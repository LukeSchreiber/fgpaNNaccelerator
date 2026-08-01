`timescale 1ns / 1ps
//=============================================================================
// tb_datapath.v -- self-checking testbench for requantize and argmax
//
// requantize cases are chosen to hit the exact boundaries where a wrong
// shift operator, a missing ReLU, or wrapping instead of saturating would
// change the answer.
//=============================================================================

module tb_datapath;

    localparam integer ACC_WIDTH = 32;
    localparam integer OUT_WIDTH = 8;
    localparam integer SHIFT     = 10;
    localparam integer COUNT     = 24;
    localparam integer IDXW      = 5;   // $clog2(24)

    integer errors = 0;
    integer k;

    //---------------------------------------------------------------
    // requantize
    //---------------------------------------------------------------
    reg  signed [ACC_WIDTH-1:0] rq_acc;
    wire        [OUT_WIDTH-1:0] rq_act;

    requantize #(
        .ACC_WIDTH(ACC_WIDTH), .OUT_WIDTH(OUT_WIDTH), .SHIFT(SHIFT)
    ) u_rq (.acc(rq_acc), .act(rq_act));

    task rq_check(input signed [ACC_WIDTH-1:0] a,
                  input [OUT_WIDTH-1:0] expect,
                  input [255:0] label);
        begin
            rq_acc = a;
            #1;
            if (rq_act !== expect) begin
                $display("FAIL requantize %0s: acc=%0d got=%0d expected=%0d",
                         label, a, rq_act, expect);
                errors = errors + 1;
            end else begin
                $display("pass requantize %0s: %0d -> %0d", label, a, rq_act);
            end
        end
    endtask

    //---------------------------------------------------------------
    // argmax
    //---------------------------------------------------------------
    reg                          clk = 0;
    reg                          rst = 1;
    reg                          am_en = 0;
    reg                          am_first = 0;
    reg  signed [ACC_WIDTH-1:0]  am_value = 0;
    reg  [IDXW-1:0]              am_index = 0;
    wire [IDXW-1:0]              am_best_idx;
    wire signed [ACC_WIDTH-1:0]  am_best_val;

    argmax #(.WIDTH(ACC_WIDTH), .COUNT(COUNT)) u_am (
        .clk(clk), .rst(rst), .en(am_en), .first(am_first),
        .value(am_value), .index(am_index),
        .best_idx(am_best_idx), .best_val(am_best_val)
    );

    always #5 clk = ~clk;

    reg signed [ACC_WIDTH-1:0] logits [0:COUNT-1];

    task am_feed(input integer i, input signed [ACC_WIDTH-1:0] v);
        begin
            am_value = v;
            am_index = i[IDXW-1:0];
            am_first = (i == 0);
            am_en    = 1;
            @(posedge clk);
            #1;
            am_en    = 0;
            am_first = 0;
        end
    endtask

    initial begin
        //-----------------------------------------------------------
        // requantize
        //-----------------------------------------------------------
        rq_check(32'sd0,        8'd0,   "zero");
        rq_check(32'sd1023,     8'd0,   "just below 1 LSB (floor)");
        rq_check(32'sd1024,     8'd1,   "exactly 1 LSB");
        rq_check(32'sd2047,     8'd1,   "floor toward zero");
        rq_check(-32'sd1,       8'd0,   "-1 (arithmetic shift -> -1, ReLU)");
        rq_check(-32'sd5000,    8'd0,   "negative (ReLU)");
        rq_check(-32'sd1048576, 8'd0,   "large negative (ReLU)");
        rq_check(32'sd261120,   8'd255, "exactly 255 LSB");
        rq_check(32'sd262143,   8'd255, "just below saturation");
        rq_check(32'sd262144,   8'd255, "saturate at 256 LSB");
        rq_check(32'sd373327,   8'd255, "observed peak acc1 (saturates)");

        //-----------------------------------------------------------
        // argmax
        //-----------------------------------------------------------
        @(posedge clk); #1;
        rst = 0;
        @(posedge clk); #1;

        // all negative -- catches a zero-initialized best_val
        for (k = 0; k < COUNT; k = k + 1)
            logits[k] = -1000 - k * 10;
        for (k = 0; k < COUNT; k = k + 1)
            am_feed(k, logits[k]);
        if (am_best_idx !== 0) begin
            $display("FAIL argmax all-negative: idx=%0d expected=0", am_best_idx);
            errors = errors + 1;
        end else
            $display("pass argmax all-negative: idx=%0d val=%0d",
                     am_best_idx, am_best_val);

        // winner in the middle
        rst = 1; @(posedge clk); #1; rst = 0;
        for (k = 0; k < COUNT; k = k + 1)
            logits[k] = k * 7 - 50;
        logits[13] = 999999;
        for (k = 0; k < COUNT; k = k + 1)
            am_feed(k, logits[k]);
        if (am_best_idx !== 13) begin
            $display("FAIL argmax mid-winner: idx=%0d expected=13", am_best_idx);
            errors = errors + 1;
        end else
            $display("pass argmax mid-winner: idx=%0d val=%0d",
                     am_best_idx, am_best_val);

        // winner last -- catches an off-by-one in the loop bound
        rst = 1; @(posedge clk); #1; rst = 0;
        for (k = 0; k < COUNT; k = k + 1)
            logits[k] = k;
        for (k = 0; k < COUNT; k = k + 1)
            am_feed(k, logits[k]);
        if (am_best_idx !== COUNT-1) begin
            $display("FAIL argmax last-winner: idx=%0d expected=%0d",
                     am_best_idx, COUNT-1);
            errors = errors + 1;
        end else
            $display("pass argmax last-winner: idx=%0d", am_best_idx);

        // tie -- must resolve to the lowest index, like numpy
        rst = 1; @(posedge clk); #1; rst = 0;
        for (k = 0; k < COUNT; k = k + 1)
            logits[k] = (k == 5 || k == 17) ? 500 : 10;
        for (k = 0; k < COUNT; k = k + 1)
            am_feed(k, logits[k]);
        if (am_best_idx !== 5) begin
            $display("FAIL argmax tie: idx=%0d expected=5 (lowest)", am_best_idx);
            errors = errors + 1;
        end else
            $display("pass argmax tie resolves to lowest index");

        $display("");
        if (errors == 0)
            $display("=== ALL TESTS PASSED ===");
        else
            $display("=== %0d FAILURE(S) ===", errors);
        $finish;
    end

endmodule
