`timescale 1ns / 1ps
//=============================================================================
// tb_mac_unit.v -- self-checking testbench for the pipelined mac_unit
//
// The DUT now has 2-cycle latency: operands presented at cycle k land in acc
// at cycle k+2. `drive` presents one operand pair per cycle; `drain` clocks
// once more so the final product reaches the accumulator before checking.
//=============================================================================

module tb_mac_unit;

    localparam integer A_WIDTH   = 8;
    localparam integer W_WIDTH   = 8;
    localparam integer ACC_WIDTH = 32;

    reg                          clk = 0;
    reg                          rst = 1;
    reg                          en  = 0;
    reg                          clr = 0;
    reg        [A_WIDTH-1:0]     a   = 0;
    reg signed [W_WIDTH-1:0]     w   = 0;
    reg signed [ACC_WIDTH-1:0]   init = 0;
    wire signed [ACC_WIDTH-1:0]  acc;

    integer errors = 0;
    integer i;

    reg signed [ACC_WIDTH-1:0] ref_acc;

    mac_unit #(
        .A_WIDTH(A_WIDTH), .W_WIDTH(W_WIDTH), .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk(clk), .rst(rst), .en(en), .clr(clr),
        .a(a), .w(w), .init(init), .acc(acc)
    );

    always #5 clk = ~clk;   // 100 MHz

    // Present one operand pair. Advances one clock.
    task drive(input [A_WIDTH-1:0] ta,
               input signed [W_WIDTH-1:0] tw,
               input tclr,
               input signed [ACC_WIDTH-1:0] tinit);
        begin
            a = ta; w = tw; clr = tclr; init = tinit; en = 1;
            ref_acc = (tclr ? tinit : ref_acc)
                      + $signed({1'b0, ta}) * tw;
            @(posedge clk);
            #1;
        end
    endtask

    // Let the last product propagate from stage 1 into the accumulator.
    task drain;
        begin
            en  = 0;
            clr = 0;
            @(posedge clk);
            #1;
        end
    endtask

    task check(input [255:0] label);
        begin
            if (acc !== ref_acc) begin
                $display("FAIL %0s: acc=%0d expected=%0d", label, acc, ref_acc);
                errors = errors + 1;
            end else begin
                $display("pass %0s: acc=%0d", label, acc);
            end
        end
    endtask

    task reset_dut;
        begin
            rst = 1; en = 0; clr = 0;
            @(posedge clk); #1;
            rst = 0;
            @(posedge clk); #1;
            ref_acc = 0;
        end
    endtask

    initial begin
        @(posedge clk);
        reset_dut;

        // zero weight -- accumulator must not move
        drive(8'd255, 8'sd0, 1'b1, 32'sd0);
        drain;
        check("zero weight");

        // max positive: 255 * 127
        reset_dut;
        drive(8'd255, 8'sd127, 1'b1, 32'sd0);
        drain;
        check("max positive (255*127)");

        // THE signedness test. A wrong unsigned/signed mix returns a large
        // positive value instead of -32640.
        reset_dut;
        drive(8'd255, -8'sd128, 1'b1, 32'sd0);
        drain;
        check("max negative (255*-128)");

        // bias fused with the first product
        reset_dut;
        drive(8'd10, 8'sd10, 1'b1, 32'sd1000);
        drain;
        check("bias preload");

        // accumulate a second term on top
        drive(8'd10, 8'sd10, 1'b0, 32'sd0);
        drain;
        check("accumulate");

        // negative product pulling the accumulator below zero
        drive(8'd200, -8'sd100, 1'b0, 32'sd0);
        drain;
        check("negative accumulate");

        // enable low: accumulator must hold
        begin : hold_test
            reg signed [ACC_WIDTH-1:0] held;
            held = acc;
            en = 0; a = 8'd255; w = 8'sd127;
            @(posedge clk); #1;
            @(posedge clk); #1;
            if (acc !== held) begin
                $display("FAIL enable gating: acc changed while en=0");
                errors = errors + 1;
            end else begin
                $display("pass enable gating");
            end
        end

        // synchronous reset
        rst = 1; @(posedge clk); #1; rst = 0; ref_acc = 0;
        if (acc !== 0) begin
            $display("FAIL reset: acc=%0d", acc);
            errors = errors + 1;
        end else begin
            $display("pass reset");
        end

        // back-to-back streaming, 784 terms -- the real layer-1 depth.
        // Operands are presented every cycle with no gaps, which is what
        // exercises the pipeline rather than just its steady state.
        reset_dut;
        drive($random, $random, 1'b1, 32'sd54321);
        for (i = 1; i < 784; i = i + 1)
            drive($random, $random, 1'b0, 32'sd0);
        drain;
        check("784-term streamed accumulation");

        $display("");
        if (errors == 0)
            $display("=== ALL TESTS PASSED ===");
        else
            $display("=== %0d FAILURE(S) ===", errors);
        $finish;
    end

endmodule
