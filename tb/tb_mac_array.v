`timescale 1ns / 1ps
//=============================================================================
// tb_mac_array.v -- self-checking testbench for mac_array
//
// Both mac_unit and the reduction are pipelined, so every burst ends with a
// drain before `sum` is valid. Counting from the edge that consumes the last
// pair of operands (which is also when mac_unit registers the product):
//
//   +1  mac_unit stage 2, the accumulator
//   +1  mac_array's registered partial sums
//
// -- two cycles. It was one before the reduction was split into halves, and a
// testbench that drains too few cycles reads the second-to-last accumulator
// state, which for a long dot product is very nearly the right answer.
//
// Checks:
//   1. Hand-computable uniform dot product across all N lanes.
//   2. Bias fusion into lane 0.
//   3. All-negative weights -- catches sign or bit-slicing errors that a
//      positive-only test would miss.
//   4. Full 784-term random dot product at N=16 (49 cycles), streamed with
//      no gaps, against a behavioral reference.
//=============================================================================

module tb_mac_array;

    localparam integer N         = 16;
    localparam integer A_WIDTH   = 8;
    localparam integer W_WIDTH   = 8;
    localparam integer ACC_WIDTH = 32;
    localparam integer DEPTH     = 784;
    localparam integer CYCLES    = DEPTH / N;   // 49
    localparam integer DRAIN     = 2;           // see the header

    reg                          clk = 0;
    reg                          rst = 1;
    reg                          en  = 0;
    reg                          clr = 0;
    reg  [N*A_WIDTH-1:0]         a_flat = 0;
    reg  [N*W_WIDTH-1:0]         w_flat = 0;
    reg  signed [ACC_WIDTH-1:0]  bias = 0;
    wire signed [ACC_WIDTH-1:0]  sum;

    integer errors = 0;
    integer c, j;

    reg signed [ACC_WIDTH-1:0] ref_sum;

    reg  [A_WIDTH-1:0]        a_mem [0:DEPTH-1];
    reg  signed [W_WIDTH-1:0] w_mem [0:DEPTH-1];

    mac_array #(
        .N(N), .A_WIDTH(A_WIDTH), .W_WIDTH(W_WIDTH), .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk(clk), .rst(rst), .en(en), .clr(clr),
        .a_flat(a_flat), .w_flat(w_flat), .bias(bias), .sum(sum)
    );

    always #5 clk = ~clk;

    task step(input first);
        begin
            clr = first;
            en  = 1;
            @(posedge clk);
            #1;
        end
    endtask

    task drain;
        integer d;
        begin
            en  = 0;
            clr = 0;
            for (d = 0; d < DRAIN; d = d + 1) begin
                @(posedge clk);
                #1;
            end
        end
    endtask

    task check(input [255:0] label);
        begin
            if (sum !== ref_sum) begin
                $display("FAIL %0s: sum=%0d expected=%0d", label, sum, ref_sum);
                errors = errors + 1;
            end else begin
                $display("pass %0s: sum=%0d", label, sum);
            end
        end
    endtask

    task reset_dut;
        begin
            rst = 1; en = 0; clr = 0;
            @(posedge clk); #1;
            rst = 0;
            @(posedge clk); #1;
        end
    endtask

    initial begin
        @(posedge clk);
        reset_dut;

        // 1. all activations 2, all weights 3, no bias -> N * 6
        for (j = 0; j < N; j = j + 1) begin
            a_flat[j*A_WIDTH +: A_WIDTH] = 8'd2;
            w_flat[j*W_WIDTH +: W_WIDTH] = 8'd3;
        end
        bias = 0;
        ref_sum = N * 6;
        step(1'b1);
        drain;
        check("uniform 2*3 across N lanes");

        // 2. same, with a bias fused into lane 0 only
        reset_dut;
        bias = 32'sd1000;
        ref_sum = N * 6 + 1000;
        step(1'b1);
        drain;
        check("bias fusion");

        // 3. all weights -128 against activations 255 -> N * -32640
        reset_dut;
        for (j = 0; j < N; j = j + 1) begin
            a_flat[j*A_WIDTH +: A_WIDTH] = 8'd255;
            w_flat[j*W_WIDTH +: W_WIDTH] = -8'sd128;
        end
        bias = 0;
        ref_sum = N * (-32640);
        step(1'b1);
        drain;
        check("all-negative weights");

        // 4. full 784-term random dot product, streamed
        reset_dut;
        for (j = 0; j < DEPTH; j = j + 1) begin
            a_mem[j] = $random;
            w_mem[j] = $random;
        end
        bias = 32'sd12345;

        ref_sum = bias;
        for (j = 0; j < DEPTH; j = j + 1)
            ref_sum = ref_sum + $signed({1'b0, a_mem[j]}) * w_mem[j];

        for (c = 0; c < CYCLES; c = c + 1) begin
            for (j = 0; j < N; j = j + 1) begin
                a_flat[j*A_WIDTH +: A_WIDTH] = a_mem[c*N + j];
                w_flat[j*W_WIDTH +: W_WIDTH] = w_mem[c*N + j];
            end
            step(c == 0);
        end
        drain;
        check("784-term random dot product");

        $display("");
        if (errors == 0)
            $display("=== ALL TESTS PASSED ===");
        else
            $display("=== %0d FAILURE(S) ===", errors);
        $finish;
    end

endmodule
