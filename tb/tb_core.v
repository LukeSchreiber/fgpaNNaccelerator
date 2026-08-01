`timescale 1ns / 1ps
//=============================================================================
// tb_core.v -- end-to-end test of nn_accel_core
//
// Runs inference on every demo image and compares the hardware against
// preds.mem and scores.mem, which golden_check.py generated from the integer
// reference model. Matching the reference -- not the true labels -- is the
// claim that means something: the quantized model is ~92% accurate, so
// checking against true labels would fail on correct hardware.
//
// WHY THE SCORE IS CHECKED AND NOT JUST THE CLASS
// -----------------------------------------------
// The class index is a lossy view of the arithmetic: it survives a dot product
// that is merely close to right. Deleting one stage from the capture chain --
// so every neuron is read one cycle early, before its last 16 terms land --
// changes the class on ONE of these 16 images and leaves the other 15 passing.
// The winning logit is bit-exact or it is not, and it caught that on image 0.
//
// The cycle count is checked for the same reason: it is the only thing here
// that notices a wrong DRAIN_CYCLES, since a pipeline that drains for too long
// still produces the right answer, just later.
//
// MEM_DIR must be an absolute path. $readmemh resolves relative paths
// against the simulator's run directory, not the project root.
//=============================================================================

module tb_core;

    localparam MEM_DIR = "/home/yetigod/fpga-nn-accel/mem";

    localparam integer N         = 16;
    localparam integer INPUTS    = 784;
    localparam integer HIDDEN    = 128;
    localparam integer NUM_CLASS = 24;
    localparam integer ACC_WIDTH = 32;
    localparam integer SHIFT1    = 10;
    localparam integer N_IMAGES  = 16;

    // Must track the localparam of the same name in nn_accel_core.v: five
    // cycles are needed to drain the pipeline (mac_unit x2, mac_array partial
    // sums, sum_r) and the RTL allows seven.
    localparam integer DRAIN_CYCLES  = 7;
    localparam integer EXPECT_CYCLES = HIDDEN * (INPUTS / N)
                                     + NUM_CLASS * (HIDDEN / N)
                                     + 2 * (DRAIN_CYCLES + 1);

    reg                        clk = 0;
    reg                        rst = 1;
    reg                        start = 0;
    reg  [3:0]                 img_sel = 0;
    wire                       busy, done;
    wire [4:0]                 pred;
    wire signed [ACC_WIDTH-1:0] pred_score;
    wire [31:0]                cycles;

    integer errors = 0;
    integer i;
    integer total_cycles = 0;

    reg [7:0]            expect_pred  [0:N_IMAGES-1];
    reg [ACC_WIDTH-1:0]  expect_score [0:N_IMAGES-1];

    nn_accel_core #(
        .N(N), .INPUTS(INPUTS), .HIDDEN(HIDDEN), .NUM_CLASS(NUM_CLASS),
        .ACC_WIDTH(ACC_WIDTH), .SHIFT1(SHIFT1), .N_IMAGES(N_IMAGES),
        .W1_FILE ({MEM_DIR, "/w1_packed.mem"}),
        .W2_FILE ({MEM_DIR, "/w2_packed.mem"}),
        .B1_FILE ({MEM_DIR, "/b1_packed.mem"}),
        .B2_FILE ({MEM_DIR, "/b2_packed.mem"}),
        .IMG_FILE({MEM_DIR, "/images_packed.mem"})
    ) dut (
        .clk(clk), .rst(rst), .start(start), .img_sel(img_sel),
        .busy(busy), .done(done),
        .pred(pred), .pred_score(pred_score), .cycles(cycles)
    );

    always #5 clk = ~clk;

    initial begin
        $readmemh({MEM_DIR, "/preds.mem"},  expect_pred);
        $readmemh({MEM_DIR, "/scores.mem"}, expect_score);

        repeat (4) @(posedge clk);
        rst = 0;
        @(posedge clk); #1;

        for (i = 0; i < N_IMAGES; i = i + 1) begin
            img_sel = i[3:0];
            start   = 1;
            @(posedge clk); #1;
            start   = 0;

            wait (done == 1);
            #1;

            total_cycles = total_cycles + cycles;

            // Compared as raw bit patterns: expect_score holds two's complement
            // straight out of scores.mem, which is what pred_score is.
            if (pred !== expect_pred[i][4:0]) begin
                $display("FAIL image %0d: class hw=%0d golden=%0d  (score %0d, %0d cycles)",
                         i, pred, expect_pred[i], pred_score, cycles);
                errors = errors + 1;
            end else if (pred_score !== expect_score[i]) begin
                $display("FAIL image %0d: class %0d correct but score hw=%0d golden=%0d",
                         i, pred, pred_score, $signed(expect_score[i]));
                errors = errors + 1;
            end else if (cycles !== EXPECT_CYCLES) begin
                $display("FAIL image %0d: %0d cycles, expected %0d",
                         i, cycles, EXPECT_CYCLES);
                errors = errors + 1;
            end else begin
                $display("pass image %2d: class %2d  score %10d  %0d cycles",
                         i, pred, pred_score, cycles);
            end

            @(posedge clk); #1;
        end

        $display("");
        $display("mean cycles/inference: %0d", total_cycles / N_IMAGES);
        $display("  = %0d ns at 100 MHz, %0d inferences/sec",
                 (total_cycles / N_IMAGES) * 10,
                 100000000 / (total_cycles / N_IMAGES));
        $display("");
        if (errors == 0)
            $display("=== ALL TESTS PASSED -- RTL matches golden model ===");
        else
            $display("=== %0d FAILURE(S) ===", errors);
        $finish;
    end

    // safety net: a stuck FSM should not hang the simulation forever
    initial begin
        #2000000;
        $display("TIMEOUT -- FSM never asserted done");
        $finish;
    end

endmodule
