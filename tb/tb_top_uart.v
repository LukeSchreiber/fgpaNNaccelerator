`timescale 1ns / 1ps
//=============================================================================
// tb_top_uart.v -- end-to-end test of the Basys 3 top level over the wire
//
// Drives real 8N1 frames into RsRx at the deployed baud rate, waits for the
// board to classify, and decodes the 5-byte reply on RsTx (class, then the
// int32 winning logit little-endian). Nothing
// reaches inside the design: this is exactly what predict.py does, minus the
// USB.
//
// WHY THIS EXISTS
// ---------------
// Every other testbench checks one module. This is the only one that exercises
// the whole chain -- uart_rx timing, img_loader packing and its watchdog, the
// start handshake, the core, argmax, and uart_tx -- against a known answer.
//
// It is also the test that makes a baud change safe to synthesize. BAUD_DIV
// and BAUD_HALF are localparams in nn_accel_top, so a wrong divisor cannot be
// caught by any module-level testbench: uart_rx passes its own tests at any
// divisor you hand it. Here the receiver's divisor and the stimulus rate are
// derived from the same number the hardware uses, so a mismatch shows up as a
// wrong class or no reply at all.
//
// The transfer is 784 * 10 * 109 = 854,560 clocks, so this runs longer than
// the other testbenches -- about 8.5 ms of simulated time.
//
// Run from the repo root; see the MEM_DIR note below to build elsewhere.
//=============================================================================

module tb_top_uart;

    `ifndef MEM_DIR
      `define MEM_DIR "mem"
    `endif
    localparam MEM_DIR = `MEM_DIR;

    // Must match nn_accel_top. 100 MHz / 921600 baud.
    localparam integer DIVISOR = 109;

    localparam integer INPUTS   = 784;
    localparam integer N_IMAGES = 16;
    localparam integer TEST_IMG = 3;     // golden class 0 ('A')

    reg clk = 0, btnC = 0, btnU = 1, RsRx = 1;
    wire RsTx;
    wire [15:0] led;
    wire [3:0]  an;
    wire [6:0]  seg;
    wire        dp;

    integer errors = 0;
    integer i;

    reg [7:0]  px          [0:N_IMAGES*INPUTS-1];
    reg [7:0]  expect_pred [0:N_IMAGES-1];
    reg [31:0] expect_score[0:N_IMAGES-1];

    nn_accel_top #(
        .W1_FILE({MEM_DIR, "/w1_packed.mem"}),
        .W2_FILE({MEM_DIR, "/w2_packed.mem"}),
        .B1_FILE({MEM_DIR, "/b1_packed.mem"}),
        .B2_FILE({MEM_DIR, "/b2_packed.mem"})
    ) dut (
        .clk(clk), .btnC(btnC), .btnU(btnU), .RsRx(RsRx), .RsTx(RsTx),
        .led(led), .an(an), .seg(seg), .dp(dp)
    );

    always #5 clk = ~clk;    // 100 MHz

    //-------------------------------------------------------------------------
    // host side of the link
    //-------------------------------------------------------------------------

    // One 8N1 frame at the deployed bit rate. The line moves 1 ns after an
    // edge so the DUT's synchronizer never samples it mid-transition.
    task uart_send(input [7:0] b);
        integer k, c;
        begin
            #1 RsRx = 1'b0;                                  // start
            for (c = 0; c < DIVISOR; c = c + 1) @(posedge clk);
            for (k = 0; k < 8; k = k + 1) begin              // LSB first
                #1 RsRx = b[k];
                for (c = 0; c < DIVISOR; c = c + 1) @(posedge clk);
            end
            #1 RsRx = 1'b1;                                  // stop
            for (c = 0; c < DIVISOR; c = c + 1) @(posedge clk);
        end
    endtask

    // Decode one frame off RsTx, sampling at the centre of each bit.
    task uart_recv(output [7:0] b);
        integer k, c;
        begin
            @(negedge RsTx);                                 // start edge
            for (c = 0; c < DIVISOR/2; c = c + 1) @(posedge clk);
            for (k = 0; k < 8; k = k + 1) begin
                for (c = 0; c < DIVISOR; c = c + 1) @(posedge clk);
                b[k] = RsTx;
            end
            for (c = 0; c < DIVISOR; c = c + 1) @(posedge clk);
        end
    endtask

    //-------------------------------------------------------------------------
    // A concurrent receiver: the reply lands ~65 us after the last byte, long
    // before the sending task returns control, so it has to be caught rather
    // than polled for afterwards.
    //-------------------------------------------------------------------------
    // The reply is 5 bytes: class, then the int32 logit little-endian.
    reg [7:0]  got_class = 8'hxx;
    reg [31:0] got_score = 32'hxxxxxxxx;
    reg        got_valid = 0;

    initial begin : receiver
        reg [7:0] b;
        integer   n;
        uart_recv(b);
        got_class = b;
        for (n = 0; n < 4; n = n + 1) begin
            uart_recv(b);
            got_score[n*8 +: 8] = b;      // little-endian
        end
        got_valid = 1'b1;
    end

    initial begin
        $readmemh({MEM_DIR, "/images.mem"}, px);
        $readmemh({MEM_DIR, "/preds.mem"},  expect_pred);
        $readmemh({MEM_DIR, "/scores.mem"}, expect_score);

        repeat (10) @(posedge clk);
        btnU = 0;                                  // release reset
        repeat (10) @(posedge clk);

        $display("streaming image %0d: %0d bytes at %0d clocks/bit",
                 TEST_IMG, INPUTS, DIVISOR);

        for (i = 0; i < INPUTS; i = i + 1)
            uart_send(px[TEST_IMG*INPUTS + i]);

        // Wait for the reply. Bounded, so a broken DUT fails instead of hanging.
        i = 0;
        while (!got_valid && i < 200000) begin
            @(posedge clk);
            i = i + 1;
        end

        if (!got_valid) begin
            $display("FAIL: no complete 5-byte reply on RsTx");
            errors = errors + 1;
        end else begin
            $display("pass: board replied with all %0d bytes", 5);

            if (got_class !== expect_pred[TEST_IMG]) begin
                $display("FAIL: class %0d, golden model says %0d",
                         got_class, expect_pred[TEST_IMG]);
                errors = errors + 1;
            end else begin
                $display("pass: class %0d matches the golden model", got_class);
            end

            // The logit is the real check: a class match only says the argmax
            // fell the same way, this says the arithmetic is identical.
            if (got_score !== expect_score[TEST_IMG]) begin
                $display("FAIL: logit %0d, golden model says %0d",
                         $signed(got_score), $signed(expect_score[TEST_IMG]));
                errors = errors + 1;
            end else begin
                $display("pass: logit %0d is bit-exact against the golden model",
                         $signed(got_score));
            end
        end

        // The watchdog must not have fired mid-transfer: it is 5,000,000 clocks
        // and the gap between bytes here is zero, but a botched idle counter
        // would have discarded the image and there would be no reply at all.
        if (got_valid && got_class === expect_pred[TEST_IMG])
            $display("pass: watchdog did not trip during a continuous stream");

        $display("");
        if (errors == 0)
            $display("=== ALL TESTS PASSED -- full chain over the wire ===");
        else
            $display("=== %0d FAILURE(S) ===", errors);
        $finish;
    end

    // 8.5 ms transfer + inference; allow generous headroom
    initial begin
        #20000000;
        $display("TIMEOUT -- no reply within 20 ms");
        $finish;
    end

endmodule
