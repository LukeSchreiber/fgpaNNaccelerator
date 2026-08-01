`timescale 1ns / 1ps
//=============================================================================
// tb_img_loader.v -- self-checking testbench for img_loader
//
// WHY THIS TESTBENCH EXISTS
// -------------------------
// The byte-packing order is the one thing in the UART image path that fails
// SILENTLY. If lane 0 ends up at the top of the word instead of the bottom,
// every pixel still lands in the buffer, the FSM still runs, done still
// asserts, and the seven-segment display still shows a class -- one drawn from
// a picture whose every group of 16 pixels is mirrored. Accuracy collapses to
// roughly chance and nothing anywhere reports an error.
//
// The check is exact rather than statistical: images_packed.mem is what the
// old internal ROM held, produced by pack_mem.py from the same pixels this
// testbench streams in. If the loader packs identically, the 49 words of
// image 0 must match rows 0..48 of that file bit for bit.
//
// A mismatch is diagnosed, not just reported: a word that equals the golden
// word with its bytes reversed means the lane order is backwards, which is by
// far the most likely way to get this wrong, so it is called out by name.
//
// ALSO COVERED
//   - img_ready rises on byte 784 and not one byte earlier
//   - bytes arriving while img_ready is high are dropped, not written
//   - img_ack clears img_ready
//   - the loader rewinds, so a second image lands correctly after the first
//
// Run from the repo root; see the MEM_DIR note below to build elsewhere.
//=============================================================================

module tb_img_loader;

    // $readmemh resolves relative paths against the SIMULATOR'S RUN DIRECTORY,
    // not this file, so the default assumes you run from the repo root:
    //     iverilog -o tb.vvp tb/tb_core.v rtl/*.v && vvp tb.vvp
    // Building elsewhere? Override without editing this file:
    //     iverilog -DMEM_DIR='"/abs/path/to/mem"' ...
    `ifndef MEM_DIR
      `define MEM_DIR "mem"
    `endif
    localparam MEM_DIR = `MEM_DIR;

    localparam integer N         = 16;
    localparam integer IMG_BYTES = 784;
    localparam integer CHUNKS    = IMG_BYTES / N;    // 49 words per image
    localparam integer N_IMAGES  = 16;

    reg                clk = 0;
    reg                rst = 1;
    reg  [7:0]         rx_data = 8'd0;
    reg                rx_valid = 0;
    reg                img_ack = 0;
    reg  [5:0]         rd_addr = 6'd0;
    wire               img_ready;
    wire [N*8-1:0]     rd_data;

    integer errors = 0;
    integer k;

    // Golden data: the raw pixel stream and the packed words pack_mem.py
    // produced from it.
    reg [7:0]      px     [0:N_IMAGES*IMG_BYTES-1];
    reg [N*8-1:0]  golden [0:N_IMAGES*CHUNKS-1];

    img_loader #(.N(N), .IMG_BYTES(IMG_BYTES)) dut (
        .clk(clk), .rst(rst),
        .rx_data(rx_data), .rx_valid(rx_valid),
        .img_ready(img_ready), .img_ack(img_ack),
        .rd_addr(rd_addr), .rd_data(rd_data)
    );

    always #5 clk = ~clk;    // 100 MHz -> 10 ns period

    //-------------------------------------------------------------------------
    // helpers
    //-------------------------------------------------------------------------

    // One accepted byte. rx_valid is a one-cycle pulse, then a couple of idle
    // cycles -- a real uart_rx leaves ~8700 clocks between strobes, and the
    // loader must not care how wide the gap is.
    task send_byte(input [7:0] b);
        begin
            rx_data  = b;
            rx_valid = 1'b1;
            @(posedge clk); #1;
            rx_valid = 1'b0;
            repeat (2) @(posedge clk); #1;
        end
    endtask

    // Present an address and return the word one cycle later: the read port is
    // registered, exactly like rom_sync.
    task read_word(input [5:0] a, output [N*8-1:0] d);
        begin
            rd_addr = a;
            @(posedge clk); #1;
            d = rd_data;
        end
    endtask

    // Byte-reverse a word, to recognize the lane-order failure by sight.
    function [N*8-1:0] byte_reverse(input [N*8-1:0] w);
        integer j;
        reg [N*8-1:0] t;
        begin
            for (j = 0; j < N; j = j + 1)
                t[j*8 +: 8] = w[(N-1-j)*8 +: 8];
            byte_reverse = t;
        end
    endfunction

    // Stream one whole image and check that img_ready rises on the last byte
    // and not before.
    task stream_image(input integer img);
        integer b;
        reg     early;      // report an early rise once, not 700 times
        begin
            early = 1'b0;
            for (b = 0; b < IMG_BYTES; b = b + 1) begin
                send_byte(px[img*IMG_BYTES + b]);

                if (b < IMG_BYTES-1 && img_ready !== 1'b0 && !early) begin
                    $display("FAIL: img_ready rose after byte %0d of %0d",
                             b+1, IMG_BYTES);
                    errors = errors + 1;
                    early   = 1'b1;
                end
            end

            if (img_ready !== 1'b1) begin
                $display("FAIL: img_ready did not assert after byte %0d",
                         IMG_BYTES);
                errors = errors + 1;
            end else begin
                $display("pass: img_ready asserts exactly at byte %0d",
                         IMG_BYTES);
            end
        end
    endtask

    // Compare all 49 buffered words against the packed golden image.
    task check_image(input integer img);
        integer c, bad;
        reg [N*8-1:0] got, want;
        begin
            bad = 0;
            for (c = 0; c < CHUNKS; c = c + 1) begin
                read_word(c[5:0], got);
                want = golden[img*CHUNKS + c];

                if (got !== want) begin
                    bad = bad + 1;
                    errors = errors + 1;
                    $display("FAIL word %2d: got  %032h", c, got);
                    $display("              want %032h", want);
                    if (got === byte_reverse(want))
                        $display("              ^^ BYTES REVERSED: lane 0 must be the LOW byte, bits [7:0]");
                end else if (c < 3 || c == CHUNKS-1) begin
                    // A few words shown so a passing run is legible; printing
                    // all 49 would bury the summary.
                    $display("pass word %2d: %032h", c, got);
                end
            end

            if (bad == 0)
                $display("pass: all %0d packed words of image %0d match images_packed.mem",
                         CHUNKS, img);
        end
    endtask

    //-------------------------------------------------------------------------
    // stimulus
    //-------------------------------------------------------------------------
    reg [N*8-1:0] w0_before, w0_after;

    initial begin
        $readmemh({MEM_DIR, "/images.mem"},        px);
        $readmemh({MEM_DIR, "/images_packed.mem"}, golden);

        repeat (4) @(posedge clk);
        rst = 0;
        repeat (2) @(posedge clk); #1;

        //---- image 0: packing ------------------------------------------
        $display("--- image 0: stream %0d bytes ---", IMG_BYTES);
        stream_image(0);
        check_image(0);

        //---- bytes arriving while ready must be dropped -----------------
        // If the loader kept writing, these would overwrite word 0 with 0xA5s
        // while the core was still reading the image out of the buffer.
        read_word(6'd0, w0_before);
        for (k = 0; k < N + 4; k = k + 1)
            send_byte(8'hA5);

        if (img_ready !== 1'b1) begin
            $display("FAIL: img_ready dropped while bytes were being ignored");
            errors = errors + 1;
        end

        read_word(6'd0, w0_after);
        if (w0_after !== w0_before) begin
            $display("FAIL: word 0 changed while img_ready was high: %032h",
                     w0_after);
            errors = errors + 1;
        end else begin
            $display("pass: %0d bytes sent while img_ready was high were dropped",
                     N + 4);
        end

        //---- img_ack clears ready ---------------------------------------
        img_ack = 1'b1;
        @(posedge clk); #1;
        img_ack = 1'b0;

        if (img_ready !== 1'b0) begin
            $display("FAIL: img_ready still set after img_ack");
            errors = errors + 1;
        end else begin
            $display("pass: img_ack clears img_ready");
        end

        //---- image 1: the loader must have rewound -----------------------
        // The dropped 0xA5s must not have left the write pointer part way
        // through a word, or image 1 lands rotated by however many were taken.
        $display("");
        $display("--- image 1: stream %0d bytes ---", IMG_BYTES);
        stream_image(1);
        check_image(1);

        //---- summary ----------------------------------------------------
        $display("");
        if (errors == 0)
            $display("=== ALL TESTS PASSED -- packing matches images_packed.mem ===");
        else
            $display("=== %0d FAILURE(S) ===", errors);
        $finish;
    end

    // safety net: a loader that never asserts img_ready would otherwise spin
    // in stream_image forever
    initial begin
        #2000000;
        $display("TIMEOUT -- img_ready never asserted");
        $finish;
    end

endmodule
