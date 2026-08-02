`timescale 1ns / 1ps
//=============================================================================
// nn_accel_top.v -- Basys 3 top level, UART image input
//
//   RsRx          784 raw pixel bytes -> inference starts by itself
//   RsTx          one byte back per image: the predicted class index, 0..23
//   BTNC          re-run the buffered image (demo convenience, no resend)
//   BTNU          reset
//   7-seg         [letter] [ ] [tens] [ones]   predicted class
//   LD15          done
//   LD14          busy
//   LD13          UART framing error since reset (sticky)
//   LD[12:0]      cycle count, upper bits (a crude latency display)
//
// The weights, biases and the display are exactly as they were; only where
// the image comes from has changed. w1/w2/b1/b2 still load from .mem files
// through $readmemh at configuration -- they are fixed for a given trained
// network and there is no reason to stream them.
//
// FLOW
// ----
//   host sends 784 bytes  ->  img_loader fills its BRAM, raises img_ready
//   img_ready             ->  one-cycle start pulse into the core, img_ack
//                             back to the loader so it will take the next image
//   core runs 6,480 cycles (64.8 us, far shorter than the 8.5 ms the transfer
//                             itself takes at 921600 baud)
//   done rises            ->  one byte out on RsTx, 7-seg updates
//
// The button start has been replaced by img_ready: an image arriving IS the
// request to classify it. btnC no longer selects when the first inference
// happens, it only re-runs whatever is already in the loader's buffer -- the
// buffer holds its last image, so the same result comes back out without a
// 68 ms retransmit. That is worth keeping for demos, where the interesting
// thing to show is the 65 us of compute, not the serial link.
//
// The image-select switches are gone: there is exactly one image in the
// buffer, so there is nothing to select.
//
// BAUD vs LATENCY: 784 bytes at 921600 8N1 is 8.5 ms, still 130x longer than
// the inference. The link, not the accelerator, sets the frame rate, which is
// why no attempt is made to overlap load and compute.
//=============================================================================

module nn_accel_top #(
    // Weight and bias initialization files, as BARE FILENAMES.
    //
    // Vivado resolves a bare $readmemh filename against the files added to the
    // project as design sources, so mem/*_packed.mem must be in sources_1 --
    // see the "Building the bitstream" section of the README. That keeps one
    // machine's absolute paths out of the repo.
    //
    // Override with a path to simulate this top level directly:
    //   nn_accel_top #(.W1_FILE("mem/w1_packed.mem"), ...) dut (...);
    //
    // A single MEM_DIR prefix is deliberately NOT used. Concatenating an empty
    // string parameter in Verilog yields a leading NUL byte, so {MEM_DIR,
    // "w1.mem"} with an empty default produces a filename that fails to open
    // and leaves the BRAMs full of zeros -- a board that runs and classifies
    // everything as class 0, with no error anywhere.
    parameter W1_FILE = "w1_packed.mem",
    parameter W2_FILE = "w2_packed.mem",
    parameter B1_FILE = "b1_packed.mem",
    parameter B2_FILE = "b2_packed.mem"
)(
    input  wire        clk,        // W5, 100 MHz
    input  wire        btnC,       // re-run the buffered image
    input  wire        btnU,       // reset
    input  wire        RsRx,       // B18, from the USB-UART bridge
    output wire        RsTx,       // A18, to the USB-UART bridge
    output wire [15:0] led,
    output wire [3:0]  an,
    output wire [6:0]  seg,
    output wire        dp
);

    localparam integer N          = 16;
    localparam integer INPUTS     = 784;
    localparam integer N_IMAGES   = 16;                      // core's address space
    localparam integer IMG_CHUNKS = INPUTS / N;              // 49 words
    localparam integer IMG_AW     = $clog2(IMG_CHUNKS);      //  6 bits, one image
    localparam integer CORE_AW    = $clog2(N_IMAGES * IMG_CHUNKS);   // 10 bits

    // 100 MHz / 921600 baud = 108.5 -> 109 clocks per bit (917,431 baud,
    // -0.45% error). An 8N1 frame is resampled from the start bit, so the error
    // only accumulates over 9.5 bit times: 4.3% of a bit at the stop bit, well
    // inside the ~50% margin before a sample lands in the wrong bit.
    //
    // HALF_DIV centres the first sample in the start bit: 109/2 = 54.5 -> 54.
    // The half-bit rounding is absorbed the same way, since every later sample
    // is timed from that point rather than from the frame start.
    //
    // 921600 is the fastest standard rate the Basys 3's FT2232H handles
    // cleanly. 784 bytes: 68.1 ms -> 8.5 ms, so the link goes from 14.7 to 117
    // images/sec while the inference stays 64.8 us.
    localparam integer BAUD_DIV  = 109;
    localparam integer BAUD_HALF = 54;

    //---------------------------------------------------------------
    // buttons: synchronize and debounce
    //
    // Both inputs are asynchronous to clk and go through two flops, because an
    // asynchronous input into a state machine can go metastable. btnC then has
    // to hold its new value for 2^20 clocks -- about 10 ms -- before it counts:
    // a mechanical switch bounces for a few milliseconds, which at 100 MHz is
    // hundreds of thousands of clocks and hence hundreds of re-run pulses.
    //
    // The FSM restarts cleanly on a mid-inference start anyway, but relying on
    // that would mean the cycle counter reports garbage.
    //---------------------------------------------------------------
    reg [1:0] btnC_sync, btnU_sync;
    always @(posedge clk) begin
        btnC_sync <= {btnC_sync[0], btnC};
        btnU_sync <= {btnU_sync[0], btnU};
    end

    wire rst = btnU_sync[1];

    // These MUST be reset. Without it btnC_stable powers up unknown in
    // simulation, `btnC_sync[1] != btnC_stable` is X, `if (X)` takes the else
    // branch, and btnC_stable is never assigned -- so it stays X forever. The X
    // reaches start_core through rerun_pulse and sticks there (x && x = x), and
    // the core can never start. On Xilinx parts every flop configures to 0, so
    // the board works anyway; the cost is that any top-level simulation is dead
    // on arrival, which is exactly where an integration bug would show up.
    reg [19:0] db_cnt;
    reg        btnC_stable, btnC_prev;
    always @(posedge clk) begin
        if (rst) begin
            db_cnt      <= 20'd0;
            btnC_stable <= 1'b0;
            btnC_prev   <= 1'b0;
        end else begin
            if (btnC_sync[1] != btnC_stable) begin
                db_cnt <= db_cnt + 1'b1;
                if (&db_cnt) btnC_stable <= btnC_sync[1];
            end else begin
                db_cnt <= 20'd0;
            end
            btnC_prev <= btnC_stable;
        end
    end

    wire rerun_pulse = btnC_stable & ~btnC_prev;   // one cycle on press

    //---------------------------------------------------------------
    // UART receive -> image buffer
    //---------------------------------------------------------------
    wire [7:0] rx_data;
    wire       rx_valid;
    wire       rx_framing_error;

    uart_rx #(.DIVISOR(BAUD_DIV), .HALF_DIV(BAUD_HALF)) u_rx (
        .clk(clk), .rst(rst), .rx(RsRx),
        .data_out(rx_data), .data_valid(rx_valid),
        .framing_error(rx_framing_error)
    );

    // Sticky, because a framing error is a single-cycle event on a link that
    // delivers 784 bytes in 8.5 ms -- nobody would ever see it flash. It stays
    // lit until reset, so "the link glitched at some point since power-up" is
    // answerable by looking at the board instead of guessing.
    reg frame_err_sticky;
    always @(posedge clk) begin
        if (rst)                  frame_err_sticky <= 1'b0;
        else if (rx_framing_error) frame_err_sticky <= 1'b1;
    end

    wire               img_ready;
    wire               img_ack;
    wire [IMG_AW-1:0]  img_rd_addr;
    wire [N*8-1:0]     img_rd_data;

    img_loader #(.N(N), .IMG_BYTES(INPUTS)) u_loader (
        .clk(clk), .rst(rst),
        .rx_data(rx_data), .rx_valid(rx_valid),
        .img_ready(img_ready), .img_ack(img_ack),
        .rd_addr(img_rd_addr), .rd_data(img_rd_data)
    );

    //---------------------------------------------------------------
    // start: a new image or a button press, either way a one-cycle pulse.
    //
    // img_ready is a level and the core wants a pulse, so the !start_core term
    // stops it issuing a second pulse on the cycle after the first, before busy
    // has had time to rise. rerun_pulse is already one cycle wide, but it goes
    // through the same register so both requests reach the core the same way.
    // The core takes start in both S_IDLE and S_DONE, so back-to-back images
    // need no idling in between.
    //---------------------------------------------------------------
    wire busy, done;
    wire start_req = (img_ready || rerun_pulse) && !busy;
    reg  start_core;

    always @(posedge clk) begin
        if (rst) start_core <= 1'b0;
        else     start_core <= start_req && !start_core;
    end

    // Released on the same cycle the core samples start, so the loader is free
    // to take the next image while this one is being scored out of its BRAM.
    // That is safe only because the transfer is 1000x slower than the
    // inference: the first byte of the next image cannot land before this run
    // has finished reading the buffer.
    //
    // A re-run press with no image pending drives ack against an already-clear
    // img_ready, which the loader ignores.
    assign img_ack = start_core;

    //---------------------------------------------------------------
    // inference core -- weights still from .mem, image from the loader
    //---------------------------------------------------------------
    wire [4:0]  pred;
    wire signed [31:0] pred_score;
    wire [31:0] cycles;
    wire [CORE_AW-1:0] img_addr_full;

    nn_accel_core #(
        .N(N), .INPUTS(INPUTS), .HIDDEN(128), .NUM_CLASS(24),
        .ACC_WIDTH(32), .SHIFT1(10), .N_IMAGES(N_IMAGES), .EXT_IMG(1),
        .W1_FILE (W1_FILE),
        .W2_FILE (W2_FILE),
        .B1_FILE (B1_FILE),
        .B2_FILE (B2_FILE)
    ) u_core (
        .clk(clk), .rst(rst), .start(start_core),
        .img_sel(4'd0),                 // one buffered image, so base is 0
        .img_addr(img_addr_full),
        .img_din(img_rd_data),
        .busy(busy), .done(done),
        .pred(pred), .pred_score(pred_score), .cycles(cycles)
    );

    // img_sel is tied off, so the address never leaves the first image's
    // 49 words and the upper bits are constant zero.
    assign img_rd_addr = img_addr_full[IMG_AW-1:0];

    //---------------------------------------------------------------
    // UART transmit: five bytes per result
    //
    //   [0]     class index 0..23
    //   [1..4]  the winning logit, int32 little-endian
    //
    // The score costs 4 bytes on a link that just moved 784, and it turns the
    // host's check from "same class" into "same number". An argmax match only
    // says the hardware landed on the same side of a comparison; the logit is
    // bit-exact or it is not -- which is the property tb_core.v checks in
    // simulation, now available on the live path too.
    //
    // It also makes a wrong answer legible. Image 5 comes back as W with a
    // logit 2.8% above V: with the score in hand the UI can say "narrowly
    // chose W over V" instead of just being wrong.
    //
    // `done` is a level that holds through S_DONE, so the burst is triggered
    // off its rising edge -- driving send from the level would re-trigger the
    // transmitter every cycle until the next image arrived.
    //---------------------------------------------------------------
    localparam integer TX_BYTES = 5;

    reg         done_d;
    reg         tx_send;
    reg  [7:0]  tx_data;
    reg  [39:0] tx_shift;    // {score[31:0], class byte}, sent low byte first
    reg  [2:0]  tx_left;     // bytes still to send
    wire        tx_busy;

    always @(posedge clk) begin
        if (rst) begin
            done_d   <= 1'b0;
            tx_send  <= 1'b0;
            tx_data  <= 8'd0;
            tx_shift <= 40'd0;
            tx_left  <= 3'd0;
        end else begin
            done_d  <= done;
            tx_send <= 1'b0;          // default; one-cycle pulse below

            if (done && !done_d && tx_left == 3'd0) begin
                // Latch the whole reply at once: pred/pred_score are stable
                // from the moment done rises until the next inference starts.
                tx_shift <= {pred_score, 3'b000, pred};
                tx_left  <= TX_BYTES[2:0];
            end else if (tx_left != 3'd0 && !tx_busy && !tx_send) begin
                // busy rises the cycle after a send is accepted, and tx_send is
                // still high on the cycle it is issued, so both terms are
                // needed to keep from double-issuing into the same frame.
                tx_data  <= tx_shift[7:0];
                tx_shift <= {8'd0, tx_shift[39:8]};
                tx_send  <= 1'b1;
                tx_left  <= tx_left - 1'b1;
            end
        end
    end

    uart_tx #(.DIVISOR(BAUD_DIV)) u_tx (
        .clk(clk), .rst(rst),
        .send(tx_send), .data_in(tx_data),
        .tx(RsTx), .busy(tx_busy)
    );

    //---------------------------------------------------------------
    // display
    //---------------------------------------------------------------
    reg result_valid;
    always @(posedge clk) begin
        if (rst)       result_valid <= 1'b0;
        else if (done) result_valid <= 1'b1;
    end

    seven_seg u_disp (
        .clk(clk), .rst(rst),
        .class_idx(pred), .valid(result_valid),
        .an(an), .seg(seg), .dp(dp)
    );

    // cycles[13:0] would never change visibly; the upper bits give a coarse
    // read of the latency that lands in a useful range for ~6,500 cycles.
    // LD13 is the sticky framing-error flag: if it is lit the serial link has
    // mistimed a frame since reset, which is the one failure the host cannot
    // see (the byte is still delivered, so nothing upstream reports it).
    assign led = {done, busy, frame_err_sticky, cycles[17:5]};

endmodule
