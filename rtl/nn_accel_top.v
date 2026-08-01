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
//   LD[13:0]      cycle count, upper bits (a crude latency display)
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
//   core runs ~6,478 cycles (65 us, far shorter than the 68 ms the transfer
//                             itself takes at 115200 baud)
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
// BAUD vs LATENCY: 784 bytes at 115200 8N1 is ~68 ms, three orders of
// magnitude longer than the inference. The link, not the accelerator, sets the
// frame rate, which is why no attempt is made to overlap load and compute.
//=============================================================================

module nn_accel_top (
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

    // 100 MHz / 115200 baud. HALF_DIV centres the first sample in the start bit.
    localparam integer BAUD_DIV  = 868;
    localparam integer BAUD_HALF = 434;

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

    reg [19:0] db_cnt;
    reg        btnC_stable, btnC_prev;
    always @(posedge clk) begin
        if (btnC_sync[1] != btnC_stable) begin
            db_cnt <= db_cnt + 1'b1;
            if (&db_cnt) btnC_stable <= btnC_sync[1];
        end else begin
            db_cnt <= 20'd0;
        end
        btnC_prev <= btnC_stable;
    end

    wire rerun_pulse = btnC_stable & ~btnC_prev;   // one cycle on press

    //---------------------------------------------------------------
    // UART receive -> image buffer
    //---------------------------------------------------------------
    wire [7:0] rx_data;
    wire       rx_valid;

    uart_rx #(.DIVISOR(BAUD_DIV), .HALF_DIV(BAUD_HALF)) u_rx (
        .clk(clk), .rst(rst), .rx(RsRx),
        .data_out(rx_data), .data_valid(rx_valid)
    );

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
        .W1_FILE ("/home/yetigod/fpga-nn-accel/mem/w1_packed.mem"),
        .W2_FILE ("/home/yetigod/fpga-nn-accel/mem/w2_packed.mem"),
        .B1_FILE ("/home/yetigod/fpga-nn-accel/mem/b1_packed.mem"),
        .B2_FILE ("/home/yetigod/fpga-nn-accel/mem/b2_packed.mem")
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
    // UART transmit: one byte per result
    //
    // `done` is a level that holds through S_DONE, so the send pulse comes off
    // its rising edge -- driving send from the level would re-trigger the
    // transmitter every cycle until the next image arrived.
    //---------------------------------------------------------------
    reg        done_d;
    reg        tx_send;
    reg  [7:0] tx_data;
    wire       tx_busy;

    always @(posedge clk) begin
        if (rst) begin
            done_d  <= 1'b0;
            tx_send <= 1'b0;
            tx_data <= 8'd0;
        end else begin
            done_d  <= done;
            tx_send <= 1'b0;
            if (done && !done_d && !tx_busy) begin
                tx_data <= {3'b000, pred};   // class index 0..23, raw byte
                tx_send <= 1'b1;
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

    // cycles[13:0] would never change visibly; bits [17:4] give a coarse
    // read of the latency that lands in a useful range for ~6,500 cycles.
    assign led = {done, busy, cycles[17:4]};

endmodule
