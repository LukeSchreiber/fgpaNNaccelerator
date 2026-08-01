`timescale 1ns / 1ps
//=============================================================================
// uart_rx.v -- 8N1 UART receiver, 115200 baud from a 100 MHz clock
//
// DIVISOR = 100e6 / 115200 = 868.06 -> 868 cycles per bit (0.007% error, far
// inside the ~5% budget an 8N1 frame allows). HALF_DIV = 434 places the first
// sample in the middle of the start bit, after which sampling every DIVISOR
// cycles stays centred through the whole frame.
//
// The rx pin is asynchronous to clk, so it passes through two flops before
// anything looks at it. Only rx_sync is used by the FSM -- sampling the raw
// pad would put a metastable value into the shift register.
//
// The mid-start recheck matters: a glitch on an idle line looks exactly like
// a start edge. If rx has gone back high by the centre of the start bit the
// frame is abandoned rather than shifting in eight bits of noise.
//
// data_valid is a one-cycle pulse coincident with data_out becoming valid;
// data_out then holds until the next frame completes.
//=============================================================================

module uart_rx #(
    parameter integer DIVISOR  = 868,   // clk cycles per bit
    parameter integer HALF_DIV = 434    // clk cycles to the centre of a bit
)(
    input  wire       clk,
    input  wire       rst,        // synchronous, active high
    input  wire       rx,         // asynchronous serial input
    output reg  [7:0] data_out,
    output reg        data_valid  // one-cycle pulse
);

    localparam [1:0] S_IDLE  = 2'd0,
                     S_START = 2'd1,
                     S_DATA  = 2'd2,
                     S_STOP  = 2'd3;

    localparam integer CNT_W = $clog2(DIVISOR);

    // Two-FF input synchronizer. Reset high because idle is high -- coming out
    // of reset with zeroes here would look like a start bit.
    reg rx_meta, rx_sync;

    always @(posedge clk) begin
        if (rst) begin
            rx_meta <= 1'b1;
            rx_sync <= 1'b1;
        end else begin
            rx_meta <= rx;
            rx_sync <= rx_meta;
        end
    end

    reg [1:0]       state;
    reg [CNT_W-1:0] cnt;      // cycles elapsed within the current bit
    reg [2:0]       bit_idx;  // which data bit is being received
    reg [7:0]       shift;

    always @(posedge clk) begin
        if (rst) begin
            state      <= S_IDLE;
            cnt        <= {CNT_W{1'b0}};
            bit_idx    <= 3'd0;
            shift      <= 8'd0;
            data_out   <= 8'd0;
            data_valid <= 1'b0;
        end else begin
            data_valid <= 1'b0;   // default; overridden for one cycle below

            case (state)

                // Idle high. A falling edge is a candidate start bit.
                S_IDLE: begin
                    cnt <= {CNT_W{1'b0}};
                    if (~rx_sync)
                        state <= S_START;
                end

                // Wait half a bit, then confirm the line is still low.
                S_START: begin
                    if (cnt == HALF_DIV - 1) begin
                        cnt <= {CNT_W{1'b0}};
                        if (~rx_sync) begin
                            bit_idx <= 3'd0;
                            state   <= S_DATA;
                        end else begin
                            state   <= S_IDLE;   // glitch, not a frame
                        end
                    end else begin
                        cnt <= cnt + 1'b1;
                    end
                end

                // Sample one bit every DIVISOR cycles. The counter was zeroed
                // at the centre of the start bit, so each expiry lands at the
                // centre of the next bit. LSB first: shift right, new bit in
                // at the top, so after eight bits shift[0] is bit 0.
                S_DATA: begin
                    if (cnt == DIVISOR - 1) begin
                        cnt   <= {CNT_W{1'b0}};
                        shift <= {rx_sync, shift[7:1]};
                        if (bit_idx == 3'd7)
                            state <= S_STOP;
                        else
                            bit_idx <= bit_idx + 1'b1;
                    end else begin
                        cnt <= cnt + 1'b1;
                    end
                end

                // Consume the stop bit so the receiver cannot mistake the tail
                // of this frame for the start of the next. The stop bit's level
                // is not checked -- a framing error would only cost the byte
                // that has already been shifted in.
                S_STOP: begin
                    if (cnt == DIVISOR - 1) begin
                        cnt        <= {CNT_W{1'b0}};
                        data_out   <= shift;
                        data_valid <= 1'b1;
                        state      <= S_IDLE;
                    end else begin
                        cnt <= cnt + 1'b1;
                    end
                end

                default: state <= S_IDLE;

            endcase
        end
    end

endmodule
