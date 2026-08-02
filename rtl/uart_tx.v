`timescale 1ns / 1ps
//=============================================================================
// uart_tx.v -- 8N1 UART transmitter, rate set by DIVISOR from a 100 MHz clock
//
// DIVISOR must match uart_rx. The deployed rate is 921600 baud -> 109 cycles
// per bit; the default below is the 115200 value uart_loopback_top still uses,
// and nn_accel_top overrides it.
//
// The transmitter needs no half-bit offset: it defines the bit boundaries
// rather than hunting for them, so a single DIVISOR counter paces the frame.
//
// tx is registered, not decoded combinationally from the state and counter.
// It drives a pin, and a glitch on a serial line is a corrupt frame at the
// far end -- one flop is cheaper than debugging that.
//
// Handshake: `send` is a one-cycle pulse, sampled only in IDLE, so a pulse
// arriving mid-frame is ignored rather than corrupting the byte in flight.
// data_in is latched on that pulse and need not be held afterwards. busy
// rises the cycle after send is accepted and falls at the end of the stop
// bit, so `!busy` is the condition for offering the next byte.
//=============================================================================

module uart_tx #(
    parameter integer DIVISOR = 868     // clk cycles per bit
)(
    input  wire       clk,
    input  wire       rst,       // synchronous, active high
    input  wire       send,      // one-cycle pulse, accepted only when idle
    input  wire [7:0] data_in,   // latched on send
    output reg        tx,        // serial output, idle high
    output reg        busy
);

    localparam [1:0] S_IDLE  = 2'd0,
                     S_START = 2'd1,
                     S_DATA  = 2'd2,
                     S_STOP  = 2'd3;

    localparam integer CNT_W = $clog2(DIVISOR);

    reg [1:0]       state;
    reg [CNT_W-1:0] cnt;      // cycles elapsed within the current bit
    reg [2:0]       bit_idx;  // which data bit is on the wire
    reg [7:0]       shift;

    // True on the last clock of a bit period. Every state advances on it, so
    // each bit occupies exactly DIVISOR cycles.
    wire bit_done = (cnt == DIVISOR - 1);

    always @(posedge clk) begin
        if (rst) begin
            state   <= S_IDLE;
            cnt     <= {CNT_W{1'b0}};
            bit_idx <= 3'd0;
            shift   <= 8'd0;
            tx      <= 1'b1;    // idle high, including during reset
            busy    <= 1'b0;
        end else begin
            case (state)

                S_IDLE: begin
                    cnt <= {CNT_W{1'b0}};
                    tx  <= 1'b1;
                    if (send) begin
                        shift <= data_in;
                        tx    <= 1'b0;    // start bit goes out immediately
                        busy  <= 1'b1;
                        state <= S_START;
                    end else begin
                        busy  <= 1'b0;
                    end
                end

                // tx already holds the start bit; just time it out.
                S_START: begin
                    if (bit_done) begin
                        cnt     <= {CNT_W{1'b0}};
                        tx      <= shift[0];    // LSB first
                        bit_idx <= 3'd0;
                        state   <= S_DATA;
                    end else begin
                        cnt <= cnt + 1'b1;
                    end
                end

                // Shift right at each bit boundary so the next LSB is at
                // shift[0], and drive it. After the eighth bit, tx goes high
                // for the stop bit.
                S_DATA: begin
                    if (bit_done) begin
                        cnt   <= {CNT_W{1'b0}};
                        shift <= {1'b0, shift[7:1]};
                        if (bit_idx == 3'd7) begin
                            tx    <= 1'b1;
                            state <= S_STOP;
                        end else begin
                            tx      <= shift[1];
                            bit_idx <= bit_idx + 1'b1;
                        end
                    end else begin
                        cnt <= cnt + 1'b1;
                    end
                end

                // Hold the stop bit for a full bit time before releasing busy,
                // so a receiver always sees the frame boundary even if the
                // next send arrives on the very next cycle.
                S_STOP: begin
                    if (bit_done) begin
                        cnt   <= {CNT_W{1'b0}};
                        tx    <= 1'b1;
                        busy  <= 1'b0;
                        state <= S_IDLE;
                    end else begin
                        cnt <= cnt + 1'b1;
                    end
                end

                default: state <= S_IDLE;

            endcase
        end
    end

endmodule
