`timescale 1ns / 1ps
//=============================================================================
// img_loader.v -- collect one image byte stream from uart_rx into a BRAM
//
// The host sends IMG_BYTES raw pixel bytes back to back, no header and no
// framing. After the last byte img_ready rises and stays high until the
// consumer pulses img_ack.
//
// PACKING -- must match pack_mem.py, or every prediction is quietly wrong
// ----------------------------------------------------------------------
// The MAC array eats N activations per cycle and slices its input as
// a_flat[j*8 +: 8], so lane j is the LOW-to-HIGH byte j of the word. Pixel p
// therefore belongs at word p/N, lane p%N. Bytes arrive in raster order, so
// shifting each new byte in at the TOP of a register and right-shifting walks
// the first byte of the group down to bits [7:0] exactly as the group closes:
//
//   byte 0 in -> [127:120] ... after 15 more shifts -> [7:0]   = lane 0
//   byte 15 in                                      -> [127:120] = lane 15
//
// This is the same layout images_packed.mem had, so nothing downstream of the
// image port changes.
//
// WHY A SHIFT REGISTER AND NOT BYTE-ENABLED WRITES
// ------------------------------------------------
// A 128-bit-wide BRAM written one byte at a time needs 16 byte-enables and the
// matching inference template. IMG_BYTES is an exact multiple of N (784 = 49*16),
// so there is never a partial word: staging N bytes in a register and writing
// the full word once costs one 128-bit register and keeps the memory a plain
// single-write-port BRAM.
//
// The word is written from {rx_data, shift[...:8]} rather than from `shift`,
// because on the cycle the last byte of a group arrives `shift` still holds
// only the previous N-1 bytes -- the new one has not been clocked in yet.
//
// BACKPRESSURE
// ------------
// While img_ready is high the buffer holds an image the core is (or is about
// to be) reading, so incoming bytes are DROPPED rather than overwriting it.
// The host is expected to wait for the result byte before sending the next
// image. Dropping is the safe failure: a torn image would be scored and the
// wrong answer reported as if it were real.
//
// There is no timeout. A short stream leaves the loader waiting for the
// missing bytes and img_ready simply never rises; the next stream continues
// filling from where the last one stopped. Recovery is a reset.
//=============================================================================

module img_loader #(
    parameter integer N         = 16,    // bytes per packed word == MAC lanes
    parameter integer IMG_BYTES = 784    // must be a multiple of N
)(
    input  wire                             clk,
    input  wire                             rst,        // synchronous, active high

    // byte stream in, straight off uart_rx
    input  wire [7:0]                       rx_data,
    input  wire                             rx_valid,   // one-cycle pulse

    // handshake with the inference core
    output reg                              img_ready,  // level, held until ack
    input  wire                             img_ack,    // one-cycle pulse

    // read port for the core: same shape and latency as rom_sync
    input  wire [$clog2(IMG_BYTES/N)-1:0]   rd_addr,
    output reg  [N*8-1:0]                   rd_data
);

    localparam integer DEPTH = IMG_BYTES / N;      // 49 words
    localparam integer AW    = $clog2(DEPTH);      //  6 bits
    localparam integer LW    = $clog2(N);          //  4 bits, N a power of two
    localparam integer CW    = $clog2(IMG_BYTES);  // 10 bits, counts 0..783

    (* ram_style = "block" *)
    reg [N*8-1:0] mem [0:DEPTH-1];

    reg [N*8-1:0] shift;      // staging register for the word being assembled
    reg [LW-1:0]  lane;       // which byte of the current word arrives next
    reg [AW-1:0]  wr_addr;
    reg [CW-1:0]  byte_cnt;   // bytes of this image received so far

    // A byte is taken only while the buffer is free.
    wire accept   = rx_valid && !img_ready;
    wire word_end = (lane == N - 1);
    wire img_end  = (byte_cnt == IMG_BYTES - 1);

    wire [N*8-1:0] word_next = {rx_data, shift[N*8-1:8]};

    //-----------------------------------------------------------------
    // write port -- one full word every N accepted bytes
    //-----------------------------------------------------------------
    always @(posedge clk) begin
        if (accept && word_end)
            mem[wr_addr] <= word_next;
    end

    //-----------------------------------------------------------------
    // read port -- registered, so data is valid one cycle after rd_addr,
    // matching rom_sync and therefore the core's pipeline alignment
    //-----------------------------------------------------------------
    always @(posedge clk) begin
        rd_data <= mem[rd_addr];
    end

    //-----------------------------------------------------------------
    // stream sequencing
    //-----------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            shift     <= {(N*8){1'b0}};
            lane      <= {LW{1'b0}};
            wr_addr   <= {AW{1'b0}};
            byte_cnt  <= {CW{1'b0}};
            img_ready <= 1'b0;
        end else begin
            if (img_ack)
                img_ready <= 1'b0;

            if (accept) begin
                shift <= word_next;

                if (word_end) begin
                    lane    <= {LW{1'b0}};
                    wr_addr <= wr_addr + 1'b1;
                end else begin
                    lane    <= lane + 1'b1;
                end

                if (img_end) begin
                    // Rewind for the next image. These override the word_end
                    // updates above, which is the intent: 784 is a multiple of
                    // N, so the last byte closes a word and finishes the image
                    // on the same cycle.
                    byte_cnt  <= {CW{1'b0}};
                    wr_addr   <= {AW{1'b0}};
                    lane      <= {LW{1'b0}};
                    // Set after the img_ack clear above so a completing image
                    // wins over an ack arriving in the same cycle.
                    img_ready <= 1'b1;
                end else begin
                    byte_cnt <= byte_cnt + 1'b1;
                end
            end
        end
    end

endmodule
