`timescale 1ns / 1ps
//=============================================================================
// nn_accel_core.v -- full inference engine
//
//   784 -> HIDDEN -> NUM_CLASS, int8 weights, int32 accumulators
//
// Contains the weight/bias/image ROMs, the MAC array, the requantizer, the
// activation buffer, the argmax and the sequencing FSM. The top level only
// has to supply a clock, a start pulse and an image index.
//
// IMAGE SOURCE: EXT_IMG=0 (default) keeps the internal image ROM loaded from
// IMG_FILE. EXT_IMG=1 drops that ROM and exposes img_addr/img_din so the image
// can come from somewhere else -- a UART-fed buffer, for instance. img_din
// must be registered on the read, i.e. valid one cycle after img_addr, exactly
// like rom_sync; the whole pipeline alignment depends on that latency.
//
// PIPELINE ALIGNMENT -- the whole design hinges on this
// -----------------------------------------------------
//   cycle k   : address issued to ROM
//   cycle k+1 : ROM data valid on the MAC inputs   (rom_sync latency 1)
//   cycle k+2 : product registered                 (mac_unit stage 1)
//   cycle k+3 : accumulator updated                (mac_unit stage 2)
//   cycle k+4 : half reductions registered         (mac_array psum_lo/psum_hi)
//   cycle k+5 : reduction result registered        (sum_r)
//
// So `en` and `clr` are delayed ONE cycle from address generation to arrive
// with the data they belong to, and the capture strobe is delayed FIVE.
// The neuron index rides the same delay chain so a captured result knows
// which output it belongs to.
//
// WHY sum IS REGISTERED, AND WHY THE REDUCTION IS SPLIT
// -----------------------------------------------------
// `sum` off mac_array reduces the DSP accumulators. Feeding that reduction
// straight into requantize AND the argmax comparator merged all three into one
// ~15-level path: 12.7 ns, WNS -3.377 ns at 100 MHz. sum_r split it --
// accumulators -> reduction in one cycle, reduction -> requantize/compare in
// the next.
//
// That left the reduction itself as the longest run: a 16-input 32-bit adder,
// 17 logic levels from the DSP P register to sum_r. mac_array now reduces
// lanes 0-7 and 8-15 into two REGISTERED partial sums, and the single add that
// combines them feeds sum_r on the following cycle.
//
// Each of those splits costs one stage on the capture chain -- c3 for sum_r,
// c4 for the partial sums -- and one drain cycle per layer. The strobe must
// follow the data: a capture chain one stage short does not fail timing, it
// files each neuron's result under the wrong index and quietly ruins argmax.
//
// WHY NEURONS CAN STREAM BACK-TO-BACK
// -----------------------------------
// Addresses run continuously; `clr` restarts the accumulator every CHUNKS
// cycles. That means neuron i+1 begins accumulating before neuron i's sum
// has been read. Working the timeline through:
//
//   neuron i   last data at cycle k    -> acc holds neuron i during k+2
//   neuron i+1 first data at cycle k+1 -> acc holds neuron i+1 from k+3
//
// Neuron i's sum is valid for exactly one cycle, and the capture register
// samples on the edge ENDING that cycle -- reading the old accumulator value
// while the new one is being written. Zero gap cycles.
//
// A per-neuron drain would have been easier to reason about and cost
// 128 * 3 = 384 extra cycles in layer 1 (about 6%).
//
// LATENCY (N=16, HIDDEN=128, NUM_CLASS=24)
//   layer 1  128 * 49 = 6272 cycles
//   layer 2   24 *  8 =  192 cycles
//   drains     2 *  8 =   16 cycles   (DRAIN_CYCLES+1 each)
//   total                6480 cycles = 64.8 us at 100 MHz
//
// Was 6476 with the reduction combinational and 6478 with sum_r; each pipeline
// stage costs one drain cycle per layer, so 2 cycles out of 6,478 -- 0.03% of
// the runtime to halve the longest path.
//=============================================================================

module nn_accel_core #(
    parameter integer N         = 16,
    parameter integer INPUTS    = 784,
    parameter integer HIDDEN    = 128,
    parameter integer NUM_CLASS = 24,
    parameter integer ACC_WIDTH = 32,
    parameter integer SHIFT1    = 10,
    parameter integer N_IMAGES  = 16,
    parameter integer EXT_IMG   = 0,    // 1: images arrive on img_din, not from IMG_FILE
    parameter         W1_FILE   = "",
    parameter         W2_FILE   = "",
    parameter         B1_FILE   = "",
    parameter         B2_FILE   = "",
    parameter         IMG_FILE  = ""
)(
    input  wire                             clk,
    input  wire                             rst,
    input  wire                             start,
    input  wire [$clog2(N_IMAGES)-1:0]      img_sel,
    // external image port, used only when EXT_IMG=1. img_din must have the
    // same one-cycle read latency as the internal ROM it replaces.
    output wire [$clog2(N_IMAGES*(INPUTS/N))-1:0] img_addr,
    input  wire [N*8-1:0]                   img_din,
    output reg                              busy,
    output reg                              done,
    output wire [$clog2(NUM_CLASS)-1:0]     pred,
    output wire signed [ACC_WIDTH-1:0]      pred_score,
    output reg  [31:0]                      cycles
);

    localparam integer L1_CHUNKS = INPUTS / N;              // 49
    localparam integer L2_CHUNKS = HIDDEN / N;              //  8
    localparam integer W1_DEPTH  = HIDDEN * L1_CHUNKS;      // 6272
    localparam integer W2_DEPTH  = NUM_CLASS * L2_CHUNKS;   //  192
    localparam integer IMG_DEPTH = N_IMAGES * L1_CHUNKS;    //  784

    localparam integer WAW  = $clog2(W1_DEPTH);
    localparam integer IAW  = $clog2(IMG_DEPTH);
    localparam integer NAW  = $clog2(HIDDEN);
    localparam integer CLSW = $clog2(NUM_CLASS);

    localparam [2:0] S_IDLE = 3'd0,
                     S_L1   = 3'd1,
                     S_L1D  = 3'd2,
                     S_L2   = 3'd3,
                     S_L2D  = 3'd4,
                     S_DONE = 3'd5;

    localparam integer DRAIN_CYCLES = 7;   // 5 needed (psum + sum_r); 2 spare

    //-----------------------------------------------------------------
    // stage A: address generation
    //-----------------------------------------------------------------
    reg  [2:0]        state;
    reg  [15:0]       cnt_chunk;
    reg  [NAW:0]      cnt_neuron;
    reg  [WAW-1:0]    addr_w;
    reg  [IAW-1:0]    img_base;
    reg  [3:0]        drain;

    wire is_l2   = (state == S_L2) || (state == S_L2D);
    wire run_a   = (state == S_L1) || (state == S_L2);

    wire [7:0]   chunks_last  = is_l2 ? (L2_CHUNKS - 1) : (L1_CHUNKS - 1);
    wire [NAW:0] neurons_last = is_l2 ? (NUM_CLASS - 1) : (HIDDEN - 1);

    wire first_a = (cnt_chunk == 8'd0);
    wire last_a  = (cnt_chunk == chunks_last);
    wire layer_done = run_a && last_a && (cnt_neuron == neurons_last);

    always @(posedge clk) begin
        if (rst) begin
            state      <= S_IDLE;
            cnt_chunk  <= 8'd0;
            cnt_neuron <= {(NAW+1){1'b0}};
            addr_w     <= {WAW{1'b0}};
            img_base   <= {IAW{1'b0}};
            drain      <= 4'd0;
            busy       <= 1'b0;
            done       <= 1'b0;
            cycles     <= 32'd0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        state      <= S_L1;
                        cnt_chunk  <= 8'd0;
                        cnt_neuron <= {(NAW+1){1'b0}};
                        addr_w     <= {WAW{1'b0}};
                        img_base   <= img_sel * L1_CHUNKS;
                        busy       <= 1'b1;
                        cycles     <= 32'd0;
                    end
                end

                S_L1, S_L2: begin
                    cycles <= cycles + 32'd1;
                    addr_w <= addr_w + {{(WAW-1){1'b0}}, 1'b1};
                    if (last_a) begin
                        cnt_chunk <= 8'd0;
                        if (cnt_neuron == neurons_last) begin
                            state <= (state == S_L1) ? S_L1D : S_L2D;
                            drain <= DRAIN_CYCLES;
                        end else begin
                            cnt_neuron <= cnt_neuron + 1'b1;
                        end
                    end else begin
                        cnt_chunk <= cnt_chunk + 8'd1;
                    end
                end

                S_L1D: begin
                    cycles <= cycles + 32'd1;
                    if (drain == 4'd0) begin
                        state      <= S_L2;
                        cnt_chunk  <= 8'd0;
                        cnt_neuron <= {(NAW+1){1'b0}};
                        addr_w     <= {WAW{1'b0}};
                    end else
                        drain <= drain - 4'd1;
                end

                S_L2D: begin
                    cycles <= cycles + 32'd1;
                    if (drain == 4'd0) begin
                        state <= S_DONE;
                        busy  <= 1'b0;
                        done  <= 1'b1;
                    end else
                        drain <= drain - 4'd1;
                end

                S_DONE: begin
                    if (start) begin
                        state      <= S_L1;
                        cnt_chunk  <= 8'd0;
                        cnt_neuron <= {(NAW+1){1'b0}};
                        addr_w     <= {WAW{1'b0}};
                        img_base   <= img_sel * L1_CHUNKS;
                        busy       <= 1'b1;
                        done       <= 1'b0;
                        cycles     <= 32'd0;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    //-----------------------------------------------------------------
    // memories -- all addressed in stage A, data arrives in stage B
    //-----------------------------------------------------------------
    wire [N*8-1:0]           w1_dout, w2_dout;
    wire [N*8-1:0]           img_dout;
    wire [ACC_WIDTH-1:0]     b1_dout, b2_dout;

    rom_sync #(.WIDTH(N*8), .DEPTH(W1_DEPTH), .INIT_FILE(W1_FILE)) u_w1 (
        .clk(clk), .en(1'b1), .addr(addr_w[WAW-1:0]), .dout(w1_dout));

    rom_sync #(.WIDTH(N*8), .DEPTH(W2_DEPTH), .INIT_FILE(W2_FILE)) u_w2 (
        .clk(clk), .en(1'b1), .addr(addr_w[$clog2(W2_DEPTH)-1:0]),
        .dout(w2_dout));

    // The image address is always driven out; with EXT_IMG the ROM that used
    // to answer it is gone and an external buffer (img_loader) answers instead.
    // Everything downstream is unchanged -- same width, same packing, same
    // one-cycle latency -- so the pipeline alignment above still holds.
    assign img_addr = img_base + cnt_chunk[IAW-1:0];

    generate
        if (EXT_IMG) begin : g_img_ext
            assign img_dout = img_din;
        end else begin : g_img_rom
            rom_sync #(.WIDTH(N*8), .DEPTH(IMG_DEPTH), .INIT_FILE(IMG_FILE)) u_img (
                .clk(clk), .en(1'b1), .addr(img_addr), .dout(img_dout));
        end
    endgenerate

    rom_sync #(.WIDTH(ACC_WIDTH), .DEPTH(HIDDEN), .INIT_FILE(B1_FILE)) u_b1 (
        .clk(clk), .en(1'b1), .addr(cnt_neuron[NAW-1:0]), .dout(b1_dout));

    rom_sync #(.WIDTH(ACC_WIDTH), .DEPTH(NUM_CLASS), .INIT_FILE(B2_FILE)) u_b2 (
        .clk(clk), .en(1'b1), .addr(cnt_neuron[CLSW-1:0]), .dout(b2_dout));

    //-----------------------------------------------------------------
    // activation buffer: layer 1 writes one byte per neuron, layer 2
    // reads N contiguous bytes per cycle. Registered on read so it lines
    // up with the ROM outputs, which have one cycle of latency.
    //-----------------------------------------------------------------
    reg [7:0] act_mem [0:HIDDEN-1];

    wire [N*8-1:0] act_bus;
    genvar g;
    generate
        for (g = 0; g < N; g = g + 1) begin : act_rd
            assign act_bus[g*8 +: 8] = act_mem[cnt_chunk * N + g];
        end
    endgenerate

    reg [N*8-1:0] act_bus_r;
    always @(posedge clk) act_bus_r <= act_bus;

    //-----------------------------------------------------------------
    // stage B: control aligned with data
    //-----------------------------------------------------------------
    reg           en_b, clr_b, last_b, l2_b;
    reg [NAW:0]   neuron_b;

    always @(posedge clk) begin
        if (rst) begin
            en_b <= 1'b0; clr_b <= 1'b0; last_b <= 1'b0; l2_b <= 1'b0;
            neuron_b <= {(NAW+1){1'b0}};
        end else begin
            en_b     <= run_a;
            clr_b    <= run_a && first_a;
            last_b   <= run_a && last_a;
            l2_b     <= is_l2;
            neuron_b <= cnt_neuron;
        end
    end

    wire [N*8-1:0]           w_data = l2_b ? w2_dout : w1_dout;
    wire [N*8-1:0]           x_data = l2_b ? act_bus_r : img_dout;
    wire signed [ACC_WIDTH-1:0] bias = l2_b ? $signed(b2_dout)
                                            : $signed(b1_dout);

    //-----------------------------------------------------------------
    // compute
    //-----------------------------------------------------------------
    wire signed [ACC_WIDTH-1:0] sum;

    mac_array #(
        .N(N), .A_WIDTH(8), .W_WIDTH(8), .ACC_WIDTH(ACC_WIDTH)
    ) u_mac (
        .clk(clk), .rst(rst), .en(en_b), .clr(clr_b),
        .a_flat(x_data), .w_flat(w_data), .bias(bias), .sum(sum)
    );

    // pipeline register between the reduction and its two consumers
    reg signed [ACC_WIDTH-1:0] sum_r;

    always @(posedge clk) begin
        if (rst) sum_r <= {ACC_WIDTH{1'b0}};
        else     sum_r <= sum;
    end

    wire [7:0] act_val;
    requantize #(
        .ACC_WIDTH(ACC_WIDTH), .OUT_WIDTH(8), .SHIFT(SHIFT1)
    ) u_rq (.acc(sum_r), .act(act_val));

    //-----------------------------------------------------------------
    // capture: last_b delayed four more cycles to land on the one cycle
    // where `sum_r` holds this neuron's finished dot product
    //
    // One stage per register between the MAC inputs and sum_r: c1/c2 for the
    // mac_unit product and accumulator, c3 for mac_array's partial sums, c4 for
    // sum_r itself. Add a pipeline stage anywhere in that path and one must be
    // added here too.
    //-----------------------------------------------------------------
    reg         last_c1, last_c2, last_c3, last_c4;
    reg         l2_c1,   l2_c2,   l2_c3,   l2_c4;
    reg [NAW:0] neuron_c1, neuron_c2, neuron_c3, neuron_c4;

    always @(posedge clk) begin
        if (rst) begin
            last_c1 <= 1'b0; last_c2 <= 1'b0; last_c3 <= 1'b0; last_c4 <= 1'b0;
            l2_c1   <= 1'b0; l2_c2   <= 1'b0; l2_c3   <= 1'b0; l2_c4   <= 1'b0;
            neuron_c1 <= {(NAW+1){1'b0}};
            neuron_c2 <= {(NAW+1){1'b0}};
            neuron_c3 <= {(NAW+1){1'b0}};
            neuron_c4 <= {(NAW+1){1'b0}};
        end else begin
            last_c1   <= last_b;    last_c2   <= last_c1;
            last_c3   <= last_c2;   last_c4   <= last_c3;
            l2_c1     <= l2_b;      l2_c2     <= l2_c1;
            l2_c3     <= l2_c2;     l2_c4     <= l2_c3;
            neuron_c1 <= neuron_b;  neuron_c2 <= neuron_c1;
            neuron_c3 <= neuron_c2; neuron_c4 <= neuron_c3;
        end
    end

    wire capture = last_c4;

    always @(posedge clk) begin
        if (capture && !l2_c4)
            act_mem[neuron_c4[NAW-1:0]] <= act_val;
    end

    argmax #(.WIDTH(ACC_WIDTH), .COUNT(NUM_CLASS)) u_am (
        .clk(clk), .rst(rst),
        .en   (capture && l2_c4),
        .first(capture && l2_c4 && (neuron_c4 == 0)),
        .value(sum_r),
        .index(neuron_c4[CLSW-1:0]),
        .best_idx(pred), .best_val(pred_score)
    );

endmodule
