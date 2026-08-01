`timescale 1ns / 1ps
//=============================================================================
// mac_array.v -- N parallel MAC lanes computing one dot product
//
// DATAFLOW
// --------
// Lane i accumulates the terms at positions i, i+N, i+2N, ... of the dot
// product. After DEPTH/N cycles each lane holds a partial sum; the reduction
// below adds them into the final result.
//
// The alternative (one lane per output neuron, broadcast activation) avoids
// the reduction entirely but needs a strided weight fetch and produces N
// results simultaneously, complicating requantization downstream. This
// arrangement fetches N contiguous weights and N contiguous activations per
// cycle -- a natural BRAM access -- and yields one neuron at a time.
//
// BIAS
// ----
// Fused into lane 0 only. Lanes 1..N-1 clear to zero. Adding it once rather
// than N times keeps the reduction a plain sum.
//
// CRITICAL PATH -- why the reduction is split
// -------------------------------------------
// The reduction used to be one combinational N-input, ACC_WIDTH-bit adder
// hanging off the DSP48 accumulator outputs: 17 logic levels from the P
// register to sum_r, and the longest path in the design.
//
// It is now cut in half. Lanes 0..N/2-1 and N/2..N-1 reduce into two
// registered partial sums, and the single ACC_WIDTH add that combines them
// feeds sum_r in the core on the following cycle:
//
//   cycle m   : acc[] valid          (DSP P registers)
//   cycle m+1 : psum_lo, psum_hi     (half-width reductions, registered here)
//   cycle m+2 : sum_r                (one 32-bit add, registered in the core)
//
// Each stage is now an 8-input tree or a single adder rather than a 16-input
// tree, so neither half carries the whole run.
//
// THIS COSTS ONE CYCLE OF LATENCY, and everything downstream has to know:
// the core's capture strobe is delayed one more stage and DRAIN_CYCLES grows
// by one. Getting that wrong does not produce a timing failure, it produces
// captured results that belong to the wrong neuron.
//
// N must be even. It is a power of two in every configuration used here, and
// the halves are the natural split; an odd N would silently drop a lane.
//
// PORT WIDTHS
// -----------
// Verilog-2001 cannot pass arrays through ports, so the N activations and N
// weights arrive as flattened vectors and are sliced with the +: operator.
//=============================================================================

module mac_array #(
    parameter integer N         = 16,   // lanes; must divide the dot-product depth
    parameter integer A_WIDTH   = 8,
    parameter integer W_WIDTH   = 8,
    parameter integer ACC_WIDTH = 32
)(
    input  wire                        clk,
    input  wire                        rst,
    input  wire                        en,      // accumulate this cycle
    input  wire                        clr,     // first cycle of a new neuron
    input  wire [N*A_WIDTH-1:0]        a_flat,  // N unsigned activations
    input  wire [N*W_WIDTH-1:0]        w_flat,  // N signed weights
    input  wire signed [ACC_WIDTH-1:0] bias,    // at accumulator scale
    output wire signed [ACC_WIDTH-1:0] sum      // acc[] reduced, one cycle behind
);

    wire signed [ACC_WIDTH-1:0] acc [0:N-1];

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : lane
            mac_unit #(
                .A_WIDTH(A_WIDTH),
                .W_WIDTH(W_WIDTH),
                .ACC_WIDTH(ACC_WIDTH)
            ) u_mac (
                .clk  (clk),
                .rst  (rst),
                .en   (en),
                .clr  (clr),
                .a    (a_flat[i*A_WIDTH +: A_WIDTH]),
                .w    ($signed(w_flat[i*W_WIDTH +: W_WIDTH])),
                .init ((i == 0) ? bias : {ACC_WIDTH{1'b0}}),
                .acc  (acc[i])
            );
        end
    endgenerate

    //-----------------------------------------------------------------
    // stage 1: two half-width reductions, registered
    //-----------------------------------------------------------------
    localparam integer HALF = N / 2;

    integer k;
    reg signed [ACC_WIDTH-1:0] psum_lo_c, psum_hi_c;

    always @(*) begin
        psum_lo_c = {ACC_WIDTH{1'b0}};
        psum_hi_c = {ACC_WIDTH{1'b0}};
        for (k = 0; k < HALF; k = k + 1)
            psum_lo_c = psum_lo_c + acc[k];
        for (k = HALF; k < N; k = k + 1)
            psum_hi_c = psum_hi_c + acc[k];
    end

    reg signed [ACC_WIDTH-1:0] psum_lo, psum_hi;

    always @(posedge clk) begin
        if (rst) begin
            psum_lo <= {ACC_WIDTH{1'b0}};
            psum_hi <= {ACC_WIDTH{1'b0}};
        end else begin
            psum_lo <= psum_lo_c;
            psum_hi <= psum_hi_c;
        end
    end

    //-----------------------------------------------------------------
    // stage 2: one add, registered by the core into sum_r
    //
    // Unconditional, like the register it feeds: each neuron's accumulator is
    // final for exactly one cycle and every value walks the same two stages,
    // so results stay in order and cannot alias. The capture strobe picks the
    // cycle that matters.
    //-----------------------------------------------------------------
    assign sum = psum_lo + psum_hi;

endmodule
