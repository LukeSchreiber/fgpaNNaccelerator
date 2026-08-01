`timescale 1ns / 1ps
//=============================================================================
// mac_unit.v -- one pipelined multiply-accumulate lane
//
//   stage 1 (posedge):  prod <= a * w          [DSP48E1 M register]
//   stage 2 (posedge):  acc  <= sel + prod     [DSP48E1 P register]
//                       sel = clr ? init : acc
//
// LATENCY: results appear 2 clocks after the operands are presented.
// Throughput is still one MAC per cycle -- the pipeline fills and drains.
// After the final operand, hold for one more clock before reading acc.
//
// WHY THE PIPELINE REGISTER EXISTS
// --------------------------------
// A DSP48E1 has a register (M) between its multiplier and its ALU. Writing
//
//     acc <= (clr ? init : acc) + a*w;
//
// puts the multiply and the add in one combinational hop, which corresponds
// to MREG=0 on the DSP -- legal, slow, and outside the pattern Vivado's
// inference engine matches. The result is that the entire multiplier gets
// built out of LUTs and carry chains while the DSP slices sit idle. Measured
// on this design at N=16: 1,826 LUTs and 0 DSPs.
//
// Registering the product matches the canonical MACC template. The `use_dsp`
// attribute alone does NOT fix this -- the mismatch is structural, not a
// cost-model preference.
//
// CONTROL ALIGNMENT
// -----------------
// en, clr and init are delayed one cycle so they arrive at stage 2 alongside
// the product they belong to. Forgetting this is the classic pipelining bug:
// the bias lands on the wrong neuron and every result is quietly off.
//
// SIGNEDNESS
// ----------
// Activations are unsigned (pixel 0..255, ReLU output 0..127); weights are
// signed. Verilog evaluates a mixed expression as unsigned, so the activation
// is zero-extended one bit and cast signed before the multiply.
//=============================================================================

(* use_dsp = "yes" *)
module mac_unit #(
    parameter integer A_WIDTH   = 8,
    parameter integer W_WIDTH   = 8,
    parameter integer ACC_WIDTH = 32
)(
    input  wire                          clk,
    input  wire                          rst,   // synchronous, active high
    input  wire                          en,
    input  wire                          clr,   // first term: load init + prod
    input  wire        [A_WIDTH-1:0]     a,     // unsigned activation
    input  wire signed [W_WIDTH-1:0]     w,     // signed weight
    input  wire signed [ACC_WIDTH-1:0]   init,  // bias, at accumulator scale
    output reg  signed [ACC_WIDTH-1:0]   acc
);

    localparam integer P_WIDTH = A_WIDTH + W_WIDTH + 1;

    wire signed [A_WIDTH:0] a_signed = $signed({1'b0, a});

    // ---- stage 1: multiply --------------------------------------------
    reg signed [P_WIDTH-1:0]   prod;
    reg                        en_d;
    reg                        clr_d;
    reg signed [ACC_WIDTH-1:0] init_d;

    always @(posedge clk) begin
        prod   <= a_signed * w;
        init_d <= init;
        if (rst) begin
            en_d  <= 1'b0;
            clr_d <= 1'b0;
        end else begin
            en_d  <= en;
            clr_d <= clr;
        end
    end

    // ---- stage 2: accumulate ------------------------------------------
    wire signed [ACC_WIDTH-1:0] prod_ext =
        {{(ACC_WIDTH-P_WIDTH){prod[P_WIDTH-1]}}, prod};

    always @(posedge clk) begin
        if (rst)
            acc <= {ACC_WIDTH{1'b0}};
        else if (en_d)
            acc <= (clr_d ? init_d : acc) + prod_ext;
    end

endmodule
