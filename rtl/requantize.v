`timescale 1ns / 1ps
//=============================================================================
// requantize.v -- ReLU + power-of-two requantization + saturation
//
//   act = clip(acc >>> SHIFT, 0, 2^OUT_WIDTH - 1)
//
// Purely combinational; the controller registers the result.
//
// MUST MATCH THE PYTHON GOLDEN MODEL EXACTLY:
//     np.clip(acc1 >> shift1, 0, 255)
//
// THREE THINGS THAT SILENTLY BREAK THIS
// -------------------------------------
// 1. >>> vs >>.  NumPy's >> on a negative integer is an ARITHMETIC shift:
//    it floors toward -inf, so -1 >> 10 == -1, not 0. Verilog's >>> on a
//    signed operand does the same; >> (logical) would turn every negative
//    accumulator into a huge positive and the ReLU would fire backwards.
//
// 2. ReLU placement. Clipping at 0 after the shift is equivalent to ReLU
//    before it, because an arithmetic right shift is monotonic. Folding
//    them into one clamp costs nothing.
//
// 3. Saturation, not wrap. An accumulator above 255<<SHIFT must clamp to
//    255. Truncating the low bits instead would alias large activations to
//    small ones -- the classic overflow bug that shows up as a handful of
//    wildly wrong classifications rather than a uniform accuracy loss.
//
// A power-of-two shift is used rather than an exact fractional rescale so
// this is a barrel shifter instead of a multiplier plus rounding unit. The
// accuracy cost is measured in the results table.
//=============================================================================

module requantize #(
    parameter integer ACC_WIDTH = 32,
    parameter integer OUT_WIDTH = 8,
    parameter integer SHIFT     = 10
)(
    input  wire signed [ACC_WIDTH-1:0] acc,
    output wire        [OUT_WIDTH-1:0] act
);

    localparam signed [ACC_WIDTH-1:0] MAXV = (1 << OUT_WIDTH) - 1;

    wire signed [ACC_WIDTH-1:0] shifted = acc >>> SHIFT;

    assign act = (shifted <= 0)    ? {OUT_WIDTH{1'b0}} :
                 (shifted >= MAXV) ? {OUT_WIDTH{1'b1}} :
                                     shifted[OUT_WIDTH-1:0];

endmodule
