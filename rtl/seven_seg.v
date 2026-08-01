`timescale 1ns / 1ps
//=============================================================================
// seven_seg.v -- Basys 3 four-digit multiplexed display driver
//
// Shows:  [letter] [blank] [tens] [ones]
//
// The predicted ASL letter on the leftmost digit and the class index in
// decimal on the two rightmost. The index is always unambiguous; the letter
// is a convenience, because FIVE OF THE 24 CLASSES HAVE NO HONEST GLYPH.
// K, M, V, W and X cannot be drawn with seven segments -- they need
// diagonals. Those show a dash rather than a misleading approximation.
//
// BASYS 3 ELECTRICAL DETAILS (reference manual section 8.1)
// ---------------------------------------------------------
// Common-anode display. Segments light when the CATHODE is driven low.
// The anode enables are inverted by the drive transistors, so BOTH the
// anode selects and the segment outputs are ACTIVE LOW. Driving them
// active-high gives an inverted image -- every segment lit except the ones
// you wanted.
//
// Only one digit is lit at a time; the scan runs fast enough that the eye
// integrates it. The manual specifies a 1-16 ms full-refresh period; below
// about 45 Hz the flicker becomes visible. Here the digit select advances
// every 2^16 clocks = 655 us, so a full refresh takes 2.6 ms (381 Hz).
//=============================================================================

module seven_seg #(
    parameter integer REFRESH_BITS = 16   // 100 MHz / 2^16 = 1526 Hz per digit
)(
    input  wire        clk,
    input  wire        rst,
    input  wire [4:0]  class_idx,   // 0..23
    input  wire        valid,       // blank the display until a result exists
    output reg  [3:0]  an,          // active low digit select
    output reg  [6:0]  seg,         // active low, {a,b,c,d,e,f,g}
    output wire        dp
);

    assign dp = 1'b1;   // decimal point off (active low)

    reg [REFRESH_BITS+1:0] refresh;
    always @(posedge clk) begin
        if (rst) refresh <= 0;
        else     refresh <= refresh + 1'b1;
    end

    wire [1:0] digit = refresh[REFRESH_BITS+1:REFRESH_BITS];

    // ---- decimal split of the class index -----------------------------
    // 0..23 only, so a compare-and-subtract beats a divider.
    wire [4:0] tens = (class_idx >= 5'd20) ? 5'd2 :
                      (class_idx >= 5'd10) ? 5'd1 : 5'd0;
    wire [4:0] ones = class_idx - (tens * 5'd10);

    // ---- glyphs, {a,b,c,d,e,f,g}, 1 = segment on ----------------------
    function [6:0] digit_glyph(input [4:0] d);
        case (d)
            5'd0: digit_glyph = 7'b1111110;
            5'd1: digit_glyph = 7'b0110000;
            5'd2: digit_glyph = 7'b1101101;
            5'd3: digit_glyph = 7'b1111001;
            5'd4: digit_glyph = 7'b0110011;
            5'd5: digit_glyph = 7'b1011011;
            5'd6: digit_glyph = 7'b1011111;
            5'd7: digit_glyph = 7'b1110000;
            5'd8: digit_glyph = 7'b1111111;
            5'd9: digit_glyph = 7'b1111011;
            default: digit_glyph = 7'b0000001;   // dash
        endcase
    endfunction

    // Class order: A B C D E F G H I K L M N O P Q R S T U V W X Y
    // (J and Z are excluded -- they require motion.)
    function [6:0] letter_glyph(input [4:0] c);
        case (c)
            5'd0:  letter_glyph = 7'b1110111;  // A
            5'd1:  letter_glyph = 7'b0011111;  // b
            5'd2:  letter_glyph = 7'b1001110;  // C
            5'd3:  letter_glyph = 7'b0111101;  // d
            5'd4:  letter_glyph = 7'b1001111;  // E
            5'd5:  letter_glyph = 7'b1000111;  // F
            5'd6:  letter_glyph = 7'b1011110;  // G
            5'd7:  letter_glyph = 7'b0110111;  // H
            5'd8:  letter_glyph = 7'b0110000;  // I
            5'd9:  letter_glyph = 7'b0000001;  // K -- no glyph, dash
            5'd10: letter_glyph = 7'b0001110;  // L
            5'd11: letter_glyph = 7'b0000001;  // M -- no glyph, dash
            5'd12: letter_glyph = 7'b0010101;  // n
            5'd13: letter_glyph = 7'b0011101;  // o
            5'd14: letter_glyph = 7'b1100111;  // P
            5'd15: letter_glyph = 7'b1110011;  // q
            5'd16: letter_glyph = 7'b0000101;  // r
            5'd17: letter_glyph = 7'b1011011;  // S
            5'd18: letter_glyph = 7'b0001111;  // t
            5'd19: letter_glyph = 7'b0111110;  // U
            5'd20: letter_glyph = 7'b0000001;  // V -- no glyph, dash
            5'd21: letter_glyph = 7'b0000001;  // W -- no glyph, dash
            5'd22: letter_glyph = 7'b0000001;  // X -- no glyph, dash
            5'd23: letter_glyph = 7'b0111011;  // y
            default: letter_glyph = 7'b0000000;
        endcase
    endfunction

    reg [6:0] glyph;
    always @(*) begin
        case (digit)
            2'd0: glyph = digit_glyph(ones);
            2'd1: glyph = digit_glyph(tens);
            2'd2: glyph = 7'b0000000;              // blank separator
            2'd3: glyph = letter_glyph(class_idx);
            default: glyph = 7'b0000000;
        endcase
        if (!valid) glyph = 7'b0000001;            // dashes before first result
    end

    // Both anode and segment outputs are active low on this board.
    always @(*) begin
        case (digit)
            2'd0: an = 4'b1110;
            2'd1: an = 4'b1101;
            2'd2: an = 4'b1011;
            2'd3: an = 4'b0111;
            default: an = 4'b1111;
        endcase
        seg = ~glyph;
    end

endmodule
