`timescale 1ns / 1ps
//=============================================================================
// argmax.v -- streaming maximum with index
//
// Values arrive one per cycle as the controller finishes each output neuron.
// Assert `first` with the first value of a pass; it loads unconditionally.
// Subsequent values replace the running best only if strictly greater, so
// ties resolve to the LOWEST index -- matching numpy's argmax.
//
// Streaming rather than a comparator tree over all COUNT logits: it matches
// the dataflow (the controller produces one logit at a time anyway) and
// avoids registering COUNT * WIDTH bits of intermediate results.
//
// No requantization is needed ahead of this. argmax is invariant under a
// positive scale factor, so the raw int32 accumulators can be compared
// directly and the final output scale never has to be computed.
//=============================================================================

module argmax #(
    parameter integer WIDTH = 32,
    parameter integer COUNT = 24
)(
    input  wire                          clk,
    input  wire                          rst,
    input  wire                          en,     // a value is present
    input  wire                          first,  // load unconditionally
    input  wire signed [WIDTH-1:0]       value,
    input  wire [$clog2(COUNT)-1:0]      index,
    output reg  [$clog2(COUNT)-1:0]      best_idx,
    output reg  signed [WIDTH-1:0]       best_val
);

    always @(posedge clk) begin
        if (rst) begin
            best_idx <= {$clog2(COUNT){1'b0}};
            best_val <= {1'b1, {(WIDTH-1){1'b0}}};   // most negative
        end else if (en) begin
            if (first || value > best_val) begin
                best_idx <= index;
                best_val <= value;
            end
        end
    end

endmodule
