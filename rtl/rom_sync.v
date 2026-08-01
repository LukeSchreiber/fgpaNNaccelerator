`timescale 1ns / 1ps
//=============================================================================
// rom_sync.v -- synchronous-read ROM, initialized from a hex file
//
// This is the canonical Xilinx block-RAM inference template: an array read
// into a REGISTER on a clock edge. The output register is what makes it a
// BRAM. Writing
//
//     assign dout = mem[addr];     // combinational read
//
// instead produces distributed RAM built from LUTs, which for 800 Kbit would
// be impossible on this part -- there are only 9,600 LUTs available as memory.
//
// LATENCY: dout is valid one cycle after addr is presented. The layer
// controller must issue addresses one cycle ahead of when the MAC array
// needs the data.
//
// WIDTH is deliberately N*8 rather than 8: the MAC array consumes N weights
// per cycle, and a single BRAM port is at most 36 bits wide, so the weights
// are packed across the word and Vivado splits the array over as many
// physical BRAMs as the width and depth require.
//=============================================================================

module rom_sync #(
    parameter integer WIDTH     = 128,
    parameter integer DEPTH     = 6272,
    parameter         INIT_FILE = ""
)(
    input  wire                          clk,
    input  wire                          en,
    input  wire [$clog2(DEPTH)-1:0]      addr,
    output reg  [WIDTH-1:0]              dout
);

    (* ram_style = "block" *)
    reg [WIDTH-1:0] mem [0:DEPTH-1];

    initial begin
        if (INIT_FILE != "")
            $readmemh(INIT_FILE, mem);
    end

    always @(posedge clk) begin
        if (en)
            dout <= mem[addr];
    end

endmodule
