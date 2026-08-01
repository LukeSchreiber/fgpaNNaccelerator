`timescale 1ns / 1ps
//=============================================================================
// uart_loopback_top.v -- Basys 3 top level: UART echo with byte display
//
//   BTNC          reset
//   RsRx          serial in  from the host (115200 8N1)
//   RsTx          serial out to the host -- echoes whatever arrived
//   LD[7:0]       last received byte  (LD[15:8] are left unconstrained)
//
// This is the bring-up harness for the UART pair: type a character in a
// terminal, see it come back and see its bit pattern on the LEDs. It proves
// the divisor, the sampling point and the pin constraints are right before
// anything is wired to the inference core.
//
// THROUGHPUT LIMIT
// ----------------
// uart_rx strobes data_valid at the CENTRE of the stop bit, half a bit time
// before the frame is actually over, and uart_tx then needs ten full bit
// times to echo it. A host streaming frames back-to-back with no inter-byte
// gap therefore produces the next data_valid roughly half a bit time before
// the echo finishes, and `~busy` drops that byte on the floor.
//
// That is deliberate: the alternative is a FIFO, and this module exists to
// test the UART, not to buffer it. Interactive typing and any host that
// leaves a byte of idle between frames echo everything. If a future user of
// this pair needs lossless streaming, put a FIFO between rx and tx rather
// than widening the handshake here.
//=============================================================================

module uart_loopback_top (
    input  wire        clk,     // W5, 100 MHz
    input  wire        btnC,    // reset, active high
    input  wire        RsRx,    // B18
    output wire        RsTx,    // A18
    output wire [7:0]  led      // U16 E19 U19 V19 W18 U15 U14 V14
);

    localparam integer DIVISOR = 868;   // 100 MHz / 115200 baud

    // btnC is a raw asynchronous input feeding the reset of two state
    // machines, so it gets two flops before use. No debounce: contact bounce
    // on a reset line just means a few extra resets, which is harmless --
    // unlike a start pulse, where each bounce would launch new work.
    reg [1:0] btnC_sync;
    always @(posedge clk)
        btnC_sync <= {btnC_sync[0], btnC};

    wire rst = btnC_sync[1];

    wire [7:0] rx_data;
    wire       rx_valid;
    wire       tx_busy;

    // RsRx goes straight in: uart_rx has its own two-FF synchronizer on the
    // pin, so adding another here would only cost latency.
    uart_rx #(
        .DIVISOR(DIVISOR), .HALF_DIV(DIVISOR/2)
    ) u_rx (
        .clk        (clk),
        .rst        (rst),
        .rx         (RsRx),
        .data_out   (rx_data),
        .data_valid (rx_valid)
    );

    // rx_valid is already a one-cycle pulse and tx_busy is low only in IDLE,
    // so the AND is a clean single-cycle send that cannot fire mid-frame.
    // uart_tx latches data_in on that pulse, and rx_data holds until the next
    // frame completes, so no separate holding register is needed.
    uart_tx #(
        .DIVISOR(DIVISOR)
    ) u_tx (
        .clk     (clk),
        .rst     (rst),
        .send    (rx_valid & ~tx_busy),
        .data_in (rx_data),
        .tx      (RsTx),
        .busy    (tx_busy)
    );

    // uart_rx holds data_out until the next byte lands, so this is the last
    // received byte without any extra state.
    assign led = rx_data;

endmodule
