`timescale 1ns / 1ps
//=============================================================================
// tb_uart_rx.v -- self-checking testbench for uart_rx
//
// Drives a real 8N1 frame at the bit rate the DUT expects: each bit is held
// for exactly DIVISOR clock cycles, so the receiver's own counters decide
// where it samples. Nothing here reaches inside the DUT.
//
// Test byte is 0x53 ('S') = 0101_0011, sent LSB first, so the line carries
//   start(0) 1 1 0 0 1 0 1 0 stop(1)
// -- both levels appear in both halves of the byte, so a receiver that shifted
// the wrong direction or dropped a bit could not accidentally pass.
//
// data_valid is caught by a concurrent monitor rather than polled after the
// frame: the DUT strobes at the CENTRE of the stop bit, roughly HALF_DIV
// cycles before the testbench finishes holding that bit, so a check that only
// started once send_byte returned would arrive too late to see the pulse.
//=============================================================================

module tb_uart_rx;

    localparam integer DIVISOR  = 868;
    localparam integer HALF_DIV = 434;

    localparam [7:0] TEST_BYTE = 8'h53;

    reg        clk = 0;
    reg        rst = 1;
    reg        rx  = 1;      // idle high
    wire [7:0] data_out;
    wire       data_valid;

    integer errors = 0;
    integer i;

    uart_rx #(
        .DIVISOR(DIVISOR), .HALF_DIV(HALF_DIV)
    ) dut (
        .clk(clk), .rst(rst), .rx(rx),
        .data_out(data_out), .data_valid(data_valid)
    );

    always #5 clk = ~clk;    // 100 MHz -> 10 ns period

    //-------------------------------------------------------------------------
    // Monitor: latch the strobe and what it published, and count how many
    // cycles it stayed high.
    //-------------------------------------------------------------------------
    reg        saw_valid    = 0;
    reg  [7:0] captured     = 0;
    integer    valid_cycles = 0;

    always @(posedge clk) begin
        if (!rst && data_valid) begin
            saw_valid    <= 1'b1;
            captured     <= data_out;
            valid_cycles <= valid_cycles + 1;
        end
    end

    //-------------------------------------------------------------------------
    // Stimulus
    //-------------------------------------------------------------------------

    // Hold the rx line at `level` for one full bit time. The line moves 1 ns
    // after a clock edge so the DUT's synchronizer never samples it mid-change.
    task send_bit(input level);
        integer c;
        begin
            #1 rx = level;
            for (c = 0; c < DIVISOR; c = c + 1)
                @(posedge clk);
        end
    endtask

    // One 8N1 frame: start, eight data bits LSB first, stop.
    task send_byte(input [7:0] b);
        integer k;
        begin
            send_bit(1'b0);
            for (k = 0; k < 8; k = k + 1)
                send_bit(b[k]);
            send_bit(1'b1);
        end
    endtask

    initial begin
        // Hold reset, then let the line sit idle long enough for the
        // synchronizer to settle before the first start edge.
        repeat (4) @(posedge clk);
        rst = 0;
        repeat (10) @(posedge clk);

        send_byte(TEST_BYTE);

        // saw_valid latches, so this succeeds whether the strobe already
        // happened during the stop bit or is still a few cycles out. Bounded
        // so a broken DUT fails instead of hanging the simulation.
        i = 0;
        while (!saw_valid && i < 2*DIVISOR) begin
            @(posedge clk);
            i = i + 1;
        end

        if (!saw_valid) begin
            $display("FAIL: data_valid never asserted after the frame");
            errors = errors + 1;
        end else begin
            $display("pass: data_valid asserted");

            if (captured !== TEST_BYTE) begin
                $display("FAIL: data_out=0x%02h expected=0x%02h",
                         captured, TEST_BYTE);
                errors = errors + 1;
            end else begin
                $display("pass: data_out=0x%02h", captured);
            end

            // The strobe must be exactly one cycle wide -- downstream logic
            // treats it as a pulse, so a level would double-consume the byte.
            if (valid_cycles !== 1) begin
                $display("FAIL: data_valid high for %0d cycles, expected 1",
                         valid_cycles);
                errors = errors + 1;
            end else begin
                $display("pass: data_valid is a one-cycle pulse");
            end

            // And the byte must hold on the output after the pulse drops.
            if (data_out !== TEST_BYTE) begin
                $display("FAIL: data_out did not hold: 0x%02h", data_out);
                errors = errors + 1;
            end else begin
                $display("pass: data_out holds after the pulse");
            end
        end

        $display("");
        if (errors == 0)
            $display("=== ALL TESTS PASSED ===");
        else
            $display("=== %0d FAILURE(S) ===", errors);
        $finish;
    end

endmodule
