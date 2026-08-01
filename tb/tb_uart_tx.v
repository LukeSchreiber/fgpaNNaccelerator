`timescale 1ns / 1ps
//=============================================================================
// tb_uart_tx.v -- self-checking testbench for uart_tx
//
// Decodes the tx line the way a real receiver would: find the start edge,
// step half a bit to the centre, then sample every DIVISOR cycles. Nothing
// reaches inside the DUT, so this checks the wire format rather than the
// implementation -- a transmitter that shifted the wrong way or mistimed a
// bit period would fail here even though its internal state looked sane.
//
// Test byte is 0x53 ('S') = 0101_0011, so the line should carry
//   start(0) 1 1 0 0 1 0 1 0 stop(1)
// -- both levels appear in both halves of the byte, so a reversed shift
// direction cannot accidentally pass.
//=============================================================================

module tb_uart_tx;

    // The divisor nn_accel_top actually deploys: 100 MHz / 921600 baud.
    localparam integer DIVISOR  = 109;
    localparam integer HALF_DIV = DIVISOR / 2;

    localparam [7:0] TEST_BYTE = 8'h53;

    reg        clk     = 0;
    reg        rst     = 1;
    reg        send    = 0;
    reg  [7:0] data_in = 0;
    wire       tx;
    wire       busy;

    integer errors = 0;
    integer i;

    reg [7:0] decoded;

    uart_tx #(
        .DIVISOR(DIVISOR)
    ) dut (
        .clk(clk), .rst(rst), .send(send), .data_in(data_in),
        .tx(tx), .busy(busy)
    );

    always #5 clk = ~clk;    // 100 MHz -> 10 ns period

    // Count the cycles busy stays high. A correct 8N1 frame is exactly ten
    // bit times, so this catches a busy that drops early or lingers.
    integer busy_cycles = 0;

    always @(posedge clk)
        if (!rst && busy)
            busy_cycles <= busy_cycles + 1;

    //-------------------------------------------------------------------------
    // Stimulus and decode
    //-------------------------------------------------------------------------

    // Present data_in and pulse send for exactly one clock edge. Signals move
    // 1 ns off the edge so the DUT never samples them mid-change.
    task pulse_send(input [7:0] b);
        begin
            #1;
            data_in = b;
            send    = 1'b1;
            @(posedge clk);
            #1;
            send    = 1'b0;
        end
    endtask

    // Advance n clock edges and settle 1 ns past the last one, so anything
    // sampled afterwards sees the DUT's post-edge value.
    task wait_cycles(input integer n);
        integer c;
        begin
            for (c = 0; c < n; c = c + 1)
                @(posedge clk);
            #1;
        end
    endtask

    // Recover a byte from the tx line by sampling at mid-bit.
    task recv_byte(output [7:0] b);
        integer k;
        integer w;
        begin
            b = 8'hxx;

            // Hunt for the start bit. tx may already be low if the caller got
            // here right after the frame began. Bounded so a transmitter that
            // never pulls the line low fails instead of hanging the sim.
            w = 0;
            while (tx !== 1'b0 && w < 2*DIVISOR) begin
                @(posedge clk);
                w = w + 1;
            end

            if (tx !== 1'b0) begin
                $display("FAIL: no start bit -- tx never went low");
                errors = errors + 1;
                disable recv_byte;
            end

            // Step to the middle of the start bit and confirm it is still low.
            wait_cycles(HALF_DIV);
            if (tx !== 1'b0) begin
                $display("FAIL: start bit not low at mid-bit");
                errors = errors + 1;
            end

            // One sample per bit time from here on, LSB first.
            for (k = 0; k < 8; k = k + 1) begin
                wait_cycles(DIVISOR);
                b[k] = tx;
            end

            // busy must still be asserted -- the stop bit has not gone out yet.
            if (busy !== 1'b1) begin
                $display("FAIL: busy dropped before the stop bit");
                errors = errors + 1;
            end

            wait_cycles(DIVISOR);
            if (tx !== 1'b1) begin
                $display("FAIL: stop bit not high at mid-bit");
                errors = errors + 1;
            end else begin
                $display("pass: framing (start low, stop high)");
            end
        end
    endtask

    initial begin
        repeat (4) @(posedge clk);
        rst = 0;
        repeat (10) @(posedge clk);

        if (tx !== 1'b1) begin
            $display("FAIL: tx not idle high before sending");
            errors = errors + 1;
        end else begin
            $display("pass: tx idle high out of reset");
        end

        pulse_send(TEST_BYTE);

        if (busy !== 1'b1) begin
            $display("FAIL: busy did not assert after send");
            errors = errors + 1;
        end else begin
            $display("pass: busy asserted after send");
        end

        recv_byte(decoded);

        if (decoded !== TEST_BYTE) begin
            $display("FAIL: decoded=0x%02h expected=0x%02h",
                     decoded, TEST_BYTE);
            errors = errors + 1;
        end else begin
            $display("pass: decoded=0x%02h", decoded);
        end

        // Decoding ended at the CENTRE of the stop bit, so busy still has
        // about half a bit time to run. Wait it out, bounded so a stuck busy
        // fails the test instead of hanging the simulation.
        i = 0;
        while (busy !== 1'b0 && i < 2*DIVISOR) begin
            @(posedge clk);
            i = i + 1;
        end
        #1;

        if (busy !== 1'b0) begin
            $display("FAIL: busy never deasserted after the stop bit");
            errors = errors + 1;
        end else begin
            $display("pass: busy deasserted after the stop bit");
        end

        if (busy_cycles !== 10*DIVISOR) begin
            $display("FAIL: busy high for %0d cycles, expected %0d",
                     busy_cycles, 10*DIVISOR);
            errors = errors + 1;
        end else begin
            $display("pass: busy high for exactly %0d cycles (10 bit times)",
                     busy_cycles);
        end

        // And the line must return to idle rather than sitting low.
        repeat (4) @(posedge clk);
        #1;
        if (tx !== 1'b1) begin
            $display("FAIL: tx did not return to idle high");
            errors = errors + 1;
        end else begin
            $display("pass: tx back to idle high");
        end

        $display("");
        if (errors == 0)
            $display("=== ALL TESTS PASSED ===");
        else
            $display("=== %0d FAILURE(S) ===", errors);
        $finish;
    end

endmodule
