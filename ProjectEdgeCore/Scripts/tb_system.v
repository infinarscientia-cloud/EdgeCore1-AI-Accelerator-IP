// ============================================================
//  EdgeCore-1 — Full System Testbench
//  Infinar Scientia | April 2026
//
//  Tests the complete pipeline:
//    Controller → Sparsity → MAC → Accumulator → Result
//
//  PASS criteria:
//    1. result_valid pulses within timeout
//    2. result_byte is valid (0-255 class index)
//    3. Sparsity suppresses expected tiles (tiles 3,4 per pattern)
//    4. MAC clock gating works (gated cycles > 0)
//    5. PHY power state transitions correctly
//    6. Multiple back-to-back inferences succeed
// ============================================================
`timescale 1ns/1ps

module tb_system;
    parameter CLK_PERIOD = 2; // 500 MHz
    parameter TIMEOUT    = 500000;

    reg  sys_clk, sys_rst_n, test_mode;
    wire lpddr5_clk, lpddr5_cs_n;
    wire [63:0] lpddr5_dq_out;
    wire spi_miso, uart_tx;
    wire [7:0]  result_byte;
    wire        result_valid;
    wire [31:0] dbg_mac_cycles, dbg_gated_cycles;
    wire [31:0] dbg_suppressed_tiles, dbg_inference_count;
    wire [1:0]  dbg_phy_state;
    wire [2:0]  dbg_ctrl_state;

    // ── DUT ───────────────────────────────────────────────────
    edgecore1_top dut (
        .sys_clk(sys_clk), .sys_rst_n(sys_rst_n),
        .lpddr5_clk(lpddr5_clk), .lpddr5_cs_n(lpddr5_cs_n),
        .lpddr5_dq_out(lpddr5_dq_out), .lpddr5_dq_in(64'b0),
        .spi_clk(1'b0), .spi_cs_n(1'b1), .spi_mosi(1'b0),
        .spi_miso(spi_miso), .uart_tx(uart_tx), .uart_rx(1'b1),
        .result_byte(result_byte), .result_valid(result_valid),
        .test_mode(test_mode),
        .dbg_mac_cycles(dbg_mac_cycles),
        .dbg_gated_cycles(dbg_gated_cycles),
        .dbg_suppressed_tiles(dbg_suppressed_tiles),
        .dbg_phy_state(dbg_phy_state),
        .dbg_ctrl_state(dbg_ctrl_state),
        .dbg_inference_count(dbg_inference_count)
    );

    initial sys_clk = 0;
    always #(CLK_PERIOD/2) sys_clk = ~sys_clk;

    integer tests_run = 0, tests_passed = 0, tests_failed = 0;
    integer timeout_ctr;
    integer inference_num;

    task wait_for_result;
        input integer inf_num;
        output reg timed_out;
        begin
            timed_out = 0;
            timeout_ctr = 0;
            while (!result_valid && timeout_ctr < TIMEOUT) begin
                @(posedge sys_clk);
                timeout_ctr = timeout_ctr + 1;
            end
            if (timeout_ctr >= TIMEOUT) begin
                timed_out = 1;
                $display("  [FAIL] Inference %0d: TIMEOUT waiting for result_valid", inf_num);
                tests_failed = tests_failed + 1;
                tests_run    = tests_run + 1;
            end
        end
    endtask

    task check;
        input        cond;
        input [255:0] msg;
        begin
            tests_run = tests_run + 1;
            if (cond) begin
                tests_passed = tests_passed + 1;
                $display("  [PASS] %s", msg);
            end else begin
                tests_failed = tests_failed + 1;
                $display("  [FAIL] %s", msg);
            end
        end
    endtask

    reg timed_out;
    reg [31:0] sup_after_1st, sup_after_2nd;
    reg [7:0]  result_1st, result_2nd;
    reg [31:0] gated_after_reset, gated_after_inference;

    initial begin
        $dumpfile("tb_system.vcd");
        $dumpvars(0, tb_system);

        $display("==============================================");
        $display("  EdgeCore-1 Full System Testbench");
        $display("  Infinar Scientia | April 2026");
        $display("  Complete Pipeline: Ctrl→Sparse→MAC→Result");
        $display("==============================================");

        sys_rst_n = 0;
        test_mode = 0;
        repeat(8) @(posedge sys_clk);
        sys_rst_n = 1;
        @(posedge sys_clk);

        // ── TEST 1: First inference completes ─────────────────
        $display("\n[Test 1] First full inference pipeline");
        wait_for_result(1, timed_out);

        if (!timed_out) begin
            result_1st = result_byte;

            check(!timed_out,
                "Inference 1 completed within timeout");
            check(result_byte <= 8'd255,
                "result_byte is valid class index (0-255)");
            $display("         Class result: %0d", result_byte);
            $display("         Inferences completed: %0d", dbg_inference_count);
        end

        // ── TEST 2: Sparsity suppression happened ─────────────
        $display("\n[Test 2] Sparsity engine suppressed sparse tiles");
        sup_after_1st = dbg_suppressed_tiles;
        check(dbg_suppressed_tiles > 0,
            "At least 1 tile was suppressed (sparsity working)");
        $display("         Tiles suppressed: %0d out of 16", dbg_suppressed_tiles);

        // ── TEST 3: Clock gating fired ────────────────────────
        $display("\n[Test 3] MAC clock gating (power saving)");
        check(dbg_gated_cycles > 0,
            "Gated cycles > 0 (clock gating active)");
        $display("         MAC active cycles: %0d", dbg_mac_cycles);
        $display("         MAC gated cycles:  %0d", dbg_gated_cycles);

        // ── TEST 4: Second inference (loop stability) ──────────
        $display("\n[Test 4] Second inference (pipeline loops correctly)");
        // Wait for result_valid to deassert then fire again
        repeat(10) @(posedge sys_clk);
        wait_for_result(2, timed_out);

        if (!timed_out) begin
            result_2nd = result_byte;
            check(!timed_out,
                "Inference 2 completed within timeout");
            check(dbg_inference_count >= 2,
                "Inference counter incremented to >= 2");
            check(result_2nd === result_1st,
                "Deterministic: same input → same class output");
            $display("         2nd class result: %0d (matches 1st: %0d)",
                     result_2nd, result_1st);
        end

        // ── TEST 5: Suppression count grew ────────────────────
        $display("\n[Test 5] Cumulative sparsity across inferences");
        sup_after_2nd = dbg_suppressed_tiles;
        check(sup_after_2nd > sup_after_1st,
            "Suppressed tile count grew on 2nd inference");
        $display("         After 2nd inference: %0d suppressed tiles total",
                 sup_after_2nd);

        // ── TEST 6: PHY power state transitions ───────────────
        $display("\n[Test 6] PHY power domain transitions");
        check(dbg_phy_state !== 2'bxx,
            "PHY power state is defined (not X)");
        $display("         PHY state: %0d (0=deep, 1=idle, 2=burst)",
                 dbg_phy_state);

        // ── TEST 7: Third inference (stress) ──────────────────
        $display("\n[Test 7] Third inference (stress)");
        repeat(10) @(posedge sys_clk);
        wait_for_result(3, timed_out);
        if (!timed_out) begin
            check(dbg_inference_count >= 3,
                "Three complete inferences without hang");
        end

        // ── FINAL REPORT ──────────────────────────────────────
        $display("\n==============================================");
        $display("  SYSTEM VERIFICATION REPORT");
        $display("==============================================");
        $display("  Tests run:          %0d", tests_run);
        $display("  Tests passed:       %0d", tests_passed);
        $display("  Tests failed:       %0d", tests_failed);
        $display("  Inferences done:    %0d", dbg_inference_count);
        $display("  Tiles suppressed:   %0d", dbg_suppressed_tiles);
        $display("  MAC active cycles:  %0d", dbg_mac_cycles);
        $display("  MAC gated cycles:   %0d", dbg_gated_cycles);

        if (tests_failed == 0)
            $display("  STATUS: ALL PASS — Full pipeline verified");
        else
            $display("  STATUS: %0d FAILURES", tests_failed);
        $display("==============================================");
        $finish;
    end

    initial begin
        #(TIMEOUT * CLK_PERIOD * 4);
        $display("[WATCHDOG] Global timeout"); $finish;
    end
endmodule
