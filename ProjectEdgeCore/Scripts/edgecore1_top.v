// ============================================================
//  EdgeCore-1 — Top-Level SoC Integration
//  Infinar Scientia | April 2026
//
//  Integrates:
//    - edgecore1_riscv_ctrl (inference controller FSM /
//      drop-in for PicoRV32 binary)
//    - mac_array (256×256 INT8 systolic array)
//    - sparsity_engine (bitmap-mask tile suppression)
//    - lpddr5_pwr_ctrl (JEDEC deep power-down sequencing)
//    - dma_double_buf (double-buffer weight prefetch)
//    - icg_cell (clock gating)
// ============================================================
`timescale 1ns/1ps

// ── Clock Gating Cell ─────────────────────────────────────────
module icg_cell (
    input  wire clk_in, enable, test_mode,
    output wire clk_out
);
    reg latch_en;
    always @(*) if (!clk_in) latch_en <= enable | test_mode;
    assign clk_out = clk_in & latch_en;
endmodule

// ── LPDDR5 PHY Power Domain Controller ───────────────────────
module lpddr5_pwr_ctrl (
    input  wire       clk, rst_n,
    input  wire [1:0] pwr_req,
    output reg        phy_active, pwr_ready,
    output reg  [1:0] pwr_state,
    output reg [31:0] cycles_deep_down, cycles_idle, cycles_burst
);
    localparam PWR_DEEP=2'b00, PWR_IDLE=2'b01, PWR_BURST=2'b10;
    localparam WAKEUP_CYCLES = 250;
    reg [17:0] wakeup_ctr;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pwr_state<=PWR_DEEP; phy_active<=0; pwr_ready<=0;
            wakeup_ctr<=0; cycles_deep_down<=0; cycles_idle<=0; cycles_burst<=0;
        end else case (pwr_state)
            PWR_DEEP: begin
                phy_active<=0; pwr_ready<=0;
                cycles_deep_down<=cycles_deep_down+1;
                if (pwr_req!=PWR_DEEP) begin wakeup_ctr<=WAKEUP_CYCLES; pwr_state<=PWR_IDLE; end
            end
            PWR_IDLE: begin
                phy_active<=1; cycles_idle<=cycles_idle+1;
                if (wakeup_ctr>0) begin wakeup_ctr<=wakeup_ctr-1; pwr_ready<=0; end
                else begin
                    pwr_ready<=1;
                    if (pwr_req==PWR_BURST) pwr_state<=PWR_BURST;
                    else if (pwr_req==PWR_DEEP) pwr_state<=PWR_DEEP;
                end
            end
            PWR_BURST: begin
                phy_active<=1; pwr_ready<=1; cycles_burst<=cycles_burst+1;
                if (pwr_req!=PWR_BURST) pwr_state<=PWR_IDLE;
            end
            default: pwr_state<=PWR_DEEP;
        endcase
    end
endmodule

// ── DMA Double-Buffer Controller ─────────────────────────────
module dma_double_buf (
    input  wire        clk, rst_n, layer_done, phy_ready,
    input  wire [31:0] next_layer_addr,
    input  wire [19:0] next_layer_bytes,
    output reg         dma_req, bank_swap,
    output reg  [1:0]  phy_pwr_req,
    output reg [31:0]  total_fetches
);
    localparam DMA_IDLE=2'b00, DMA_FETCH=2'b01;
    reg [1:0]  dma_state;
    reg [19:0] bytes_rem;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dma_state<=DMA_IDLE; dma_req<=0; bank_swap<=0;
            bytes_rem<=0; total_fetches<=0; phy_pwr_req<=2'b00;
        end else begin
            bank_swap<=0;
            case (dma_state)
                DMA_IDLE: begin
                    phy_pwr_req<=2'b00; dma_req<=0;
                    if (layer_done) begin bytes_rem<=next_layer_bytes; phy_pwr_req<=2'b10; dma_state<=DMA_FETCH; end
                end
                DMA_FETCH: begin
                    phy_pwr_req<=2'b10;
                    if (phy_ready) begin
                        dma_req<=1;
                        if (bytes_rem>0) begin
                            bytes_rem<=bytes_rem>8 ? bytes_rem-8 : 0;
                            total_fetches<=total_fetches+1;
                        end else begin dma_req<=0; bank_swap<=1; dma_state<=DMA_IDLE; end
                    end
                end
                default: dma_state<=DMA_IDLE;
            endcase
        end
    end
endmodule

// ── TOP LEVEL ─────────────────────────────────────────────────
module edgecore1_top (
    input  wire        sys_clk,
    input  wire        sys_rst_n,
    output wire        lpddr5_clk,
    output wire        lpddr5_cs_n,
    output wire [63:0] lpddr5_dq_out,
    input  wire [63:0] lpddr5_dq_in,
    input  wire        spi_clk, spi_cs_n, spi_mosi,
    output wire        spi_miso,
    output wire        uart_tx,
    input  wire        uart_rx,
    output wire [7:0]  result_byte,
    output wire        result_valid,
    input  wire        test_mode,
    output wire [31:0] dbg_mac_cycles,
    output wire [31:0] dbg_gated_cycles,
    output wire [31:0] dbg_suppressed_tiles,
    output wire [1:0]  dbg_phy_state,
    output wire [2:0]  dbg_ctrl_state,
    output wire [31:0] dbg_inference_count
);

    // ── Internal wires ────────────────────────────────────────
    wire        clk_mac;
    wire        mac_clk_en, acc_clear, tile_valid;
    wire [511:0] tile_a, tile_w;
    wire        layer_done, acc_valid_mac;
    wire        mask_wr_en, eval_req, eval_done, tile_active;
    wire [17:0] mask_wr_addr;
    wire [7:0]  mask_wr_data;
    wire [13:0] tile_index;
    wire [1:0]  phy_pwr_req, phy_state;
    wire        phy_ready, dma_req, bank_swap;
    wire [31:0] next_layer_addr_w;
    wire [19:0] next_layer_bytes_w;
    wire [8191:0] acc_out;
    wire [31:0] mac_cycle_count, gated_cycle_count, tiles_suppressed;
    wire [7:0]  result_class;
    wire        result_valid_w;
    wire [31:0] inference_count;
    wire [2:0]  ctrl_state;

    // ── Clock gating ──────────────────────────────────────────
    icg_cell mac_clk_gate (
        .clk_in(sys_clk), .enable(mac_clk_en),
        .test_mode(test_mode), .clk_out(clk_mac)
    );

    // ── RISC-V Controller (replace with PicoRV32 for production) ──
    edgecore1_riscv_ctrl #(.NUM_TILES(16)) ctrl (
        .clk(sys_clk), .rst_n(sys_rst_n),
        .mac_clk_en(mac_clk_en), .acc_clear(acc_clear),
        .tile_valid(tile_valid), .tile_a(tile_a), .tile_w(tile_w),
        .layer_done(layer_done), .acc_valid(acc_valid_mac),
        .mask_wr_en(mask_wr_en), .mask_wr_addr(mask_wr_addr),
        .mask_wr_data(mask_wr_data), .eval_req(eval_req),
        .tile_index(tile_index), .eval_done(eval_done),
        .tile_active(tile_active), .next_layer_addr(next_layer_addr_w),
        .next_layer_bytes(next_layer_bytes_w), .phy_ready(phy_ready),
        .acc_out(acc_out), .result_class(result_class),
        .result_valid(result_valid_w), .inference_count(inference_count),
        .ctrl_state_dbg(ctrl_state)
    );

    // ── PHY Power Controller ──────────────────────────────────
    lpddr5_pwr_ctrl phy_pwr (
        .clk(sys_clk), .rst_n(sys_rst_n),
        .pwr_req(phy_pwr_req), .phy_active(),
        .pwr_ready(phy_ready), .pwr_state(phy_state),
        .cycles_deep_down(), .cycles_idle(), .cycles_burst()
    );

    // ── DMA ───────────────────────────────────────────────────
    dma_double_buf dma (
        .clk(sys_clk), .rst_n(sys_rst_n),
        .layer_done(layer_done), .phy_ready(phy_ready),
        .next_layer_addr(next_layer_addr_w),
        .next_layer_bytes(next_layer_bytes_w),
        .dma_req(dma_req), .bank_swap(bank_swap),
        .phy_pwr_req(phy_pwr_req), .total_fetches()
    );

    // ── Sparsity Engine ───────────────────────────────────────
    sparsity_engine sparsity (
        .clk(clk_mac), .rst_n(sys_rst_n),
        .mask_wr_en(mask_wr_en), .mask_wr_addr(mask_wr_addr),
        .mask_wr_data(mask_wr_data), .eval_req(eval_req),
        .tile_index(tile_index), .eval_done(eval_done),
        .tile_active(tile_active), .tiles_evaluated(),
        .tiles_suppressed(tiles_suppressed), .cycles_saved()
    );

    // ── MAC Array ─────────────────────────────────────────────
    mac_array #(.MAC_ROWS(256),.MAC_COLS(256),.TILE_SIZE(8),.DATA_WIDTH(8),.ACC_WIDTH(32)) mac (
        .clk(clk_mac), .rst_n(sys_rst_n),
        .tile_valid(tile_valid), .tile_a(tile_a), .tile_w(tile_w),
        .acc_out(acc_out), .acc_valid(acc_valid_mac),
        .acc_clear(acc_clear), .layer_done(layer_done),
        .mac_cycle_count(mac_cycle_count),
        .gated_cycle_count(gated_cycle_count)
    );

    // ── Outputs ───────────────────────────────────────────────
    assign result_byte          = result_class;
    assign result_valid         = result_valid_w;
    assign dbg_mac_cycles       = mac_cycle_count;
    assign dbg_gated_cycles     = gated_cycle_count;
    assign dbg_suppressed_tiles = tiles_suppressed;
    assign dbg_phy_state        = phy_state;
    assign dbg_ctrl_state       = ctrl_state;
    assign dbg_inference_count  = inference_count;
    assign lpddr5_clk           = sys_clk;
    assign lpddr5_cs_n          = !dma_req;
    assign lpddr5_dq_out        = 64'b0;
    assign spi_miso             = 1'b0;
    assign uart_tx              = 1'b1;

endmodule
