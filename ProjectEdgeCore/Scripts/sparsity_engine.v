// ============================================================
//  EdgeCore-1 — Bitmap Sparsity Engine
//  Infinar Scientia | April 2026
//
//  Module: sparsity_engine
//  Description:
//    64-bit bitmap comparator and popcount unit.
//    Evaluates one 8x8 weight tile (64 elements) per clock cycle.
//    Suppresses dispatch to MAC array when active weight count
//    falls below SPARSITY_THRESH (default: 13/64 = 20%).
//
//    Estimated area: ~0.27 mm² on 28nm
//      - 32KB mask SRAM: ~0.21 mm²
//      - Comparator + popcount tree: ~0.04 mm²
//      - Control logic: ~0.02 mm²
//
//  Verified against: edgecore1_golden.cpp SparsityEngine::evaluate_tile()
// ============================================================

`timescale 1ns/1ps

module sparsity_engine #(
    parameter TILE_BITS      = 64,   // 8x8 tile = 64 elements = 64 bits
    parameter MASK_SRAM_BITS = 262144, // 32KB = 262144 bits
    parameter SPARSITY_THRESH = 13   // Suppress if popcount < 13 (20% of 64)
) (
    input  wire        clk,
    input  wire        rst_n,

    // Mask SRAM write port (from RISC-V DMA during weight load)
    input  wire        mask_wr_en,
    input  wire [17:0] mask_wr_addr,  // Byte address in 32KB SRAM
    input  wire [7:0]  mask_wr_data,

    // Tile evaluation interface
    input  wire        eval_req,      // Request tile evaluation
    input  wire [13:0] tile_index,    // Which tile to evaluate (byte addr = tile*8)
    output reg         eval_done,     // Result ready (1 cycle latency)
    output reg         tile_active,   // 1 = compute tile, 0 = suppress

    // Statistics (for simulation and power reporting)
    output reg [31:0]  tiles_evaluated,
    output reg [31:0]  tiles_suppressed,
    output reg [31:0]  cycles_saved
);

    // ── 32KB Mask SRAM ──────────────────────────────────────────
    // One bit per weight element. Stored as bytes.
    // 32768 bytes × 8 bits = 262144 bits = 4096 tiles of 64 bits each.
    reg [7:0] mask_sram [0:32767];

    // ── 64-bit bitmap read (8 consecutive bytes) ────────────────
    reg [63:0] tile_bitmap;
    reg [13:0] tile_index_r;
    reg        eval_req_r;

    // ── 6-bit popcount tree ─────────────────────────────────────
    // Counts active (set) bits in a 64-bit word.
    // Implemented as a balanced binary tree for minimum depth (6 levels).
    wire [3:0] pop8  [0:7];   // Eight 8-bit groups → 4-bit popcount each
    wire [4:0] pop16 [0:3];   // Four pairs → 5-bit sum
    wire [5:0] pop32 [0:1];   // Two pairs → 6-bit sum
    wire [6:0] pop64;         // Final 7-bit popcount (max = 64)

    // Level 1: 8-bit popcount (LUT-based on 28nm standard cells)
    generate
        genvar gi;
        for (gi = 0; gi < 8; gi = gi + 1) begin : pop8_gen
            wire [7:0] byte_val = tile_bitmap[gi*8 +: 8];
            // Sum of bits using Wallace tree (synthesis will optimize)
            assign pop8[gi] = byte_val[0] + byte_val[1] + byte_val[2] + byte_val[3] +
                              byte_val[4] + byte_val[5] + byte_val[6] + byte_val[7];
        end
    endgenerate

    // Level 2: pair sums
    assign pop16[0] = pop8[0] + pop8[1];
    assign pop16[1] = pop8[2] + pop8[3];
    assign pop16[2] = pop8[4] + pop8[5];
    assign pop16[3] = pop8[6] + pop8[7];

    // Level 3: quad sums
    assign pop32[0] = pop16[0] + pop16[1];
    assign pop32[1] = pop16[2] + pop16[3];

    // Level 4: final sum
    assign pop64 = pop32[0] + pop32[1];

    // ── Evaluation pipeline (2-cycle latency) ──────────────────
    // Cycle 0: Read 8 bytes from mask SRAM
    // Cycle 1: Popcount → compare → output result

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            eval_done        <= 0;
            tile_active      <= 0;
            tiles_evaluated  <= 0;
            tiles_suppressed <= 0;
            cycles_saved     <= 0;
            tile_bitmap      <= 0;
            tile_index_r     <= 0;
            eval_req_r       <= 0;

            for (i = 0; i < 32768; i = i + 1)
                mask_sram[i] <= 8'hFF; // Default: all active
        end else begin
            // ── SRAM write (RISC-V DMA fills mask during weight load) ──
            if (mask_wr_en)
                mask_sram[mask_wr_addr] <= mask_wr_data;

            eval_done <= 0;
            eval_req_r <= eval_req;
            tile_index_r <= tile_index;

            // ── Cycle 0: SRAM read ─────────────────────────────
            if (eval_req) begin
                // Read 8 consecutive bytes starting at tile_index * 8
                // (Each tile occupies 64 bits = 8 bytes in the mask SRAM)
                tile_bitmap[7:0]   <= mask_sram[{tile_index, 3'b000}];
                tile_bitmap[15:8]  <= mask_sram[{tile_index, 3'b001}];
                tile_bitmap[23:16] <= mask_sram[{tile_index, 3'b010}];
                tile_bitmap[31:24] <= mask_sram[{tile_index, 3'b011}];
                tile_bitmap[39:32] <= mask_sram[{tile_index, 3'b100}];
                tile_bitmap[47:40] <= mask_sram[{tile_index, 3'b101}];
                tile_bitmap[55:48] <= mask_sram[{tile_index, 3'b110}];
                tile_bitmap[63:56] <= mask_sram[{tile_index, 3'b111}];
            end

            // ── Cycle 1: Popcount compare → result ─────────────
            if (eval_req_r) begin
                tiles_evaluated <= tiles_evaluated + 1;

                if (pop64 < SPARSITY_THRESH) begin
                    // Tile is sparse — suppress
                    tile_active      <= 1'b0;
                    tiles_suppressed <= tiles_suppressed + 1;
                    cycles_saved     <= cycles_saved + 64; // 8x8 tile = 64 MAC ops saved
                end else begin
                    // Tile is dense enough — compute
                    tile_active <= 1'b1;
                end

                eval_done <= 1'b1;
            end
        end
    end

endmodule
