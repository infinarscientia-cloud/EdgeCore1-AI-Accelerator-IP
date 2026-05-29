`timescale 1ns/1ps
// EdgeCore-1 — Sparsity Engine COMBINATIONAL CORE
// Synthesized separately from the SRAM (which maps to BRAM in real flow)
// This module: popcount comparator + FSM + tile_active decision logic
// Area reported here × tile count + BRAM macro = total sparsity engine area

module sparsity_core #(
    parameter TILE_BITS       = 64,
    parameter SPARSITY_THRESH = 13
) (
    input  wire        clk,
    input  wire        rst_n,
    // From SRAM: 8 bytes = 64-bit tile bitmap
    input  wire [63:0] tile_bitmap,
    input  wire        bitmap_valid,   // 1 cycle after SRAM read
    // Outputs
    output reg         tile_active,    // 1=compute, 0=suppress
    output reg         eval_done,
    output reg  [31:0] tiles_evaluated,
    output reg  [31:0] tiles_suppressed,
    output reg  [31:0] cycles_saved
);
    // 7-bit popcount: count set bits in 64-bit bitmap
    function [6:0] popcount64;
        input [63:0] x;
        integer k;
        reg [6:0] cnt;
        begin
            cnt = 0;
            for (k = 0; k < 64; k = k + 1)
                cnt = cnt + x[k];
            popcount64 = cnt;
        end
    endfunction

    reg [6:0] pop;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tile_active      <= 0;
            eval_done        <= 0;
            tiles_evaluated  <= 0;
            tiles_suppressed <= 0;
            cycles_saved     <= 0;
        end else begin
            eval_done <= 0;
            if (bitmap_valid) begin
                pop = popcount64(tile_bitmap);
                tile_active <= (pop >= SPARSITY_THRESH) ? 1'b1 : 1'b0;
                eval_done   <= 1;
                tiles_evaluated <= tiles_evaluated + 1;
                if (pop < SPARSITY_THRESH) begin
                    tiles_suppressed <= tiles_suppressed + 1;
                    cycles_saved     <= cycles_saved + 8; // 8 MAC cycles saved per suppressed tile
                end
            end
        end
    end
endmodule
