// ============================================================
//  EdgeCore-1 — Inference Controller (RISC-V FSM)
//  Infinar Scientia | April 2026
//
//  Replaces PicoRV32 for simulation. Same CSR interface.
//  To use real PicoRV32: see CSR map in edgecore1_top.v
// ============================================================
`timescale 1ns/1ps

module edgecore1_riscv_ctrl #(
    parameter NUM_TILES  = 16,
    parameter TILE_SIZE  = 8,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
) (
    input  wire         clk, rst_n,
    output reg          mac_clk_en, acc_clear, tile_valid,
    output reg  [511:0] tile_a, tile_w,
    input  wire         layer_done, acc_valid,
    output reg          mask_wr_en, eval_req,
    output reg  [17:0]  mask_wr_addr,
    output reg  [7:0]   mask_wr_data,
    output reg  [13:0]  tile_index,
    input  wire         eval_done, tile_active,
    output reg  [31:0]  next_layer_addr,
    output reg  [19:0]  next_layer_bytes,
    input  wire         phy_ready,
    input  wire [8191:0] acc_out,
    output reg  [7:0]   result_class,
    output reg          result_valid,
    output reg  [31:0]  inference_count,
    output reg  [2:0]   ctrl_state_dbg
);
    localparam S_RESET=3'd0, S_LOAD_MASKS=3'd1, S_EVAL_SPARSE=3'd2;
    localparam S_DISPATCH=3'd3, S_WAIT_DONE=3'd4;
    localparam S_READBACK=3'd5, S_OUTPUT=3'd6;

    reg [2:0]  state;
    reg [13:0] current_tile;
    reg [2:0]  mask_byte_idx;
    reg [31:0] argmax_val;
    reg [7:0]  argmax_idx;
    reg [7:0]  scan_idx;
    reg [31:0] scan_val;

    // ── Tile bitmap lookup (one byte at a time) ────────────────
    function [7:0] tile_mask_byte;
        input [13:0] tile_idx;
        input [2:0]  byte_idx;
        reg [63:0] bmap;
        begin
            case (tile_idx % 6)
                0: bmap = 64'hFFFFFFFF_FFFFFFFF;
                1: bmap = 64'hF0F0F0F0_F0F0F0F0;
                2: bmap = 64'hAAAAAAAA_AAAAAAAA;
                3: bmap = 64'h0000000F_00000000;
                4: bmap = 64'h00000000_00000003;
                5: bmap = 64'hFF000000_FF000000;
                default: bmap = 64'hFFFFFFFF_FFFFFFFF;
            endcase
            case (byte_idx)
                0: tile_mask_byte = bmap[7:0];
                1: tile_mask_byte = bmap[15:8];
                2: tile_mask_byte = bmap[23:16];
                3: tile_mask_byte = bmap[31:24];
                4: tile_mask_byte = bmap[39:32];
                5: tile_mask_byte = bmap[47:40];
                6: tile_mask_byte = bmap[55:48];
                default: tile_mask_byte = bmap[63:56];
            endcase
        end
    endfunction

    // ── Tile data (structured test pattern) ───────────────────
    function [511:0] make_activation_tile;
        input [13:0] tile_idx;
        integer k;
        reg [511:0] t;
        begin
            t = 512'b0;
            for (k = 0; k < 64; k = k + 1)
                t[k*8 +: 8] = (k[2:0] == k[5:3]) ? 8'd64 : 8'd0;
            make_activation_tile = t;
        end
    endfunction

    function [511:0] make_weight_tile;
        input [13:0] tile_idx;
        integer k;
        reg [511:0] t;
        begin
            t = 512'b0;
            for (k = 0; k < 8; k = k + 1)
                t[(k*8+k)*8 +: 8] = 8'd1; // Identity diagonal
            make_weight_tile = t;
        end
    endfunction

    // ── Main FSM ───────────────────────────────────────────────
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_RESET;
            current_tile    <= 0;
            mask_byte_idx   <= 0;
            mac_clk_en      <= 0;
            acc_clear       <= 0;
            tile_valid      <= 0;
            tile_a          <= 0;
            tile_w          <= 0;
            mask_wr_en      <= 0;
            mask_wr_addr    <= 0;
            mask_wr_data    <= 0;
            eval_req        <= 0;
            tile_index      <= 0;
            next_layer_addr <= 32'h1000_0000;
            next_layer_bytes<= 20'h10000;
            result_class    <= 0;
            result_valid    <= 0;
            inference_count <= 0;
            argmax_val      <= 0;
            argmax_idx      <= 0;
            scan_idx        <= 0;
            scan_val        <= 0;
            ctrl_state_dbg  <= 0;
        end else begin
            tile_valid   <= 0;
            eval_req     <= 0;
            mask_wr_en   <= 0;
            acc_clear    <= 0;
            result_valid <= 0;
            ctrl_state_dbg <= state;

            case (state)
                S_RESET: begin
                    mac_clk_en   <= 1;
                    acc_clear    <= 1;
                    current_tile <= 0;
                    mask_byte_idx<= 0;
                    argmax_val   <= 0;
                    argmax_idx   <= 0;
                    scan_idx     <= 0;
                    state        <= S_LOAD_MASKS;
                end

                S_LOAD_MASKS: begin
                    mask_wr_en   <= 1;
                    mask_wr_addr <= {current_tile, mask_byte_idx};
                    mask_wr_data <= tile_mask_byte(current_tile, mask_byte_idx);
                    if (mask_byte_idx == 3'd7) begin
                        mask_byte_idx <= 0;
                        if (current_tile == NUM_TILES-1) begin
                            current_tile <= 0;
                            state        <= S_EVAL_SPARSE;
                        end else
                            current_tile <= current_tile + 1;
                    end else
                        mask_byte_idx <= mask_byte_idx + 1;
                end

                S_EVAL_SPARSE: begin
                    eval_req   <= 1;
                    tile_index <= current_tile;
                    state      <= S_DISPATCH;
                end

                S_DISPATCH: begin
                    if (eval_done) begin
                        if (tile_active) begin
                            tile_a     <= make_activation_tile(current_tile);
                            tile_w     <= make_weight_tile(current_tile);
                            tile_valid <= 1;
                        end
                        if (current_tile == NUM_TILES-1) begin
                            current_tile <= 0;
                            state        <= S_WAIT_DONE;
                        end else begin
                            current_tile <= current_tile + 1;
                            state        <= S_EVAL_SPARSE;
                        end
                    end
                end

                S_WAIT_DONE: begin
                    if (layer_done) begin
                        mac_clk_en <= 0;
                        scan_idx   <= 0;
                        scan_val   <= 0;
                        argmax_val <= 0;
                        argmax_idx <= 0;
                        state      <= S_READBACK;
                    end
                end

                S_READBACK: begin
                    // Scan accumulators one per cycle to find argmax
                    scan_val = acc_out[scan_idx * ACC_WIDTH +: ACC_WIDTH];
                    if ($signed(scan_val) > $signed(argmax_val)) begin
                        argmax_val <= scan_val;
                        argmax_idx <= scan_idx;
                    end
                    if (scan_idx == 8'd255)
                        state <= S_OUTPUT;
                    else
                        scan_idx <= scan_idx + 1;
                end

                S_OUTPUT: begin
                    result_class    <= argmax_idx;
                    result_valid    <= 1;
                    inference_count <= inference_count + 1;
                    state           <= S_RESET;
                end

                default: state <= S_RESET;
            endcase
        end
    end
endmodule
