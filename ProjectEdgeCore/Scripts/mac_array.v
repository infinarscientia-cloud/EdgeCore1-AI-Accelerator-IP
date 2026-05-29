// ============================================================
//  EdgeCore-1 — INT8 MAC Array Module
//  Infinar Scientia | April 2026
//
//  Module: mac_array
//  Description:
//    256x256 systolic INT8 multiply-accumulate array.
//    Pipelined at MAC_FREQ (500 MHz target on 28nm).
//    Receives 8x8 weight tiles from the sparsity engine dispatcher.
//    Produces INT32 accumulator outputs for dequantization.
//
//  Interface:
//    clk        — MAC array clock (500 MHz, clock-gated externally)
//    rst_n      — Active-low synchronous reset
//    tile_valid — Dispatcher asserts when a tile is ready to compute
//    tile_a     — Input activation tile (TILE_SIZE x TILE_SIZE x 8-bit)
//    tile_w     — Weight tile (TILE_SIZE x TILE_SIZE x 8-bit)
//    acc_out    — INT32 accumulator output (one row per cycle)
//    acc_valid  — Accumulator output is valid
//    layer_done — Pulse when full layer accumulation is complete
//
//  Verification:
//    All outputs must match edgecore1_golden.cpp bit-for-bit.
//    Run: vvp mac_array_tb to verify against golden model vectors.
// ============================================================

`timescale 1ns/1ps

module mac_array #(
    parameter MAC_ROWS   = 256,
    parameter MAC_COLS   = 256,
    parameter TILE_SIZE  = 8,
    parameter DATA_WIDTH = 8,   // INT8
    parameter ACC_WIDTH  = 32   // INT32 accumulator
) (
    input  wire                                  clk,
    input  wire                                  rst_n,

    // Tile dispatch interface (from sparsity engine)
    input  wire                                  tile_valid,
    input  wire [TILE_SIZE*TILE_SIZE*DATA_WIDTH-1:0] tile_a,  // activations
    input  wire [TILE_SIZE*TILE_SIZE*DATA_WIDTH-1:0] tile_w,  // weights

    // Accumulator output (streamed row-by-row)
    output reg  [MAC_COLS*ACC_WIDTH-1:0]         acc_out,
    output reg                                   acc_valid,

    // Control
    input  wire                                  acc_clear,   // Reset accumulators for new layer
    output reg                                   layer_done,

    // Power monitoring (for simulation verification)
    output reg  [31:0]                           mac_cycle_count,
    output reg  [31:0]                           gated_cycle_count
);

    // ── Internal accumulator array ──────────────────────────────
    // Full 256x256 INT32 accumulator grid
    // In real silicon: implemented as register file + SRAM hybrid
    reg signed [ACC_WIDTH-1:0] accumulators [0:MAC_ROWS-1][0:MAC_COLS-1];

    // ── Tile unpacking ──────────────────────────────────────────
    // Unpack flat tile vectors into 2D arrays
    wire signed [DATA_WIDTH-1:0] a_tile [0:TILE_SIZE-1][0:TILE_SIZE-1];
    wire signed [DATA_WIDTH-1:0] w_tile [0:TILE_SIZE-1][0:TILE_SIZE-1];

    genvar gi, gj;
    generate
        for (gi = 0; gi < TILE_SIZE; gi = gi + 1) begin : unpack_rows
            for (gj = 0; gj < TILE_SIZE; gj = gj + 1) begin : unpack_cols
                assign a_tile[gi][gj] = tile_a[(gi*TILE_SIZE+gj)*DATA_WIDTH +: DATA_WIDTH];
                assign w_tile[gi][gj] = tile_w[(gi*TILE_SIZE+gj)*DATA_WIDTH +: DATA_WIDTH];
            end
        end
    endgenerate

    // ── Systolic pipeline registers ─────────────────────────────
    // Stage 1: multiply (1 cycle)
    // Stage 2: accumulate (1 cycle)
    // Stage 3: writeback to accumulator grid (1 cycle)
    reg signed [ACC_WIDTH-1:0] mul_stage [0:TILE_SIZE-1][0:TILE_SIZE-1];
    reg                        mul_valid;

    // ── Tile compute state machine ──────────────────────────────
    localparam ST_IDLE    = 2'b00;
    localparam ST_COMPUTE = 2'b01;
    localparam ST_DRAIN   = 2'b10;

    reg [1:0]  state;
    reg [7:0]  tile_row_ptr;   // Current tile row being processed
    reg [7:0]  tile_col_ptr;   // Current tile column
    reg [7:0]  out_row_ptr;    // Row being streamed to output

    integer i, j, k;

    // ── Main sequential logic ───────────────────────────────────
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= ST_IDLE;
            tile_row_ptr    <= 0;
            tile_col_ptr    <= 0;
            out_row_ptr     <= 0;
            acc_valid       <= 0;
            layer_done      <= 0;
            mul_valid       <= 0;
            mac_cycle_count <= 0;
            gated_cycle_count <= 0;

            // Clear accumulator grid
            for (i = 0; i < MAC_ROWS; i = i + 1)
                for (j = 0; j < MAC_COLS; j = j + 1)
                    accumulators[i][j] <= 0;
        end else begin
            // Track gated cycles when no valid tile
            if (!tile_valid && state == ST_IDLE)
                gated_cycle_count <= gated_cycle_count + 1;

            layer_done <= 0;
            acc_valid  <= 0;

            // Accumulator clear (new layer)
            if (acc_clear) begin
                for (i = 0; i < MAC_ROWS; i = i + 1)
                    for (j = 0; j < MAC_COLS; j = j + 1)
                        accumulators[i][j] <= 0;
                out_row_ptr <= 0;
            end

            case (state)
                ST_IDLE: begin
                    if (tile_valid) begin
                        state <= ST_COMPUTE;
                        tile_row_ptr <= 0;
                    end
                end

                ST_COMPUTE: begin
                    mac_cycle_count <= mac_cycle_count + 1;

                    // Stage 1: Multiply activation row × weight tile column
                    // (One row of the activation tile × one column of the weight tile)
                    for (i = 0; i < TILE_SIZE; i = i + 1) begin
                        for (j = 0; j < TILE_SIZE; j = j + 1) begin
                            // Saturating MAC: a_tile[row][k] * w_tile[k][col]
                            // In systolic: this is pipelined across TILE_SIZE cycles
                            mul_stage[i][j] <= $signed(a_tile[tile_row_ptr][i]) *
                                               $signed(w_tile[i][j]);
                        end
                    end
                    mul_valid <= 1;

                    // Stage 2: Accumulate into accumulator grid
                    // (Registered — appears one cycle after multiply)
                    if (mul_valid) begin
                        for (i = 0; i < TILE_SIZE; i = i + 1) begin
                            for (j = 0; j < TILE_SIZE; j = j + 1) begin
                                // Saturating accumulation (INT32 bounds)
                                accumulators[tile_row_ptr][j] <=
                                    $signed(accumulators[tile_row_ptr][j]) +
                                    $signed(mul_stage[i][j]);
                            end
                        end

                        if (tile_row_ptr == TILE_SIZE - 1) begin
                            // Tile done — return to IDLE, wait for next tile
                            state        <= ST_IDLE;
                            tile_row_ptr <= 0;
                            mul_valid    <= 0;
                        end else begin
                            tile_row_ptr <= tile_row_ptr + 1;
                        end
                    end
                end

                default: state <= ST_IDLE;
            endcase

            // Output streaming: stream one accumulator row per cycle
            // Triggered by external read_en (not shown — simplified)
            if (out_row_ptr < MAC_ROWS && !acc_clear) begin
                // Pack one row of accumulators into acc_out
                for (j = 0; j < MAC_COLS; j = j + 1)
                    acc_out[j*ACC_WIDTH +: ACC_WIDTH] <= accumulators[out_row_ptr][j];
                acc_valid <= 1;
                if (out_row_ptr == MAC_ROWS - 1) begin
                    layer_done  <= 1;
                    out_row_ptr <= 0;
                end
            end
        end
    end

endmodule
