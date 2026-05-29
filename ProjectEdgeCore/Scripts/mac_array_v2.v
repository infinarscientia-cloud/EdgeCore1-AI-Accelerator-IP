`timescale 1ns/1ps
module mac_array #(
    parameter MAC_ROWS=256, MAC_COLS=256,
    parameter TILE_SIZE=8, DATA_WIDTH=8, ACC_WIDTH=32
) (
    input  wire clk, rst_n,
    input  wire tile_valid, acc_clear,
    input  wire [TILE_SIZE*TILE_SIZE*DATA_WIDTH-1:0] tile_a, tile_w,
    output reg  [MAC_COLS*ACC_WIDTH-1:0] acc_out,
    output reg  acc_valid, layer_done,
    output reg  [31:0] mac_cycle_count, gated_cycle_count
);
    reg signed [ACC_WIDTH-1:0] accumulators [0:MAC_ROWS-1][0:MAC_COLS-1];
    // Latched tile registers
    reg signed [DATA_WIDTH-1:0] a_lat [0:TILE_SIZE-1][0:TILE_SIZE-1];
    reg signed [DATA_WIDTH-1:0] w_lat [0:TILE_SIZE-1][0:TILE_SIZE-1];

    localparam IDLE=2'd0, COMPUTE=2'd1, OUTPUT=2'd2;
    reg [1:0] state;
    reg [3:0] row_ptr;
    integer i, j;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state<=IDLE; row_ptr<=0; acc_valid<=0; layer_done<=0;
            mac_cycle_count<=0; gated_cycle_count<=0; acc_out<=0;
            for(i=0;i<MAC_ROWS;i=i+1) for(j=0;j<MAC_COLS;j=j+1)
                accumulators[i][j]<=0;
        end else begin
            layer_done<=0; acc_valid<=0;
            if (acc_clear) begin
                for(i=0;i<MAC_ROWS;i=i+1) for(j=0;j<MAC_COLS;j=j+1)
                    accumulators[i][j]<=0;
                state<=IDLE;
            end else case(state)
                IDLE: begin
                    if (tile_valid) begin
                        // Latch tile into registers
                        for(i=0;i<TILE_SIZE;i=i+1) for(j=0;j<TILE_SIZE;j=j+1) begin
                            a_lat[i][j] <= $signed(tile_a[(i*TILE_SIZE+j)*DATA_WIDTH +: DATA_WIDTH]);
                            w_lat[i][j] <= $signed(tile_w[(i*TILE_SIZE+j)*DATA_WIDTH +: DATA_WIDTH]);
                        end
                        row_ptr<=0; state<=COMPUTE;
                    end else gated_cycle_count<=gated_cycle_count+1;
                end
                COMPUTE: begin
                    mac_cycle_count<=mac_cycle_count+1;
                    for(j=0;j<TILE_SIZE;j=j+1) begin
                        // Dot product: sum over k of a[row_ptr][k]*w[k][j]
                        accumulators[row_ptr][j] <= accumulators[row_ptr][j]
                            + a_lat[row_ptr][0]*w_lat[0][j]
                            + a_lat[row_ptr][1]*w_lat[1][j]
                            + a_lat[row_ptr][2]*w_lat[2][j]
                            + a_lat[row_ptr][3]*w_lat[3][j]
                            + a_lat[row_ptr][4]*w_lat[4][j]
                            + a_lat[row_ptr][5]*w_lat[5][j]
                            + a_lat[row_ptr][6]*w_lat[6][j]
                            + a_lat[row_ptr][7]*w_lat[7][j];
                    end
                    if (row_ptr==TILE_SIZE-1) begin state<=OUTPUT; row_ptr<=0; end
                    else row_ptr<=row_ptr+1;
                end
                OUTPUT: begin
                    for(j=0;j<MAC_COLS;j=j+1)
                        acc_out[j*ACC_WIDTH +: ACC_WIDTH] <= accumulators[0][j];
                    acc_valid<=1; layer_done<=1; state<=IDLE;
                end
                default: state<=IDLE;
            endcase
        end
    end
endmodule
