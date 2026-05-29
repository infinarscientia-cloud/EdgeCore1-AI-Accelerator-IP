# ============================================================
#  EdgeCore-1 — Runtime Driver
#  Infinar Scientia | April 2026
#
#  Bridges compiler output (tile_schedule.json, sparsity_masks.bin)
#  to the EdgeCore-1 hardware interface.
#
#  In simulation: drives the Verilog DUT via cocotb or
#  generates C stimulus for the RTL testbench.
#
#  In production: this compiles to RISC-V binary that runs on
#  the PicoRV32 core inside EdgeCore-1 and controls the MAC
#  array via memory-mapped CSRs.
#
#  USAGE (Python simulation):
#    driver = EdgeCore1Runtime("output/")
#    result = driver.run_inference(input_tensor)
#    print(f"Class: {result.argmax()}")
#
#  USAGE (RISC-V binary generation):
#    driver = EdgeCore1Runtime("output/", target="riscv")
#    driver.compile_firmware("edgecore1_firmware.hex")
# ============================================================

import json
import struct
import numpy as np
from typing import Optional, Dict, List
import os


# CSR memory map — must match edgecore1_top.v comments
CSR_BASE        = 0x4000_0000
CSR_MAC_CTRL    = CSR_BASE + 0x00   # [0]=start [1]=clear [2]=clk_en
CSR_TILE_INDEX  = CSR_BASE + 0x04
CSR_SPARSE_WR   = CSR_BASE + 0x08   # [0]=wr_en [25:8]=addr [33:26]=data
CSR_SPARSE_REQ  = CSR_BASE + 0x0C   # [0]=eval_req
CSR_SPARSE_RD   = CSR_BASE + 0x10   # [0]=tile_active [1]=eval_done (RO)
CSR_ACC_CTRL    = CSR_BASE + 0x14   # [0]=acc_clear
CSR_DMA_ADDR    = CSR_BASE + 0x18
CSR_DMA_LEN     = CSR_BASE + 0x1C
CSR_STATUS      = CSR_BASE + 0x20   # [0]=layer_done [1]=phy_ready
ACC_BASE        = 0x8000_0000        # Accumulator read-back


class EdgeCore1Runtime:
    """
    Software runtime driver for EdgeCore-1.
    Orchestrates inference by feeding tiles to hardware.
    """

    def __init__(self, schedule_dir: str = "output", target: str = "sim"):
        self.schedule_dir = schedule_dir
        self.target = target  # "sim" or "riscv"
        self._schedule = None
        self._masks    = None
        self._config   = None
        self._sim_accumulators = np.zeros(256, dtype=np.int32)
        self._load_artifacts()

    def _load_artifacts(self):
        """Load compiler-generated artifacts."""
        sched_path  = os.path.join(self.schedule_dir, "tile_schedule.json")
        mask_path   = os.path.join(self.schedule_dir, "sparsity_masks.bin")
        config_path = os.path.join(self.schedule_dir, "runtime_config.json")

        if os.path.exists(sched_path):
            with open(sched_path) as f:
                self._schedule = json.load(f)

        if os.path.exists(mask_path):
            raw = np.frombuffer(open(mask_path, "rb").read(), dtype=np.uint8)
            n_tiles = len(raw) // 8
            self._masks = raw[:n_tiles*8].reshape(n_tiles, 8)

        if os.path.exists(config_path):
            with open(config_path) as f:
                self._config = json.load(f)

    # ── Hardware interface (simulation model) ─────────────────
    def _csr_write(self, addr: int, value: int):
        """Simulate a CSR write (in production: RISC-V sw instruction)."""
        pass  # Simulation: state tracked in Python objects

    def _csr_read(self, addr: int) -> int:
        """Simulate a CSR read."""
        return 0

    def _load_sparsity_masks(self, masks: np.ndarray):
        """Write all tile masks to sparsity engine SRAM."""
        # In production RISC-V code:
        #   for each tile t, for each byte b:
        #     sw (addr=t*8+b, data=mask[t,b]) → CSR_SPARSE_WR
        self._loaded_masks = masks

    def _evaluate_tile_sparsity(self, tile_id: int) -> bool:
        """
        Ask sparsity engine if tile should compute.
        Returns True = compute, False = suppress.
        """
        if self._loaded_masks is None or tile_id >= len(self._loaded_masks):
            return True  # Default: always compute
        bitmap = self._loaded_masks[tile_id]
        popcount = int(np.unpackbits(bitmap).sum())
        thresh = self._config["hw_params"]["sparsity_thresh"] if self._config else 13
        return popcount >= thresh

    def _execute_mac_tile(
        self,
        activation_tile: np.ndarray,
        weight_tile: np.ndarray
    ) -> np.ndarray:
        """
        Execute one 8×8 INT8 MAC tile.
        Returns 8-element INT32 partial sum row.
        In production: hardware does this in 8 clock cycles.
        """
        a = activation_tile.astype(np.int32)
        w = weight_tile.astype(np.int32)
        # Matrix multiply: (8,8) × (8,8) = (8,8), return row 0
        result = a @ w
        return result[0]  # Row 0 of the tile output

    def _clear_accumulators(self):
        """Assert acc_clear for one cycle."""
        self._sim_accumulators = np.zeros(256, dtype=np.int32)

    def _read_accumulators(self) -> np.ndarray:
        """Read 256 INT32 accumulator values."""
        return self._sim_accumulators.copy()

    # ── Inference pipeline ────────────────────────────────────
    def run_inference(
        self,
        input_tensor: np.ndarray,
        weight_tensor: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Run one complete inference pass.

        Args:
            input_tensor: INT8 input activations, shape (C, H, W) or flat
            weight_tensor: INT8 weights (optional — uses schedule if None)

        Returns dict with:
            class_idx:       Predicted class (argmax)
            acc_values:      Raw accumulator outputs
            tiles_computed:  Number of MAC tiles executed
            tiles_suppressed: Number of tiles skipped (sparsity)
            cycles_estimated: Estimated clock cycles
        """
        # Step 1: Load sparsity masks
        if self._masks is not None:
            self._load_sparsity_masks(self._masks)
        else:
            self._loaded_masks = None

        # Step 2: Clear accumulators
        self._clear_accumulators()

        # Step 3: Process tiles
        tiles_computed   = 0
        tiles_suppressed = 0
        input_flat = input_tensor.flatten().astype(np.int8)

        if weight_tensor is not None:
            weight_flat = weight_tensor.flatten().astype(np.int8)
        else:
            # Generate synthetic weights for simulation
            weight_flat = np.tile(
                np.eye(8, dtype=np.int8).flatten(), 1024
            )[:len(input_flat)]

        schedule = self._schedule["tiles"] if self._schedule else []

        if len(schedule) == 0:
            # No schedule: generate simple tiled pass
            n_tiles = max(1, len(input_flat) // 64)
            schedule = [{"tile_id": i, "op": "dense"} for i in range(n_tiles)]

        for tile_desc in schedule:
            tile_id = tile_desc["tile_id"]

            # Sparsity check
            if not self._evaluate_tile_sparsity(tile_id):
                tiles_suppressed += 1
                continue

            # Extract 8×8 activation and weight tiles
            offset = (tile_id * 64) % max(len(input_flat) - 64, 1)
            a_tile = input_flat[offset:offset+64]
            if len(a_tile) < 64:
                a_tile = np.pad(a_tile, (0, 64-len(a_tile)))
            a_tile = a_tile.reshape(8, 8)

            w_offset = (tile_id * 64) % max(len(weight_flat) - 64, 1)
            w_tile = weight_flat[w_offset:w_offset+64]
            if len(w_tile) < 64:
                w_tile = np.pad(w_tile, (0, 64-len(w_tile)))
            w_tile = w_tile.reshape(8, 8)

            # Execute tile on MAC array
            partial_sum = self._execute_mac_tile(a_tile, w_tile)

            # Accumulate into first 8 outputs
            acc_idx = (tile_id * 8) % 256
            self._sim_accumulators[acc_idx:acc_idx+8] += partial_sum
            tiles_computed += 1

        # Step 4: Read accumulators and find argmax (classification)
        acc = self._read_accumulators()
        class_idx = int(np.argmax(acc))

        cycles = (tiles_computed * 8) + (tiles_suppressed * 1)  # 8 cycles compute, 1 cycle skip

        return {
            "class_idx":        class_idx,
            "acc_values":       acc,
            "tiles_total":      len(schedule),
            "tiles_computed":   tiles_computed,
            "tiles_suppressed": tiles_suppressed,
            "sparsity_ratio":   tiles_suppressed / max(len(schedule), 1),
            "cycles_estimated": cycles,
            "fps_at_500mhz":    500e6 / max(cycles, 1),
        }

    def generate_riscv_firmware(self, output_path: str = "edgecore1_firmware.c"):
        """
        Generate C firmware that runs on PicoRV32 inside EdgeCore-1.
        Compile with: riscv32-unknown-elf-gcc -march=rv32imc -O2
        """
        n_tiles = len(self._schedule["tiles"]) if self._schedule else 0
        thresh  = self._config["hw_params"]["sparsity_thresh"] if self._config else 13

        c_code = f"""/* EdgeCore-1 Firmware — Auto-generated by edgecore1_runtime.py
 * Infinar Scientia | April 2026
 * Compile: riscv32-unknown-elf-gcc -march=rv32imc -O2 -o firmware.elf firmware.c
 * Link:    riscv32-unknown-elf-objcopy -O ihex firmware.elf firmware.hex
 */
#include <stdint.h>

/* CSR Memory Map */
#define CSR_MAC_CTRL   (*(volatile uint32_t*)0x40000000)
#define CSR_TILE_INDEX (*(volatile uint32_t*)0x40000004)
#define CSR_SPARSE_WR  (*(volatile uint32_t*)0x40000008)
#define CSR_SPARSE_REQ (*(volatile uint32_t*)0x4000000C)
#define CSR_SPARSE_RD  (*(volatile uint32_t*)0x40000010)
#define CSR_ACC_CTRL   (*(volatile uint32_t*)0x40000014)
#define CSR_STATUS     (*(volatile uint32_t*)0x40000020)
#define ACC_BASE       ((volatile int32_t*)0x80000000)

#define SPARSITY_THRESH {thresh}
#define NUM_TILES       {n_tiles}

static inline void write_mask_byte(uint32_t tile, uint32_t byte_idx, uint8_t val) {{
    CSR_SPARSE_WR = (1u) | ((tile*8 + byte_idx) << 8) | ((uint32_t)val << 26);
    CSR_SPARSE_WR = 0; /* deassert wr_en */
}}

static inline int tile_is_active(uint32_t tile_idx) {{
    CSR_TILE_INDEX = tile_idx;
    CSR_SPARSE_REQ = 1;
    CSR_SPARSE_REQ = 0;
    while (!(CSR_SPARSE_RD & 0x2)); /* wait eval_done */
    return (CSR_SPARSE_RD & 0x1);  /* tile_active */
}}

static inline void wait_layer_done(void) {{
    while (!(CSR_STATUS & 0x1)); /* poll layer_done */
}}

void edgecore1_inference(const int8_t *activations, const int8_t *weights,
                          uint8_t *result_class) {{
    uint32_t t, b;

    /* 1. Clear accumulators */
    CSR_ACC_CTRL = 1; CSR_ACC_CTRL = 0;

    /* 2. Load sparsity masks (from weight tensor, pre-computed by TVM) */
    /* In production: masks DMA'd from LPDDR5 before this function */
    /* Here: placeholder — real masks written by DMA engine */

    /* 3. Enable MAC clock */
    CSR_MAC_CTRL = 0x4; /* clk_en=1 */

    /* 4. Dispatch tiles */
    for (t = 0; t < NUM_TILES; t++) {{
        if (tile_is_active(t)) {{
            /* Write tile_a and tile_w via DMA (simplified: direct write) */
            /* In production: DMA transfers 512 bits from SRAM to MAC input regs */
            CSR_MAC_CTRL = 0x5; /* start=1, clk_en=1 */
            CSR_MAC_CTRL = 0x4; /* deassert start */
        }}
    }}

    /* 5. Wait for completion */
    wait_layer_done();

    /* 6. Gate MAC clock to save power */
    CSR_MAC_CTRL = 0x0;

    /* 7. Argmax over 256 accumulators */
    int32_t max_val = ACC_BASE[0];
    uint8_t max_idx = 0;
    for (uint32_t i = 1; i < 256; i++) {{
        if (ACC_BASE[i] > max_val) {{
            max_val = ACC_BASE[i];
            max_idx = (uint8_t)i;
        }}
    }}
    *result_class = max_idx;
}}

int main(void) {{
    /* Placeholder: in production, activations loaded from LPDDR5 */
    static int8_t activations[512] = {{0}};
    static int8_t weights[512]     = {{0}};
    uint8_t result = 0;
    while (1) {{
        edgecore1_inference(activations, weights, &result);
        /* Result written to output register — host reads via SPI */
    }}
    return 0;
}}
"""
        with open(output_path, "w") as f:
            f.write(c_code)
        return output_path

    # ── Self-test ─────────────────────────────────────────────
    @classmethod
    def self_test(cls):
        import tempfile, sys
        from edgecore1_target import EdgeCore1Target

        print("EdgeCore-1 Runtime Driver — Self-Test")
        print("=" * 44)
        failures = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate schedule
            target = EdgeCore1Target()
            target.tile_dense("fc1", 64, 32)
            w = np.random.randint(-5, 5, (64,), dtype=np.int8)
            masks, ratio = target.generate_masks_from_weights(w, "fc1")
            target.export_schedule(tmpdir)

            driver = cls(tmpdir)

            # Test 1: inference returns expected structure
            inp = np.random.randint(-10, 10, (64,), dtype=np.int8)
            result = driver.run_inference(inp)
            ok = all(k in result for k in
                     ["class_idx","tiles_computed","tiles_suppressed","fps_at_500mhz"])
            print(f"  [{'PASS' if ok else 'FAIL'}] Inference returns complete result dict")
            if not ok: failures += 1

            # Test 2: class index in valid range
            ok = 0 <= result["class_idx"] <= 255
            print(f"  [{'PASS' if ok else 'FAIL'}] class_idx={result['class_idx']} in [0,255]")
            if not ok: failures += 1

            # Test 3: sparsity ratio reasonable
            ok = 0.0 <= result["sparsity_ratio"] <= 1.0
            print(f"  [{'PASS' if ok else 'FAIL'}] sparsity_ratio={result['sparsity_ratio']:.2f} in [0,1]")
            if not ok: failures += 1

            # Test 4: FPS estimate > 0
            ok = result["fps_at_500mhz"] > 0
            print(f"  [{'PASS' if ok else 'FAIL'}] fps_at_500mhz={result['fps_at_500mhz']:.1f} > 0")
            if not ok: failures += 1

            # Test 5: deterministic (same input → same result)
            result2 = driver.run_inference(inp)
            ok = result["class_idx"] == result2["class_idx"]
            print(f"  [{'PASS' if ok else 'FAIL'}] Deterministic: two runs → same class {result['class_idx']}")
            if not ok: failures += 1

            # Test 6: firmware generation
            fw_path = os.path.join(tmpdir, "firmware.c")
            driver.generate_riscv_firmware(fw_path)
            ok = os.path.exists(fw_path) and os.path.getsize(fw_path) > 500
            print(f"  [{'PASS' if ok else 'FAIL'}] RISC-V firmware C file generated ({os.path.getsize(fw_path)} bytes)")
            if not ok: failures += 1

        print("=" * 44)
        print(f"  {'ALL PASS' if failures == 0 else str(failures) + ' FAILURES'}")
        return failures


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    failures = EdgeCore1Runtime.self_test()
    sys.exit(failures)
