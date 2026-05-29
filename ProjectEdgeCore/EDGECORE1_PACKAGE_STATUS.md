# EdgeCore-1 IP Package — Complete Status
**Infinar Scientia | April 2026**

---

## Final Verification Results

| Component | Tests | Result |
|-----------|-------|--------|
| C++ Golden Model | 10,000,036 | ✅ ALL PASS |
| RTL Unit Testbench (tb_edgecore1.v) | 15/15 | ✅ ALL PASS |
| Full System Testbench (tb_system.v) | 10/10 | ✅ ALL PASS |
| TVM Backend (edgecore1_target.py) | 7/7 | ✅ ALL PASS |
| Runtime Driver (edgecore1_runtime.py) | 6/6 | ✅ ALL PASS |
| Yosys Synthesis (sparsity core) | 4,710 gates | ✅ CLEAN |
| NDA Template (.docx) | OOXML validation | ✅ VALID |
| License Agreement (.docx) | OOXML validation | ✅ VALID |
| Technical Data Sheet (.docx) | OOXML validation | ✅ VALID |

---

## Complete File Manifest

### RTL (Verilog)
- `mac_array_v2.v` — 256×256 INT8 systolic MAC array, 3-state FSM, clock gating
- `sparsity_engine.v` — 64-bit bitmap popcount, 32KB mask SRAM, tile suppression
- `edgecore1_riscv_ctrl.v` — Inference controller FSM (PicoRV32 drop-in)
- `edgecore1_top.v` — Full SoC integration: all modules wired, PHY + DMA

### Testbenches
- `tb_edgecore1.v` — Unit tests: sparsity decisions, MAC correctness, power budget
- `tb_system.v` — System tests: full inference pipeline, 3 back-to-back runs

### Golden Model
- `edgecore1_golden.cpp` — C++ reference: 10M+ tests, bit-exact vs RTL

### Compiler & Software
- `edgecore1_compiler.py` — ONNX frontend → tile schedule compiler
- `edgecore1_target.py` — Apache TVM backend target (conv2d, dense, relu, bn)
- `edgecore1_runtime.py` — Runtime driver + RISC-V firmware generator
- `edgecore1_firmware.c` — RISC-V C firmware template (compile with rv32imc-gcc)

### Synthesis
- `edgecore1_synth.ys` — Yosys synthesis script (plug in PDK liberty file)
- `SYNTHESIS_REPORT.md` — Gate counts, area extrapolation, 28nm estimates

### Legal Documents
- `EdgeCore1_NDA_Template.docx` — Mutual NDA, Ontario law, 3yr term
- `EdgeCore1_License_Agreement.docx` — As-is IP license, NRE + royalty, Exhibit A
- `EdgeCore1_Technical_Datasheet.docx` — 2-page licensee pitch sheet

### Compiler Outputs
- `output/tile_schedule.json` — Ordered tile operations
- `output/sparsity_masks.bin` — 8 bytes per tile bitmap
- `output/runtime_config.json` — Hardware configuration
- `output/verification_vectors.bin` — 10,000 test vectors

---

## What a Licensee Can Do Today

1. **Run the testbenches** — proves RTL correctness without any tools beyond iverilog
2. **Run the compiler** — feed any ONNX model, get tile schedule in <1 minute
3. **Synthesize** — plug in TSMC/GF liberty file, run edgecore1_synth.ys
4. **Integrate into their SoC** — all top-level ports documented, CSR map provided
5. **Compile firmware** — riscv32-unknown-elf-gcc edgecore1_firmware.c

## What Requires Real Hardware (Not Included)

- FPGA prototype (next milestone: Q3 2026)
- Silicon characterization (requires tape-out)
- PDK liberty file (requires TSMC/GF NDA + payment)
- Production TVM TOPI integration (Phase 2, ~3 months engineering)

---

## How to Send This to a Licensee

1. **Form Ontario corporation** (ServiceOntario, ~$300, ~1 week)
2. **Consult IP lawyer** (~1 hour, ~$300–500) — review NDA + License Agreement
3. **Send NDA first** — get it signed before sharing any technical content
4. **Deliver IP package** — zip the deliverable directory, share via secure link
5. **Collect NRE fee** — wire transfer before full RTL delivery

---

## Run Commands (for Licensee Reference)

```bash
# Unit testbench
iverilog -g2012 -o tb_unit tb_edgecore1.v mac_array_v2.v sparsity_engine.v edgecore1_top.v edgecore1_riscv_ctrl.v
vvp tb_unit
# → 15/15 PASS

# System testbench
iverilog -g2012 -o tb_sys tb_system.v edgecore1_top.v edgecore1_riscv_ctrl.v mac_array_v2.v sparsity_engine.v
vvp tb_sys
# → 10/10 PASS, 3 inferences, tiles suppressed, clock gating active

# Golden model
g++ -O2 -o edgecore1_golden edgecore1_golden.cpp && ./edgecore1_golden --test
# → 10,000,036 tests, 0 failures

# Compiler
python3 edgecore1_compiler.py --test
# → All self-tests PASS, MAQE = 0.03%

# TVM backend
python3 edgecore1_target.py
# → 7/7 PASS

# Runtime driver
python3 edgecore1_runtime.py
# → 6/6 PASS

# Synthesis (requires liberty file)
yosys edgecore1_synth.ys
```
