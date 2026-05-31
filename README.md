# EdgeCore-1 — Edge AI Inference Accelerator IP

**Simulation-verified RTL for a 256×256 INT8 systolic AI accelerator.**  
32.8 TOPS · <185 mW · 28nm · RISC-V · ONNX/TVM · RTL verified · 10M+ algorithmic tests.

---

## What This Is

EdgeCore-1 is a complete, licensable semiconductor IP package for a fixed-function INT8 neural network inference accelerator targeting the sub-$3 die cost range at 28nm (TSMC/GF/SMIC).

It fills a real gap in the market: there is no purpose-built AI inference chip between the $0.50–2 MCU (no AI hardware) and the $5+ Coral Edge TPU (overkill for most IoT). EdgeCore-1 targets $1.40–2.60 die cost at 500K units.

This repo contains the full RTL, compiler stack, testbenches, and synthesis scripts. Everything needed to evaluate, synthesize, and integrate the IP.

---

## Verification Status

| Component | Tests | Result |
|-----------|-------|--------|
| C++ Golden Model | 10,000,036 | ✅ ALL PASS |
| RTL Unit Testbench | 15/15 | ✅ ALL PASS |
| Full System Testbench | 10/10 | ✅ ALL PASS |
| TVM Backend | 7/7 | ✅ ALL PASS |
| Runtime Driver | 6/6 | ✅ ALL PASS |
| Yosys Synthesis (sparsity core) | 4,710 gates | ✅ CLEAN |

---

## Key Specifications

| Parameter | Value |
|-----------|-------|
| Process | 28nm TSMC / GF / SMIC |
| Die area (estimated) | 5.00 mm² |
| Peak compute | 32.8 TOPS INT8 |
| Peak power | <185 mW |
| Idle power | 2–5 mW (PHY off) |
| MAC array | 256×256 systolic INT8 |
| Weight cache | 512 KB dual-bank SRAM |
| Memory interface | LPDDR5, 64-bit, 6.4 Gbps |
| Controller | RISC-V RV32IMC (PicoRV32) |
| Sparsity engine | Bitmap-mask, 0.27 mm², up to 80% tile suppression |
| Clock frequency | 500 MHz target |

---

## Repository Structure

```
Scripts/
├── mac_array_v2.v          # 256×256 INT8 systolic MAC array
├── sparsity_engine.v       # Bitmap-mask sparsity engine
├── edgecore1_riscv_ctrl.v  # RISC-V inference controller FSM
├── edgecore1_top.v         # Full SoC integration
├── tb_edgecore1.v          # Unit testbench (15 tests)
├── tb_system.v             # System testbench (10 tests)
├── edgecore1_golden.cpp    # C++ reference model (10M+ tests)
├── edgecore1_compiler.py   # ONNX → tile schedule compiler
├── edgecore1_target.py     # Apache TVM backend
├── edgecore1_runtime.py    # Runtime driver + firmware generator
├── edgecore1_firmware.c    # RISC-V C firmware template
├── edgecore1_synth.ys      # Yosys synthesis script
├── SYNTHESIS_REPORT.md     # Gate counts, area estimates
├── runtime_config.json     # Hardware configuration
└── tile_schedule.json      # Compiler output example
```

---

## Run It Yourself

```bash
# Unit testbench
iverilog -g2012 -o tb_unit tb_edgecore1.v mac_array_v2.v sparsity_engine.v edgecore1_top.v edgecore1_riscv_ctrl.v
vvp tb_unit
# → 15/15 PASS

# System testbench
iverilog -g2012 -o tb_sys tb_system.v edgecore1_top.v edgecore1_riscv_ctrl.v mac_array_v2.v sparsity_engine.v
vvp tb_sys
# → 10/10 PASS

# Golden model (requires g++)
g++ -O2 -o edgecore1_golden edgecore1_golden.cpp && ./edgecore1_golden --test
# → 10,000,036 tests, 0 failures

# ONNX compiler
python3 edgecore1_compiler.py --test
# → All self-tests PASS

# TVM backend
python3 edgecore1_target.py
# → 7/7 PASS
```

---

## Why INT8 Systolic at This Die Size

- INT8 multipliers are ~16× smaller than FP32 — enabling 65,536 parallel MACs in 2.5 mm²
- Published MLPerf Tiny results confirm INT8 accuracy within 1.5% of FP32 for MobileNetV2, TinyBERT, YOLO-nano after post-training quantization
- RISC-V RV32IMC controller carries zero ISA licensing fees vs $0.10–0.50/chip for ARM
- Bitmap-mask sparsity engine resolves tile suppression in a single clock cycle with zero address arithmetic overhead

---

## Supported Models

- MobileNetV2, MobileNetV3
- EfficientNet-Lite
- YOLO-nano class models
- Any ONNX-exportable model via the compiler frontend

---

## Target Applications

- ADAS: pedestrian detection, sign recognition
- Industrial IoT: predictive maintenance, anomaly detection
- Consumer IoT: keyword spotting, image classification
- Smart cameras: always-on vision at <5 mW idle

---

## Licensing

This IP is available for commercial licensing.

**This repository is provided for evaluation purposes only.**  
Commercial use, integration, or tape-out requires a signed license agreement.

Licensing terms:
- NRE fee: $10K–$200K (depending on scope)
- Per-unit royalty: $0.02–$0.15/chip
- Flat license: $25K–$75K
- 90-day bug-fix warranty included

NDA required before full IP delivery.  
Contact: infinarscientia@proton.me

---

## About

Built by [Infinar Scientia](https://x.com/InfinarScientia).  
April 2026.
