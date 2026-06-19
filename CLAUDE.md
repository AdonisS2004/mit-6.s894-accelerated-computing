# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Self-guided run-through of MIT 6.S894 (Accelerated Computing, Fall 2024). Each lab is a git submodule pointing to `git@github.com:accelerated-computing-class/lab<N>.git`. Labs 1–11 cover GPU programming from basic CUDA through H100-specific features and multi-device collectives.

Labs are run on MIT's Engaging HPC cluster. The full cluster workflow is documented in `docs/6S894-engaging-workflow.md`.

## Repo Structure

```
.
├── docs/
│   └── 6S894-engaging-workflow.md   # Engaging cluster runbook
├── lab.mk                           # Reusable Makefile, copy into any lab
├── lab1/ … lab11/                   # Git submodules
├── CLAUDE.md
└── README.md
```

## Submodule Setup

After cloning, initialize all labs with:
```
git submodule update --init --recursive
```

## Building and Running Labs

A reusable `lab.mk` lives at the repo root. Copy it into any lab before building:
```bash
cp ../lab.mk Makefile
make                                          # labs 1–8
make ARCH=sm_90a LDFLAGS="-lcuda -lcublas"   # labs 9–10
```

`make` auto-creates the `out/` directory that the harness writes images to.

### GPU architecture per lab
| Labs  | GPU           | `ARCH`    |
|-------|---------------|-----------|
| 1–8   | L40S (Ada)    | `sm_89`   |
| 9–10  | H100 (Hopper) | `sm_90a`  |

`sm_90a` (not plain `sm_90`) is required for labs 9–10 — it enables TMA and WGMMA.

### Explicit nvcc commands (without the Makefile)
```bash
# Labs 1–8
nvcc -O3 -std=c++17 -arch=sm_89 <file>.cu -o <file>
# Labs 9–10
nvcc -O3 -std=c++17 -arch=sm_90a -lcuda -lcublas <file>.cu -o <file>
# CPU files
g++ -O3 -std=c++17 -march=native <file>.cpp -o <file>
```

Labs with `.clang-format` (labs 2–8) use 4-space indent and 90-column limit — run `clang-format -i <file>.cu` to format.

## Engaging Cluster — Loading CUDA

`cuda/12.4.0` requires a prerequisite module on Engaging:
```bash
module load deprecated-modules
module load cuda/12.4.0
```

## Lab Progression

| Lab | Topic | Key Files |
|-----|-------|-----------|
| 1 | GPU basics — Mandelbrot set (scalar + vector kernels) | `mandelbrot_gpu.cu` |
| 2 | FMA latency, warp scheduling, Mandelbrot optimization | `fma_latency.cu`, `warp_scheduler.cu` |
| 3 | Memory coalescing, memory latency | `coalesced-loads.cu`, `mem-latency.cu` |
| 4 | Matrix multiplication with L1/shared memory tiling | `matmul.cu` |
| 5 | Matmul occupancy tuning across multiple shapes | `matmul_2.cu`, `occupancy.cu` |
| 6 | Tensor core (MMA) matrix multiplication | `matmul_3.cu`, `exercise_mma.cu` |
| 7 | Scan, shuffle, RLE compression/decompression | `scan.cu`, `shuffle.cu`, `rle_*.cu` |
| 8 | Circles rendering (parallel primitives) | `circles.cu` |
| 9 | H100 TMA — load, store, reduce, swizzle | `0-tma-*.cu` through `5-tma-swizzle.cu` |
| 10 | H100 WGMMA + TMA matrix multiplication | `h100-matmul.cu`, `wgmma-interface.cuh` |
| 11 | Multi-device collectives with JAX Pallas on TPU | `collective_matmul.py`, `collectives.py` |

## Key Patterns

**CUDA_CHECK macro** — all labs define this wrapper around CUDA API calls that prints file/line on error and exits. Use it for every `cuda*` call.

**BENCHPRESS macro** (labs 1–6) — runs a kernel 5 outer × 3 inner iterations, sorts times, and prints the minimum. The kernel launch function is passed as the first argument.

**TL+ directives** (labs 9–10) — comment-based metadata at the top of `.cu` files for the course's `telerun` tool (not used here, but don't remove them):
```cpp
// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh"]}
// TL+ {"compile_flags": ["-lcuda", "-lcublas"]}
```

**Test data** — labs 4–6 ship binary `.bin` files for correctness checking. `gen_test_data.py` regenerates them. The matmul labs read `test_<M>x<K>x<N>_{a,b,c}.bin` and compare GPU output against the stored reference.

**Student code regions** — marked with `/// <--- your code here --->` and `/* TODO: your code here */`. The harness below these regions handles correctness checking, benchmarking, and I/O — do not modify it.

## Lab 11 (JAX/Python)

Lab 11 uses JAX with Pallas for TPU programming — not a CUDA lab. Dependencies: `jax`, `numpy`. The `ENABLE_DEBUG` flag in `collective_matmul.py` enables `jax.debug.print`, out-of-bounds detection, and race-condition detection at the cost of performance. Set it to `False` before timing runs.
