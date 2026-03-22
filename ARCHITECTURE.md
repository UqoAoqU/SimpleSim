# SimpleSim Architecture Guide

A detailed walkthrough of every component, the data flow between them, and
the design decisions that connect them.

---

## Table of Contents

1. [Project Layout](#1-project-layout)
2. [Design Philosophy](#2-design-philosophy)
3. [Component Reference](#3-component-reference)
   - 3.1 [hardware.py — the GPU model](#31-hardwarepy--the-gpu-model)
   - 3.2 [workload.py — algorithm description](#32-workloadpy--algorithm-description)
   - 3.3 [simulator.py — the bottleneck engine](#33-simulatorpy--the-bottleneck-engine)
   - 3.4 [analysis.py — reports & roofline](#34-analysispy--reports--roofline)
4. [Data-Flow Diagram](#4-data-flow-diagram)
5. [Key Connection: MMAOp → SMEM Cycles](#5-key-connection-mmaop--smem-cycles)
6. [Granularity Convention: per-SM vs chip-level](#6-granularity-convention-per-sm-vs-chip-level)
7. [YAML Hardware Configs](#7-yaml-hardware-configs)
8. [Reproducing FA-4 Table 1](#8-reproducing-fa-4-table-1)
9. [Extending the Simulator](#9-extending-the-simulator)

---

## 1. Project Layout

```
SimpleSim/
├── simplesim/                      # Python package — the simulator library
│   ├── __init__.py                 # Public API exports
│   ├── hardware.py                 # GPUConfig, ComputeUnit, MemoryLevel, YAML loader
│   ├── workload.py                 # Workload (basic), MMAOp + TiledWorkload (MMA-aware)
│   ├── simulator.py                # CycleSimulator: simulate() and simulate_tiled()
│   ├── analysis.py                 # print_report(), roofline_point(), print_roofline()
│   ├── pipeline.py                 # Stage / Pipeline data model
│   ├── timeline_sim.py             # Dependency-aware timeline simulation
│   ├── resource_scheduler.py       # Resource-constrained scheduling helpers
│   └── viz.py                      # Optional matplotlib timeline/utilization plots
│
├── configs/                        # Hardware YAML files (one per GPU SKU)
│   ├── h100.yaml                   # NVIDIA H100 SXM (Hopper, 132 SMs @ 1.83 GHz)
│   └── b200.yaml                   # NVIDIA B200 SXM (Blackwell, 148 SMs @ 1.85 GHz)
│
├── examples/
│   ├── flash_attention.py          # FA-4 forward/backward validation against Table 1/3
│   ├── fa4_pipeline.py             # End-to-end forward/backward pipeline demo
│   ├── fa4_tiled_pipeline.py       # KV-tile pipeline demo
│   ├── fa4_fine_grained_pipeline.py# Fine-grained stage overlap demo
│   ├── fa4_resource_scheduled.py   # Automatic resource scheduling demo
│   ├── fa4_forward_breakdown.py    # Single-tile forward stage breakdown
│   └── mla_decode.py               # MLA decode analysis and MHA comparison
│
├── tests/
│   └── test_examples_and_regressions.py  # Import safety + FA-4 regression checks
├── pyproject.toml                  # Packaging metadata and optional viz dependency
├── requirements.txt                # Minimal runtime dependencies
├── plan.md                         # Original design plan (reference)
└── ARCHITECTURE.md                 # This file
```

---

## 2. Design Philosophy

SimpleSim answers the question: *given a set of hardware throughput numbers and
an algorithm's compute/memory demands, which resource becomes the bottleneck and
how many cycles does it take?*

### The bottleneck ("Feeds and Speeds") model

Every hardware resource (tensor core, SFU, SMEM, HBM, …) is modelled as an
independent pipeline with a fixed throughput.  The total execution time is
determined by the *slowest* resource:

```
total_cycles = max(cycles_tc, cycles_smem, cycles_sfu, cycles_hbm, ...)
```

This is identical to the "Feeds and Speeds" analysis framework used in the
FlashAttention-4 paper (Zadouri et al., 2026).

### What SimpleSim adds beyond a plain Roofline

The classical Roofline model uses only two numbers: peak FLOPS and peak HBM
bandwidth.  It cannot capture the SMEM bottleneck that dominates FA-4.

SimpleSim models the full memory hierarchy per-SM, and—crucially—correctly
computes **how many times each SMEM operand stripe is re-read** when the output
tile is larger than the hardware MMA tile.  This is what `MMAOp.smem_bytes()`
implements.

---

## 3. Component Reference

### 3.1 `hardware.py` — the GPU model

**Purpose**: encode everything the simulator needs to know about the target GPU.

#### Data classes

| Class | Granularity | Key field |
|-------|-------------|-----------|
| `ComputeUnit` | per-SM per-cycle | `ops_per_cycle` |
| `MemoryLevel` | per-SM *or* chip-level | `bandwidth_bytes_per_cycle`, `is_per_sm` |
| `GPUConfig` | whole chip | `num_sms`, `clock_ghz`, dicts of the above |

`MemoryLevel.is_per_sm` is the flag that tells the simulator which
granularity to use:

```
is_per_sm = True   → shared_memory, L1 (each SM owns its copy of the tile)
is_per_sm = False  → HBM, L2       (shared across all SMs)
```

#### YAML loader

`load_gpu_config(path)` reads a YAML file and returns a `GPUConfig`.  The
`is_per_sm` flag is inferred automatically from the level name (see
`_parse_memory_level`): any level whose name is in `{"shared_memory",
"l1_cache", "rf", "tmem"}` is treated as per-SM.

---

### 3.2 `workload.py` — algorithm description

Two abstraction levels exist so that users can trade accuracy for simplicity.

#### `Workload` (basic mode)

```python
Workload(
    name="my_kernel",
    compute_ops={"tensor_core": 4_000_000, "sfu": 16_384},
    memory_bytes={"hbm": 2_097_152, "shared_memory": 98_304},
)
```

The caller pre-computes all totals.  The simulator just divides each number by
the corresponding hardware throughput and takes the max.

#### `TiledWorkload` + `MMAOp` (MMA-aware mode)

```
TiledWorkload
├── mma_ops: list[MMAOp]      ← one entry per GEMM in the kernel
├── elementwise_ops: dict     ← sfu / cuda_core totals
├── hbm_bytes: int            ← chip-level total I/O
└── extra_smem_bytes: int     ← non-MMA SMEM traffic (e.g. dS write-back)
```

Each `MMAOp` records:
- The *logical* output tile shape `(M, N)`.
- The reduction dimension `K`.
- The source of each operand (`"smem"`, `"tmem"`, or `"rmem"`).
- The element size in bytes.
- The *hardware* MMA tile size (default 128×128 for Blackwell).

`MMAOp.smem_bytes()` is the core formula — see Section 5.

#### Why two levels?

The basic `Workload` is useful when you already know the SMEM traffic (e.g.,
you are porting a hand-analysis from a paper).  The `TiledWorkload` is the
right choice when you want the simulator to *derive* SMEM traffic from the
tiling structure, especially when changing tile sizes (M, N, d) to explore the
design space.

---

### 3.3 `simulator.py` — the bottleneck engine

`CycleSimulator` owns a `GPUConfig` and exposes two methods:

#### `simulate(workload: Workload) → SimResult`

```
for each compute unit in workload.compute_ops:
    cycles = ceil(total_ops / ops_per_cycle)        # per-SM

for each memory level in workload.memory_bytes:
    cycles = ceil(total_bytes / bandwidth_per_cycle)  # per-SM or chip-level

total_cycles  = max(all cycles)
bottleneck    = units where cycles == total_cycles
utilization   = each unit's cycles / total_cycles
```

#### `simulate_tiled(workload: TiledWorkload) → SimResult`

Extends the basic flow with:
1. **Tensor-core cycles**: `ceil(workload.total_mma_ops() / tc.ops_per_cycle)`
2. **SMEM cycles**: `ceil(workload.total_smem_bytes() / smem_bw)` — where
   `total_smem_bytes()` sums `MMAOp.smem_bytes()` for every op (plus
   `extra_smem_bytes`).
3. **Elementwise cycles**: same as basic mode for `sfu`/`cuda_core`.
4. **HBM cycles**: `ceil(hbm_bytes / hbm_bw)` — chip-level bandwidth.

#### `SimResult`

```python
SimResult:
    compute_results : dict[str, UnitResult]   # tensor_core, sfu, cuda_core
    memory_results  : dict[str, UnitResult]   # shared_memory, hbm, l2_cache
    total_cycles    : int
    bottleneck_units: list[str]
    utilization     : dict[str, float]
    smem_breakdown  : list[SmemBreakdownEntry]  # per-MMAOp in mma_aware mode
```

`UnitResult` bundles the raw traffic, throughput, derived cycle count, wall
time, and the `is_bottleneck` flag in one place.

---

### 3.4 `analysis.py` — reports & roofline

#### `print_report(result)`

Prints a table with one row per hardware unit.  In MMA-aware mode, the
`shared_memory` row is expanded to show each `MMAOp`'s contribution and its
**amplification factor** (actual SMEM bytes ÷ naïve matrix-size bytes).

```
Resource                  Total Traffic   Throughput/cyc  Cycles  Time(µs)  Util%
tensor_core <<<           8.389 M ops     8192/cyc         1024    0.55     100.0%
sfu         <<<           16.4 K ops        16/cyc         1024    0.55     100.0%
cuda_core                 65.5 K ops       256/cyc          256    0.14      25.0%
shared_memory             98.3 K B         128/cyc          768    0.42      75.0%
  └─ QKt                  65.5 K B         amp=1.00x
  └─ PV                   32.8 K B         amp=1.00x
hbm                      131.1 K B        4324/cyc           31    0.02       3.0%
```

#### `roofline_point(result, hw)` + `print_roofline(rp)`

Computes the classical Roofline metrics:

```
AI           = total_FLOPs / hbm_bytes
ridge_point  = peak_FLOPS / peak_HBM_bandwidth
attainable   = min(peak_FLOPS, AI × peak_HBM_bandwidth)
is_compute_bound = AI >= ridge_point
```

---

## 4. Data-Flow Diagram

```
  ┌─────────────────────┐   load_gpu_config()   ┌──────────────┐
  │  configs/b200.yaml  │ ────────────────────► │  GPUConfig   │
  └─────────────────────┘                       │  ├ ComputeUnit│
                                                │  └ MemoryLevel│
                                                └──────┬───────┘
                                                       │
  ┌─────────────────────────────┐                      │
  │  TiledWorkload              │                      │
  │  ├ MMAOp("QKt", SS, M, N, d)│                      ▼
  │  ├ MMAOp("PV",  TS, M, d, N)│           ┌─────────────────────┐
  │  ├ elementwise_ops           │──────────►│  CycleSimulator     │
  │  └ hbm_bytes                │           │  .simulate_tiled()  │
  └─────────────────────────────┘           └────────┬────────────┘
                                                     │
                                          ┌──────────▼──────────┐
                                          │  SimResult           │
                                          │  ├ compute_results   │
                                          │  ├ memory_results    │
                                          │  ├ total_cycles      │
                                          │  ├ bottleneck_units  │
                                          │  └ smem_breakdown    │
                                          └──────────┬──────────┘
                                                     │
                          ┌──────────────────────────┤
                          │                          │
                 ┌────────▼────────┐       ┌─────────▼──────────┐
                 │  print_report() │       │  roofline_point()  │
                 │  (text table)   │       │  + print_roofline()│
                 └─────────────────┘       └────────────────────┘
```

---

## 5. Key Connection: MMAOp → SMEM Cycles

This is the most important analytical insight in SimpleSim, derived from the
FA-4 paper.

### The problem with naïve SMEM counting

For a QK^T GEMM of shape (M × N), inner dimension d, in BF16:
- Naïve estimate: `(M*d + d*N) * 2` bytes.
- For M=N=d=128: `(128*128 + 128*128) * 2 = 65 536` bytes.
- At 128 B/cycle: **512 cycles**.

But the paper says **768 cycles** for M=N=d=128.  Why?

### MMA tile tiling causes operand re-reads

A Blackwell MMA instruction processes one 128×128 *hardware tile* at a time.
If the output tile is (M, N), the hardware needs `ceil(M/128) × ceil(N/128)`
MMA instructions.  Each instruction re-reads its operand stripes:

- Operand A (M-row stripe × K cols): `hw_tile_m × K` elements per instruction.
- Operand B (K rows × N-col stripe): `K × hw_tile_n` elements per instruction.

Total reads = `ceil(M/128) × ceil(N/128) × (per_tile_A + per_tile_B)`.

For M=N=d=128 QK^T (SS mode):
```
num_tiles_m = 1,  num_tiles_n = 1
bytes = 1 × 1 × (128×128×2 + 128×128×2) = 65 536 bytes
```

For PV MMA of shape (M=128, d=128), K=N=128 (TS mode, A from TMEM):
```
num_tiles_m = 1,  num_tiles_n = 1
bytes = 1 × 1 × (0 + 128×128×2) = 32 768 bytes   (A is free from TMEM)
```

Total SMEM = 65 536 + 32 768 = **98 304 bytes** → 98 304 / 128 = **768 cycles**. ✓

For M=256 (doubling tile height):
```
QKt: num_tiles_m=2 → bytes = 2 × 65 536 = 131 072
PV:  num_tiles_m=2 → bytes = 2 × 32 768 =  65 536
Total = 196 608 bytes → 196 608 / 128 = 1536 cycles ✓
```

### Code path

```
TiledWorkload.total_smem_bytes()
  └── sum(MMAOp.smem_bytes() for op in mma_ops) + extra_smem_bytes
        └── MMAOp.smem_bytes():
              num_tiles_m = ceil(output_shape[0] / hw_tile_m)
              num_tiles_n = ceil(output_shape[1] / hw_tile_n)
              per_tile_bytes = 0
              if operand_a_source == "smem":
                  per_tile_bytes += hw_tile_m * K * dtype_bytes
              if operand_b_source == "smem":
                  per_tile_bytes += K * hw_tile_n * dtype_bytes
              return num_tiles_m * num_tiles_n * per_tile_bytes
```

`CycleSimulator.simulate_tiled()` then computes:
```
smem_cycles = ceil(total_smem_bytes / smem_bw_per_cycle)
```

---

## 6. Granularity Convention: per-SM vs chip-level

A common source of confusion in GPU performance modelling is whether
throughputs are per-SM or whole-chip.

SimpleSim uses **per-SM granularity for all SM-local resources** and
**chip-level granularity for shared memory subsystems**:

| Resource | Granularity | Rationale |
|----------|-------------|-----------|
| `tensor_core` | per SM | Each SM runs its tile independently |
| `cuda_core` | per SM | Each SM runs its tile independently |
| `sfu` | per SM | Each SM runs its tile independently |
| `shared_memory` | per SM | Each SM has its own SRAM bank |
| `hbm` | chip (shared) | Single HBM bus shared by all SMs |
| `l2_cache` | chip (shared) | Single L2 shared by all SMs |

In the YAML config, `bandwidth_bytes_per_cycle` for HBM/L2 is the *total*
chip bandwidth divided by the clock frequency.

The simulator uses `MemoryLevel.is_per_sm` to select the right formula.

---

## 7. YAML Hardware Configs

Both files follow the same schema:

```yaml
name: "B200 SXM"
architecture: "Blackwell"
num_sms: 148
clock_ghz: 1.85

compute_units:
  tensor_core:
    ops_per_cycle: 8192       # FP16 FLOPS per SM per cycle
    supported_dtypes: [fp16, bf16, fp8, fp4, tf32]
  cuda_core:
    ops_per_cycle: 256
    supported_dtypes: [fp32, fp16, int32]
  sfu:
    ops_per_cycle: 16
    supported_dtypes: [fp32]

memory_levels:
  shared_memory:
    bandwidth_bytes_per_cycle: 128   # per-SM (inferred from is_per_sm)
    capacity_bytes: 262144
  hbm:
    bandwidth_bytes_per_cycle: 4324  # chip-level
    capacity_bytes: 214748364800
```

### Deriving B200 values from FA-4 Table 1

The FA-4 paper gives closed-form cycle-count formulae for the forward pass:

| Formula | Source | Derived config value |
|---------|--------|----------------------|
| `T_mma  = 4MNd / 8192` | Tensor core throughput | `tensor_core.ops_per_cycle = 8192` |
| `T_smem = 3MNd / 8192` | SMEM MMA feed rate | `shared_memory.bw = 128 B/cycle` (derived in §5) |
| `T_exp  = MN / 16`     | SFU exp throughput | `sfu.ops_per_cycle = 16` |

---

## 8. Reproducing FA-4 Table 1

Run the example:

```
python -m examples.flash_attention
```

Run the automated regression tests:

```
python -m unittest discover -s tests -v
```

Expected output (Table 1 portion):

```
[PASS] M=128, N=128, d=128
  MMA  : got= 1024  expected= 1024  ✓
  SMEM : got=  768  expected=  768  ✓
  SFU  : got= 1024  expected= 1024  ✓

[PASS] M=256, N=128, d=128
  MMA  : got= 2048  expected= 2048  ✓
  SMEM : got= 1536  expected= 1536  ✓
  SFU  : got= 2048  expected= 2048  ✓
```

The simulation confirms:
- Doubling M doubles both MMA and SFU cycles (linear in M).
- SMEM cycles also double because `QKt.num_tiles_m` goes from 1 → 2, causing Q and K stripes to be re-read twice.
- The joint MMA + SFU bottleneck means SMEM at 75% utilisation is *not* the bottleneck for the forward pass.

---

## 9. Extending the Simulator

### Adding a new GPU

1. Copy `configs/h100.yaml` to e.g. `configs/h200.yaml`.
2. Fill in `num_sms`, `clock_ghz`, and the throughput values.
3. Load with `load_gpu_config("configs/h200.yaml")`.

### Adding a new workload

For MMA-aware analysis, define a `TiledWorkload` with one `MMAOp` per GEMM:

```python
my_wl = TiledWorkload(
    name="my_kernel",
    mma_ops=[
        MMAOp("GEMM1", output_shape=(M, N), reduction_dim=K,
              operand_a_source="smem", operand_b_source="smem",
              dtype_bytes=2),
    ],
    elementwise_ops={"sfu": M * N},
    hbm_bytes=...,
)
```

### Modelling 2-CTA MMA (Blackwell)

In 2-CTA mode, two CTAs collaborate on one MMA, halving the SMEM traffic per
CTA.  Model this by setting `hw_tile_m=256` (or `hw_tile_n=256`) in `MMAOp`
— the `smem_bytes()` formula then uses the wider hardware tile, reducing
`num_tiles_m` by half.

### Adding bank-conflict or wave-quantisation effects

Subclass `CycleSimulator` and override `simulate_tiled` to apply multipliers
before the cycle-count step:

```python
class DetailedSimulator(CycleSimulator):
    def simulate_tiled(self, wl):
        result = super().simulate_tiled(wl)
        # Apply 10% bank-conflict penalty to SMEM
        r = result.memory_results.get("shared_memory")
        if r:
            r.cycles = int(r.cycles * 1.10)
        return self._rebuild(result)
```
