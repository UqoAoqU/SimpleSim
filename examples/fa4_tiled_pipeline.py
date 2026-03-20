"""
examples/fa4_tiled_pipeline.py
==============================
Demonstrates Stage.repeat with hardware-aware pipelining.

Models the FA-4 forward pass outer KV-tile loop:
  • Each iteration loads a K+V tile from HBM (async copy / DMA)
  • Then runs QK^T → Softmax → Rescale O → PV
  • After all tiles: Update O (once)

Pipelining model
----------------
  HBM loads   →  shared / overlappable  (async DMA, independent of shader)
  tensor_core →  exclusive             (one tile at a time)
  sfu         →  exclusive
  cuda_core   →  exclusive
  shared_memory → overlappable         (SMEM capacity allows double-buffer)

Expected overlap pattern:
  tensor_core:    [0, 512] [512,1024] [1024,1536] ...  (back-to-back)
  hbm:            [0, 15]  [15, 30]   [30,  45]  ...  (≈15 cyc/tile, fully hidden)

  → HBM latency is completely hidden behind compute; effective throughput
    ≈ 512 cycles/tile (tensor-core bound).

Run
---
    cd SimpleSim
    MPLBACKEND=Agg python -m examples.fa4_tiled_pipeline
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simplesim import (
    load_gpu_config,
    MMAOp, TiledWorkload, Workload,
    Stage, Pipeline,
    TimelineSimulator,
    plot_timeline, plot_utilization,
    print_report,
)

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
b200 = load_gpu_config(os.path.join(CONFIG_DIR, "b200.yaml"))

# ---------------------------------------------------------------------------
# Workload parameters
# ---------------------------------------------------------------------------
M          = 128   # query tile rows (per SM)
N_tile     = 128   # KV tile columns
d          = 128   # head dimension
seq_kv     = 1024  # total KV sequence length
n_tiles    = seq_kv // N_tile   # = 8  (number of KV-tile loop iterations)
dtype_bytes = 2    # BF16


# ---------------------------------------------------------------------------
# Per-tile combined workload
# ---------------------------------------------------------------------------
# Each iteration of the KV-tile loop does:
#   1. Load K_tile + V_tile from HBM into SMEM  (overlappable DMA)
#   2. QK^T  (SS MMA: Q(M,d) × K^T(d,N))
#   3. Softmax (exp of M×N scores)
#   4. Rescale O (online softmax accumulator update)
#   5. PV (TS MMA: P(M,N) × V(N,d))

# HBM bytes per tile: K_tile + V_tile = 2 × N_tile × d × dtype
hbm_per_tile = 2 * N_tile * d * dtype_bytes   # = 65 536 B  → ~15 HBM cycles on B200

kv_tile_wl = TiledWorkload(
    name="kv_tile",
    mma_ops=[
        # QK^T: output (M,N), K=d, both from SMEM → SS MMA
        MMAOp("QKt", output_shape=(M, N_tile), reduction_dim=d,
              operand_a_source="smem", operand_b_source="smem",
              dtype_bytes=dtype_bytes),
        # PV: output (M,d), K=N, P from TMEM, V from SMEM → TS MMA
        MMAOp("PV",  output_shape=(M, d), reduction_dim=N_tile,
              operand_a_source="tmem", operand_b_source="smem",
              dtype_bytes=dtype_bytes),
    ],
    elementwise_ops={
        # Softmax: exp (SFU) + fmax/fsub/fadd (cuda_core)
        "sfu":       M * N_tile,          # exp of every score
        "cuda_core": 3 * M * N_tile       # fmax + fsub + fadd
                   + M * d + 2 * M,       # O*=α + l update (rescale)
    },
    hbm_bytes=hbm_per_tile,
)

# Final normalisation (once, after all tiles)
update_o_wl = Workload(
    name="update_o",
    compute_ops={
        "sfu":       M,       # rcp.approx(l)
        "cuda_core": M * d,   # O *= rcp(l)
    },
)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
pipe = Pipeline(f"FA-4 fwd (seq_kv={seq_kv}, {n_tiles} tiles)", stages=[
    Stage("kv_tile", workload=kv_tile_wl, repeat=n_tiles),
    Stage("update_o", workload=update_o_wl, depends_on=["kv_tile"]),
])

# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------
sim = TimelineSimulator(b200)
tl  = sim.simulate(pipe)

# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------
print()
print("╔══════════════════════════════════════════════════════════════╗")
print("║  FA-4 KV-Tile Pipeline — Pipelining Demo (B200)              ║")
print("╚══════════════════════════════════════════════════════════════╝")
print(f"  seq_kv={seq_kv}, n_tiles={n_tiles}, M=N={N_tile}, d={d}, BF16")
print(f"  HBM per tile : {hbm_per_tile:,} B")
print()

# Per-tile single analysis for comparison
single_res = sim._inner.simulate_tiled(kv_tile_wl)
hbm_ur  = single_res.memory_results.get("hbm")
tc_ur   = single_res.compute_results.get("tensor_core")
hbm_cyc = hbm_ur.cycles  if hbm_ur  else 0
tc_cyc  = tc_ur.cycles   if tc_ur   else 0

print(f"  Per-tile bottleneck    : {single_res.total_cycles} cyc "
      f"({'+'.join(single_res.bottleneck_units)})")
print(f"  Per-tile HBM           : {hbm_cyc} cyc  "
      f"({'FULLY hidden by compute' if hbm_cyc <= tc_cyc else 'visible'})")

# Serial model: each tile runs fully before the next, including HBM
serial_per_tile = single_res.total_cycles  # bottleneck already accounts for HBM
serial_tiles_total = serial_per_tile * n_tiles

# Pipelined tiles total: compute exclusive → n_tiles × bottleneck
# HBM pipelined → adds only max(hbm_cyc, 0) for the very first iter
# (subsequent HBM runs overlap with previous compute)
tile_scheds = [s for s in tl.schedules if s.stage_name.startswith("kv_tile")]
pipelined_tiles_end = tile_scheds[-1].end_cycle
pipeline_overhead = max(0, hbm_cyc - tc_cyc)  # extra cycles if HBM > TC

print(f"  Serial total  ({n_tiles} tiles): {serial_tiles_total:,} cyc")
print(f"  Pipelined tiles end    : {pipelined_tiles_end:,} cyc  "
      f"({b200.cycles_to_us(pipelined_tiles_end):.3f} µs)")
if pipeline_overhead == 0:
    print(f"  HBM latency hidden     : fully (HBM {hbm_cyc} cyc ≤ TC {tc_cyc} cyc)")
else:
    print(f"  Pipeline overhead      : +{pipeline_overhead} cyc/iter (HBM > TC)")
print()

# First-few-iterations table
print("  Iteration schedule (first 4 + last):")
col = (16, 10, 10, 36)
hdr = ("Stage", "Start", "End", "Unit intervals (cycles)")
sep = "  ".join("-"*c for c in col)
print("  " + "  ".join(h.ljust(c) for h, c in zip(hdr, col)))
print("  " + sep)
shown = [s for s in tl.schedules if s.iteration < 4 or s.iteration == n_tiles - 1]
for s in shown:
    units_str = "  ".join(
        f"{u}:{iv.unit_cycles}" for u, iv in sorted(s.unit_intervals.items())
    )
    if s.iteration == n_tiles - 1 and n_tiles > 5:
        print("  ...")
    print("  " + "  ".join([
        s.stage_name.ljust(col[0]),
        str(s.start_cycle).ljust(col[1]),
        str(s.end_cycle).ljust(col[2]),
        units_str,
    ]))
print()

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(__file__)

fig_tl = plot_timeline(
    tl,
    time_unit="cycles",
    max_preview_iters=n_tiles,   # show all 8 iterations
    title=f"FA-4 KV-Tile Pipeline (×{n_tiles} tiles, seq_kv={seq_kv}) — B200",
    figsize=(16, 5),
)
tl_path = os.path.join(out_dir, "fa4_tiled_pipeline_timeline.png")
fig_tl.savefig(tl_path, dpi=150, bbox_inches="tight")
print(f"  Timeline chart  → {tl_path}")

fig_ut = plot_utilization(
    tl,
    title=f"FA-4 KV-Tile Pipeline Utilization (×{n_tiles} tiles) — B200",
)
ut_path = os.path.join(out_dir, "fa4_tiled_pipeline_utilization.png")
fig_ut.savefig(ut_path, dpi=150, bbox_inches="tight")
print(f"  Utilization     → {ut_path}")
print()
