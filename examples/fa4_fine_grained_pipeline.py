"""
examples/fa4_fine_grained_pipeline.py
======================================
FA-4 forward pass: fine-grained sub-stage pipelining across KV-tile iterations.

Each KV-tile iteration is broken into 5 separate pipeline stages:

  hbm_load[i]  — async DMA: load K_i + V_i tiles from HBM
  QKT[i]       — S = Q × K_i^T  (SS-MMA, tensor_core + shared_memory)
  Softmax[i]   — P = softmax(S), update m/l  (SFU + cuda_core)
  RescaleO[i]  — O *= exp(m_old - m_new)  (cuda_core)
  PV[i]        — O += P × V_i  (TS-MMA, tensor_core)

Cross-iteration dependencies that create pipelining
----------------------------------------------------
  hbm_load[i] → hbm_load[i-1]          (HBM is exclusive: serial DMA)
  QKT[i]      → hbm_load[i]            (need tile data)
               + PV[i-1]               (tensor_core must be free)
  Softmax[i]  → QKT[i]
  RescaleO[i] → Softmax[i]
  PV[i]       → RescaleO[i]
  update_o    → PV[n_tiles-1]

Expected behaviour (M=N=d=128, seq_kv=1024, 8 tiles, B200)
-----------------------------------------------------------
  HBM per tile  :   16 cycles   (fully hidden, << 2178-cycle compute)
  Compute/tile  : 2178 cycles   (512 QKT + 1024 Softmax + 130 Rescale + 512 PV)
  Total (8+1)   : 8*2178 + 128 + 16 ≈ 17568 cycles

Run
---
    cd SimpleSim
    MPLBACKEND=Agg python -m examples.fa4_fine_grained_pipeline
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
n_tiles    = seq_kv // N_tile   # = 8
dtype_bytes = 2    # BF16

hbm_per_tile = 2 * N_tile * d * dtype_bytes   # K_tile + V_tile = 65 536 B

# ---------------------------------------------------------------------------
# Workload objects (shared across all tile iterations)
# ---------------------------------------------------------------------------

# HBM async load: K_i + V_i from HBM → SMEM (no compute)
hbm_load_wl = Workload(
    name="hbm_load",
    memory_bytes={"hbm": hbm_per_tile},
)

# QKT: S(M,N) = Q(M,d) × K^T(d,N)  — both operands from SMEM (SS-MMA)
qkt_wl = TiledWorkload(
    name="QKT",
    mma_ops=[MMAOp(
        name="QKt",
        output_shape=(M, N_tile),
        reduction_dim=d,
        operand_a_source="smem",
        operand_b_source="smem",
        dtype_bytes=dtype_bytes,
    )],
)

# Softmax: exp of every score (SFU) + fmax/fsub/fadd (cuda_core)
softmax_wl = Workload(
    name="Softmax",
    compute_ops={
        "sfu":       M * N_tile,
        "cuda_core": 3 * M * N_tile,
    },
)

# Rescale O: exp(m_old - m_new) per row (SFU) + O*=α, l update (cuda_core)
rescale_o_wl = Workload(
    name="RescaleO",
    compute_ops={
        "sfu":       M,
        "cuda_core": M * d + 2 * M,
    },
)

# PV: O(M,d) += P(M,N) × V(N,d)  — P in TMEM (free), V from SMEM (TS-MMA)
pv_wl = TiledWorkload(
    name="PV",
    mma_ops=[MMAOp(
        name="PV",
        output_shape=(M, d),
        reduction_dim=N_tile,
        operand_a_source="tmem",   # P in TMEM — no SMEM traffic
        operand_b_source="smem",   # V from SMEM
        dtype_bytes=dtype_bytes,
    )],
)

# Update O: final normalisation (once, after all tiles)
update_o_wl = Workload(
    name="update_o",
    compute_ops={
        "sfu":       M,
        "cuda_core": M * d,
    },
)

# ---------------------------------------------------------------------------
# Build fine-grained pipeline  (QKTs back-to-back, Softmax overlaps next QKT)
# ---------------------------------------------------------------------------
#
# Key insight: QKT uses tensor_core + shared_memory; Softmax/RescaleO use
# SFU + cuda_core.  These are *different* hardware units, so QKT[i+1] can
# run in parallel with Softmax[i] — they only need to wait for each other's
# *own* unit to be free.
#
# Dependency rules:
#   hbm_load[i]  → hbm_load[i-1]          (HBM DMA serialized)
#   QKT[i]       → QKT[i-1]  + hbm_load[i]  (TC serialized; need K data)
#   Softmax[i]   → QKT[i]                  (need scores from QKT)
#   RescaleO[i]  → Softmax[i]
#   PV[0]        → RescaleO[0] + QKT[last] (TC free after all QKTs)
#   PV[i>0]      → RescaleO[i] + PV[i-1]  (TC serialized; need P[i])
#   update_o     → PV[last]
#
# Result: QKT[0] QKT[1] ... QKT[N-1] run back-to-back on tensor_core.
#         Softmax[i] runs in parallel with QKT[i+1] on SFU.
#         PV[0..N-1] run back-to-back on tensor_core after all QKTs finish.

stages: list[Stage] = []

for i in range(n_tiles):
    # HBM load: serialized (only one DMA engine), but async w.r.t. compute.
    # hbm_load[i+1] starts right after hbm_load[i] finishes (16 cyc each),
    # so all 8 loads complete in 128 cycles — fully hidden inside tile 0's
    # 2178-cycle compute window.
    stages.append(Stage(
        name=f"hbm_load[{i}]",
        workload=hbm_load_wl,
        depends_on=[f"hbm_load[{i-1}]"] if i > 0 else [],
    ))

    # QKT[i]: algorithmic dep on hbm_load[i] (K[i] in SMEM) AND PV[i-1]
    # (TC free, AND SMEM double-buffer slot freed so K[i]+V[i] can reside).
    # SMEM check: at QKT[i], active K[i]+V[i] (64KB) + pre-load K[i+1]+V[i+1]
    # (64KB) = 128KB << 256KB.  Only one V tile stays in SMEM at a time.
    qkt_deps = [f"hbm_load[{i}]"]
    if i > 0:
        qkt_deps.append(f"PV[{i-1}]")   # TC free + SMEM buffer freed by PV
    stages.append(Stage(
        name=f"QKT[{i}]",
        workload=qkt_wl,
        depends_on=qkt_deps,
    ))

    # Softmax[i]: needs S[i] scores from QKT[i].  SFU/CUDA are naturally
    # free because Softmax[i-1] ran inside the previous tile's pipeline.
    stages.append(Stage(
        name=f"Softmax[{i}]",
        workload=softmax_wl,
        depends_on=[f"QKT[{i}]"],
    ))

    # RescaleO[i]: needs m_new/l from Softmax[i].
    stages.append(Stage(
        name=f"RescaleO[{i}]",
        workload=rescale_o_wl,
        depends_on=[f"Softmax[{i}]"],
    ))

    # PV[i]: needs P[i] (from RescaleO[i]) and V[i] in SMEM.
    # TC is free: last TC user was QKT[i], which ended before Softmax[i].
    stages.append(Stage(
        name=f"PV[{i}]",
        workload=pv_wl,
        depends_on=[f"RescaleO[{i}]"],
    ))

stages.append(Stage(
    name="update_o",
    workload=update_o_wl,
    depends_on=[f"PV[{n_tiles - 1}]"],
))

pipe = Pipeline(
    f"FA-4 fine-grained (seq_kv={seq_kv}, {n_tiles} tiles)",
    stages=stages,
)

# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------
sim = TimelineSimulator(b200)
tl  = sim.simulate(pipe)

# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------
print()
print("╔══════════════════════════════════════════════════════════════════╗")
print("║  FA-4 Fine-Grained KV-Tile Pipelining (B200)                     ║")
print("╚══════════════════════════════════════════════════════════════════╝")
print(f"  seq_kv={seq_kv}, n_tiles={n_tiles}, M=N={N_tile}, d={d}, BF16")
print(f"  HBM per tile : {hbm_per_tile:,} B  ({b200.cycles_to_us.__func__ and ''})")
print()

# Print schedule for first 2 tiles + last + update_o
print("  Per-stage schedule (tile 0, 1, ..., last):")
col = (14, 8, 8, 8, 44)
hdr = ("Stage", "Start", "End", "Dur", "Unit intervals")
sep = "  ".join("-" * c for c in col)
print("  " + "  ".join(h.ljust(c) for h, c in zip(hdr, col)))
print("  " + sep)

shown_tiles = {0, 1, n_tiles - 1}
prev_tile = -1
for s in tl.schedules:
    # Extract tile index from name
    tile_idx = None
    for stage_prefix in ("hbm_load[", "QKT[", "Softmax[", "RescaleO[", "PV["):
        if s.stage_name.startswith(stage_prefix):
            tile_idx = int(s.stage_name.split("[")[1].rstrip("]"))
            break

    if tile_idx is not None and tile_idx not in shown_tiles:
        if tile_idx == 2 and prev_tile in shown_tiles:
            print("  ...")
        prev_tile = tile_idx
        continue
    prev_tile = tile_idx if tile_idx is not None else -1

    units_str = "  ".join(
        f"{u}:[{iv.start_cycle},{iv.end_cycle}]"
        for u, iv in sorted(s.unit_intervals.items())
    )
    print("  " + "  ".join([
        s.stage_name.ljust(col[0]),
        str(s.start_cycle).ljust(col[1]),
        str(s.end_cycle).ljust(col[2]),
        str(s.stage_cycles).ljust(col[3]),
        units_str,
    ]))

print()
print(f"  Total: {tl.total_cycles:,} cycles  /  {tl.total_time_us:.3f} µs")

# HBM hiding analysis
hbm_sched   = next(s for s in tl.schedules if s.stage_name == "hbm_load[0]")
qkt0_sched  = next(s for s in tl.schedules if s.stage_name == "QKT[0]")
pv0_sched   = next(s for s in tl.schedules if s.stage_name == "PV[0]")
hbm1_sched  = next(s for s in tl.schedules if s.stage_name == "hbm_load[1]")
qkt1_sched  = next(s for s in tl.schedules if s.stage_name == "QKT[1]")

hbm_cyc  = hbm_sched.stage_cycles
idle_gap = qkt1_sched.start_cycle - hbm1_sched.end_cycle
print(f"  HBM load per tile  : {hbm_cyc} cycles")
print(f"  hbm_load[1] done   : cycle {hbm1_sched.end_cycle}")
print(f"  QKT[1] start       : cycle {qkt1_sched.start_cycle}  "
      f"(waits for PV[0].end={pv0_sched.end_cycle})")
print(f"  HBM overlap saving : {hbm_cyc} cyc/tile × {n_tiles} tiles "
      f"= {hbm_cyc * n_tiles} cycles hidden")
print()

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(__file__)

fig_tl = plot_timeline(
    tl,
    time_unit="cycles",
    max_preview_iters=n_tiles,
    title=f"FA-4 Fine-Grained Pipeline (×{n_tiles} tiles, seq_kv={seq_kv}) — B200",
    figsize=(20, 6),
)
tl_path = os.path.join(out_dir, "fa4_fine_grained_timeline.png")
fig_tl.savefig(tl_path, dpi=150, bbox_inches="tight")
print(f"  Timeline chart  → {tl_path}")

fig_ut = plot_utilization(
    tl,
    title=f"FA-4 Fine-Grained Utilization (×{n_tiles} tiles) — B200",
)
ut_path = os.path.join(out_dir, "fa4_fine_grained_utilization.png")
fig_ut.savefig(ut_path, dpi=150, bbox_inches="tight")
print(f"  Utilization     → {ut_path}")
print()
