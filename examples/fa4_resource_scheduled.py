"""
examples/fa4_resource_scheduled.py
===================================
FA-4 forward pass with **automatic** resource-constrained scheduling
and **per-tensor HBM loading**.

Key improvements over ``fa4_fine_grained_pipeline.py``:

1. **Data dependencies only** — resource conflicts (tensor_core exclusion,
   SMEM capacity) are detected automatically by ``ResourceScheduler``.

2. **Per-tensor HBM loads** — each tensor (K, V, Q) is loaded as a separate
   stage via ``DataTensor.load_workload()``.  This enables finer overlap:
   QKT[i] can start as soon as K[i] arrives, while V[i] loads in parallel.

Run
---
    cd SimpleSim
    MPLBACKEND=Agg python -m examples.fa4_resource_scheduled
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simplesim import (
    load_gpu_config,
    DataTensor, MMAOp, TiledWorkload, Workload,
    Stage, Pipeline,
    ResourceScheduler,
    plot_timeline, plot_timeline_with_smem, plot_utilization,
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
dtype      = "bf16"
dtype_bytes = 2

smem_per_kv = N_tile * d * dtype_bytes   # one K or V tile = 32 KB

# ---------------------------------------------------------------------------
# DataTensors — define once, reuse for loads and compute stages
# ---------------------------------------------------------------------------
# Q is loaded once and stays in SMEM for all tiles
Q_tensor = DataTensor("Q", (M, d), dtype, "hbm")

# Per-tile K and V tensors (in HBM, will be loaded individually)
def K_hbm(i: int) -> DataTensor:
    return DataTensor(f"K_{i}", (N_tile, d), dtype, "hbm")

def V_hbm(i: int) -> DataTensor:
    return DataTensor(f"V_{i}", (N_tile, d), dtype, "hbm")

# ---------------------------------------------------------------------------
# Compute workload factories
# ---------------------------------------------------------------------------

def make_qkt(i: int) -> TiledWorkload:
    """QKT: S(M,N) = Q(M,d) × K^T(d,N)  — both from SMEM (SS-MMA)."""
    return TiledWorkload(
        name="QKT",
        mma_ops=[MMAOp(
            name="QKt",
            output_shape=(M, N_tile),
            reduction_dim=d,
            operand_a_source="smem",
            operand_b_source="smem",
            dtype_bytes=dtype_bytes,
        )],
        inputs=[
            DataTensor("Q", (M, d), dtype, "smem"),
            DataTensor(f"K_{i}", (N_tile, d), dtype, "smem"),
        ],
        outputs=[
            DataTensor(f"S_{i}", (M, N_tile), dtype, "smem"),
        ],
        # Q (32KB) + K_tile (32KB) + S output (32KB)
        smem_capacity_bytes=M * d * dtype_bytes + smem_per_kv + M * N_tile * dtype_bytes,
    )


def make_softmax(i: int) -> Workload:
    """Softmax: exp(S) → P, update running max/sum."""
    return Workload(
        name="Softmax",
        compute_ops={
            "sfu":       M * N_tile,
            "cuda_core": 3 * M * N_tile,
        },
        inputs=[DataTensor(f"S_{i}", (M, N_tile), dtype, "smem")],
        outputs=[DataTensor(f"P_{i}", (M, N_tile), dtype, "tmem")],
        smem_capacity_bytes=M * N_tile * dtype_bytes,
    )


def make_rescale(i: int) -> Workload:
    """RescaleO: O *= exp(m_old - m_new)."""
    return Workload(
        name="RescaleO",
        compute_ops={
            "sfu":       M,
            "cuda_core": M * d + 2 * M,
        },
        smem_capacity_bytes=0,
    )


def make_pv(i: int) -> TiledWorkload:
    """PV: O += P × V  — P from TMEM (free), V from SMEM (TS-MMA)."""
    return TiledWorkload(
        name="PV",
        mma_ops=[MMAOp(
            name="PV",
            output_shape=(M, d),
            reduction_dim=N_tile,
            operand_a_source="tmem",
            operand_b_source="smem",
            dtype_bytes=dtype_bytes,
        )],
        inputs=[
            DataTensor(f"P_{i}", (M, N_tile), dtype, "tmem"),
            DataTensor(f"V_{i}", (N_tile, d), dtype, "smem"),
        ],
        outputs=[
            DataTensor("O_partial", (M, d), dtype, "tmem"),
        ],
        smem_capacity_bytes=smem_per_kv,
    )


update_o_wl = Workload(
    name="update_o",
    compute_ops={
        "sfu":       M,
        "cuda_core": M * d,
    },
)

# ---------------------------------------------------------------------------
# Build pipeline with per-tensor loads and DATA dependencies only
# ---------------------------------------------------------------------------
# Per-tensor loading enables:
#   - QKT[i] starts as soon as K[i] is loaded (no need to wait for V[i])
#   - load_V[i] overlaps with QKT[i] compute
#   - PV[i] depends on both RescaleO[i] (data) and load_V[i] (data)

stages: list[Stage] = []

# Load Q once at the start
stages.append(Stage(
    name="load_Q",
    workload=Q_tensor.load_workload(),
))

for i in range(n_tiles):
    # --- Per-tensor HBM loads ---
    # K[i]: serial w.r.t. previous DMA (HBM exclusive), parallel with compute
    load_k_deps = ["load_Q"] if i == 0 else [f"load_V[{i-1}]"]
    stages.append(Stage(
        name=f"load_K[{i}]",
        workload=K_hbm(i).load_workload(),
        depends_on=load_k_deps,
    ))

    # V[i]: serial DMA after K[i]
    stages.append(Stage(
        name=f"load_V[{i}]",
        workload=V_hbm(i).load_workload(),
        depends_on=[f"load_K[{i}]"],
    ))

    # --- Compute stages ---
    # QKT[i]: needs Q (always in SMEM) + K[i] loaded
    #         does NOT need V[i] → V load overlaps with QKT compute
    stages.append(Stage(
        name=f"QKT[{i}]",
        workload=make_qkt(i),
        depends_on=[f"load_K[{i}]"],
    ))

    stages.append(Stage(
        name=f"Softmax[{i}]",
        workload=make_softmax(i),
        depends_on=[f"QKT[{i}]"],
    ))

    stages.append(Stage(
        name=f"RescaleO[{i}]",
        workload=make_rescale(i),
        depends_on=[f"Softmax[{i}]"],
    ))

    # PV[i]: needs P[i] (from RescaleO) AND V[i] loaded
    stages.append(Stage(
        name=f"PV[{i}]",
        workload=make_pv(i),
        depends_on=[f"RescaleO[{i}]", f"load_V[{i}]"],
    ))

stages.append(Stage(
    name="update_o",
    workload=update_o_wl,
    depends_on=[f"PV[{n_tiles - 1}]"],
))

pipe = Pipeline(
    f"FA-4 auto-scheduled (seq_kv={seq_kv}, {n_tiles} tiles)",
    stages=stages,
)

# ---------------------------------------------------------------------------
# Schedule with automatic resource constraints
# ---------------------------------------------------------------------------
scheduler = ResourceScheduler(b200)
tl = scheduler.schedule(pipe)

# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("  FA-4 Resource-Constrained Auto-Scheduling (B200)")
print("  Per-tensor HBM loading + automatic overlap detection")
print("=" * 70)
print(f"  seq_kv={seq_kv}, n_tiles={n_tiles}, M=N={N_tile}, d={d}, BF16")
print(f"  HBM bw/SM = {b200.hbm_bytes_per_cycle():.1f} B/cycle")
print(f"  Per tensor load = {smem_per_kv:,} B / {b200.hbm_bytes_per_cycle():.1f} "
      f"= {smem_per_kv / b200.hbm_bytes_per_cycle():.0f} cycles")
print()

# Resource profiles
print("  Resource profiles (auto-detected):")
profiles = scheduler.get_profiles(pipe)
shown = ["load_Q", "load_K[0]", "load_V[0]", "QKT[0]", "Softmax[0]",
         "RescaleO[0]", "PV[0]"]
for name in shown:
    prof = profiles[name]
    cu_str = ", ".join(sorted(prof.compute_units)) or "(none)"
    print(f"    {name:16s}  compute: {cu_str:24s}  "
          f"smem: {prof.smem_capacity_bytes:>7,} B  "
          f"hbm: {'yes' if prof.uses_hbm else 'no'}")
print()

# Schedule table
print("  Per-stage schedule (first 2 tiles + last):")
col = (16, 8, 8, 8, 55)
hdr = ("Stage", "Start", "End", "Dur", "Unit intervals")
sep = "  ".join("-" * c for c in col)
print("  " + "  ".join(h.ljust(c) for h, c in zip(hdr, col)))
print("  " + sep)

shown_tiles = {0, 1, n_tiles - 1}
prev_tile = -1
for s in tl.schedules:
    # Show load_Q always
    if s.stage_name == "load_Q":
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
        continue

    tile_idx = None
    for prefix in ("load_K[", "load_V[", "QKT[", "Softmax[", "RescaleO[", "PV["):
        if s.stage_name.startswith(prefix):
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
print(f"  Total: {tl.total_cycles:,} cycles  /  {tl.total_time_us:.3f} us")
print()

# Overlap analysis
print("  Key overlaps:")
for i in range(min(2, n_tiles)):
    load_v = next(s for s in tl.schedules if s.stage_name == f"load_V[{i}]")
    qkt_i  = next(s for s in tl.schedules if s.stage_name == f"QKT[{i}]")
    overlap = min(load_v.end_cycle, qkt_i.end_cycle) - max(load_v.start_cycle, qkt_i.start_cycle)
    if overlap > 0:
        print(f"    load_V[{i}] overlaps QKT[{i}]: {overlap} cycles "
              f"(HBM DMA || tensor_core)")
    else:
        print(f"    load_V[{i}] and QKT[{i}]: sequential "
              f"(V ends {load_v.end_cycle}, QKT starts {qkt_i.start_cycle})")
print()

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(__file__)

# Standard timeline
fig_tl = plot_timeline(
    tl,
    time_unit="cycles",
    max_preview_iters=n_tiles,
    title=f"FA-4 Per-Tensor Loading + Auto-Scheduling ({n_tiles} tiles) — B200",
    figsize=(22, 6),
)
tl_path = os.path.join(out_dir, "fa4_resource_scheduled_timeline.png")
fig_tl.savefig(tl_path, dpi=150, bbox_inches="tight")
print(f"  Timeline chart        -> {tl_path}")

# Timeline with SMEM capacity track
smem_steps = scheduler.get_smem_timeline(tl, pipe)
smem_cap = b200.memory_levels["shared_memory"].capacity_bytes or 0
fig_smem = plot_timeline_with_smem(
    tl, smem_steps, smem_cap,
    time_unit="cycles",
    max_preview_iters=n_tiles,
    title=f"FA-4 Per-Tensor + SMEM Capacity ({n_tiles} tiles) — B200",
    figsize=(22, 8),
)
smem_path = os.path.join(out_dir, "fa4_resource_scheduled_smem.png")
fig_smem.savefig(smem_path, dpi=150, bbox_inches="tight")
print(f"  Timeline + SMEM chart -> {smem_path}")

# Utilization
fig_ut = plot_utilization(
    tl,
    title=f"FA-4 Per-Tensor Auto-Scheduled Utilization ({n_tiles} tiles) — B200",
)
ut_path = os.path.join(out_dir, "fa4_resource_scheduled_utilization.png")
fig_ut.savefig(ut_path, dpi=150, bbox_inches="tight")
print(f"  Utilization chart     -> {ut_path}")
print()
