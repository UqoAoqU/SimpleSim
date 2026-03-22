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
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
b200 = load_gpu_config(os.path.join(CONFIG_DIR, "b200.yaml"))

def run(
    m: int = 128,
    n_tile: int = 128,
    d: int = 128,
    seq_kv: int = 1024,
    dtype_bytes: int = 2,
) -> None:
    n_tiles = seq_kv // n_tile

    hbm_per_tile = 2 * n_tile * d * dtype_bytes

    kv_tile_wl = TiledWorkload(
        name="kv_tile",
        mma_ops=[
            MMAOp(
                "QKt",
                output_shape=(m, n_tile),
                reduction_dim=d,
                operand_a_source="smem",
                operand_b_source="smem",
                dtype_bytes=dtype_bytes,
            ),
            MMAOp(
                "PV",
                output_shape=(m, d),
                reduction_dim=n_tile,
                operand_a_source="tmem",
                operand_b_source="smem",
                dtype_bytes=dtype_bytes,
            ),
        ],
        elementwise_ops={
            "sfu": m * n_tile,
            "cuda_core": 3 * m * n_tile + m * d + 2 * m,
        },
        hbm_bytes=hbm_per_tile,
    )

    update_o_wl = Workload(
        name="update_o",
        compute_ops={
            "sfu": m,
            "cuda_core": m * d,
        },
    )

    pipe = Pipeline(f"FA-4 fwd (seq_kv={seq_kv}, {n_tiles} tiles)", stages=[
        Stage("kv_tile", workload=kv_tile_wl, repeat=n_tiles),
        Stage("update_o", workload=update_o_wl, depends_on=["kv_tile"]),
    ])

    sim = TimelineSimulator(b200)
    tl = sim.simulate(pipe)

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  FA-4 KV-Tile Pipeline — Pipelining Demo (B200)              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  seq_kv={seq_kv}, n_tiles={n_tiles}, M=N={n_tile}, d={d}, BF16")
    print(f"  HBM per tile : {hbm_per_tile:,} B")
    print()

    single_res = sim._inner.simulate_tiled(kv_tile_wl)
    hbm_ur = single_res.memory_results.get("hbm")
    tc_ur = single_res.compute_results.get("tensor_core")
    hbm_cyc = hbm_ur.cycles if hbm_ur else 0
    tc_cyc = tc_ur.cycles if tc_ur else 0

    print(f"  Per-tile bottleneck    : {single_res.total_cycles} cyc "
          f"({'+'.join(single_res.bottleneck_units)})")
    print(f"  Per-tile HBM           : {hbm_cyc} cyc  "
          f"({'FULLY hidden by compute' if hbm_cyc <= tc_cyc else 'visible'})")

    serial_per_tile = single_res.total_cycles
    serial_tiles_total = serial_per_tile * n_tiles

    tile_scheds = [s for s in tl.schedules if s.stage_name.startswith("kv_tile")]
    pipelined_tiles_end = tile_scheds[-1].end_cycle
    pipeline_overhead = max(0, hbm_cyc - tc_cyc)

    print(f"  Serial total  ({n_tiles} tiles): {serial_tiles_total:,} cyc")
    print(f"  Pipelined tiles end    : {pipelined_tiles_end:,} cyc  "
          f"({b200.cycles_to_us(pipelined_tiles_end):.3f} µs)")
    if pipeline_overhead == 0:
        print(f"  HBM latency hidden     : fully (HBM {hbm_cyc} cyc ≤ TC {tc_cyc} cyc)")
    else:
        print(f"  Pipeline overhead      : +{pipeline_overhead} cyc/iter (HBM > TC)")
    print()

    print("  Iteration schedule (first 4 + last):")
    col = (16, 10, 10, 36)
    hdr = ("Stage", "Start", "End", "Unit intervals (cycles)")
    sep = "  ".join("-" * c for c in col)
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

    out_dir = os.path.dirname(__file__)

    fig_tl = plot_timeline(
        tl,
        time_unit="cycles",
        max_preview_iters=n_tiles,
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


def main() -> None:
    run()


if __name__ == "__main__":
    main()
