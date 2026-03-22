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

    hbm_load_wl = Workload(
        name="hbm_load",
        memory_bytes={"hbm": hbm_per_tile},
    )

    qkt_wl = TiledWorkload(
        name="QKT",
        mma_ops=[MMAOp(
            name="QKt",
            output_shape=(m, n_tile),
            reduction_dim=d,
            operand_a_source="smem",
            operand_b_source="smem",
            dtype_bytes=dtype_bytes,
        )],
    )

    softmax_wl = Workload(
        name="Softmax",
        compute_ops={
            "sfu": m * n_tile,
            "cuda_core": 3 * m * n_tile,
        },
    )

    rescale_o_wl = Workload(
        name="RescaleO",
        compute_ops={
            "sfu": m,
            "cuda_core": m * d + 2 * m,
        },
    )

    pv_wl = TiledWorkload(
        name="PV",
        mma_ops=[MMAOp(
            name="PV",
            output_shape=(m, d),
            reduction_dim=n_tile,
            operand_a_source="tmem",
            operand_b_source="smem",
            dtype_bytes=dtype_bytes,
        )],
    )

    update_o_wl = Workload(
        name="update_o",
        compute_ops={
            "sfu": m,
            "cuda_core": m * d,
        },
    )

    stages: list[Stage] = []
    for i in range(n_tiles):
        stages.append(Stage(
            name=f"hbm_load[{i}]",
            workload=hbm_load_wl,
            depends_on=[f"hbm_load[{i-1}]"] if i > 0 else [],
        ))

        qkt_deps = [f"hbm_load[{i}]"]
        if i > 0:
            qkt_deps.append(f"PV[{i-1}]")
        stages.append(Stage(
            name=f"QKT[{i}]",
            workload=qkt_wl,
            depends_on=qkt_deps,
        ))

        stages.append(Stage(
            name=f"Softmax[{i}]",
            workload=softmax_wl,
            depends_on=[f"QKT[{i}]"],
        ))

        stages.append(Stage(
            name=f"RescaleO[{i}]",
            workload=rescale_o_wl,
            depends_on=[f"Softmax[{i}]"],
        ))

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

    sim = TimelineSimulator(b200)
    tl = sim.simulate(pipe)

    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  FA-4 Fine-Grained KV-Tile Pipelining (B200)                     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  seq_kv={seq_kv}, n_tiles={n_tiles}, M=N={n_tile}, d={d}, BF16")
    print(f"  HBM per tile : {hbm_per_tile:,} B")
    print()

    print("  Per-stage schedule (tile 0, 1, ..., last):")
    col = (14, 8, 8, 8, 44)
    hdr = ("Stage", "Start", "End", "Dur", "Unit intervals")
    sep = "  ".join("-" * c for c in col)
    print("  " + "  ".join(h.ljust(c) for h, c in zip(hdr, col)))
    print("  " + sep)

    shown_tiles = {0, 1, n_tiles - 1}
    prev_tile = -1
    for s in tl.schedules:
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

    hbm_sched = next(s for s in tl.schedules if s.stage_name == "hbm_load[0]")
    pv0_sched = next(s for s in tl.schedules if s.stage_name == "PV[0]")
    hbm1_sched = next(s for s in tl.schedules if s.stage_name == "hbm_load[1]")
    qkt1_sched = next(s for s in tl.schedules if s.stage_name == "QKT[1]")

    hbm_cyc = hbm_sched.stage_cycles
    print(f"  HBM load per tile  : {hbm_cyc} cycles")
    print(f"  hbm_load[1] done   : cycle {hbm1_sched.end_cycle}")
    print(f"  QKT[1] start       : cycle {qkt1_sched.start_cycle}  "
          f"(waits for PV[0].end={pv0_sched.end_cycle})")
    print(f"  HBM overlap saving : {hbm_cyc} cyc/tile × {n_tiles} tiles "
          f"= {hbm_cyc * n_tiles} cycles hidden")
    print()

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


def main() -> None:
    run()


if __name__ == "__main__":
    main()
