"""
examples/fa4_pipeline.py
========================
Demonstrates the Pipeline / TimelineSimulator / plot_timeline API using a
realistic FlashAttention-4 sequence on B200.

Pipeline structure
------------------

    [load_qkv] ──► [fa4_forward] ──► [fa4_backward] ──► [write_grad]

Each stage has explicit data dependencies:
  - fa4_forward  depends on load_qkv    (needs Q/K/V in HBM before starting)
  - fa4_backward depends on fa4_forward (needs forward activations / softmax stats)
  - write_grad   depends on fa4_backward (writes dQ, dK, dV to HBM)

What this example shows
-----------------------
1. How to wrap existing TiledWorkload instances in Stage objects.
2. How to declare dependencies with ``depends_on``.
3. How to call ``TimelineSimulator.simulate(pipeline)`` → ``TimelineResult``.
4. How to generate and save the Gantt timeline and utilization charts.

Run
---
    cd SimpleSim
    python -m examples.fa4_pipeline
"""

import os

from simplesim import (
    load_gpu_config,
    Workload,
    Stage,
    Pipeline,
    TimelineSimulator,
    plot_timeline,
    plot_utilization,
    print_report,
)

# Re-use workload factory functions from the existing flash_attention example
from examples.flash_attention import fa4_forward, fa4_backward

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
b200 = load_gpu_config(os.path.join(CONFIG_DIR, "b200.yaml"))

def run(
    m: int = 128,
    n: int = 128,
    d: int = 128,
    dtype_bytes: int = 2,
    seq_len: int = 2048,
) -> None:
    # Stage 0: load Q, K, V from HBM into SRAM  (pure memory transfer)
    load_qkv = Workload(
        name="load_qkv",
        compute_ops={},
        memory_bytes={"hbm": 3 * seq_len * d * dtype_bytes},
    )

    fwd_wl = fa4_forward(m, n, d, dtype_bytes=dtype_bytes, batch_seqlen=seq_len)
    bwd_wl = fa4_backward(m, n, d, dtype_bytes=dtype_bytes, batch_seqlen=seq_len)

    # Stage 3: write dQ, dK, dV gradients back to HBM
    write_grad = Workload(
        name="write_grad",
        compute_ops={},
        memory_bytes={"hbm": 3 * seq_len * d * dtype_bytes},
    )

    pipe = Pipeline("FA-4 fwd+bwd", stages=[
        Stage("load_qkv", workload=load_qkv),
        Stage("fa4_forward", workload=fwd_wl, depends_on=["load_qkv"]),
        Stage("fa4_backward", workload=bwd_wl, depends_on=["fa4_forward"]),
        Stage("write_grad", workload=write_grad, depends_on=["fa4_backward"]),
    ])

    sim = TimelineSimulator(b200)
    tl = sim.simulate(pipe)

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║  FA-4 Pipeline Timeline — B200                       ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Pipeline : {tl.pipeline_name}")
    print(f"  Hardware : {tl.hardware_name}")
    print(f"  Total    : {tl.total_cycles:,} cycles  /  {tl.total_time_us:.3f} µs")
    print()

    col_w = (20, 10, 10, 10, 36)
    header = ("Stage", "Start(cyc)", "End(cyc)", "Dur(cyc)", "Unit intervals (cycles)")
    sep = "  ".join("-" * w for w in col_w)
    print("  " + "  ".join(h.ljust(w) for h, w in zip(header, col_w)))
    print("  " + sep)

    for sched in tl.schedules:
        units_str = "  ".join(
            f"{u}:{iv.unit_cycles}" for u, iv in sorted(sched.unit_intervals.items())
        )
        print("  " + "  ".join([
            sched.stage_name.ljust(col_w[0]),
            str(sched.start_cycle).ljust(col_w[1]),
            str(sched.end_cycle).ljust(col_w[2]),
            str(sched.stage_cycles).ljust(col_w[3]),
            units_str.ljust(col_w[4]),
        ]))

    print()

    for sched in tl.schedules:
        if sched.sim_result is not None:
            print_report(sched.sim_result)

    out_dir = os.path.dirname(__file__)

    fig_tl = plot_timeline(tl, time_unit="us")
    tl_path = os.path.join(out_dir, "fa4_pipeline_timeline.png")
    fig_tl.savefig(tl_path, dpi=150, bbox_inches="tight")
    print(f"  Timeline chart saved → {tl_path}")

    fig_ut = plot_utilization(tl)
    ut_path = os.path.join(out_dir, "fa4_pipeline_utilization.png")
    fig_ut.savefig(ut_path, dpi=150, bbox_inches="tight")
    print(f"  Utilization chart saved → {ut_path}")
    print()

    try:
        import matplotlib.pyplot as plt
        plt.show()
    except Exception:
        pass


def main() -> None:
    run()


if __name__ == "__main__":
    main()
