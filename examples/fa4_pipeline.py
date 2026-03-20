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
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
b200 = load_gpu_config(os.path.join(CONFIG_DIR, "b200.yaml"))

# ---------------------------------------------------------------------------
# Workload parameters
# ---------------------------------------------------------------------------
M, N, d = 128, 128, 128   # tile dims matching FA-4 Table 1
DTYPE_BYTES = 2            # BF16
SEQ_LEN = 2048             # total sequence length (for HBM estimate)

# ---------------------------------------------------------------------------
# Stage workloads
# ---------------------------------------------------------------------------

# Stage 0: load Q, K, V from HBM into SRAM  (pure memory transfer)
#   HBM bytes = 3 tensors × SEQ_LEN × d × dtype_bytes
load_qkv = Workload(
    name="load_qkv",
    compute_ops={},
    memory_bytes={"hbm": 3 * SEQ_LEN * d * DTYPE_BYTES},
)

# Stage 1: FA-4 forward pass (one tile at a time; per-tile analysis)
fwd_wl = fa4_forward(M, N, d, dtype_bytes=DTYPE_BYTES, batch_seqlen=SEQ_LEN)

# Stage 2: FA-4 backward pass
bwd_wl = fa4_backward(M, N, d, dtype_bytes=DTYPE_BYTES, batch_seqlen=SEQ_LEN)

# Stage 3: write dQ, dK, dV gradients back to HBM
#   HBM bytes = 3 gradient tensors × SEQ_LEN × d × dtype_bytes
write_grad = Workload(
    name="write_grad",
    compute_ops={},
    memory_bytes={"hbm": 3 * SEQ_LEN * d * DTYPE_BYTES},
)

# ---------------------------------------------------------------------------
# Assemble pipeline
# ---------------------------------------------------------------------------
pipe = Pipeline("FA-4 fwd+bwd", stages=[
    Stage("load_qkv",     workload=load_qkv),
    Stage("fa4_forward",  workload=fwd_wl,   depends_on=["load_qkv"]),
    Stage("fa4_backward", workload=bwd_wl,   depends_on=["fa4_forward"]),
    Stage("write_grad",   workload=write_grad, depends_on=["fa4_backward"]),
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

# Per-stage SimResult details
for sched in tl.schedules:
    if sched.sim_result is not None:
        print_report(sched.sim_result)

# ---------------------------------------------------------------------------
# Visualize
# ---------------------------------------------------------------------------
OUT_DIR = os.path.dirname(__file__)

fig_tl = plot_timeline(tl, time_unit="us")
tl_path = os.path.join(OUT_DIR, "fa4_pipeline_timeline.png")
fig_tl.savefig(tl_path, dpi=150, bbox_inches="tight")
print(f"  Timeline chart saved → {tl_path}")

fig_ut = plot_utilization(tl)
ut_path = os.path.join(OUT_DIR, "fa4_pipeline_utilization.png")
fig_ut.savefig(ut_path, dpi=150, bbox_inches="tight")
print(f"  Utilization chart saved → {ut_path}")
print()

# ---------------------------------------------------------------------------
# Optional: show in interactive window if display is available
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    plt.show()
except Exception:
    pass
