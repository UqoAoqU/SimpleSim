"""
examples/fa4_forward_breakdown.py
==================================
Fine-grained 5-stage breakdown of the FlashAttention-4 forward pass (one KV tile).

Stages and their hardware footprint
-------------------------------------
Within one iteration of the outer KV-tile loop:

  QK^T        S(M,N) = Q(M,d) × K^T(d,N)          SS-MMA  → tensor_core + shared_memory
  Softmax     P = softmax(S), update m / sum_p      SFU + CUDA core
  Rescale O   O *= exp(m_old-m_new), l update       CUDA core (+ tiny SFU)
  PV          O += P(M,N) × V(N,d)                 TS-MMA  → tensor_core + shared_memory

After all KV tiles (once):
  Update O    O /= l  (final normalisation)         CUDA core + tiny SFU

Dependencies: QK^T → Softmax → Rescale O → PV → Update O  (linear chain)

Work breakdown (M=N=d=128, BF16, B200)
-----------------------------------------
                    tensor_core   shared_memory   sfu        cuda_core
  QK^T              2·128³ ops    65 536 B        —          —
                    = 4 194 304   → 512 cyc       —          —
                    → 512 cyc
  Softmax           —             —               128² ops   3·128² ops
                                                  = 16 384   = 49 152
                                                  → 1024 cyc → 384 cyc
  Rescale O         —             —               128 ops    128²+2·128 ops
                                                  = 128      = 16 640
                                                  → 8 cyc    → 130 cyc
  PV                2·128³ ops    32 768 B        —          —
                    = 4 194 304   → 256 cyc       —          —
                    → 512 cyc
  Update O          —             —               128 ops    128² ops
                                                  = 128      = 16 384
                                                  → 8 cyc    → 128 cyc

  Bottleneck:       512 (tie)     512 (tie)       1024       130 / 128
  Per-stage BN:     512           —               1024       130 / 128
  Total (serial):   512 + 1024 + 130 + 512 + 128 = 2306 cycles

Run
---
    cd SimpleSim
    python -m examples.fa4_forward_breakdown
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


# ---------------------------------------------------------------------------
# Workload factory
# ---------------------------------------------------------------------------

def fa4_forward_stages(M: int, N: int, d: int, dtype_bytes: int = 2) -> list[Stage]:
    """Return the 5 fine-grained stages of the FA-4 forward pass for one KV tile.

    Parameters
    ----------
    M           : query tile rows
    N           : key/value tile columns
    d           : head dimension (reduction dim for both MMAs)
    dtype_bytes : element size (2 = BF16)
    """

    # ── Stage 1: QK^T ───────────────────────────────────────────────────────
    # S(M,N) = Q(M,d) × K^T(d,N)
    # Both Q and K from SMEM  →  SS-MMA
    #
    # tensor_core : 2·M·N·d   FLOPs
    # shared_memory: nm·nn·(hw_m·d + d·hw_n)·dtype bytes
    #              = 1·1·(M·d + d·N)·2  =  2·M·d·2  (since M=N=d=128)
    qkt_wl = TiledWorkload(
        name="QK^T",
        mma_ops=[MMAOp(
            name="QKt",
            output_shape=(M, N),
            reduction_dim=d,
            operand_a_source="smem",   # Q from SMEM
            operand_b_source="smem",   # K from SMEM
            dtype_bytes=dtype_bytes,
        )],
    )

    # ── Stage 2: Softmax ────────────────────────────────────────────────────
    # Online softmax update of attention scores S → P, m_new, sum_p
    #
    # (a) m_new  = rowmax(S)    M·N  fmax    [cuda_core FP32]
    # (b) S     -= m_new        M·N  fsub    [cuda_core FP32]
    # (c) P      = exp(S)       M·N  exp     [SFU]
    # (d) sum_p  = rowsum(P)    M·N  fadd    [cuda_core FP32]
    softmax_wl = Workload(
        name="Softmax",
        compute_ops={
            "sfu":       M * N,          # (c) exp of every score
            "cuda_core": 3 * M * N,      # (a) fmax  + (b) fsub  + (d) fadd
        },
    )

    # ── Stage 3: Rescale O ──────────────────────────────────────────────────
    # Bring the accumulated O from previous tiles up to date before PV adds
    # this tile's contribution.
    #
    # (e) α      = exp(m_old - m_new)   M   exp     [SFU]   (one per row)
    # (f) O     *= α                    M·d fmul    [cuda_core]
    # (g) l     *= α                    M   fmul    [cuda_core]
    # (h) l     += sum_p                M   fadd    [cuda_core]
    rescale_wl = Workload(
        name="Rescale O",
        compute_ops={
            "sfu":       M,                 # (e) exp per row
            "cuda_core": M * d + 2 * M,     # (f) O·=α  +  (g)(h) l update
        },
    )

    # ── Stage 4: PV ─────────────────────────────────────────────────────────
    # O(M,d) += P(M,N) × V(N,d)
    # P lives in TMEM (zero SMEM cost), V from SMEM  →  TS-MMA
    #
    # tensor_core : 2·M·d·N   FLOPs
    # shared_memory: nm·nn·(K·hw_n)·dtype  (only B operand from SMEM)
    #              = 1·1·N·d·2  =  32 768 bytes   (for M=N=d=128)
    pv_wl = TiledWorkload(
        name="PV",
        mma_ops=[MMAOp(
            name="PV",
            output_shape=(M, d),
            reduction_dim=N,
            operand_a_source="tmem",   # P in TMEM — no SMEM traffic
            operand_b_source="smem",   # V from SMEM
            dtype_bytes=dtype_bytes,
        )],
    )

    # ── Stage 5: Update O ───────────────────────────────────────────────────
    # Final normalisation — runs ONCE at kernel end after all KV tiles.
    #
    # (i-rcp) rcp.approx(l)    M    rcp     [SFU]
    # (i-mul) O *= rcp(l)      M·d  fmul    [cuda_core]
    update_o_wl = Workload(
        name="Update O",
        compute_ops={
            "sfu":       M,         # rcp.approx(l) per row
            "cuda_core": M * d,     # O *= rcp(l)
        },
    )

    # ── Dependency chain ────────────────────────────────────────────────────
    # QK^T must finish before softmax can exponentiate the scores.
    # Rescale needs m_new from softmax, and must update O before PV adds to it.
    # PV needs P from softmax (in TMEM) and a rescaled O accumulator.
    # Update O runs last, needs all tiles processed.
    return [
        Stage("QK^T",     workload=qkt_wl),
        Stage("Softmax",  workload=softmax_wl,  depends_on=["QK^T"]),
        Stage("Rescale O",workload=rescale_wl,  depends_on=["Softmax"]),
        Stage("PV",       workload=pv_wl,       depends_on=["Rescale O"]),
        Stage("Update O", workload=update_o_wl, depends_on=["PV"]),
    ]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(M: int = 128, N: int = 128, d: int = 128, dtype_bytes: int = 2) -> None:
    stages = fa4_forward_stages(M, N, d, dtype_bytes)
    pipe   = Pipeline(f"FA-4 Forward (M={M},N={N},d={d})", stages=stages)

    sim = TimelineSimulator(b200)
    tl  = sim.simulate(pipe)

    # ── Text summary ────────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  FA-4 Forward Pass — Fine-grained Stage Breakdown (B200)         ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  M={M}, N={N}, d={d},  dtype={'BF16' if dtype_bytes==2 else 'FP8'}")
    print(f"  Total (serial stages): {tl.total_cycles:,} cycles  /  {tl.total_time_us:.3f} µs")

    # Compare to flat single-workload bottleneck
    from simplesim import CycleSimulator
    from examples.flash_attention import fa4_forward
    flat_res = CycleSimulator(b200).simulate_tiled(fa4_forward(M, N, d, dtype_bytes))
    print(f"  Flat single-workload:  {flat_res.total_cycles:,} cycles  "
          f"(bottleneck: {'+'.join(flat_res.bottleneck_units)})")
    print()

    # Per-stage table
    col = (12, 10, 10, 10, 22, 44)
    hdr = ("Stage", "Start", "End", "Duration", "Bottleneck unit(s)", "Unit durations (cycles)")
    sep = "  ".join("-" * c for c in col)
    print("  " + "  ".join(h.ljust(c) for h, c in zip(hdr, col)))
    print("  " + sep)
    for s in tl.schedules:
        units_str = "  ".join(
            f"{u}:{iv.unit_cycles}" for u, iv in sorted(s.unit_intervals.items())
        )
        bn = " + ".join(s.sim_result.bottleneck_units) if s.sim_result else "?"
        row = [
            s.stage_name.ljust(col[0]),
            str(s.start_cycle).ljust(col[1]),
            str(s.end_cycle).ljust(col[2]),
            str(s.stage_cycles).ljust(col[3]),
            bn.ljust(col[4]),
            units_str,
        ]
        print("  " + "  ".join(row))
    print()

    # Per-stage SimResult details
    for s in tl.schedules:
        if s.sim_result is not None:
            print_report(s.sim_result)

    # ── Plots ────────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(__file__)

    fig_tl = plot_timeline(
        tl,
        time_unit="cycles",
        title=f"FA-4 Forward Stage Breakdown (M={M},N={N},d={d}) — B200",
    )
    tl_path = os.path.join(out_dir, "fa4_fwd_breakdown_timeline.png")
    fig_tl.savefig(tl_path, dpi=150, bbox_inches="tight")
    print(f"  Timeline chart     → {tl_path}")

    fig_ut = plot_utilization(
        tl,
        title=f"FA-4 Forward Unit Utilization (M={M},N={N},d={d}) — B200",
    )
    ut_path = os.path.join(out_dir, "fa4_fwd_breakdown_utilization.png")
    fig_ut.savefig(ut_path, dpi=150, bbox_inches="tight")
    print(f"  Utilization chart  → {ut_path}")
    print()

    try:
        import matplotlib.pyplot as plt
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    run(M=128, N=128, d=128)
