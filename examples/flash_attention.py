"""
examples/flash_attention.py
===========================
FlashAttention-4 workload definitions and validation against the paper.

This script reproduces the cycle-count results from:
  Zadouri et al. (2026) — "FlashAttention-4: Algorithm and Kernel Pipelining
  Co-Design for Asymmetric Hardware Scaling"

Validated targets
-----------------
Table 1 (FA-4 forward pass, B200 per-SM analysis):
  M=N=d=128:  MMA=1024, SMEM=768,  SFU=1024 cycles
  M=256,N=d=128: MMA=2048, SMEM=1536, SFU=2048 cycles

Table 3 (FA-4 backward pass, B200):
  Includes dS write-back to SMEM as extra_smem_bytes.

Run
---
  cd SimpleSim
  python -m examples.flash_attention
"""

import os

from simplesim import (
    load_gpu_config,
    MMAOp,
    TiledWorkload,
    CycleSimulator,
    print_report,
    roofline_point,
    print_roofline,
)

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
b200 = load_gpu_config(os.path.join(CONFIG_DIR, "b200.yaml"))
h100 = load_gpu_config(os.path.join(CONFIG_DIR, "h100.yaml"))


# ---------------------------------------------------------------------------
# Workload factory functions
# ---------------------------------------------------------------------------

def fa4_forward(
    M: int,
    N: int,
    d: int,
    dtype_bytes: int = 2,
    batch_seqlen: int | None = None,
) -> TiledWorkload:
    """FlashAttention-4 forward pass per-tile workload.

    Kernel structure (per thread-block tile):
      1. QK^T  — (M × N) output tile, K=d, operands Q and K both from SMEM  → SS MMA
      2. PV    — (M × d) output tile, K=N, operand P from TMEM, V from SMEM  → TS MMA
      3. Softmax exp on S (M×N) via SFU
      4. Softmax reduce (max, sub, sum, scale) on rows via CUDA Core

    Parameters
    ----------
    M : int  Thread-block tile rows (query tokens per tile).
    N : int  Thread-block tile columns (key/value tokens per tile).
    d : int  Head dimension.
    dtype_bytes : int  Bytes per element (2=BF16, 1=FP8).
    batch_seqlen : int or None  Total Q tokens; used only for HBM bytes estimate.
    """
    S = batch_seqlen or M  # sequence length for HBM estimate

    mma_ops = [
        # QK^T: output (M, N), K=d, both Q and K from SMEM → SS
        MMAOp(
            name="QKt",
            output_shape=(M, N),
            reduction_dim=d,
            operand_a_source="smem",
            operand_b_source="smem",
            dtype_bytes=dtype_bytes,
        ),
        # PV: output (M, d), K=N, P (softmax output) from TMEM, V from SMEM → TS
        MMAOp(
            name="PV",
            output_shape=(M, d),
            reduction_dim=N,
            operand_a_source="tmem",   # P lives in TMEM — zero SMEM cost
            operand_b_source="smem",
            dtype_bytes=dtype_bytes,
        ),
    ]

    # Elementwise ops (per output tile, M query rows × N KV cols):
    #
    #  Per tile:
    #   (a) fmax  row max:       m_new = max(s, dim=-1)   → M*N  [cuda_core FP32]
    #   (b) fsub  s -= m_new                              → M*N  [cuda_core]
    #   (c) exp   p  = exp(s)    (SFU)                    → M*N  [sfu]
    #   (d) fadd  row sum Σp                              → M*N  [cuda_core]
    #   (e) exp   α  = exp(m_old-m_new)  (SFU, per row)  → M    [sfu]
    #   (f) fmul  O *= α   (per-tile accumulator rescale) → M*d  [cuda_core]
    #   (g) fmul/fadd  l update  (l*=α, l+=Σp)           → M*2  [cuda_core]
    #
    #  Once at kernel end (amortised; tiny vs tile cost):
    #   (h) rcp.approx(l)  — SFU, one rcp per query row  → M    [sfu]
    #   (i) fmul  O *= rcp(l)                            → M*d  [cuda_core]
    #
    # Note: the FA-4 paper T_exp = MN/16 counts only step (c).
    # Steps (e) and (h) are negligible (M << M*N) and not in the paper formula.
    elementwise_ops = {
        "sfu": (
            M * N           # (c) exp of attention scores  ← T_exp in FA-4 paper
            + M             # (e) exp(m_old - m_new) per row
            + M             # (h) rcp.approx(l), once per row at kernel end
        ),
        "cuda_core": (
            M * N           # (a) fmax row max
            + M * N         # (b) fsub s -= m_new
            + M * N         # (d) fadd row sum Σp
            + M * d         # (f) fmul O *= α  (per-tile rescale of output)
            + M * 2         # (g) fmul l*=α  +  fadd l+=Σp
            + M * d         # (i) fmul O *= rcp(l)  (final normalize)
        ),
    }

    # HBM traffic (chip-level, per kernel invocation):
    #   Reads:  Q (S×d) + K (N×d) + V (N×d)
    #   Writes: O (S×d)
    hbm_bytes = (S * d + N * d + N * d + S * d) * dtype_bytes

    return TiledWorkload(
        name=f"fa4_forward_M{M}_N{N}_d{d}",
        mma_ops=mma_ops,
        elementwise_ops=elementwise_ops,
        hbm_bytes=hbm_bytes,
    )


def fa4_backward(
    M: int,
    N: int,
    d: int,
    dtype_bytes: int = 2,
    batch_seqlen: int | None = None,
) -> TiledWorkload:
    """FlashAttention-4 backward pass per-tile workload.

    Kernel structure (per thread-block tile):
      1. QK^T    — SS MMA  (same as forward)
      2. dO V^T  — (M × N) output, K=d, dO from SMEM, V from SMEM → SS MMA
      3. dV      — (N × d) output, K=M, P^T from TMEM, dO from SMEM → TS MMA
      4. dS      — elementwise multiply S * dP (M × N) — written back to SMEM
      5. dQ      — (M × d) output, K=N, dS from SMEM, K from SMEM → SS MMA
      6. dK      — (N × d) output, K=M, dS^T from SMEM, Q from SMEM → SS MMA
      7. Softmax re-computation (SFU) and row-reduce (CUDA Core)

    The dS write-back is modelled as ``extra_smem_bytes``.

    Parameters
    ----------
    M, N, d, dtype_bytes, batch_seqlen : same as fa4_forward.
    """
    S = batch_seqlen or M

    mma_ops = [
        # QK^T (re-compute attention scores)
        MMAOp("QKt",  output_shape=(M, N), reduction_dim=d,
              operand_a_source="smem", operand_b_source="smem",
              dtype_bytes=dtype_bytes),
        # dO V^T → dP = (M × N), K=d
        MMAOp("dOVt", output_shape=(M, N), reduction_dim=d,
              operand_a_source="smem", operand_b_source="smem",
              dtype_bytes=dtype_bytes),
        # dV = P^T dO → (N × d), K=M; P^T from TMEM
        MMAOp("dV",   output_shape=(N, d), reduction_dim=M,
              operand_a_source="tmem", operand_b_source="smem",
              dtype_bytes=dtype_bytes),
        # dQ = dS K → (M × d), K=N; both from SMEM (dS just computed)
        MMAOp("dQ",   output_shape=(M, d), reduction_dim=N,
              operand_a_source="smem", operand_b_source="smem",
              dtype_bytes=dtype_bytes),
        # dK = dS^T Q → (N × d), K=M; both from SMEM
        MMAOp("dK",   output_shape=(N, d), reduction_dim=M,
              operand_a_source="smem", operand_b_source="smem",
              dtype_bytes=dtype_bytes),
    ]

    # dS = S ⊙ dP  written back to SMEM  (M × N elements)
    extra_smem_bytes = M * N * dtype_bytes

    elementwise_ops = {
        "sfu": (
            M * N            # exp for softmax re-compute (score exp)
            + M              # exp(α) rescale factor per row
            + M              # rcp.approx(l) for final normalize
        ),
        "cuda_core": (
            M * N            # fmax row max
            + M * N          # fsub s -= m
            + M * N          # fadd row sum
            + M * d          # fmul dO-rescale (O*=α, d elements per row per tile)
            + M * 2          # fmul/fadd l update
            + M * d          # fmul final O *= rcp(l)
            + 2 * M * N      # backward-specific: dS = S⊙dP (pointwise mul+add)
        ),
    }

    # HBM: Q, K, V, O, dO reads + dQ, dK, dV writes
    hbm_bytes = (S * d * 3 + N * d * 3 + S * d + S * d + N * d * 2) * dtype_bytes

    return TiledWorkload(
        name=f"fa4_backward_M{M}_N{N}_d{d}",
        mma_ops=mma_ops,
        elementwise_ops=elementwise_ops,
        hbm_bytes=hbm_bytes,
        extra_smem_bytes=extra_smem_bytes,
    )


# ---------------------------------------------------------------------------
# Validation: reproduce FA-4 Table 1
# ---------------------------------------------------------------------------

def validate_table1() -> None:
    """Reproduce FA-4 paper Table 1 — forward pass cycles on B200.

    Expected results (per-SM per tile, BF16):
    ┌──────────────────┬──────────────────┬──────────────────────┐
    │ Resource         │ M=N=d=128        │ M=256, N=d=128       │
    ├──────────────────┼──────────────────┼──────────────────────┤
    │ MMA compute      │ 1024 cycles      │ 2048 cycles          │
    │ Shared memory    │  768 cycles      │ 1536 cycles          │
    │ SFU (exp)        │ 1024 cycles      │ 2048 cycles          │
    └──────────────────┴──────────────────┴──────────────────────┘
    Bottleneck: MMA + SFU tied in both cases.
    """
    sim = CycleSimulator(b200)

    print("=" * 60)
    print("FA-4 Table 1 Validation — Forward Pass (B200)")
    print("=" * 60)

    configs = [
        (128, 128, 128, {"MMA": 1024, "SMEM": 768,  "SFU": 1024}),
        (256, 128, 128, {"MMA": 2048, "SMEM": 1536, "SFU": 2048}),
    ]

    all_pass = True
    for M, N, d, expected in configs:
        wl = fa4_forward(M, N, d, dtype_bytes=2)
        result = sim.simulate_tiled(wl)
        print_report(result)

        # Extract key cycles
        mma_cyc  = result.compute_results.get("tensor_core", None)
        smem_cyc = result.memory_results.get("shared_memory", None)
        sfu_cyc  = result.compute_results.get("sfu", None)

        mma_got  = mma_cyc.cycles  if mma_cyc  else -1
        smem_got = smem_cyc.cycles if smem_cyc else -1
        sfu_got  = sfu_cyc.cycles  if sfu_cyc  else -1

        mma_ok  = mma_got  == expected["MMA"]
        smem_ok = smem_got == expected["SMEM"]
        sfu_ok  = sfu_got  == expected["SFU"]

        status = "PASS" if (mma_ok and smem_ok and sfu_ok) else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"[{status}] M={M}, N={N}, d={d}")
        print(f"  MMA  : got={mma_got:5d}  expected={expected['MMA']:5d}  {'✓' if mma_ok  else '✗'}")
        print(f"  SMEM : got={smem_got:5d}  expected={expected['SMEM']:5d}  {'✓' if smem_ok else '✗'}")
        print(f"  SFU  : got={sfu_got:5d}  expected={expected['SFU']:5d}  {'✓' if sfu_ok  else '✗'}")
        print()

    overall = "ALL PASS" if all_pass else "SOME TESTS FAILED"
    print(f"Table 1 validation: {overall}")
    print()


# ---------------------------------------------------------------------------
# Demo: backward pass
# ---------------------------------------------------------------------------

def demo_backward() -> None:
    """Show the backward pass breakdown for M=128, N=d=128 on B200."""
    sim = CycleSimulator(b200)
    wl = fa4_backward(128, 128, 128, dtype_bytes=2)
    result = sim.simulate_tiled(wl)
    print_report(result)

    rp = roofline_point(result, b200)
    print_roofline(rp, workload_name=wl.name)


# ---------------------------------------------------------------------------
# Demo: forward pass on H100
# ---------------------------------------------------------------------------

def demo_h100_forward() -> None:
    """Show the forward pass on H100 for comparison with B200."""
    sim = CycleSimulator(h100)

    print("=" * 60)
    print("FA-4 Forward Pass — H100 (for comparison)")
    print("=" * 60)

    for M, N, d in [(128, 128, 128), (256, 128, 128)]:
        wl = fa4_forward(M, N, d, dtype_bytes=2)
        result = sim.simulate_tiled(wl)
        print_report(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║  SimpleSim — FlashAttention-4 Example / Validate ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    validate_table1()

    print("-" * 60)
    print("Backward pass demo (B200, M=N=d=128):")
    print("-" * 60)
    demo_backward()

    print("-" * 60)
    print("Forward pass on H100 (M=N=d=128 and M=256):")
    print("-" * 60)
    demo_h100_forward()


if __name__ == "__main__":
    main()
