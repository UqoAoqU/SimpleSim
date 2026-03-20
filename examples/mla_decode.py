"""
examples/mla_decode.py
======================
MLA (Multi-head Latent Attention) decoding phase simulation.

MLA Overview
------------
In standard MHA, the KV cache stores K and V tensors separately:
  K_cache: (seq_kv, num_kv_heads × d_head)
  V_cache: (seq_kv, num_kv_heads × d_head)
  → Both must be loaded from HBM, 2× bandwidth demand.

MLA (DeepSeek-V2/V3) compresses the KV cache into a single latent tensor c:
  c_cache: (seq_kv, d_c)   where d_c is the KV latent dimension

Using the "absorbed" projection trick:
  - W_UK (key up-projection) is absorbed into the query projection → Q_absorbed
  - W_UV (value up-projection) is absorbed into the output projection → O_absorbed
  - The attention kernel directly computes Q_absorbed @ c^T and P @ c
  - c is loaded ONCE from HBM and serves as both K and V

This has three important effects:
  1. HBM bandwidth: only seq_kv × d_c × dtype_bytes (vs 2× for MHA)
  2. SMEM pressure: c is READ MULTIPLE TIMES — once as K (for scores) and
     multiple times as V (once per output d_c tile)
  3. Q re-reads: the query matrix is re-read from SMEM once per KV tile

Workload Parameters (this example)
-----------------------------------
  seq_q  = 64       (query tokens per SM, e.g. batch × 1 for decode)
  seq_kv = 16384 / 32768  (KV cache context length)
  d_c    = 576      (MLA latent dimension)
  dtype  = BF16 (2 bytes)

MMA Tile Choice for Decoding
-----------------------------
For seq_q = 64 < 128, the hardware uses a 64×128 MMA tile instead of 128×128.
We model this with hw_tile_m=64 in MMAOp for accuracy.

Kernel Structure (FlashAttention-style outer loop over KV tiles)
-----------------------------------------------------------------
For each KV tile of size N_tile=128 loaded into SMEM:
  Step 1: QK^T  — (seq_q × N_tile) = Q_absorbed(seq_q, d_c) × c_tile^T(d_c, N_tile)
              Both Q and c_tile are in SMEM → SS MMA
  Step 2: Online softmax update (SFU exp + CUDA core reduce)
  Step 3: PV    — (seq_q × d_c) += P_tile(seq_q, N_tile) × c_tile(N_tile, d_c)
              P_tile from TMEM (accumulator), c_tile reused from SMEM → TS MMA

Key insight: c_tile is loaded once from HBM into SMEM per iteration, but it is
read from SMEM:
  - 1× for QK^T (as K)
  - ceil(d_c / hw_tile_n) = ceil(576/128) = 5× for PV (as V, once per d_c output tile)

And Q is re-read from SMEM once per KV tile:
  - num_kv_tiles = seq_kv / N_tile = 128 (for seq_kv=16384)
  - Q re-reads: 128× the base Q size
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simplesim import (
    load_gpu_config,
    MMAOp,
    TiledWorkload,
    CycleSimulator,
    print_report,
    roofline_point,
    print_roofline,
)

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
b200 = load_gpu_config(os.path.join(CONFIG_DIR, "b200.yaml"))
h100 = load_gpu_config(os.path.join(CONFIG_DIR, "h100.yaml"))


# ---------------------------------------------------------------------------
# MLA Decode workload factory
# ---------------------------------------------------------------------------

def mla_decode(
    seq_q: int,
    seq_kv: int,
    d_c: int,
    dtype_bytes: int = 2,
    N_tile: int = 128,
) -> TiledWorkload:
    """MLA decode per-SM workload using the absorbed projection formulation.

    The full kernel loops over seq_kv in tiles of N_tile.  We express the
    total SMEM traffic by using the full output_shape across the seq_kv
    dimension, which is mathematically equivalent to summing per-tile costs
    (see ARCHITECTURE.md §5 for proof).

    SMEM traffic breakdown per SM
    ------------------------------
    QK^T (SS): Q is re-read once per KV tile; c is read once total as K.
      bytes = ceil(seq_q/hw_m) × ceil(seq_kv/hw_n) × (hw_m×d_c + d_c×hw_n) × dtype
            = 1 × (seq_kv/128) × (seq_q×d_c + d_c×128) × 2     [hw_m=seq_q, hw_n=128]

    PV (TS): P from TMEM (free); c re-read ceil(d_c/128) times total as V.
      bytes = ceil(seq_q/hw_m) × ceil(d_c/hw_n) × (seq_kv×hw_n) × dtype
            = 1 × ceil(d_c/128) × seq_kv×128 × 2                [K=seq_kv]

    HBM traffic
    -----------
    c loaded ONCE: seq_kv × d_c × dtype_bytes
    Q loaded once: seq_q × d_c × dtype_bytes   (tiny)
    O written once: seq_q × d_c × dtype_bytes  (tiny)

    Parameters
    ----------
    seq_q   : query tokens handled by this SM (e.g. batch size for decode)
    seq_kv  : KV context length
    d_c     : MLA latent (KV compressed) dimension
    dtype_bytes : element size (2=BF16, 1=FP8)
    N_tile  : KV tile size (determines SMEM per iteration; 128 is standard)
    """
    # For decoding with seq_q < 128, use seq_q as the hardware tile-M size
    # so the SMEM formula correctly counts one Q read per KV tile (not two).
    hw_m = min(seq_q, 128)   # effective MMA tile M for decode

    mma_ops = [
        # ── Step 1: Score computation ──────────────────────────────────
        # Q_absorbed(seq_q, d_c) × c^T(d_c, seq_kv) → S(seq_q, seq_kv)
        # Both Q and c come from SMEM in each KV tile iteration → SS MMA.
        # The N dimension (seq_kv) tiles over KV iterations; each iteration
        # reads hw_m×d_c bytes of Q and d_c×128 bytes of c from SMEM.
        MMAOp(
            name="QKt",
            output_shape=(seq_q, seq_kv),
            reduction_dim=d_c,
            operand_a_source="smem",   # Q: re-read from SMEM per KV tile
            operand_b_source="smem",   # c (as K): streamed from SMEM per KV tile
            dtype_bytes=dtype_bytes,
            hw_tile_m=hw_m,
            hw_tile_n=N_tile,
        ),
        # ── Step 3: Output accumulation ────────────────────────────────
        # P(seq_q, seq_kv) × c(seq_kv, d_c) → O(seq_q, d_c)
        # P lives in TMEM (zero SMEM cost); c is reused from SMEM → TS MMA.
        # The d_c output is tiled into ceil(d_c/128) = 5 tiles, and c is
        # re-read from SMEM for each output tile.
        MMAOp(
            name="PV",
            output_shape=(seq_q, d_c),
            reduction_dim=seq_kv,
            operand_a_source="tmem",   # P: from TMEM (accumulator), zero SMEM cost
            operand_b_source="smem",   # c (as V): re-read ceil(d_c/128) times
            dtype_bytes=dtype_bytes,
            hw_tile_m=hw_m,
            hw_tile_n=N_tile,
        ),
    ]

    # ── Online softmax elementwise ops ──────────────────────────────────
    # The FlashAttention-style outer loop runs num_tiles = seq_kv / N_tile
    # iterations.  Each iteration does:
    #
    #  Per tile (seq_q rows × N_tile cols of score matrix):
    #   (a) row max:        m_new[i] = max(s[i,:])       seq_q*N_tile compare  [CUDA]
    #   (b) score subtract: s[i,j]  -= m_new[i]          seq_q*N_tile sub      [CUDA]
    #   (c) score exp:      p[i,j]   = exp(s[i,j])       seq_q*N_tile exp      [SFU]
    #   (d) row sum:        l_new[i]+= sum(p[i,:])        seq_q*N_tile add      [CUDA]
    #   (e) rescale exp:    α[i]     = exp(m_old-m_new)   seq_q       exp       [SFU]
    #   (f) O rescale:      O[i,:]  *= α[i]              seq_q*d_c   mul       [CUDA]
    #       (O is in TMEM on Blackwell; still costs compute cycles)
    #   (g) l rescale:      l[i]    *= α[i]              seq_q       mul       [CUDA]
    #   (h) l update:       l[i]    += l_new[i]           seq_q       add       [CUDA]
    #
    #  After all tiles (once):
    #   (i) final normalize: O[i,:] /= l[i]              seq_q*d_c   div       [CUDA]
    #
    # Note: m_prev, l_prev, O accumulator live in registers/TMEM — no SMEM cost.
    num_tiles = seq_kv // N_tile

    elementwise_ops = {
        "sfu": (
            seq_q * seq_kv              # (c) exp of every attention score
            + num_tiles * seq_q         # (e) exp of per-row rescale factor α
            + seq_q                     # (i-rcp) rcp.approx(l): one SFU rcp per row
                                        #   FP32 divide is NOT native; compiles to
                                        #   rcp.approx.f32 [SFU] + mul.f32 [cuda_core]
        ),
        "cuda_core": (
            seq_q * seq_kv              # (a) fmax per score (FP32, 128/SM/cyc)
            + seq_q * seq_kv            # (b) fsub s -= m_new  (FP32)
            + seq_q * seq_kv            # (d) fadd row sum of p  (FP32)
            + num_tiles * seq_q * d_c   # (f) fmul O *= α  ← dominant (FP32)
            + num_tiles * seq_q * 2     # (g)+(h) fmul l*=α  +  fadd l+=Σp  (FP32)
            + seq_q * d_c               # (i-mul) fmul O *= rcp(l)  (FP32)
        ),
    }

    # HBM traffic (chip-level):
    #   c loaded once (dominant term, the whole point of MLA)
    #   Q and O are small but included for completeness
    hbm_bytes = (
        seq_kv * d_c * dtype_bytes              # c: loaded once ← key MLA saving
        + seq_q * d_c * dtype_bytes             # Q (read)
        + seq_q * d_c * dtype_bytes             # O (written)
    )

    return TiledWorkload(
        name=f"mla_decode_sq{seq_q}_skv{seq_kv}_d{d_c}",
        mma_ops=mma_ops,
        elementwise_ops=elementwise_ops,
        hbm_bytes=hbm_bytes,
    )


# ---------------------------------------------------------------------------
# Naive MHA decode (for comparison) — K and V stored separately
# ---------------------------------------------------------------------------

def mha_decode_naive(
    seq_q: int,
    seq_kv: int,
    d_kv: int,
    dtype_bytes: int = 2,
    N_tile: int = 128,
) -> TiledWorkload:
    """Standard MHA decode where K and V are stored and loaded separately.

    Unlike MLA:
    - K cache: (seq_kv, d_kv)  — loaded from HBM once for QK^T
    - V cache: (seq_kv, d_kv)  — loaded from HBM again for PV
    → Total HBM = 2 × seq_kv × d_kv × dtype_bytes

    SMEM-wise, K and V can also both be in SMEM per tile (SS + TS same as MLA),
    but K and V are separate tensors so they don't share SMEM reads.
    """
    hw_m = min(seq_q, 128)

    mma_ops = [
        # QK^T: Q(seq_q, d_kv) × K^T(d_kv, seq_kv) → S(seq_q, seq_kv)
        MMAOp(
            name="QKt",
            output_shape=(seq_q, seq_kv),
            reduction_dim=d_kv,
            operand_a_source="smem",
            operand_b_source="smem",
            dtype_bytes=dtype_bytes,
            hw_tile_m=hw_m,
            hw_tile_n=N_tile,
        ),
        # PV: P(seq_q, seq_kv) × V(seq_kv, d_kv) → O(seq_q, d_kv)
        MMAOp(
            name="PV",
            output_shape=(seq_q, d_kv),
            reduction_dim=seq_kv,
            operand_a_source="tmem",
            operand_b_source="smem",
            dtype_bytes=dtype_bytes,
            hw_tile_m=hw_m,
            hw_tile_n=N_tile,
        ),
    ]

    # Same online softmax structure as MLA; d_c → d_kv here.
    num_tiles = seq_kv // N_tile

    elementwise_ops = {
        "sfu": (
            seq_q * seq_kv              # exp of every attention score
            + num_tiles * seq_q         # exp(α) rescale factor
            + seq_q                     # rcp.approx(l) for final normalize
        ),
        "cuda_core": (
            seq_q * seq_kv              # fmax row max
            + seq_q * seq_kv            # fsub s -= m_new
            + seq_q * seq_kv            # fadd row sum of p
            + num_tiles * seq_q * d_kv  # fmul O *= α  (dominant)
            + num_tiles * seq_q * 2     # fmul l*=α  +  fadd l+=Σp
            + seq_q * d_kv              # fmul O *= rcp(l)
        ),
    }

    # HBM: K and V are SEPARATE tensors — loaded from HBM TWICE
    hbm_bytes = (
        seq_kv * d_kv * dtype_bytes * 2         # K + V (two separate loads)
        + seq_q * d_kv * dtype_bytes            # Q
        + seq_q * d_kv * dtype_bytes            # O
    )

    return TiledWorkload(
        name=f"mha_decode_sq{seq_q}_skv{seq_kv}_d{d_kv}",
        mma_ops=mma_ops,
        elementwise_ops=elementwise_ops,
        hbm_bytes=hbm_bytes,
    )


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _smem_amplification(wl: TiledWorkload) -> float:
    """Ratio of total SMEM reads to HBM bytes (SMEM amplification factor)."""
    return wl.total_smem_bytes() / wl.hbm_bytes if wl.hbm_bytes > 0 else float("inf")


def run_analysis(hw, wl: TiledWorkload, label: str = "") -> None:
    sim = CycleSimulator(hw)
    result = sim.simulate_tiled(wl)
    print_report(result)

    rp = roofline_point(result, hw)
    print_roofline(rp, workload_name=wl.name)

    amp = _smem_amplification(wl)
    print(f"  SMEM amplification over HBM : {amp:.2f}×")
    print(f"  (SMEM reads {wl.total_smem_bytes()/1e6:.1f} MB  vs  "
          f"HBM {wl.hbm_bytes/1e6:.1f} MB loaded)")
    print()


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def demo_mla_decode() -> None:
    """Run MLA decode analysis for both context lengths on B200."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  MLA Decode — Feeds & Speeds Analysis (B200)             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print("  seq_q=64, d_c=576, BF16")
    print("  c is loaded ONCE from HBM (serves as both K and V)")
    print()

    for seq_kv in [16384, 32768]:
        print(f"{'─'*60}")
        print(f"  seq_kv = {seq_kv}")
        print(f"{'─'*60}")
        wl = mla_decode(seq_q=64, seq_kv=seq_kv, d_c=576)
        run_analysis(b200, wl)


def demo_mla_vs_mha() -> None:
    """Compare MLA vs naive MHA decode HBM and SMEM costs on B200."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  MLA vs Standard MHA Decode — HBM Savings (B200)        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    seq_kv = 16384
    seq_q  = 64
    d_c    = 576   # MLA latent dim
    d_kv   = 576   # MHA head dim (same size for fair comparison)

    print(f"  seq_q={seq_q}, seq_kv={seq_kv}, d={d_c}, BF16\n")

    sim = CycleSimulator(b200)

    wl_mla = mla_decode(seq_q, seq_kv, d_c)
    wl_mha = mha_decode_naive(seq_q, seq_kv, d_kv)

    res_mla = sim.simulate_tiled(wl_mla)
    res_mha = sim.simulate_tiled(wl_mha)

    print("── MLA (c loaded once, serves as K and V) ──")
    print_report(res_mla)
    print("── Naive MHA (K and V loaded separately) ──")
    print_report(res_mha)

    # Summary comparison
    hbm_mla = wl_mla.hbm_bytes / 1e6
    hbm_mha = wl_mha.hbm_bytes / 1e6
    mla_tc = res_mla.total_cycles
    mha_tc = res_mha.total_cycles

    print(f"{'═'*60}")
    print(f"  {'Metric':<32} {'MLA':>10} {'MHA':>10}")
    print(f"  {'─'*52}")
    print(f"  {'HBM bytes loaded (MB)':<32} {hbm_mla:>10.1f} {hbm_mha:>10.1f}")
    print(f"  {'Total SMEM bytes (MB)':<32} "
          f"{wl_mla.total_smem_bytes()/1e6:>10.1f} "
          f"{wl_mha.total_smem_bytes()/1e6:>10.1f}")
    print(f"  {'Total cycles (bottleneck)':<32} {mla_tc:>10,} {mha_tc:>10,}")
    print(f"  {'Bottleneck unit(s)':<32} "
          f"{'  '.join(res_mla.bottleneck_units):>10} "
          f"{'  '.join(res_mha.bottleneck_units):>10}")
    print(f"  {'SMEM amplification over HBM':<32} "
          f"{_smem_amplification(wl_mla):>9.2f}× "
          f"{_smem_amplification(wl_mha):>9.2f}×")
    print(f"{'═'*60}")
    print()
    print("  Key insight: MLA halves HBM traffic vs MHA (c replaces K+V),")
    print("  but SMEM becomes the binding constraint because:")
    print("  • Q is re-read from SMEM for every KV tile")
    print("  • c is re-read from SMEM ceil(d_c/128) = "
          f"{(d_c+127)//128}× as V (once per output tile along d_c)")
    print()


def demo_h100_comparison() -> None:
    """MLA decode on H100 vs B200."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  MLA Decode — H100 vs B200 Comparison                   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    wl = mla_decode(seq_q=64, seq_kv=16384, d_c=576)
    sim_b200 = CycleSimulator(b200)
    sim_h100 = CycleSimulator(h100)

    res_b200 = sim_b200.simulate_tiled(wl)
    res_h100 = sim_h100.simulate_tiled(wl)

    print("── B200 ──")
    print_report(res_b200)
    print("── H100 ──")
    print_report(res_h100)

    speedup = res_h100.total_cycles / res_b200.total_cycles if res_b200.total_cycles else 0
    print(f"  B200 speedup over H100: {speedup:.2f}×")
    print(f"  B200 bottleneck: {' + '.join(res_b200.bottleneck_units)}")
    print(f"  H100 bottleneck: {' + '.join(res_h100.bottleneck_units)}")
    print()


if __name__ == "__main__":
    demo_mla_decode()
    demo_mla_vs_mha()
    demo_h100_comparison()
