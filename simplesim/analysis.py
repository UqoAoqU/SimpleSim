"""
analysis.py — Human-readable reports and roofline analysis.

Two outputs are provided:

1. ``print_report(result)``
   A text table showing per-unit cycle counts, throughput, utilization,
   and bottleneck flags.  In MMA-aware mode the shared-memory row is
   expanded to show per-MMAOp contributions and amplification factors.

2. ``roofline_point(result)``
   Computes the arithmetic intensity (AI) and places the workload on the
   classical Roofline model: compute-bound if AI > ridge point, else
   memory-bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from .simulator import SimResult


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

_COL_WIDTHS = (24, 22, 18, 8, 8, 6)
_HEADER = ("Resource", "Total Traffic", "Throughput/cyc", "Cycles", "Time(µs)", "Util%")


def _row(cols: tuple, widths=_COL_WIDTHS) -> str:
    return "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))


def _hr() -> str:
    return "-" * (sum(_COL_WIDTHS) + 2 * (len(_COL_WIDTHS) - 1))


def print_report(result: SimResult, *, show_smem_breakdown: bool = True) -> None:
    """Print a formatted performance report to stdout.

    Parameters
    ----------
    result : SimResult
        Output from ``CycleSimulator.simulate`` or ``simulate_tiled``.
    show_smem_breakdown : bool
        When *True* and the result was produced by the MMA-aware mode,
        print per-MMAOp SMEM contributions below the shared-memory row.
    """
    hw_info = f"{result.hardware_name}"
    print()
    print(f"{'=' * 8}  SimpleSim Report: {result.workload_name}  {'=' * 8}")
    print(f"Hardware : {hw_info}  |  Mode: {result.mode}")
    print(f"Bottleneck: {' + '.join(result.bottleneck_units)}  "
          f"({result.total_cycles} cycles,  {result.total_time_us:.2f} µs)")
    print()

    print(_row(_HEADER))
    print(_hr())

    all_entries = list(result.compute_results.items()) + list(result.memory_results.items())
    for unit_name, r in all_entries:
        util_pct = result.utilization.get(unit_name, 0.0) * 100
        bottleneck_flag = " <<<" if r.is_bottleneck else ""
        if r.total_ops_or_bytes >= 1_000_000:
            traffic_str = f"{r.total_ops_or_bytes / 1_000_000:.3f} M"
        elif r.total_ops_or_bytes >= 1_000:
            traffic_str = f"{r.total_ops_or_bytes / 1_000:.1f} K"
        else:
            traffic_str = str(r.total_ops_or_bytes)

        suffix = " ops" if unit_name in result.compute_results else " B"
        label = f"{unit_name}{bottleneck_flag}"
        print(_row((
            label,
            traffic_str + suffix,
            f"{r.throughput_per_cycle:.0f}/cyc",
            str(r.cycles),
            f"{r.time_us:.2f}",
            f"{util_pct:.1f}%",
        )))

        # SMEM breakdown (MMA-aware mode only)
        if (unit_name == "shared_memory" and show_smem_breakdown
                and result.smem_breakdown):
            for entry in result.smem_breakdown:
                if entry.bytes_contributed >= 1_000:
                    b_str = f"{entry.bytes_contributed / 1_000:.1f} K"
                else:
                    b_str = str(entry.bytes_contributed)
                amp_str = f"amp={entry.amplification:.2f}x"
                print(_row((
                    f"  └─ {entry.op_name}",
                    b_str + " B",
                    amp_str,
                    "",
                    "",
                    "",
                )))

    print(_hr())
    print()


# ---------------------------------------------------------------------------
# Roofline analysis
# ---------------------------------------------------------------------------

@dataclass
class RooflinePoint:
    """A workload's position on the Roofline model.

    Attributes
    ----------
    arithmetic_intensity : float
        FLOPs per byte of HBM traffic.
    ridge_point : float
        FLOPs/byte at which the workload transitions from memory-bound to
        compute-bound: ``peak_compute / peak_bandwidth``.
    is_compute_bound : bool
        ``True`` if ``arithmetic_intensity >= ridge_point``.
    peak_compute_tflops : float
        GPU peak tensor-core TFLOPS used for the ridge-point calculation.
    peak_bandwidth_tb_s : float
        GPU peak HBM bandwidth in TB/s.
    attainable_tflops : float
        Roofline performance ceiling at this AI:
        ``min(peak_compute, AI × peak_bandwidth)``.
    """

    arithmetic_intensity: float
    ridge_point: float
    is_compute_bound: bool
    peak_compute_tflops: float
    peak_bandwidth_tb_s: float
    attainable_tflops: float


def roofline_point(result: SimResult, hw, dtype: str = "fp16") -> RooflinePoint:
    """Compute the Roofline model position for a simulation result.

    Parameters
    ----------
    result : SimResult
    hw : GPUConfig
        Hardware model (needed for clock and bandwidth info).
    dtype : str
        Precision string for peak-compute lookup (informational only).

    Returns
    -------
    RooflinePoint
    """
    # Total FLOPs (from compute results)
    total_flops = sum(r.total_ops_or_bytes for r in result.compute_results.values())

    # HBM bytes
    hbm_result = result.memory_results.get("hbm")
    total_hbm_bytes = hbm_result.total_ops_or_bytes if hbm_result else 1

    ai = total_flops / total_hbm_bytes if total_hbm_bytes > 0 else float("inf")

    # Peak compute (TFLOPS) from tensor core
    tc = hw.compute_units.get("tensor_core")
    peak_compute_flops_per_s = (
        tc.ops_per_cycle * hw.num_sms * hw.clock_ghz * 1e9 if tc else 1e12
    )
    peak_compute_tflops = peak_compute_flops_per_s / 1e12

    # Peak HBM bandwidth
    hbm_ml = hw.memory_levels.get("hbm")
    peak_bw_bytes_per_s = (
        hbm_ml.bandwidth_bytes_per_cycle * hw.clock_ghz * 1e9 if hbm_ml else 1e12
    )
    peak_bw_tb_s = peak_bw_bytes_per_s / 1e12

    ridge_point = (peak_compute_flops_per_s / peak_bw_bytes_per_s
                   if peak_bw_bytes_per_s > 0 else float("inf"))

    attainable = min(peak_compute_tflops, ai * peak_bw_tb_s)

    return RooflinePoint(
        arithmetic_intensity=ai,
        ridge_point=ridge_point,
        is_compute_bound=(ai >= ridge_point),
        peak_compute_tflops=peak_compute_tflops,
        peak_bandwidth_tb_s=peak_bw_tb_s,
        attainable_tflops=attainable,
    )


def print_roofline(rp: RooflinePoint, workload_name: str = "") -> None:
    """Print a brief Roofline summary.

    Parameters
    ----------
    rp : RooflinePoint
    workload_name : str
        Label for the printout.
    """
    bound = "compute-bound" if rp.is_compute_bound else "memory-bound"
    label = f" ({workload_name})" if workload_name else ""
    print(f"Roofline{label}:")
    print(f"  AI              = {rp.arithmetic_intensity:.1f} FLOP/B")
    print(f"  Ridge point     = {rp.ridge_point:.1f} FLOP/B")
    print(f"  Regime          : {bound}")
    print(f"  Peak compute    = {rp.peak_compute_tflops:.0f} TFLOPS")
    print(f"  Peak bandwidth  = {rp.peak_bandwidth_tb_s:.2f} TB/s")
    print(f"  Attainable perf = {rp.attainable_tflops:.1f} TFLOPS")
    print()
