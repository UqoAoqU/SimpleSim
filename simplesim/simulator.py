"""
simulator.py — Cycle-level performance simulation engine.

The simulator implements the "Feeds and Speeds" bottleneck model:
- Compute each hardware unit's required cycle count independently.
- Total execution time is dominated by the *slowest* unit (max of all cycles).
- All units are assumed to be fully pipelined and overlap perfectly.

Two simulation modes
--------------------
``simulate(workload: Workload)``
    Basic mode.  The caller supplies raw totals for each resource.
    Useful for quick estimates when operand re-read amplification is already
    baked into the numbers.

``simulate_tiled(workload: TiledWorkload)``
    MMA-aware mode.  The simulator computes SMEM traffic from ``MMAOp``
    descriptors, correctly accounting for operand re-reads caused by tiling.
    This mode reproduces the results in FlashAttention-4 Table 1 / Table 3.

Granularity conventions
------------------------
Per-SM resources (tensor_core, cuda_core, sfu, shared_memory):
    ``cycles = ceil(total_ops_or_bytes / ops_or_bytes_per_cycle)``
    Because every SM processes the same tile independently, the cycle count
    is per-SM — a single SM's latency.

Chip-level resources (hbm, l2_cache):
    ``cycles = ceil(total_bytes / bandwidth_bytes_per_cycle)``
    The bandwidth is shared across all SMs; this gives the *minimum* time
    imposed by the memory subsystem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Union

from .hardware import GPUConfig
from .workload import TiledWorkload, Workload


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class UnitResult:
    """Cycle breakdown for a single hardware unit.

    Attributes
    ----------
    unit_name : str
        Same key as in ``GPUConfig.compute_units`` or ``memory_levels``.
    total_ops_or_bytes : int
        Raw work assigned to this unit (ops for compute, bytes for memory).
    throughput_per_cycle : float
        Hardware capacity per cycle (per-SM for SM-local resources,
        chip-level for HBM/L2).
    cycles : int
        ``ceil(total / throughput)`` — minimum latency imposed by this unit.
    time_us : float
        Wall-clock time in microseconds derived from ``cycles`` and clock GHz.
    is_bottleneck : bool
        ``True`` if this unit equals the maximum cycle count.
    """

    unit_name: str
    total_ops_or_bytes: int
    throughput_per_cycle: float
    cycles: int
    time_us: float
    is_bottleneck: bool = False


@dataclass
class SmemBreakdownEntry:
    """Per-MMAOp entry in the SMEM breakdown table."""
    op_name: str
    bytes_contributed: int
    amplification: float


@dataclass
class SimResult:
    """Full simulation result for one workload on one GPU.

    Attributes
    ----------
    workload_name : str
    hardware_name : str
    mode : str
        ``"basic"`` or ``"mma_aware"``.
    compute_results : dict[str, UnitResult]
        One entry per compute unit in the hardware config.
    memory_results : dict[str, UnitResult]
        One entry per memory level in the hardware config.
    total_cycles : int
        Maximum across all unit cycles — the bottleneck cycle count.
    total_time_us : float
        Total time in microseconds.
    bottleneck_units : list[str]
        Names of all units tied at ``total_cycles``.
    utilization : dict[str, float]
        Fraction of total_cycles each unit is busy (0–1).
    smem_breakdown : list[SmemBreakdownEntry]
        Non-empty in MMA-aware mode; one entry per MMAOp.
    """

    workload_name: str
    hardware_name: str
    mode: str
    compute_results: dict[str, UnitResult] = field(default_factory=dict)
    memory_results: dict[str, UnitResult] = field(default_factory=dict)
    total_cycles: int = 0
    total_time_us: float = 0.0
    bottleneck_units: list[str] = field(default_factory=list)
    utilization: dict[str, float] = field(default_factory=dict)
    smem_breakdown: list[SmemBreakdownEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class CycleSimulator:
    """GPU cycle-level performance simulator.

    Parameters
    ----------
    hw : GPUConfig
        Target hardware model loaded from a YAML config.
    """

    def __init__(self, hw: GPUConfig) -> None:
        self.hw = hw

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(self, workload: Workload) -> SimResult:
        """Basic-mode simulation: user supplies pre-aggregated totals.

        Parameters
        ----------
        workload : Workload
            Raw ops/bytes per resource.

        Returns
        -------
        SimResult
            Populated result including per-unit cycles, utilization, and
            bottleneck identification.
        """
        hw = self.hw
        compute_results: dict[str, UnitResult] = {}
        memory_results: dict[str, UnitResult] = {}

        # --- Compute units (per-SM) ---
        for unit_name, total_ops in workload.compute_ops.items():
            cu = hw.compute_units.get(unit_name)
            if cu is None:
                continue
            throughput = cu.ops_per_cycle  # per SM per cycle
            cycles = math.ceil(total_ops / throughput) if throughput > 0 else 0
            compute_results[unit_name] = UnitResult(
                unit_name=unit_name,
                total_ops_or_bytes=total_ops,
                throughput_per_cycle=throughput,
                cycles=cycles,
                time_us=hw.cycles_to_us(cycles),
            )

        # --- Memory levels ---
        for level_name, total_bytes in workload.memory_bytes.items():
            ml = hw.memory_levels.get(level_name)
            if ml is None:
                continue
            # Per-SM resources: each SM handles its own tile independently.
            # Chip-level resources: total bytes shared across all SMs.
            if ml.is_per_sm:
                throughput = ml.bandwidth_bytes_per_cycle  # per SM
            else:
                throughput = ml.bandwidth_bytes_per_cycle  # chip-level
            cycles = math.ceil(total_bytes / throughput) if throughput > 0 else 0
            memory_results[level_name] = UnitResult(
                unit_name=level_name,
                total_ops_or_bytes=total_bytes,
                throughput_per_cycle=throughput,
                cycles=cycles,
                time_us=hw.cycles_to_us(cycles),
            )

        return self._build_result(workload.name, "basic", compute_results,
                                  memory_results, [])

    def simulate_tiled(self, workload: TiledWorkload) -> SimResult:
        """MMA-aware simulation using per-MMAOp SMEM operand re-read counting.

        This method replicates the "Feeds and Speeds" analysis from the
        FlashAttention-4 paper (Zadouri et al., 2026).  SMEM traffic is
        computed via ``MMAOp.smem_bytes()``, which accounts for how many
        times each operand stripe is re-read when the output tile exceeds
        the hardware MMA tile size.

        Parameters
        ----------
        workload : TiledWorkload

        Returns
        -------
        SimResult
        """
        hw = self.hw
        compute_results: dict[str, UnitResult] = {}
        memory_results: dict[str, UnitResult] = {}

        # --- Tensor core (MMA compute) ---
        tc = hw.compute_units.get("tensor_core")
        if tc is not None and workload.mma_ops:
            total_mma_ops = workload.total_mma_ops()
            cycles = math.ceil(total_mma_ops / tc.ops_per_cycle)
            compute_results["tensor_core"] = UnitResult(
                unit_name="tensor_core",
                total_ops_or_bytes=total_mma_ops,
                throughput_per_cycle=tc.ops_per_cycle,
                cycles=cycles,
                time_us=hw.cycles_to_us(cycles),
            )

        # --- Elementwise compute units (SFU, CUDA Core, ...) ---
        for unit_name, total_ops in workload.elementwise_ops.items():
            cu = hw.compute_units.get(unit_name)
            if cu is None:
                continue
            cycles = math.ceil(total_ops / cu.ops_per_cycle) if cu.ops_per_cycle > 0 else 0
            compute_results[unit_name] = UnitResult(
                unit_name=unit_name,
                total_ops_or_bytes=total_ops,
                throughput_per_cycle=cu.ops_per_cycle,
                cycles=cycles,
                time_us=hw.cycles_to_us(cycles),
            )

        # --- Shared memory (MMA-aware, per-SM) ---
        smem_ml = hw.memory_levels.get("shared_memory")
        smem_breakdown: list[SmemBreakdownEntry] = []
        if smem_ml is not None:
            total_smem = workload.total_smem_bytes()
            for op in workload.mma_ops:
                smem_breakdown.append(SmemBreakdownEntry(
                    op_name=op.name,
                    bytes_contributed=op.smem_bytes(),
                    amplification=op.smem_amplification(),
                ))
            if workload.extra_smem_bytes > 0:
                smem_breakdown.append(SmemBreakdownEntry(
                    op_name="extra",
                    bytes_contributed=workload.extra_smem_bytes,
                    amplification=1.0,
                ))
            if total_smem > 0:
                bw = smem_ml.bandwidth_bytes_per_cycle
                cycles = math.ceil(total_smem / bw) if bw > 0 else 0
                memory_results["shared_memory"] = UnitResult(
                    unit_name="shared_memory",
                    total_ops_or_bytes=total_smem,
                    throughput_per_cycle=bw,
                    cycles=cycles,
                    time_us=hw.cycles_to_us(cycles),
                )

        # --- HBM (chip-level) ---
        hbm_ml = hw.memory_levels.get("hbm")
        if hbm_ml is not None and workload.hbm_bytes > 0:
            bw = hbm_ml.bandwidth_bytes_per_cycle
            cycles = math.ceil(workload.hbm_bytes / bw) if bw > 0 else 0
            memory_results["hbm"] = UnitResult(
                unit_name="hbm",
                total_ops_or_bytes=workload.hbm_bytes,
                throughput_per_cycle=bw,
                cycles=cycles,
                time_us=hw.cycles_to_us(cycles),
            )

        # --- Other memory levels (L2, etc.) from basic-mode field ---
        for level_name, ml in hw.memory_levels.items():
            if level_name in ("shared_memory", "hbm"):
                continue  # already handled

        return self._build_result(workload.name, "mma_aware", compute_results,
                                  memory_results, smem_breakdown)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_result(
        self,
        workload_name: str,
        mode: str,
        compute_results: dict[str, UnitResult],
        memory_results: dict[str, UnitResult],
        smem_breakdown: list[SmemBreakdownEntry],
    ) -> SimResult:
        all_results = {**compute_results, **memory_results}
        if not all_results:
            return SimResult(
                workload_name=workload_name,
                hardware_name=self.hw.name,
                mode=mode,
                compute_results=compute_results,
                memory_results=memory_results,
                smem_breakdown=smem_breakdown,
            )

        total_cycles = max(r.cycles for r in all_results.values())
        total_time_us = self.hw.cycles_to_us(total_cycles)

        bottleneck_units = [
            name for name, r in all_results.items()
            if r.cycles == total_cycles
        ]

        utilization = {
            name: (r.cycles / total_cycles if total_cycles > 0 else 0.0)
            for name, r in all_results.items()
        }

        # Mark bottleneck flag on each UnitResult
        for name, r in all_results.items():
            r.is_bottleneck = (r.cycles == total_cycles)

        return SimResult(
            workload_name=workload_name,
            hardware_name=self.hw.name,
            mode=mode,
            compute_results=compute_results,
            memory_results=memory_results,
            total_cycles=total_cycles,
            total_time_us=total_time_us,
            bottleneck_units=bottleneck_units,
            utilization=utilization,
            smem_breakdown=smem_breakdown,
        )
