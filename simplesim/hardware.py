"""
hardware.py — GPU hardware model dataclasses + YAML loader.

All per-SM throughput values (compute_units and shared_memory) are expressed
per-SM per-cycle.  HBM and L2 are expressed per-chip per-cycle because they
are shared across all SMs.  The simulator uses this distinction when summing
cycles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ComputeUnit:
    """A logical compute unit on one SM.

    Attributes
    ----------
    name : str
        Identifier, e.g. ``"tensor_core"``, ``"cuda_core"``, ``"sfu"``.
    ops_per_cycle : float
        FLOPs (or ops) the unit can retire **per SM per cycle**.
        For tensor cores this counts both the multiply and the add of each
        FMA, so 1 FP16 FMA = 2 ops.
    supported_dtypes : list[str]
        Data types the unit can handle, e.g. ``["fp16", "bf16", "fp8"]``.
    """

    name: str
    ops_per_cycle: float
    supported_dtypes: list[str] = field(default_factory=list)


@dataclass
class MemoryLevel:
    """A memory level in the hierarchy.

    Attributes
    ----------
    name : str
        E.g. ``"hbm"``, ``"l2_cache"``, ``"shared_memory"``.
    bandwidth_bytes_per_cycle : float
        Throughput in **bytes per cycle**.
        *shared_memory* — per-SM bandwidth.
        *hbm* / *l2_cache* — per-chip bandwidth (shared across all SMs).
    capacity_bytes : int or None
        Total capacity; ``None`` if not relevant.
    is_per_sm : bool
        ``True`` for shared memory (per-SM resource);
        ``False`` for HBM / L2 (chip-level resource).
    """

    name: str
    bandwidth_bytes_per_cycle: float
    capacity_bytes: Optional[int] = None
    is_per_sm: bool = False


@dataclass
class GPUConfig:
    """Complete hardware description of one GPU SKU.

    Attributes
    ----------
    name : str
        Human-readable GPU name, e.g. ``"B200 SXM"``.
    architecture : str
        NVIDIA micro-architecture, e.g. ``"Blackwell"``.
    num_sms : int
        Number of streaming multiprocessors.
    clock_ghz : float
        Boost clock in GHz, used to convert cycles → wall time.
    compute_units : dict[str, ComputeUnit]
        Mapping from unit name to its descriptor.  Key is the same as
        ``ComputeUnit.name``.
    memory_levels : dict[str, MemoryLevel]
        Mapping from level name to its descriptor.
    """

    name: str
    architecture: str
    num_sms: int
    clock_ghz: float
    compute_units: dict[str, ComputeUnit]
    memory_levels: dict[str, MemoryLevel]

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def tensor_core_ops_per_cycle(self) -> float:
        """Return tensor-core ops/cycle/SM, or 0 if absent."""
        cu = self.compute_units.get("tensor_core")
        return cu.ops_per_cycle if cu else 0.0

    def smem_bytes_per_cycle(self) -> float:
        """Return shared-memory bandwidth in bytes/cycle/SM, or 0 if absent."""
        ml = self.memory_levels.get("shared_memory")
        return ml.bandwidth_bytes_per_cycle if ml else 0.0

    def hbm_bytes_per_cycle(self) -> float:
        """Return HBM bandwidth in bytes/cycle/SM (per-SM share), or 0 if absent."""
        ml = self.memory_levels.get("hbm")
        return ml.bandwidth_bytes_per_cycle if ml else 0.0

    def total_flops_tflops(self, dtype: str = "fp16") -> float:
        """Compute peak chip TFLOPS for the given dtype (approximate)."""
        tc = self.compute_units.get("tensor_core")
        if tc is None:
            return 0.0
        total_flops_per_s = tc.ops_per_cycle * self.num_sms * self.clock_ghz * 1e9
        return total_flops_per_s / 1e12

    def cycles_to_us(self, cycles: float) -> float:
        """Convert a cycle count to microseconds using the chip's clock."""
        return cycles / (self.clock_ghz * 1e3)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def _parse_memory_level(name: str, data: dict) -> MemoryLevel:
    per_sm_levels = {"shared_memory", "l1_cache", "rf", "tmem"}
    # Explicit YAML override takes precedence over name-based detection
    if "is_per_sm" in data:
        is_per_sm = bool(data["is_per_sm"])
    else:
        is_per_sm = name in per_sm_levels
    return MemoryLevel(
        name=name,
        bandwidth_bytes_per_cycle=float(data["bandwidth_bytes_per_cycle"]),
        capacity_bytes=data.get("capacity_bytes"),
        is_per_sm=is_per_sm,
    )


def load_gpu_config(path: str | Path) -> GPUConfig:
    """Load a ``GPUConfig`` from a YAML file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    GPUConfig
        Fully populated hardware model.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    compute_units: dict[str, ComputeUnit] = {}
    for cu_name, cu_data in raw.get("compute_units", {}).items():
        compute_units[cu_name] = ComputeUnit(
            name=cu_name,
            ops_per_cycle=float(cu_data["ops_per_cycle"]),
            supported_dtypes=cu_data.get("supported_dtypes", []),
        )

    memory_levels: dict[str, MemoryLevel] = {}
    for ml_name, ml_data in raw.get("memory_levels", {}).items():
        memory_levels[ml_name] = _parse_memory_level(ml_name, ml_data)

    return GPUConfig(
        name=raw["name"],
        architecture=raw.get("architecture", "Unknown"),
        num_sms=int(raw["num_sms"]),
        clock_ghz=float(raw["clock_ghz"]),
        compute_units=compute_units,
        memory_levels=memory_levels,
    )
