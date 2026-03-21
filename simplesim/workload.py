"""
workload.py — Algorithm workload models.

Two levels of abstraction:

1. ``Workload`` (basic) — user directly specifies per-unit totals.
   Suitable for quick back-of-the-envelope estimates.

2. ``TiledWorkload`` (MMA-aware) — models MMA tiling and shared-memory
   operand re-reads exactly, following the "Feeds and Speeds" methodology
   in FlashAttention-4 (Zadouri et al., 2026).

The key insight in MMA-aware modeling
--------------------------------------
When an output tile (M, N) is larger than the hardware MMA tile (hw_m, hw_n),
multiple MMA instructions are needed.  Each instruction reads a full copy of
its operands from SRAM, so operands are read ``ceil(M/hw_m) * ceil(N/hw_n)``
times — not once.  ``MMAOp.smem_bytes()`` captures this amplification.

Example (FA-4 forward, M=256, N=d=128, B200):
  QKt (SS): num_tiles_m=2, num_tiles_n=1 → operands read 2× → 131 072 bytes
  PV  (TS): A from TMEM (free), B from SMEM → read 2× → 65 536 bytes
  Total SMEM = 196 608 bytes / 128 B·cycle⁻¹ = 1536 cycles  ✓ (Table 1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


OperandSource = Literal["smem", "tmem", "rmem"]

# Supported dtype name → bytes per element
_DTYPE_BYTES: dict[str, float] = {
    "fp32": 4, "tf32": 4,
    "bf16": 2, "fp16": 2,
    "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1,
    "fp4": 0.5,
}


@dataclass
class DataTensor:
    """A named data buffer with type, shape, and storage location.

    Used to describe the inputs and outputs of a workload stage so the
    resource-constrained scheduler can track SMEM capacity and data flow.

    Parameters
    ----------
    name : str
        Descriptive label, e.g. ``"K_tile"``, ``"V_tile"``, ``"S_scores"``.
    shape : tuple[int, ...]
        Element counts per dimension, e.g. ``(128, 128)``.
    dtype : str
        Data type string: ``"bf16"``, ``"fp16"``, ``"fp8"``, ``"fp32"``, etc.
    location : str
        Where the buffer resides: ``"smem"``, ``"hbm"``, ``"rmem"``, ``"tmem"``.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str
    location: str  # "smem", "hbm", "rmem", "tmem"

    @property
    def dtype_bytes(self) -> float:
        """Bytes per element for this dtype."""
        if self.dtype not in _DTYPE_BYTES:
            raise ValueError(f"Unknown dtype {self.dtype!r}; "
                             f"known: {list(_DTYPE_BYTES)}")
        return _DTYPE_BYTES[self.dtype]

    @property
    def size_bytes(self) -> int:
        """Total buffer size in bytes."""
        return int(math.prod(self.shape) * self.dtype_bytes)

    @property
    def smem_bytes(self) -> int:
        """Bytes in shared memory (0 if not stored in SMEM)."""
        return self.size_bytes if self.location == "smem" else 0

    def load_workload(self, *, dst: str = "smem") -> "Workload":
        """Create a ``Workload`` that loads this tensor from HBM to *dst*.

        Models one async DMA transfer of ``size_bytes`` from HBM into the
        destination (typically SMEM).  The resulting stage uses only HBM
        bandwidth (no compute units) and occupies ``size_bytes`` of SMEM
        capacity in the destination.

        Parameters
        ----------
        dst : str
            Destination memory, default ``"smem"``.

        Returns
        -------
        Workload
            A pure-memory workload suitable for use as a pipeline ``Stage``.

        Example
        -------
        ::

            K = DataTensor("K_tile", (128, 128), "bf16", "hbm")
            load_K_stage = Stage("load_K", workload=K.load_workload())
        """
        nbytes = self.size_bytes
        dst_tensor = DataTensor(
            name=self.name,
            shape=self.shape,
            dtype=self.dtype,
            location=dst,
        )
        return Workload(
            name=f"load_{self.name}",
            memory_bytes={"hbm": nbytes},
            outputs=[dst_tensor],
            smem_capacity_bytes=nbytes if dst == "smem" else 0,
        )


@dataclass
class MMAOp:
    """Describes the SMEM access pattern of one matrix-multiply instruction.

    Parameters
    ----------
    name : str
        Descriptive label, e.g. ``"QKt"`` or ``"PV"``.
    output_shape : tuple[int, int]
        (M, N) — the logical output tile computed by this operation.
    reduction_dim : int
        K — the inner (contraction) dimension shared by A and B.
    operand_a_source : OperandSource
        Where operand A is read from.  Use ``"smem"`` for shared memory,
        ``"tmem"`` for Blackwell TMEM (contributes zero SMEM traffic),
        ``"rmem"`` for register file.
    operand_b_source : OperandSource
        Where operand B is read from (same choices as A).
    dtype_bytes : int
        Bytes per element, e.g. 2 for BF16/FP16, 1 for FP8.
    hw_tile_m : int
        Hardware MMA tile size along M (default 128 for Blackwell).
    hw_tile_n : int
        Hardware MMA tile size along N (default 128 for Blackwell).
    """

    name: str
    output_shape: tuple[int, int]
    reduction_dim: int
    operand_a_source: OperandSource
    operand_b_source: OperandSource
    dtype_bytes: int
    hw_tile_m: int = 128
    hw_tile_n: int = 128

    # ------------------------------------------------------------------
    # Core derived quantities
    # ------------------------------------------------------------------

    def num_tiles_m(self) -> int:
        """Number of hardware MMA tiles along the M dimension."""
        return math.ceil(self.output_shape[0] / self.hw_tile_m)

    def num_tiles_n(self) -> int:
        """Number of hardware MMA tiles along the N dimension."""
        return math.ceil(self.output_shape[1] / self.hw_tile_n)

    def smem_bytes(self) -> int:
        """Total SMEM bytes read by this MMA op, including operand re-reads.

        When the logical tile (M, N) exceeds the hardware tile (hw_m, hw_n),
        each operand stripe is re-read for every tile in the other dimension.
        This function implements Equations 1–3 from the FA-4 paper.

        Returns
        -------
        int
            Total SMEM read bytes = num_tiles_m × num_tiles_n × per_tile_bytes,
            where per_tile_bytes counts only operands sourced from SMEM.
        """
        nm = self.num_tiles_m()
        nn = self.num_tiles_n()
        K = self.reduction_dim

        per_tile_bytes = 0
        if self.operand_a_source == "smem":
            # A stripe: hw_tile_m rows × K cols
            per_tile_bytes += self.hw_tile_m * K * self.dtype_bytes
        if self.operand_b_source == "smem":
            # B stripe: K rows × hw_tile_n cols
            per_tile_bytes += K * self.hw_tile_n * self.dtype_bytes

        return nm * nn * per_tile_bytes

    def compute_ops(self) -> int:
        """FLOPs performed by this MMA op (multiply + add = 2 per element)."""
        M, N = self.output_shape
        return 2 * M * N * self.reduction_dim

    def smem_amplification(self) -> float:
        """Ratio of actual SMEM traffic to naive (matrix-size × dtype_bytes).

        A value > 1 indicates operand re-reads due to tiling.
        """
        M, N = self.output_shape
        K = self.reduction_dim
        naive = 0
        if self.operand_a_source == "smem":
            naive += M * K * self.dtype_bytes
        if self.operand_b_source == "smem":
            naive += K * N * self.dtype_bytes
        actual = self.smem_bytes()
        return actual / naive if naive > 0 else 1.0


# ---------------------------------------------------------------------------
# Basic workload (direct specification)
# ---------------------------------------------------------------------------

@dataclass
class Workload:
    """Simple workload specified by total operation counts.

    Attributes
    ----------
    name : str
        Human-readable label.
    compute_ops : dict[str, int]
        Mapping from compute-unit name (must match ``GPUConfig.compute_units``)
        to the total number of operations.
    memory_bytes : dict[str, int]
        Mapping from memory-level name (must match ``GPUConfig.memory_levels``)
        to the total number of bytes transferred.
    inputs : list[DataTensor]
        Data buffers read by this workload (for resource tracking).
    outputs : list[DataTensor]
        Data buffers produced by this workload (for resource tracking).
    smem_capacity_bytes : int
        Shared memory capacity occupied while this workload runs (bytes).
        Unlike SMEM bandwidth (tracked via ``memory_bytes``), this is the
        *footprint* — how much SMEM is simultaneously live.
    """

    name: str
    compute_ops: dict[str, int] = field(default_factory=dict)
    memory_bytes: dict[str, int] = field(default_factory=dict)
    inputs: list[DataTensor] = field(default_factory=list)
    outputs: list[DataTensor] = field(default_factory=list)
    smem_capacity_bytes: int = 0

    def auto_smem_capacity(self) -> int:
        """Compute SMEM capacity from inputs/outputs if not set explicitly."""
        if self.smem_capacity_bytes > 0:
            return self.smem_capacity_bytes
        return sum(t.smem_bytes for t in self.inputs) + sum(
            t.smem_bytes for t in self.outputs
        )


# ---------------------------------------------------------------------------
# MMA-aware workload
# ---------------------------------------------------------------------------

@dataclass
class TiledWorkload:
    """MMA-aware workload that precisely models SMEM operand re-reads.

    This is the preferred representation for any kernel that uses matrix-
    multiply units (Tensor Cores on NVIDIA GPUs), because it correctly
    accounts for the SMEM bandwidth amplification caused by tiling.

    Attributes
    ----------
    name : str
        Human-readable label.
    mma_ops : list[MMAOp]
        All matrix-multiply operations performed per thread-block tile.
    elementwise_ops : dict[str, int]
        Non-MMA scalar / vector operations.  Key must match a compute-unit
        name in ``GPUConfig.compute_units``, e.g. ``"sfu"`` or ``"cuda_core"``.
    hbm_bytes : int
        Total HBM bytes read + written by the entire kernel (chip-level).
        This is usually just the Q, K, V input loads plus the O output store.
    extra_smem_bytes : int
        Additional SMEM bytes not captured by ``mma_ops``, e.g. writing back
        the dS matrix in the backward pass.
    inputs : list[DataTensor]
        Data buffers read by this workload (for resource tracking).
    outputs : list[DataTensor]
        Data buffers produced by this workload (for resource tracking).
    smem_capacity_bytes : int
        Shared memory capacity occupied while this workload runs (bytes).
    """

    name: str
    mma_ops: list[MMAOp] = field(default_factory=list)
    elementwise_ops: dict[str, int] = field(default_factory=dict)
    hbm_bytes: int = 0
    extra_smem_bytes: int = 0
    inputs: list[DataTensor] = field(default_factory=list)
    outputs: list[DataTensor] = field(default_factory=list)
    smem_capacity_bytes: int = 0

    def auto_smem_capacity(self) -> int:
        """Compute SMEM capacity from inputs/outputs if not set explicitly."""
        if self.smem_capacity_bytes > 0:
            return self.smem_capacity_bytes
        return sum(t.smem_bytes for t in self.inputs) + sum(
            t.smem_bytes for t in self.outputs
        )

    # ------------------------------------------------------------------
    # Aggregate helpers (used by the simulator)
    # ------------------------------------------------------------------

    def total_mma_ops(self) -> int:
        """Sum of FLOPs across all MMA operations."""
        return sum(op.compute_ops() for op in self.mma_ops)

    def total_smem_bytes(self) -> int:
        """Sum of SMEM bytes across all MMA operations plus extras."""
        return sum(op.smem_bytes() for op in self.mma_ops) + self.extra_smem_bytes

    def smem_breakdown(self) -> list[tuple[str, int, float]]:
        """Per-MMAOp SMEM breakdown: [(name, bytes, amplification), ...]."""
        return [
            (op.name, op.smem_bytes(), op.smem_amplification())
            for op in self.mma_ops
        ]

    def to_basic_workload(self) -> Workload:
        """Convert to a ``Workload`` for display / basic-mode simulation."""
        compute: dict[str, int] = {}
        if self.mma_ops:
            compute["tensor_core"] = self.total_mma_ops()
        for unit, ops in self.elementwise_ops.items():
            compute[unit] = compute.get(unit, 0) + ops

        memory: dict[str, int] = {"hbm": self.hbm_bytes}
        smem_total = self.total_smem_bytes()
        if smem_total > 0:
            memory["shared_memory"] = smem_total

        return Workload(name=self.name, compute_ops=compute, memory_bytes=memory)
