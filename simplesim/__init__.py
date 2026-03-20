"""
SimpleSim — A cycle-level GPU performance simulator for attention kernels.

Quick start (single workload)
------------------------------
>>> from simplesim import load_gpu_config, TiledWorkload, MMAOp, CycleSimulator
>>> from simplesim.analysis import print_report
>>> hw = load_gpu_config("configs/b200.yaml")
>>> wl = TiledWorkload(
...     name="fa4_fwd_M128",
...     mma_ops=[
...         MMAOp("QKt", (128, 128), 128, "smem", "smem", dtype_bytes=2),
...         MMAOp("PV",  (128, 128), 128, "tmem", "smem", dtype_bytes=2),
...     ],
...     elementwise_ops={"sfu": 128*128, "cuda_core": 128*128},
...     hbm_bytes=(128+128+128)*128*2 + 128*128*2,
... )
>>> result = CycleSimulator(hw).simulate_tiled(wl)
>>> print_report(result)

Quick start (pipeline with dependencies)
-----------------------------------------
>>> from simplesim import Stage, Pipeline, TimelineSimulator, plot_timeline
>>> pipe = Pipeline("fa4", stages=[
...     Stage("forward",  workload=fwd_wl),
...     Stage("backward", workload=bwd_wl, depends_on=["forward"]),
... ])
>>> tl = TimelineSimulator(hw).simulate(pipe)
>>> fig = plot_timeline(tl)
>>> fig.savefig("timeline.png", dpi=150, bbox_inches="tight")
"""

from .hardware import ComputeUnit, GPUConfig, MemoryLevel, load_gpu_config
from .workload import MMAOp, TiledWorkload, Workload
from .simulator import CycleSimulator, SimResult, UnitResult
from .analysis import print_report, print_roofline, roofline_point
from .pipeline import Pipeline, Stage
from .timeline_sim import StageSchedule, TimelineResult, TimelineSimulator, UnitInterval
from .viz import plot_timeline, plot_utilization

__all__ = [
    # hardware
    "ComputeUnit",
    "GPUConfig",
    "MemoryLevel",
    "load_gpu_config",
    # workload
    "MMAOp",
    "TiledWorkload",
    "Workload",
    # simulator
    "CycleSimulator",
    "SimResult",
    "UnitResult",
    # analysis
    "print_report",
    "print_roofline",
    "roofline_point",
    # pipeline
    "Pipeline",
    "Stage",
    # timeline
    "StageSchedule",
    "TimelineResult",
    "TimelineSimulator",
    "UnitInterval",
    # visualization
    "plot_timeline",
    "plot_utilization",
]
