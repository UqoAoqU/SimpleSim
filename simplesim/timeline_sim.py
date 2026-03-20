"""
timeline_sim.py — Pipeline-level time-line scheduler.

Extends the single-workload ``CycleSimulator`` to a multi-stage pipeline
model that respects data dependencies between stages.

Scheduling model
----------------
* All hardware units within a stage are assumed to run in **parallel**
  (the same "Feeds & Speeds" assumption as ``CycleSimulator``).
* A stage can only **start** once every stage listed in its ``depends_on``
  has **finished** (i.e., its bottleneck unit has completed).
* Within each stage, every unit starts at ``stage.start_cycle`` and runs
  for its own ``unit_cycles`` — units that finish before the bottleneck
  have visible slack in the timeline.

This gives a conservative lower-bound on total pipeline latency that
matches the per-kernel analysis from the FA-4 paper when the pipeline
contains a single stage.

Data structures
---------------
``UnitInterval``
    Half-open cycle range ``[start, end)`` for one hardware unit inside
    one stage.

``StageSchedule``
    All unit intervals for one stage, plus the underlying ``SimResult``.

``TimelineResult``
    The complete scheduled pipeline: one ``StageSchedule`` per stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from .hardware import GPUConfig
from .pipeline import Pipeline, Stage
from .simulator import CycleSimulator, SimResult
from .workload import TiledWorkload, Workload


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class UnitInterval:
    """Cycle range for one hardware unit within one stage.

    Attributes
    ----------
    unit_name : str
    start_cycle : int
        Absolute cycle at which this unit begins work (equals stage start).
    end_cycle : int
        Absolute cycle at which this unit finishes (start + unit_cycles).
    unit_cycles : int
        Duration of this unit's work, i.e. ``end_cycle - start_cycle``.
    is_bottleneck : bool
        True if this unit equals the stage bottleneck (longest unit).
    """

    unit_name: str
    start_cycle: int
    end_cycle: int
    unit_cycles: int
    is_bottleneck: bool = False


@dataclass
class StageSchedule:
    """Scheduled placement of a single stage on the timeline.

    Attributes
    ----------
    stage_name : str
    start_cycle : int
        Absolute cycle when the stage starts.
    end_cycle : int
        Absolute cycle when the stage ends (start + bottleneck cycles).
    stage_cycles : int
        Bottleneck duration of this stage.
    unit_intervals : dict[str, UnitInterval]
        Per-unit intervals keyed by unit name.
    sim_result : SimResult
        Raw simulation result from ``CycleSimulator``.
    """

    stage_name: str
    start_cycle: int
    end_cycle: int
    stage_cycles: int
    unit_intervals: dict[str, UnitInterval] = field(default_factory=dict)
    sim_result: SimResult | None = None


@dataclass
class TimelineResult:
    """Complete scheduled timeline for a pipeline.

    Attributes
    ----------
    pipeline_name : str
    hardware_name : str
    schedules : list[StageSchedule]
        One entry per stage, in topological execution order.
    total_cycles : int
        End cycle of the last stage to finish.
    total_time_us : float
        Total wall-clock time in microseconds.
    """

    pipeline_name: str
    hardware_name: str
    schedules: list[StageSchedule] = field(default_factory=list)
    total_cycles: int = 0
    total_time_us: float = 0.0

    # Convenience lookup: stage_name -> StageSchedule
    def get(self, stage_name: str) -> StageSchedule | None:
        for s in self.schedules:
            if s.stage_name == stage_name:
                return s
        return None


# ---------------------------------------------------------------------------
# TimelineSimulator
# ---------------------------------------------------------------------------

class TimelineSimulator:
    """Schedule a ``Pipeline`` onto hardware and produce a ``TimelineResult``.

    Parameters
    ----------
    hw : GPUConfig
        Target hardware model.

    Examples
    --------
    ::

        sim = TimelineSimulator(b200)
        result = sim.simulate(pipeline)
        plot_timeline(result)
    """

    def __init__(self, hw: GPUConfig) -> None:
        self.hw = hw
        self._inner = CycleSimulator(hw)

    def simulate(self, pipeline: Pipeline) -> TimelineResult:
        """Schedule all stages respecting dependencies.

        Parameters
        ----------
        pipeline : Pipeline
            Must have a valid (acyclic) dependency graph.

        Returns
        -------
        TimelineResult
        """
        ordered = pipeline.topo_order()   # raises ValueError on cycles

        # stage_name -> absolute end_cycle (used to compute successors' starts)
        finish: dict[str, int] = {}
        schedules: list[StageSchedule] = []

        for stage in ordered:
            # Start as soon as all predecessors have finished
            start = max((finish[dep] for dep in stage.depends_on), default=0)

            # Simulate the stage workload
            sim_result = self._run_stage(stage)

            # Map per-unit cycle counts to absolute [start, end] intervals
            all_unit_results = {
                **sim_result.compute_results,
                **sim_result.memory_results,
            }
            unit_intervals: dict[str, UnitInterval] = {}
            for unit_name, ur in all_unit_results.items():
                unit_intervals[unit_name] = UnitInterval(
                    unit_name=unit_name,
                    start_cycle=start,
                    end_cycle=start + ur.cycles,
                    unit_cycles=ur.cycles,
                    is_bottleneck=ur.is_bottleneck,
                )

            stage_cycles = sim_result.total_cycles
            end = start + stage_cycles
            finish[stage.name] = end

            schedules.append(StageSchedule(
                stage_name=stage.name,
                start_cycle=start,
                end_cycle=end,
                stage_cycles=stage_cycles,
                unit_intervals=unit_intervals,
                sim_result=sim_result,
            ))

        total_cycles = max((s.end_cycle for s in schedules), default=0)

        return TimelineResult(
            pipeline_name=pipeline.name,
            hardware_name=self.hw.name,
            schedules=schedules,
            total_cycles=total_cycles,
            total_time_us=self.hw.cycles_to_us(total_cycles),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_stage(self, stage: Stage) -> SimResult:
        wl = stage.workload
        if isinstance(wl, TiledWorkload):
            return self._inner.simulate_tiled(wl)
        return self._inner.simulate(wl)
