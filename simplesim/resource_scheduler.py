"""
resource_scheduler.py — Resource-constrained pipeline scheduler.

Unlike ``TimelineSimulator`` which requires users to manually encode
resource conflicts as ``depends_on`` edges, ``ResourceScheduler``
automatically detects which hardware resources each stage uses and
prevents illegal overlaps.

Resource constraint model
-------------------------
* **Exclusive compute units** (tensor_core, sfu, cuda_core):
  Only one stage may use a given compute unit at a time.  Two stages
  that use *different* compute units (e.g., QKT uses tensor_core while
  Softmax uses sfu+cuda_core) can run in parallel.

* **SMEM capacity**:
  The sum of all concurrently-running stages' SMEM footprints must not
  exceed the hardware capacity.  This models the physical SMEM limit.

* **HBM bandwidth**:
  HBM DMA can overlap with compute (async copy engines), but only one
  HBM transfer runs at a time (serial DMA).

Scheduling algorithm
--------------------
List scheduling with resource-availability checks:

1. Topologically sort stages by data dependencies.
2. Maintain a ready queue of stages whose data dependencies are met.
3. For each ready stage, find the earliest start time where:
   - All data predecessors have finished.
   - All required compute units are free.
   - SMEM capacity is sufficient.
4. Schedule the stage and update resource occupation timelines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Union

from .hardware import GPUConfig
from .pipeline import Pipeline, Stage
from .simulator import CycleSimulator, SimResult
from .timeline_sim import StageSchedule, TimelineResult, UnitInterval
from .workload import TiledWorkload, Workload


# ---------------------------------------------------------------------------
# Resource profile: auto-derived from a workload
# ---------------------------------------------------------------------------

@dataclass
class ResourceProfile:
    """Hardware resource usage of a single stage, auto-derived from its workload.

    Attributes
    ----------
    compute_units : set[str]
        Names of compute units this stage occupies (e.g. {"tensor_core"}).
    smem_capacity_bytes : int
        SMEM footprint while this stage runs.
    uses_hbm : bool
        Whether this stage performs HBM transfers.
    """

    compute_units: set[str] = field(default_factory=set)
    smem_capacity_bytes: int = 0
    uses_hbm: bool = False


def extract_resource_profile(stage: Stage) -> ResourceProfile:
    """Derive resource usage from a stage's workload fields.

    For ``TiledWorkload``, tensor_core is used if ``mma_ops`` is non-empty,
    and other compute units are inferred from ``elementwise_ops``.

    For ``Workload``, compute units are inferred from ``compute_ops``,
    and SMEM usage from ``memory_bytes["shared_memory"]`` or
    ``smem_capacity_bytes``.

    SMEM capacity is resolved in priority order:
    1. Explicit ``smem_capacity_bytes`` on the workload
    2. ``auto_smem_capacity()`` (sum of SMEM-located DataTensors)
    3. ``memory_bytes["shared_memory"]`` (bandwidth bytes as fallback)
    """
    wl = stage.workload
    profile = ResourceProfile()

    if isinstance(wl, TiledWorkload):
        if wl.mma_ops:
            profile.compute_units.add("tensor_core")
        for unit_name, ops in wl.elementwise_ops.items():
            if ops > 0:
                profile.compute_units.add(unit_name)
        if wl.hbm_bytes > 0:
            profile.uses_hbm = True
        # SMEM capacity
        cap = wl.auto_smem_capacity()
        if cap == 0:
            # Fallback: use total SMEM bandwidth bytes as rough capacity
            cap = wl.total_smem_bytes()
        profile.smem_capacity_bytes = cap

    elif isinstance(wl, Workload):
        for unit_name, ops in wl.compute_ops.items():
            if ops > 0:
                profile.compute_units.add(unit_name)
        if wl.memory_bytes.get("hbm", 0) > 0:
            profile.uses_hbm = True
        # SMEM capacity
        cap = wl.auto_smem_capacity()
        if cap == 0:
            cap = wl.memory_bytes.get("shared_memory", 0)
        profile.smem_capacity_bytes = cap

    # Remove "shared_memory" from compute_units — it's tracked separately
    # via smem_capacity_bytes (shared_memory is a memory level, not a CU)
    profile.compute_units.discard("shared_memory")

    return profile


# ---------------------------------------------------------------------------
# Resource occupation timeline
# ---------------------------------------------------------------------------

class _IntervalTracker:
    """Tracks non-overlapping occupied intervals for an exclusive resource.

    Stores sorted list of (start, end) intervals.  Supports querying the
    earliest time >= a given minimum where a new interval of given duration
    can be placed without overlap.
    """

    def __init__(self) -> None:
        self._intervals: list[tuple[int, int]] = []

    def earliest_free(self, min_start: int, duration: int) -> int:
        """Find earliest start >= min_start where duration fits."""
        candidate = min_start
        for s, e in self._intervals:
            if candidate + duration <= s:
                return candidate
            if candidate < e:
                candidate = e
        return candidate

    def occupy(self, start: int, end: int) -> None:
        """Mark [start, end) as occupied.  Assumes no overlap."""
        # Insert in sorted order
        idx = 0
        for i, (s, _e) in enumerate(self._intervals):
            if start < s:
                break
            idx = i + 1
        self._intervals.insert(idx, (start, end))


class _SmemCapacityTracker:
    """Tracks SMEM capacity usage over time.

    Maintains a list of (start, end, bytes) intervals representing
    concurrent SMEM allocations.  Supports querying the earliest time
    where adding a new allocation would not exceed capacity.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._allocs: list[tuple[int, int, int]] = []  # (start, end, bytes)

    def usage_at(self, t: int) -> int:
        """Total SMEM bytes in use at cycle t."""
        return sum(b for s, e, b in self._allocs if s <= t < e)

    def max_usage_in_range(self, start: int, end: int) -> int:
        """Maximum SMEM usage at any point in [start, end)."""
        # Collect all boundary points within [start, end)
        points = {start}
        for s, e, _b in self._allocs:
            if start < s < end:
                points.add(s)
            if start < e < end:
                points.add(e)
        return max(self.usage_at(t) for t in points) if points else 0

    def can_fit(self, start: int, end: int, new_bytes: int) -> bool:
        """Check if adding new_bytes in [start, end) stays within capacity."""
        if new_bytes == 0:
            return True
        # Check all boundary points
        points = {start}
        for s, e, _b in self._allocs:
            if start <= s < end:
                points.add(s)
            if start < e <= end:
                points.add(e)
        for t in points:
            if self.usage_at(t) + new_bytes > self._capacity:
                return False
        return True

    def earliest_fit(self, min_start: int, duration: int, new_bytes: int) -> int:
        """Find earliest start >= min_start where new_bytes fits for duration."""
        if new_bytes == 0 or self._capacity == 0:
            return min_start

        # Collect all change points from existing allocations
        change_points = sorted({min_start} | {
            t for s, e, _b in self._allocs
            for t in (s, e)
            if t >= min_start
        })

        for cp in change_points:
            candidate = max(cp, min_start)
            if self.can_fit(candidate, candidate + duration, new_bytes):
                return candidate

        # After all existing allocations end, it must fit
        if self._allocs:
            last_end = max(e for _, e, _ in self._allocs)
            candidate = max(last_end, min_start)
            return candidate
        return min_start

    def allocate(self, start: int, end: int, nbytes: int) -> None:
        """Record a SMEM allocation for [start, end)."""
        if nbytes > 0:
            self._allocs.append((start, end, nbytes))


# ---------------------------------------------------------------------------
# ResourceScheduler
# ---------------------------------------------------------------------------

class ResourceScheduler:
    """Resource-constrained pipeline scheduler.

    Unlike ``TimelineSimulator`` which trusts user-specified ``depends_on``
    to encode resource conflicts, this scheduler automatically detects
    resource usage from workload fields and prevents illegal overlaps.

    Parameters
    ----------
    hw : GPUConfig
        Target hardware model.

    Examples
    --------
    ::

        from simplesim import Pipeline, Stage, load_gpu_config
        from simplesim.resource_scheduler import ResourceScheduler

        hw = load_gpu_config("configs/b200.yaml")
        pipe = Pipeline("fa4", stages=[
            Stage("QKT",     workload=qkt_wl),
            Stage("Softmax", workload=sm_wl, depends_on=["QKT"]),
            Stage("PV",      workload=pv_wl, depends_on=["Softmax"]),
        ])
        sched = ResourceScheduler(hw)
        result = sched.schedule(pipe)
    """

    def __init__(self, hw: GPUConfig) -> None:
        self.hw = hw
        self._inner = CycleSimulator(hw)

    def schedule(self, pipeline: Pipeline) -> TimelineResult:
        """Schedule all stages with automatic resource constraint detection.

        Parameters
        ----------
        pipeline : Pipeline
            Must have a valid (acyclic) dependency graph.  ``depends_on``
            should only encode **data dependencies** — resource conflicts
            are detected automatically.

        Returns
        -------
        TimelineResult
            Same format as ``TimelineSimulator.simulate()``.
        """
        ordered = pipeline.topo_order()

        # Pre-compute resource profiles and sim results
        profiles: dict[str, ResourceProfile] = {}
        sim_results: dict[str, SimResult] = {}
        for stage in ordered:
            profiles[stage.name] = extract_resource_profile(stage)
            sim_results[stage.name] = self._run_stage(stage)

        # Resource trackers
        compute_trackers: dict[str, _IntervalTracker] = {}
        for stage in ordered:
            for cu in profiles[stage.name].compute_units:
                if cu not in compute_trackers:
                    compute_trackers[cu] = _IntervalTracker()

        hbm_tracker = _IntervalTracker()

        smem_cap = 0
        smem_ml = self.hw.memory_levels.get("shared_memory")
        if smem_ml and smem_ml.capacity_bytes:
            smem_cap = smem_ml.capacity_bytes
        smem_tracker = _SmemCapacityTracker(smem_cap)

        # Schedule
        finish: dict[str, int] = {}
        schedules: list[StageSchedule] = []

        for stage in ordered:
            sr = sim_results[stage.name]
            prof = profiles[stage.name]

            # Data dependency: earliest start from predecessors
            dep_ready = max((finish[dep] for dep in stage.depends_on), default=0)

            # Compute exclusive resource constraints
            compute_ready = dep_ready
            stage_duration = sr.total_cycles
            for cu_name in prof.compute_units:
                tracker = compute_trackers[cu_name]
                cu_start = tracker.earliest_free(dep_ready, stage_duration)
                compute_ready = max(compute_ready, cu_start)

            # HBM constraint (serial DMA)
            hbm_ready = compute_ready
            if prof.uses_hbm:
                hbm_ready = hbm_tracker.earliest_free(
                    dep_ready, stage_duration
                )

            # Start must satisfy all constraints
            candidate = max(compute_ready, hbm_ready)

            # SMEM capacity constraint
            if prof.smem_capacity_bytes > 0 and smem_cap > 0:
                candidate = smem_tracker.earliest_fit(
                    candidate, stage_duration, prof.smem_capacity_bytes
                )
                # Re-check compute constraints at new candidate
                for cu_name in prof.compute_units:
                    tracker = compute_trackers[cu_name]
                    cu_start = tracker.earliest_free(candidate, stage_duration)
                    if cu_start > candidate:
                        # Iterate: find consensus (simple convergence)
                        candidate = cu_start
                        candidate = smem_tracker.earliest_fit(
                            candidate, stage_duration,
                            prof.smem_capacity_bytes,
                        )

            start = candidate
            end = start + stage_duration

            # Record occupations
            for cu_name in prof.compute_units:
                compute_trackers[cu_name].occupy(start, end)
            if prof.uses_hbm:
                hbm_tracker.occupy(start, end)
            if prof.smem_capacity_bytes > 0:
                smem_tracker.allocate(start, end, prof.smem_capacity_bytes)

            # Build unit intervals
            all_units = {**sr.compute_results, **sr.memory_results}
            unit_intervals: dict[str, UnitInterval] = {}
            for unit_name, ur in all_units.items():
                unit_intervals[unit_name] = UnitInterval(
                    unit_name=unit_name,
                    start_cycle=start,
                    end_cycle=start + ur.cycles,
                    unit_cycles=ur.cycles,
                    is_bottleneck=ur.is_bottleneck,
                )

            finish[stage.name] = end
            schedules.append(StageSchedule(
                stage_name=stage.name,
                start_cycle=start,
                end_cycle=end,
                stage_cycles=stage_duration,
                unit_intervals=unit_intervals,
                sim_result=sr,
                iteration=0,
                total_iters=1,
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
    # Convenience: access profiles for debugging / visualization
    # ------------------------------------------------------------------

    def get_profiles(self, pipeline: Pipeline) -> dict[str, ResourceProfile]:
        """Return resource profiles for all stages (useful for debugging)."""
        return {
            stage.name: extract_resource_profile(stage)
            for stage in pipeline.stages
        }

    def get_smem_timeline(
        self, result: TimelineResult, pipeline: Pipeline,
    ) -> list[tuple[int, int]]:
        """Return SMEM capacity usage over time as (cycle, bytes) steps.

        Useful for plotting the SMEM capacity track in visualizations.
        """
        profiles = self.get_profiles(pipeline)

        # Collect all change points
        events: list[tuple[int, int]] = []  # (cycle, delta_bytes)
        for sched in result.schedules:
            base_name = sched.stage_name
            # Handle [i] suffix from manual expansion
            prof = profiles.get(base_name)
            if prof is None:
                # Try stripping [N] suffix
                import re
                stripped = re.sub(r'\[\d+\]$', '', base_name)
                prof = profiles.get(stripped)
            if prof and prof.smem_capacity_bytes > 0:
                events.append((sched.start_cycle, prof.smem_capacity_bytes))
                events.append((sched.end_cycle, -prof.smem_capacity_bytes))

        if not events:
            return [(0, 0)]

        events.sort(key=lambda x: (x[0], x[1]))

        # Build step function
        steps: list[tuple[int, int]] = []
        current = 0
        for cycle, delta in events:
            current += delta
            steps.append((cycle, current))

        return steps

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_stage(self, stage: Stage) -> SimResult:
        wl = stage.workload
        if isinstance(wl, TiledWorkload):
            return self._inner.simulate_tiled(wl)
        return self._inner.simulate(wl)
