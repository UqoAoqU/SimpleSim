"""
pipeline.py — Multi-stage operator pipeline with dependency graph.

Abstractions
------------
``Stage``
    A single operator step: wraps any ``Workload`` or ``TiledWorkload`` and
    declares which earlier stages it depends on.  This is the primary unit
    users compose to describe a kernel pipeline.

``Pipeline``
    An ordered collection of ``Stage`` objects.  Provides topological
    scheduling order and validates that the dependency graph is acyclic.

Usage example
-------------
::

    from simplesim import Stage, Pipeline, Workload, TiledWorkload

    load = Stage("load_qkv",  workload=load_wl)
    fwd  = Stage("forward",   workload=fwd_wl,  depends_on=["load_qkv"])
    bwd  = Stage("backward",  workload=bwd_wl,  depends_on=["forward"])

    pipe = Pipeline("fa4", stages=[load, fwd, bwd])

The pipeline can then be handed to ``TimelineSimulator.simulate(pipe)`` to
produce a ``TimelineResult`` suitable for ``plot_timeline()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from .workload import Workload, TiledWorkload

AnyWorkload = Union[Workload, TiledWorkload]


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

@dataclass
class Stage:
    """A single operator step in a pipeline.

    Parameters
    ----------
    name : str
        Unique identifier within the pipeline.  Used to reference this stage
        in ``depends_on`` lists of later stages.
    workload : Workload | TiledWorkload
        Resource demands for this step.  Both basic and MMA-aware workloads
        are accepted; the simulator picks the appropriate simulation path.
    depends_on : list[str]
        Names of stages that must complete before this one can start.
        An empty list (default) means the stage can start immediately.
    """

    name: str
    workload: AnyWorkload
    depends_on: list[str] = field(default_factory=list)
    repeat: int = 1
    """Number of times this stage repeats (e.g. KV-tile loop count).

    When ``repeat > 1`` the ``TimelineSimulator`` applies hardware-aware
    pipelining between iterations:

    * **Memory units** (``hbm``, ``l2_cache``, ``shared_memory``) —
      iteration *k+1* can begin as soon as the same unit finishes iteration
      *k*, independent of the compute schedule.  This models async-copy /
      double-buffering.  A SMEM capacity check ensures two tiles fit
      simultaneously; if not, SMEM falls back to serial scheduling.

    * **Compute units** (``tensor_core``, ``sfu``, ``cuda_core``) —
      iteration *k+1* starts only when **all** compute units from iteration
      *k* have finished (exclusive use).
    """


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """An ordered collection of stages with a dependency graph.

    Parameters
    ----------
    name : str
        Human-readable label for the pipeline.
    stages : list[Stage] | None
        Initial stages.  More can be added via ``add_stage()``.

    Raises
    ------
    ValueError
        On duplicate stage names or cyclic dependencies (detected lazily when
        ``topo_order()`` is called).
    """

    def __init__(self, name: str, stages: list[Stage] | None = None) -> None:
        self.name = name
        self._stages: dict[str, Stage] = {}   # insertion-ordered (Python 3.7+)
        for s in (stages or []):
            self.add_stage(s)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_stage(self, stage: Stage) -> None:
        """Append a stage to the pipeline.

        Parameters
        ----------
        stage : Stage

        Raises
        ------
        ValueError
            If a stage with the same name already exists.
        """
        if stage.name in self._stages:
            raise ValueError(f"Duplicate stage name: '{stage.name}'")
        self._stages[stage.name] = stage

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def stages(self) -> list[Stage]:
        """All stages in insertion order."""
        return list(self._stages.values())

    def topo_order(self) -> list[Stage]:
        """Return stages in a valid topological execution order.

        Uses Kahn's algorithm (BFS).  Insertion order is used as a
        tie-breaker so the result is deterministic.

        Returns
        -------
        list[Stage]
            Stages sorted so that every stage appears after all its
            dependencies.

        Raises
        ------
        ValueError
            If an unknown dependency name is referenced or the graph
            contains a cycle.
        """
        # Validate all dependency references
        for stage in self._stages.values():
            for dep in stage.depends_on:
                if dep not in self._stages:
                    raise ValueError(
                        f"Stage '{stage.name}' depends on unknown stage '{dep}'"
                    )

        # Build in-degree and adjacency list
        in_degree: dict[str, int] = {name: 0 for name in self._stages}
        successors: dict[str, list[str]] = {name: [] for name in self._stages}

        for stage in self._stages.values():
            for dep in stage.depends_on:
                in_degree[stage.name] += 1
                successors[dep].append(stage.name)

        # Kahn's BFS — use insertion order for the ready queue
        from collections import deque
        ready: deque[str] = deque(
            name for name in self._stages if in_degree[name] == 0
        )
        order: list[Stage] = []

        while ready:
            name = ready.popleft()
            order.append(self._stages[name])
            for succ in successors[name]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    ready.append(succ)

        if len(order) != len(self._stages):
            # Find which nodes remain (they form cycles)
            processed = {s.name for s in order}
            remaining = [n for n in self._stages if n not in processed]
            raise ValueError(
                f"Cyclic dependency detected among stages: {remaining}"
            )

        return order

    def __repr__(self) -> str:
        names = list(self._stages.keys())
        return f"Pipeline(name={self.name!r}, stages={names})"
