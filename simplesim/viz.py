"""
viz.py — Timeline and utilization visualization for SimpleSim.

Functions
---------
``plot_timeline(result, *, time_unit, figsize, title)``
    Gantt-style chart: hardware units on Y-axis, time on X-axis.
    Each stage is a distinct color; solid bars show active unit work,
    faded bars show slack (unit idle while stage still "in flight").

``plot_utilization(result, *, figsize, title)``
    Stacked horizontal-bar chart: fraction of total pipeline time each
    unit spends active, broken down by stage.

Both functions return a ``matplotlib.figure.Figure`` so callers can
save, display, or embed the chart as they wish.

Dependencies
------------
Only ``matplotlib`` is required (no seaborn / plotly).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from .timeline_sim import TimelineResult


# ---------------------------------------------------------------------------
# Color palette (color-blind-friendly, up to 10 stages)
# ---------------------------------------------------------------------------

_STAGE_COLORS = [
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
    "#937860",  # brown
    "#DA8BC3",  # pink
    "#8C8C8C",  # grey
    "#CCB974",  # yellow
    "#64B5CD",  # cyan
]

# Preferred display order for hardware units (bottom → top in chart)
_UNIT_ORDER = [
    "hbm",
    "l2_cache",
    "shared_memory",
    "cuda_core",
    "sfu",
    "tensor_core",
]


def _unit_display_order(unit_names: list[str]) -> list[str]:
    """Sort unit names by preferred display order; unknowns go to top."""
    known = [u for u in _UNIT_ORDER if u in unit_names]
    unknown = sorted(u for u in unit_names if u not in _UNIT_ORDER)
    return known + unknown


def _cycles_to_x(cycles: int, clock_ghz: float, time_unit: str) -> float:
    """Convert a cycle count to the requested time unit."""
    if time_unit == "cycles":
        return float(cycles)
    # Convert to µs
    return cycles / (clock_ghz * 1e3)


# ---------------------------------------------------------------------------
# plot_timeline
# ---------------------------------------------------------------------------

def plot_timeline(
    result: TimelineResult,
    *,
    time_unit: str = "us",
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    clock_ghz: float | None = None,
    max_preview_iters: int = 8,
) -> "Figure":
    """Gantt-style hardware-unit occupancy chart.

    Parameters
    ----------
    result : TimelineResult
        Output from ``TimelineSimulator.simulate()``.
    time_unit : str
        ``"us"`` (default) or ``"cycles"`` — sets the X-axis unit.
    figsize : tuple[float, float] | None
        Matplotlib figure size in inches.  Auto-sized if None.
    title : str | None
        Custom plot title.  Defaults to the pipeline name.
    clock_ghz : float | None
        Clock speed for cycle→µs conversion.  If None the function tries
        to infer it from ``result.total_time_us`` and ``result.total_cycles``.
    max_preview_iters : int
        Maximum number of repeated iterations to draw per stage group.
        If a stage repeats more than this, only the first
        ``max_preview_iters - 1`` and the last iteration are rendered;
        the rest are annotated as "⋯ ×N total".

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # --- Resolve clock for unit conversion ---
    if clock_ghz is None and result.total_cycles > 0:
        clock_ghz = result.total_cycles / (result.total_time_us * 1e3)
    if clock_ghz is None:
        clock_ghz = 1.0   # fallback — won't matter if time_unit="cycles"

    def to_x(cycles: int) -> float:
        return _cycles_to_x(cycles, clock_ghz, time_unit)

    # --- Collect all unit names across all stages ---
    all_units: set[str] = set()
    for sched in result.schedules:
        all_units.update(sched.unit_intervals.keys())

    units = _unit_display_order(list(all_units))  # bottom → top
    n_units = len(units)
    unit_y = {u: i for i, u in enumerate(units)}

    # --- Build stage groups (for repeated stages, group by base name) ---
    # group_name -> list of schedules in iteration order
    seen_groups: list[str] = []
    groups: dict[str, list] = {}
    for sched in result.schedules:
        # Base name strips "[i]" suffix from repeated stages
        base = sched.stage_name.rsplit("[", 1)[0] if sched.total_iters > 1 else sched.stage_name
        if base not in groups:
            groups[base] = []
            seen_groups.append(base)
        groups[base].append(sched)

    # Assign a stable color to each group
    group_color = {
        name: _STAGE_COLORS[i % len(_STAGE_COLORS)]
        for i, name in enumerate(seen_groups)
    }

    # --- Figure layout ---
    if figsize is None:
        w = max(12.0, to_x(result.total_cycles) * 0.04 + 8)
        h = max(4.0, n_units * 0.75 + 2.0)
        figsize = (min(w, 26), h)

    fig, ax = plt.subplots(figsize=figsize)
    bar_height = 0.55
    slack_alpha = 0.15

    legend_patches: list[mpatches.Patch] = []

    for group_name, g_scheds in groups.items():
        color = group_color[group_name]
        N = g_scheds[0].total_iters

        # Decide which iterations to render
        if N <= max_preview_iters:
            render_iters = list(range(N))
            ellipsis_after = None
        else:
            # Show first (max_preview_iters-1) + last
            render_iters = list(range(max_preview_iters - 1)) + [N - 1]
            ellipsis_after = max_preview_iters - 2   # insert annotation after this index

        for render_idx, iter_i in enumerate(render_iters):
            sched = g_scheds[iter_i]
            # Alpha: full for first iter, slightly faded for later ones
            alpha_bar = max(0.35, 1.0 - iter_i / max(N - 1, 1) * 0.55)
            stage_end_x = to_x(sched.end_cycle)

            for unit in units:
                y = unit_y[unit]

                if unit in sched.unit_intervals:
                    iv = sched.unit_intervals[unit]
                    work_start_x = to_x(iv.start_cycle)
                    work_end_x   = to_x(iv.end_cycle)
                    work_width   = work_end_x - work_start_x

                    # Active work bar
                    ax.barh(
                        y, work_width,
                        left=work_start_x,
                        height=bar_height,
                        color=color,
                        alpha=alpha_bar,
                        edgecolor="white",
                        linewidth=0.4,
                        zorder=3,
                    )

                    # Slack bar (unit finishes before compute bottleneck)
                    slack_width = stage_end_x - work_end_x
                    if slack_width > 0:
                        ax.barh(
                            y, slack_width,
                            left=work_end_x,
                            height=bar_height,
                            color=color,
                            alpha=slack_alpha,
                            edgecolor="none",
                            zorder=2,
                        )

                    # Bottleneck marker (only on first iteration to avoid clutter)
                    if iv.is_bottleneck and iter_i == 0:
                        mid_x = (work_start_x + work_end_x) / 2
                        ax.annotate(
                            "▼",
                            xy=(mid_x, y + bar_height / 2),
                            ha="center", va="bottom",
                            fontsize=7, color="black", zorder=4,
                        )

            # Ellipsis annotation between skipped iters
            if ellipsis_after is not None and render_idx == ellipsis_after:
                # Place "⋯ ×N" label in the gap between last shown and final iter
                gap_start_x = to_x(g_scheds[render_idx].end_cycle)
                gap_end_x   = to_x(g_scheds[N - 1].start_cycle)
                mid_x = (gap_start_x + gap_end_x) / 2
                mid_y = max(unit_y.values()) / 2
                ax.text(
                    mid_x, mid_y,
                    f"⋯ ×{N} total",
                    ha="center", va="center",
                    fontsize=9, color=color,
                    fontstyle="italic", zorder=5,
                )

        # Legend: for repeated stages include the repeat count
        label = f"{group_name} ×{N}" if N > 1 else group_name
        legend_patches.append(mpatches.Patch(color=color, label=label))

    # --- Axes formatting ---
    ax.set_yticks(range(n_units))
    ax.set_yticklabels(units, fontsize=10)
    ax.set_ylim(-0.6, n_units - 0.3)

    xlabel = "Time (µs)" if time_unit == "us" else "Cycles"
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Hardware Unit", fontsize=11)

    plot_title = title or f"Pipeline Timeline — {result.pipeline_name} on {result.hardware_name}"
    ax.set_title(plot_title, fontsize=12, fontweight="bold")

    # Total time annotation
    total_x = to_x(result.total_cycles)
    ax.axvline(total_x, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(
        total_x, n_units - 0.1,
        f" {result.total_time_us:.2f} µs total",
        va="top", ha="left", fontsize=9, color="black", alpha=0.7,
    )

    ax.legend(
        handles=legend_patches,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        fontsize=9,
        title="Stages",
        title_fontsize=9,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_utilization
# ---------------------------------------------------------------------------

def plot_utilization(
    result: TimelineResult,
    *,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> "Figure":
    """Stacked horizontal bar chart of per-unit active fraction by stage.

    For each hardware unit the chart shows what fraction of the total
    pipeline time the unit is actively working, split by stage.

    Parameters
    ----------
    result : TimelineResult
    figsize : tuple[float, float] | None
    title : str | None

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    all_units: set[str] = set()
    for sched in result.schedules:
        all_units.update(sched.unit_intervals.keys())

    units = _unit_display_order(list(all_units))
    n_units = len(units)
    total = result.total_cycles or 1

    if figsize is None:
        figsize = (9, max(3.5, n_units * 0.65 + 1.5))

    fig, ax = plt.subplots(figsize=figsize)
    bar_height = 0.55

    left_offset = {u: 0.0 for u in units}

    for idx, sched in enumerate(result.schedules):
        color = _STAGE_COLORS[idx % len(_STAGE_COLORS)]
        for unit in units:
            y = units.index(unit)
            if unit in sched.unit_intervals:
                frac = sched.unit_intervals[unit].unit_cycles / total
            else:
                frac = 0.0
            if frac > 0:
                ax.barh(
                    y, frac,
                    left=left_offset[unit],
                    height=bar_height,
                    color=color,
                    edgecolor="white",
                    linewidth=0.5,
                )
                left_offset[unit] += frac

    ax.set_yticks(range(n_units))
    ax.set_yticklabels(units, fontsize=10)
    ax.set_ylim(-0.6, n_units - 0.3)
    ax.set_xlabel("Fraction of total pipeline time", fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}")
    )

    plot_title = title or f"Unit Utilization — {result.pipeline_name} on {result.hardware_name}"
    ax.set_title(plot_title, fontsize=12, fontweight="bold")

    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color=_STAGE_COLORS[i % len(_STAGE_COLORS)], label=sched.stage_name)
        for i, sched in enumerate(result.schedules)
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        fontsize=9,
        title="Stages",
        title_fontsize=9,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return fig
