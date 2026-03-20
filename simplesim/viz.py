"""
viz.py — Timeline and utilization visualization for SimpleSim.

Design goals (inspired by FA-4 paper pipeline diagram)
-------------------------------------------------------
* Each active bar carries a **text label** showing the stage/operation name.
* Idle periods are shown as **dashed outlines** so every unit row has a
  visible background spanning the full pipeline time.
* **Vertical dashed lines** at every stage boundary help align operations
  across rows, making pipeline overlaps immediately legible.
* **Horizontal row separators** keep unit rows visually distinct.
* Colors are assigned per *stage group* (all iterations of the same stage
  share one color, fading slightly for later iterations).

Functions
---------
``plot_timeline``   — main pipeline Gantt diagram (reference-style)
``plot_utilization`` — stacked utilization bar chart (summary view)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from .timeline_sim import TimelineResult


# ---------------------------------------------------------------------------
# Color palette  (distinguishable on screen and in print)
# ---------------------------------------------------------------------------

_STAGE_COLORS = [
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
    "#937860",  # brown
    "#DA8BC3",  # pink
    "#4E9AAF",  # teal
    "#CCB974",  # yellow-brown
    "#8C8C8C",  # grey
]

# Preferred display order for hardware-unit rows (bottom → top)
_UNIT_ORDER = [
    "hbm",
    "l2_cache",
    "shared_memory",
    "cuda_core",
    "sfu",
    "tensor_core",
]

# Human-readable unit labels for Y-axis
_UNIT_LABELS = {
    "tensor_core":    "Tensor Core",
    "sfu":            "SFU (exp/rcp)",
    "cuda_core":      "CUDA Core",
    "shared_memory":  "Shared Mem",
    "l2_cache":       "L2 Cache",
    "hbm":            "HBM",
}


def _unit_display_order(unit_names: list[str]) -> list[str]:
    known   = [u for u in _UNIT_ORDER   if u in unit_names]
    unknown = sorted(u for u in unit_names if u not in _UNIT_ORDER)
    return known + unknown


def _to_x(cycles: int, clock_ghz: float, time_unit: str) -> float:
    if time_unit == "cycles":
        return float(cycles)
    return cycles / (clock_ghz * 1e3)   # → µs


def _resolve_clock(result: TimelineResult, clock_ghz: float | None) -> float:
    if clock_ghz is not None:
        return clock_ghz
    if result.total_cycles > 0 and result.total_time_us > 0:
        return result.total_cycles / (result.total_time_us * 1e3)
    return 1.0


_ITER_SUFFIX_RE = re.compile(r'\[\d+\]$')


def _build_groups(
    result: TimelineResult,
) -> tuple[list[str], dict[str, list]]:
    """Return (seen_groups, groups) where groups maps base_name → list[StageSchedule].

    Groups by stripping trailing ``[N]`` suffixes from stage names, so both
    ``Stage.repeat``-generated names (e.g. ``kv_tile[0]``) and manually
    expanded names (e.g. ``QKT[0]``, ``QKT[1]``) are handled uniformly.
    """
    seen: list[str] = []
    groups: dict[str, list] = {}
    for sched in result.schedules:
        base = _ITER_SUFFIX_RE.sub("", sched.stage_name) or sched.stage_name
        if base not in groups:
            groups[base] = []
            seen.append(base)
        groups[base].append(sched)
    return seen, groups


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
    """Pipeline schedule diagram.

    Layout
    ------
    * **Y-axis** — one row per hardware unit.
    * **X-axis** — time (µs or cycles).
    * **Solid colored box** — unit is active; the stage name is printed inside.
    * **Faded box** — unit is idle but the stage's bottleneck is still running
      (slack time).
    * **Dashed outline** — unit has no work in this time window (true idle).
    * **Vertical dotted lines** — stage boundaries for easy cross-row alignment.

    Parameters
    ----------
    result : TimelineResult
    time_unit : "us" | "cycles"
    figsize : optional (width, height) in inches
    title : custom title
    clock_ghz : override clock speed for cycle→µs conversion
    max_preview_iters : max iterations drawn per repeated stage group
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    ghz   = _resolve_clock(result, clock_ghz)
    to_x  = lambda c: _to_x(c, ghz, time_unit)

    # ── Geometry ────────────────────────────────────────────────────────────
    all_units: set[str] = set()
    for s in result.schedules:
        all_units.update(s.unit_intervals.keys())
    units   = _unit_display_order(list(all_units))
    n_units = len(units)
    unit_y  = {u: i for i, u in enumerate(units)}

    seen_groups, groups = _build_groups(result)
    group_color = {
        name: _STAGE_COLORS[i % len(_STAGE_COLORS)]
        for i, name in enumerate(seen_groups)
    }

    total_x = to_x(result.total_cycles)

    if figsize is None:
        w = max(14.0, total_x * 0.008 + 6)
        h = max(3.5, n_units * 1.05 + 1.8)
        figsize = (min(w, 30), h)

    fig, ax = plt.subplots(figsize=figsize)

    BAR_H       = 0.74   # bar height in data units
    SLACK_ALPHA = 0.12
    IDLE_COLOR  = "#BBBBBB"

    # ── Dashed idle background for every unit row ────────────────────────────
    for unit in units:
        y = unit_y[unit]
        ax.barh(
            y, total_x, left=0, height=BAR_H,
            fill=False,
            edgecolor=IDLE_COLOR, linewidth=0.9, linestyle="--",
            zorder=1,
        )

    # ── Active bars + text labels ────────────────────────────────────────────
    legend_patches: list[mpatches.Patch] = []
    stage_boundaries: set[int] = {0, result.total_cycles}

    for group_name, g_scheds in groups.items():
        color = group_color[group_name]
        # N = number of schedules in this group (works for both Stage.repeat
        # and manually-expanded [i]-suffixed stages)
        N = len(g_scheds)

        # Which schedules to render
        if N <= max_preview_iters:
            render_scheds  = g_scheds
            ellipsis_after = None
        else:
            render_scheds  = g_scheds[:max_preview_iters - 1] + [g_scheds[-1]]
            ellipsis_after = max_preview_iters - 2

        for render_idx, sched in enumerate(render_scheds):
            # iteration index: use sched.iteration when from Stage.repeat,
            # else fall back to render_idx
            iter_i = sched.iteration if sched.total_iters > 1 else render_idx

            stage_boundaries.add(sched.start_cycle)
            stage_boundaries.add(sched.end_cycle)

            bar_label    = group_name if N == 1 else f"{group_name}[{iter_i}]"
            alpha_active = max(0.42, 1.0 - render_idx / max(N - 1, 1) * 0.48)
            stage_end_x  = to_x(sched.end_cycle)

            for unit in units:
                y = unit_y[unit]

                if unit not in sched.unit_intervals:
                    continue

                iv            = sched.unit_intervals[unit]
                work_start_x  = to_x(iv.start_cycle)
                work_end_x    = to_x(iv.end_cycle)
                work_width    = work_end_x - work_start_x

                # Active bar
                ax.barh(
                    y, work_width, left=work_start_x, height=BAR_H,
                    color=color, alpha=alpha_active,
                    edgecolor="white", linewidth=0.9,
                    zorder=3,
                )

                # Slack bar (stage still running, this unit finished early)
                slack_w = stage_end_x - work_end_x
                if slack_w > 0:
                    ax.barh(
                        y, slack_w, left=work_end_x, height=BAR_H,
                        color=color, alpha=SLACK_ALPHA,
                        edgecolor=color, linewidth=0.5, linestyle=":",
                        zorder=2,
                    )

                # ── Text label inside active bar ─────────────────────────
                mid_x = (work_start_x + work_end_x) / 2
                frac  = work_width / max(total_x, 1e-9)

                if frac >= 0.045:           # wide enough: horizontal label
                    ax.text(
                        mid_x, y, bar_label,
                        ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold",
                        clip_on=True, zorder=5,
                    )
                elif frac >= 0.012:         # narrow: vertical label
                    short = bar_label[:8]
                    ax.text(
                        mid_x, y, short,
                        ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="bold",
                        rotation=90, clip_on=True, zorder=5,
                    )
                # else: too narrow for text (e.g. HBM = 16 cycles)

                # Bottleneck marker (▼ above bar, first iteration only)
                if iv.is_bottleneck and render_idx == 0:
                    ax.text(
                        mid_x, y + BAR_H / 2 + 0.03, "▼",
                        ha="center", va="bottom",
                        fontsize=7, color="#222222", zorder=6,
                    )

            # "⋯ ×N" annotation for collapsed iterations
            if ellipsis_after is not None and render_idx == ellipsis_after:
                gap_x0  = to_x(render_scheds[render_idx].end_cycle)
                gap_x1  = to_x(g_scheds[-1].start_cycle)
                ax.text(
                    (gap_x0 + gap_x1) / 2, n_units / 2,
                    f"⋯ ×{N} total",
                    ha="center", va="center",
                    fontsize=9, color=color, fontstyle="italic", zorder=6,
                )

        label = f"{group_name}  ×{N}" if N > 1 else group_name
        legend_patches.append(mpatches.Patch(color=color, label=label))

    # ── Vertical stage-boundary lines ────────────────────────────────────────
    for cyc in sorted(stage_boundaries):
        bx = to_x(cyc)
        ax.axvline(bx, color="#999999", linewidth=0.6,
                   linestyle=":", alpha=0.55, zorder=0)

    # ── Horizontal row separator lines ───────────────────────────────────────
    for y_sep in range(n_units + 1):
        ax.axhline(y_sep - 0.5, color="#DDDDDD", linewidth=0.7, zorder=0)

    # ── Axis labels & ticks ──────────────────────────────────────────────────
    ax.set_xlim(0, total_x * 1.02)
    ax.set_yticks(range(n_units))
    ax.set_yticklabels(
        [_UNIT_LABELS.get(u, u) for u in units],
        fontsize=10,
    )
    ax.set_ylim(-0.55, n_units - 0.45)

    ax.set_xlabel("Time (µs)" if time_unit == "us" else "Cycles", fontsize=11)
    ax.set_ylabel("Hardware Unit", fontsize=11)

    plot_title = (title or
                  f"Pipeline Timeline — {result.pipeline_name}"
                  f"  [{result.hardware_name}]")
    ax.set_title(plot_title, fontsize=12, fontweight="bold", pad=8)

    # Total-time marker
    ax.axvline(total_x, color="#333333", linewidth=1.2,
               linestyle="--", alpha=0.7)
    ax.text(
        total_x * 0.998, n_units - 0.52,
        f" {result.total_time_us:.3f} µs",
        va="bottom", ha="right",
        fontsize=8.5, color="#333333", alpha=0.85,
    )

    ax.legend(
        handles=legend_patches,
        loc="upper left", bbox_to_anchor=(1.01, 1),
        borderaxespad=0, fontsize=9,
        title="Stages", title_fontsize=9,
        framealpha=0.9,
    )
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
    """Stacked horizontal bar chart: fraction of total pipeline time each
    unit is active, broken down by stage group.

    Repeated stages are collapsed into a single group entry so the chart
    stays readable regardless of iteration count.
    """
    import matplotlib.pyplot as plt

    all_units: set[str] = set()
    for s in result.schedules:
        all_units.update(s.unit_intervals.keys())
    units   = _unit_display_order(list(all_units))
    n_units = len(units)
    total   = result.total_cycles or 1

    seen_groups, groups = _build_groups(result)
    group_color = {
        name: _STAGE_COLORS[i % len(_STAGE_COLORS)]
        for i, name in enumerate(seen_groups)
    }

    if figsize is None:
        figsize = (10, max(3.5, n_units * 0.75 + 1.5))

    fig, ax = plt.subplots(figsize=figsize)
    BAR_H = 0.62

    left_offset = {u: 0.0 for u in units}

    for group_name, g_scheds in groups.items():
        color = group_color[group_name]
        for unit in units:
            y    = units.index(unit)
            frac = sum(
                s.unit_intervals[unit].unit_cycles
                for s in g_scheds if unit in s.unit_intervals
            ) / total
            if frac > 0:
                bar = ax.barh(
                    y, frac, left=left_offset[unit], height=BAR_H,
                    color=color, edgecolor="white", linewidth=0.6,
                )
                # Label if wide enough
                if frac >= 0.04:
                    ax.text(
                        left_offset[unit] + frac / 2, y,
                        f"{frac:.0%}",
                        ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold",
                        clip_on=True,
                    )
                left_offset[unit] += frac

    ax.set_yticks(range(n_units))
    ax.set_yticklabels(
        [_UNIT_LABELS.get(u, u) for u in units],
        fontsize=10,
    )
    ax.set_ylim(-0.55, n_units - 0.45)
    ax.set_xlabel("Fraction of total pipeline time", fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}")
    )

    plot_title = (title or
                  f"Unit Utilization — {result.pipeline_name}"
                  f"  [{result.hardware_name}]")
    ax.set_title(plot_title, fontsize=12, fontweight="bold", pad=8)

    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color=group_color[n], label=n) for n in seen_groups
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper left", bbox_to_anchor=(1.01, 1),
        borderaxespad=0, fontsize=9,
        title="Stages", title_fontsize=9,
        framealpha=0.9,
    )
    ax.axhline(y=-0.5, color="#DDDDDD", linewidth=0.7)
    for y_sep in range(n_units):
        ax.axhline(y_sep - 0.5, color="#DDDDDD", linewidth=0.7, zorder=0)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return fig
