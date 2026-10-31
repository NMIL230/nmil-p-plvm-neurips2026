"""Visualization helpers for the paired-task foraging and cleaning simulator."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from simulation.config import (
    LABEL_HOPPER,
    LABEL_PLANNER,
    TASK_NAME_CLEANING,
    TASK_NAME_FORAGING,
)
from simulation.environments import TaskEnvironment


FORAGING_TERRAIN_COLORS = {
    0: "#e7f4c9",
    1: "#7cc6fe",
    2: "#4f772d",
    3: "#8d6e63",
    4: "#b08968",
}

CLEANING_TERRAIN_COLORS = {
    0: "#fff8ef",
    1: "#3d405b",
}

COMPARISON_HIGHLIGHT_COUNT = 12
COMPARISON_ROLE_COLORS = {
    LABEL_PLANNER: "#355070",
    LABEL_HOPPER: "#9c6644",
}
COMPARISON_FREE_TERRAIN_COLOR = "#fbfaf7"
COMPARISON_FORAGING_BLOCKED_COLOR = "#d7dce1"
COMPARISON_CLEANING_BLOCKED_COLOR = "#5b6374"
COMPARISON_TARGET_EDGE_COLOR = "#9aa3ad"
COMPARISON_HIGH_VALUE_COLOR = "#d4a72c"
COMPARISON_COLLECTED_HIGH_VALUE_COLOR = "#f3df97"
COMPARISON_ROUTE_COLOR = "#374151"
PUBLICATION_PANEL_FIGSIZE = (2.13, 2.13)
COMPARISON_FIGSIZE = (3.2, 2.22)


def plot_environment(environment: TaskEnvironment, output_path: Path) -> None:
    """Plot one task layout as a compact, simplified panel."""
    figure, axis = plt.subplots(figsize=PUBLICATION_PANEL_FIGSIZE)
    figure.patch.set_facecolor("white")
    _draw_publication_layout_axis(axis, environment)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_trajectory(
    environment: TaskEnvironment,
    trajectory_payload: Dict[str, object],
    output_path: Path,
) -> None:
    """Plot one example trajectory on top of the shared task layout."""
    figure, axis = plt.subplots(figsize=(9, 9))
    _draw_trajectory_axis(axis, environment, trajectory_payload)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_planner_vs_hopper(
    environment: TaskEnvironment,
    planner_payload: Dict[str, object],
    hopper_payload: Dict[str, object],
    output_path: Path,
    highlight_completion_count: int = COMPARISON_HIGHLIGHT_COUNT,
) -> None:
    """Plot a side-by-side Planner/Hopper comparison on the same task layout."""
    resolved_highlight_count = min(
        int(highlight_completion_count),
        len(_completion_events(planner_payload)),
        len(_completion_events(hopper_payload)),
    )
    figure, axes = plt.subplots(1, 2, figsize=COMPARISON_FIGSIZE, sharex=True, sharey=True)
    figure.patch.set_facecolor("white")

    _draw_publication_comparison_axis(
        axes[0],
        environment=environment,
        trajectory_payload=planner_payload,
        highlight_completion_count=resolved_highlight_count,
    )
    _draw_publication_comparison_axis(
        axes[1],
        environment=environment,
        trajectory_payload=hopper_payload,
        highlight_completion_count=resolved_highlight_count,
    )
    axes[0].set_title(f"Planner {planner_payload['player']['player_id']}", fontsize=5.2, pad=1.5)
    axes[1].set_title(f"Hopper {hopper_payload['player']['player_id']}", fontsize=5.2, pad=1.5)
    figure.suptitle(f"{environment.task_name.title()} Example Trajectories", fontsize=6.2, y=0.946)
    figure.text(
        0.5,
        0.898,
        (
            f"Numbered markers show the first {resolved_highlight_count} completed targets. "
            "Remaining targets are shown in gray for context."
        ),
        ha="center",
        va="center",
        fontsize=3.35,
        color="#4b5563",
    )
    figure.legend(
        handles=_comparison_legend_handles(environment, resolved_highlight_count),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.032),
        ncol=5 if environment.task_name == TASK_NAME_FORAGING else 4,
        frameon=False,
        fontsize=4.0,
        columnspacing=0.8,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    figure.subplots_adjust(left=0.045, right=0.995, top=0.77, bottom=0.145, wspace=0.06)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _draw_publication_comparison_axis(
    axis,
    environment: TaskEnvironment,
    trajectory_payload: Dict[str, object],
    highlight_completion_count: int,
) -> None:
    """Render one comparison panel using simplified styling."""
    _draw_comparison_environment(axis, environment)

    player = dict(trajectory_payload["player"])
    summary = dict(trajectory_payload["summary"])
    role_color = COMPARISON_ROLE_COLORS.get(player["label"], COMPARISON_ROUTE_COLOR)
    prefix_events, completion_events = _trajectory_prefix_until_completion(
        trajectory_payload,
        highlight_completion_count=highlight_completion_count,
    )
    route_positions = _deduplicate_positions(
        [environment.start_pos] + [(int(event["x"]), int(event["y"])) for event in prefix_events]
    )
    if len(route_positions) >= 2:
        xs = [position[0] for position in route_positions]
        ys = [position[1] for position in route_positions]
        axis.plot(
            xs,
            ys,
            color=role_color,
            linewidth=1.15,
            alpha=0.88,
            solid_capstyle="round",
            zorder=2,
        )

    _draw_comparison_targets(axis, environment)
    _draw_collected_high_value_underlays(axis, environment, completion_events)

    for event in completion_events:
        completion_fill_color = matplotlib.colors.to_rgba(role_color, alpha=0.78)
        axis.scatter(
            int(event["x"]),
            int(event["y"]),
            s=52,
            marker="o",
            c=[completion_fill_color],
            edgecolors="white",
            linewidths=0.85,
            zorder=6,
        )
        axis.text(
            int(event["x"]),
            int(event["y"]),
            str(int(event["completion_rank"])),
            ha="center",
            va="center",
            fontsize=3.7,
            color="white",
            fontweight="bold",
            zorder=7,
        )

    _draw_publication_start_marker(axis, environment.start_pos)
    axis.text(
        0.02,
        0.98,
        _comparison_summary_text(environment, summary),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=4.2,
        color="#374151",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#d1d5db",
            "alpha": 0.94,
        },
        zorder=8,
    )


def _draw_publication_layout_axis(axis, environment: TaskEnvironment) -> None:
    """Draw a standalone layout panel using the simplified panel style."""
    _draw_comparison_environment(axis, environment)
    _draw_comparison_targets(axis, environment)
    _draw_publication_start_marker(axis, environment.start_pos)


def _draw_publication_start_marker(axis, start_pos: Tuple[int, int]) -> None:
    """Draw the start marker shared by the layout and comparison panels."""
    axis.scatter(
        start_pos[0],
        start_pos[1],
        s=70,
        marker="^",
        c=["white"],
        edgecolors="#111827",
        linewidths=0.95,
        zorder=7,
    )


def _draw_comparison_environment(axis, environment: TaskEnvironment) -> None:
    """Draw a simplified, two-tone environment for the comparison plots."""
    blocked_color = _comparison_blocked_color(environment)
    terrain = np.where(environment.terrain == 0, 0, 1)
    axis.imshow(
        terrain,
        cmap=matplotlib.colors.ListedColormap([COMPARISON_FREE_TERRAIN_COLOR, blocked_color]),
        origin="upper",
        interpolation="nearest",
    )
    axis.set_xticks(np.arange(-0.5, environment.grid_size, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, environment.grid_size, 1), minor=True)
    axis.grid(which="minor", color="#d9dde3", linewidth=0.45, alpha=0.35)
    axis.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    axis.set_xlim(-0.5, environment.grid_size - 0.5)
    axis.set_ylim(environment.grid_size - 0.5, -0.5)
    for spine in axis.spines.values():
        spine.set_color("#d1d5db")
        spine.set_linewidth(0.9)


def _draw_comparison_targets(axis, environment: TaskEnvironment) -> None:
    """Draw muted target locations and keep only high-value food visually distinct."""
    standard_targets = [target for target in environment.targets if not target.is_high_value]
    if standard_targets:
        axis.scatter(
            [target.x for target in standard_targets],
            [target.y for target in standard_targets],
            s=20,
            marker="o",
            facecolors="white",
            edgecolors=COMPARISON_TARGET_EDGE_COLOR,
            linewidths=0.75,
            zorder=4,
        )

    high_value_targets = [target for target in environment.targets if target.is_high_value]
    if high_value_targets:
        axis.scatter(
            [target.x for target in high_value_targets],
            [target.y for target in high_value_targets],
            s=72,
            marker="*",
            c=[COMPARISON_HIGH_VALUE_COLOR],
            edgecolors="#6b7280",
            linewidths=0.7,
            zorder=5,
        )


def _comparison_summary_text(environment: TaskEnvironment, summary: Dict[str, object]) -> str:
    """Return the small per-panel summary shown on the comparison plots."""
    summary_lines = [f"patch switches: {int(summary['patch_switch_count'])}"]
    if environment.task_name == TASK_NAME_FORAGING:
        summary_lines.append(f"early high-value: {summary['early_high_value_rate']:.2f}")
    else:
        summary_lines.append(f"mean patch run: {summary['mean_patch_run_length']:.1f}")
    return "\n".join(summary_lines)


def _draw_collected_high_value_underlays(
    axis,
    environment: TaskEnvironment,
    completion_events: List[Dict[str, object]],
) -> None:
    """Keep collected high-value food visible beneath numbered completion markers."""
    if environment.task_name != TASK_NAME_FORAGING:
        return

    high_value_events = [event for event in completion_events if _is_high_value_completion(event)]
    if not high_value_events:
        return

    axis.scatter(
        [int(event["x"]) for event in high_value_events],
        [int(event["y"]) for event in high_value_events],
        s=130,
        marker="*",
        c=[COMPARISON_COLLECTED_HIGH_VALUE_COLOR],
        edgecolors="#b08900",
        linewidths=0.6,
        alpha=0.95,
        zorder=5.8,
    )


def _trajectory_prefix_until_completion(
    trajectory_payload: Dict[str, object],
    highlight_completion_count: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Return the event prefix and completion markers up to the requested rank."""
    prefix_events: List[Dict[str, object]] = []
    completion_events: List[Dict[str, object]] = []
    for event in trajectory_payload["events"]:
        prefix_events.append(event)
        if event["completion_rank"] is None:
            continue
        completion_events.append(event)
        if int(event["completion_rank"]) >= highlight_completion_count:
            break
    return prefix_events, completion_events


def _completion_events(trajectory_payload: Dict[str, object]) -> List[Dict[str, object]]:
    """Return completion events for one trajectory payload."""
    return [event for event in trajectory_payload["events"] if event["completion_rank"] is not None]


def _is_high_value_completion(event: Dict[str, object]) -> bool:
    """Return whether one completion event collected high-value food.

    Compact events carry the target reward but not the ``is_high_value`` flag; with
    the default rewards (1.0 ordinary, 3.0 high-value) a reward above 1.0 marks
    high-value food.
    """
    return float(event["target_reward"]) > 1.0


def _deduplicate_positions(positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Remove consecutive duplicate positions so route overlays stay visually clean."""
    deduplicated: List[Tuple[int, int]] = []
    for position in positions:
        if deduplicated and position == deduplicated[-1]:
            continue
        deduplicated.append(position)
    return deduplicated


def _comparison_legend_handles(
    environment: TaskEnvironment, highlight_completion_count: int
) -> List[object]:
    """Build a compact legend for the comparison plots."""
    handles: List[object] = [
        Patch(
            facecolor=_comparison_blocked_color(environment),
            edgecolor="none",
            label="Impassable cell",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            label="Start position",
            markerfacecolor="white",
            markeredgecolor="#111827",
            markersize=4.8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Target location",
            markerfacecolor="white",
            markeredgecolor=COMPARISON_TARGET_EDGE_COLOR,
            markersize=4.2,
        ),
        Line2D(
            [0],
            [0],
            color=COMPARISON_ROUTE_COLOR,
            linewidth=1.15,
            marker="o",
            markerfacecolor=COMPARISON_ROUTE_COLOR,
            markeredgecolor="white",
            markersize=4.2,
            label=f"Numbered route: first {highlight_completion_count} completions",
        ),
    ]
    high_value_rewards = [
        float(target.reward) for target in environment.targets if target.is_high_value
    ]
    if environment.task_name == TASK_NAME_FORAGING and high_value_rewards:
        handles.insert(
            3,
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                label="High-value food (+{:g})".format(max(high_value_rewards)),
                markerfacecolor=COMPARISON_HIGH_VALUE_COLOR,
                markeredgecolor="#6b7280",
                markersize=6.5,
            ),
        )
    return handles


def _comparison_blocked_color(environment: TaskEnvironment) -> str:
    """Return the blocked-cell color used in comparison plots."""
    if environment.task_name == TASK_NAME_CLEANING:
        return COMPARISON_CLEANING_BLOCKED_COLOR
    return COMPARISON_FORAGING_BLOCKED_COLOR


def plot_latent_distribution(players_df: pd.DataFrame, output_path: Path) -> None:
    """Plot latent P against latent O as a compact, simplified panel."""
    figure, axis = plt.subplots(figsize=PUBLICATION_PANEL_FIGSIZE)
    figure.patch.set_facecolor("white")
    color_map = {
        LABEL_PLANNER: COMPARISON_ROLE_COLORS[LABEL_PLANNER],
        LABEL_HOPPER: COMPARISON_ROLE_COLORS[LABEL_HOPPER],
    }
    display_label_map = {
        LABEL_PLANNER: "Type A",
        LABEL_HOPPER: "Type B",
    }
    for label, group in players_df.groupby("label", sort=True):
        axis.scatter(
            group["latent_P"],
            group["latent_O"],
            s=7,
            alpha=0.52,
            c=color_map.get(label, "#264653"),
            label=display_label_map.get(label, label),
            linewidths=0.0,
            rasterized=len(group) > 200,
        )
    axis.axhline(0.0, color="#cbd5e1", linewidth=0.8, linestyle="--", zorder=0)
    axis.axvline(0.0, color="#cbd5e1", linewidth=0.8, linestyle="--", zorder=0)
    axis.set_xlabel("P", fontsize=7, labelpad=0.5)
    axis.set_ylabel("O", fontsize=7, labelpad=0.5)
    axis.tick_params(axis="both", labelsize=6, length=2.2, width=0.6, colors="#4b5563")
    axis.grid(True, linestyle=":", linewidth=0.5, alpha=0.24, color="#cbd5e1")
    axis.legend(
        loc="upper left",
        frameon=False,
        fontsize=5.2,
        handletextpad=0.3,
        borderpad=0.1,
        borderaxespad=0.2,
        labelspacing=0.2,
        markerscale=1.0,
    )
    for spine in axis.spines.values():
        spine.set_color("#d1d5db")
        spine.set_linewidth(0.8)
    figure.subplots_adjust(left=0.15, right=0.995, bottom=0.12, top=0.995)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_behavior_summaries(summaries_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot per-label patch-switch counts, patch run lengths and early high-value rates."""
    _plot_metric_by_task(
        summaries_df=summaries_df,
        metric="patch_switch_count",
        title="Patch Switch Count by Label",
        output_path=output_dir / "patch_switch_by_label.pdf",
    )
    _plot_metric_by_task(
        summaries_df=summaries_df,
        metric="mean_patch_run_length",
        title="Mean Patch Run Length by Label",
        output_path=output_dir / "mean_patch_run_by_label.pdf",
    )

    figure, axis = plt.subplots(figsize=(7, 6))
    foraging_df = summaries_df[summaries_df["task_name"] == TASK_NAME_FORAGING]
    data = [
        foraging_df.loc[foraging_df["label"] == LABEL_PLANNER, "early_high_value_rate"]
        .dropna()
        .values,
        foraging_df.loc[foraging_df["label"] == LABEL_HOPPER, "early_high_value_rate"]
        .dropna()
        .values,
    ]
    bp = axis.boxplot(data, labels=[LABEL_PLANNER, LABEL_HOPPER], patch_artist=True)
    colors = ["#2a9d8f", "#e76f51"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axis.set_ylabel("Fraction of high-value targets captured in first quarter")
    axis.set_title("Foraging Early High-Value Rate", fontsize=14)
    axis.grid(True, axis="y", linestyle=":", alpha=0.4)
    figure.tight_layout()
    figure.savefig(output_dir / "early_high_value_rate_by_label.pdf", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_metric_by_task(
    summaries_df: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    task_order = [TASK_NAME_FORAGING, TASK_NAME_CLEANING]
    color_map = {
        LABEL_PLANNER: "#2a9d8f",
        LABEL_HOPPER: "#e76f51",
    }

    for axis, task_name in zip(axes, task_order):
        task_df = summaries_df[summaries_df["task_name"] == task_name]
        data = [
            task_df.loc[task_df["label"] == LABEL_PLANNER, metric].dropna().values,
            task_df.loc[task_df["label"] == LABEL_HOPPER, metric].dropna().values,
        ]
        bp = axis.boxplot(data, labels=[LABEL_PLANNER, LABEL_HOPPER], patch_artist=True)
        for box, label in zip(bp["boxes"], [LABEL_PLANNER, LABEL_HOPPER]):
            box.set_facecolor(color_map[label])
            box.set_alpha(0.65)
        axis.set_title(task_name.title())
        axis.grid(True, axis="y", linestyle=":", alpha=0.4)
        axis.set_ylabel(metric)

    figure.suptitle(title, fontsize=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _draw_trajectory_axis(
    axis, environment: TaskEnvironment, trajectory_payload: Dict[str, object]
) -> None:
    patch_colors = _patch_color_map(environment)
    _draw_base_environment(axis, environment)

    events = list(trajectory_payload["events"])
    player = dict(trajectory_payload["player"])
    summary = dict(trajectory_payload["summary"])

    for target in environment.targets:
        axis.scatter(
            target.x,
            target.y,
            s=55 if not target.is_high_value else 90,
            c=[patch_colors[target.patch_id]],
            marker="*"
            if target.is_high_value
            else ("o" if environment.task_name == TASK_NAME_FORAGING else "s"),
            edgecolors="white",
            linewidths=0.4,
            alpha=0.55,
            zorder=3,
        )

    xs = [environment.start_pos[0]] + [int(event["x"]) for event in events]
    ys = [environment.start_pos[1]] + [int(event["y"]) for event in events]
    axis.plot(xs, ys, color="#111111", linewidth=1.4, alpha=0.85, zorder=2)

    completion_events = [event for event in events if event["completion_rank"] is not None]
    for event in completion_events:
        patch_id = int(event["target_patch"])
        axis.scatter(
            int(event["x"]),
            int(event["y"]),
            s=115,
            c=[patch_colors[patch_id]],
            marker="o",
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )
        axis.text(
            int(event["x"]) + 0.12,
            int(event["y"]) - 0.15,
            str(int(event["completion_rank"])),
            fontsize=6,
            color="black",
            zorder=5,
        )

    axis.scatter(
        environment.start_pos[0],
        environment.start_pos[1],
        s=170,
        marker="P",
        c=["#bc4749"],
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
    )
    axis.set_title(
        (
            f"{player['label']} | P={player['latent_P']:.2f} O={player['latent_O']:.2f} | "
            f"steps={summary['total_steps']}"
        ),
        fontsize=12,
    )


def _draw_base_environment(axis, environment: TaskEnvironment) -> None:
    terrain = environment.terrain
    if environment.task_name == TASK_NAME_FORAGING:
        colors = FORAGING_TERRAIN_COLORS
    elif environment.task_name == TASK_NAME_CLEANING:
        colors = CLEANING_TERRAIN_COLORS
    else:
        raise ValueError(f"Unsupported environment task_name: {environment.task_name}")

    max_code = int(np.max(terrain))
    color_list = [colors.get(code, "#cccccc") for code in range(max_code + 1)]
    axis.imshow(terrain, cmap=matplotlib.colors.ListedColormap(color_list), origin="upper")
    axis.set_xticks(np.arange(-0.5, environment.grid_size, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, environment.grid_size, 1), minor=True)
    axis.grid(which="minor", color="#ced4da", linewidth=0.6, alpha=0.45)
    axis.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    axis.set_xlim(-0.5, environment.grid_size - 0.5)
    axis.set_ylim(environment.grid_size - 0.5, -0.5)


def _patch_color_map(environment: TaskEnvironment) -> Dict[int, str]:
    patch_ids = sorted({target.patch_id for target in environment.targets})
    cmap = plt.get_cmap("tab20", max(len(patch_ids), 1))
    return {
        patch_id: matplotlib.colors.to_hex(cmap(index)) for index, patch_id in enumerate(patch_ids)
    }
