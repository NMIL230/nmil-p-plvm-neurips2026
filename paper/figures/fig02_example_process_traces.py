#!/usr/bin/env python3
"""Generate Figure 2: one player's observed trajectories on the subtask maps.

Overlays player ``p191`` attempt 1 (by default) on the canonical subtask-map
points for Backgarden, Bungalow and the Fire Station target, read from the
per-attempt player sequences and the subtask task-map CSV.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from powerwash.analysis.plot_subtask_task_maps import (
    _load_traj_for_pid,
    derive_task_rows,
    padded_limits,
)
from paper.output_paths import resolve_figures_dir
from plvm import artifact_paths


RAW_INPUT_DIR = artifact_paths.PLAYER_SEQUENCES_DIR
TASK_MAP_CSV = artifact_paths.SUBTASK_TASK_MAP_POINTS_CSV
FIGURES_DIR = resolve_figures_dir()
DEFAULT_OUTPUT_PATH = FIGURES_DIR / "Figure_2.pdf"
DEFAULT_PLAYER_ID = "p191"
DEFAULT_ATTEMPT_NUMBER = 1
DEFAULT_DPI = 600
FIGURE_WIDTH_INCHES = 6.4
FIGURE_HEIGHT_INCHES = 3.2
SUPTITLE_FONT_SIZE = 12
LEGEND_FONT_SIZE = 9.5
SUBPLOT_LEFT = 0.075
SUBPLOT_RIGHT = 0.99
SUBPLOT_BOTTOM = 0.17
SUBPLOT_TOP = 0.68
SUBPLOT_WSPACE = 0.18
LEGEND_Y = 0.895
SUPTITLE_Y = 0.975
TARGET_MARKER_SIZE = 10

TASK_PANELS = [
    ("back_garden", "Backgarden (observed)"),
    ("bungalow", "Bungalow (observed)"),
    ("fire_station", "Fire Station (Target)"),
]
TRAJECTORY_COLORS = {
    "back_garden": "#2f8aa3",
    "bungalow": "#2f8aa3",
    "fire_station": "#b94d5d",
}
NUMBER_OFFSETS = [
    (10, 10),
    (-10, 10),
    (10, -10),
    (-10, -10),
    (13, 0),
    (-13, 0),
    (0, 13),
    (0, -13),
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the Figure 2 renderer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--player_id",
        default=DEFAULT_PLAYER_ID,
        help="Player identifier to plot.",
    )
    parser.add_argument(
        "--attempt_number",
        type=int,
        default=DEFAULT_ATTEMPT_NUMBER,
        help="Attempt number to load for each task.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=RAW_INPUT_DIR,
        help="Directory containing PowerWash per-attempt CSVs.",
    )
    parser.add_argument(
        "--task_map_csv",
        type=Path,
        default=TASK_MAP_CSV,
        help="Canonical task-map CSV produced by plot_subtask_task_maps.py.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output PDF path for Figure 2.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Save DPI for the output figure.",
    )
    return parser.parse_args(argv)


def configure_plot_style() -> None:
    """Apply the static plotting defaults for Figure 2."""
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": DEFAULT_DPI,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _parse_optional_float(raw_value: Optional[str]) -> Optional[float]:
    """Return a float for one CSV field when parseable, else ``None``."""
    cleaned = str(raw_value or "").strip()
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if not np.isfinite(value):
        return None
    return value


def load_task_rows_from_csv(
    task_map_csv: Path,
    task_names: Sequence[str],
) -> Dict[str, List[dict]]:
    """Load canonical subtask-map rows from the aggregated CSV when available."""
    if not task_map_csv.exists():
        return {}

    task_name_set = set(task_names)
    rows_by_task: Dict[str, List[dict]] = {task_name: [] for task_name in task_names}
    with task_map_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            task_name = str(row.get("task") or "").strip()
            if task_name not in task_name_set:
                continue
            rows_by_task[task_name].append(
                {
                    "task": task_name,
                    "point_id": _parse_optional_float(row.get("point_id")),
                    "subtask_name": str(row.get("subtask_name") or "").strip(),
                    "x": _parse_optional_float(row.get("x")),
                    "z": _parse_optional_float(row.get("z")),
                }
            )
    return {task_name: rows for task_name, rows in rows_by_task.items() if rows}


def resolve_task_rows(
    input_dir: Path,
    task_map_csv: Path,
    task_names: Sequence[str],
) -> Dict[str, List[dict]]:
    """Resolve canonical subtask rows from the cached CSV with a raw-data fallback."""
    rows_by_task = load_task_rows_from_csv(task_map_csv, task_names)
    missing_tasks = [
        task_name
        for task_name in task_names
        if not any(
            row.get("x") is not None and row.get("z") is not None
            for row in rows_by_task.get(task_name, [])
        )
    ]
    if missing_tasks:
        derived_rows = derive_task_rows(input_dir, missing_tasks)
        rows_by_task.update(derived_rows)
    return rows_by_task


def load_player_trajectories(
    input_dir: Path,
    player_id: str,
    attempt_number: int,
    task_names: Sequence[str],
) -> Dict[str, dict]:
    """Load one player's usable trajectories for each requested task."""
    trajectories = {}
    for task_name in task_names:
        trajectory = _load_traj_for_pid(input_dir, task_name, player_id, attempt_number)
        if trajectory is None:
            raise FileNotFoundError(
                "No usable trajectory found for player '{}' task '{}' attempt {}.".format(
                    player_id,
                    task_name,
                    attempt_number,
                )
            )
        trajectories[task_name] = trajectory
    return trajectories


def build_axis_limits(
    task_rows: Sequence[dict], positions: Sequence[Tuple[float, float]]
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute padded axis limits from both canonical targets and trajectory positions."""
    target_x = [row["x"] for row in task_rows if row.get("x") is not None]
    target_z = [row["z"] for row in task_rows if row.get("z") is not None]
    path_x = [position[0] for position in positions]
    path_z = [position[1] for position in positions]
    x_values = np.asarray(target_x + path_x, dtype="float64")
    z_values = np.asarray(target_z + path_z, dtype="float64")
    return padded_limits(x_values), padded_limits(z_values)


def draw_targets(ax: plt.Axes, task_rows: Sequence[dict]) -> None:
    """Draw canonical subtask locations as an unlabeled background layer."""
    xs = [row["x"] for row in task_rows if row.get("x") is not None]
    zs = [row["z"] for row in task_rows if row.get("z") is not None]
    if not xs:
        return
    ax.scatter(
        xs,
        zs,
        s=TARGET_MARKER_SIZE,
        color="#d4d4d8",
        edgecolors="#a1a1aa",
        linewidths=0.55,
        alpha=0.95,
        zorder=1,
    )


def draw_direction_arrows(
    ax: plt.Axes,
    xs: np.ndarray,
    zs: np.ndarray,
    color: str,
    max_arrows: int = 18,
) -> None:
    """Overlay regularly sampled direction arrows on a polyline trajectory."""
    if xs.size < 2:
        return

    step = max(1, int(np.ceil((xs.size - 1) / float(max_arrows))))
    for start_idx in range(0, xs.size - 1, step):
        end_idx = start_idx + 1
        dx = xs[end_idx] - xs[start_idx]
        dz = zs[end_idx] - zs[start_idx]
        if dx == 0.0 and dz == 0.0:
            continue
        ax.add_patch(
            FancyArrowPatch(
                (xs[start_idx], zs[start_idx]),
                (xs[end_idx], zs[end_idx]),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.25,
                color=color,
                alpha=0.80,
                shrinkA=0.0,
                shrinkB=0.0,
                zorder=4,
            )
        )


def annotate_first_positions(
    ax: plt.Axes,
    xs: np.ndarray,
    zs: np.ndarray,
    color: str,
    n_positions: int = 10,
) -> None:
    """Label the first ``n_positions`` movement steps after the start."""
    if xs.size < 2:
        return

    label_count = min(int(n_positions), int(xs.size - 1))
    for index in range(label_count):
        point_index = index + 1
        offset = NUMBER_OFFSETS[index % len(NUMBER_OFFSETS)]
        ax.annotate(
            str(index + 1),
            xy=(xs[point_index], zs[point_index]),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color="#111827",
            bbox={
                "boxstyle": "circle,pad=0.28",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 1.0,
                "alpha": 0.97,
            },
            zorder=7,
        )


def draw_start_end_markers(ax: plt.Axes, xs: np.ndarray, zs: np.ndarray) -> None:
    """Draw highly visible start/end markers without text callouts."""
    ax.scatter(
        xs[:1],
        zs[:1],
        marker="*",
        s=320,
        color="#2563eb",
        edgecolors="white",
        linewidths=1.5,
        zorder=8,
    )
    ax.scatter(
        xs[-1:],
        zs[-1:],
        marker="X",
        s=220,
        color="#dc2626",
        edgecolors="white",
        linewidths=1.5,
        zorder=8,
    )


def draw_trajectory(ax: plt.Axes, positions: Sequence[Tuple[float, float]], color: str) -> None:
    """Draw the full trajectory path, directional arrows, and first-step labels."""
    xs = np.asarray([position[0] for position in positions], dtype="float64")
    zs = np.asarray([position[1] for position in positions], dtype="float64")
    ax.plot(
        xs,
        zs,
        color=color,
        linewidth=2.1,
        alpha=0.88,
        solid_capstyle="round",
        zorder=3,
    )
    draw_direction_arrows(ax, xs, zs, color=color)
    annotate_first_positions(ax, xs, zs, color=color, n_positions=10)
    draw_start_end_markers(ax, xs, zs)


def style_axis(
    ax: plt.Axes,
    title: str,
    x_limits: Tuple[float, float],
    z_limits: Tuple[float, float],
    show_ylabel: bool,
) -> None:
    """Apply the Figure 2 axis styling."""
    ax.set_title(title, pad=12)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*z_limits)
    ax.set_box_aspect(1)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x position")
    ax.set_ylabel("z position" if show_ylabel else "")
    ax.grid(False)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.tick_params(length=3.5, width=0.8, colors="#374151")


def build_figure(
    player_id: str,
    task_rows_by_task: Dict[str, List[dict]],
    trajectories_by_task: Dict[str, dict],
    output_path: Path,
    dpi: int,
) -> Path:
    """Render and save the Figure 2 trajectory figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        len(TASK_PANELS),
        figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES),
        gridspec_kw={"width_ratios": [1, 1, 1]},
    )
    fig.subplots_adjust(
        left=SUBPLOT_LEFT,
        right=SUBPLOT_RIGHT,
        bottom=SUBPLOT_BOTTOM,
        top=SUBPLOT_TOP,
        wspace=SUBPLOT_WSPACE,
    )
    for index, (ax, (task_name, panel_title)) in enumerate(zip(axes, TASK_PANELS)):
        task_rows = task_rows_by_task.get(task_name, [])
        if not task_rows:
            raise ValueError("No canonical task-map rows available for '{}'.".format(task_name))
        trajectory = trajectories_by_task[task_name]
        line_color = TRAJECTORY_COLORS.get(task_name, "#1f2937")
        x_limits, z_limits = build_axis_limits(task_rows, trajectory["positions"])
        draw_targets(ax, task_rows)
        draw_trajectory(ax, trajectory["positions"], color=line_color)
        style_axis(
            ax,
            title=panel_title,
            x_limits=x_limits,
            z_limits=z_limits,
            show_ylabel=index == 0,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=7,
            markerfacecolor="#d4d4d8",
            markeredgecolor="#a1a1aa",
            label="Subtask target",
        ),
        Line2D([0], [0], color="#1f2937", linewidth=2.1, label="Trajectory"),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="None",
            markersize=13,
            markerfacecolor="#2563eb",
            markeredgecolor="white",
            label="Start",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            linestyle="None",
            markersize=10,
            markerfacecolor="#dc2626",
            markeredgecolor="white",
            label="End",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, LEGEND_Y),
        ncol=4,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.45,
        fontsize=LEGEND_FONT_SIZE,
    )
    fig.suptitle(
        "Example Player Trajectories: {}".format(player_id),
        fontsize=SUPTITLE_FONT_SIZE,
        fontweight="bold",
        y=SUPTITLE_Y,
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Generate Figure 2, the example player trajectory figure."""
    args = parse_args(argv)
    if args.attempt_number <= 0:
        print("ERROR: --attempt_number must be positive", file=sys.stderr)
        return 1
    if args.dpi <= 0:
        print("ERROR: --dpi must be positive", file=sys.stderr)
        return 1

    input_dir = args.input_dir.expanduser().resolve()
    task_map_csv = args.task_map_csv.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print("ERROR: input directory does not exist: {}".format(input_dir), file=sys.stderr)
        return 1

    configure_plot_style()
    task_names = [task_name for task_name, _ in TASK_PANELS]
    try:
        task_rows_by_task = resolve_task_rows(input_dir, task_map_csv, task_names)
        trajectories_by_task = load_player_trajectories(
            input_dir=input_dir,
            player_id=str(args.player_id),
            attempt_number=int(args.attempt_number),
            task_names=task_names,
        )
        figure_path = build_figure(
            player_id=str(args.player_id),
            task_rows_by_task=task_rows_by_task,
            trajectories_by_task=trajectories_by_task,
            output_path=output_path,
            dpi=int(args.dpi),
        )
    except (FileNotFoundError, ValueError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1

    print("Wrote {}".format(figure_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
