#!/usr/bin/env python3
"""
Build canonical x/z subtask maps for the PowerWash study tasks.

Reads the attempt CSVs in ``powerwash/data/player_sequences/<task>/``, keeps
"clean" attempts whose normalized subtask set equals the modal set, and places
each subtask at the median x/z of its first completion across them. Writes
map figures to ``powerwash/analysis/figures/task_maps/`` and the point table
``powerwash/analysis/tables/subtask_task_map_points.csv``. With
``--n_traj_players N`` it also draws N sampled players' trajectories over the
maps into ``powerwash/analysis/figures/player_trajectories/``.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from powerwash.data.derive_features_for_transformer import POSITION_PATTERN
from powerwash.analysis.trajectory_utils import compute_zigzag_score, compute_path_length
from powerwash.data.powerwash_subtask_cleaning import EXPECTED_NORMALIZED_SUBTASKS_BY_TASK
from plvm import artifact_paths


INPUT_DIR = artifact_paths.PLAYER_SEQUENCES_DIR
FIGURES_DIR = artifact_paths.POWERWASH_ANALYSIS_FIGURES_DIR / "task_maps"
TRAJ_FIGURES_DIR = artifact_paths.POWERWASH_ANALYSIS_FIGURES_DIR / "player_trajectories"
TABLES_DIR = artifact_paths.POWERWASH_ANALYSIS_TABLES_DIR
OUTPUT_CSV = artifact_paths.SUBTASK_TASK_MAP_POINTS_CSV

TASK_ORDER = list(EXPECTED_NORMALIZED_SUBTASKS_BY_TASK)
TASK_COLORS = {
    "back_garden": "#1f7a8c",
    "bungalow": "#c95d38",
    "fire_station": "#b23a48",
    "storybook_house": "#6c9a3b",
}
CSV_COLUMNS = [
    "task",
    "point_id",
    "subtask_name",
    "x",
    "z",
    "n_clean_attempts_with_position",
    "n_position_rows",
    "median_radial_deviation",
    "p90_radial_deviation",
    "x_min",
    "x_max",
    "z_min",
    "z_max",
]
ANNOTATION_OFFSETS = [
    (5, 5),
    (-5, 5),
    (5, -5),
    (-5, -5),
    (7, 0),
    (-7, 0),
    (0, 7),
    (0, -7),
]


def parse_args(argv=None):
    """Parse command-line arguments for the task-map analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=INPUT_DIR,
        help="Directory containing extracted per-attempt sequence CSVs.",
    )
    parser.add_argument(
        "--figures_dir",
        type=Path,
        default=FIGURES_DIR,
        help="Directory where task-map figures will be written.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=OUTPUT_CSV,
        help="Output CSV path for the aggregated subtask coordinate table.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASK_ORDER),
        choices=list(TASK_ORDER),
        help="Task names to process.",
    )
    parser.add_argument(
        "--png_dpi",
        type=int,
        default=600,
        help="DPI used when saving PNG task-map figures.",
    )
    parser.add_argument(
        "--traj_figures_dir",
        type=Path,
        default=TRAJ_FIGURES_DIR,
        help="Directory where player-trajectory figures will be written.",
    )
    parser.add_argument(
        "--n_traj_players",
        type=int,
        default=0,
        help="Number of players to sample for trajectory overlay maps (0 = skip).",
    )
    parser.add_argument(
        "--traj_attempt",
        type=int,
        default=1,
        help="Attempt number to use when loading player trajectories.",
    )
    parser.add_argument(
        "--traj_random_state",
        type=int,
        default=42,
        help="Random seed used when sampling trajectory players.",
    )
    traj_paired_group = parser.add_mutually_exclusive_group()
    traj_paired_group.add_argument(
        "--traj_paired",
        dest="traj_paired",
        action="store_true",
        default=True,
        help=(
            "Generate side-by-side 1×N trajectory figures (one subplot per task), "
            "sampling only players present in all tasks (default: enabled). "
            "Either way the same player IDs are used across tasks."
        ),
    )
    traj_paired_group.add_argument(
        "--no_traj_paired",
        dest="traj_paired",
        action="store_false",
        help="Produce separate per-task trajectory PDFs instead of side-by-side figures.",
    )
    parser.add_argument(
        "--no_task_maps",
        action="store_true",
        default=False,
        help="Skip generating canonical subtask map figures (useful when only trajectories are wanted).",
    )
    return parser.parse_args(argv)


def maybe_set_plot_style():
    """Apply plot settings for high-resolution static figures."""
    for candidate in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        if candidate in plt.style.available:
            plt.style.use(candidate)
            break
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
        }
    )


def normalize_subtask_name(raw_name):
    """Collapse a raw subtask label to its leaf name."""
    cleaned = (raw_name or "").strip()
    if not cleaned:
        return ""
    return cleaned.split(".")[-1].strip()


def parse_xz_position(raw_position):
    """Parse a ``CurrentPosition`` string into an ``(x, z)`` tuple."""
    cleaned = (raw_position or "").strip()
    if not cleaned:
        return None
    match = POSITION_PATTERN.match(cleaned)
    if match is None:
        return None
    return float(match.group(1)), float(match.group(3))


def task_label(task_name):
    """Convert an internal task name into a presentation label."""
    return task_name.replace("_", " ").title()


def discover_attempt_files(input_dir, task_name):
    """Return sorted attempt CSV paths for one task."""
    task_dir = input_dir / task_name
    if not task_dir.exists() or not task_dir.is_dir():
        return []

    csv_paths = []
    for csv_path in sorted(task_dir.glob("*.csv")):
        if not csv_path.is_file():
            continue
        if ".ipynb_checkpoints" in csv_path.parts:
            continue
        csv_paths.append(csv_path)
    return csv_paths


def load_attempt_summary(csv_path):
    """
    Load a compact summary for one attempt CSV.

    The summary retains:
      - the normalized subtask set for canonical-set detection
      - the first parseable x/z position seen for each normalized subtask
      - the total number of parseable rows seen for each normalized subtask
    """
    normalized_subtasks = set()
    first_positions = {}
    parseable_position_row_counts = Counter()

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("event_type") or "").strip() != "subtask_completed":
                continue

            subtask_name = normalize_subtask_name(row.get("LastSubtaskCompleted"))
            if not subtask_name:
                continue

            normalized_subtasks.add(subtask_name)
            position = parse_xz_position(row.get("CurrentPosition"))
            if position is None:
                continue

            parseable_position_row_counts[subtask_name] += 1
            if subtask_name not in first_positions:
                first_positions[subtask_name] = position

    return {
        "path": csv_path,
        "normalized_subtasks": tuple(sorted(normalized_subtasks)),
        "first_positions": first_positions,
        "parseable_position_row_counts": parseable_position_row_counts,
    }


def select_canonical_subtask_set(attempt_summaries):
    """Pick the modal normalized subtask set for one task."""
    set_counts = Counter(
        summary["normalized_subtasks"]
        for summary in attempt_summaries
        if summary["normalized_subtasks"]
    )
    if not set_counts:
        return None, 0

    ranked = sorted(
        set_counts.items(),
        key=lambda item: (-item[1], -len(item[0]), item[0]),
    )
    canonical_subtasks, count = ranked[0]
    return canonical_subtasks, count


def compute_position_summary(points):
    """Summarize a set of x/z positions around their median location."""
    xs = np.asarray([point[0] for point in points], dtype="float64")
    zs = np.asarray([point[1] for point in points], dtype="float64")

    x_coord = float(np.median(xs))
    z_coord = float(np.median(zs))
    radial_deviation = np.sqrt((xs - x_coord) ** 2 + (zs - z_coord) ** 2)

    return {
        "x": x_coord,
        "z": z_coord,
        "median_radial_deviation": float(np.median(radial_deviation)),
        "p90_radial_deviation": float(np.percentile(radial_deviation, 90)),
        "x_min": float(xs.min()),
        "x_max": float(xs.max()),
        "z_min": float(zs.min()),
        "z_max": float(zs.max()),
    }


def build_task_rows(task_name, attempt_summaries, canonical_subtasks):
    """Aggregate canonical subtask rows for one task."""
    clean_attempts = [
        summary
        for summary in attempt_summaries
        if summary["normalized_subtasks"] == canonical_subtasks
    ]

    rows = []
    missing_position_subtasks = []
    for subtask_name in canonical_subtasks:
        first_points = []
        n_position_rows = 0

        for summary in clean_attempts:
            n_position_rows += summary["parseable_position_row_counts"].get(subtask_name, 0)
            if subtask_name in summary["first_positions"]:
                first_points.append(summary["first_positions"][subtask_name])

        row = {
            "task": task_name,
            "point_id": None,
            "subtask_name": subtask_name,
            "x": None,
            "z": None,
            "n_clean_attempts_with_position": len(first_points),
            "n_position_rows": n_position_rows,
            "median_radial_deviation": None,
            "p90_radial_deviation": None,
            "x_min": None,
            "x_max": None,
            "z_min": None,
            "z_max": None,
        }

        if first_points:
            row.update(compute_position_summary(first_points))
        else:
            missing_position_subtasks.append(subtask_name)

        rows.append(row)

    plotted_rows = sorted(
        [row for row in rows if row["x"] is not None and row["z"] is not None],
        key=lambda row: (-row["z"], row["x"], row["subtask_name"]),
    )
    for point_id, row in enumerate(plotted_rows, start=1):
        row["point_id"] = point_id

    rows.sort(
        key=lambda row: (
            TASK_ORDER.index(row["task"]) if row["task"] in TASK_ORDER else len(TASK_ORDER),
            row["point_id"] if row["point_id"] is not None else 10**9,
            row["subtask_name"],
        )
    )
    return rows, clean_attempts, missing_position_subtasks


def finite_or_none(value):
    """Return the float value when finite, otherwise ``None``."""
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric_value):
        return None
    return numeric_value


def write_output_csv(rows, output_csv):
    """Write the aggregated task-map rows to one CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "task": row["task"],
                    "point_id": row["point_id"],
                    "subtask_name": row["subtask_name"],
                    "x": finite_or_none(row["x"]),
                    "z": finite_or_none(row["z"]),
                    "n_clean_attempts_with_position": row["n_clean_attempts_with_position"],
                    "n_position_rows": row["n_position_rows"],
                    "median_radial_deviation": finite_or_none(row["median_radial_deviation"]),
                    "p90_radial_deviation": finite_or_none(row["p90_radial_deviation"]),
                    "x_min": finite_or_none(row["x_min"]),
                    "x_max": finite_or_none(row["x_max"]),
                    "z_min": finite_or_none(row["z_min"]),
                    "z_max": finite_or_none(row["z_max"]),
                }
            )


def merge_existing_task_rows(rows, output_csv, task_names):
    """Merge newly generated rows into an existing task-map CSV by task name."""
    if not output_csv.exists():
        return rows

    requested_tasks = {str(task_name).strip() for task_name in task_names}
    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        existing_rows = list(csv.DictReader(handle))

    preserved_rows = [
        row for row in existing_rows if str(row.get("task", "")).strip() not in requested_tasks
    ]
    return preserved_rows + rows


def load_player_trajectory(csv_path):
    """
    Parse one attempt CSV and return a dict with:
      - positions: ordered list of (x, z) tuples from subtask-completion rows only
      - total_time_s: total time in seconds from the last subtask-completion row
      - completion: completed expected subtasks / expected subtasks for the task;
                    falls back to the last subtask-completion
                    LevelProgressionAmount when no expected subtask set is
                    configured for the task
    """
    positions = []
    total_time_s = None
    raw_completion = None
    completed_subtasks = set()
    task_name = csv_path.parent.name.strip()
    expected_subtasks = EXPECTED_NORMALIZED_SUBTASKS_BY_TASK.get(task_name)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("event_type") or "").strip() != "subtask_completed":
                continue

            position = parse_xz_position(row.get("CurrentPosition"))
            if position is not None:
                positions.append(position)
            subtask_name = normalize_subtask_name(row.get("LastSubtaskCompleted"))
            if subtask_name:
                completed_subtasks.add(subtask_name)
            raw_time = (row.get("time_elapsed_since_job_start") or "").strip()
            if raw_time:
                try:
                    total_time_s = int(float(raw_time))
                except ValueError:
                    pass
            raw_lvl = (row.get("LevelProgressionAmount") or "").strip()
            if raw_lvl:
                try:
                    raw_completion = float(raw_lvl)
                except ValueError:
                    pass
    completion = raw_completion
    if expected_subtasks:
        completion = len(completed_subtasks.intersection(expected_subtasks)) / float(
            len(expected_subtasks)
        )
    return {"positions": positions, "total_time_s": total_time_s, "completion": completion}


def discover_attempt_files_by_number(input_dir, task_name, attempt_number):
    """Return sorted CSV paths whose filename matches a specific attempt number."""
    task_dir = input_dir / task_name
    if not task_dir.exists() or not task_dir.is_dir():
        return []
    suffix = "_attempt{}.csv".format(attempt_number)
    paths = []
    for csv_path in sorted(task_dir.glob("*.csv")):
        if csv_path.name.endswith(suffix) and ".ipynb_checkpoints" not in csv_path.parts:
            paths.append(csv_path)
    return paths


def _load_traj_for_pid(input_dir, task_name, pid, attempt_number):
    """Load trajectory data for one (pid, task, attempt) triple; returns None if unusable."""
    suffix = "_attempt{}.csv".format(attempt_number)
    csv_path = input_dir / task_name / "{}_{}{}".format(pid, task_name, suffix)
    if not csv_path.exists():
        return None
    traj = load_player_trajectory(csv_path)
    if len(traj["positions"]) < 2:
        return None
    return {
        "pid": pid,
        "path": csv_path,
        **{k: traj[k] for k in ("positions", "total_time_s", "completion")},
    }


def sample_paired_trajectory_players(
    input_dir, task_names, n_players, random_state, attempt_number=1
):
    """
    Sample up to n_players who have usable attempt CSVs for every task in task_names.

    Returns a list of dicts:
        {pid, tasks: {task_name: {pid, path, positions, total_time_s, completion}}}
    Players are drawn from the intersection of PIDs present across all tasks so
    both subplots always show the same individuals.
    """
    pid_sets = []
    for task_name in task_names:
        paths = discover_attempt_files_by_number(input_dir, task_name, attempt_number)
        pids = set()
        for p in paths:
            pid = p.stem.split("_{}".format(task_name))[0]
            pids.add(pid)
        pid_sets.append(pids)

    common_pids = sorted(pid_sets[0].intersection(*pid_sets[1:]))
    if not common_pids:
        return []

    rng = np.random.RandomState(random_state)
    shuffled_pids = [common_pids[i] for i in rng.permutation(len(common_pids))]

    selected = []
    for pid in shuffled_pids:
        task_trajs = {}
        for task_name in task_names:
            traj = _load_traj_for_pid(input_dir, task_name, pid, attempt_number)
            if traj is None:
                task_trajs = None
                break
            task_trajs[task_name] = traj
        if task_trajs is None:
            continue
        selected.append({"pid": pid, "tasks": task_trajs})
        if len(selected) >= n_players:
            break
    return selected


def _pub_rcparams():
    """Return a dict of rcParams for high-resolution trajectory figures."""
    return {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "#f8f8f8",
        "figure.facecolor": "white",
        "grid.color": "#cccccc",
        "grid.linewidth": 0.6,
        "grid.linestyle": "--",
        "legend.fontsize": 10,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#cccccc",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    }


def _style_traj_ax(ax, x_lim, z_lim, legend_handles, legend_labels):
    """Apply shared axis styling for trajectory panels."""
    ax.set_xlabel("x (game units)", labelpad=6)
    ax.set_ylabel("z (game units)", labelpad=6)
    if x_lim is not None:
        ax.set_xlim(*x_lim)
        ax.set_ylim(*z_lim)

    ax.set_axisbelow(True)
    ax.grid(True)
    ax.tick_params(length=3, width=0.7)
    if legend_handles:
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper right",
            handlelength=1.4,
            handletextpad=0.5,
            borderpad=0.6,
        )


def _draw_subtask_background(ax, plotted_rows, label_stride: int = 1):
    """Draw canonical subtask points as a muted reference layer; return a legend handle.

    label_stride controls how many subtask IDs are labelled: 1 = all, 5 = every 5th.
    All points are always drawn; only the stride-selected ones get text annotations.
    """
    if not plotted_rows:
        return None
    x_bg = np.asarray([r["x"] for r in plotted_rows], dtype="float64")
    z_bg = np.asarray([r["z"] for r in plotted_rows], dtype="float64")
    handle = ax.scatter(
        x_bg,
        z_bg,
        s=90,
        color="#cccccc",
        edgecolors="#999999",
        linewidths=0.6,
        zorder=2,
    )
    stride = max(1, int(label_stride))
    for idx, row in enumerate(plotted_rows):
        if idx % stride != 0:
            continue
        ax.annotate(
            str(row["point_id"]),
            xy=(row["x"], row["z"]),
            xytext=(4, 4),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=6,
            color="#888888",
            zorder=3,
        )
    return handle


def _draw_player_trajectory(ax, player, color):
    """Draw one player's path, start, and end; return (start_handle, end_handle)."""
    xs = np.asarray([p[0] for p in player["positions"]], dtype="float64")
    zs = np.asarray([p[1] for p in player["positions"]], dtype="float64")
    ax.plot(xs, zs, color=color, linewidth=1.6, alpha=0.80, zorder=4, solid_capstyle="round")
    h_start = ax.scatter(
        xs[:1],
        zs[:1],
        marker="o",
        s=70,
        color="#d73027",
        edgecolors="white",
        linewidths=0.8,
        zorder=6,
    )
    h_end = ax.scatter(
        xs[-1:],
        zs[-1:],
        marker="s",
        s=70,
        color="#1a9641",
        edgecolors="white",
        linewidths=0.8,
        zorder=6,
    )
    return h_start, h_end


def _player_page_title(task_name, player, zigzag=None, path_length=None):
    """Build the suptitle string for one player page."""
    time_str = (
        "time={} s".format(player["total_time_s"])
        if player["total_time_s"] is not None
        else "time=N/A"
    )
    completion_str = (
        "completion={:.1%}".format(player["completion"])
        if player["completion"] is not None
        else "completion=N/A"
    )
    zz_str = " | reversals={}".format(zigzag) if zigzag is not None else ""
    pl_str = " | path length={:.0f}".format(path_length) if path_length is not None else ""
    return "{} | {} | {} | {}{}{}".format(
        task_label(task_name), player["pid"], time_str, completion_str, zz_str, pl_str
    )


def plot_trajectory_pages(
    task_name, plotted_rows, sampled_players, attempt_number, traj_figures_dir, png_dpi
):
    """
    Write a multi-page PDF with one page per sampled player.

    Axis limits come from the canonical task-map points only, so trajectories
    may extend beyond the plot. The PNG shows only the first page.
    """
    if not sampled_players:
        return None, None

    traj_figures_dir.mkdir(parents=True, exist_ok=True)

    x_limits, z_limits = _taskmap_limits(task_name, {task_name: plotted_rows})
    if x_limits is None:
        return None, None

    from matplotlib.backends.backend_pdf import PdfPages

    stem = traj_figures_dir / "{}_player_trajectories".format(task_name)
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")

    pub_rc = _pub_rcparams()
    with PdfPages(pdf_path) as pdf:
        for page_idx, player in enumerate(sampled_players):
            with plt.rc_context(pub_rc):
                fig, ax = plt.subplots(figsize=(8, 8))

                h_subtask = _draw_subtask_background(ax, plotted_rows)
                h_start, h_end = _draw_player_trajectory(ax, player, color="#2166ac")

                legend_handles, legend_labels = [], []
                if h_subtask is not None:
                    legend_handles.append(h_subtask)
                    legend_labels.append("Subtask location")
                legend_handles += [h_start, h_end]
                legend_labels += ["Start", "End"]

                zigzag = compute_zigzag_score(player["positions"])
                path_length = compute_path_length(player["positions"])
                fig.suptitle(
                    _player_page_title(task_name, player, zigzag=zigzag, path_length=path_length),
                    fontsize=11,
                    fontweight="bold",
                    y=0.98,
                )
                _style_traj_ax(ax, x_limits, z_limits, legend_handles, legend_labels)
                fig.text(
                    0.99,
                    0.005,
                    "{}/{}".format(page_idx + 1, len(sampled_players)),
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color="#aaaaaa",
                )
                fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.95])

                pdf.savefig(fig, dpi=png_dpi, bbox_inches="tight")
                if page_idx == 0:
                    fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight", facecolor="white")
                plt.close(fig)

    return pdf_path, png_path


def _subplot_title(task_name, traj, zigzag=None, path_length=None):
    """Subtitle for one task panel."""
    time_str = (
        "time={} s".format(traj["total_time_s"]) if traj["total_time_s"] is not None else "time=N/A"
    )
    completion_str = (
        "completion={:.1%}".format(traj["completion"])
        if traj["completion"] is not None
        else "completion=N/A"
    )
    zz_str = " | reversals={}".format(zigzag) if zigzag is not None else ""
    pl_str = " | path length={:.0f}".format(path_length) if path_length is not None else ""
    return "{} | {} | {}{}{}".format(
        task_label(task_name), time_str, completion_str, zz_str, pl_str
    )


def _taskmap_limits(task_name, plotted_rows_by_task):
    """Compute x/z limits from task-map canonical points only (no player positions)."""
    rows = plotted_rows_by_task.get(task_name, [])
    xs = [r["x"] for r in rows if r.get("x") is not None]
    zs = [r["z"] for r in rows if r.get("z") is not None]
    if not xs:
        return None, None
    return (
        padded_limits(np.asarray(xs, dtype="float64")),
        padded_limits(np.asarray(zs, dtype="float64")),
    )


def plot_paired_trajectory_pages(
    task_names, plotted_rows_by_task, paired_players, attempt_number, traj_figures_dir, png_dpi
):
    """
    Write a multi-page PDF with one 1×N page per sampled player, tasks in ``task_names`` order.

    Axis limits are fixed per task column and come from that task's canonical map
    points only. The PNG shows only the first page.
    """
    if not paired_players or len(task_names) < 2:
        return None, None

    traj_figures_dir.mkdir(parents=True, exist_ok=True)

    task_limits = {t: _taskmap_limits(t, plotted_rows_by_task) for t in task_names}

    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = traj_figures_dir / "paired_player_trajectories.pdf"
    png_path = traj_figures_dir / "paired_player_trajectories.png"

    pub_rc = _pub_rcparams()
    with PdfPages(pdf_path) as pdf:
        for page_idx, player in enumerate(paired_players):
            with plt.rc_context(pub_rc):
                fig, axes = plt.subplots(1, len(task_names), figsize=(8 * len(task_names), 8))

                for col_idx, (ax, task_name) in enumerate(zip(axes, task_names)):
                    plotted_rows = plotted_rows_by_task.get(task_name, [])
                    traj = player["tasks"][task_name]
                    x_lim, z_lim = task_limits[task_name]

                    h_subtask = _draw_subtask_background(ax, plotted_rows)
                    h_start, h_end = _draw_player_trajectory(ax, traj, color="#2166ac")

                    zigzag = compute_zigzag_score(traj["positions"])
                    path_length = compute_path_length(traj["positions"])
                    ax.set_title(
                        _subplot_title(task_name, traj, zigzag=zigzag, path_length=path_length),
                        fontsize=11,
                        pad=10,
                    )

                    legend_handles, legend_labels = [], []
                    if col_idx == 0:
                        if h_subtask is not None:
                            legend_handles.append(h_subtask)
                            legend_labels.append("Subtask location")
                        legend_handles += [h_start, h_end]
                        legend_labels += ["Start", "End"]

                    _style_traj_ax(ax, x_lim, z_lim, legend_handles, legend_labels)

                fig.suptitle(
                    "Player: {}".format(player["pid"]),
                    fontsize=13,
                    fontweight="bold",
                    y=1.01,
                )
                fig.text(
                    0.99,
                    0.005,
                    "{}/{}".format(page_idx + 1, len(paired_players)),
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color="#aaaaaa",
                )
                fig.tight_layout()

                pdf.savefig(fig, dpi=png_dpi, bbox_inches="tight")
                if page_idx == 0:
                    fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight", facecolor="white")
                plt.close(fig)

    return pdf_path, png_path


def build_trajectory_artifacts(
    input_dir,
    traj_figures_dir,
    task_names,
    n_players,
    random_state,
    attempt_number,
    task_rows_by_task,
    png_dpi,
    paired=False,
):
    """
    Generate trajectory PDFs for players with a usable ``attempt_number`` attempt in every task.

    ``paired=True`` (the CLI default, needs two or more tasks) writes one PDF with
    a 1×N page per player; ``paired=False`` writes one PDF per task.
    ``task_rows_by_task`` maps task name -> canonical subtask map rows.
    """
    plotted_rows_by_task = {
        task_name: [
            r for r in task_rows_by_task.get(task_name, []) if r.get("point_id") is not None
        ]
        for task_name in task_names
    }

    if paired:
        if len(task_names) < 2:
            print("WARNING: --traj_paired requires at least 2 tasks", file=sys.stderr)
            return
        paired_players = sample_paired_trajectory_players(
            input_dir, task_names, n_players, random_state, attempt_number
        )
        if not paired_players:
            print(
                "WARNING: no players with usable attempt{} trajectories found across all tasks".format(
                    attempt_number
                ),
                file=sys.stderr,
            )
            return
        pdf_path, png_path = plot_paired_trajectory_pages(
            task_names,
            plotted_rows_by_task,
            paired_players,
            attempt_number,
            traj_figures_dir,
            png_dpi,
        )
        if pdf_path is not None:
            print("Paired trajectories")
            print("  Players sampled : {:,}".format(len(paired_players)))
            print("  PDF             : {}".format(pdf_path))
            print("  PNG (page 1)    : {}".format(png_path))
        return

    # Sample a shared PID set so every per-task PDF shows the same players.
    paired_players = sample_paired_trajectory_players(
        input_dir, task_names, n_players, random_state, attempt_number
    )
    if not paired_players:
        print(
            "WARNING: no players with usable attempt{} trajectories found across all tasks".format(
                attempt_number
            ),
            file=sys.stderr,
        )
        return

    for task_name in task_names:
        sampled = [
            {
                "pid": p["pid"],
                "path": p["tasks"][task_name]["path"],
                "positions": p["tasks"][task_name]["positions"],
                "total_time_s": p["tasks"][task_name]["total_time_s"],
                "completion": p["tasks"][task_name]["completion"],
            }
            for p in paired_players
        ]
        pdf_path, png_path = plot_trajectory_pages(
            task_name,
            plotted_rows_by_task[task_name],
            sampled,
            attempt_number,
            traj_figures_dir,
            png_dpi,
        )
        if pdf_path is None:
            continue
        print("Task: {} (trajectories)".format(task_name))
        print("  Players sampled : {:,}".format(len(sampled)))
        print("  PDF             : {}".format(pdf_path))
        print("  PNG (page 1)    : {}".format(png_path))


def padded_limits(values):
    """Compute padded axis limits for one coordinate vector."""
    lower = float(np.min(values))
    upper = float(np.max(values))
    if lower == upper:
        pad = max(1.0, abs(lower) * 0.1 + 0.5)
        return lower - pad, upper + pad
    pad = max(0.5, (upper - lower) * 0.08)
    return lower - pad, upper + pad


def save_task_figure(task_name, rows, clean_attempt_count, figures_dir, png_dpi):
    """Render and save the task-map figure for one task."""
    plotted_rows = [row for row in rows if row["point_id"] is not None]
    if not plotted_rows:
        raise ValueError("Task '{}' has no plotted subtasks".format(task_name))

    figures_dir.mkdir(parents=True, exist_ok=True)
    color = TASK_COLORS.get(task_name, "#2f5d8a")

    x_values = np.asarray([row["x"] for row in plotted_rows], dtype="float64")
    z_values = np.asarray([row["z"] for row in plotted_rows], dtype="float64")
    x_limits = padded_limits(x_values)
    z_limits = padded_limits(z_values)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(
        x_values,
        z_values,
        s=72,
        color=color,
        edgecolors="#1f1f1f",
        linewidths=0.7,
        zorder=3,
    )

    for row in plotted_rows:
        offset = ANNOTATION_OFFSETS[(row["point_id"] - 1) % len(ANNOTATION_OFFSETS)]
        ax.annotate(
            str(row["point_id"]),
            xy=(row["x"], row["z"]),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "#4a4a4a",
                "linewidth": 0.5,
                "alpha": 0.95,
            },
            zorder=4,
        )

    canonical_count = len(rows)
    missing_count = canonical_count - len(plotted_rows)
    fig.suptitle(
        "{} Canonical Subtask Map".format(task_label(task_name)),
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    ax.set_title(
        "{} canonical subtasks | {:,} clean attempts | {} plotted".format(
            canonical_count,
            clean_attempt_count,
            len(plotted_rows),
        ),
        fontsize=13,
        pad=12,
    )
    ax.set_xlabel("x position")
    ax.set_ylabel("z position")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*z_limits)

    ax.set_axisbelow(True)
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.02,
        0.02,
        (
            "Point IDs map to `point_id` in subtask_task_map_points.csv. "
            "Coordinates are median first-completion x/z across clean attempts."
            + (
                " {} canonical subtasks had no parseable x/z position.".format(missing_count)
                if missing_count
                else ""
            )
        ),
        ha="left",
        va="bottom",
        fontsize=10,
    )
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.95])

    stem = figures_dir / "{}_subtask_task_map".format(task_name)
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return pdf_path, png_path


def summarize_task(
    task_name, total_attempt_files, canonical_subtasks, clean_attempts, rows, pdf_path, png_path
):
    """Print a compact runtime summary for one task."""
    plotted_count = sum(1 for row in rows if row["point_id"] is not None)
    print("Task: {}".format(task_name))
    print("  Attempt files scanned : {:,}".format(total_attempt_files))
    print("  Canonical set size    : {:,}".format(len(canonical_subtasks)))
    print("  Clean attempts        : {:,}".format(len(clean_attempts)))
    print("  Subtasks plotted      : {:,}".format(plotted_count))
    print("  PDF                   : {}".format(pdf_path))
    print("  PNG                   : {}".format(png_path))


def derive_task_rows(input_dir, task_names):
    """
    Derive canonical subtask map rows for each task without writing any output.

    Returns a dict mapping task_name -> list of row dicts (same structure as
    build_task_rows output). Used by trajectory generation when task-map figures
    are not needed.
    """
    task_rows_by_task = {}
    for task_name in task_names:
        attempt_files = discover_attempt_files(input_dir, task_name)
        if not attempt_files:
            continue
        attempt_summaries = [load_attempt_summary(path) for path in attempt_files]
        canonical_subtasks, _ = select_canonical_subtask_set(attempt_summaries)
        if not canonical_subtasks:
            continue
        rows, _, _ = build_task_rows(task_name, attempt_summaries, canonical_subtasks)
        task_rows_by_task[task_name] = rows
    return task_rows_by_task


def build_task_map_artifacts(input_dir, figures_dir, output_csv, task_names, png_dpi):
    """Run the task-map aggregation and write all requested artifacts."""
    all_rows = []
    task_rows_by_task = {}
    task_summaries = []
    for task_name in task_names:
        attempt_files = discover_attempt_files(input_dir, task_name)
        if not attempt_files:
            raise ValueError("No attempt CSVs found for task '{}'".format(task_name))

        attempt_summaries = [load_attempt_summary(path) for path in attempt_files]
        canonical_subtasks, modal_count = select_canonical_subtask_set(attempt_summaries)
        if not canonical_subtasks:
            raise ValueError("No modal canonical subtask set found for task '{}'".format(task_name))

        rows, clean_attempts, missing_position_subtasks = build_task_rows(
            task_name,
            attempt_summaries,
            canonical_subtasks,
        )
        if missing_position_subtasks:
            print(
                "WARNING: task '{}' had {} canonical subtasks without parseable x/z positions: {}".format(
                    task_name,
                    len(missing_position_subtasks),
                    ", ".join(missing_position_subtasks[:10])
                    + (" ..." if len(missing_position_subtasks) > 10 else ""),
                ),
                file=sys.stderr,
            )

        pdf_path, png_path = save_task_figure(
            task_name,
            rows,
            clean_attempt_count=len(clean_attempts),
            figures_dir=figures_dir,
            png_dpi=png_dpi,
        )
        summarize_task(
            task_name,
            total_attempt_files=len(attempt_files),
            canonical_subtasks=canonical_subtasks,
            clean_attempts=clean_attempts,
            rows=rows,
            pdf_path=pdf_path,
            png_path=png_path,
        )

        all_rows.extend(rows)
        task_rows_by_task[task_name] = rows
        task_summaries.append(
            {
                "task": task_name,
                "total_attempt_files": len(attempt_files),
                "modal_count": modal_count,
                "canonical_set_size": len(canonical_subtasks),
                "clean_attempt_count": len(clean_attempts),
                "pdf_path": pdf_path,
                "png_path": png_path,
            }
        )

    merged_rows = merge_existing_task_rows(all_rows, output_csv, task_names)
    write_output_csv(merged_rows, output_csv)
    print("Aggregated CSV: {}".format(output_csv))
    return all_rows, task_rows_by_task, task_summaries


def main(argv=None):
    """Run the canonical PowerWash subtask task-map analysis."""
    args = parse_args(argv)
    input_dir = args.input_dir.expanduser().resolve()
    figures_dir = args.figures_dir.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    traj_figures_dir = args.traj_figures_dir.expanduser().resolve()

    if args.png_dpi <= 0:
        print("ERROR: --png_dpi must be positive", file=sys.stderr)
        return 1
    if args.n_traj_players < 0:
        print("ERROR: --n_traj_players must be non-negative", file=sys.stderr)
        return 1
    if not input_dir.exists() or not input_dir.is_dir():
        print("ERROR: input directory does not exist: {}".format(input_dir), file=sys.stderr)
        return 1

    maybe_set_plot_style()
    task_rows_by_task = {}

    if not args.no_task_maps:
        try:
            _, task_rows_by_task, _ = build_task_map_artifacts(
                input_dir=input_dir,
                figures_dir=figures_dir,
                output_csv=output_csv,
                task_names=args.tasks,
                png_dpi=args.png_dpi,
            )
        except ValueError as exc:
            print("ERROR: {}".format(exc), file=sys.stderr)
            return 1

    if args.n_traj_players > 0:
        if not task_rows_by_task:
            task_rows_by_task = derive_task_rows(input_dir, args.tasks)
        build_trajectory_artifacts(
            input_dir=input_dir,
            traj_figures_dir=traj_figures_dir,
            task_names=args.tasks,
            n_players=args.n_traj_players,
            random_state=args.traj_random_state,
            attempt_number=args.traj_attempt,
            task_rows_by_task=task_rows_by_task,
            png_dpi=args.png_dpi,
            paired=args.traj_paired,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
