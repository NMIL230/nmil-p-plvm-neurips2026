#!/usr/bin/env python3
"""
Generate Figure 5c: the 2-D PLVM latent space coloured by zone-planning label.

Reuses saved PowerWash PLVM runs (no retraining). By default it picks the
highest-validation-accuracy ``plvm_both_partial`` run with ``latent_dim=2`` on
the clean 2-class zone-planning target that saved a train latent-space CSV, and
plots its training-split latents at the 1800 s cutoff. Outputs go to the
PowerWash model-comparison figures folder, with a copy at
``manuscript_figures/figures/Figure_5c.pdf``.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper.output_paths import copy_figure_to_path, resolve_figures_dir  # noqa: E402
from plvm import artifact_paths  # noqa: E402
from powerwash.analysis.plot_subtask_task_maps import maybe_set_plot_style  # noqa: E402
from powerwash.data.create_firestation_zone_planning_labels import LABEL_COLORS  # noqa: E402

DEFAULT_TRAINING_LOGS_DIR = artifact_paths.powerwash_family_logs_dir("plvm_models")
DEFAULT_MODEL_PERFORMANCE_CSV = DEFAULT_TRAINING_LOGS_DIR / "model_performance.csv"
DEFAULT_OUTPUT_DIR = artifact_paths.POWERWASH_MODEL_COMPARISON_FIGURES_DIR
DEFAULT_MANUSCRIPT_OUTPUT_PDF = resolve_figures_dir() / "Figure_5c.pdf"
DEFAULT_PARAMETERS_TO_PREDICT = "player_label_2_class_clean_zonehopping_label"
DEFAULT_MODEL_TYPE = "plvm_both_partial"
DEFAULT_LATENT_DIM = 2
DEFAULT_CUTOFF_SECONDS = 1800.0
DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_SCATTER_POINTS = 4000
DEFAULT_MAX_KDE_POINTS_PER_LABEL = 5000
MANUSCRIPT_FIGURE_WIDTH_INCHES = 2.13
MANUSCRIPT_FIGURE_HEIGHT_INCHES = 2.2
CAPTION_TEXT = "(c) Latent Representation"
SCATTER_MARKER_SIZE = 9.0
SCATTER_ALPHA = 0.28
OUTPUT_PNG_DPI = 600
AXIS_TICK_INTERVAL = 0.5
PLANNER_LABEL = "Zone Planner"
HOPPER_LABEL = "Zone Hopper"


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training_logs_dir",
        type=Path,
        default=DEFAULT_TRAINING_LOGS_DIR,
        help="Directory containing saved PowerWash PLVM run subdirectories "
        "(default: <artifact root>/models/powerwash/hierarchical/training_logs/plvm_models).",
    )
    parser.add_argument(
        "--model_performance_csv",
        type=Path,
        default=DEFAULT_MODEL_PERFORMANCE_CSV,
        help="Run registry used to select the best saved run "
        "(default: model_performance.csv in the default --training_logs_dir).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help="Optional explicit saved run name. If set, skips automatic best-run selection.",
    )
    parser.add_argument(
        "--parameters_to_predict",
        type=str,
        default=DEFAULT_PARAMETERS_TO_PREDICT,
        help="Classification target used when selecting the best saved run.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=DEFAULT_MODEL_TYPE,
        help="Model type used when selecting the best saved run.",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=DEFAULT_LATENT_DIM,
        help="Latent dimensionality used when selecting the best saved run.",
    )
    parser.add_argument(
        "--cutoff_seconds",
        type=float,
        default=DEFAULT_CUTOFF_SECONDS,
        help="Single observation cutoff to plot (one row per player).",
    )
    parser.add_argument(
        "--max_scatter_points",
        type=int,
        default=DEFAULT_MAX_SCATTER_POINTS,
        help="Maximum number of points to draw in the scatter layers.",
    )
    parser.add_argument(
        "--max_kde_points_per_label",
        type=int,
        default=DEFAULT_MAX_KDE_POINTS_PER_LABEL,
        help="Maximum number of points per label used when fitting KDE contours.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used for any scatter/KDE subsampling.",
    )
    parser.add_argument(
        "--output_pdf",
        type=Path,
        default=None,
        help="Optional output PDF path. Defaults under "
        "<artifact root>/powerwash/analysis/figures/model_comparisons/.",
    )
    parser.add_argument(
        "--output_png",
        type=Path,
        default=None,
        help="Optional output PNG path. Defaults next to the PDF.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional output CSV path for the joined latent metadata.",
    )
    parser.add_argument(
        "--manuscript_output_pdf",
        type=Path,
        default=None,
        help=(
            "Path of the Figure 5c PDF copy. Defaults to manuscript_figures/figures/Figure_5c.pdf."
        ),
    )
    return parser.parse_args(argv)


def resolve_output_paths(
    output_dir: Path,
    cutoff_seconds: float,
    output_pdf: Optional[Path],
    output_png: Optional[Path],
    output_csv: Optional[Path],
) -> Tuple[Path, Path, Path]:
    """Resolve default artifact paths for one latent-space plotting run."""
    cutoff_tag = "{}s".format(int(round(float(cutoff_seconds))))
    if output_pdf is None:
        pdf_path = output_dir / "powerwash_plvm_best_ld2_latent_space_{}.pdf".format(cutoff_tag)
    else:
        pdf_path = Path(output_pdf)
    if output_png is None:
        png_path = pdf_path.with_suffix(".png")
    else:
        png_path = Path(output_png)
    if output_csv is None:
        stem = (
            pdf_path.stem
            if output_pdf is not None
            else "powerwash_plvm_best_ld2_latent_space_{}".format(cutoff_tag)
        )
        csv_path = pdf_path.parent / "{}.csv".format(stem)
    else:
        csv_path = Path(output_csv)
    return pdf_path, png_path, csv_path


def resolve_manuscript_output_pdf(manuscript_output_pdf: Optional[Path]) -> Path:
    """Resolve the path of the Figure 5c PDF copy."""
    if manuscript_output_pdf is None:
        return DEFAULT_MANUSCRIPT_OUTPUT_PDF
    return Path(manuscript_output_pdf)


def manuscript_figure_size_inches(
    manuscript_width: float = MANUSCRIPT_FIGURE_WIDTH_INCHES,
    manuscript_height: float = MANUSCRIPT_FIGURE_HEIGHT_INCHES,
) -> Tuple[float, float]:
    """Return the Figure 5c size in inches."""
    if manuscript_width <= 0.0 or manuscript_height <= 0.0:
        raise ValueError("Figure 5c dimensions must be positive.")
    return float(manuscript_width), float(manuscript_height)


def apply_latent_space_plot_style() -> None:
    """Apply a compact style tuned for a one-column latent-space figure."""
    maybe_set_plot_style()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
            "font.size": 6.2,
            "axes.labelsize": 6.4,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "figure.dpi": OUTPUT_PNG_DPI,
            "savefig.dpi": OUTPUT_PNG_DPI,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _safe_float(raw_value) -> float:
    """Return a numeric scalar or ``nan`` when conversion fails."""
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return float("nan")


def load_model_performance_rows(model_performance_csv: Path) -> List[Dict[str, str]]:
    """Load the saved PowerWash model-performance rows."""
    model_performance_csv = Path(model_performance_csv).expanduser().resolve()
    with model_performance_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _latent_csv_exists(training_logs_dir: Path, model_name: str) -> bool:
    """Return whether one saved run has the required train latent CSV."""
    run_dir = Path(training_logs_dir) / str(model_name)
    return (run_dir / "plots" / "train_classification_latent_space_plot_data.csv").exists()


def select_best_run(
    rows: Sequence[Dict[str, str]],
    training_logs_dir: Path,
    parameters_to_predict: str,
    model_type: str,
    latent_dim: int,
) -> Dict[str, str]:
    """Select the best saved classification run matching the requested filters."""
    candidates: List[Tuple[float, Dict[str, str]]] = []
    for row in rows:
        if str(row.get("model_type", "")).strip() != str(model_type):
            continue
        if str(row.get("parameters_to_predict", "")).strip() != str(parameters_to_predict):
            continue
        if int(_safe_float(row.get("latent_dim"))) != int(latent_dim):
            continue
        model_name = str(row.get("model_name", "")).strip()
        if not model_name or not _latent_csv_exists(training_logs_dir, model_name):
            continue
        val_accuracy = _safe_float(row.get("val_accuracy"))
        if not math.isfinite(val_accuracy):
            continue
        candidates.append((val_accuracy, row))
    if not candidates:
        raise ValueError(
            "No saved runs matched model_type='{}', parameters_to_predict='{}', latent_dim={} "
            "with a train latent CSV under {}.".format(
                model_type,
                parameters_to_predict,
                latent_dim,
                training_logs_dir,
            )
        )
    candidates.sort(key=lambda item: (item[0], str(item[1].get("model_name", ""))), reverse=True)
    return candidates[0][1]


def resolve_run_dir(training_logs_dir: Path, model_name: str) -> Path:
    """Resolve one saved run directory and verify that it exists."""
    run_dir = Path(training_logs_dir).expanduser().resolve() / str(model_name)
    if not run_dir.exists():
        raise FileNotFoundError("Saved run directory does not exist: {}".format(run_dir))
    return run_dir


def _candidate_frame_from_payload(
    payload: Dict[str, object],
    split_name: str,
) -> Optional[pd.DataFrame]:
    """Return the candidate rows for one split as a frame, if available."""
    split_rows = payload.get(str(split_name))
    if not isinstance(split_rows, list) or not split_rows:
        return None
    candidate_frame = pd.DataFrame(split_rows)
    candidate_frame.insert(0, "sample_index", np.arange(len(candidate_frame), dtype=np.int64))
    return candidate_frame


def load_latent_plot_frame(run_dir: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load saved latent coordinates and join per-sample candidate metadata when possible."""
    latent_csv = run_dir / "plots" / "train_classification_latent_space_plot_data.csv"
    metadata_json = run_dir / "plots" / "train_classification_latent_space_plot_metadata.json"
    candidate_splits_json = run_dir / "candidate_splits.json"
    if not latent_csv.exists():
        raise FileNotFoundError("Missing latent CSV: {}".format(latent_csv))
    if not metadata_json.exists():
        raise FileNotFoundError("Missing latent metadata JSON: {}".format(metadata_json))

    frame = pd.read_csv(latent_csv)
    with metadata_json.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    split_name = str(metadata.get("split", "train") or "train").strip().lower()
    if candidate_splits_json.exists():
        with candidate_splits_json.open("r", encoding="utf-8") as handle:
            candidate_payload = json.load(handle)
        candidate_frame = _candidate_frame_from_payload(candidate_payload, split_name)
        if candidate_frame is not None and len(candidate_frame) == len(frame):
            frame = frame.merge(
                candidate_frame, on="sample_index", how="left", validate="one_to_one"
            )

    return frame, metadata


def filter_frame_to_cutoff(frame: pd.DataFrame, cutoff_seconds: float) -> pd.DataFrame:
    """Keep only rows from one cutoff and verify one observation per player when possible."""
    if "cutoff_seconds" not in frame.columns:
        raise ValueError("Latent frame does not contain a cutoff_seconds column.")
    cutoff_seconds = float(cutoff_seconds)
    filtered = frame[
        np.isclose(frame["cutoff_seconds"].to_numpy(dtype=np.float64), cutoff_seconds)
    ].copy()
    if filtered.empty:
        available_cutoffs = sorted(set(frame["cutoff_seconds"].dropna().tolist()))
        raise ValueError(
            "No latent rows found at cutoff_seconds={}. Available cutoffs: {}".format(
                cutoff_seconds,
                available_cutoffs,
            )
        )
    if "pid" in filtered.columns:
        duplicate_counts = filtered["pid"].value_counts()
        duplicate_players = duplicate_counts[duplicate_counts > 1]
        if not duplicate_players.empty:
            raise ValueError(
                "Expected one observation per player at cutoff_seconds={}, but found duplicates for {} players.".format(
                    cutoff_seconds,
                    len(duplicate_players),
                )
            )
    return filtered.reset_index(drop=True)


def sample_frame_for_scatter(
    frame: pd.DataFrame,
    max_points: int,
    random_seed: int,
    label_column: str = "true_class_label",
) -> pd.DataFrame:
    """Return a label-stratified scatter sample."""
    if max_points <= 0 or len(frame) <= max_points:
        return frame.copy()
    pieces: List[pd.DataFrame] = []
    group_sizes = frame[label_column].value_counts(sort=False)
    total_count = float(group_sizes.sum())
    base_seed = int(random_seed)
    for group_index, (label_value, group_count) in enumerate(group_sizes.items()):
        group = frame[frame[label_column] == label_value]
        group_target = max(1, int(round(max_points * (float(group_count) / total_count))))
        group_target = min(group_target, len(group))
        pieces.append(group.sample(n=group_target, random_state=base_seed + group_index))
    sampled = pd.concat(pieces, axis=0, ignore_index=False)
    if len(sampled) > max_points:
        sampled = sampled.sample(n=max_points, random_state=base_seed + 999)
    return sampled.sort_values("sample_index").reset_index(drop=True)


def _axis_limits(frame: pd.DataFrame) -> Tuple[float, float, float, float]:
    """Compute square plotting limits with a small shared margin."""
    x_values = frame["latent_x"].to_numpy(dtype=np.float64)
    y_values = frame["latent_y"].to_numpy(dtype=np.float64)
    x_min = float(np.nanmin(x_values))
    x_max = float(np.nanmax(x_values))
    y_min = float(np.nanmin(y_values))
    y_max = float(np.nanmax(y_values))
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    half_span = 0.5 * max(x_max - x_min, y_max - y_min)
    pad = max(0.08 * (2.0 * half_span), 0.12)
    half_extent = half_span + pad
    return (
        x_center - half_extent,
        x_center + half_extent,
        y_center - half_extent,
        y_center + half_extent,
    )


def _label_sample(
    frame: pd.DataFrame,
    label_name: str,
    max_points: int,
    random_seed: int,
) -> pd.DataFrame:
    """Return up to ``max_points`` rows for one label."""
    label_frame = frame[frame["true_class_label"] == label_name]
    if max_points <= 0 or len(label_frame) <= max_points:
        return label_frame
    return label_frame.sample(n=max_points, random_state=random_seed)


def add_kde_contours(
    ax,
    frame: pd.DataFrame,
    label_name: str,
    color: str,
    limits: Tuple[float, float, float, float],
    max_points: int,
    random_seed: int,
    line_alpha: float = 0.35,
    fill_alphas: Tuple[float, float, float, float] = (0.04, 0.07, 0.10, 0.14),
    linewidths: float = 0.8,
) -> None:
    """Overlay filled 2D KDE contours for one class label."""
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        return

    label_frame = _label_sample(frame, label_name, max_points, random_seed)
    if len(label_frame) < 8:
        return

    xy = label_frame[["latent_x", "latent_y"]].to_numpy(dtype=np.float64).T
    if np.allclose(xy[0], xy[0][0]) or np.allclose(xy[1], xy[1][0]):
        return

    try:
        kde = gaussian_kde(xy)
    except np.linalg.LinAlgError:
        return

    x_min, x_max, y_min, y_max = limits
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 220),
        np.linspace(y_min, y_max, 220),
    )
    density = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
    peak = float(np.nanmax(density))
    if not math.isfinite(peak) or peak <= 0.0:
        return
    density = density / peak
    fill_levels = [0.12, 0.28, 0.48, 0.68, 1.01]
    fill_colors = [to_rgba(color, alpha=a) for a in fill_alphas]
    ax.contourf(grid_x, grid_y, density, levels=fill_levels, colors=fill_colors, antialiased=True)
    ax.contour(
        grid_x,
        grid_y,
        density,
        levels=[0.20, 0.40, 0.60, 0.80],
        colors=[to_rgba(color, alpha=line_alpha)],
        linewidths=linewidths,
    )


def plot_latent_space_figure(
    frame: pd.DataFrame,
    output_pdf: Path,
    output_png: Path,
    max_scatter_points: int,
    max_kde_points_per_label: int,
    random_seed: int,
) -> None:
    """Render and save a single-panel latent-space figure."""
    apply_latent_space_plot_style()

    scatter_frame = sample_frame_for_scatter(frame, max_scatter_points, random_seed)
    limits = _axis_limits(frame)
    planner_color = LABEL_COLORS[PLANNER_LABEL]
    hopper_color = LABEL_COLORS[HOPPER_LABEL]
    fig, ax = plt.subplots(figsize=manuscript_figure_size_inches())
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.18, right=0.99, bottom=0.22, top=0.97)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.set_xlabel("Latent dimension 1")
    ax.set_ylabel("Latent dimension 2")
    ax.xaxis.set_major_locator(MultipleLocator(AXIS_TICK_INTERVAL))
    ax.yaxis.set_major_locator(MultipleLocator(AXIS_TICK_INTERVAL))
    ax.tick_params(length=2.4, width=0.6, pad=1.4)

    add_kde_contours(
        ax,
        frame,
        HOPPER_LABEL,
        hopper_color,
        limits,
        max_kde_points_per_label,
        random_seed + 23,
    )
    add_kde_contours(
        ax,
        frame,
        PLANNER_LABEL,
        planner_color,
        limits,
        max_kde_points_per_label,
        random_seed + 11,
        line_alpha=0.65,
        fill_alphas=(0.08, 0.14, 0.20, 0.28),
        linewidths=1.1,
    )

    # Draw both classes in one randomly interleaved call so that neither class is always
    # drawn on top of the other where points overlap.
    color_by_label = {PLANNER_LABEL: planner_color, HOPPER_LABEL: hopper_color}
    interleaved_frame = scatter_frame.sample(frac=1.0, random_state=random_seed).reset_index(
        drop=True
    )
    ax.scatter(
        interleaved_frame["latent_x"],
        interleaved_frame["latent_y"],
        s=SCATTER_MARKER_SIZE,
        alpha=SCATTER_ALPHA,
        linewidths=0.0,
        c=interleaved_frame["true_class_label"].map(color_by_label),
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=planner_color,
            markersize=5.8,
            label=PLANNER_LABEL,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=hopper_color,
            markersize=5.8,
            label=HOPPER_LABEL,
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        borderpad=0.28,
        handletextpad=0.35,
        labelspacing=0.25,
    )

    pos = ax.get_position()
    fig.text(
        (pos.x0 + pos.x1) / 2.0,
        0.03,
        CAPTION_TEXT,
        ha="center",
        va="bottom",
        fontsize=7.0,
        fontweight="bold",
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=OUTPUT_PNG_DPI)
    plt.close(fig)


def attach_run_metadata(frame: pd.DataFrame, selected_row: Dict[str, str]) -> pd.DataFrame:
    """Attach a few useful run-level metadata columns to the exported frame."""
    output = frame.copy()
    output["selected_model_name"] = str(selected_row.get("model_name", "")).strip()
    output["selected_model_type"] = str(selected_row.get("model_type", "")).strip()
    output["selected_parameters_to_predict"] = str(
        selected_row.get("parameters_to_predict", "")
    ).strip()
    output["selected_val_accuracy"] = _safe_float(selected_row.get("val_accuracy"))
    return output


def main(argv=None) -> int:
    """Run the artifact-driven latent-space plotting workflow."""
    args = parse_args(argv)
    rows = load_model_performance_rows(args.model_performance_csv)
    training_logs_dir = Path(args.training_logs_dir).expanduser().resolve()

    if str(args.model_name).strip():
        selected_row = next(
            (
                row
                for row in rows
                if str(row.get("model_name", "")).strip() == str(args.model_name).strip()
            ),
            None,
        )
        if selected_row is None:
            raise ValueError(
                "No row with model_name='{}' found in {}.".format(
                    args.model_name,
                    args.model_performance_csv,
                )
            )
    else:
        selected_row = select_best_run(
            rows=rows,
            training_logs_dir=training_logs_dir,
            parameters_to_predict=args.parameters_to_predict,
            model_type=args.model_type,
            latent_dim=args.latent_dim,
        )

    model_name = str(selected_row.get("model_name", "")).strip()
    run_dir = resolve_run_dir(training_logs_dir, model_name)
    frame, metadata = load_latent_plot_frame(run_dir)
    frame = filter_frame_to_cutoff(frame, args.cutoff_seconds)

    output_pdf, output_png, output_csv = resolve_output_paths(
        output_dir=DEFAULT_OUTPUT_DIR,
        cutoff_seconds=float(args.cutoff_seconds),
        output_pdf=args.output_pdf,
        output_png=args.output_png,
        output_csv=args.output_csv,
    )
    manuscript_output_pdf = resolve_manuscript_output_pdf(args.manuscript_output_pdf)

    plot_latent_space_figure(
        frame=frame,
        output_pdf=output_pdf,
        output_png=output_png,
        max_scatter_points=int(args.max_scatter_points),
        max_kde_points_per_label=int(args.max_kde_points_per_label),
        random_seed=int(args.random_seed),
    )
    attach_run_metadata(frame, selected_row).to_csv(output_csv, index=False)
    if output_pdf.expanduser().resolve() != manuscript_output_pdf.expanduser().resolve():
        copy_figure_to_path(output_pdf, manuscript_output_pdf)

    print("Selected model:", model_name)
    print("Validation accuracy:", "{:.4f}".format(_safe_float(selected_row.get("val_accuracy"))))
    print("Cutoff seconds:", "{:.1f}".format(float(args.cutoff_seconds)))
    print("PDF:", output_pdf)
    print("PNG:", output_png)
    print("CSV:", output_csv)
    print("Figure 5c PDF:", manuscript_output_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
