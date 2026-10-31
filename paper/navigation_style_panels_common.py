#!/usr/bin/env python3
"""Shared helpers for the navigation-style figure panels (Figures 3a and 3c)."""

import json
import math
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper.output_paths import resolve_figures_dir
from plvm import artifact_paths


DEFAULT_FIRE_STATION_ARTIFACT_DIR = artifact_paths.FIRESTATION_LABELS_DIR / "fold_00"

FIGURE_3A_FILENAME = "Figure_3a.pdf"
FIGURE_3C_FILENAME = "Figure_3c.pdf"

ZONE_PALETTE = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#999999",
]
LABEL_COLORS = {
    "Zone Planner": "#377eb8",
    "Mixed": "#4daf4a",
    "Zone Hopper": "#e41a1c",
}
SCORE_HISTOGRAM_BIN_WIDTH = 0.5
PUBLICATION_PANEL_WIDTH_IN = 2.13
PUBLICATION_HISTOGRAM_HEIGHT_IN = 2.10
PUBLICATION_PANEL_HEIGHT_IN = PUBLICATION_HISTOGRAM_HEIGHT_IN
PUBLICATION_LABEL_FONTSIZE = 7.0
PUBLICATION_TICK_FONTSIZE = 6.0
PUBLICATION_LEGEND_FONTSIZE = 5.8
PUBLICATION_ZONE_NUMBER_FONTSIZE = 8.8
PUBLICATION_ZONE_CENTROID_MARKER_SIZE = 120
PUBLICATION_LIGHT_BORDER_COLOR = "#d1d5db"
PUBLICATION_ZONE_NUMBER_OUTLINE_COLOR = "#1f2937"
PUBLICATION_SAVEFIG_DPI = 600


def resolve_fire_station_output_paths(
    output_dir: Optional[Path],
    figure_3a_pdf: Optional[Path],
    figure_3c_pdf: Optional[Path],
) -> Tuple[Path, Path]:
    """Resolve the output paths for Figures 3a and 3c."""
    resolved_output_dir = resolve_figures_dir(output_dir)
    zones_path = (
        Path(figure_3a_pdf)
        if figure_3a_pdf is not None
        else resolved_output_dir / FIGURE_3A_FILENAME
    )
    score_path = (
        Path(figure_3c_pdf)
        if figure_3c_pdf is not None
        else resolved_output_dir / FIGURE_3C_FILENAME
    )
    return zones_path, score_path


def load_fire_station_artifacts(
    artifact_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load the saved Fire Station label and zone-map artifacts needed for plotting."""
    resolved_artifact_dir = Path(artifact_dir).expanduser().resolve()
    labels_path = resolved_artifact_dir / "firestation_zone_labels.csv"
    zone_map_path = resolved_artifact_dir / "firestation_subtask_zone_map.csv"
    centroids_path = resolved_artifact_dir / "firestation_zone_centroids.csv"

    missing_paths = [
        path for path in (labels_path, zone_map_path, centroids_path) if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Missing Fire Station artifacts: {}.".format(
                ", ".join(str(path) for path in missing_paths)
            )
        )

    labeled_df = pd.read_csv(labels_path)
    df_map = pd.read_csv(zone_map_path)
    centroids_df = pd.read_csv(centroids_path)

    required_label_columns = {"label", "zone_hopping_score"}
    missing_label_columns = sorted(required_label_columns.difference(labeled_df.columns))
    if missing_label_columns:
        raise ValueError(
            "Missing required Fire Station label columns: {}.".format(
                ", ".join(missing_label_columns)
            )
        )

    required_map_columns = {"x", "z", "zone_id"}
    missing_map_columns = sorted(required_map_columns.difference(df_map.columns))
    if missing_map_columns:
        raise ValueError(
            "Missing required Fire Station zone-map columns: {}.".format(
                ", ".join(missing_map_columns)
            )
        )

    required_centroid_columns = {"centroid_x", "centroid_z"}
    missing_centroid_columns = sorted(required_centroid_columns.difference(centroids_df.columns))
    if missing_centroid_columns:
        raise ValueError(
            "Missing required Fire Station centroid columns: {}.".format(
                ", ".join(missing_centroid_columns)
            )
        )

    zone_centroids = centroids_df[["centroid_x", "centroid_z"]].to_numpy(dtype="float64")
    return labeled_df, df_map, zone_centroids


def _configure_powerwash_plot_style() -> None:
    """Apply compact static plotting defaults for the Fire Station panels."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _zone_color(zone_id: int) -> str:
    """Return the fixed color for one zone id."""
    return ZONE_PALETTE[int(zone_id) % len(ZONE_PALETTE)]


def _score_histogram_bin_edges(
    scores: pd.Series,
    bin_width: float = SCORE_HISTOGRAM_BIN_WIDTH,
) -> np.ndarray:
    """Return fixed-width histogram bin edges spanning all finite score values."""
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")

    numeric_scores = pd.to_numeric(scores, errors="coerce")
    finite_scores = numeric_scores[np.isfinite(numeric_scores.to_numpy(dtype="float64"))]
    if finite_scores.empty:
        half_width = bin_width / 2.0
        return np.asarray([-half_width, half_width], dtype="float64")

    lower = math.floor(float(finite_scores.min()) / bin_width) * bin_width
    upper = math.ceil(float(finite_scores.max()) / bin_width) * bin_width
    if math.isclose(lower, upper):
        upper = lower + bin_width

    n_bins = max(1, int(round((upper - lower) / bin_width)))
    return lower + (np.arange(n_bins + 1, dtype="float64") * bin_width)


def _save_publication_figure(fig: plt.Figure, path: Path) -> None:
    """Save one fixed-size figure."""
    fig.savefig(path, dpi=PUBLICATION_SAVEFIG_DPI)


def _apply_publication_axis_style(
    ax: plt.Axes,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    show_ticks: bool = True,
) -> None:
    """Apply the compact axis styling used by the Fire Station panels."""
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=PUBLICATION_LABEL_FONTSIZE, labelpad=1.5)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=PUBLICATION_LABEL_FONTSIZE, labelpad=1.5)

    if show_ticks:
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=PUBLICATION_TICK_FONTSIZE,
            length=2.5,
            pad=1.2,
        )
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def _three_class_publication_labels(labeled_df: pd.DataFrame) -> Sequence[str]:
    """Return compact legend labels with class counts for the score distribution."""
    labels = []
    for long_label, short_label in (
        ("Zone Planner", "Planner"),
        ("Mixed", "Mixed"),
        ("Zone Hopper", "Hopper"),
    ):
        count = int((labeled_df["label"] == long_label).sum())
        labels.append("{} (n = {})".format(short_label, count))
    return labels


def render_fire_station_zones(
    df_map: pd.DataFrame,
    zone_centroids: np.ndarray,
    output_path: Path,
) -> Path:
    """Render the Fire Station zone panel (Figure 3a) to the requested path."""
    _configure_powerwash_plot_style()
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(PUBLICATION_PANEL_WIDTH_IN, PUBLICATION_PANEL_HEIGHT_IN))
    x_min = float(df_map["x"].min())
    x_max = float(df_map["x"].max())
    z_min = float(df_map["z"].min())
    z_max = float(df_map["z"].max())
    x_pad = max((x_max - x_min) * 0.003, 0.05)
    z_pad = max((z_max - z_min) * 0.003, 0.05)

    for zone_id in sorted(df_map["zone_id"].unique()):
        sub = df_map[df_map["zone_id"] == zone_id]
        ax.scatter(
            sub["x"],
            sub["z"],
            s=9,
            alpha=0.88,
            linewidths=0,
            color=_zone_color(zone_id),
            zorder=2,
        )
        ax.scatter(
            zone_centroids[int(zone_id), 0],
            zone_centroids[int(zone_id), 1],
            s=PUBLICATION_ZONE_CENTROID_MARKER_SIZE,
            marker="o",
            color=_zone_color(zone_id),
            edgecolors=PUBLICATION_LIGHT_BORDER_COLOR,
            linewidths=0.85,
            zorder=4,
        )
        zone_label = ax.text(
            zone_centroids[int(zone_id), 0],
            zone_centroids[int(zone_id), 1],
            str(int(zone_id)),
            ha="center",
            va="center",
            fontsize=PUBLICATION_ZONE_NUMBER_FONTSIZE,
            fontweight="bold",
            color="white",
            zorder=5,
        )
        zone_label.set_path_effects(
            [
                patheffects.withStroke(
                    linewidth=1.1,
                    foreground=PUBLICATION_ZONE_NUMBER_OUTLINE_COLOR,
                    alpha=0.8,
                )
            ]
        )

    ax.set_aspect("auto")
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(z_min - z_pad, z_max + z_pad)
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_edgecolor(PUBLICATION_LIGHT_BORDER_COLOR)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    _save_publication_figure(fig, resolved_output_path)
    plt.close(fig)
    return resolved_output_path


def load_label_thresholds(artifact_dir: Path) -> Optional[Tuple[float, float]]:
    """Return the (lower, upper) score thresholds of ``label_thresholds.json``, or None if absent."""
    path = Path(artifact_dir).expanduser().resolve() / "label_thresholds.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return float(data["lower_threshold"]), float(data["upper_threshold"])


def render_navigation_style_score_distribution(
    labeled_df: pd.DataFrame,
    output_path: Path,
    thresholds: Optional[Tuple[float, float]] = None,
) -> Path:
    """Render the Fire Station 3-class score distribution (Figure 3c) to one PDF.

    ``thresholds`` = (lower, upper) score cut-offs between the classes; when given, each is
    drawn as a dashed vertical line in the colour of the class beyond it (Planner below the
    lower one, Hopper above the upper one).
    """
    _configure_powerwash_plot_style()
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    bins = _score_histogram_bin_edges(
        labeled_df["zone_hopping_score"],
        bin_width=SCORE_HISTOGRAM_BIN_WIDTH,
    )
    planner_color = LABEL_COLORS["Zone Planner"]
    mixed_color = LABEL_COLORS["Mixed"]
    hopper_color = LABEL_COLORS["Zone Hopper"]

    fig, ax = plt.subplots(figsize=(PUBLICATION_PANEL_WIDTH_IN, PUBLICATION_HISTOGRAM_HEIGHT_IN))
    for label, color in (
        ("Zone Planner", planner_color),
        ("Mixed", mixed_color),
        ("Zone Hopper", hopper_color),
    ):
        sub = labeled_df.loc[labeled_df["label"] == label, "zone_hopping_score"]
        if not sub.empty:
            ax.hist(sub, bins=bins, alpha=0.7, color=color, label=label)
    if thresholds is not None:
        lower, upper = thresholds
        ax.axvline(lower, color=planner_color, linestyle="--", linewidth=1.4)
        ax.axvline(upper, color=hopper_color, linestyle="--", linewidth=1.4)

    ax.grid(axis="y", linewidth=0.35, alpha=0.25)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    ax.set_xlim(float(bins[0]), float(bins[-1]))
    _apply_publication_axis_style(ax, xlabel="Score", ylabel="Players")

    planner_label, mixed_label, hopper_label = _three_class_publication_labels(labeled_df)
    ax.legend(
        handles=[
            mpatches.Patch(color=planner_color, label=planner_label),
            mpatches.Patch(color=mixed_color, label=mixed_label),
            mpatches.Patch(color=hopper_color, label=hopper_label),
        ],
        fontsize=PUBLICATION_LEGEND_FONTSIZE,
        frameon=False,
        loc="best",
        handlelength=0.9,
        labelspacing=0.25,
        handletextpad=0.4,
    )

    fig.tight_layout(pad=0.25)
    _save_publication_figure(fig, resolved_output_path)
    plt.close(fig)
    return resolved_output_path
