#!/usr/bin/env python3
"""Generates Figure 7: simulation accuracy over observed steps per condition.

Its loaders are also used by Figure 5 and several tables. Conditions come from
the PLVM, outcome random-forest baseline, and the two logistic-probe families
(both-task "Logistic Fuser" and single-task). Each condition is one run (no
folds): the highest validation accuracy, ties broken by train accuracy.

Analysis outputs go under ``<artifact root>/analysis/model_training_analysis/``;
``--manuscript_figure`` also renders ``manuscript_figures/figures/Figure_7.pdf``
(only for ``--max_train_val_players 8000``, the default).

    python paper/figures/fig07_simulation_accuracy_curves.py --manuscript_figure
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from paper.plot_style import maybe_set_plot_style
from paper.output_paths import (
    resolve_figures_dir,
)
from plvm import artifact_paths


DEFAULT_PLVM_LOGS_DIR = artifact_paths.simulation_family_logs_dir("plvm_label")
DEFAULT_PLVM_PERF_CSV = DEFAULT_PLVM_LOGS_DIR / "model_performance.csv"
DEFAULT_OUTCOME_LOGS_DIR = artifact_paths.simulation_family_logs_dir(
    "outcome_observation_baseline"
)
DEFAULT_OUTCOME_PERF_CSV = DEFAULT_OUTCOME_LOGS_DIR / "model_performance.csv"
DEFAULT_LOGISTIC_LOGS_DIR = artifact_paths.simulation_family_logs_dir(
    "plvm_label_concat_logistic_probe"
)
DEFAULT_LOGISTIC_PERF_CSV = DEFAULT_LOGISTIC_LOGS_DIR / "model_performance.csv"
DEFAULT_SINGLE_TASK_LOGISTIC_LOGS_DIR = artifact_paths.simulation_family_logs_dir(
    "plvm_label_single_task_logistic_probe"
)
DEFAULT_SINGLE_TASK_LOGISTIC_PERF_CSV = (
    DEFAULT_SINGLE_TASK_LOGISTIC_LOGS_DIR / "model_performance.csv"
)
DEFAULT_OUTPUT_ROOT = artifact_paths.SIMULATION_PERFORMANCE_ANALYSIS_DIR
DEFAULT_MANUSCRIPT_FIGURES_DIR = resolve_figures_dir()

PROCESS_SINGLE_CONDITIONS = [
    ("Process: Foraging", "foraging", "#d67c4a"),
    ("Process: Cleaning", "cleaning", "#b84f2a"),
]
OUTCOME_SINGLE_CONDITIONS = [
    ("Outcome: Foraging", "foraging", "#7fb069"),
    ("Outcome: Cleaning", "cleaning", "#4c956c"),
]
OUTCOME_BOTH_CONDITION = ("Outcome: Both", "cleaning,foraging", "#2d6a4f")
LOGISTIC_FUSER_CONDITION = ("Process: Logistic Fuser", "cleaning,foraging", "2", "#7c6fb3")
SINGLE_TASK_LOGISTIC_CONDITIONS = [
    ("Process: Foraging (Logistic)", "foraging", "#e3a857"),
    ("Process: Cleaning (Logistic)", "cleaning", "#8f4a2e"),
]
# The "condition" names above are lookup keys (also used by fig05_combined_overview.py);
# CONDITION_LABELS below maps them to legend text.
# Sort keys for legend ordering.
LEGEND_SORT_ALIAS = {
    "Process: Single Task (Foraging)": "Single task (Foraging), nonlinear head",
    "Process: Single Task (Cleaning)": "Single task (Cleaning), nonlinear head",
    "Process: Single Task (Foraging), linear head": "Single task (Foraging), linear head",
    "Process: Single Task (Cleaning), linear head": "Single task (Cleaning), linear head",
    "Process: PLVM, linear fusion": "Both tasks, linear fusion",
    "Process: PLVM": "Both tasks, nonlinear fusion (PLVM-2)",
}
LEGEND_SORT_ALIAS.update(
    {"Process: PLVM-%d" % k: "Both tasks, nonlinear fusion (PLVM-%d)" % k for k in (1, 3, 4, 5)}
)

CONDITION_LABELS = {
    "Process: Foraging": "Process: Single Task (Foraging)",
    "Process: Cleaning": "Process: Single Task (Cleaning)",
    "Outcome: Foraging": "Outcome (Foraging)",
    "Outcome: Cleaning": "Outcome (Cleaning)",
    "Outcome: Both": "Outcome (Both)",
    "Process: Logistic Fuser": "Process: PLVM, linear fusion",
    "Process: Foraging (Logistic)": "Process: Single Task (Foraging), linear head",
    "Process: Cleaning (Logistic)": "Process: Single Task (Cleaning), linear head",
    "Process: PLVM-1": "Process: PLVM-1",
    "Process: PLVM-2": "Process: PLVM",
    "Process: PLVM-3": "Process: PLVM-3",
    "Process: PLVM-4": "Process: PLVM-4",
    "Process: PLVM-5": "Process: PLVM-5",
}
# Same blue as PLVM_SINGLE_DIM_COLOR in the PowerWash curve script, so Figures 6 and 7 match.
# PLVM runs with different latent dims share this color and differ in opacity (_plvm_alphas).
PLVM_BASE_COLOR = "#1f7a8c"
MODEL_MARKERS = {
    "single_task": "s",
    "plvm_both_partial": "o",
}
CURVE_TITLE = "Type recovery accuracy"
BAR_TITLE = "Type recovery accuracy"
Y_AXIS_LABEL = "Accuracy (%)"
TWO_CLASS_YMIN_PERCENT = 45.0
MANUSCRIPT_CURVE_FILENAMES = {
    8000: "Figure_7.pdf",
}
MANUSCRIPT_CURVE_FIGSIZE = (6.4, 4.4)
SOURCE_FAMILY_LABELS = {
    "process": "plvm_label",
    "outcome": "outcome_observation_baseline",
    "logistic": "plvm_label_concat_logistic_probe",
    "single_task_logistic": "plvm_label_single_task_logistic_probe",
}


def _apply_pub_style() -> None:
    """Set the plotting defaults shared by the curve and bar figures."""
    maybe_set_plot_style()
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "normal",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#cccccc",
        "legend.borderpad": 0.4,
        "legend.labelspacing": 0.3,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.linewidth": 1.6,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    }
    plt.rcParams.update(style)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plvm_model_performance_csv",
        type=Path,
        default=DEFAULT_PLVM_PERF_CSV,
        help="Process-model performance CSV from the minigrid plvm_label family.",
    )
    parser.add_argument(
        "--outcome_model_performance_csv",
        type=Path,
        default=DEFAULT_OUTCOME_PERF_CSV,
        help="Outcome-baseline performance CSV from the minigrid outcome family.",
    )
    parser.add_argument(
        "--plvm_training_logs_dir",
        type=Path,
        default=DEFAULT_PLVM_LOGS_DIR,
        help="Process-model training_logs directory containing per-run cutoff metrics.",
    )
    parser.add_argument(
        "--outcome_training_logs_dir",
        type=Path,
        default=DEFAULT_OUTCOME_LOGS_DIR,
        help="Outcome-baseline training_logs directory containing per-run cutoff metrics.",
    )
    parser.add_argument(
        "--logistic_model_performance_csv",
        type=Path,
        default=DEFAULT_LOGISTIC_PERF_CSV,
        help="Logistic-head performance CSV from the minigrid plvm_label_concat_logistic_probe family.",
    )
    parser.add_argument(
        "--logistic_training_logs_dir",
        type=Path,
        default=DEFAULT_LOGISTIC_LOGS_DIR,
        help="Logistic-head training_logs directory containing per-run cutoff metrics.",
    )
    parser.add_argument(
        "--single_task_logistic_model_performance_csv",
        type=Path,
        default=DEFAULT_SINGLE_TASK_LOGISTIC_PERF_CSV,
        help="Logistic-head performance CSV from the minigrid plvm_label_single_task_logistic_probe family.",
    )
    parser.add_argument(
        "--single_task_logistic_training_logs_dir",
        type=Path,
        default=DEFAULT_SINGLE_TASK_LOGISTIC_LOGS_DIR,
        help="Single-task logistic-head training_logs directory containing per-run cutoff metrics.",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=200,
        help="Process-model sequence length used to filter plvm_label runs.",
    )
    parser.add_argument(
        "--max_train_val_players",
        type=int,
        default=8000,
        help="Balanced player cap used to filter all families. Use <=0 to disable the filter.",
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        default="run",
        choices=["run", "debug"],
        help="Filter runs by run_mode.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for generated artifacts.",
    )
    parser.add_argument(
        "--output_pdf",
        type=Path,
        default=None,
        help="Optional output PDF path for the cutoff-curve figure.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional output CSV path for the long-form cutoff metrics.",
    )
    parser.add_argument(
        "--manuscript_figure",
        action="store_true",
        help=(
            "Also render Figure 7 at its fixed size into --manuscript_figure_dir "
            "(Figure_7.pdf; only for --max_train_val_players 8000). The analysis outputs are "
            "written either way."
        ),
    )
    parser.add_argument(
        "--manuscript_figure_dir",
        type=Path,
        default=DEFAULT_MANUSCRIPT_FIGURES_DIR,
        help=(
            "Directory for the Figure 7 PDF when --manuscript_figure is set "
            "(default: manuscript_figures/figures)."
        ),
    )
    return parser.parse_args(argv)


def _default_output_dir(seq_length: int, max_train_val_players: int) -> Path:
    player_token = (
        "all_players"
        if int(max_train_val_players) <= 0
        else "players{}".format(int(max_train_val_players))
    )
    return DEFAULT_OUTPUT_ROOT / "label_prediction_seq{}_{}".format(int(seq_length), player_token)


def _resolve_output_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    """Resolve output paths for curves, bars, and CSV exports."""
    output_dir = args.output_dir or _default_output_dir(
        args.seq_length,
        args.max_train_val_players,
    )
    stem = "minigrid_label_prediction_seq{}".format(int(args.seq_length))
    output_pdf = args.output_pdf or (output_dir / "{}_val_curves.pdf".format(stem))
    output_csv = args.output_csv or (output_dir / "{}_cutoff_metrics.csv".format(stem))
    output_bar_pdf = output_pdf.with_name(
        output_pdf.stem.replace("_val_curves", "_val_bar_summary") + ".pdf"
    )
    return output_pdf, output_bar_pdf, output_csv


def _parse_json_dict(raw_payload: object) -> Dict[str, object]:
    """Best-effort parser for the `all_arguments_json` payload column."""
    text = str(raw_payload or "").strip()
    if not text or text.lower() == "nan":
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_float(value: object, default: float = float("nan")) -> float:
    """Convert one scalar to float with a fallback."""
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: object, default: int = -1) -> int:
    """Convert one scalar to int with a fallback."""
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return int(float(value))
    except Exception:
        return default


def _normalize_task_scope(raw_value: object) -> str:
    """Normalize predicted task names to a stable comma-separated token."""
    tokens = [token.strip().lower() for token in str(raw_value or "").split(",") if token.strip()]
    if not tokens:
        return ""
    unique_tokens = sorted(dict.fromkeys(tokens))
    return ",".join(unique_tokens)


def _plvm_alphas(n: int) -> List[float]:
    """Return increasing opacities for multiple PLVM latent dimensions."""
    if n <= 1:
        return [1.0]
    return [0.35 + 0.65 * index / float(n - 1) for index in range(n)]


def load_candidate_rows(
    csv_path: Path,
    *,
    source_family: str,
    logs_dir: Path,
    run_mode: str,
    seq_length: int,
    max_train_val_players: int,
) -> List[dict]:
    """Load family-specific model-performance rows and attach parsed metadata."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []

    df = pd.read_csv(csv_path)
    rows: List[dict] = []
    for _, row in df.iterrows():
        payload = _parse_json_dict(row.get("all_arguments_json"))
        model_type = str(row.get("model_type") or payload.get("model_type") or "").strip().lower()
        if model_type not in {"single_task", "plvm_both_partial"}:
            continue
        if (
            str(
                row.get("parameters_to_predict") or payload.get("parameters_to_predict") or ""
            ).strip()
            != "minigrid_player_label"
        ):
            continue
        effective_run_mode = (
            str(row.get("run_mode") or payload.get("run_mode") or "").strip().lower()
        )
        if effective_run_mode != str(run_mode).strip().lower():
            continue

        effective_max_players = _safe_int(
            payload.get("max_train_val_players", row.get("max_train_val_players")),
            default=-1,
        )
        if int(max_train_val_players) > 0 and effective_max_players != int(max_train_val_players):
            continue

        effective_seq_length = _safe_int(
            row.get("seq_length", payload.get("seq_length", payload.get("embedding_seq_length"))),
            default=-1,
        )
        if source_family in (
            "process",
            "logistic",
            "single_task_logistic",
        ) and effective_seq_length != int(seq_length):
            continue

        predicted_next_task = _normalize_task_scope(
            row.get("predicted_next_task") or payload.get("predicted_next_task")
        )
        if not predicted_next_task:
            continue

        latent_dim = str(row.get("latent_dim") or payload.get("latent_dim") or "").strip()
        rows.append(
            {
                "model_name": str(row.get("model_name") or "").strip(),
                "model_type": model_type,
                "predicted_next_task": predicted_next_task,
                "prediction_head_type": str(
                    row.get("prediction_head_type") or payload.get("prediction_head_type") or ""
                ).strip(),
                "val_accuracy": _safe_float(row.get("val_accuracy")),
                "train_accuracy": _safe_float(row.get("train_accuracy")),
                "naive_val_accuracy": _safe_float(row.get("naive_val_accuracy")),
                "naive_train_accuracy": _safe_float(row.get("naive_train_accuracy")),
                "latent_dim": latent_dim,
                "seq_length": effective_seq_length,
                "max_train_val_players": effective_max_players,
                "run_mode": effective_run_mode,
                "source_family": source_family,
                "source_family_label": SOURCE_FAMILY_LABELS[source_family],
                "logs_dir": Path(logs_dir),
                "all_arguments_json": payload,
            }
        )
    return rows


def select_best_rows(rows: Sequence[dict]) -> Tuple[List[dict], Dict[str, dict]]:
    """Select the single best run for each plotting condition."""
    process_rows = [row for row in rows if row["source_family"] == "process"]
    outcome_rows = [row for row in rows if row["source_family"] == "outcome"]
    logistic_rows = [row for row in rows if row["source_family"] == "logistic"]
    single_task_logistic_rows = [
        row for row in rows if row["source_family"] == "single_task_logistic"
    ]
    process_plvm_rows = [row for row in process_rows if row["model_type"] == "plvm_both_partial"]
    latent_dims = sorted(
        {
            row["latent_dim"]
            for row in process_plvm_rows
            if row["latent_dim"] not in {"", "nan", "None"}
        },
        key=lambda value: int(value),
    )
    alphas = _plvm_alphas(len(latent_dims))

    condition_specs: List[dict] = []
    for label, task_name, color in PROCESS_SINGLE_CONDITIONS:
        condition_specs.append(
            {
                "condition": label,
                "color": color,
                "line_alpha": 1.0,
                "source_family": "process",
                "model_type": "single_task",
                "predicted_next_task": task_name,
                "latent_dim_filter": None,
            }
        )
    for label, task_name, color in OUTCOME_SINGLE_CONDITIONS:
        condition_specs.append(
            {
                "condition": label,
                "color": color,
                "line_alpha": 1.0,
                "source_family": "outcome",
                "model_type": "single_task",
                "predicted_next_task": task_name,
                "latent_dim_filter": None,
            }
        )
    condition_specs.append(
        {
            "condition": OUTCOME_BOTH_CONDITION[0],
            "color": OUTCOME_BOTH_CONDITION[2],
            "line_alpha": 1.0,
            "source_family": "outcome",
            "model_type": "plvm_both_partial",
            "predicted_next_task": OUTCOME_BOTH_CONDITION[1],
            "latent_dim_filter": None,
        }
    )
    condition_specs.append(
        {
            "condition": LOGISTIC_FUSER_CONDITION[0],
            "color": LOGISTIC_FUSER_CONDITION[3],
            "line_alpha": 1.0,
            "source_family": "logistic",
            "model_type": "plvm_both_partial",
            "predicted_next_task": LOGISTIC_FUSER_CONDITION[1],
            "latent_dim_filter": LOGISTIC_FUSER_CONDITION[2],
        }
    )
    for label, task_name, color in SINGLE_TASK_LOGISTIC_CONDITIONS:
        condition_specs.append(
            {
                "condition": label,
                "color": color,
                "line_alpha": 1.0,
                "source_family": "single_task_logistic",
                "model_type": "single_task",
                "predicted_next_task": task_name,
                "latent_dim_filter": None,
            }
        )
    for index, latent_dim in enumerate(latent_dims):
        condition_specs.append(
            {
                "condition": "Process: PLVM-{}".format(latent_dim),
                "color": PLVM_BASE_COLOR,
                "line_alpha": alphas[index],
                "source_family": "process",
                "model_type": "plvm_both_partial",
                "predicted_next_task": "cleaning,foraging",
                "latent_dim_filter": latent_dim,
            }
        )

    selected_meta: List[dict] = []
    selected_rows: Dict[str, dict] = {}
    family_rows_by_source = {
        "process": process_rows,
        "outcome": outcome_rows,
        "logistic": logistic_rows,
        "single_task_logistic": single_task_logistic_rows,
    }
    for spec in condition_specs:
        family_rows = family_rows_by_source[spec["source_family"]]
        candidates = [
            row
            for row in family_rows
            if row["model_type"] == spec["model_type"]
            and row["predicted_next_task"] == spec["predicted_next_task"]
            and (
                spec["latent_dim_filter"] is None or row["latent_dim"] == spec["latent_dim_filter"]
            )
        ]
        if not candidates:
            print(
                "Warning: no completed minigrid runs found for condition '{}'.".format(
                    spec["condition"]
                ),
                file=sys.stderr,
            )
            continue

        best_row = max(
            candidates,
            key=lambda row: (
                _safe_float(row["val_accuracy"], default=float("-inf")),
                _safe_float(row["train_accuracy"], default=float("-inf")),
            ),
        )
        selected = dict(best_row)
        selected["condition"] = spec["condition"]
        selected["condition_label"] = CONDITION_LABELS.get(spec["condition"], spec["condition"])
        selected["color"] = spec["color"]
        selected["line_alpha"] = spec["line_alpha"]
        selected_meta.append(selected)
        selected_rows[spec["condition"]] = selected
    return selected_meta, selected_rows


def load_cutoff_frame(run_row: dict, split_name: str) -> pd.DataFrame:
    """Load one selected run's cutoff metrics for the requested split."""
    path = (
        Path(run_row["logs_dir"])
        / str(run_row["model_name"])
        / "plots"
        / split_name
        / "cutoff_metrics"
        / "metrics_by_cutoff.csv"
    )
    if not path.exists():
        raise FileNotFoundError("Missing cutoff metrics: {}".format(path))

    frame = pd.read_csv(path)
    required_columns = ["cutoff_value", "accuracy", "macro_f1", "cross_entropy", "support"]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError("Missing columns {} in {}".format(missing_columns, path))
    for column in required_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.copy()
    frame["split"] = split_name
    frame["model_name"] = str(run_row["model_name"])
    frame["condition"] = str(run_row["condition"])
    frame["condition_label"] = str(run_row["condition_label"])
    frame["model_type"] = str(run_row["model_type"])
    frame["source_family"] = str(run_row["source_family"])
    frame["predicted_next_task"] = str(run_row["predicted_next_task"])
    frame["latent_dim"] = str(run_row["latent_dim"])
    frame["val_accuracy_best_run"] = float(run_row["val_accuracy"])
    frame["naive_val_accuracy_best_run"] = float(run_row["naive_val_accuracy"])
    frame["seq_length"] = int(run_row["seq_length"])
    return frame


def build_long_frame(selected_rows: Dict[str, dict]) -> pd.DataFrame:
    """Build a long-form cutoff-metrics frame across all selected conditions."""
    pieces = []
    for run_row in selected_rows.values():
        for split_name in ("train", "val"):
            try:
                pieces.append(load_cutoff_frame(run_row, split_name))
            except FileNotFoundError as exc:
                print("Warning: {}".format(exc), file=sys.stderr)
    if not pieces:
        raise ValueError("No cutoff-metric frames could be loaded for the selected runs.")
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(
            ["split", "condition", "cutoff_value"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _to_percent(values):
    """Convert one scalar or array-like accuracy value from [0, 1] to [0, 100]."""
    return np.asarray(values, dtype="float64") * 100.0


def _resolve_percent_ylim(all_acc_percent, reference_baseline_percent: float):
    """Resolve a percent-scaled y-range for the simulation figures."""
    # The lower limit is fixed; the upper limit leaves 5 points above the highest value.
    y_min = TWO_CLASS_YMIN_PERCENT
    if len(all_acc_percent) == 0:
        y_max = min(100.0, reference_baseline_percent + 5.0)
    else:
        y_max = min(100.0, max(max(all_acc_percent), reference_baseline_percent) + 5.0)
    if y_max <= y_min:
        y_max = min(100.0, y_min + 10.0)
    return y_min, y_max


def _infer_num_classes(frame: pd.DataFrame, default: int = 2) -> int:
    """Infer class count from cutoff-metric metadata when available."""
    if "class_names_json" in frame.columns:
        for raw_value in frame["class_names_json"].dropna().tolist():
            try:
                parsed = json.loads(str(raw_value))
            except Exception:
                continue
            if isinstance(parsed, list) and parsed:
                return int(len(parsed))
    return int(default)


def _resolve_named_manuscript_curve_path(max_train_val_players: int) -> Optional[Path]:
    """Return the default Figure 7 path for a player-count setting, if it has one."""
    filename = MANUSCRIPT_CURVE_FILENAMES.get(int(max_train_val_players))
    if filename is None:
        return None
    return DEFAULT_MANUSCRIPT_FIGURES_DIR / filename


def plot_curves(
    long_frame: pd.DataFrame,
    condition_meta: Sequence[dict],
    selected_rows: Dict[str, dict],
    output_pdf: Path,
    *,
    split_name: str = "val",
    figsize: Tuple[float, float] = (6.5, 3.8),
    num_classes: int = 2,
    legend_loc: str = "best",
    legend_ncol: int = 2,
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
) -> None:
    """Render the cutoff-wise accuracy curves for the selected runs."""
    _apply_pub_style()
    fig, ax = plt.subplots(figsize=figsize)

    split_frame = long_frame.loc[long_frame["split"] == split_name].copy()
    if split_frame.empty:
        raise ValueError("No {} cutoff metrics are available for plotting.".format(split_name))

    reference_baseline = 1.0 / float(max(int(num_classes), 1))
    reference_baseline_percent = float(_to_percent(reference_baseline))

    for row in condition_meta:
        condition_df = (
            split_frame.loc[split_frame["condition"] == row["condition"]]
            .sort_values("cutoff_value", kind="stable")
            .copy()
        )
        if condition_df.empty:
            continue
        x = condition_df["cutoff_value"].astype(float)
        y = _to_percent(condition_df["accuracy"].astype(float))
        ax.plot(
            x,
            y,
            marker=MODEL_MARKERS.get(row.get("model_type"), "o"),
            linewidth=2.0,
            markersize=5.0,
            color=row["color"],
            alpha=row.get("line_alpha", 1.0),
            label=row["condition_label"],
        )

    ax.axhline(
        reference_baseline_percent,
        color="#555555",
        linestyle="--",
        linewidth=1.3,
        label="_nolegend_",
    )

    all_acc_percent = _to_percent(split_frame["accuracy"].dropna().tolist())
    y_min, y_max = _resolve_percent_ylim(all_acc_percent, reference_baseline_percent)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Observed player steps")
    ax.set_ylabel(Y_AXIS_LABEL)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, axis="y")

    handles, labels = ax.get_legend_handles_labels()

    def _legend_sort_key(label: str) -> Tuple[int, str]:
        label = LEGEND_SORT_ALIAS.get(label, label)
        if label.startswith("Outcome"):
            return (0, label)
        if label.startswith("Single task"):
            return (1, label)
        if label.startswith("Both tasks"):
            return (2, label)
        return (3, label)

    if handles:
        pairs = sorted(zip(handles, labels), key=lambda pair: _legend_sort_key(pair[1]))
        sorted_handles, sorted_labels = zip(*pairs)
        legend_kwargs = {
            "loc": str(legend_loc),
            "frameon": True,
            "ncol": int(legend_ncol),
        }
        if legend_bbox_to_anchor is not None:
            legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
        ax.legend(sorted_handles, sorted_labels, **legend_kwargs)

    title = ax.set_title(CURVE_TITLE)
    title.set_wrap(True)
    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_bar_summary(
    condition_meta: Sequence[dict],
    selected_rows: Dict[str, dict],
    output_pdf: Path,
    *,
    split_name: str = "val",
    figsize: Tuple[float, float] = (6.0, 3.4),
    num_classes: int = 2,
) -> None:
    """Render a bar-chart summary of selected-run accuracies."""
    _apply_pub_style()
    fig, ax = plt.subplots(figsize=figsize)

    metric_key = "{}_accuracy".format(str(split_name).strip().lower())
    labels = [row["condition_label"] for row in condition_meta]
    colors = [row["color"] for row in condition_meta]
    alphas = [row.get("line_alpha", 1.0) for row in condition_meta]
    means = [
        float(_to_percent(_safe_float(selected_rows[row["condition"]].get(metric_key))))
        for row in condition_meta
    ]

    x = np.arange(len(labels))
    bars = ax.bar(x, means, color=colors)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)

    chance_baseline = 1.0 / float(max(int(num_classes), 1))
    ax.axhline(
        float(_to_percent(chance_baseline)),
        color="#555555",
        linestyle="--",
        linewidth=1.3,
        label="_nolegend_",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel(Y_AXIS_LABEL)
    baseline_percent = float(_to_percent(chance_baseline))
    ax.set_ylim(_resolve_percent_ylim(means, baseline_percent))
    ax.grid(True, axis="y")
    ax.set_title(BAR_TITLE)

    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def write_selected_runs_csv(selected_rows: Dict[str, dict], output_csv: Path) -> None:
    """Write a compact selected-runs summary CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition_label",
        "source_family",
        "model_name",
        "model_type",
        "prediction_head_type",
        "predicted_next_task",
        "latent_dim",
        "seq_length",
        "max_train_val_players",
        "val_accuracy",
        "naive_val_accuracy",
        "train_accuracy",
        "run_mode",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for condition in sorted(selected_rows):
            row = dict(selected_rows[condition])
            row["condition_label"] = row["condition"]
            writer.writerow(row)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the minigrid cutoff-curve summarization workflow."""
    args = parse_args(argv)
    output_pdf, output_bar_pdf, output_csv = _resolve_output_paths(args)
    selected_runs_csv = output_pdf.with_name(
        output_pdf.stem.replace("_val_curves", "_selected_runs") + ".csv"
    )

    rows = []
    rows.extend(
        load_candidate_rows(
            args.plvm_model_performance_csv,
            source_family="process",
            logs_dir=args.plvm_training_logs_dir,
            run_mode=args.run_mode,
            seq_length=args.seq_length,
            max_train_val_players=args.max_train_val_players,
        )
    )
    rows.extend(
        load_candidate_rows(
            args.outcome_model_performance_csv,
            source_family="outcome",
            logs_dir=args.outcome_training_logs_dir,
            run_mode=args.run_mode,
            seq_length=args.seq_length,
            max_train_val_players=args.max_train_val_players,
        )
    )
    rows.extend(
        load_candidate_rows(
            args.logistic_model_performance_csv,
            source_family="logistic",
            logs_dir=args.logistic_training_logs_dir,
            run_mode=args.run_mode,
            seq_length=args.seq_length,
            max_train_val_players=args.max_train_val_players,
        )
    )
    rows.extend(
        load_candidate_rows(
            args.single_task_logistic_model_performance_csv,
            source_family="single_task_logistic",
            logs_dir=args.single_task_logistic_training_logs_dir,
            run_mode=args.run_mode,
            seq_length=args.seq_length,
            max_train_val_players=args.max_train_val_players,
        )
    )
    if not rows:
        raise ValueError("No minigrid model-performance rows matched the requested filters.")

    condition_meta, selected_rows = select_best_rows(rows)
    if not selected_rows:
        raise ValueError("No completed minigrid conditions were available for plotting.")

    long_frame = build_long_frame(selected_rows)
    num_classes = _infer_num_classes(long_frame)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    long_frame.to_csv(output_csv, index=False)
    write_selected_runs_csv(selected_rows, selected_runs_csv)
    plot_curves(
        long_frame,
        condition_meta,
        selected_rows,
        output_pdf,
        split_name="val",
        num_classes=num_classes,
    )
    plot_bar_summary(
        condition_meta,
        selected_rows,
        output_bar_pdf,
        split_name="val",
        num_classes=num_classes,
    )

    if args.manuscript_figure:
        named_manuscript_curve = _resolve_named_manuscript_curve_path(args.max_train_val_players)
        if named_manuscript_curve is not None:
            manuscript_output = Path(args.manuscript_figure_dir) / named_manuscript_curve.name
            plot_curves(
                long_frame,
                condition_meta,
                selected_rows,
                manuscript_output,
                split_name="val",
                figsize=MANUSCRIPT_CURVE_FIGSIZE,
                num_classes=num_classes,
                legend_loc="upper center",
                legend_ncol=2,
                legend_bbox_to_anchor=(0.5, -0.18),
            )
            print("Wrote {}".format(manuscript_output))

    print("Wrote {}".format(output_pdf))
    print("Wrote {}".format(output_bar_pdf))
    print("Wrote {}".format(output_csv))
    print("Wrote {}".format(selected_runs_csv))
    for row in condition_meta:
        selected = selected_rows[row["condition"]]
        print(
            "{}: val={:.4f} train={:.4f} model={}".format(
                row["condition_label"],
                _safe_float(selected.get("val_accuracy")),
                _safe_float(selected.get("train_accuracy")),
                selected.get("model_name"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
