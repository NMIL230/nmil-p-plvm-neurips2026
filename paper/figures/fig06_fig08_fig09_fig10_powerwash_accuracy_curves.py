#!/usr/bin/env python3
"""
Generates Figures 6, 8, 9 and 10: PowerWash accuracy over observation time per condition.

Also owns the PowerWash run selection and fold aggregation reused by
``fig05_combined_overview.py`` and the PowerWash table scripts. Conditions are
single-task, RF outcome baselines, each PLVM latent dim and the logistic probe.
For each (condition, fold) the run with the highest validation accuracy is kept;
each condition is then averaged over the folds where it has runs (mean +/-1 std).

Analysis outputs go under ``<artifact root>/analysis/model_training_analysis/``;
``--manuscript_figure`` also copies the curve PDFs to ``manuscript_figures/figures/``
(Figures 6 and 8 show PLVM-2 only; 9 and 10 show every latent dim).

    python paper/figures/fig06_fig08_fig09_fig10_powerwash_accuracy_curves.py \
        --manuscript_figure
"""

import argparse
import csv
import json
import sys
from pathlib import Path

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
    copy_figure_to_path,
    resolve_figures_dir,
    resolve_tables_dir,
)
from plvm import artifact_paths
from powerwash.train.powerwash_hierarchical_log_utils import (
    load_powerwash_hierarchical_performance_rows,
    resolve_powerwash_hierarchical_run_dir,
)


def _apply_pub_style(compact: bool = False):
    """Set the plotting rcParams; ``compact`` selects a compact variant for small figures."""
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
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    }
    if compact:
        style.update(
            {
                "font.size": 5.8,
                "axes.titlesize": 6.0,
                "axes.labelsize": 5.7,
                "xtick.labelsize": 5.1,
                "ytick.labelsize": 5.1,
                "legend.fontsize": 4.9,
                "legend.borderpad": 0.16,
                "legend.labelspacing": 0.12,
                "legend.handlelength": 0.9,
                "legend.handletextpad": 0.25,
                "axes.linewidth": 0.55,
                "xtick.major.width": 0.55,
                "ytick.major.width": 0.55,
                "grid.linewidth": 0.4,
                "lines.linewidth": 1.1,
            }
        )
    plt.rcParams.update(style)


# Sort keys for legend ordering.
LEGEND_SORT_ALIAS = {
    "Process: Single Task (Bungalow)": "Single task (Bungalow), nonlinear head",
    "Process: Single Task (Back Garden)": "Single task (Back Garden), nonlinear head",
    "Process: Single Task (Back Garden), linear head": "Single task (Back Garden), linear head",
    "Process: Single Task (Bungalow), linear head": "Single task (Bungalow), linear head",
    "Process: PLVM, linear fusion": "Both tasks, linear fusion",
    "Process: PLVM": "Both tasks, nonlinear fusion (PLVM-2)",
}
LEGEND_SORT_ALIAS.update(
    {"Process: PLVM-%d" % k: "Both tasks, nonlinear fusion (PLVM-%d)" % k for k in range(1, 6)}
)

CONDITION_LABELS = {
    "Bungalow Alone": "Process: Single Task (Bungalow)",
    "Back Garden Alone": "Process: Single Task (Back Garden)",
    "RF Back Garden": "Outcome (Back Garden)",
    "RF Bungalow": "Outcome (Bungalow)",
    "RF Both": "Outcome (Both)",
    "PLVM-1": "Process: PLVM-1",
    "PLVM-2": "Process: PLVM-2",
    "PLVM-3": "Process: PLVM-3",
    "PLVM-4": "Process: PLVM-4",
    "PLVM-5": "Process: PLVM-5",
    "Logistic probe-1": "Process: PLVM, linear fusion",
    "Logistic probe-2": "Process: PLVM, linear fusion",
    "Back Garden Logistic": "Process: Single Task (Back Garden), linear head",
    "Bungalow Logistic": "Process: Single Task (Bungalow), linear head",
}
SINGLE_TASK_LOGISTIC_CONDITIONS = [
    ("Back Garden Logistic", "back_garden", "#e3a857"),
    ("Bungalow Logistic", "bungalow", "#8f4a2e"),
]
SUMMARY_TIMEPOINT_MINUTES = (20.0, 60.0, 120.0)
SUMMARY_AUC_MAX_MINUTES = 120.0
SUMMARY_FAMILY_ORDER = (
    "Chance baseline",
    "Outcome baseline",
    "Single-task process",
    "PLVM",
)
SUMMARY_FAMILY_INPUTS = {
    "Chance baseline": "Class prior",
    "Outcome baseline": "BG + B outcome summaries",
    "Single-task process": "Best of BG or B trace",
    "PLVM": "BG + B process traces",
}

TRAINING_LOGS_DIR = artifact_paths.POWERWASH_TRAINING_LOGS_DIR
DEFAULT_PERF_CSV = TRAINING_LOGS_DIR
DEFAULT_OUTPUT_ROOT = artifact_paths.POWERWASH_PERFORMANCE_ANALYSIS_DIR
DEFAULT_FIGURES_DIR = resolve_figures_dir()
DEFAULT_TABLES_DIR = resolve_tables_dir()
DEFAULT_TARGET_MODES = [
    "player_label_2_class_clean_zonehopping",
    "player_label_3_class_zonehopping",
]

# Single-task baselines plus matched RF baselines
SINGLE_CONDITIONS = [
    ("Back Garden Alone", "single_task", "back_garden", "#d67c4a"),
    ("Bungalow Alone", "single_task", "bungalow", "#b84f2a"),
    ("RF Back Garden", "rf_baseline", "back_garden", "#7fb069"),
    ("RF Bungalow", "rf_baseline", "bungalow", "#4c956c"),
]
RF_BOTH_CONDITION = ("RF Both", "rf_baseline", "both", "#2d6a4f")
MODEL_MARKERS = {
    "single_task": "s",
    "rf_baseline": "D",
    "plvm_both_partial": "o",
}
CURVE_TITLE = "Future navigation style prediction accuracy (mean ± sd over 10 folds)"
COMPACT_CURVE_TITLE = "Future Navigation Style Prediction"
BAR_TITLE = "Future navigation style prediction accuracy summary (mean ± sd over 10 folds)"
Y_AXIS_LABEL = "Accuracy (%)"
TWO_CLASS_YMIN_PERCENT = 45.0
MANUSCRIPT_CURVE_FILENAMES = {
    "player_label_2_class_clean_zonehopping": "Figure_6.pdf",
    "player_label_3_class_zonehopping": "Figure_8.pdf",
}
MANUSCRIPT_MULTI_PLVM_CURVE_FILENAMES = {
    "player_label_2_class_clean_zonehopping": "Figure_9.pdf",
    "player_label_3_class_zonehopping": "Figure_10.pdf",
}
MANUSCRIPT_CURVE_FIGSIZE = (6.4, 4.4)


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_performance_csv",
        type=Path,
        default=DEFAULT_PERF_CSV,
        help=(
            "Path to one PowerWash hierarchical model_performance.csv file, or "
            "the training_logs root containing family-specific CSVs."
        ),
    )
    parser.add_argument(
        "--training_logs_dir",
        type=Path,
        default=TRAINING_LOGS_DIR,
        help="Directory containing the per-run training logs.",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=120,
        help="Sequence length used to filter the candidate runs.",
    )
    parser.add_argument(
        "--target_mode",
        type=str,
        action="append",
        default=None,
        help=(
            "Target mode used to filter runs. Repeat the flag to request more "
            "than one target mode. Defaults to player_label_2_class_clean_zonehopping "
            "and player_label_3_class_zonehopping."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Directory for generated artifacts. Defaults to "
            "<artifact root>/analysis/model_training_analysis/powerwash_hierarchical_performance/"
            "<target_mode>/zone_planning_seq<seq_length>/."
        ),
    )
    parser.add_argument(
        "--output_pdf",
        type=Path,
        default=None,
        help="Output PDF path for the cutoff-curve figure. Overrides --output_dir if set.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Output CSV path for the long-form cutoff metrics. Overrides --output_dir if set.",
    )
    parser.add_argument(
        "--manuscript_figure",
        action="store_true",
        help=(
            "Also copy the PDFs of Figures 6, 8, 9 and 10 (Figure_6/8/9/10.pdf, for the two default "
            "targets) into --manuscript_figure_dir. The analysis outputs are written either way."
        ),
    )
    parser.add_argument(
        "--manuscript_figure_dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help=(
            "Directory for the Figure 6, 8, 9 and 10 PDFs when --manuscript_figure is set "
            "(default: manuscript_figures/figures)."
        ),
    )
    return parser.parse_args(argv)


def _default_output_dir(target_mode: str, seq_length: int) -> Path:
    return DEFAULT_OUTPUT_ROOT / target_mode / "zone_planning_seq{}".format(int(seq_length))


def _default_output_stem(target_mode: str, seq_length: int) -> str:
    return "{}__best_curves".format(target_mode)


def _resolve_output_paths(args, target_mode: str, target_modes_count: int):
    if target_modes_count > 1 and (args.output_pdf or args.output_csv):
        raise ValueError(
            "Explicit --output_pdf/--output_csv only support a single target_mode. "
            "Use --output_dir or run one target at a time."
        )
    if args.output_dir is not None:
        output_dir = args.output_dir / target_mode if target_modes_count > 1 else args.output_dir
    else:
        output_dir = _default_output_dir(target_mode, args.seq_length)
    output_stem = _default_output_stem(target_mode, args.seq_length)
    output_pdf = args.output_pdf or (output_dir / "{}.pdf".format(output_stem))
    output_csv = args.output_csv or (output_dir / "{}.csv".format(output_stem))
    return output_pdf, output_csv


def resolve_target_modes(args):
    """Return the requested ``--target_mode`` values (or the defaults), de-duplicated in order."""
    target_modes = args.target_mode or list(DEFAULT_TARGET_MODES)
    ordered_unique = []
    seen = set()
    for target_mode in target_modes:
        normalized = str(target_mode).strip()
        if not normalized or normalized in seen:
            continue
        ordered_unique.append(normalized)
        seen.add(normalized)
    return ordered_unique


def load_candidate_rows(csv_path: Path, seq_length: int, target_mode: str) -> list:
    """Load target/seq_length-filtered model-performance rows with fold ID attached."""
    rows = []
    for row in load_powerwash_hierarchical_performance_rows(
        csv_path,
        model_types=("plvm_both_partial", "single_task", "rf_baseline"),
    ):
        payload = json.loads(row["all_arguments_json"]) if row.get("all_arguments_json") else {}
        effective_target_mode = str(
            row.get("target_mode") or payload.get("target_mode") or ""
        ).strip()
        if effective_target_mode != str(target_mode):
            continue
        effective_seq_length = (
            int(row["seq_length"]) if row.get("seq_length") else int(payload.get("seq_length", -1))
        )
        if effective_seq_length != int(seq_length):
            continue
        fold_number = row.get("fold_number") or payload.get("fold_number")
        fold_id = (
            "fold_{:02d}".format(int(fold_number)) if fold_number is not None else "fold_unknown"
        )
        rows.append(
            {
                "model_name": row["model_name"],
                "model_type": row["model_type"],
                "predicted_next_task": row["predicted_next_task"],
                "prediction_head_type": row["prediction_head_type"],
                "val_accuracy": float(row["val_accuracy"]),
                "naive_val_accuracy": float(row["naive_val_accuracy"]),
                "target_mode": effective_target_mode,
                "num_classes": int(
                    payload.get(
                        "num_classes",
                        2
                        if "2_class" in effective_target_mode
                        else 3
                        if "3_class" in effective_target_mode
                        else 2,
                    )
                ),
                "num_epochs": int(row["num_epochs"])
                if row.get("num_epochs")
                else int(payload.get("num_epochs", -1)),
                "max_files_per_task": int(payload.get("max_files_per_task", -1)),
                "dim_reduction_method": row.get("dim_reduction_method", ""),
                "latent_dim": str(row.get("latent_dim") or payload.get("latent_dim") or ""),
                "seq_length": effective_seq_length,
                "fold_id": fold_id,
            }
        )
    return rows


# Distinct, ordered blues (ColorBrewer "Blues") so each PLVM latent dim has its own line,
# which keeps overlapping curves readable in Figures 9 and 10.
PLVM_DIM_COLORS = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]
# Color of the single PLVM curve when only PLVM-2 is shown (Figures 6 and 8).
PLVM_SINGLE_DIM_COLOR = "#1f7a8c"


def _plvm_dim_color(index: int) -> str:
    return PLVM_DIM_COLORS[index % len(PLVM_DIM_COLORS)]


def select_all_rows_by_fold(rows: list) -> tuple:
    """
    Select the best run per (condition, fold) independently for each condition.

    Returns:
        condition_meta: list of display-property dicts (one per condition, from first fold)
        rows_by_condition_fold: {condition_label: {fold_id: best_row}}
    """
    plvm_rows = [r for r in rows if r["model_type"] == "plvm_both_partial"]
    latent_dims = sorted(
        {r["latent_dim"] for r in plvm_rows if r["latent_dim"] not in ("", "nan", "None")},
        key=lambda x: int(x),
    )

    condition_specs = []
    for label, model_type, predicted_next_task, color in SINGLE_CONDITIONS:
        condition_specs.append(
            {
                "condition": label,
                "condition_label": label,
                "color": color,
                "line_alpha": 1.0,
                "model_type": model_type,
                "predicted_next_task": predicted_next_task,
                "latent_dim_filter": None,
            }
        )
    condition_specs.append(
        {
            "condition": RF_BOTH_CONDITION[0],
            "condition_label": RF_BOTH_CONDITION[0],
            "color": RF_BOTH_CONDITION[3],
            "line_alpha": 1.0,
            "model_type": RF_BOTH_CONDITION[1],
            "predicted_next_task": RF_BOTH_CONDITION[2],
            "latent_dim_filter": None,
        }
    )
    for i, ld in enumerate(latent_dims):
        condition_specs.append(
            {
                "condition": "PLVM-{}".format(ld),
                "condition_label": "PLVM-{}".format(ld),
                "color": _plvm_dim_color(i),
                "line_alpha": 1.0,
                "model_type": "plvm_both_partial",
                "predicted_next_task": None,
                "latent_dim_filter": ld,
            }
        )

    rows_by_condition_fold = {}
    condition_meta = []

    for spec in condition_specs:
        label = spec["condition"]
        candidates = [
            r
            for r in rows
            if r["model_type"] == spec["model_type"]
            and (
                spec["predicted_next_task"] is None
                or r["predicted_next_task"] == spec["predicted_next_task"]
            )
            and (spec["latent_dim_filter"] is None or r["latent_dim"] == spec["latent_dim_filter"])
        ]
        if not candidates:
            print("Warning: no runs found for condition '{}'.".format(label), file=sys.stderr)
            continue

        # Pick best run per fold
        by_fold = {}
        for r in candidates:
            fid = r["fold_id"]
            if fid not in by_fold or r["val_accuracy"] > by_fold[fid]["val_accuracy"]:
                row = dict(r)
                row["condition"] = label
                row["condition_label"] = spec["condition_label"]
                row["color"] = spec["color"]
                row["line_alpha"] = spec["line_alpha"]
                by_fold[fid] = row

        rows_by_condition_fold[label] = by_fold
        condition_meta.append(dict(next(iter(by_fold.values()))))

    return condition_meta, rows_by_condition_fold


def load_logistic_rows(csv_path: Path, seq_length: int, target_mode: str) -> list:
    """Load logistic-probe rows (e.g. from concat_logistic_probe) for one target/seq_length."""
    rows = []
    for row in load_powerwash_hierarchical_performance_rows(Path(csv_path)):
        payload = json.loads(row["all_arguments_json"]) if row.get("all_arguments_json") else {}
        effective_target_mode = str(
            row.get("target_mode") or payload.get("target_mode") or ""
        ).strip()
        if effective_target_mode != str(target_mode):
            continue
        effective_seq_length = (
            int(row["seq_length"]) if row.get("seq_length") else int(payload.get("seq_length", -1))
        )
        if effective_seq_length != int(seq_length):
            continue
        if str(row.get("prediction_head_type", "")).strip() != "logistic":
            continue
        fold_number = row.get("fold_number") or payload.get("fold_number")
        fold_id = (
            "fold_{:02d}".format(int(fold_number)) if fold_number is not None else "fold_unknown"
        )
        rows.append(
            {
                "model_name": row["model_name"],
                "model_type": row["model_type"],
                "predicted_next_task": row["predicted_next_task"],
                "prediction_head_type": row["prediction_head_type"],
                "val_accuracy": float(row["val_accuracy"]),
                "naive_val_accuracy": float(row["naive_val_accuracy"]),
                "target_mode": effective_target_mode,
                "num_epochs": int(row["num_epochs"])
                if row.get("num_epochs")
                else int(payload.get("num_epochs", -1)),
                "max_files_per_task": int(payload.get("max_files_per_task", -1)),
                "dim_reduction_method": row.get("dim_reduction_method", ""),
                "latent_dim": str(row.get("latent_dim") or payload.get("latent_dim") or ""),
                "seq_length": effective_seq_length,
                "fold_id": fold_id,
                "_source_csv": str(row.get("_source_csv", "")),
            }
        )
    return rows


def select_best_logistic_condition(training_logs_dir: Path, logistic_rows: list):
    """Select the logistic condition with the best fold-mean validation accuracy.

    Returns (condition_label, {fold_id: row}) for the best-scoring logistic
    condition, or None when no logistic rows are available.
    """
    if not logistic_rows:
        return None

    condition_rows: dict = {}
    for row in logistic_rows:
        latent_dim = str(row.get("latent_dim", "")).strip()
        condition = "Logistic probe" if not latent_dim else "Logistic probe-{}".format(latent_dim)
        fold_bucket = condition_rows.setdefault(condition, {})
        fold_id = str(row["fold_id"])
        current = fold_bucket.get(fold_id)
        if current is None or float(row["val_accuracy"]) > float(current["val_accuracy"]):
            selected = dict(row)
            selected["condition"] = condition
            selected["condition_label"] = condition
            selected["color"] = "#7c6fb3"
            selected["line_alpha"] = 1.0
            fold_bucket[fold_id] = selected

    fold_frame = build_fold_long_frame(Path(training_logs_dir), condition_rows)
    scores = _condition_average_accuracy_scores(fold_frame, "val")
    if not scores:
        return None
    best_condition = max(scores.items(), key=lambda item: float(item[1]))[0]
    return best_condition, condition_rows[best_condition]


def select_single_task_logistic_conditions(logistic_rows: list, task_specs=None) -> list:
    """Select best-per-fold runs for each single-task logistic condition.

    ``task_specs`` entries are (condition_label, predicted_next_task, color).
    Returns one (condition_label, {fold_id: row}) pair per task with data.
    """
    if task_specs is None:
        task_specs = SINGLE_TASK_LOGISTIC_CONDITIONS
    results = []
    for condition_label, task_name, color in task_specs:
        candidates = [row for row in logistic_rows if row.get("predicted_next_task") == task_name]
        if not candidates:
            continue
        by_fold = {}
        for row in candidates:
            fold_id = str(row["fold_id"])
            current = by_fold.get(fold_id)
            if current is None or float(row["val_accuracy"]) > float(current["val_accuracy"]):
                selected = dict(row)
                selected["condition"] = condition_label
                selected["condition_label"] = condition_label
                selected["color"] = color
                selected["line_alpha"] = 1.0
                by_fold[fold_id] = selected
        results.append((condition_label, by_fold))
    return results


def load_cutoff_frame(training_logs_dir: Path, run_name: str, split_name: str) -> pd.DataFrame:
    """Load ``plots/<split>/cutoff_metrics/metrics_by_cutoff.csv`` of one saved run."""
    path = (
        resolve_powerwash_hierarchical_run_dir(training_logs_dir, run_name)
        / "plots"
        / split_name
        / "cutoff_metrics"
        / "metrics_by_cutoff.csv"
    )
    if not path.exists():
        raise FileNotFoundError("Missing cutoff metrics: {}".format(path))
    frame = pd.read_csv(path)
    required = ["cutoff_value", "accuracy", "macro_f1", "cross_entropy", "support"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError("Missing columns {} in {}".format(missing, path))
    frame = frame.copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["split"] = split_name
    frame["model_name"] = run_name
    return frame


def build_fold_long_frame(training_logs_dir: Path, rows_by_condition_fold: dict) -> pd.DataFrame:
    """Build a long-form frame across all conditions, folds, and both splits."""
    pieces = []
    for condition, fold_rows in rows_by_condition_fold.items():
        for fold_id, row in sorted(fold_rows.items()):
            for split_name in ["train", "val"]:
                try:
                    frame = load_cutoff_frame(training_logs_dir, row["model_name"], split_name)
                except FileNotFoundError as exc:
                    print("Warning: {}".format(exc), file=sys.stderr)
                    continue
                frame["fold_id"] = fold_id
                frame["condition"] = row["condition"]
                frame["model_type"] = row["model_type"]
                frame["predicted_next_task"] = row["predicted_next_task"]
                frame["prediction_head_type"] = row["prediction_head_type"]
                frame["val_accuracy_best_run"] = row["val_accuracy"]
                frame["naive_val_accuracy_best_run"] = row["naive_val_accuracy"]
                frame["num_epochs"] = row["num_epochs"]
                frame["max_files_per_task"] = row["max_files_per_task"]
                frame["dim_reduction_method"] = row["dim_reduction_method"]
                frame["latent_dim"] = row["latent_dim"]
                frame["target_mode"] = row["target_mode"]
                frame["condition_label"] = row["condition_label"]
                pieces.append(frame)
    if not pieces:
        raise ValueError("No cutoff-metric frames could be loaded.")
    fold_frame = pd.concat(pieces, ignore_index=True)
    fold_frame = fold_frame.sort_values(
        ["fold_id", "split", "condition", "cutoff_value"]
    ).reset_index(drop=True)
    return fold_frame


def aggregate_folds(fold_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and std of accuracy across folds for each (condition, split, cutoff)."""
    agg = (
        fold_frame.groupby(["condition", "split", "cutoff_value"])["accuracy"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "accuracy_mean", "std": "accuracy_std", "count": "n_folds"})
        .reset_index()
    )
    agg["accuracy_std"] = agg["accuracy_std"].fillna(0.0)
    return agg


def load_train_accuracy_by_model_name(*log_sources: Path) -> dict:
    """Return {model_name: train_accuracy} from the model_performance.csv files under each source."""
    train_accuracy_by_model = {}
    for log_source in log_sources:
        if not Path(log_source).exists():
            continue
        for row in load_powerwash_hierarchical_performance_rows(Path(log_source)):
            model_name = str(row.get("model_name", "") or "").strip()
            raw_value = str(row.get("train_accuracy", "") or "").strip()
            if model_name and raw_value and model_name not in train_accuracy_by_model:
                train_accuracy_by_model[model_name] = float(raw_value)
    return train_accuracy_by_model


def collect_fold_train_accs(rows_by_condition_fold: dict, train_accuracy_by_model: dict) -> dict:
    """Return {condition: [train accuracy per fold]} for the selected runs."""
    return {
        condition: [
            train_accuracy_by_model[r["model_name"]]
            for r in fold_rows.values()
            if r["model_name"] in train_accuracy_by_model
        ]
        for condition, fold_rows in rows_by_condition_fold.items()
    }


def collect_fold_val_accs(rows_by_condition_fold: dict) -> tuple:
    """Return (val_accs, naive_accs): {condition: [value per fold]}."""
    val_accs = {}
    naive_accs = {}
    for condition, fold_rows in rows_by_condition_fold.items():
        val_accs[condition] = [r["val_accuracy"] for r in fold_rows.values()]
        naive_accs[condition] = [
            float(r["naive_val_accuracy"])
            for r in fold_rows.values()
            if r.get("naive_val_accuracy") is not None
        ]
    return val_accs, naive_accs


def mean_naive_baseline_accuracy(fold_naive_accs: dict) -> float:
    """Return the pooled mean naive validation accuracy across conditions/folds."""
    all_naive = [value for values in fold_naive_accs.values() for value in values]
    if not all_naive:
        return float("nan")
    return float(np.mean(all_naive))


def _to_percent(values):
    """Convert one scalar or array-like accuracy value from [0, 1] to [0, 100]."""
    return np.asarray(values, dtype="float64") * 100.0


def _resolve_percent_ylim(
    *,
    all_acc_percent,
    reference_baseline_percent: float,
    num_classes: int,
    explicit_y_lim: tuple = None,
):
    """Resolve percent-scaled y-limits shared by curve and bar plots."""
    if explicit_y_lim is not None:
        return float(explicit_y_lim[0]), float(explicit_y_lim[1])

    if len(all_acc_percent) == 0:
        y_min = max(0.0, reference_baseline_percent - 9.0)
        y_max = min(100.0, reference_baseline_percent + 9.0)
    else:
        y_min = max(0.0, min(min(all_acc_percent), reference_baseline_percent) - 4.0)
        y_max = min(100.0, max(max(all_acc_percent), reference_baseline_percent) + 4.0)
        if (y_max - y_min) < 18.0:
            mid = 0.5 * (y_min + y_max)
            half_span = 9.0
            y_min = max(0.0, mid - half_span)
            y_max = min(100.0, mid + half_span)

    if int(num_classes) == 2:
        y_min = TWO_CLASS_YMIN_PERCENT
        if y_max <= y_min:
            y_max = min(100.0, y_min + 10.0)
    return y_min, y_max


def plot_curves(
    agg_frame: pd.DataFrame,
    fold_frame: pd.DataFrame,
    condition_meta: list,
    fold_naive_accs: dict,
    output_pdf: Path,
    split_name: str = "val",
    figsize: tuple = (6.5, 3.5),
    show_title: bool = False,
    marker_every_minutes: int = None,
    y_lim: tuple = None,
    compact_layout: bool = False,
    compact_legend_loc: str = "best",
    compact_legend_ncol: int = 2,
    legend_bbox_to_anchor: tuple = None,
):
    """Render a cutoff-curve figure: fold mean with ±1 std shading."""
    _apply_pub_style(compact=compact_layout)

    if not condition_meta:
        raise ValueError("condition_meta is empty.")

    fig, ax = plt.subplots(figsize=figsize)

    num_classes = int(condition_meta[0].get("num_classes", 2))
    reference_baseline = 1.0 / float(max(num_classes, 1))
    reference_baseline_percent = float(_to_percent(reference_baseline))

    split_agg = agg_frame[agg_frame["split"] == split_name].copy()
    split_fold = (
        fold_frame[fold_frame["split"] == split_name].copy()
        if "split" in fold_frame.columns
        else fold_frame.copy()
    )

    all_acc_percent = _to_percent(split_fold["accuracy"].dropna().tolist())
    y_min, y_max = _resolve_percent_ylim(
        all_acc_percent=all_acc_percent,
        reference_baseline_percent=reference_baseline_percent,
        num_classes=num_classes,
        explicit_y_lim=y_lim,
    )

    uses_seconds = (
        "cutoff_kind" in split_fold.columns
        and not split_fold.empty
        and split_fold["cutoff_kind"].eq("observation_cutoff_seconds").all()
    )

    use_minutes_axis = (
        uses_seconds
        and "cutoff_value" in split_fold.columns
        and not split_fold["cutoff_value"].dropna().empty
        and all(
            abs((float(v) / 60.0) - round(float(v) / 60.0)) < 1e-6
            for v in split_fold["cutoff_value"].dropna().tolist()
        )
    )

    for row in condition_meta:
        condition = row["condition"]

        cdf = split_agg[split_agg["condition"] == condition].sort_values("cutoff_value").copy()

        if cdf.empty:
            continue

        x = cdf["cutoff_value"].astype(float)
        if use_minutes_axis:
            x = x / 60.0

        y_mean = _to_percent(cdf["accuracy_mean"].astype(float))
        y_std = _to_percent(cdf["accuracy_std"].fillna(0.0).astype(float))

        x_vals = x.values if hasattr(x, "values") else list(x)
        if marker_every_minutes is not None and use_minutes_axis:
            mark_indices = [i for i, v in enumerate(x_vals) if abs(v % marker_every_minutes) < 1e-6]
        else:
            mark_indices = None

        ax.plot(
            x,
            y_mean,
            marker=MODEL_MARKERS.get(row.get("model_type"), "s"),
            linewidth=1.1 if compact_layout else 2.2,
            markersize=2.7 if compact_layout else 5.5,
            markevery=mark_indices,
            color=row["color"],
            alpha=row.get("line_alpha", 1.0),
            label=_condition_label(row),
        )

        ax.fill_between(
            x,
            y_mean - y_std,
            y_mean + y_std,
            color=row["color"],
            alpha=0.12,
        )

    ax.axhline(
        reference_baseline_percent,
        color="#555555",
        linestyle="--",
        linewidth=1.4,
        label="_nolegend_",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Observed player time (minutes)" if use_minutes_axis else "Observed player time")
    ax.set_ylabel(Y_AXIS_LABEL)
    ax.grid(True, axis="y")
    ax.set_ylim(y_min, y_max)
    handles, labels = ax.get_legend_handles_labels()

    def _legend_sort_key(label):
        key = LEGEND_SORT_ALIAS.get(label, label)
        if key.startswith("Outcome"):
            return (0, key)
        if key.startswith("Single task"):
            return (1, key)
        if key.startswith("Both tasks"):
            return (2, key)
        return (3, key)

    if handles:
        pairs = sorted(zip(handles, labels), key=lambda p: _legend_sort_key(p[1]))
        sorted_handles, sorted_labels = zip(*pairs)
        legend_kwargs = {
            "loc": str(compact_legend_loc),
            "frameon": True,
            "ncol": int(compact_legend_ncol),
        }
        if legend_bbox_to_anchor is not None:
            legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
        if compact_layout:
            legend_kwargs.update(
                columnspacing=0.3,
                handlelength=0.9,
                handletextpad=0.22,
                markerscale=0.65,
            )
        ax.legend(sorted_handles, sorted_labels, **legend_kwargs)

    if show_title:
        title = ax.set_title(
            COMPACT_CURVE_TITLE if compact_layout else CURVE_TITLE,
            pad=1.0 if compact_layout else None,
        )
        title.set_wrap(True)

    if compact_layout:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), pad=0.12)
    else:
        fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    if compact_layout:
        fig.savefig(output_pdf)
    else:
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_bar_summary(
    fold_accs: dict,
    fold_naive_accs: dict,
    condition_meta: list,
    output_pdf: Path,
    split_name: str = "val",
    figsize: tuple = (6.0, 3.2),
    show_title: bool = False,
    y_lim: tuple = None,
):
    """Render a bar chart of mean accuracy ± std per condition.

    ``fold_accs`` is {condition: [accuracy per fold]} for ``split_name`` (the selected runs'
    validation or training accuracy).
    """
    _apply_pub_style()
    fig, ax = plt.subplots(figsize=figsize)

    labels = [_condition_label(row) for row in condition_meta]
    conditions = [row["condition"] for row in condition_meta]
    colors = [row["color"] for row in condition_meta]
    alphas = [row["line_alpha"] for row in condition_meta]

    means, stds = [], []
    for condition in conditions:
        vals = fold_accs.get(condition, [])
        means.append(float(_to_percent(np.mean(vals))) if vals else 0.0)
        stds.append(float(_to_percent(np.std(vals))) if len(vals) > 1 else 0.0)

    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        means,
        color=colors,
        yerr=stds,
        capsize=4,
        error_kw={"elinewidth": 1.4, "ecolor": "#333333"},
    )
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)

    num_classes = int(condition_meta[0].get("num_classes", 2)) if condition_meta else 2
    chance_baseline = 1.0 / float(max(num_classes, 1))
    ax.axhline(
        float(_to_percent(chance_baseline)),
        color="#555555",
        linestyle="--",
        linewidth=1.4,
        label="_nolegend_",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(Y_AXIS_LABEL)
    upper_vals = [m + s for m, s in zip(means, stds)]
    lower_vals = [m - s for m, s in zip(means, stds)]
    y_min, y_max = _resolve_percent_ylim(
        all_acc_percent=lower_vals + upper_vals,
        reference_baseline_percent=float(_to_percent(chance_baseline)),
        num_classes=num_classes,
        explicit_y_lim=y_lim,
    )
    ax.set_ylim(y_min, y_max)
    ax.grid(True, axis="y")

    if show_title:
        ax.set_title(BAR_TITLE)
    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def write_selected_runs_csv(rows_by_condition_fold: dict, output_csv: Path):
    """Write a CSV with one row per condition × fold: best-run name and metadata."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "fold_id",
        "condition_label",
        "model_name",
        "val_accuracy",
        "naive_val_accuracy",
        "latent_dim",
        "dim_reduction_method",
        "num_epochs",
        "target_mode",
        "seq_length",
    ]
    rows = []
    for condition, fold_rows in rows_by_condition_fold.items():
        for fold_id, row in sorted(fold_rows.items()):
            r = dict(row)
            r["fold_id"] = fold_id
            rows.append(r)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _condition_label(row) -> str:
    """Display label for one condition row, honouring a per-figure override."""
    return row.get("label_override") or CONDITION_LABELS.get(row["condition"], row["condition"])


def _plvm2_only_meta(condition_meta: list) -> list:
    """Return condition_meta with all hierarchical PLVM dims removed except PLVM-2.

    Logistic-probe conditions share ``model_type=plvm_both_partial`` but are always
    kept. PLVM-2 is recoloured to the fixed PLVM blue.
    """
    filtered = [
        row
        for row in condition_meta
        if row["model_type"] != "plvm_both_partial"
        or row.get("prediction_head_type") != "hierarchical"
        or row["condition"] == "PLVM-2"
    ]
    # Copy rather than mutate in place: these dicts are shared with the
    # unfiltered condition_meta used by the multi-dimension Figure 9/10 plots,
    # which must keep their own per-dimension palette colors.
    return [
        dict(row, color=PLVM_SINGLE_DIM_COLOR, label_override="Process: PLVM")
        if row["condition"] == "PLVM-2"
        else row
        for row in filtered
    ]


def _resolve_named_manuscript_curve_path(target_mode: str) -> Path:
    """Return the default Figure 6 or 8 path for a PowerWash target, if it has one."""
    filename = MANUSCRIPT_CURVE_FILENAMES.get(str(target_mode).strip())
    if filename is None:
        return None
    return DEFAULT_FIGURES_DIR / filename


def _resolve_named_manuscript_multi_plvm_curve_path(target_mode: str) -> Path:
    """Return the default Figure 9 or 10 path (every PLVM latent dim), if any."""
    filename = MANUSCRIPT_MULTI_PLVM_CURVE_FILENAMES.get(str(target_mode).strip())
    if filename is None:
        return None
    return DEFAULT_FIGURES_DIR / filename


def _summary_family_for_model_type(model_type: str):
    """Map one selected-condition model type onto the summary-table families."""
    if model_type == "rf_baseline":
        return "Outcome baseline"
    if model_type == "single_task":
        return "Single-task process"
    if model_type == "plvm_both_partial":
        return "PLVM"
    return None


def _format_summary_stat(mean_value: float, std_value: float, scale: float, decimals: int) -> str:
    """Format one mean±std summary cell, omitting the spread when unavailable."""
    if not np.isfinite(mean_value):
        return "NA"
    scaled_mean = float(mean_value) * float(scale)
    if std_value is None or not np.isfinite(std_value):
        return "{:.{decimals}f}".format(scaled_mean, decimals=decimals)
    scaled_std = float(std_value) * float(scale)
    if scaled_std <= 0.0:
        return "{:.{decimals}f}".format(scaled_mean, decimals=decimals)
    return "{:.{decimals}f} ± {:.{decimals}f}".format(
        scaled_mean,
        scaled_std,
        decimals=decimals,
    )


def _format_constant_summary_stat(value: float, scale: float, decimals: int) -> str:
    """Format one deterministic baseline cell without a spread term."""
    if not np.isfinite(value):
        return "NA"
    return "{:.{decimals}f}".format(float(value) * float(scale), decimals=decimals)


def _condition_average_accuracy_scores(
    fold_frame: pd.DataFrame,
    split_name: str,
) -> dict:
    """Return mean validation accuracy across all available cutoffs for each condition."""
    split_frame = fold_frame.loc[fold_frame["split"] == split_name].copy()
    if split_frame.empty:
        return {}
    per_fold_scores = split_frame.groupby(["condition", "fold_id"])["accuracy"].mean().reset_index()
    grouped = per_fold_scores.groupby("condition")["accuracy"].mean().to_dict()
    return {
        str(condition): float(score) for condition, score in grouped.items() if np.isfinite(score)
    }


def _normalized_accuracy_auc_by_fold(
    condition_frame: pd.DataFrame,
    max_minutes: float,
) -> list:
    """
    Compute per-fold normalized area under the accuracy-time curve.

    The curve is integrated over the observed cutoff grid up to ``max_minutes``
    and divided by the observed time span so the result stays on the same
    [0, 1] scale as accuracy.
    """
    if condition_frame.empty:
        return []

    cutoff_limit_seconds = float(max_minutes) * 60.0
    subset = condition_frame.loc[
        pd.to_numeric(condition_frame["cutoff_value"], errors="coerce") <= cutoff_limit_seconds
    ].copy()
    if subset.empty:
        return []

    auc_values = []
    integrate_fn = getattr(np, "trapezoid", None)
    if integrate_fn is None:
        integrate_fn = getattr(np, "trapz")
    for _, fold_subset in subset.groupby("fold_id"):
        ordered = fold_subset.sort_values("cutoff_value").copy()
        x_minutes = ordered["cutoff_value"].astype(float).to_numpy() / 60.0
        y_accuracy = ordered["accuracy"].astype(float).to_numpy()
        if x_minutes.size < 2:
            continue
        span_minutes = float(x_minutes[-1] - x_minutes[0])
        if span_minutes <= 0.0:
            continue
        auc_values.append(float(integrate_fn(y_accuracy, x_minutes) / span_minutes))
    return auc_values


def build_family_summary_rows(
    fold_frame: pd.DataFrame,
    condition_meta: list,
    split_name: str = "val",
    timepoints_minutes: tuple = SUMMARY_TIMEPOINT_MINUTES,
    auc_max_minutes: float = SUMMARY_AUC_MAX_MINUTES,
) -> list:
    """
    Build the per-family summary table for one PowerWash target.

    Each learnable family keeps the condition with the highest mean validation
    accuracy across all available cutoffs, then reports fold mean ± std at the
    requested time points plus a normalized accuracy-curve AUC through
    ``auc_max_minutes``.
    """
    if not condition_meta:
        raise ValueError("condition_meta is empty.")

    split_frame = fold_frame.loc[fold_frame["split"] == split_name].copy()
    if split_frame.empty:
        raise ValueError("No rows found for split '{}'.".format(split_name))

    condition_types = {
        str(row["condition"]): str(row.get("model_type", "")) for row in condition_meta
    }
    condition_scores = _condition_average_accuracy_scores(fold_frame, split_name)
    best_condition_by_family = {}
    for condition, model_type in sorted(condition_types.items()):
        family = _summary_family_for_model_type(model_type)
        if family is None:
            continue
        score = condition_scores.get(condition)
        if score is None:
            continue
        current = best_condition_by_family.get(family)
        if current is None or score > current["score"]:
            best_condition_by_family[family] = {
                "condition": condition,
                "score": float(score),
            }

    num_classes = int(condition_meta[0].get("num_classes", 2))
    chance_accuracy = 1.0 / float(max(num_classes, 1))
    rows = [
        {
            "Model family": "Chance baseline",
            "Inputs": SUMMARY_FAMILY_INPUTS["Chance baseline"],
            "20 min Acc.": _format_constant_summary_stat(chance_accuracy, scale=100.0, decimals=1),
            "60 min Acc.": _format_constant_summary_stat(chance_accuracy, scale=100.0, decimals=1),
            "120 min Acc.": _format_constant_summary_stat(chance_accuracy, scale=100.0, decimals=1),
            "AUC at 120 min": _format_constant_summary_stat(chance_accuracy, scale=1.0, decimals=2),
        }
    ]

    for family in SUMMARY_FAMILY_ORDER[1:]:
        selected = best_condition_by_family.get(family)
        if selected is None:
            continue

        condition = selected["condition"]
        condition_frame = split_frame.loc[split_frame["condition"] == condition].copy()
        if condition_frame.empty:
            continue

        row = {
            "Model family": family,
            "Inputs": SUMMARY_FAMILY_INPUTS[family],
        }
        for minute_value in timepoints_minutes:
            cutoff_seconds = float(minute_value) * 60.0
            cutoff_frame = condition_frame.loc[
                np.isclose(
                    condition_frame["cutoff_value"].astype(float).to_numpy(),
                    cutoff_seconds,
                    atol=1e-6,
                )
            ].copy()
            per_fold_values = [
                float(value)
                for value in cutoff_frame["accuracy"].astype(float).tolist()
                if np.isfinite(value)
            ]
            mean_value = float(np.mean(per_fold_values)) if per_fold_values else float("nan")
            std_value = float(np.std(per_fold_values, ddof=1)) if len(per_fold_values) > 1 else 0.0
            row["{} min Acc.".format(int(minute_value))] = _format_summary_stat(
                mean_value,
                std_value,
                scale=100.0,
                decimals=1,
            )

        auc_values = _normalized_accuracy_auc_by_fold(condition_frame, auc_max_minutes)
        auc_mean = float(np.mean(auc_values)) if auc_values else float("nan")
        auc_std = float(np.std(auc_values, ddof=1)) if len(auc_values) > 1 else 0.0
        row["AUC at 120 min"] = _format_summary_stat(
            auc_mean,
            auc_std,
            scale=1.0,
            decimals=2,
        )
        rows.append(row)

    return rows


def write_family_summary_table(rows: list, output_csv: Path):
    """Write the per-family summary table to CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(output_csv, index=False)


def main(argv=None):
    """Run the fold-averaged cutoff-curve plotting workflow."""
    args = parse_args(argv)
    target_modes = resolve_target_modes(args)
    wrote_any = False

    for target_mode in target_modes:
        output_pdf, output_csv = _resolve_output_paths(args, target_mode, len(target_modes))
        output_bar_pdf = output_pdf.with_name(output_pdf.stem + "_bar_summary.pdf")
        output_runs_csv = output_pdf.with_name(output_pdf.stem + "_selected_runs.csv")

        rows = load_candidate_rows(args.model_performance_csv, args.seq_length, target_mode)
        if not rows:
            print(
                "Skipping {}: no rows found for seq_length={} in {}.".format(
                    target_mode,
                    args.seq_length,
                    args.model_performance_csv,
                ),
                file=sys.stderr,
            )
            continue

        condition_meta, rows_by_condition_fold = select_all_rows_by_fold(rows)
        if not rows_by_condition_fold:
            print(
                "Skipping {}: no conditions with available runs.".format(target_mode),
                file=sys.stderr,
            )
            continue

        logistic_condition_names = []
        logistic_csv_path = (
            Path(args.training_logs_dir) / "concat_logistic_probe" / "model_performance.csv"
        )
        if logistic_csv_path.exists():
            logistic_rows = load_logistic_rows(logistic_csv_path, args.seq_length, target_mode)
            logistic_selection = select_best_logistic_condition(
                args.training_logs_dir, logistic_rows
            )
            if logistic_selection is not None:
                logistic_condition_name, logistic_fold_rows = logistic_selection
                rows_by_condition_fold[logistic_condition_name] = logistic_fold_rows
                condition_meta.append(dict(next(iter(logistic_fold_rows.values()))))
                logistic_condition_names.append(logistic_condition_name)

        single_task_logistic_csv_path = (
            Path(args.training_logs_dir) / "single_task_logistic_probe" / "model_performance.csv"
        )
        if single_task_logistic_csv_path.exists():
            single_task_logistic_rows = load_logistic_rows(
                single_task_logistic_csv_path, args.seq_length, target_mode
            )
            for condition_name, fold_rows in select_single_task_logistic_conditions(
                single_task_logistic_rows
            ):
                rows_by_condition_fold[condition_name] = fold_rows
                condition_meta.append(dict(next(iter(fold_rows.values()))))
                logistic_condition_names.append(condition_name)

        try:
            fold_frame = build_fold_long_frame(args.training_logs_dir, rows_by_condition_fold)
        except ValueError as exc:
            print("Skipping {}: {}".format(target_mode, exc), file=sys.stderr)
            continue

        agg_frame = aggregate_folds(fold_frame)
        fold_val_accs, fold_naive_accs = collect_fold_val_accs(rows_by_condition_fold)
        fold_accs_by_split = {
            "val": fold_val_accs,
            "train": collect_fold_train_accs(
                rows_by_condition_fold,
                load_train_accuracy_by_model_name(
                    args.model_performance_csv, args.training_logs_dir
                ),
            ),
        }

        base_dir = output_pdf.parent
        paper_dir = base_dir / "paper_figures"
        detail_dir = base_dir / "detailed_figures"
        tables_dir = DEFAULT_TABLES_DIR
        stem = output_pdf.stem
        bar_stem = output_bar_pdf.stem

        detail_csv = detail_dir / output_csv.name
        detail_csv.parent.mkdir(parents=True, exist_ok=True)
        fold_frame.to_csv(detail_csv, index=False)

        meta_plvm2 = _plvm2_only_meta(condition_meta)

        # Figures 6 and 8 (paper_figures/): PLVM-2 only, validation split.
        paper_y_lim = (25.0, 50.0) if "3_class" in target_mode else (45.0, 80.0)
        paper_curves = paper_dir / "{}_val.pdf".format(stem)
        plot_curves(
            agg_frame,
            fold_frame,
            meta_plvm2,
            fold_naive_accs,
            paper_curves,
            split_name="val",
            figsize=MANUSCRIPT_CURVE_FIGSIZE,
            marker_every_minutes=10,
            y_lim=paper_y_lim,
            compact_layout=False,
            compact_legend_loc="upper center",
            compact_legend_ncol=2,
            legend_bbox_to_anchor=(0.5, -0.18),
        )
        paper_bar = paper_dir / "{}_val.pdf".format(bar_stem)
        plot_bar_summary(
            fold_val_accs,
            fold_naive_accs,
            meta_plvm2,
            paper_bar,
            split_name="val",
            figsize=(6.0, 3.8),
            y_lim=paper_y_lim,
        )
        print("Wrote {}".format(paper_curves))
        print("Wrote {}".format(paper_bar))
        if args.manuscript_figure:
            named_manuscript_curve = _resolve_named_manuscript_curve_path(target_mode)
            if named_manuscript_curve is not None:
                named_curve_output = copy_figure_to_path(
                    paper_curves,
                    Path(args.manuscript_figure_dir) / named_manuscript_curve.name,
                )
                print("Wrote {}".format(named_curve_output))

        # Detailed figures: all conditions, both splits.
        dimensionality_meta = [
            row
            for row in condition_meta
            if row["condition"] == "Bungalow Alone"
            or (
                row["model_type"] == "plvm_both_partial"
                and row.get("prediction_head_type") == "hierarchical"
            )
        ]
        for split in ("val", "train"):
            curves_pdf = detail_dir / "{}_detailed_{}.pdf".format(stem, split)
            plot_curves(
                agg_frame,
                fold_frame,
                condition_meta,
                fold_naive_accs,
                curves_pdf,
                split_name=split,
                figsize=MANUSCRIPT_CURVE_FIGSIZE,
                compact_legend_loc="upper center",
                compact_legend_ncol=2,
                legend_bbox_to_anchor=(0.5, -0.18),
            )
            bar_pdf = detail_dir / "{}_detailed_{}.pdf".format(bar_stem, split)
            plot_bar_summary(
                fold_accs_by_split[split],
                fold_naive_accs,
                condition_meta,
                bar_pdf,
                split_name=split,
                figsize=(6.0, 3.2),
            )
            curves_plvm2_pdf = detail_dir / "{}_plvm2_{}.pdf".format(stem, split)
            plot_curves(
                agg_frame,
                fold_frame,
                meta_plvm2,
                fold_naive_accs,
                curves_plvm2_pdf,
                split_name=split,
                figsize=(5.0, 3.5),
                marker_every_minutes=10,
            )
            bar_plvm2_pdf = detail_dir / "{}_plvm2_{}.pdf".format(bar_stem, split)
            plot_bar_summary(
                fold_accs_by_split[split],
                fold_naive_accs,
                meta_plvm2,
                bar_plvm2_pdf,
                split_name=split,
                figsize=(5.0, 3.2),
            )
            print("Wrote detailed + plvm2 figures for split={}".format(split))
            if args.manuscript_figure and split == "val":
                named_multi_plvm_curve = _resolve_named_manuscript_multi_plvm_curve_path(
                    target_mode
                )
                if named_multi_plvm_curve is not None:
                    if named_multi_plvm_curve.name == "Figure_9.pdf":
                        # Figure 9 compares PLVM latent dims: the nonlinear Bungalow
                        # single-task reference plus every hierarchical PLVM latent dim.
                        source_curve = detail_dir / "{}_dimensionality_{}.pdf".format(stem, split)
                        plot_curves(
                            agg_frame,
                            fold_frame,
                            dimensionality_meta,
                            fold_naive_accs,
                            source_curve,
                            split_name=split,
                            figsize=MANUSCRIPT_CURVE_FIGSIZE,
                            marker_every_minutes=10,
                            compact_legend_loc="upper center",
                            compact_legend_ncol=2,
                            legend_bbox_to_anchor=(0.5, -0.18),
                        )
                    else:
                        source_curve = curves_pdf
                    named_multi_plvm_output = copy_figure_to_path(
                        source_curve,
                        Path(args.manuscript_figure_dir) / named_multi_plvm_curve.name,
                    )
                    print("Wrote {}".format(named_multi_plvm_output))

        detail_runs_csv = detail_dir / output_runs_csv.name
        write_selected_runs_csv(rows_by_condition_fold, detail_runs_csv)
        family_summary_csv = tables_dir / "{}_family_summary_table.csv".format(stem)
        family_summary_condition_meta = [
            row for row in condition_meta if row["condition"] not in logistic_condition_names
        ]
        family_summary_rows = build_family_summary_rows(fold_frame, family_summary_condition_meta)
        write_family_summary_table(family_summary_rows, family_summary_csv)
        wrote_any = True

        print("Wrote {}".format(detail_csv))
        print("Wrote {}".format(detail_runs_csv))
        print("Wrote {}".format(family_summary_csv))
        for row in condition_meta:
            vals = fold_val_accs.get(row["condition"], [])
            print(
                "{}: mean_val={:.4f} ± {:.4f} ({} folds)".format(
                    CONDITION_LABELS.get(row["condition"], row["condition"]),
                    float(np.mean(vals)) if vals else float("nan"),
                    float(np.std(vals)) if len(vals) > 1 else 0.0,
                    len(vals),
                )
            )

    if not wrote_any:
        raise ValueError(
            "No target-mode plots were generated for seq_length={} from {}.".format(
                args.seq_length,
                args.model_performance_csv,
            )
        )


if __name__ == "__main__":
    main()
