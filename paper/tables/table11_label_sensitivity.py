#!/usr/bin/env python3
"""Generates Table 11: Fire Station label-formula sensitivity, PLVM-2 vs Bungalow single task.

Compares the original zone-planning labels with four leave-one-component-out
variants from saved runs. Per variant, target and fold it keeps the run with the
lowest ``val_loss`` (Tables 2 and 10 select by validation accuracy instead) and
scores its validation predictions at fixed cutoffs, with paired Wilcoxon tests.
Writes to ``manuscript_figures/tables/table11_label_sensitivity/``.
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, log_loss


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from powerwash.train.powerwash_hierarchical_log_utils import (  # noqa: E402
    resolve_powerwash_hierarchical_run_dir,
)
from paper.output_paths import resolve_tables_dir  # noqa: E402
from plvm import artifact_paths  # noqa: E402


TRAINING_LOGS_ROOT = artifact_paths.POWERWASH_TRAINING_LOGS_DIR
DEFAULT_PLVM_FAMILY_DIR = TRAINING_LOGS_ROOT / "plvm_models"
DEFAULT_SINGLE_TASK_FAMILY_DIR = TRAINING_LOGS_ROOT / "single_task"
DEFAULT_SENSITIVITY_PLVM_FAMILY_DIR = TRAINING_LOGS_ROOT / "plvm_models_firestation_sensitivity"
DEFAULT_SENSITIVITY_SINGLE_TASK_FAMILY_DIR = (
    TRAINING_LOGS_ROOT / "single_task_firestation_sensitivity"
)
DEFAULT_OUTPUT_DIR = resolve_tables_dir() / "table11_label_sensitivity"
DEFAULT_CUTOFF_MINUTES = (20, 60, 120)
VALID_TARGETS = (
    "player_label_2_class_clean_zonehopping",
    "player_label_3_class_zonehopping",
)
LABEL_VARIANT_ORDER = (
    "original",
    "drop_zone_switch_rate",
    "drop_return_rate",
    "drop_mean_zone_run_length",
    "drop_mean_zone_jump_distance",
)
LABEL_VARIANT_DISPLAY = {
    "original": "Original",
    "drop_zone_switch_rate": "Drop zone switch rate",
    "drop_return_rate": "Drop return rate",
    "drop_mean_zone_run_length": "Drop mean zone run length",
    "drop_mean_zone_jump_distance": "Drop mean zone jump distance",
}
TARGET_DISPLAY = {
    "player_label_2_class_clean_zonehopping": "2-class",
    "player_label_3_class_zonehopping": "3-class",
}
FIXED_SINGLE_TASK_NAME = "bungalow"
FIXED_PLVM_LATENT_DIM = 2


@dataclass(frozen=True)
class SelectedRun:
    """Metadata for one chosen Fire Station run."""

    label_variant: str
    family: str
    run_name: str
    fold_number: int
    model_type: str
    target_name: str
    predicted_next_task: str
    val_loss: float
    val_accuracy: float
    prediction_head_type: str
    run_root: Path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plvm_family_dir",
        type=Path,
        default=DEFAULT_PLVM_FAMILY_DIR,
        help="Directory containing original Fire Station PLVM runs.",
    )
    parser.add_argument(
        "--single_task_family_dir",
        type=Path,
        default=DEFAULT_SINGLE_TASK_FAMILY_DIR,
        help="Directory containing original Fire Station single-task runs.",
    )
    parser.add_argument(
        "--sensitivity_plvm_family_dir",
        type=Path,
        default=DEFAULT_SENSITIVITY_PLVM_FAMILY_DIR,
        help="Directory containing sensitivity-only PLVM runs.",
    )
    parser.add_argument(
        "--sensitivity_single_task_family_dir",
        type=Path,
        default=DEFAULT_SENSITIVITY_SINGLE_TASK_FAMILY_DIR,
        help="Directory containing sensitivity-only single-task runs.",
    )
    parser.add_argument(
        "--cutoff_minutes",
        type=str,
        default=",".join(str(minutes) for minutes in DEFAULT_CUTOFF_MINUTES),
        help="Comma-separated cutoff minutes to summarize.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the CSV, JSON and markdown outputs are written.",
    )
    return parser.parse_args(argv)


def _parse_cutoff_minutes(raw_value: str) -> List[int]:
    """Parse and validate requested cutoff minutes."""
    values: List[int] = []
    for token in str(raw_value or "").split(","):
        token = token.strip()
        if not token:
            continue
        minutes = int(token)
        if minutes <= 0:
            raise ValueError("Cutoff minutes must be positive integers.")
        values.append(minutes)
    if not values:
        raise ValueError("At least one cutoff minute is required.")
    return sorted(set(values))


def _parse_json_dict(raw_value: object) -> dict:
    """Best-effort parser for one JSON payload column."""
    if isinstance(raw_value, dict):
        return raw_value
    if not isinstance(raw_value, str) or not raw_value.strip():
        return {}
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_target_name(parameters_to_predict: str) -> str:
    """Strip the trailing `_label` suffix from player-label targets."""
    normalized = str(parameters_to_predict or "").strip()
    if normalized.endswith("_label"):
        return normalized[: -len("_label")]
    return normalized


def _extract_label_variant(player_label_csv: str) -> str:
    """Infer the Fire Station label variant from the saved player-label CSV path."""
    normalized = str(player_label_csv or "").replace("\\", "/").strip().lower()
    sensitivity_token = "/firestation_zone_labels_sensitivity/"
    original_token = "/firestation_zone_labels/"
    if sensitivity_token in normalized:
        suffix = normalized.split(sensitivity_token, 1)[1]
        parts = [part for part in suffix.split("/") if part]
        if parts:
            return parts[0]
    if original_token in normalized:
        return "original"
    return ""


def _read_model_performance_rows(csv_path: Path, family: str) -> List[dict]:
    """Load raw model-performance rows from one Fire Station family CSV."""
    if not csv_path.exists():
        raise FileNotFoundError("model_performance.csv not found: {}".format(csv_path))
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        payload = _parse_json_dict(row.get("all_arguments_json"))
        parameters_to_predict = str(
            row.get("parameters_to_predict") or payload.get("parameters_to_predict") or ""
        ).strip()
        target_name = _normalize_target_name(parameters_to_predict)
        player_label_csv = str(
            payload.get("player_label_csv") or row.get("player_label_csv") or ""
        ).strip()
        row["_family"] = family
        row["_run_root"] = str(csv_path.parent)
        row["_all_arguments"] = payload
        row["_target_name"] = target_name
        row["_player_label_csv"] = player_label_csv
        row["_label_variant"] = _extract_label_variant(player_label_csv)
    return rows


def _filter_candidate_rows(rows: Sequence[dict]) -> List[dict]:
    """Keep only Fire Station zone-hopping rows for original or drop variants."""
    filtered: List[dict] = []
    for row in rows:
        target_name = str(row.get("_target_name", "")).strip()
        label_variant = str(row.get("_label_variant", "")).strip()
        if target_name not in VALID_TARGETS:
            continue
        if label_variant not in LABEL_VARIANT_ORDER:
            continue
        filtered.append(row)
    return filtered


def _select_fixed_plvm_runs_by_fold(
    rows: Sequence[dict],
) -> Dict[Tuple[str, str, int], SelectedRun]:
    """Select the PLVM-2 run with the lowest ``val_loss`` per (label_variant, target, fold)."""
    best: Dict[Tuple[str, str, int], SelectedRun] = {}
    for row in rows:
        latent_dim = int(row.get("latent_dim") or -1)
        if latent_dim != FIXED_PLVM_LATENT_DIM:
            continue
        label_variant = str(row["_label_variant"])
        target_name = str(row["_target_name"])
        fold_number = int(row.get("fold_number") or -1)
        key = (label_variant, target_name, fold_number)
        candidate = SelectedRun(
            label_variant=label_variant,
            family="plvm",
            run_name=str(row.get("model_name", "")).strip(),
            fold_number=fold_number,
            model_type=str(row.get("model_type", "")).strip(),
            target_name=target_name,
            predicted_next_task=str(row.get("predicted_next_task", "")).strip(),
            val_loss=float(row.get("val_loss") or "inf"),
            val_accuracy=float(row.get("val_accuracy") or "nan"),
            prediction_head_type=str(row.get("prediction_head_type", "")).strip(),
            run_root=Path(str(row["_run_root"])),
        )
        current = best.get(key)
        if current is None or candidate.val_loss < current.val_loss:
            best[key] = candidate
    return best


def _select_best_single_task_runs_by_fold(
    rows: Sequence[dict],
) -> Dict[Tuple[str, str, str, int], SelectedRun]:
    """Select the lowest-``val_loss`` single-task run per (label_variant, target, task, fold)."""
    best: Dict[Tuple[str, str, str, int], SelectedRun] = {}
    for row in rows:
        label_variant = str(row["_label_variant"])
        target_name = str(row["_target_name"])
        task_name = str(row.get("predicted_next_task", "")).strip()
        fold_number = int(row.get("fold_number") or -1)
        key = (label_variant, target_name, task_name, fold_number)
        candidate = SelectedRun(
            label_variant=label_variant,
            family="single_task",
            run_name=str(row.get("model_name", "")).strip(),
            fold_number=fold_number,
            model_type=str(row.get("model_type", "")).strip(),
            target_name=target_name,
            predicted_next_task=task_name,
            val_loss=float(row.get("val_loss") or "inf"),
            val_accuracy=float(row.get("val_accuracy") or "nan"),
            prediction_head_type=str(row.get("prediction_head_type", "")).strip(),
            run_root=Path(str(row["_run_root"])),
        )
        current = best.get(key)
        if current is None or candidate.val_loss < current.val_loss:
            best[key] = candidate
    return best


def _load_val_predictions(run: SelectedRun) -> pd.DataFrame:
    """Load one saved val_predictions.csv file."""
    run_dir = resolve_powerwash_hierarchical_run_dir(
        run.run_root,
        run.run_name,
        model_type=run.model_type,
    )
    path = run_dir / "val_predictions.csv"
    frame = pd.read_csv(path)
    required = {
        "player_id",
        "task_name",
        "observation_cutoff_seconds",
        "target",
        "target_label",
        "prediction_class_id",
        "prediction_label",
        "prediction_probabilities_json",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("Missing columns {} in {}".format(missing, path))
    frame = frame.copy()
    frame["observation_cutoff_seconds"] = pd.to_numeric(
        frame["observation_cutoff_seconds"], errors="coerce"
    )
    frame["target"] = pd.to_numeric(frame["target"], errors="coerce")
    frame["prediction_class_id"] = pd.to_numeric(frame["prediction_class_id"], errors="coerce")
    frame = frame.dropna(
        subset=["observation_cutoff_seconds", "target", "prediction_class_id"]
    ).copy()
    frame["target"] = frame["target"].astype(int)
    frame["prediction_class_id"] = frame["prediction_class_id"].astype(int)
    frame["probabilities"] = frame["prediction_probabilities_json"].apply(json.loads)
    return frame


def _compute_classification_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    """Compute classification metrics for one cutoff slice."""
    if frame.empty:
        return {
            "n_rows": 0,
            "n_players": 0,
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "cross_entropy": float("nan"),
        }

    y_true = frame["target"].to_numpy(dtype=np.int64)
    y_pred = frame["prediction_class_id"].to_numpy(dtype=np.int64)
    prob_matrix = np.asarray(frame["probabilities"].tolist(), dtype=np.float64)
    class_ids = list(range(prob_matrix.shape[1]))

    return {
        "n_rows": int(len(frame)),
        "n_players": int(frame["player_id"].nunique()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cross_entropy": float(log_loss(y_true, prob_matrix, labels=class_ids)),
    }


def _summarize_run_at_cutoffs(
    run: SelectedRun,
    cutoff_minutes: Sequence[int],
) -> pd.DataFrame:
    """Summarize one selected run at the requested cutoffs."""
    frame = _load_val_predictions(run)
    rows: List[dict] = []
    for minutes in cutoff_minutes:
        cutoff_seconds = float(minutes * 60)
        cutoff_frame = frame[np.isclose(frame["observation_cutoff_seconds"], cutoff_seconds)].copy()
        metrics = _compute_classification_metrics(cutoff_frame)
        rows.append(
            {
                "label_variant": run.label_variant,
                "target_name": run.target_name,
                "run_family": run.family,
                "run_name": run.run_name,
                "fold_number": run.fold_number,
                "model_type": run.model_type,
                "predicted_next_task": run.predicted_next_task,
                "prediction_head_type": run.prediction_head_type,
                "cutoff_minutes": int(minutes),
                "cutoff_seconds": cutoff_seconds,
                "val_loss": run.val_loss,
                "val_accuracy": run.val_accuracy,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _select_fixed_single_task_rows(single_task_rows: pd.DataFrame) -> pd.DataFrame:
    """Keep only the fixed single-task comparator used in Table 11."""
    selected = single_task_rows[
        single_task_rows["predicted_next_task"].astype(str).str.strip() == FIXED_SINGLE_TASK_NAME
    ].copy()
    if selected.empty:
        return pd.DataFrame(columns=list(single_task_rows.columns))
    selected["run_family"] = "single_task_fixed"
    return selected.reset_index(drop=True)


def _format_percentage(value: float, digits: int = 1) -> str:
    """Format one accuracy-like value as a percent."""
    if pd.isna(value):
        return "NA"
    return "{:.{digits}f}".format(float(value) * 100.0, digits=digits)


def _format_mean_sd_percentage(mean_value: float, sd_value: float) -> str:
    """Format one mean/sd accuracy pair as a percentage string."""
    if pd.isna(mean_value):
        return "NA"
    if pd.isna(sd_value):
        return _format_percentage(mean_value)
    return "{} ± {}".format(
        _format_percentage(mean_value),
        _format_percentage(sd_value),
    )


def _format_pvalue(value: float) -> str:
    """Format one p-value compactly for the tables."""
    if pd.isna(value):
        return "NA"
    if float(value) < 1e-3:
        return "<0.001"
    return "{:.3f}".format(float(value))


def _wilcoxon_pvalue(values_a: pd.Series, values_b: pd.Series) -> float:
    """Return the Wilcoxon signed-rank p-value for one shared-fold comparison."""
    deltas = values_a.astype(float).to_numpy() - values_b.astype(float).to_numpy()
    if deltas.size <= 1 or np.allclose(deltas, 0.0):
        return float("nan")
    try:
        return float(stats.wilcoxon(deltas, zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def _build_accuracy_table_rows(
    plvm_rows: pd.DataFrame,
    best_single_task_rows: pd.DataFrame,
    cutoff_minutes: Sequence[int],
) -> pd.DataFrame:
    """Aggregate mean ± s.d. accuracies for the formatted outputs."""
    output_rows: List[dict] = []
    for label_variant in LABEL_VARIANT_ORDER:
        for target_name in VALID_TARGETS:
            for family_name, frame in [
                ("Single-task process", best_single_task_rows),
                ("PLVM", plvm_rows),
            ]:
                subset = frame[
                    (frame["label_variant"] == label_variant)
                    & (frame["target_name"] == target_name)
                ].copy()
                if subset.empty:
                    continue
                row = {
                    "label_variant": label_variant,
                    "formula": LABEL_VARIANT_DISPLAY[label_variant],
                    "target_name": target_name,
                    "target_label": TARGET_DISPLAY[target_name],
                    "model_family": family_name,
                }
                for minutes in cutoff_minutes:
                    cutoff_subset = subset[subset["cutoff_minutes"] == int(minutes)].copy()
                    row["{} min mean accuracy".format(minutes)] = float(
                        cutoff_subset["accuracy"].mean()
                    )
                    row["{} min sd accuracy".format(minutes)] = float(
                        cutoff_subset["accuracy"].std(ddof=1)
                    )
                output_rows.append(row)
    return pd.DataFrame(output_rows)


def _build_significance_rows(
    plvm_rows: pd.DataFrame,
    best_single_task_rows: pd.DataFrame,
    cutoff_minutes: Sequence[int],
) -> pd.DataFrame:
    """Build one Wilcoxon comparison row per formula, target, and cutoff."""
    rows: List[dict] = []
    for label_variant in LABEL_VARIANT_ORDER:
        for target_name in VALID_TARGETS:
            plvm_subset = plvm_rows[
                (plvm_rows["label_variant"] == label_variant)
                & (plvm_rows["target_name"] == target_name)
            ].copy()
            single_subset = best_single_task_rows[
                (best_single_task_rows["label_variant"] == label_variant)
                & (best_single_task_rows["target_name"] == target_name)
            ].copy()
            if plvm_subset.empty or single_subset.empty:
                continue
            for minutes in cutoff_minutes:
                paired = plvm_subset[["fold_number", "accuracy"]][
                    plvm_subset["cutoff_minutes"] == int(minutes)
                ].merge(
                    single_subset[["fold_number", "accuracy"]][
                        single_subset["cutoff_minutes"] == int(minutes)
                    ],
                    on="fold_number",
                    how="inner",
                    suffixes=("_plvm", "_single_task"),
                )
                if paired.empty:
                    continue
                deltas = paired["accuracy_plvm"] - paired["accuracy_single_task"]
                rows.append(
                    {
                        "label_variant": label_variant,
                        "formula": LABEL_VARIANT_DISPLAY[label_variant],
                        "target_name": target_name,
                        "target_label": TARGET_DISPLAY[target_name],
                        "cutoff_minutes": int(minutes),
                        "n_shared_folds": int(len(paired)),
                        "mean_delta_accuracy": float(deltas.mean()),
                        "sd_delta_accuracy": float(deltas.std(ddof=1)),
                        "wilcoxon_p": _wilcoxon_pvalue(
                            paired["accuracy_plvm"],
                            paired["accuracy_single_task"],
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _metadata_payload(
    cutoff_minutes: Sequence[int],
    plvm_runs: Iterable[SelectedRun],
    single_task_runs: Iterable[SelectedRun],
) -> dict:
    """Build a metadata payload describing the chosen runs."""
    return {
        "cutoff_minutes": [int(value) for value in cutoff_minutes],
        "label_variants": list(LABEL_VARIANT_ORDER),
        "targets": list(VALID_TARGETS),
        "fixed_plvm_latent_dim": FIXED_PLVM_LATENT_DIM,
        "fixed_single_task_name": FIXED_SINGLE_TASK_NAME,
        "plvm_runs": [
            {
                "label_variant": run.label_variant,
                "target_name": run.target_name,
                "fold_number": run.fold_number,
                "run_name": run.run_name,
                "run_root": artifact_paths.display_path(run.run_root),
                "model_type": run.model_type,
                "predicted_next_task": run.predicted_next_task,
                "val_loss": run.val_loss,
                "val_accuracy": run.val_accuracy,
                "prediction_head_type": run.prediction_head_type,
            }
            for run in plvm_runs
        ],
        "single_task_runs": [
            {
                "label_variant": run.label_variant,
                "target_name": run.target_name,
                "fold_number": run.fold_number,
                "run_name": run.run_name,
                "run_root": artifact_paths.display_path(run.run_root),
                "model_type": run.model_type,
                "predicted_next_task": run.predicted_next_task,
                "val_loss": run.val_loss,
                "val_accuracy": run.val_accuracy,
                "prediction_head_type": run.prediction_head_type,
            }
            for run in single_task_runs
        ],
    }


def _build_markdown_summary(
    accuracy_table: pd.DataFrame,
    significance_table: pd.DataFrame,
    cutoff_minutes: Sequence[int],
) -> str:
    """Render the Markdown summary."""
    lines: List[str] = [
        "# Fire Station Label Sensitivity Analysis",
        "",
        "| Formula | Target | Model family | {} |".format(
            " | ".join("{} min Acc.".format(minutes) for minutes in cutoff_minutes)
        ),
        "| --- | --- | --- | {} |".format(" | ".join(["---"] * len(cutoff_minutes))),
    ]

    for _, row in accuracy_table.iterrows():
        lines.append(
            "| {} | {} | {} | {} |".format(
                row["formula"],
                row["target_label"],
                row["model_family"],
                " | ".join(
                    _format_mean_sd_percentage(
                        row["{} min mean accuracy".format(minutes)],
                        row["{} min sd accuracy".format(minutes)],
                    )
                    for minutes in cutoff_minutes
                ),
            )
        )

    if not significance_table.empty:
        lines.extend(
            [
                "",
                "| Formula | Target | Cutoff (min) | Delta acc. (PLVM - single-task) | Wilcoxon p |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for _, row in significance_table.iterrows():
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    row["formula"],
                    row["target_label"],
                    int(row["cutoff_minutes"]),
                    _format_mean_sd_percentage(
                        row["mean_delta_accuracy"],
                        row["sd_delta_accuracy"],
                    ),
                    _format_pvalue(row["wilcoxon_p"]),
                )
            )

    return "\n".join(lines).strip() + "\n"


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the Fire Station label sensitivity analysis."""
    args = parse_args(argv)
    cutoff_minutes = _parse_cutoff_minutes(args.cutoff_minutes)

    plvm_rows = _filter_candidate_rows(
        _read_model_performance_rows(
            args.plvm_family_dir / "model_performance.csv",
            family="plvm",
        )
        + _read_model_performance_rows(
            args.sensitivity_plvm_family_dir / "model_performance.csv",
            family="plvm",
        )
    )
    single_task_rows = _filter_candidate_rows(
        _read_model_performance_rows(
            args.single_task_family_dir / "model_performance.csv",
            family="single_task",
        )
        + _read_model_performance_rows(
            args.sensitivity_single_task_family_dir / "model_performance.csv",
            family="single_task",
        )
    )

    selected_plvm_runs = _select_fixed_plvm_runs_by_fold(plvm_rows)
    selected_single_task_runs = _select_best_single_task_runs_by_fold(single_task_rows)
    if not selected_plvm_runs:
        raise ValueError("No Fire Station PLVM runs found for the requested targets.")
    if not selected_single_task_runs:
        raise ValueError("No Fire Station single-task runs found for the requested targets.")

    plvm_summary_frames = [
        _summarize_run_at_cutoffs(run, cutoff_minutes) for run in selected_plvm_runs.values()
    ]
    single_task_summary_frames = [
        _summarize_run_at_cutoffs(run, cutoff_minutes) for run in selected_single_task_runs.values()
    ]
    plvm_summary = pd.concat(plvm_summary_frames, ignore_index=True).sort_values(
        ["label_variant", "target_name", "fold_number", "cutoff_minutes"]
    )
    single_task_summary = pd.concat(
        single_task_summary_frames,
        ignore_index=True,
    ).sort_values(
        [
            "label_variant",
            "target_name",
            "fold_number",
            "predicted_next_task",
            "cutoff_minutes",
        ]
    )
    fixed_single_task_summary = _select_fixed_single_task_rows(single_task_summary).sort_values(
        ["label_variant", "target_name", "fold_number", "cutoff_minutes"]
    )

    accuracy_table = _build_accuracy_table_rows(
        plvm_summary,
        fixed_single_task_summary,
        cutoff_minutes,
    ).sort_values(["label_variant", "target_name", "model_family"])
    significance_table = _build_significance_rows(
        plvm_summary,
        fixed_single_task_summary,
        cutoff_minutes,
    ).sort_values(["label_variant", "target_name", "cutoff_minutes"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plvm_csv = args.output_dir / "firestation_label_sensitivity_plvm_fold_metrics.csv"
    single_csv = args.output_dir / "firestation_label_sensitivity_single_task_fold_metrics.csv"
    best_single_csv = (
        args.output_dir / "firestation_label_sensitivity_best_single_task_fold_metrics.csv"
    )
    accuracy_csv = args.output_dir / "firestation_label_sensitivity_accuracy_table.csv"
    significance_csv = args.output_dir / "firestation_label_sensitivity_significance_table.csv"
    metadata_json = args.output_dir / "firestation_label_sensitivity_metadata.json"
    markdown_summary = args.output_dir / "firestation_label_sensitivity_summary.md"

    plvm_summary.to_csv(plvm_csv, index=False)
    single_task_summary.to_csv(single_csv, index=False)
    fixed_single_task_summary.to_csv(best_single_csv, index=False)
    accuracy_table.to_csv(accuracy_csv, index=False)
    significance_table.to_csv(significance_csv, index=False)
    metadata_json.write_text(
        json.dumps(
            _metadata_payload(
                cutoff_minutes,
                selected_plvm_runs.values(),
                selected_single_task_runs.values(),
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_summary.write_text(
        _build_markdown_summary(
            accuracy_table,
            significance_table,
            cutoff_minutes,
        ),
        encoding="utf-8",
    )

    print("Wrote Fire Station sensitivity analysis artifacts to {}".format(args.output_dir))


if __name__ == "__main__":
    main()
