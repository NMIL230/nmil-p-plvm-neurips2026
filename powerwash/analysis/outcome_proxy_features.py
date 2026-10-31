#!/usr/bin/env python3
"""
Outcome-feature helpers for the observed-task outcome table.

Compares simple outcome metrics on a source task (level completion, subtasks
completed, subtasks/min) at a fixed cutoff between players labelled Zone Planner
and Zone Hopper in one Fire Station fold and split. Features use the cutoff
definition of ``train_rf_zone_planning_baseline.py`` on each player's first
attempt at the source task (the attempt the models observe).

Used by ``paper/tables/table01_outcome_proxy.py``.
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plvm import artifact_paths  # noqa: E402
from powerwash.train.train_rf_zone_planning_baseline import (  # noqa: E402
    ATTEMPT_POLICY_FIRST,
    _features_at_cutoff,
    _load_selected_attempt_events,
    load_split_manifest,
)

DEFAULT_CUTOFF_SECONDS = 20.0 * 60.0
DEFAULT_LABELS_ROOT = artifact_paths.FIRESTATION_LABELS_DIR
DEFAULT_INPUT_DIR = artifact_paths.PLAYER_SEQUENCES_DIR
LABEL_COLORS = {
    "Zone Planner": "#377eb8",
    "Mixed": "#4daf4a",
    "Zone Hopper": "#e41a1c",
    "": "#cccccc",
}
SUMMARY_TABLE_METRIC_SPECS = (
    {
        "metric_name": "level_completion",
        "row_label_template": "{task_label} completion (%)",
        "scale": 100.0,
        "value_decimals": 1,
    },
    {
        "metric_name": "subtask_count",
        "row_label_template": "{task_label} subtasks completed",
        "scale": 1.0,
        "value_decimals": 1,
    },
    {
        "metric_name": "subtasks_per_minute",
        "row_label_template": "{task_label} subtasks/min",
        "scale": 1.0,
        "value_decimals": 2,
    },
)
LABEL_ORDER = ["Zone Planner", "Zone Hopper"]
# Number of distinct subtasks in each level. An attempt that logs more
# subtask completions than this by the cutoff has duplicated completion events,
# so filter_invalid_subtask_counts drops it.
MAX_SUBTASK_COUNT_BY_TASK = {
    "back_garden": 110.0,
    "bungalow": 53.0,
}


def fold_dir_for_number(fold_number: int) -> Path:
    """Return the canonical Fire Station fold directory for one fold number."""
    return DEFAULT_LABELS_ROOT / "fold_{:02d}".format(int(fold_number))


def resolve_default_labels_csv(fold_number: int) -> Path:
    """Return the default 2-class clean labels CSV for one Fire Station fold."""
    return fold_dir_for_number(fold_number) / "firestation_zone_labels_2_class_clean.csv"


def resolve_default_split_manifest(fold_number: int) -> Path:
    """Return the default player split manifest for one Fire Station fold."""
    return fold_dir_for_number(fold_number) / "player_split_manifest.csv"


def resolve_split_player_ids(
    player_split_manifest: Optional[Path],
    split_name: str,
) -> Tuple[Optional[set], str]:
    """Resolve the selected split's player IDs and a human-readable split label."""
    normalized_split = str(split_name).strip().lower()
    if normalized_split == "all":
        return None, "all"
    if player_split_manifest is None:
        raise ValueError("A split manifest is required unless --split all is used.")

    train_ids, val_ids, fold_name = load_split_manifest(player_split_manifest)
    if normalized_split == "train":
        return set(train_ids), "{} train".format(fold_name)
    if normalized_split == "val":
        return set(val_ids), "{} val".format(fold_name)
    raise ValueError("Unsupported split='{}'.".format(split_name))


def load_zone_labels(
    labels_csv: Path,
    selected_player_ids: Optional[set] = None,
) -> pd.DataFrame:
    """Load planner/hopper labels, optionally filtered to a selected split."""
    labels_df = pd.read_csv(labels_csv).copy()
    labels_df["player_id"] = labels_df["player_id"].astype(str)
    if "excluded" in labels_df.columns:
        labels_df = labels_df[labels_df["excluded"] == False].copy()  # noqa: E712
    labels_df = labels_df[labels_df["label"].isin(LABEL_ORDER)].copy()
    if selected_player_ids is not None:
        labels_df = labels_df[labels_df["player_id"].isin(selected_player_ids)].copy()
    return labels_df


def build_feature_frame(
    source_task: str,
    task_dir: Path,
    labels_df: pd.DataFrame,
    cutoff_seconds: float,
) -> pd.DataFrame:
    """Build per-player outcome features for one source task at one cutoff."""
    player_ids = set(labels_df["player_id"].astype(str))
    attempt_events = _load_selected_attempt_events(
        task_dir, source_task, player_ids, attempt_policy=ATTEMPT_POLICY_FIRST
    )
    label_lookup = labels_df.set_index("player_id").to_dict(orient="index")

    rows = []
    for player_id, events in attempt_events.items():
        label_row = label_lookup.get(str(player_id))
        if not label_row:
            continue
        feature_values = _features_at_cutoff(events, float(cutoff_seconds))
        rows.append(
            {
                "player_id": str(player_id),
                "label": str(label_row["label"]),
                "subtask_count": float(feature_values[0]),
                "level_completion": float(feature_values[1]),
            }
        )

    return pd.DataFrame(rows)


def task_display_label(task_name: str) -> str:
    """Convert an internal task name into a title-cased display label."""
    return str(task_name or "").replace("_", " ").title()


def pooled_standardized_mean_difference(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
) -> float:
    """Compute a pooled-SD standardized mean difference."""
    values_a = np.asarray(sample_a, dtype=np.float64)
    values_b = np.asarray(sample_b, dtype=np.float64)
    if len(values_a) < 2 or len(values_b) < 2:
        return float("nan")
    pooled_var = (
        ((len(values_a) - 1) * np.var(values_a, ddof=1))
        + ((len(values_b) - 1) * np.var(values_b, ddof=1))
    ) / float(len(values_a) + len(values_b) - 2)
    pooled_sd = math.sqrt(pooled_var) if pooled_var > 0.0 else float("nan")
    if not np.isfinite(pooled_sd) or pooled_sd == 0.0:
        return float("nan")
    return float((np.mean(values_a) - np.mean(values_b)) / pooled_sd)


def format_mean_plus_minus(mean_value: float, std_value: float, decimals: int) -> str:
    """Format one mean ± std string with fixed decimal precision."""
    return "{mean:.{d}f} ± {std:.{d}f}".format(
        mean=float(mean_value),
        std=float(std_value),
        d=int(decimals),
    )


def outcome_metric_column_name(cutoff_seconds: float) -> str:
    """Return the first column name of the outcome table, e.g. ``20-min observed-task outcome metric``."""
    cutoff_minutes = float(cutoff_seconds) / 60.0
    if abs(cutoff_minutes - round(cutoff_minutes)) < 1e-6:
        return "{}-min observed-task outcome metric".format(int(round(cutoff_minutes)))
    return "{:.1f}-min observed-task outcome metric".format(cutoff_minutes)


def build_summary_table_rows(
    feature_frame: pd.DataFrame,
    source_task: str,
    cutoff_seconds: float,
) -> List[Dict[str, object]]:
    """Build the outcome-table rows (one per outcome metric) for one observed source task."""
    summary_frame = feature_frame.copy()
    cutoff_minutes = float(cutoff_seconds) / 60.0
    if cutoff_minutes <= 0.0:
        raise ValueError("cutoff_seconds must be positive.")
    summary_frame["subtasks_per_minute"] = (
        summary_frame["subtask_count"].astype(np.float64) / cutoff_minutes
    )
    task_label = task_display_label(source_task)

    rows = []
    for metric_spec in SUMMARY_TABLE_METRIC_SPECS:
        metric_name = str(metric_spec["metric_name"])
        values = summary_frame[metric_name].astype(np.float64)
        planners = values[summary_frame["label"] == "Zone Planner"].to_numpy(dtype=np.float64)
        hoppers = values[summary_frame["label"] == "Zone Hopper"].to_numpy(dtype=np.float64)
        scale = float(metric_spec["scale"])
        decimals = int(metric_spec["value_decimals"])
        auc = auc_from_scores(
            (summary_frame["label"] == "Zone Hopper").astype(np.int64).to_numpy(),
            values.to_numpy(dtype=np.float64),
        )
        std_diff = pooled_standardized_mean_difference(hoppers, planners)
        rows.append(
            {
                outcome_metric_column_name(cutoff_seconds): str(
                    metric_spec["row_label_template"]
                ).format(task_label=task_label),
                "Future Planner": format_mean_plus_minus(
                    np.mean(planners) * scale,
                    np.std(planners, ddof=1) * scale,
                    decimals,
                ),
                "Future Hopper": format_mean_plus_minus(
                    np.mean(hoppers) * scale,
                    np.std(hoppers, ddof=1) * scale,
                    decimals,
                ),
                "Std. diff.": "{:.2f}".format(float(std_diff)),
                "Single-metric AUC": "{:.2f}".format(float(auc)),
            }
        )
    return rows


def filter_invalid_subtask_counts(
    feature_frame: pd.DataFrame,
    source_task: str,
) -> pd.DataFrame:
    """Drop rows whose subtask counts exceed task-specific validity caps."""
    max_subtask_count = MAX_SUBTASK_COUNT_BY_TASK.get(str(source_task))
    if max_subtask_count is None or feature_frame.empty:
        return feature_frame.copy()
    filtered = feature_frame.loc[
        feature_frame["subtask_count"].astype(np.float64) <= float(max_subtask_count)
    ].copy()
    return filtered


def auc_from_scores(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Return the probability that a hopper gets a higher score than a planner."""
    labels_arr = np.asarray(labels, dtype=np.int64)
    scores_arr = np.asarray(scores, dtype=np.float64)
    positives = scores_arr[labels_arr == 1]
    negatives = scores_arr[labels_arr == 0]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    greater = (positives[:, None] > negatives[None, :]).mean()
    equal = (positives[:, None] == negatives[None, :]).mean()
    return float(greater + (0.5 * equal))
