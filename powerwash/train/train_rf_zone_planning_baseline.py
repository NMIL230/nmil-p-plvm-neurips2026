#!/usr/bin/env python3
"""
Random-forest outcome baseline for Fire Station zone-planning classification.

At every observation cutoff (5 to 120 min) computes three outcome features per source task
(Back Garden, Bungalow or both): ``subtask_count`` (subtask-completed events so far),
``level_completion`` (last LevelProgressionAmount) and ``net_displacement`` (x/z distance
from the first to the last visible position). One forest is trained on all train-fold
player/cutoff examples, with the normalized cutoff appended as a feature. Outputs use the
train_plvm_powerwash.py run layout, so the accuracy-curve figure scripts read them directly.
Each player contributes one attempt per source task (``--attempt_policy``, default attempt 1).

Usage (one run per fold, label set and task set; relative paths use the artifact root)::

    python powerwash/train/train_rf_zone_planning_baseline.py \\
        --player_label_csv powerwash/outputs/firestation_zone_labels/fold_00/firestation_zone_labels_2_class_clean.csv \\
        --player_split_manifest powerwash/outputs/firestation_zone_labels/fold_00/player_split_manifest.csv \\
        --data_dir powerwash/data/player_sequences \\
        --task_names back_garden,bungalow \\
        --target_mode player_label_2_class_clean_zonehopping
"""

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from plvm import artifact_paths
from powerwash.data.powerwash_subtask_cleaning import normalize_subtask_label
from powerwash.data.derive_features_for_transformer import POSITION_PATTERN
from powerwash.train.powerwash_hierarchical_log_utils import (
    default_powerwash_hierarchical_logs_root,
    resolve_powerwash_hierarchical_log_base,
    resolve_powerwash_hierarchical_performance_csv_path,
)
from powerwash.train.powerwash_plvm_data_utils import (
    BOTH_PARTIAL_CUTOFF_SECONDS,
    PLAYER_LABEL_2_CLASS_NAMES,
    PLAYER_LABEL_3_CLASS_NAMES,
)
from plvm.model_training_utils import (
    generate_model_name,
    update_hierarchical_model_performance_csv,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_TYPE = "rf_baseline"
PREDICTION_HEAD_TYPE = "rf"
DEFAULT_LOG_BASE = default_powerwash_hierarchical_logs_root()
DEFAULT_PERF_CSV = resolve_powerwash_hierarchical_performance_csv_path(
    DEFAULT_LOG_BASE / "model_performance.csv",
    MODEL_TYPE,
)
DEFAULT_N_ESTIMATORS = 200
DEFAULT_SEQ_LENGTH = 120

# Source-task attempt used per player (``--attempt_policy``). "first" (attempt 1 only)
# matches the default of train_plvm_powerwash.py, so both use the same session per player.
ATTEMPT_POLICY_FIRST = "first"
ATTEMPT_POLICY_BEST = "best"
ATTEMPT_POLICIES = (ATTEMPT_POLICY_FIRST, ATTEMPT_POLICY_BEST)

_TARGET_MODE_CLASS_NAMES = {
    "player_label_2_class": PLAYER_LABEL_2_CLASS_NAMES,
    "player_label_2_class_clean_zonehopping": PLAYER_LABEL_2_CLASS_NAMES,
    "player_label_2_class_zonehopping": PLAYER_LABEL_2_CLASS_NAMES,
    "player_label_2_class_clean_entropy": PLAYER_LABEL_2_CLASS_NAMES,
    "player_label_2_class_entropy": PLAYER_LABEL_2_CLASS_NAMES,
    "player_label_3_class": PLAYER_LABEL_3_CLASS_NAMES,
    "player_label_3_class_zonehopping": PLAYER_LABEL_3_CLASS_NAMES,
    "player_label_3_class_entropy": PLAYER_LABEL_3_CLASS_NAMES,
    "player_label_2_class_clean": PLAYER_LABEL_2_CLASS_NAMES,
}
RF_FEATURE_NAMES = (
    "subtask_count",
    "level_completion",
    "net_displacement",
)


class AttemptObservation(NamedTuple):
    """One elapsed-time observation from a player's selected attempt."""

    elapsed_seconds: float
    subtask_name: str
    completion: Optional[float]
    position_xz: Optional[Tuple[float, float]]


def _coerce_attempt_observation(event) -> AttemptObservation:
    """Return ``event`` as an :class:`AttemptObservation`.

    Accepts an ``AttemptObservation`` or a dict with the same field names.
    """
    if isinstance(event, AttemptObservation):
        return event
    if isinstance(event, dict):
        return AttemptObservation(
            elapsed_seconds=float(event.get("elapsed_seconds", 0.0)),
            subtask_name=str(event.get("subtask_name", "") or ""),
            completion=event.get("completion"),
            position_xz=event.get("position_xz"),
        )
    raise ValueError("Unsupported event record shape: {!r}".format(event))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    """Parse the command-line arguments of the RF baseline."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--player_label_csv",
        required=True,
        help="Fold-specific zone-planning labels CSV (player_id, label columns).",
    )
    p.add_argument(
        "--player_split_manifest",
        required=True,
        help="Fold-specific player_split_manifest.csv (player_id, split, fold).",
    )
    p.add_argument(
        "--data_dir",
        required=True,
        help="Root of player_sequences directory (contains one subdir per task).",
    )
    p.add_argument(
        "--task_names",
        required=True,
        help="Comma-separated source task(s) whose features are used, "
        "e.g. 'back_garden' or 'back_garden,bungalow'.",
    )
    p.add_argument(
        "--target_mode",
        required=True,
        choices=list(_TARGET_MODE_CLASS_NAMES),
        help=(
            "Label schema of --player_label_csv: 2-class (Zone Planner / Zone Hopper) "
            "or 3-class (adds Mixed). The *_zonehopping and *_entropy suffixes name "
            "the labelling method and do not change the classes."
        ),
    )
    p.add_argument(
        "--player_id_col",
        default="player_id",
        help="Player-ID column in --player_label_csv.",
    )
    p.add_argument(
        "--player_label_col",
        default="label",
        help="Label column in --player_label_csv.",
    )
    p.add_argument(
        "--log_dir_base",
        type=Path,
        default=DEFAULT_LOG_BASE,
        help=(
            "Base directory for the run subdirectory. The shared PowerWash "
            "training_logs/debug_logs root maps to its rf_baselines subfolder."
        ),
    )
    p.add_argument(
        "--model_performance_csv",
        type=Path,
        default=DEFAULT_PERF_CSV,
        help="Path to the RF baseline model_performance.csv file.",
    )
    p.add_argument(
        "--n_estimators",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
        help="Number of trees in the random forest.",
    )
    p.add_argument(
        "--seq_length",
        type=int,
        default=DEFAULT_SEQ_LENGTH,
        help=(
            "Value written to the seq_length column of model_performance.csv, "
            "which the accuracy-curve plotters use to select runs. Not used by the RF."
        ),
    )
    p.add_argument(
        "--attempt_policy",
        choices=list(ATTEMPT_POLICIES),
        default=ATTEMPT_POLICY_FIRST,
        help="Which attempt of each source task a player contributes: 'first' (default) uses "
        "attempt 1 only (players without it have no features for that task); 'best' uses "
        "the highest-completion attempt (lowest attempt number on ties).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed of the random forest.")
    p.add_argument(
        "--run_mode",
        choices=["run", "dry_run"],
        default="run",
        help="'dry_run' prints the config without writing any files.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _resolve_path(path_str: str) -> Path:
    """Resolve a path argument, taking a relative one against the artifact root."""
    p = Path(path_str)
    if not p.is_absolute():
        p = artifact_paths.ARTIFACT_ROOT / p
    return p


def load_player_labels(
    label_csv: Path,
    player_id_col: str,
    label_col: str,
    class_names: list,
) -> dict:
    """Return {player_id: label_str} keeping only rows with a recognized label."""
    df = pd.read_csv(label_csv)
    df[player_id_col] = df[player_id_col].astype(str)
    df = df[df[label_col].isin(class_names)].copy()
    return dict(zip(df[player_id_col], df[label_col]))


def load_split_manifest(manifest_csv: Path) -> tuple:
    """Return (train_ids: set[str], val_ids: set[str], fold_name: str)."""
    df = pd.read_csv(manifest_csv)
    df["player_id"] = df["player_id"].astype(str)
    train_ids = set(df.loc[df["split"] == "train", "player_id"])
    val_ids = set(df.loc[df["split"] == "val", "player_id"])
    fold_name = str(df["fold"].iloc[0]) if "fold" in df.columns and not df.empty else "fold_unknown"
    return train_ids, val_ids, fold_name


def _parse_xz_position(raw_position: str) -> Optional[Tuple[float, float]]:
    """Parse a ``CurrentPosition`` string into an ``(x, z)`` tuple."""
    cleaned = (raw_position or "").strip()
    if not cleaned:
        return None
    match = POSITION_PATTERN.match(cleaned)
    if match is None:
        return None
    return float(match.group(1)), float(match.group(3))


def _load_selected_attempt_events(
    task_dir: Path,
    task_name: str,
    player_ids: set,
    attempt_policy: str = ATTEMPT_POLICY_FIRST,
) -> dict:
    """Return ``{player_id: [AttemptObservation, ...]}`` from one attempt per player, sorted by time.

    Uses attempt 1 when ``attempt_policy="first"``, otherwise the highest-completion attempt
    (lowest attempt number on ties). Keeps every row with a completion value, a completed
    subtask or a parseable x/z position.
    """
    task_token = "_{}_".format(task_name)
    csv_paths = [p for p in sorted(task_dir.glob("*.csv")) if task_token in p.stem]

    by_player: dict = defaultdict(list)
    for p in csv_paths:
        pid = p.stem.split(task_token)[0]
        if player_ids and pid not in player_ids:
            continue
        if attempt_policy == ATTEMPT_POLICY_FIRST and not re.search(r"_attempt0*1$", p.stem):
            continue
        by_player[pid].append(p)

    result = {}
    for pid, paths in by_player.items():
        best_completion = -1.0
        best_attempt_num = 9999
        best_events = None

        for path in paths:
            m = re.search(r"attempt(\d+)$", path.stem)
            attempt_num = int(m.group(1)) if m else 999

            events = []
            with path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    event_type = (row.get("event_type") or "").strip()
                    raw_completion = row.get("LevelProgressionAmount")
                    has_completion = str(raw_completion or "").strip() != ""
                    raw_subtask = row.get("LastSubtaskCompleted", "") or ""
                    position_xz = _parse_xz_position(row.get("CurrentPosition", ""))
                    subtask = (
                        normalize_subtask_label(raw_subtask)
                        if event_type == "subtask_completed"
                        else ""
                    )
                    if not subtask and not has_completion and position_xz is None:
                        continue
                    try:
                        elapsed = float(row.get("time_elapsed_since_job_start_raw") or 0)
                    except (ValueError, TypeError):
                        elapsed = 0.0
                    completion = None
                    if has_completion:
                        try:
                            completion = float(raw_completion or 0)
                        except (ValueError, TypeError):
                            completion = 0.0
                    events.append(
                        AttemptObservation(
                            elapsed_seconds=elapsed,
                            subtask_name=subtask,
                            completion=completion,
                            position_xz=position_xz,
                        )
                    )

            if not events:
                continue
            completion_values = [
                float(obs.completion) for obs in events if obs.completion is not None
            ]
            attempt_completion = max(completion_values) if completion_values else 0.0
            if attempt_completion > best_completion or (
                attempt_completion == best_completion and attempt_num < best_attempt_num
            ):
                best_completion = attempt_completion
                best_attempt_num = attempt_num
                best_events = sorted(events, key=lambda obs: obs.elapsed_seconds)

        if best_events:
            result[pid] = best_events

    return result


def load_all_task_events(
    data_dir: Path,
    task_names: list,
    player_ids: set,
    attempt_policy: str = ATTEMPT_POLICY_FIRST,
) -> dict:
    """
    Load events for all requested tasks (one attempt per player, see ``attempt_policy``).

    Returns
    -------
    dict: task_name -> {player_id -> [AttemptObservation, ...]}
    """
    out = {}
    for task_name in task_names:
        task_dir = _resolve_task_dir(data_dir, task_name)
        out[task_name] = _load_selected_attempt_events(
            task_dir, task_name, player_ids, attempt_policy=attempt_policy
        )
        logging.info("[%s] Loaded events for %d players.", task_name, len(out[task_name]))
    return out


def _resolve_task_dir(data_dir: Path, task_name: str) -> Path:
    task_subdir = data_dir / task_name
    if task_subdir.is_dir():
        return task_subdir
    return data_dir


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


def _features_at_cutoff(
    events: list,
    cutoff_s: float,
) -> np.ndarray:
    """
    Compute the 3 raw cutoff features for one player at one cutoff.

    Features
    --------
    subtask_count    : number of subtask-completed events up to cutoff_s
    level_completion : last observed LevelProgressionAmount up to cutoff_s
    net_displacement : x/z displacement between first and last visible positions
    """
    visible = []
    for raw_observation in events:
        observation = _coerce_attempt_observation(raw_observation)
        if observation.elapsed_seconds <= cutoff_s:
            visible.append(observation)
    subtask_count = sum(1 for obs in visible if obs.subtask_name)
    level_completion = 0.0
    positions = []
    for obs in visible:
        if obs.completion is not None:
            level_completion = float(obs.completion)
        if obs.position_xz is not None:
            positions.append(obs.position_xz)
    net_displacement = 0.0
    if len(positions) >= 2:
        net_displacement = float(
            np.linalg.norm(
                np.asarray(positions[-1], dtype=np.float64)
                - np.asarray(positions[0], dtype=np.float64)
            )
        )
    return np.array(
        [
            float(subtask_count),
            float(level_completion),
            float(net_displacement),
        ],
        dtype=np.float64,
    )


def build_feature_matrix(
    player_ids: list,
    task_events: dict,
    task_names: list,
    cutoff_s: float,
) -> tuple:
    """
    Build (feature_matrix, valid_mask) for a list of players at one cutoff.

    Returns
    -------
    features   : ndarray shape (n_players, n_tasks * len(RF_FEATURE_NAMES))
    valid_mask : boolean array; False if the player is missing any task's data
    """
    n_players = len(player_ids)
    n_tasks = len(task_names)
    features = np.zeros((n_players, n_tasks * len(RF_FEATURE_NAMES)), dtype=np.float64)
    valid_mask = np.ones(n_players, dtype=bool)

    for t_idx, task_name in enumerate(task_names):
        events_by_pid = task_events[task_name]
        col_start = t_idx * len(RF_FEATURE_NAMES)

        for p_idx, pid in enumerate(player_ids):
            if pid not in events_by_pid:
                valid_mask[p_idx] = False
                continue
            features[p_idx, col_start : col_start + len(RF_FEATURE_NAMES)] = _features_at_cutoff(
                events_by_pid[pid], cutoff_s
            )

    return features, valid_mask


def _append_cutoff_feature(features: np.ndarray, cutoff_s: float) -> np.ndarray:
    """Append the normalized observation cutoff as one additional feature."""
    if features.ndim != 2:
        raise ValueError("features must be a 2D array.")
    max_cutoff_s = float(max(BOTH_PARTIAL_CUTOFF_SECONDS)) if BOTH_PARTIAL_CUTOFF_SECONDS else 1.0
    cutoff_fraction = float(cutoff_s) / max(max_cutoff_s, 1.0)
    cutoff_column = np.full((features.shape[0], 1), cutoff_fraction, dtype=np.float64)
    return np.hstack([features, cutoff_column])


def build_stacked_cutoff_dataset(
    player_ids: list,
    labels: np.ndarray,
    task_events: dict,
    task_names: list,
    cutoff_values: tuple,
) -> tuple:
    """Return stacked examples across all cutoffs for one split.

    Each valid player contributes one example per cutoff. The returned cutoff
    vector holds each example's cutoff in seconds so metrics can be grouped by
    cutoff in ``metrics_by_cutoff.csv``.
    """
    feature_blocks = []
    target_blocks = []
    cutoff_blocks = []
    # Player ids are carried through so per-player predictions can be written
    # out alongside the aggregated cutoff metrics.
    player_id_blocks = []

    for cutoff_s in cutoff_values:
        features, valid_mask = build_feature_matrix(
            player_ids,
            task_events,
            task_names,
            float(cutoff_s),
        )
        if not np.any(valid_mask):
            continue
        valid_features = _append_cutoff_feature(features[valid_mask], float(cutoff_s))
        valid_labels = labels[valid_mask]
        feature_blocks.append(valid_features)
        target_blocks.append(valid_labels)
        cutoff_blocks.append(np.full(valid_labels.shape[0], float(cutoff_s), dtype=np.float64))
        player_id_blocks.append(np.asarray(player_ids, dtype=object)[valid_mask])

    if not feature_blocks:
        n_base_features = len(task_names) * len(RF_FEATURE_NAMES) + 1
        return (
            np.zeros((0, n_base_features), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=object),
        )

    return (
        np.vstack(feature_blocks),
        np.concatenate(target_blocks),
        np.concatenate(cutoff_blocks),
        np.concatenate(player_id_blocks),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: list,
) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, log_loss

    n_classes = len(class_names)
    labels = list(range(n_classes))
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    try:
        cross_entropy = float(log_loss(y_true, y_proba, labels=labels))
    except ValueError:
        cross_entropy = float("nan")
    target_counts = [int((y_true == cls).sum()) for cls in labels]
    pred_counts = [int((y_pred == cls).sum()) for cls in labels]
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "cross_entropy": cross_entropy,
        "support": int(len(y_true)),
        "target_class_counts_json": json.dumps(target_counts),
        "predicted_class_counts_json": json.dumps(pred_counts),
        "class_names_json": json.dumps(class_names),
    }


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

_CUTOFF_CSV_FIELDS = [
    "cutoff_kind",
    "cutoff_value",
    "support",
    "accuracy",
    "macro_f1",
    "cross_entropy",
    "target_class_counts_json",
    "predicted_class_counts_json",
    "class_names_json",
]


def save_cutoff_metrics_csv(
    log_dir: Path,
    split_name: str,
    cutoff_rows: list,
) -> None:
    """Write ``plots/<split>/cutoff_metrics/metrics_by_cutoff.csv`` for one split."""
    cutoff_dir = log_dir / "plots" / split_name / "cutoff_metrics"
    cutoff_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cutoff_dir / "metrics_by_cutoff.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CUTOFF_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(cutoff_rows)
    logging.info("Saved %s cutoff metrics → %s", split_name, csv_path)


def build_cutoff_metric_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    cutoff_values: np.ndarray,
    class_names: list,
) -> list:
    """Group predictions by cutoff and return metrics rows in CSV-ready format."""
    rows = []
    for cutoff_s in sorted(float(v) for v in np.unique(cutoff_values)):
        cutoff_mask = np.isclose(cutoff_values, cutoff_s)
        cutoff_metrics = _compute_metrics(
            y_true[cutoff_mask],
            y_pred[cutoff_mask],
            y_proba[cutoff_mask],
            class_names,
        )
        rows.append(
            {
                "cutoff_kind": "observation_cutoff_seconds",
                "cutoff_value": float(cutoff_s),
                **cutoff_metrics,
            }
        )
    return rows


VAL_PREDICTIONS_FILENAME = "val_predictions.csv"
# Same columns, in the same order, as ``_save_val_predictions`` in
# train_plvm_powerwash.py, so one loader reads PLVM, single-task and RF runs.
VAL_PREDICTIONS_FIELDS = [
    "target_mode",
    "target_name",
    "target_type",
    "player_id",
    "task_name",
    "attempt_id",
    "source_type",
    "source_row_index",
    "observation_cutoff_seconds",
    "target",
    "target_label",
    "prediction",
    "prediction_class_id",
    "prediction_label",
    "prediction_probabilities_json",
    "sq_error",
    "model_type",
    "prediction_head_type",
    "model_name",
]


def build_val_predictions_frame(
    target_mode: str,
    task_names: list,
    player_ids: np.ndarray,
    cutoff_values: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: list,
    model_name: str,
) -> pd.DataFrame:
    """Return one row per validation player and cutoff in the PLVM ``val_predictions.csv`` schema.

    ``prediction`` is the probability of the predicted class, as in the PLVM
    trainer. Each row summarizes one selected attempt per source task (see
    ``--attempt_policy``) rather than one source row, so ``task_name`` is the
    source task (or ``both``) and ``attempt_id`` and ``source_row_index`` are -1.
    """
    y_proba = np.asarray(y_proba, dtype=np.float64)
    return pd.DataFrame(
        {
            "target_mode": target_mode,
            "target_name": target_mode,
            "target_type": "classification",
            "player_id": np.asarray(player_ids, dtype=object),
            "task_name": _predicted_next_task(task_names),
            "attempt_id": -1,
            "source_type": target_mode,
            "source_row_index": -1,
            "observation_cutoff_seconds": np.asarray(cutoff_values, dtype=np.float64),
            "target": np.asarray(y_true, dtype=np.int64),
            "target_label": [class_names[int(c)] for c in y_true],
            "prediction": y_proba.max(axis=1) if len(y_proba) else np.zeros(0),
            "prediction_class_id": np.asarray(y_pred, dtype=np.int64),
            "prediction_label": [class_names[int(c)] for c in y_pred],
            "prediction_probabilities_json": [
                json.dumps([float(x) for x in row]) for row in y_proba
            ],
            "sq_error": float("nan"),
            "model_type": MODEL_TYPE,
            "prediction_head_type": PREDICTION_HEAD_TYPE,
            "model_name": model_name,
        },
        columns=VAL_PREDICTIONS_FIELDS,
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def _predicted_next_task(task_names: list) -> str:
    if len(task_names) == 1:
        return task_names[0]
    return "both"


def _parse_task_names(task_names_arg: str) -> list:
    """Return a de-duplicated task list with canonical ordering for two-task runs."""
    ordered = []
    seen = set()
    for raw_task_name in str(task_names_arg or "").split(","):
        task_name = raw_task_name.strip()
        if not task_name or task_name in seen:
            continue
        ordered.append(task_name)
        seen.add(task_name)
    if not ordered:
        raise ValueError("--task_names must contain at least one task name.")
    if len(ordered) > 1:
        ordered = sorted(ordered)
    return ordered


def run(args):
    """Train one RF baseline run and write its metrics, predictions and registry row."""
    from sklearn.ensemble import RandomForestClassifier

    # ---- Resolve paths ----
    label_csv = _resolve_path(args.player_label_csv)
    manifest_csv = _resolve_path(args.player_split_manifest)
    data_dir = _resolve_path(args.data_dir)
    task_names = _parse_task_names(args.task_names)

    class_names = _TARGET_MODE_CLASS_NAMES[args.target_mode]
    label_to_idx = {lbl: i for i, lbl in enumerate(class_names)}

    # ---- Load labels and split ----
    all_labels = load_player_labels(
        label_csv, args.player_id_col, args.player_label_col, class_names
    )
    train_ids, val_ids, fold_name = load_split_manifest(manifest_csv)

    train_labeled = {pid: lbl for pid, lbl in all_labels.items() if pid in train_ids}
    val_labeled = {pid: lbl for pid, lbl in all_labels.items() if pid in val_ids}

    logging.info(
        "Fold %s | target_mode=%s | tasks=%s | train=%d labeled, val=%d labeled",
        fold_name,
        args.target_mode,
        task_names,
        len(train_labeled),
        len(val_labeled),
    )

    if not train_labeled or not val_labeled:
        raise ValueError(
            "After filtering to labeled players, train={} val={}. "
            "Check --player_label_csv and --player_split_manifest.".format(
                len(train_labeled), len(val_labeled)
            )
        )

    # ---- Load source-task events ----
    all_player_ids = set(train_labeled) | set(val_labeled)
    task_events = load_all_task_events(
        data_dir, task_names, all_player_ids, attempt_policy=args.attempt_policy
    )

    train_pids = sorted(train_labeled)
    val_pids = sorted(val_labeled)
    train_y = np.array([label_to_idx[train_labeled[p]] for p in train_pids], dtype=np.int64)
    val_y = np.array([label_to_idx[val_labeled[p]] for p in val_pids], dtype=np.int64)

    # ---- Build one stacked train/val dataset across all cutoffs ----
    train_X, train_y_all, train_cutoff_values, _train_pid_rows = build_stacked_cutoff_dataset(
        train_pids,
        train_y,
        task_events,
        task_names,
        BOTH_PARTIAL_CUTOFF_SECONDS,
    )
    val_X, val_y_all, val_cutoff_values, val_pid_rows = build_stacked_cutoff_dataset(
        val_pids,
        val_y,
        task_events,
        task_names,
        BOTH_PARTIAL_CUTOFF_SECONDS,
    )

    if len(train_X) == 0 or len(np.unique(train_y_all)) < 2:
        raise ValueError(
            "Insufficient stacked train data after cutoff expansion. "
            "Check --player_label_csv and source task data."
        )
    if len(val_X) == 0:
        raise ValueError(
            "No stacked val data after cutoff expansion. "
            "Check --player_split_manifest and source task data."
        )

    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        n_jobs=-1,
    )
    rf.fit(train_X, train_y_all)

    train_pred = rf.predict(train_X)
    train_proba = rf.predict_proba(train_X)
    val_pred = rf.predict(val_X)
    val_proba = rf.predict_proba(val_X)

    train_metrics = _compute_metrics(train_y_all, train_pred, train_proba, class_names)
    val_metrics = _compute_metrics(val_y_all, val_pred, val_proba, class_names)
    train_cutoff_rows = build_cutoff_metric_rows(
        train_y_all,
        train_pred,
        train_proba,
        train_cutoff_values,
        class_names,
    )
    val_cutoff_rows = build_cutoff_metric_rows(
        val_y_all,
        val_pred,
        val_proba,
        val_cutoff_values,
        class_names,
    )

    for train_row, val_row in zip(train_cutoff_rows, val_cutoff_rows):
        cutoff_s = float(train_row["cutoff_value"])
        logging.info(
            "Cutoff %4ds (%3.0f min) | train_acc=%.3f  val_acc=%.3f  val_support=%d",
            int(cutoff_s),
            cutoff_s / 60.0,
            float(train_row["accuracy"]),
            float(val_row["accuracy"]),
            int(val_row["support"]),
        )

    best_val_cutoff_acc = max(float(row["accuracy"]) for row in val_cutoff_rows)

    # ---- Write cutoff metrics CSVs ----
    model_name = generate_model_name()
    log_dir_base = resolve_powerwash_hierarchical_log_base(
        args.log_dir_base,
        MODEL_TYPE,
        run_mode=args.run_mode,
    )
    log_dir = Path(log_dir_base) / model_name

    if args.run_mode == "dry_run":
        logging.info("[dry_run] Would write to %s", log_dir)
        return

    log_dir.mkdir(parents=True, exist_ok=True)

    save_cutoff_metrics_csv(log_dir, "train", train_cutoff_rows)
    save_cutoff_metrics_csv(log_dir, "val", val_cutoff_rows)

    # Per-player validation predictions. The aggregated cutoff metrics above
    # cannot support paired per-player tests (McNemar, per-player bootstrap) or
    # ROC AUC, so the individual predictions and class probabilities are saved
    # in the same schema the process-model trainers use.
    build_val_predictions_frame(
        target_mode=args.target_mode,
        task_names=task_names,
        player_ids=val_pid_rows,
        cutoff_values=val_cutoff_values,
        y_true=val_y_all,
        y_pred=val_pred,
        y_proba=val_proba,
        class_names=class_names,
        model_name=model_name,
    ).to_csv(log_dir / VAL_PREDICTIONS_FILENAME, index=False)
    logging.info("Saved per-player val predictions → %s", log_dir / VAL_PREDICTIONS_FILENAME)

    # ---- Majority-class naive baseline on stacked train/val examples ----
    train_counts = np.bincount(train_y_all, minlength=len(class_names))
    val_counts = np.bincount(val_y_all, minlength=len(class_names))
    naive_train_accuracy = (
        float(train_counts.max()) / float(len(train_y_all)) if len(train_y_all) > 0 else 0.0
    )
    naive_val_accuracy = (
        float(val_counts.max()) / float(len(val_y_all)) if len(val_y_all) > 0 else 0.0
    )

    # ---- Append to model_performance.csv ----
    fold_match = re.search(r"fold_(\d+)", fold_name)
    fold_number = int(fold_match.group(1)) if fold_match else -1

    args_dict = vars(args).copy()
    args_dict.update(
        {
            "model_name": model_name,
            "model_type": MODEL_TYPE,
            "prediction_head_type": PREDICTION_HEAD_TYPE,
            "predicted_next_task": _predicted_next_task(task_names),
            "parameters_to_predict": args.target_mode,
            "target_mode": args.target_mode,
            "fold_number": fold_number,
            "player_split_manifest": str(manifest_csv),
            "player_label_csv": str(label_csv),
            "num_classes": len(class_names),
            "seq_length": int(args.seq_length),
            "latent_dim": -1,
            "loss_type": "classification",
            "tasks": task_names,
            "rf_training_mode": "single_model_all_cutoffs",
            "rf_includes_cutoff_feature": True,
            "rf_feature_names": list(RF_FEATURE_NAMES),
        }
    )

    perf_csv = resolve_powerwash_hierarchical_performance_csv_path(
        args.model_performance_csv,
        MODEL_TYPE,
        run_mode=args.run_mode,
    )
    perf_csv.parent.mkdir(parents=True, exist_ok=True)

    registry_args = argparse.Namespace(**args_dict)
    update_hierarchical_model_performance_csv(
        csv_file=str(perf_csv),
        args=registry_args,
        train_size=int(len(train_y_all)),
        val_size=int(len(val_y_all)),
        val_accuracy=float(val_metrics["accuracy"]),
        train_accuracy=float(train_metrics["accuracy"]),
        naive_train_accuracy=naive_train_accuracy,
        naive_val_accuracy=naive_val_accuracy,
        train_loss=0.0,
        val_loss=0.0,
        naive_train_loss=0.0,
        naive_val_loss=0.0,
        train_mse=0.0,
        val_mse=0.0,
    )
    logging.info("Appended to %s  (model_name=%s)", perf_csv, model_name)

    logging.info(
        "=== Done | fold=%s | task=%s | target=%s | overall_val_acc=%.4f | best_cutoff_val_acc=%.4f ===",
        fold_name,
        _predicted_next_task(task_names),
        args.target_mode,
        float(val_metrics["accuracy"]),
        best_val_cutoff_acc,
    )


def main(argv=None):
    """Command-line entry point."""
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
