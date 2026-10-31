#!/usr/bin/env python3
"""
Turn extracted attempt CSVs into feature-enriched CSVs for transformer training.

Reads ``powerwash/data/player_sequences/`` and writes one CSV per kept attempt,
keeping task subfolders, to ``powerwash/data/player_sequence_features/``. An
attempt is dropped when its subtask vocabulary does not match its task, its
clock is non-monotonic, or it exceeds ``--max_session_minutes``. Kept attempts
are reduced to deduplicated ``subtask_completed`` rows and get sequence
features plus future-progress and future-subtask targets.

Fallback: the gap-corrected clock can decrease (see
``extract_player_sequences.py``). Such a file is replaced by the same-named
file from the uncorrected set (``--fallback_input_folder``; default
``player_sequences_no_gap_correction/``, else the uncorrected copy in
``player_sequence_features/player_sequencesy/``) when that file passes every
check.
"""

import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plvm import artifact_paths  # noqa: E402

from powerwash.data.powerwash_subtask_cleaning import (  # noqa: E402
    EXPECTED_NORMALIZED_SUBTASKS_BY_TASK,
    normalize_subtask_label,
)


DEFAULT_INPUT_DIR = artifact_paths.PLAYER_SEQUENCES_DIR
DEFAULT_OUTPUT_DIR = artifact_paths.PLAYER_SEQUENCE_FEATURES_DIR
# Candidates for --fallback_input_folder (first existing one wins): the output of
# `extract_player_sequences.py --implicit_gap_threshold_seconds 0`, then the uncorrected
# copy in player_sequence_features/player_sequencesy/.
DEFAULT_FALLBACK_INPUT_DIRS = artifact_paths.DEFAULT_FALLBACK_SEQUENCE_DIRS
SCHEMA_FILENAME = "transformer_feature_schema.json"
TASK_MAP_TABLE_PATH = artifact_paths.SUBTASK_TASK_MAP_POINTS_CSV
REVISIT_CELL_SIZE = 1.0
if "fire_station" in EXPECTED_NORMALIZED_SUBTASKS_BY_TASK:
    TASK_PROCESSING_ORDER = ("fire_station",) + tuple(
        task_name
        for task_name in EXPECTED_NORMALIZED_SUBTASKS_BY_TASK
        if task_name != "fire_station"
    )
else:
    TASK_PROCESSING_ORDER = tuple(EXPECTED_NORMALIZED_SUBTASKS_BY_TASK)
METADATA_COLUMNS = [
    "pid",
    "task",
    "attempt_id",
    "Time_utc",
    "job_start_utc",
    "job_end_utc",
    "CurrentGameMode",
    "CurrentJobName",
]

REDUNDANT_RAW_COLUMNS = [
    "CurrentPosition",
    "IsIdleInGame",
    "IsInMenu",
    "LastSubtaskCompleted",
    "CampaignProgressionAmount",
    "CurrentCreditAmount",
    "CurrentSessionLength",
    "CumulativeTimeSpentInLevel",
    "time_elapsed_since_job_start_raw",
]

LEAKAGE_COLUMNS = [
    "time_remaining_proxy",
]

TRAINING_FEATURE_COLUMNS = [
    "event_type",
    "is_subtask_event",
    "is_state_event",  # constant placeholder (only subtask rows are kept); keeps the input layout
    "time_elapsed_since_job_start",
    "LevelProgressionAmount",
    "delta_time",
    "time_since_last_subtask",
    "time_since_last_state",  # constant placeholder, like is_state_event
    "delta_progress",
    "progress_rate",
    "rolling_progress_60s",
    "pos_x",
    "pos_y",
    "pos_z",
    "delta_x_10s",
    "delta_y_10s",
    "delta_z_10s",
    "displacement_10s",
    "distance_traveled_60s",
    "CurrentWasher",
    "CurrentNozzle",
    "CurrentExtension",
    "CrouchState",
    "washer_changed",
    "nozzle_changed",
    "extension_changed",
    "tool_change_count_60s",
    "crouch_changed",
    "idle_flag",
    "menu_flag",
    "subtask_count_60s",
]

TARGET_COLUMNS = [
    "progress_next_30s",
    "progress_next_60s",
    "progress_acceleration_next_60s",
    "subtask_in_next_10s",
    "subtask_count_next_30s",
    "subtask_count_next_60s",
    "subtask_count_next_120s",
    "progress_efficiency_next_30s",
    "progress_efficiency_next_60s",
    "progress_efficiency_next_120s",
    "path_length_next_30s",
    "path_length_next_60s",
    "path_length_next_120s",
    "total_angular_turn_next_30s",
    "total_angular_turn_next_60s",
    "total_angular_turn_next_120s",
    "direction_reversals_next_30s",
    "direction_reversals_next_60s",
    "direction_reversals_next_120s",
    "reversal_density_next_30s",
    "reversal_density_next_60s",
    "reversal_density_next_120s",
    "revisit_ratio_next_30s",
    "revisit_ratio_next_60s",
    "revisit_ratio_next_120s",
    "next_choice_rank_next_60s",
    "mean_choice_rank_next_60s",
    "mean_choice_rank_next_120s",
    "next_subtask_standoff_distance",
    "mean_standoff_distance_next_60s",
    "mean_standoff_distance_next_120s",
    "progress_next_10s",
    "delta_x_next_10s",
    "delta_y_next_10s",
    "delta_z_next_10s",
    "delta_x_to_next_subtask",
    "delta_y_to_next_subtask",
    "delta_z_to_next_subtask",
    "displacement_to_next_subtask",
    "xz_displacement_to_next_subtask",
    "time_to_next_subtask",
    "target_next_subtask_position_available",
    "target_10s_available",
    "target_30s_available",
    "target_60s_available",
    "target_subtask_10s_available",
]

REQUIRED_COLUMNS = [
    "pid",
    "task",
    "event_type",
    "Time_utc",
    "time_elapsed_since_job_start",
    "time_elapsed_since_job_start_raw",
    "CurrentGameMode",
    "CurrentJobName",
    "LevelProgressionAmount",
    "CurrentWasher",
    "CurrentNozzle",
    "CurrentExtension",
    "CurrentPosition",
    "CrouchState",
    "CampaignProgressionAmount",
    "CurrentCreditAmount",
    "CurrentSessionLength",
    "CumulativeTimeSpentInLevel",
    "IsIdleInGame",
    "IsInMenu",
    "LastSubtaskCompleted",
]

DERIVED_COLUMNS = [
    "is_subtask_event",
    "is_state_event",
    "delta_time",
    "time_since_last_subtask",
    "time_since_last_state",
    "delta_progress",
    "progress_rate",
    "rolling_progress_30s",
    "rolling_progress_60s",
    "pos_x",
    "pos_y",
    "pos_z",
    "delta_x",
    "delta_y",
    "delta_z",
    "distance_from_prev",
    "speed_from_prev",
    "delta_x_10s",
    "delta_y_10s",
    "delta_z_10s",
    "displacement_10s",
    "distance_traveled_30s",
    "distance_traveled_60s",
    "washer_changed",
    "nozzle_changed",
    "extension_changed",
    "tool_change_count_30s",
    "tool_change_count_60s",
    "crouch_changed",
    "idle_flag",
    "menu_flag",
    "idle_count_60s",
    "menu_count_60s",
    "subtask_count_30s",
    "subtask_count_60s",
    "rows_in_last_30s",
    "rows_in_last_60s",
    "time_remaining_proxy",
    "progress_next_30s",
    "progress_next_60s",
    "progress_acceleration_next_60s",
    "subtask_in_next_10s",
    "subtask_count_next_30s",
    "subtask_count_next_60s",
    "subtask_count_next_120s",
    "progress_efficiency_next_30s",
    "progress_efficiency_next_60s",
    "progress_efficiency_next_120s",
    "path_length_next_30s",
    "path_length_next_60s",
    "path_length_next_120s",
    "total_angular_turn_next_30s",
    "total_angular_turn_next_60s",
    "total_angular_turn_next_120s",
    "direction_reversals_next_30s",
    "direction_reversals_next_60s",
    "direction_reversals_next_120s",
    "reversal_density_next_30s",
    "reversal_density_next_60s",
    "reversal_density_next_120s",
    "revisit_ratio_next_30s",
    "revisit_ratio_next_60s",
    "revisit_ratio_next_120s",
    "next_choice_rank_next_60s",
    "mean_choice_rank_next_60s",
    "mean_choice_rank_next_120s",
    "next_subtask_standoff_distance",
    "mean_standoff_distance_next_60s",
    "mean_standoff_distance_next_120s",
    "progress_next_10s",
    "delta_x_next_10s",
    "delta_y_next_10s",
    "delta_z_next_10s",
    "delta_x_to_next_subtask",
    "delta_y_to_next_subtask",
    "delta_z_to_next_subtask",
    "displacement_to_next_subtask",
    "xz_displacement_to_next_subtask",
    "time_to_next_subtask",
    "target_next_subtask_position_available",
    "target_10s_available",
    "target_30s_available",
    "target_60s_available",
    "target_subtask_10s_available",
]

POSITION_PATTERN = re.compile(
    r"^\(\s*"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*"
    r"\)$"
)

TRUE_STRINGS = {"1", "true", "t", "yes", "y"}
FALSE_STRINGS = {"0", "false", "f", "no", "n"}

D_MIN = 1.0


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_folder",
        nargs="?",
        default=str(DEFAULT_INPUT_DIR),
        help="Folder containing extracted attempt CSVs",
    )
    parser.add_argument(
        "output_folder",
        nargs="?",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder where processed CSVs will be written",
    )
    parser.add_argument(
        "--max_session_minutes",
        type=float,
        default=120.0,
        help="Skip sessions whose total elapsed time exceeds this many minutes (default: 120).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASK_PROCESSING_ORDER),
        choices=list(TASK_PROCESSING_ORDER),
        help=(
            "Task names to regenerate. Defaults to all configured tasks, "
            "processing fire_station first."
        ),
    )
    parser.add_argument(
        "--time_axis",
        choices=("elapsed", "active"),
        default="active",
        help=(
            "Sequence clock to use for duration filtering, rolling windows, and "
            "future targets. 'elapsed' keeps raw wall-clock time since the initial "
            "job_started; 'active' uses the pause-adjusted elapsed column "
            "prepared during extraction from job_exited -> job_resumed pairs."
        ),
    )
    parser.add_argument(
        "--fallback_input_folder",
        default=None,
        help=(
            "Sequences extracted without the implicit-gap correction. An input file that "
            "fails the non-monotonic elapsed-time check is replaced by the same-named file "
            "here if that file passes every check. "
            "Default: the first existing of {}; 'none' disables the fallback.".format(
                ", ".join(
                    artifact_paths.display_path(d) for d in DEFAULT_FALLBACK_INPUT_DIRS
                )
            )
        ),
    )
    parser.add_argument(
        "--clean_output",
        action="store_true",
        help=(
            "Delete existing CSVs in the output task folders before writing, so that files "
            "from earlier runs cannot mix with this run's output. Without it, such files "
            "are only reported as stale."
        ),
    )
    args = parser.parse_args()
    args.fallback_input_folder = resolve_fallback_input_folder(args.fallback_input_folder)
    return args


def resolve_fallback_input_folder(value, candidates=DEFAULT_FALLBACK_INPUT_DIRS):
    """Return the fallback folder as a Path, or None when disabled/not found.

    ``None`` selects the first existing directory of ``candidates``; ``"none"`` or
    ``""`` disables the fallback.
    """
    if value is None:
        for candidate in candidates:
            if Path(candidate).is_dir():
                return Path(candidate)
        return None
    if str(value).strip().lower() in ("", "none"):
        return None
    return Path(value)


def ensure_required_columns(df):
    """Add any missing expected input columns so processing stays conservative."""
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df


def clean_string_series(series):
    """Normalize blank strings to missing values while preserving real strings."""
    cleaned = series.copy().astype(object)
    blank_mask = cleaned.map(lambda value: isinstance(value, str) and not value.strip())
    cleaned.loc[blank_mask] = np.nan
    return cleaned


def normalized_unique_subtasks(df):
    """Return the set of normalized subtasks completed in one attempt file."""
    subtask_rows = df.loc[df["event_type"].eq("subtask_completed"), "LastSubtaskCompleted"]
    return {
        normalized
        for normalized in (normalize_subtask_label(value) for value in subtask_rows.tolist())
        if normalized
    }


def resolve_attempt_task(df):
    """Infer the task name encoded by one attempt file."""
    if "task" not in df.columns:
        return ""
    task_values = [
        value.strip() for value in df["task"].astype(str).tolist() if value and value.strip()
    ]
    if not task_values:
        return ""
    return task_values[0]


def attempt_matches_expected_subtasks(df):
    """
    Return True when an attempt's normalized subtask vocabulary matches its task.

    Files outside the configured target tasks, empty attempts, contaminated
    attempts, and attempts with incomplete/wrong subtask vocabularies are
    rejected.
    """
    task = resolve_attempt_task(df)
    expected_subtasks = EXPECTED_NORMALIZED_SUBTASKS_BY_TASK.get(task)
    if expected_subtasks is None:
        return False
    return normalized_unique_subtasks(df) == expected_subtasks


def attempt_elapsed_is_monotonic(df):
    """
    Return True when time_elapsed_since_job_start is non-decreasing after sort.

    A decrease in elapsed time after chronological sorting indicates an
    inconsistent active clock. In the PowerWash data it comes from the
    extractor's implicit-gap correction subtracting an explicit exit -> resume
    pause a second time (see ``extract_player_sequences.py``). A tolerance of
    1 ms absorbs sub-second rounding.
    """
    col = "time_elapsed_since_job_start"
    if col not in df.columns:
        return True
    sorted_df = keep_only_subtask_rows(sort_attempt_rows(ensure_required_columns(df.copy())))
    elapsed = pd.to_numeric(sorted_df[col], errors="coerce").dropna()
    if len(elapsed) < 2:
        return True
    return bool((elapsed.diff().dropna() >= -0.001).all())


def normalize_flag(series):
    """Normalize boolean-like text into 1.0, 0.0, or NaN."""
    cleaned = clean_string_series(series).astype(str).str.strip().str.lower()
    result = pd.Series(np.nan, index=series.index, dtype="float64")
    result[cleaned.isin(TRUE_STRINGS)] = 1.0
    result[cleaned.isin(FALSE_STRINGS)] = 0.0
    return result


def resolve_raw_elapsed_time_axis(df):
    """Return the raw wall-clock elapsed series, using the backup column when present."""
    raw_column = "time_elapsed_since_job_start_raw"
    if raw_column in df.columns:
        raw_elapsed = pd.to_numeric(df[raw_column], errors="coerce").astype("float64")
        if raw_elapsed.notna().any():
            return raw_elapsed
    return pd.to_numeric(df["time_elapsed_since_job_start"], errors="coerce").astype("float64")


def build_active_time_axis(df):
    """
    Return the pause-adjusted elapsed axis prepared during extraction.

    Extracted attempt files keep the pause-adjusted active clock in
    ``time_elapsed_since_job_start``; the unadjusted wall-clock value is in
    ``time_elapsed_since_job_start_raw`` (see ``resolve_raw_elapsed_time_axis``).
    """
    return pd.to_numeric(df["time_elapsed_since_job_start"], errors="coerce").astype("float64")


def resolve_elapsed_time_axis(df, time_axis="active"):
    """Return the elapsed-time series used to derive temporal features."""
    if time_axis == "active":
        active_elapsed = build_active_time_axis(df)
        if active_elapsed.notna().any():
            return active_elapsed
    return resolve_raw_elapsed_time_axis(df)


def safe_changed(series):
    """Return 1 when a non-missing value changes from the previous row."""
    cleaned = clean_string_series(series)
    previous = cleaned.shift(1)
    changed = cleaned.notna() & previous.notna() & cleaned.ne(previous)
    return changed.astype("int8")


def build_sort_time_proxy(parsed_time_utc, elapsed_seconds):
    """
    Build a robust time-based sort key.

    ``Time_utc`` stays the primary ordering signal. When timestamps are missing
    or malformed, elapsed time is used as a conservative fallback so rows can
    still be sorted without dropping them.
    """
    sort_proxy = parsed_time_utc.copy()
    valid_pairs = parsed_time_utc.notna() & elapsed_seconds.notna()
    if valid_pairs.any():
        anchor_time = parsed_time_utc[valid_pairs].min()
        anchor_elapsed = elapsed_seconds[valid_pairs].min()
        fallback = anchor_time + pd.to_timedelta(elapsed_seconds - anchor_elapsed, unit="s")
        sort_proxy = sort_proxy.where(sort_proxy.notna(), fallback)
    return sort_proxy


def sort_attempt_rows(df):
    """Sort rows by Time_utc, breaking ties with elapsed time and input order."""
    sorted_df = df.copy()
    sorted_df["_original_order"] = np.arange(len(sorted_df))
    sorted_df["_elapsed_numeric"] = resolve_raw_elapsed_time_axis(sorted_df)
    parsed_time_utc = pd.to_datetime(sorted_df["Time_utc"], errors="coerce", utc=True)
    sorted_df["_sort_time_proxy"] = build_sort_time_proxy(
        parsed_time_utc, sorted_df["_elapsed_numeric"]
    )

    sorted_df = sorted_df.sort_values(
        by=["_sort_time_proxy", "_elapsed_numeric", "_original_order"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)

    return sorted_df.drop(columns=["_original_order", "_elapsed_numeric", "_sort_time_proxy"])


def deduplicate_attempt_rows(df):
    """
    Remove duplicate rows and repeated subtask completions within one attempt.

    Each subtask completes once per attempt; replayed completions keep the
    earliest occurrence after sorting.
    """
    deduped = df.copy()
    deduped["LastSubtaskCompleted"] = deduped["LastSubtaskCompleted"].map(normalize_subtask_label)
    deduped = deduped.drop_duplicates().reset_index(drop=True)

    subtask_names = (
        clean_string_series(deduped["LastSubtaskCompleted"]).fillna("").astype(str).str.strip()
    )
    is_subtask_row = deduped["event_type"].eq("subtask_completed")
    repeated_subtask_completion = (
        is_subtask_row & subtask_names.notna() & subtask_names.duplicated(keep="first")
    )
    if repeated_subtask_completion.any():
        deduped = deduped.loc[~repeated_subtask_completion].reset_index(drop=True)

    return deduped


def keep_only_subtask_rows(df):
    """Return only ``subtask_completed`` rows for feature derivation."""
    kept = df.loc[df["event_type"].eq("subtask_completed")].copy()
    return kept.reset_index(drop=True)


def parse_position_columns(series):
    """Safely parse a ``CurrentPosition`` string into x/y/z float columns."""
    extracted = clean_string_series(series).fillna("").astype(str).str.extract(POSITION_PATTERN)
    return (
        pd.to_numeric(extracted[0], errors="coerce"),
        pd.to_numeric(extracted[1], errors="coerce"),
        pd.to_numeric(extracted[2], errors="coerce"),
    )


def _task_coord_rows_to_lookup(task_rows_by_task):
    """Convert canonical task-map rows into ``{task: {subtask: (x, z)}}``."""
    lookup = {task_name: {} for task_name in EXPECTED_NORMALIZED_SUBTASKS_BY_TASK}
    for task_name, rows in task_rows_by_task.items():
        task_lookup = lookup.setdefault(str(task_name), {})
        for row in rows:
            subtask_name = str(row.get("subtask_name") or "").strip()
            x_coord = row.get("x")
            z_coord = row.get("z")
            if not subtask_name:
                continue
            if x_coord is None or z_coord is None:
                continue
            if not np.isfinite(x_coord) or not np.isfinite(z_coord):
                continue
            task_lookup[subtask_name] = (float(x_coord), float(z_coord))
    return lookup


@lru_cache(maxsize=8)
def load_canonical_subtask_coordinates(input_dir=""):
    """Load canonical subtask x/z coordinates from CSV or derive them on demand."""
    lookup = {task_name: {} for task_name in EXPECTED_NORMALIZED_SUBTASKS_BY_TASK}

    if TASK_MAP_TABLE_PATH.exists():
        table = pd.read_csv(TASK_MAP_TABLE_PATH)
        grouped_rows = {}
        for _, row in table.iterrows():
            grouped_rows.setdefault(str(row.get("task") or ""), []).append(
                {
                    "subtask_name": row.get("subtask_name"),
                    "x": pd.to_numeric(pd.Series([row.get("x")]), errors="coerce").iloc[0],
                    "z": pd.to_numeric(pd.Series([row.get("z")]), errors="coerce").iloc[0],
                }
            )
        lookup = _task_coord_rows_to_lookup(grouped_rows)

    missing_tasks = [task_name for task_name, task_lookup in lookup.items() if not task_lookup]
    if not missing_tasks:
        return lookup

    if not input_dir:
        return lookup

    input_root = Path(input_dir).expanduser().resolve()
    if not input_root.exists() or not input_root.is_dir():
        return lookup

    from powerwash.analysis.plot_subtask_task_maps import derive_task_rows

    derived_lookup = _task_coord_rows_to_lookup(derive_task_rows(input_root, missing_tasks))
    for task_name, task_lookup in derived_lookup.items():
        if task_lookup:
            lookup[task_name] = task_lookup
    return lookup


def _forward_fill_numeric(values):
    """Forward-fill a numeric array while preserving leading missing values."""
    return pd.Series(values, dtype="float64").ffill().to_numpy(dtype="float64")


def _safe_ratio_target(numerator, denominator):
    """Return ``numerator / denominator`` with ``NaN`` for non-positive denominators."""
    result = np.full(len(numerator), np.nan, dtype="float64")
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0.0)
    if valid.any():
        result[valid] = numerator[valid] / denominator[valid]
    return result


def future_revisit_ratio_target(
    elapsed_seconds,
    pos_x_values,
    pos_z_values,
    horizon_seconds,
    min_future_seconds,
    revisit_cell_size=REVISIT_CELL_SIZE,
):
    """Fraction of future path length spent in x/z cells seen before the current row."""
    target = np.full(len(elapsed_seconds), np.nan, dtype="float64")
    (
        trajectory_row_indices,
        _trajectory_times,
        valid_current_positions,
        valid_current_rows,
        horizon_end_positions,
        _available,
    ) = resolve_future_xz_trajectory_windows(
        elapsed_seconds,
        pos_x_values,
        pos_z_values,
        horizon_seconds,
        min_future_seconds,
    )
    if trajectory_row_indices is None or valid_current_rows is None:
        return target

    trajectory_points = np.column_stack(
        (pos_x_values[trajectory_row_indices], pos_z_values[trajectory_row_indices])
    )
    step_lengths = np.zeros(len(trajectory_row_indices), dtype="float64")
    if len(trajectory_row_indices) > 1:
        step_lengths[1:] = np.linalg.norm(np.diff(trajectory_points, axis=0), axis=1)

    x_cells = np.floor(trajectory_points[:, 0] / revisit_cell_size).astype("int64")
    z_cells = np.floor(trajectory_points[:, 1] / revisit_cell_size).astype("int64")
    first_occurrence = np.zeros(len(trajectory_row_indices), dtype="int64")
    first_seen = {}
    for idx, cell_key in enumerate(zip(x_cells.tolist(), z_cells.tolist())):
        if cell_key not in first_seen:
            first_seen[cell_key] = idx
        first_occurrence[idx] = first_seen[cell_key]

    prefix = np.concatenate(([0.0], np.cumsum(step_lengths)))
    for current_pos, current_row, end_pos in zip(
        valid_current_positions,
        valid_current_rows,
        horizon_end_positions,
    ):
        total_path = prefix[end_pos + 1] - prefix[current_pos + 1]
        if total_path <= 0.0:
            continue
        future_start = current_pos + 1
        revisit_mask = first_occurrence[future_start : end_pos + 1] <= current_pos
        revisit_dist = step_lengths[future_start : end_pos + 1][revisit_mask].sum()
        target[current_row] = revisit_dist / total_path

    return target


def future_choice_rank_targets(
    elapsed_seconds,
    pos_x_values,
    pos_z_values,
    event_types,
    subtask_names,
    task_name,
    canonical_coords_by_task,
):
    """Compute normalized next-choice and mean-choice rank targets from current x/z state."""
    next_choice_rank_next_60s = np.full(len(elapsed_seconds), np.nan, dtype="float64")
    mean_choice_rank_next_60s = np.full(len(elapsed_seconds), np.nan, dtype="float64")
    mean_choice_rank_next_120s = np.full(len(elapsed_seconds), np.nan, dtype="float64")

    task_lookup = canonical_coords_by_task.get(str(task_name), {})
    expected_subtasks = EXPECTED_NORMALIZED_SUBTASKS_BY_TASK.get(str(task_name), frozenset())
    rankable_subtasks = [
        subtask_name for subtask_name in sorted(expected_subtasks) if subtask_name in task_lookup
    ]
    if len(rankable_subtasks) <= 1:
        return (
            next_choice_rank_next_60s,
            mean_choice_rank_next_60s,
            mean_choice_rank_next_120s,
        )

    current_x = _forward_fill_numeric(pos_x_values)
    current_z = _forward_fill_numeric(pos_z_values)

    subtask_event_rows = np.flatnonzero(
        (event_types == "subtask_completed") & np.isfinite(elapsed_seconds) & (subtask_names != "")
    )
    subtask_event_times = elapsed_seconds[subtask_event_rows]
    subtask_event_names = subtask_names[subtask_event_rows]

    completed_so_far = set()
    for row_idx in range(len(elapsed_seconds)):
        if event_types[row_idx] == "subtask_completed" and subtask_names[row_idx]:
            completed_so_far.add(str(subtask_names[row_idx]))

        if not np.isfinite(elapsed_seconds[row_idx]):
            continue
        if not np.isfinite(current_x[row_idx]) or not np.isfinite(current_z[row_idx]):
            continue

        remaining_subtasks = [
            subtask_name
            for subtask_name in rankable_subtasks
            if subtask_name not in completed_so_far
        ]
        if len(remaining_subtasks) <= 1:
            continue

        current_xz = np.array([current_x[row_idx], current_z[row_idx]], dtype="float64")
        ranked_remaining = sorted(
            remaining_subtasks,
            key=lambda name: (
                float(np.sum((np.asarray(task_lookup[name], dtype="float64") - current_xz) ** 2)),
                name,
            ),
        )
        denom = float(len(ranked_remaining) - 1)
        rank_lookup = {
            subtask_name: float(rank_idx) / denom
            for rank_idx, subtask_name in enumerate(ranked_remaining)
        }

        future_start = np.searchsorted(subtask_event_times, elapsed_seconds[row_idx], side="right")
        if future_start >= len(subtask_event_times):
            continue

        future_end_60 = np.searchsorted(
            subtask_event_times, elapsed_seconds[row_idx] + 60.0, side="right"
        )
        future_end_120 = np.searchsorted(
            subtask_event_times, elapsed_seconds[row_idx] + 120.0, side="right"
        )

        if future_start < future_end_60:
            next_name = str(subtask_event_names[future_start])
            if next_name in rank_lookup:
                next_choice_rank_next_60s[row_idx] = rank_lookup[next_name]

            future_names_60 = [
                str(subtask_name)
                for subtask_name in subtask_event_names[future_start:future_end_60]
                if str(subtask_name) in rank_lookup
            ]
            if future_names_60:
                mean_choice_rank_next_60s[row_idx] = float(
                    np.mean([rank_lookup[subtask_name] for subtask_name in future_names_60])
                )

        if future_start < future_end_120:
            future_names_120 = [
                str(subtask_name)
                for subtask_name in subtask_event_names[future_start:future_end_120]
                if str(subtask_name) in rank_lookup
            ]
            if future_names_120:
                mean_choice_rank_next_120s[row_idx] = float(
                    np.mean([rank_lookup[subtask_name] for subtask_name in future_names_120])
                )

    return (
        next_choice_rank_next_60s,
        mean_choice_rank_next_60s,
        mean_choice_rank_next_120s,
    )


def future_standoff_distance_targets(
    elapsed_seconds,
    pos_x_values,
    pos_z_values,
    event_types,
    subtask_names,
    task_name,
    canonical_coords_by_task,
):
    """
    Compute x/z standoff distances to the canonical coordinates of future subtasks.

    A subtask's standoff is the mean x/z distance to its canonical coordinate over
    its work window. ``next_subtask_standoff_distance`` uses the next completed
    subtask, windowed from the current row to its completion.
    ``mean_standoff_distance_next_{60,120}s`` averages the standoffs of all
    subtasks completed within the horizon, each later window starting at the
    previous completion.
    """
    next_subtask_standoff_distance = np.full(len(elapsed_seconds), np.nan, dtype="float64")
    mean_standoff_distance_next_60s = np.full(len(elapsed_seconds), np.nan, dtype="float64")
    mean_standoff_distance_next_120s = np.full(len(elapsed_seconds), np.nan, dtype="float64")

    task_lookup = canonical_coords_by_task.get(str(task_name), {})
    if not task_lookup:
        return (
            next_subtask_standoff_distance,
            mean_standoff_distance_next_60s,
            mean_standoff_distance_next_120s,
        )

    current_x = _forward_fill_numeric(pos_x_values)
    current_z = _forward_fill_numeric(pos_z_values)

    valid_position_mask = np.isfinite(current_x) & np.isfinite(current_z)
    subtask_event_rows = np.flatnonzero(
        (event_types == "subtask_completed") & np.isfinite(elapsed_seconds) & (subtask_names != "")
    )
    if subtask_event_rows.size == 0:
        return (
            next_subtask_standoff_distance,
            mean_standoff_distance_next_60s,
            mean_standoff_distance_next_120s,
        )

    subtask_event_times = elapsed_seconds[subtask_event_rows]
    subtask_event_names = subtask_names[subtask_event_rows]

    for row_idx in range(len(elapsed_seconds)):
        if not np.isfinite(elapsed_seconds[row_idx]):
            continue
        if not valid_position_mask[row_idx]:
            continue

        future_start = np.searchsorted(subtask_event_times, elapsed_seconds[row_idx], side="right")
        if future_start >= len(subtask_event_rows):
            continue

        future_end_60 = np.searchsorted(
            subtask_event_times,
            elapsed_seconds[row_idx] + 60.0,
            side="right",
        )
        future_end_120 = np.searchsorted(
            subtask_event_times,
            elapsed_seconds[row_idx] + 120.0,
            side="right",
        )

        standoff_by_event = []
        for event_offset in range(future_start, future_end_120):
            event_row = int(subtask_event_rows[event_offset])
            subtask_name = str(subtask_event_names[event_offset])
            if subtask_name not in task_lookup:
                standoff_by_event.append(np.nan)
                continue

            window_start_row = (
                row_idx
                if event_offset == future_start
                else int(subtask_event_rows[event_offset - 1])
            )
            window_mask = valid_position_mask[window_start_row : event_row + 1]
            if not window_mask.any():
                standoff_by_event.append(np.nan)
                continue

            target_xz = np.asarray(task_lookup[subtask_name], dtype="float64")
            window_points = np.column_stack(
                (
                    current_x[window_start_row : event_row + 1][window_mask],
                    current_z[window_start_row : event_row + 1][window_mask],
                )
            )
            distances = np.linalg.norm(window_points - target_xz, axis=1)
            standoff_by_event.append(float(np.mean(distances)))

        if not standoff_by_event:
            continue

        first_standoff = standoff_by_event[0]
        if np.isfinite(first_standoff):
            next_subtask_standoff_distance[row_idx] = first_standoff

        standoff_60 = [
            value
            for value in standoff_by_event[: future_end_60 - future_start]
            if np.isfinite(value)
        ]
        if standoff_60:
            mean_standoff_distance_next_60s[row_idx] = float(np.mean(standoff_60))

        standoff_120 = [value for value in standoff_by_event if np.isfinite(value)]
        if standoff_120:
            mean_standoff_distance_next_120s[row_idx] = float(np.mean(standoff_120))

    return (
        next_subtask_standoff_distance,
        mean_standoff_distance_next_60s,
        mean_standoff_distance_next_120s,
    )


def rolling_sum_over_elapsed(values, elapsed_seconds, window_seconds):
    """Sum values over the trailing elapsed-time window, inclusive of the current row."""
    result = np.full(len(values), np.nan, dtype="float64")
    valid = np.isfinite(elapsed_seconds)
    if not valid.any():
        return result

    valid_times = elapsed_seconds[valid]
    valid_values = np.where(np.isfinite(values[valid]), values[valid], 0.0)
    starts = np.searchsorted(valid_times, valid_times - window_seconds, side="left")
    prefix = np.concatenate(([0.0], np.cumsum(valid_values)))
    ends = np.arange(valid_times.size) + 1
    result[valid] = prefix[ends] - prefix[starts]
    return result


def rolling_difference_over_elapsed(cumulative_values, elapsed_seconds, window_seconds):
    """
    Compute cumulative gain over the trailing elapsed-time window.

    The baseline is the last available cumulative value at or before the window
    start. If the window reaches before the beginning of the sequence, the first
    available cumulative value is used, which yields 0 on the first valid row.
    """
    result = np.full(len(cumulative_values), np.nan, dtype="float64")
    valid = np.isfinite(elapsed_seconds) & np.isfinite(cumulative_values)
    if not valid.any():
        return result

    valid_times = elapsed_seconds[valid]
    valid_values = cumulative_values[valid]
    baseline_positions = (
        np.searchsorted(
            valid_times,
            valid_times - window_seconds,
            side="right",
        )
        - 1
    )
    baseline_positions = np.where(baseline_positions >= 0, baseline_positions, 0)
    result[valid] = valid_values - valid_values[baseline_positions]
    return result


def future_progress_target(elapsed_seconds, progress_values, horizon_seconds, min_future_seconds):
    """
    Compute future progress gain using a time-based lookahead window.

    For each row at time ``t``, the target uses the latest observed row at or
    before ``t + horizon_seconds``. The target is kept only when that row is at
    least ``min_future_seconds`` ahead of ``t``. This avoids weak labels near
    the end of a sequence.
    """
    target = np.full(len(elapsed_seconds), np.nan, dtype="float64")
    available = np.zeros(len(elapsed_seconds), dtype="int8")

    finite_time_mask = np.isfinite(elapsed_seconds)
    if not finite_time_mask.any():
        return target, available

    finite_row_indices = np.flatnonzero(finite_time_mask)
    finite_times = elapsed_seconds[finite_time_mask]
    current_times = elapsed_seconds[finite_row_indices]

    candidate_positions = (
        np.searchsorted(
            finite_times,
            current_times + horizon_seconds,
            side="right",
        )
        - 1
    )

    candidate_exists = candidate_positions >= 0
    if not candidate_exists.any():
        return target, available

    valid_current_rows = finite_row_indices[candidate_exists]
    valid_candidate_positions = candidate_positions[candidate_exists]
    candidate_times = finite_times[valid_candidate_positions]
    candidate_rows = finite_row_indices[valid_candidate_positions]

    enough_future = candidate_times >= (elapsed_seconds[valid_current_rows] + min_future_seconds)
    if not enough_future.any():
        return target, available

    valid_current_rows = valid_current_rows[enough_future]
    candidate_rows = candidate_rows[enough_future]

    current_progress = progress_values[valid_current_rows]
    future_progress = progress_values[candidate_rows]
    finite_progress = np.isfinite(current_progress) & np.isfinite(future_progress)
    if not finite_progress.any():
        return target, available

    valid_current_rows = valid_current_rows[finite_progress]
    candidate_rows = candidate_rows[finite_progress]

    target[valid_current_rows] = (
        progress_values[candidate_rows] - progress_values[valid_current_rows]
    )
    available[valid_current_rows] = 1
    return target, available


def future_position_change_to_next_subtask_target(
    elapsed_seconds,
    pos_x_values,
    pos_y_values,
    pos_z_values,
    event_types,
):
    """
    Compute event-aligned movement from the current row to the next subtask completion.

    For each current event row, the target is the signed x/y/z change between the
    current position and the first strictly future ``subtask_completed`` row. This
    defines a variable-horizon window that ends at the next completion event, so
    adjacent subtask rows create adjacent, non-overlapping target windows.
    """
    n_rows = len(elapsed_seconds)
    delta_x = np.full(n_rows, np.nan, dtype="float64")
    delta_y = np.full(n_rows, np.nan, dtype="float64")
    delta_z = np.full(n_rows, np.nan, dtype="float64")
    displacement = np.full(n_rows, np.nan, dtype="float64")
    xz_displacement = np.full(n_rows, np.nan, dtype="float64")
    time_to_next = np.full(n_rows, np.nan, dtype="float64")
    available = np.zeros(n_rows, dtype="int8")

    event_types = np.asarray(event_types).astype(str)
    subtask_event_rows = np.flatnonzero(
        (event_types == "subtask_completed") & np.isfinite(elapsed_seconds)
    )
    if subtask_event_rows.size < 2:
        return (
            delta_x,
            delta_y,
            delta_z,
            displacement,
            xz_displacement,
            time_to_next,
            available,
        )

    current_rows = subtask_event_rows[:-1]
    next_rows = subtask_event_rows[1:]

    valid = (
        np.isfinite(pos_x_values[current_rows])
        & np.isfinite(pos_y_values[current_rows])
        & np.isfinite(pos_z_values[current_rows])
        & np.isfinite(pos_x_values[next_rows])
        & np.isfinite(pos_y_values[next_rows])
        & np.isfinite(pos_z_values[next_rows])
        & np.isfinite(elapsed_seconds[current_rows])
        & np.isfinite(elapsed_seconds[next_rows])
        & (elapsed_seconds[next_rows] >= elapsed_seconds[current_rows])
    )
    if not valid.any():
        return (
            delta_x,
            delta_y,
            delta_z,
            displacement,
            xz_displacement,
            time_to_next,
            available,
        )

    current_rows = current_rows[valid]
    next_rows = next_rows[valid]

    dx = pos_x_values[next_rows] - pos_x_values[current_rows]
    dy = pos_y_values[next_rows] - pos_y_values[current_rows]
    dz = pos_z_values[next_rows] - pos_z_values[current_rows]

    delta_x[current_rows] = dx
    delta_y[current_rows] = dy
    delta_z[current_rows] = dz
    displacement[current_rows] = np.sqrt(dx**2 + dy**2 + dz**2)
    xz_displacement[current_rows] = np.sqrt(dx**2 + dz**2)
    time_to_next[current_rows] = elapsed_seconds[next_rows] - elapsed_seconds[current_rows]
    available[current_rows] = 1

    return (
        delta_x,
        delta_y,
        delta_z,
        displacement,
        xz_displacement,
        time_to_next,
        available,
    )


def future_subtask_count_target(
    elapsed_seconds, is_subtask_event, horizon_seconds, min_future_seconds
):
    """
    Count distinct ``subtask_completed`` events within the next horizon.

    ``derive_features()`` deduplicates repeated subtask completions first, so
    these counts represent unique subtasks rather than raw replayed rows. Only
    rows where the sequence extends at least ``min_future_seconds`` beyond the
    current row are labelled, to avoid weak labels near the end of a session.
    """
    target = np.full(len(elapsed_seconds), np.nan, dtype="float64")
    available = np.zeros(len(elapsed_seconds), dtype="int8")

    finite_time_mask = np.isfinite(elapsed_seconds)
    if not finite_time_mask.any():
        return target, available

    finite_row_indices = np.flatnonzero(finite_time_mask)
    finite_times = elapsed_seconds[finite_time_mask]
    current_times = elapsed_seconds[finite_row_indices]

    # Find the last row within the horizon for each current row
    horizon_end_positions = (
        np.searchsorted(
            finite_times,
            current_times + horizon_seconds,
            side="right",
        )
        - 1
    )

    candidate_exists = horizon_end_positions >= 0
    enough_future = np.zeros(len(horizon_end_positions), dtype=bool)
    enough_future[candidate_exists] = finite_times[horizon_end_positions[candidate_exists]] >= (
        current_times[candidate_exists] + min_future_seconds
    )
    if not enough_future.any():
        return target, available

    valid_current_positions = np.flatnonzero(enough_future)
    valid_current_rows = finite_row_indices[valid_current_positions]
    horizon_start_positions = np.searchsorted(
        finite_times,
        current_times[valid_current_positions],
        side="right",
    )
    horizon_end_positions = horizon_end_positions[enough_future]

    subtask_values = np.asarray(is_subtask_event, dtype=bool)[finite_row_indices].astype("float64")
    prefix = np.concatenate(([0.0], np.cumsum(subtask_values)))
    target[valid_current_rows] = prefix[horizon_end_positions + 1] - prefix[horizon_start_positions]
    available[valid_current_rows] = 1

    return target, available


def future_subtask_target(elapsed_seconds, is_subtask_event, horizon_seconds, min_future_seconds):
    """
    Compute a binary target: 1 if any subtask_completed event occurs within the
    next ``horizon_seconds``, 0 otherwise.

    The input sequence is assumed to have been deduplicated already, so the
    target reflects distinct future subtasks. Only rows where the sequence
    extends at least ``min_future_seconds`` beyond the current row are labelled,
    to avoid weak labels near the end of a session.
    """
    count_target, available = future_subtask_count_target(
        elapsed_seconds,
        is_subtask_event,
        horizon_seconds,
        min_future_seconds,
    )
    target = np.full(len(elapsed_seconds), np.nan, dtype="float64")
    valid = np.isfinite(count_target)
    target[valid] = (count_target[valid] > 0).astype("float64")
    return target, available


def future_distance_sum_target(
    elapsed_seconds, distance_values, horizon_seconds, min_future_seconds
):
    """
    Sum ``distance_from_prev`` values for rows strictly after the current row
    and within ``horizon_seconds``.

    Only rows where the sequence extends at least ``min_future_seconds`` beyond
    the current row are labelled, matching the availability condition used by
    the subtask count targets.
    """
    target = np.full(len(elapsed_seconds), np.nan, dtype="float64")

    finite_time_mask = np.isfinite(elapsed_seconds)
    if not finite_time_mask.any():
        return target

    finite_row_indices = np.flatnonzero(finite_time_mask)
    finite_times = elapsed_seconds[finite_time_mask]
    current_times = elapsed_seconds[finite_row_indices]

    horizon_end_positions = (
        np.searchsorted(
            finite_times,
            current_times + horizon_seconds,
            side="right",
        )
        - 1
    )

    candidate_exists = horizon_end_positions >= 0
    enough_future = np.zeros(len(horizon_end_positions), dtype=bool)
    enough_future[candidate_exists] = finite_times[horizon_end_positions[candidate_exists]] >= (
        current_times[candidate_exists] + min_future_seconds
    )
    if not enough_future.any():
        return target

    valid_current_positions = np.flatnonzero(enough_future)
    valid_current_rows = finite_row_indices[valid_current_positions]
    horizon_start_positions = np.searchsorted(
        finite_times,
        current_times[valid_current_positions],
        side="right",
    )
    horizon_end_positions = horizon_end_positions[enough_future]

    dist_values = np.where(
        np.isfinite(distance_values[finite_row_indices]), distance_values[finite_row_indices], 0.0
    )
    prefix = np.concatenate(([0.0], np.cumsum(dist_values)))
    target[valid_current_rows] = prefix[horizon_end_positions + 1] - prefix[horizon_start_positions]
    return target


def resolve_future_xz_trajectory_windows(
    elapsed_seconds, pos_x_values, pos_z_values, horizon_seconds, min_future_seconds
):
    """
    Resolve future window bounds over the parseable x/z trajectory rows only.

    Rows with unparseable positions are skipped, matching how the plotting
    utilities compute trajectory metrics.
    """
    available = np.zeros(len(elapsed_seconds), dtype="int8")
    valid_mask = (
        np.isfinite(elapsed_seconds) & np.isfinite(pos_x_values) & np.isfinite(pos_z_values)
    )
    if not valid_mask.any():
        return None, None, None, None, None, available

    trajectory_row_indices = np.flatnonzero(valid_mask)
    trajectory_times = elapsed_seconds[valid_mask]
    horizon_end_positions = (
        np.searchsorted(
            trajectory_times,
            trajectory_times + horizon_seconds,
            side="right",
        )
        - 1
    )

    enough_future = trajectory_times[horizon_end_positions] >= (
        trajectory_times + min_future_seconds
    )
    if not enough_future.any():
        return trajectory_row_indices, trajectory_times, None, None, None, available

    valid_current_positions = np.flatnonzero(enough_future)
    valid_current_rows = trajectory_row_indices[valid_current_positions]
    available[valid_current_rows] = 1

    return (
        trajectory_row_indices,
        trajectory_times,
        valid_current_positions,
        valid_current_rows,
        horizon_end_positions[enough_future],
        available,
    )


def future_path_length_target(
    elapsed_seconds, pos_x_values, pos_z_values, horizon_seconds, min_future_seconds
):
    """
    Sum future x/z path length within the requested horizon.

    The target is defined on the parseable x/z trajectory, matching
    ``compute_path_length()`` used by the PowerWash analysis plots.
    """
    target = np.full(len(elapsed_seconds), np.nan, dtype="float64")
    (
        trajectory_row_indices,
        _trajectory_times,
        valid_current_positions,
        valid_current_rows,
        horizon_end_positions,
        _available,
    ) = resolve_future_xz_trajectory_windows(
        elapsed_seconds,
        pos_x_values,
        pos_z_values,
        horizon_seconds,
        min_future_seconds,
    )
    if trajectory_row_indices is None or valid_current_rows is None:
        return target

    trajectory_points = np.column_stack(
        (pos_x_values[trajectory_row_indices], pos_z_values[trajectory_row_indices])
    )
    step_lengths = np.zeros(len(trajectory_row_indices), dtype="float64")
    if len(trajectory_row_indices) > 1:
        step_lengths[1:] = np.linalg.norm(np.diff(trajectory_points, axis=0), axis=1)
    prefix = np.concatenate(([0.0], np.cumsum(step_lengths)))
    target[valid_current_rows] = (
        prefix[horizon_end_positions + 1] - prefix[valid_current_positions + 1]
    )
    return target


def build_xz_heading_change_prefixes(trajectory_points):
    """
    Build cumulative future-turn summaries over an x/z trajectory.

    Returns prefix sums for both absolute heading change in radians and the
    binary reversal indicator (>90 degrees). Zero-length steps are ignored so
    stationary rows do not create artificial turning.
    """
    if len(trajectory_points) < 2:
        zero_prefix = np.array([0.0], dtype="float64")
        return zero_prefix, zero_prefix

    diffs = np.diff(trajectory_points, axis=0)
    if len(diffs) < 2:
        zero_prefix = np.array([0.0], dtype="float64")
        return zero_prefix, zero_prefix

    norms = np.linalg.norm(diffs, axis=1)
    nonzero = norms > 0
    dirs = np.zeros_like(diffs)
    dirs[nonzero] = diffs[nonzero] / norms[nonzero, None]
    dots = np.clip((dirs[:-1] * dirs[1:]).sum(axis=1), -1.0, 1.0)
    valid_pairs = nonzero[:-1] & nonzero[1:]

    turn_angles = np.zeros(len(dots), dtype="float64")
    turn_angles[valid_pairs] = np.arccos(dots[valid_pairs])
    reversal_flags = ((dots < 0.0) & valid_pairs).astype("float64")

    return (
        np.concatenate(([0.0], np.cumsum(turn_angles))),
        np.concatenate(([0.0], np.cumsum(reversal_flags))),
    )


def future_total_angular_turn_target(
    elapsed_seconds,
    pos_x_values,
    pos_z_values,
    horizon_seconds,
    min_future_seconds,
):
    """
    Sum future absolute heading change in x/z within the requested horizon.

    The target is measured in radians over consecutive non-zero x/z movement
    steps and uses the same parseable-trajectory windowing as the path-length
    and reversal targets.
    """
    target = np.full(len(elapsed_seconds), np.nan, dtype="float64")
    (
        trajectory_row_indices,
        _trajectory_times,
        valid_current_positions,
        valid_current_rows,
        horizon_end_positions,
        _available,
    ) = resolve_future_xz_trajectory_windows(
        elapsed_seconds,
        pos_x_values,
        pos_z_values,
        horizon_seconds,
        min_future_seconds,
    )
    if trajectory_row_indices is None or valid_current_rows is None:
        return target

    trajectory_points = np.column_stack(
        (pos_x_values[trajectory_row_indices], pos_z_values[trajectory_row_indices])
    )
    angle_prefix, _reversal_prefix = build_xz_heading_change_prefixes(trajectory_points)
    angular_end_positions = np.maximum(horizon_end_positions - 1, valid_current_positions)
    target[valid_current_rows] = (
        angle_prefix[angular_end_positions] - angle_prefix[valid_current_positions]
    )
    return target


def future_direction_reversal_target(
    elapsed_seconds,
    pos_x_values,
    pos_z_values,
    horizon_seconds,
    min_future_seconds,
):
    """
    Count future x/z heading reversals (>90 degrees) within the horizon.

    This mirrors ``compute_zigzag_score()`` by ignoring zero-length steps and
    counting only consecutive step pairs whose normalized dot product is
    negative.
    """
    target = np.full(len(elapsed_seconds), np.nan, dtype="float64")
    (
        trajectory_row_indices,
        _trajectory_times,
        valid_current_positions,
        valid_current_rows,
        horizon_end_positions,
        _available,
    ) = resolve_future_xz_trajectory_windows(
        elapsed_seconds,
        pos_x_values,
        pos_z_values,
        horizon_seconds,
        min_future_seconds,
    )
    if trajectory_row_indices is None or valid_current_rows is None:
        return target

    trajectory_points = np.column_stack(
        (pos_x_values[trajectory_row_indices], pos_z_values[trajectory_row_indices])
    )
    _angle_prefix, reversal_prefix = build_xz_heading_change_prefixes(trajectory_points)
    reversal_end_positions = np.maximum(horizon_end_positions - 1, valid_current_positions)
    target[valid_current_rows] = (
        reversal_prefix[reversal_end_positions] - reversal_prefix[valid_current_positions]
    )
    return target


def derive_features(df, canonical_coords_by_task=None, time_axis="active"):
    """Derive transformer-friendly sequence features for one attempt DataFrame."""
    processed = sort_attempt_rows(ensure_required_columns(df.copy()))
    # Ignore sparse state-update rows so every sequence uses the same event type.
    processed = keep_only_subtask_rows(processed)
    processed = deduplicate_attempt_rows(processed)
    original_columns = list(processed.columns)
    task_name = resolve_attempt_task(processed)
    canonical_coords_by_task = canonical_coords_by_task or {}

    elapsed = resolve_elapsed_time_axis(processed, time_axis=time_axis)
    processed["time_elapsed_since_job_start"] = elapsed
    progress = pd.to_numeric(processed["LevelProgressionAmount"], errors="coerce")
    subtask_names = (
        clean_string_series(processed["LastSubtaskCompleted"]).fillna("").astype(str).str.strip()
    )

    # Only subtask rows are kept, so is_subtask_event (1), is_state_event (0) and
    # time_since_last_state (NaN) are constant placeholders that keep the encoder input layout.
    processed["is_subtask_event"] = np.ones(len(processed), dtype="int8")
    processed["is_state_event"] = np.zeros(len(processed), dtype="int8")

    processed["delta_time"] = elapsed.diff()

    last_subtask_elapsed = elapsed.ffill().shift(1)
    last_state_elapsed = pd.Series(np.nan, index=processed.index, dtype="float64")
    processed["time_since_last_subtask"] = elapsed - last_subtask_elapsed
    processed["time_since_last_state"] = elapsed - last_state_elapsed

    processed["delta_progress"] = progress.diff()
    processed["progress_rate"] = np.where(
        processed["delta_time"] > 0,
        processed["delta_progress"] / processed["delta_time"],
        np.nan,
    )

    elapsed_np = elapsed.to_numpy(dtype="float64")
    progress_np = progress.to_numpy(dtype="float64")
    processed["rolling_progress_30s"] = rolling_difference_over_elapsed(
        progress_np, elapsed_np, 30.0
    )
    processed["rolling_progress_60s"] = rolling_difference_over_elapsed(
        progress_np, elapsed_np, 60.0
    )

    pos_x, pos_y, pos_z = parse_position_columns(processed["CurrentPosition"])
    processed["pos_x"] = pos_x
    processed["pos_y"] = pos_y
    processed["pos_z"] = pos_z

    processed["delta_x"] = processed["pos_x"].diff()
    processed["delta_y"] = processed["pos_y"].diff()
    processed["delta_z"] = processed["pos_z"].diff()
    processed["distance_from_prev"] = np.sqrt(
        processed["delta_x"] ** 2 + processed["delta_y"] ** 2 + processed["delta_z"] ** 2
    )
    processed["speed_from_prev"] = np.where(
        processed["delta_time"] > 0,
        processed["distance_from_prev"] / processed["delta_time"],
        np.nan,
    )

    distance_np = processed["distance_from_prev"].to_numpy(dtype="float64")
    processed["distance_traveled_30s"] = rolling_sum_over_elapsed(distance_np, elapsed_np, 30.0)
    processed["distance_traveled_60s"] = rolling_sum_over_elapsed(distance_np, elapsed_np, 60.0)

    pos_x_np = pos_x.to_numpy(dtype="float64")
    pos_y_np = pos_y.to_numpy(dtype="float64")
    pos_z_np = pos_z.to_numpy(dtype="float64")
    processed["delta_x_10s"] = rolling_difference_over_elapsed(pos_x_np, elapsed_np, 10.0)
    processed["delta_y_10s"] = rolling_difference_over_elapsed(pos_y_np, elapsed_np, 10.0)
    processed["delta_z_10s"] = rolling_difference_over_elapsed(pos_z_np, elapsed_np, 10.0)
    processed["displacement_10s"] = np.sqrt(
        processed["delta_x_10s"] ** 2
        + processed["delta_y_10s"] ** 2
        + processed["delta_z_10s"] ** 2
    )

    processed["washer_changed"] = safe_changed(processed["CurrentWasher"])
    processed["nozzle_changed"] = safe_changed(processed["CurrentNozzle"])
    processed["extension_changed"] = safe_changed(processed["CurrentExtension"])

    any_tool_changed = (
        processed["washer_changed"] | processed["nozzle_changed"] | processed["extension_changed"]
    ).astype("float64")
    processed["tool_change_count_30s"] = rolling_sum_over_elapsed(
        any_tool_changed, elapsed_np, 30.0
    )
    processed["tool_change_count_60s"] = rolling_sum_over_elapsed(
        any_tool_changed, elapsed_np, 60.0
    )

    processed["crouch_changed"] = safe_changed(processed["CrouchState"])
    processed["idle_flag"] = normalize_flag(processed["IsIdleInGame"])
    processed["menu_flag"] = normalize_flag(processed["IsInMenu"])

    processed["idle_count_60s"] = rolling_sum_over_elapsed(
        processed["idle_flag"].to_numpy(dtype="float64"),
        elapsed_np,
        60.0,
    )
    processed["menu_count_60s"] = rolling_sum_over_elapsed(
        processed["menu_flag"].to_numpy(dtype="float64"),
        elapsed_np,
        60.0,
    )

    is_subtask_float = processed["is_subtask_event"].to_numpy(dtype="float64")
    processed["subtask_count_30s"] = rolling_sum_over_elapsed(is_subtask_float, elapsed_np, 30.0)
    processed["subtask_count_60s"] = rolling_sum_over_elapsed(is_subtask_float, elapsed_np, 60.0)

    row_weights = np.ones(len(processed), dtype="float64")
    processed["rows_in_last_30s"] = rolling_sum_over_elapsed(row_weights, elapsed_np, 30.0)
    processed["rows_in_last_60s"] = rolling_sum_over_elapsed(row_weights, elapsed_np, 60.0)

    max_elapsed = elapsed.max(skipna=True)
    processed["time_remaining_proxy"] = max_elapsed - elapsed if pd.notna(max_elapsed) else np.nan

    progress_next_30s, target_30s_available = future_progress_target(
        elapsed_np,
        progress_np,
        horizon_seconds=30.0,
        min_future_seconds=20.0,
    )
    progress_next_60s, target_60s_available = future_progress_target(
        elapsed_np,
        progress_np,
        horizon_seconds=60.0,
        min_future_seconds=50.0,
    )
    is_subtask_np = processed["is_subtask_event"].to_numpy(dtype=bool)
    subtask_in_next_10s, target_subtask_10s_available = future_subtask_target(
        elapsed_np,
        is_subtask_np,
        horizon_seconds=10.0,
        min_future_seconds=8.0,
    )
    subtask_count_next_30s, subtask_avail_30s = future_subtask_count_target(
        elapsed_np,
        is_subtask_np,
        horizon_seconds=30.0,
        min_future_seconds=20.0,
    )
    subtask_count_next_60s, subtask_avail_60s = future_subtask_count_target(
        elapsed_np,
        is_subtask_np,
        horizon_seconds=60.0,
        min_future_seconds=50.0,
    )
    subtask_count_next_120s, subtask_avail_120s = future_subtask_count_target(
        elapsed_np,
        is_subtask_np,
        horizon_seconds=120.0,
        min_future_seconds=110.0,
    )

    distance_np = processed["distance_from_prev"].to_numpy(dtype="float64")
    dist_next_30s = future_distance_sum_target(elapsed_np, distance_np, 30.0, 20.0)
    dist_next_60s = future_distance_sum_target(elapsed_np, distance_np, 60.0, 50.0)
    dist_next_120s = future_distance_sum_target(elapsed_np, distance_np, 120.0, 110.0)
    path_length_next_30s = future_path_length_target(elapsed_np, pos_x_np, pos_z_np, 30.0, 20.0)
    path_length_next_60s = future_path_length_target(elapsed_np, pos_x_np, pos_z_np, 60.0, 50.0)
    path_length_next_120s = future_path_length_target(elapsed_np, pos_x_np, pos_z_np, 120.0, 110.0)
    total_angular_turn_next_30s = future_total_angular_turn_target(
        elapsed_np,
        pos_x_np,
        pos_z_np,
        30.0,
        20.0,
    )
    total_angular_turn_next_60s = future_total_angular_turn_target(
        elapsed_np,
        pos_x_np,
        pos_z_np,
        60.0,
        50.0,
    )
    total_angular_turn_next_120s = future_total_angular_turn_target(
        elapsed_np,
        pos_x_np,
        pos_z_np,
        120.0,
        110.0,
    )
    direction_reversals_next_30s = future_direction_reversal_target(
        elapsed_np,
        pos_x_np,
        pos_z_np,
        30.0,
        20.0,
    )
    direction_reversals_next_60s = future_direction_reversal_target(
        elapsed_np,
        pos_x_np,
        pos_z_np,
        60.0,
        50.0,
    )
    direction_reversals_next_120s = future_direction_reversal_target(
        elapsed_np,
        pos_x_np,
        pos_z_np,
        120.0,
        110.0,
    )
    revisit_ratio_next_30s = future_revisit_ratio_target(elapsed_np, pos_x_np, pos_z_np, 30.0, 20.0)
    revisit_ratio_next_60s = future_revisit_ratio_target(elapsed_np, pos_x_np, pos_z_np, 60.0, 50.0)
    revisit_ratio_next_120s = future_revisit_ratio_target(
        elapsed_np, pos_x_np, pos_z_np, 120.0, 110.0
    )
    (
        next_choice_rank_next_60s,
        mean_choice_rank_next_60s,
        mean_choice_rank_next_120s,
    ) = future_choice_rank_targets(
        elapsed_np,
        pos_x_np,
        pos_z_np,
        processed["event_type"].astype(str).to_numpy(),
        subtask_names.to_numpy(dtype=object),
        task_name,
        canonical_coords_by_task,
    )
    (
        next_subtask_standoff_distance,
        mean_standoff_distance_next_60s,
        mean_standoff_distance_next_120s,
    ) = future_standoff_distance_targets(
        elapsed_np,
        pos_x_np,
        pos_z_np,
        processed["event_type"].astype(str).to_numpy(),
        subtask_names.to_numpy(dtype=object),
        task_name,
        canonical_coords_by_task,
    )

    def _efficiency(dist, subtask_count, avail):
        eff = dist / (subtask_count + D_MIN)
        eff[avail == 0] = np.nan
        return eff

    processed["progress_next_30s"] = progress_next_30s
    processed["progress_next_60s"] = progress_next_60s
    processed["progress_acceleration_next_60s"] = progress_next_60s - processed[
        "rolling_progress_60s"
    ].to_numpy(dtype="float64")
    processed["subtask_in_next_10s"] = subtask_in_next_10s
    processed["subtask_count_next_30s"] = subtask_count_next_30s
    processed["subtask_count_next_60s"] = subtask_count_next_60s
    processed["subtask_count_next_120s"] = subtask_count_next_120s
    processed["progress_efficiency_next_30s"] = _efficiency(
        dist_next_30s, subtask_count_next_30s, subtask_avail_30s
    )
    processed["progress_efficiency_next_60s"] = _efficiency(
        dist_next_60s, subtask_count_next_60s, subtask_avail_60s
    )
    processed["progress_efficiency_next_120s"] = _efficiency(
        dist_next_120s, subtask_count_next_120s, subtask_avail_120s
    )
    processed["path_length_next_30s"] = path_length_next_30s
    processed["path_length_next_60s"] = path_length_next_60s
    processed["path_length_next_120s"] = path_length_next_120s
    processed["total_angular_turn_next_30s"] = total_angular_turn_next_30s
    processed["total_angular_turn_next_60s"] = total_angular_turn_next_60s
    processed["total_angular_turn_next_120s"] = total_angular_turn_next_120s
    processed["direction_reversals_next_30s"] = direction_reversals_next_30s
    processed["direction_reversals_next_60s"] = direction_reversals_next_60s
    processed["direction_reversals_next_120s"] = direction_reversals_next_120s
    processed["reversal_density_next_30s"] = _safe_ratio_target(
        direction_reversals_next_30s, path_length_next_30s
    )
    processed["reversal_density_next_60s"] = _safe_ratio_target(
        direction_reversals_next_60s, path_length_next_60s
    )
    processed["reversal_density_next_120s"] = _safe_ratio_target(
        direction_reversals_next_120s, path_length_next_120s
    )
    processed["revisit_ratio_next_30s"] = revisit_ratio_next_30s
    processed["revisit_ratio_next_60s"] = revisit_ratio_next_60s
    processed["revisit_ratio_next_120s"] = revisit_ratio_next_120s
    processed["next_choice_rank_next_60s"] = next_choice_rank_next_60s
    processed["mean_choice_rank_next_60s"] = mean_choice_rank_next_60s
    processed["mean_choice_rank_next_120s"] = mean_choice_rank_next_120s
    processed["next_subtask_standoff_distance"] = next_subtask_standoff_distance
    processed["mean_standoff_distance_next_60s"] = mean_standoff_distance_next_60s
    processed["mean_standoff_distance_next_120s"] = mean_standoff_distance_next_120s

    progress_next_10s, target_10s_available = future_progress_target(
        elapsed_np,
        progress_np,
        horizon_seconds=10.0,
        min_future_seconds=8.0,
    )
    delta_x_next_10s, _ = future_progress_target(elapsed_np, pos_x_np, 10.0, 8.0)
    delta_y_next_10s, _ = future_progress_target(elapsed_np, pos_y_np, 10.0, 8.0)
    delta_z_next_10s, _ = future_progress_target(elapsed_np, pos_z_np, 10.0, 8.0)
    (
        delta_x_to_next_subtask,
        delta_y_to_next_subtask,
        delta_z_to_next_subtask,
        displacement_to_next_subtask,
        xz_displacement_to_next_subtask,
        time_to_next_subtask,
        target_next_subtask_position_available,
    ) = future_position_change_to_next_subtask_target(
        elapsed_np,
        pos_x_np,
        pos_y_np,
        pos_z_np,
        processed["event_type"].astype(str).to_numpy(),
    )

    processed["progress_next_10s"] = progress_next_10s
    processed["delta_x_next_10s"] = delta_x_next_10s
    processed["delta_y_next_10s"] = delta_y_next_10s
    processed["delta_z_next_10s"] = delta_z_next_10s
    processed["delta_x_to_next_subtask"] = delta_x_to_next_subtask
    processed["delta_y_to_next_subtask"] = delta_y_to_next_subtask
    processed["delta_z_to_next_subtask"] = delta_z_to_next_subtask
    processed["displacement_to_next_subtask"] = displacement_to_next_subtask
    processed["xz_displacement_to_next_subtask"] = xz_displacement_to_next_subtask
    processed["time_to_next_subtask"] = time_to_next_subtask
    processed["target_next_subtask_position_available"] = target_next_subtask_position_available
    processed["target_10s_available"] = target_10s_available
    processed["target_30s_available"] = target_30s_available
    processed["target_60s_available"] = target_60s_available
    processed["target_subtask_10s_available"] = target_subtask_10s_available

    return processed[original_columns + DERIVED_COLUMNS]


def select_training_ready_columns(df):
    """Restrict the final saved CSV to leakage/debug columns plus training features."""
    retained_columns = LEAKAGE_COLUMNS + TRAINING_FEATURE_COLUMNS + TARGET_COLUMNS
    trimmed = df.copy()
    for column in retained_columns:
        if column not in trimmed.columns:
            trimmed[column] = np.nan
    return trimmed[retained_columns]


def session_duration_seconds(df, time_axis="active"):
    """Return total session duration in seconds under the selected time axis."""
    if "time_elapsed_since_job_start" not in df.columns:
        return None
    processed = keep_only_subtask_rows(sort_attempt_rows(ensure_required_columns(df.copy())))
    t = resolve_elapsed_time_axis(processed, time_axis=time_axis).dropna()
    return float(t.iloc[-1]) if not t.empty else None


def write_schema_file(output_root):
    """Write a small schema file describing retained output columns."""
    schema = {
        "leakage_columns": LEAKAGE_COLUMNS,
        "training_feature_columns": TRAINING_FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "removed_metadata_columns": METADATA_COLUMNS,
        "removed_redundant_raw_columns": REDUNDANT_RAW_COLUMNS,
    }
    schema_path = output_root / SCHEMA_FILENAME
    with schema_path.open("w", encoding="utf-8") as handle:
        json.dump(schema, handle, indent=2, sort_keys=True)


def infer_input_task_name(input_path, input_root):
    """Infer a task name from an input CSV path when folders are task-structured."""
    try:
        relative_path = input_path.relative_to(input_root)
    except ValueError:
        relative_path = input_path

    if relative_path.parts and relative_path.parts[0] in EXPECTED_NORMALIZED_SUBTASKS_BY_TASK:
        return relative_path.parts[0]

    if input_path.parent.name in EXPECTED_NORMALIZED_SUBTASKS_BY_TASK:
        return input_path.parent.name

    return ""


def collect_csvs_under(search_root, output_root_resolved, output_inside_input):
    """Collect sorted CSV paths under one root while avoiding output recursion."""
    csv_files = []
    for path in sorted(search_root.rglob("*.csv")):
        if not path.is_file():
            continue
        if output_inside_input and str(path.resolve()).startswith(
            str(output_root_resolved) + str(Path("/"))
        ):
            continue
        csv_files.append(path)
    return csv_files


def list_input_csvs(input_root, output_root, task_names=None):
    """List input CSVs, respecting task order when the input root is task-structured."""
    input_root_resolved = input_root.resolve()
    output_root_resolved = output_root.resolve()
    output_inside_input = (
        str(output_root_resolved).startswith(str(input_root_resolved) + str(Path("/")))
        or output_root_resolved == input_root_resolved
    )

    if task_names and input_root.name in EXPECTED_NORMALIZED_SUBTASKS_BY_TASK:
        if input_root.name not in task_names:
            return []
        return collect_csvs_under(input_root, output_root_resolved, output_inside_input)

    if task_names:
        ordered_csv_files = []
        found_task_dirs = False
        seen_paths = set()

        for task_name in task_names:
            task_dir = input_root / task_name
            if not task_dir.exists() or not task_dir.is_dir():
                continue

            found_task_dirs = True
            for path in collect_csvs_under(task_dir, output_root_resolved, output_inside_input):
                resolved_path = str(path.resolve())
                if resolved_path in seen_paths:
                    continue
                seen_paths.add(resolved_path)
                ordered_csv_files.append(path)

        if found_task_dirs:
            return ordered_csv_files

    return collect_csvs_under(input_root, output_root_resolved, output_inside_input)


def collect_existing_output_csvs(output_root, input_root, csv_files, task_names):
    """Return the CSVs already present in the folders this run writes to.

    Only the task folders (``<output>/<task>/``) and the parent folders of the
    planned outputs are listed, non-recursively, so other subfolders of the
    output root (for example ``player_sequencesy/``) are never
    touched.
    """
    folders = {output_root / task_name for task_name in task_names}
    folders.update((output_root / path.relative_to(input_root)).parent for path in csv_files)
    existing = set()
    for folder in folders:
        if folder.is_dir():
            existing.update(path for path in folder.glob("*.csv") if path.is_file())
    return existing


def read_attempt_if_valid(path, selected_tasks):
    """Read an attempt CSV and return it if it passes the task, vocabulary and clock checks.

    Used for fallback files; returns None when the file is rejected.
    """
    df = pd.read_csv(path, dtype=object, keep_default_na=False)
    task_name = resolve_attempt_task(df)
    if task_name and task_name not in selected_tasks:
        return None
    if not attempt_matches_expected_subtasks(df):
        return None
    if not attempt_elapsed_is_monotonic(df):
        return None
    return df


def main():
    """Run feature derivation across all attempt CSVs in the input folder."""
    args = parse_args()
    input_root = Path(args.input_folder).expanduser().resolve()
    output_root = Path(args.output_folder).expanduser().resolve()
    selected_tasks = tuple(args.tasks)

    if not input_root.exists() or not input_root.is_dir():
        print(
            "ERROR: input folder does not exist or is not a directory: {}".format(input_root),
            file=sys.stderr,
        )
        return 1

    if output_root == input_root:
        print("ERROR: output folder must be different from input folder", file=sys.stderr)
        return 1

    fallback_root = args.fallback_input_folder
    if fallback_root is not None:
        fallback_root = Path(fallback_root).expanduser().resolve()
        if not fallback_root.is_dir():
            print(
                "ERROR: fallback input folder is not a directory: {}".format(fallback_root),
                file=sys.stderr,
            )
            return 1
        if fallback_root in (input_root, output_root):
            print(
                "ERROR: fallback input folder must differ from the input and output folders",
                file=sys.stderr,
            )
            return 1

    output_root.mkdir(parents=True, exist_ok=True)
    write_schema_file(output_root)
    csv_files = list_input_csvs(input_root, output_root, selected_tasks)
    if not csv_files:
        print("No CSV files found under {}".format(input_root))
        return 0

    # Files left by earlier runs would mix with this run's output; remove them on request,
    # otherwise report them at the end.
    existing_outputs = collect_existing_output_csvs(
        output_root, input_root, csv_files, selected_tasks
    )
    if args.clean_output and existing_outputs:
        for path in existing_outputs:
            path.unlink()
        print("Deleted {} existing output CSV(s) (--clean_output)".format(len(existing_outputs)))
        existing_outputs = set()
    written_outputs = set()

    max_session_seconds = args.max_session_minutes * 60
    print(
        "Session duration cutoff: {:.0f} min ({:.0f} s)".format(
            args.max_session_minutes, max_session_seconds
        )
    )
    print("Selected tasks: {}".format(", ".join(selected_tasks)))
    print("Time axis: {}".format(args.time_axis))
    print("Fallback input folder: {}".format(fallback_root or "none"))
    canonical_coords_by_task = load_canonical_subtask_coordinates(str(input_root))

    failures = 0
    skipped_task_filter = 0
    skipped_subtask = 0
    skipped_duration = 0
    skipped_nonmonotonic = 0
    used_fallback = 0
    total = len(csv_files)
    print("Processing {} CSV files from {}".format(total, input_root))

    for index, input_path in enumerate(csv_files, start=1):
        relative_path = input_path.relative_to(input_root)
        output_path = output_root / relative_path
        print("[{}/{}] {}".format(index, total, relative_path))
        try:
            inferred_task = infer_input_task_name(input_path, input_root)
            if inferred_task and inferred_task not in selected_tasks:
                skipped_task_filter += 1
                print("  skipped: task '{}' not selected".format(inferred_task))
                continue

            df_raw = pd.read_csv(input_path, dtype=object, keep_default_na=False)
            from_fallback = False
            task_name = resolve_attempt_task(df_raw)
            if task_name and task_name not in selected_tasks:
                skipped_task_filter += 1
                print("  skipped: task '{}' not selected".format(task_name))
                continue
            if not attempt_matches_expected_subtasks(df_raw):
                skipped_subtask += 1
                print("  skipped: subtask vocabulary does not match expected task")
                continue
            if not attempt_elapsed_is_monotonic(df_raw):
                fallback_path = fallback_root / relative_path if fallback_root is not None else None
                df_fallback = (
                    read_attempt_if_valid(fallback_path, selected_tasks)
                    if fallback_path is not None and fallback_path.is_file()
                    else None
                )
                if df_fallback is None:
                    skipped_nonmonotonic += 1
                    print("  skipped: non-monotonic elapsed time")
                    continue
                # Same attempt without the implicit-gap correction; it still goes through
                # the duration filter below.
                df_raw = df_fallback
                from_fallback = True
                print("  non-monotonic elapsed time: using {}".format(fallback_path))
            duration = session_duration_seconds(df_raw, time_axis=args.time_axis)
            if duration is not None and duration > max_session_seconds:
                skipped_duration += 1
                print(
                    "  skipped: session duration {:.0f}s > {:.0f}s cutoff".format(
                        duration, max_session_seconds
                    )
                )
                continue
            processed = select_training_ready_columns(
                derive_features(
                    df_raw,
                    canonical_coords_by_task=canonical_coords_by_task,
                    time_axis=args.time_axis,
                )
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            processed.to_csv(output_path, index=False)
            written_outputs.add(output_path)
            used_fallback += int(from_fallback)
        except Exception as exc:  # pragma: no cover - CLI safety net
            failures += 1
            print("  ERROR: {}".format(exc), file=sys.stderr)

    if failures:
        print("Finished with {} failed file(s)".format(failures), file=sys.stderr)
        return 1

    print("Skipped {} file(s) outside selected tasks".format(skipped_task_filter))
    print("Skipped {} invalid subtask file(s)".format(skipped_subtask))
    print("Skipped {} file(s) with non-monotonic elapsed time".format(skipped_nonmonotonic))
    print("Wrote {} file(s) derived from the fallback input folder".format(used_fallback))
    print(
        "Skipped {} file(s) exceeding {:.0f}-min duration cutoff".format(
            skipped_duration, args.max_session_minutes
        )
    )
    stale_outputs = sorted(existing_outputs - written_outputs)
    if stale_outputs:
        print(
            "WARNING: {} CSV(s) in the output folders were not written by this run (stale "
            "files from an earlier run, e.g. {}); rerun with --clean_output to remove them".format(
                len(stale_outputs), stale_outputs[0]
            ),
            file=sys.stderr,
        )
    print("Finished. Processed files written to {}".format(output_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
