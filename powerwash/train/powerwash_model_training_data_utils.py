#!/usr/bin/env python3
"""Reusable utilities for PowerWash task-specific transformer training.

Used by the encoder trainer and by the PLVM data utilities. The module:
- discovers processed per-attempt CSV files
- reads the PowerWash feature schema when available
- filters leakage/debug/target columns out of the model inputs
- fits train-only feature metadata for numeric scaling and categorical one-hot encoding
- builds variable-length per-file sequences for transformer training
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from powerwash.train.powerwash_target_utils import TARGET_CHOICES
from plvm import artifact_paths
from plvm.model_training_utils import FILL_MISSING_VALUE


DEFAULT_DATA_DIR = artifact_paths.PLAYER_SEQUENCE_FEATURES_DIR
DEFAULT_SCHEMA_FILENAME = "transformer_feature_schema.json"
# Bump this when feature-engineering logic changes to force cache invalidation.
CACHE_VERSION = 1
MISSING_CATEGORY_TOKEN = "__MISSING__"
UNKNOWN_CATEGORY_TOKEN = "__UNKNOWN__"
PREFERRED_CATEGORICAL_COLUMNS = (
    "event_type",
    "CurrentWasher",
    "CurrentNozzle",
    "CurrentExtension",
    "CrouchState",
)
STATEFUL_CATEGORICAL_COLUMNS = (
    "CurrentWasher",
    "CurrentNozzle",
    "CurrentExtension",
    "CrouchState",
)
STATEFUL_NUMERIC_COLUMNS = (
    "pos_x",
    "pos_y",
    "pos_z",
)
NON_FEATURE_KEYWORDS = ("debug", "leak")
FILENAME_PATTERN = re.compile(r"^(?P<pid>[^_]+)_(?P<task>.+)_attempt(?P<attempt>\d+)\.csv$")


class PowerWashTrainingSchema:
    """Container for PowerWash processed-data schema metadata."""

    def __init__(
        self,
        *,
        schema_path: Optional[Path],
        training_feature_columns: Sequence[str],
        target_columns: Sequence[str],
        leakage_columns: Sequence[str],
        removed_metadata_columns: Optional[Sequence[str]] = None,
        removed_redundant_raw_columns: Optional[Sequence[str]] = None,
    ) -> None:
        self.schema_path = schema_path
        self.training_feature_columns = [str(col) for col in training_feature_columns]
        self.target_columns = [str(col) for col in target_columns]
        self.leakage_columns = [str(col) for col in leakage_columns]
        self.removed_metadata_columns = [str(col) for col in (removed_metadata_columns or [])]
        self.removed_redundant_raw_columns = [
            str(col) for col in (removed_redundant_raw_columns or [])
        ]

    def get_safe_feature_columns(self, target_cols: List[str]) -> List[str]:
        """Return feature columns safe for model input given the requested targets."""
        for col in target_cols:
            if col not in self.target_columns:
                raise ValueError(
                    "Requested target column '{}' not present in schema targets={}".format(
                        col, self.target_columns
                    )
                )

        excluded = set(self.target_columns) | set(self.leakage_columns)
        safe_columns: List[str] = []
        for column in self.training_feature_columns:
            column_lower = str(column).strip().lower()
            if column in excluded:
                continue
            if any(keyword in column_lower for keyword in NON_FEATURE_KEYWORDS):
                continue
            safe_columns.append(column)
        if not safe_columns:
            raise ValueError("No safe feature columns were found for PowerWash training.")
        return safe_columns


class PowerWashSequenceMetadata:
    """Metadata describing one processed player x task-attempt CSV."""

    def __init__(
        self,
        *,
        path: Path,
        relative_path: str,
        sequence_id: str,
        pid: str,
        task_name: str,
        attempt_id: str,
    ) -> None:
        self.path = path
        self.relative_path = relative_path
        self.sequence_id = sequence_id
        self.pid = pid
        self.task_name = task_name
        self.attempt_id = attempt_id


class PowerWashFeatureManifest:
    """Train-only feature metadata used to transform raw CSVs into model inputs."""

    def __init__(
        self,
        *,
        base_feature_columns: Sequence[str],
        numeric_columns: Sequence[str],
        categorical_columns: Sequence[str],
        categorical_levels: Dict[str, Sequence[str]],
        numeric_fill_values: Dict[str, float],
        numeric_mins: Dict[str, float],
        numeric_maxs: Dict[str, float],
        feature_columns: Sequence[str],
        schema_path: Optional[str],
    ) -> None:
        self.base_feature_columns = [str(col) for col in base_feature_columns]
        self.numeric_columns = [str(col) for col in numeric_columns]
        self.categorical_columns = [str(col) for col in categorical_columns]
        self.categorical_levels = {
            str(key): [str(value) for value in values] for key, values in categorical_levels.items()
        }
        self.numeric_fill_values = {
            str(key): float(value) for key, value in numeric_fill_values.items()
        }
        self.numeric_mins = {str(key): float(value) for key, value in numeric_mins.items()}
        self.numeric_maxs = {str(key): float(value) for key, value in numeric_maxs.items()}
        self.feature_columns = [str(col) for col in feature_columns]
        self.schema_path = schema_path

    @property
    def feature_dim(self) -> int:
        """Return the final feature dimensionality."""
        return len(self.feature_columns)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the manifest to a JSON-friendly dictionary."""
        return {
            "base_feature_columns": self.base_feature_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "categorical_levels": self.categorical_levels,
            "numeric_fill_values": self.numeric_fill_values,
            "numeric_mins": self.numeric_mins,
            "numeric_maxs": self.numeric_maxs,
            "feature_columns": self.feature_columns,
            "schema_path": self.schema_path,
        }


class PowerWashSequenceExample:
    """One transformed variable-length training example."""

    def __init__(
        self,
        *,
        metadata: PowerWashSequenceMetadata,
        features: np.ndarray,
        targets: np.ndarray,
        original_length: int,
        effective_length: int,
    ) -> None:
        self.metadata = metadata
        self.features = features
        self.targets = targets
        self.original_length = int(original_length)
        self.effective_length = int(effective_length)


class PowerWashSequenceDataset(Dataset):
    """Torch dataset over transformed PowerWash variable-length sequences."""

    def __init__(self, examples: Sequence[PowerWashSequenceExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        return (
            torch.as_tensor(example.features, dtype=torch.float32),
            torch.as_tensor(example.targets, dtype=torch.float32),
            example.effective_length,
            example.metadata.sequence_id,
        )


def parse_csv_items(raw: str) -> List[str]:
    """Parse a comma-separated CLI string into a list of non-empty items."""
    if not raw:
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def read_powerwash_csv(path: Path) -> pd.DataFrame:
    """Read one processed PowerWash CSV, returning an empty frame for empty/headerless files."""
    try:
        return pd.read_csv(path, dtype=object, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def clean_string_series(series: pd.Series) -> pd.Series:
    """Normalize blank strings to missing values."""
    return series.replace(r"^\s*$", np.nan, regex=True)


def to_numeric_series(series: pd.Series) -> pd.Series:
    """Convert mixed string/bool series to numeric values where possible."""
    if series.dtype == bool:
        return series.astype(float)

    text = series.astype(str).str.strip().str.lower()
    mapped = text.map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric.notna(), mapped)


def resolve_schema_path(data_dir: Path, schema_path: str = "") -> Optional[Path]:
    """Resolve the PowerWash schema path from an explicit override or nearby files."""
    if schema_path:
        resolved = Path(schema_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError("Schema file does not exist: {}".format(resolved))
        return resolved

    candidate = Path(data_dir).expanduser().resolve()
    search_roots = [candidate] + list(candidate.parents[:5])
    for root in search_roots:
        schema_candidate = root / DEFAULT_SCHEMA_FILENAME
        if schema_candidate.exists():
            return schema_candidate
    return None


def load_training_schema(data_dir: Path, schema_path: str = "") -> PowerWashTrainingSchema:
    """Load the PowerWash processed-data schema or fall back to conservative defaults."""
    resolved_schema_path = resolve_schema_path(data_dir, schema_path=schema_path)
    if resolved_schema_path is None:
        logging.warning(
            "No PowerWash schema file found near %s. Falling back to conservative defaults.",
            data_dir,
        )
        return PowerWashTrainingSchema(
            schema_path=None,
            training_feature_columns=[],
            target_columns=list(TARGET_CHOICES),
            leakage_columns=["time_remaining_proxy"],
        )

    with resolved_schema_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return PowerWashTrainingSchema(
        schema_path=resolved_schema_path,
        training_feature_columns=payload.get("training_feature_columns", []),
        target_columns=payload.get("target_columns", []),
        leakage_columns=payload.get("leakage_columns", []),
        removed_metadata_columns=payload.get("removed_metadata_columns", []),
        removed_redundant_raw_columns=payload.get("removed_redundant_raw_columns", []),
    )


def parse_sequence_metadata(path: Path, data_dir: Path) -> PowerWashSequenceMetadata:
    """Parse pid/task/attempt metadata from a processed PowerWash CSV path."""
    relative_path = path.resolve().relative_to(Path(data_dir).expanduser().resolve())
    match = FILENAME_PATTERN.match(path.name)
    if match is None:
        raise ValueError("Unexpected PowerWash filename format: {}".format(path.name))

    task_name = match.group("task")
    if len(relative_path.parts) > 1:
        task_name = relative_path.parts[0]

    return PowerWashSequenceMetadata(
        path=path.resolve(),
        relative_path=str(relative_path),
        sequence_id=str(relative_path.with_suffix("")),
        pid=match.group("pid"),
        task_name=task_name,
        attempt_id=match.group("attempt"),
    )


def list_powerwash_sequence_metadata(
    data_dir: Path,
    task_names: Optional[Sequence[str]] = None,
) -> List[PowerWashSequenceMetadata]:
    """Recursively list processed PowerWash per-attempt CSVs."""
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError("PowerWash data directory does not exist: {}".format(data_dir))

    requested_tasks = set(str(task_name) for task_name in (task_names or []))
    items: List[PowerWashSequenceMetadata] = []
    for path in sorted(data_dir.rglob("*.csv")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        # Skip Jupyter autosave copies.
        if ".ipynb_checkpoints" in path.relative_to(data_dir).parts:
            continue
        try:
            metadata = parse_sequence_metadata(path, data_dir)
        except ValueError:
            continue
        if requested_tasks and metadata.task_name not in requested_tasks:
            continue
        items.append(metadata)

    if not items:
        raise FileNotFoundError(
            "No PowerWash processed CSVs found under {} for task filter {}".format(
                data_dir, sorted(requested_tasks) if requested_tasks else "ALL"
            )
        )
    return items


def _build_split_indices(
    num_groups: int,
    val_fraction: float,
    test_fraction: float,
) -> Tuple[int, int]:
    """Return the number of validation and test groups to allocate."""
    if val_fraction < 0.0 or test_fraction < 0.0:
        raise ValueError("val_fraction and test_fraction must be non-negative.")
    if (val_fraction + test_fraction) >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0.")
    if num_groups < 3:
        raise ValueError(
            "Need at least 3 groups for train/val/test splitting. Found {}".format(num_groups)
        )

    n_test = int(round(test_fraction * num_groups)) if test_fraction > 0 else 0
    n_val = int(round(val_fraction * num_groups)) if val_fraction > 0 else 0

    if test_fraction > 0:
        n_test = max(1, n_test)
    if val_fraction > 0:
        n_val = max(1, n_val)

    while n_test + n_val >= num_groups:
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
        else:
            break

    if test_fraction > 0 and n_test == 0:
        raise ValueError("Unable to allocate a non-empty test split.")
    if val_fraction > 0 and n_val == 0:
        raise ValueError("Unable to allocate a non-empty validation split.")

    return n_val, n_test


def split_sequence_metadata(
    items: Sequence[PowerWashSequenceMetadata],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    split_group: str = "player",
) -> Tuple[
    List[PowerWashSequenceMetadata],
    List[PowerWashSequenceMetadata],
    List[PowerWashSequenceMetadata],
]:
    """Split PowerWash sequence metadata into train/val/test partitions."""
    items = list(items)
    if len(items) < 3:
        raise ValueError("Need at least 3 sequences to build train/val/test splits.")

    if split_group not in {"player", "sequence"}:
        raise ValueError("split_group must be one of {'player', 'sequence'}.")

    group_to_items: Dict[str, List[PowerWashSequenceMetadata]] = {}
    for item in items:
        group_key = item.pid if split_group == "player" else item.sequence_id
        group_to_items.setdefault(group_key, []).append(item)

    unique_groups = sorted(group_to_items)
    if split_group == "player" and len(unique_groups) < 3:
        logging.warning(
            "Only %d unique players available. Falling back to sequence-level split.",
            len(unique_groups),
        )
        return split_sequence_metadata(
            items,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
            split_group="sequence",
        )

    n_val, n_test = _build_split_indices(len(unique_groups), val_fraction, test_fraction)
    rng = np.random.RandomState(seed)
    shuffled_groups = np.asarray(unique_groups, dtype=object)
    rng.shuffle(shuffled_groups)

    test_groups = set(shuffled_groups[:n_test].tolist())
    val_groups = set(shuffled_groups[n_test : n_test + n_val].tolist())
    train_groups = set(shuffled_groups[n_test + n_val :].tolist())

    train_items: List[PowerWashSequenceMetadata] = []
    val_items: List[PowerWashSequenceMetadata] = []
    test_items: List[PowerWashSequenceMetadata] = []

    for item in items:
        group_key = item.pid if split_group == "player" else item.sequence_id
        if group_key in test_groups:
            test_items.append(item)
        elif group_key in val_groups:
            val_items.append(item)
        elif group_key in train_groups:
            train_items.append(item)

    if not train_items or not val_items or not test_items:
        raise ValueError(
            "Split produced an empty partition: train={} val={} test={}".format(
                len(train_items), len(val_items), len(test_items)
            )
        )

    return train_items, val_items, test_items


def filter_model_ready_sequence_metadata(
    items: Sequence[PowerWashSequenceMetadata],
    target_cols: List[str],
) -> Tuple[List[PowerWashSequenceMetadata], Dict[str, int]]:
    """Drop empty/no-target sequences before train/val/test splitting."""
    kept_items: List[PowerWashSequenceMetadata] = []
    stats = {
        "n_sequence_files": int(len(items)),
        "n_sequences_kept": 0,
        "n_sequences_skipped_empty": 0,
        "n_sequences_skipped_no_target": 0,
    }

    for item in items:
        df = read_powerwash_csv(item.path)
        if df.empty:
            stats["n_sequences_skipped_empty"] += 1
            continue
        present_cols = [col for col in target_cols if col in df.columns]
        if not present_cols:
            stats["n_sequences_skipped_no_target"] += 1
            continue
        has_any_valid = any(
            bool(pd.to_numeric(df[col], errors="coerce").notna().any()) for col in present_cols
        )
        if not has_any_valid:
            stats["n_sequences_skipped_no_target"] += 1
            continue
        kept_items.append(item)

    stats["n_sequences_kept"] = int(len(kept_items))
    if not kept_items:
        raise ValueError(
            "No PowerWash sequence files with valid targets {} were found.".format(target_cols)
        )
    return kept_items, stats


def _ensure_dataframe_columns(df: pd.DataFrame, required_columns: Sequence[str]) -> pd.DataFrame:
    """Ensure expected columns exist so downstream preprocessing stays conservative."""
    out = df.copy()
    for column in required_columns:
        if column not in out.columns:
            out[column] = ""
    return out


def _sort_sequence_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Sort a processed sequence conservatively by elapsed time when available."""
    if "time_elapsed_since_job_start" not in df.columns:
        return df.reset_index(drop=True)

    ordered = df.copy()
    ordered["_sequence_order"] = np.arange(len(ordered))
    ordered["_elapsed_seconds"] = pd.to_numeric(
        ordered["time_elapsed_since_job_start"],
        errors="coerce",
    )
    ordered = ordered.sort_values(
        by=["_elapsed_seconds", "_sequence_order"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    return ordered.drop(columns=["_sequence_order", "_elapsed_seconds"])


def _prepare_numeric_feature(series: pd.Series, column_name: str) -> pd.Series:
    """Prepare one numeric feature column prior to fill/scaling."""
    numeric = to_numeric_series(series)
    if column_name in STATEFUL_NUMERIC_COLUMNS:
        numeric = numeric.ffill().bfill()
    return numeric.astype("float32")


def _prepare_categorical_feature(series: pd.Series, column_name: str) -> pd.Series:
    """Prepare one categorical feature column prior to one-hot encoding."""
    cleaned = clean_string_series(series).astype(object)
    if column_name in STATEFUL_CATEGORICAL_COLUMNS:
        cleaned = cleaned.ffill().bfill()
    return cleaned.fillna(MISSING_CATEGORY_TOKEN).astype(str)


def _fit_numeric_statistics(
    train_items: Sequence[PowerWashSequenceMetadata],
    numeric_columns: Sequence[str],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Fit numeric fill/min/max statistics from the training split only."""
    numeric_values: Dict[str, List[np.ndarray]] = {column: [] for column in numeric_columns}

    for item in train_items:
        df = read_powerwash_csv(item.path)
        df = _ensure_dataframe_columns(df, numeric_columns)
        df = _sort_sequence_dataframe(df)
        for column in numeric_columns:
            values = _prepare_numeric_feature(df[column], column)
            finite_values = values[np.isfinite(values.to_numpy())]
            if finite_values.empty:
                continue
            numeric_values[column].append(finite_values.to_numpy(dtype=np.float32))

    fill_values: Dict[str, float] = {}
    min_values: Dict[str, float] = {}
    max_values: Dict[str, float] = {}
    for column in numeric_columns:
        if numeric_values[column]:
            concatenated = np.concatenate(numeric_values[column]).astype(np.float32)
            fill_values[column] = float(np.median(concatenated))
            min_values[column] = float(np.min(concatenated))
            max_values[column] = float(np.max(concatenated))
        else:
            fill_values[column] = 0.0
            min_values[column] = 0.0
            max_values[column] = 1.0
    return fill_values, min_values, max_values


def _fit_categorical_levels(
    train_items: Sequence[PowerWashSequenceMetadata],
    categorical_columns: Sequence[str],
) -> Dict[str, List[str]]:
    """Collect categorical levels from the training split only."""
    categorical_values: Dict[str, set] = {
        column: {MISSING_CATEGORY_TOKEN, UNKNOWN_CATEGORY_TOKEN} for column in categorical_columns
    }

    for item in train_items:
        df = read_powerwash_csv(item.path)
        df = _ensure_dataframe_columns(df, categorical_columns)
        df = _sort_sequence_dataframe(df)
        for column in categorical_columns:
            values = _prepare_categorical_feature(df[column], column)
            categorical_values[column].update(values.astype(str).tolist())

    categorical_levels: Dict[str, List[str]] = {}
    for column, values in categorical_values.items():
        stable_values = sorted(
            value
            for value in values
            if value not in {MISSING_CATEGORY_TOKEN, UNKNOWN_CATEGORY_TOKEN}
        )
        categorical_levels[column] = stable_values + [
            MISSING_CATEGORY_TOKEN,
            UNKNOWN_CATEGORY_TOKEN,
        ]
    return categorical_levels


def fit_powerwash_feature_manifest(
    train_items: Sequence[PowerWashSequenceMetadata],
    schema: PowerWashTrainingSchema,
    target_cols: List[str],
) -> PowerWashFeatureManifest:
    """Fit a train-only feature manifest for PowerWash model inputs."""
    if not train_items:
        raise ValueError("Cannot fit PowerWash feature manifest with no training items.")

    base_feature_columns = schema.get_safe_feature_columns(target_cols)
    if not base_feature_columns and schema.schema_path is None:
        raise ValueError(
            "PowerWash schema is required when training_feature_columns are unavailable."
        )

    categorical_columns = [
        column for column in base_feature_columns if column in PREFERRED_CATEGORICAL_COLUMNS
    ]
    numeric_columns = [
        column for column in base_feature_columns if column not in categorical_columns
    ]

    numeric_fill_values, numeric_mins, numeric_maxs = _fit_numeric_statistics(
        train_items,
        numeric_columns,
    )
    categorical_levels = _fit_categorical_levels(train_items, categorical_columns)

    feature_columns: List[str] = list(numeric_columns)
    for column in categorical_columns:
        for level in categorical_levels[column]:
            feature_columns.append("{}__{}".format(column, level))

    return PowerWashFeatureManifest(
        base_feature_columns=base_feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        categorical_levels=categorical_levels,
        numeric_fill_values=numeric_fill_values,
        numeric_mins=numeric_mins,
        numeric_maxs=numeric_maxs,
        feature_columns=feature_columns,
        schema_path=str(schema.schema_path) if schema.schema_path is not None else None,
    )


def _scale_numeric_column(
    values: np.ndarray, fill_value: float, min_value: float, max_value: float
) -> np.ndarray:
    """Fill and min-max scale one numeric feature column."""
    filled = np.where(np.isfinite(values), values, fill_value).astype(np.float32)
    range_value = float(max_value - min_value)
    if not np.isfinite(range_value) or range_value <= 0.0:
        return np.zeros_like(filled, dtype=np.float32)
    scaled = (filled - float(min_value)) / range_value
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def transform_powerwash_sequence(
    metadata: PowerWashSequenceMetadata,
    manifest: PowerWashFeatureManifest,
    target_cols: List[str],
    max_seq_length: int = 0,
) -> Optional[PowerWashSequenceExample]:
    """Load and transform one PowerWash processed CSV into a training example."""
    required_columns = list(manifest.base_feature_columns) + list(target_cols)
    df = read_powerwash_csv(metadata.path)
    original_length = int(len(df))
    if df.empty:
        return None

    df = _ensure_dataframe_columns(df, required_columns)
    df = _sort_sequence_dataframe(df)

    target_df = pd.DataFrame({col: pd.to_numeric(df[col], errors="coerce") for col in target_cols})
    valid_target_mask = target_df.notna().any(axis=1)
    if not valid_target_mask.any():
        return None

    df = df.loc[valid_target_mask].reset_index(drop=True)
    target_array = target_df.loc[valid_target_mask].to_numpy(dtype=np.float32)

    if max_seq_length and max_seq_length > 0 and len(df) > max_seq_length:
        df = df.iloc[-max_seq_length:].reset_index(drop=True)
        target_array = target_array[-max_seq_length:]

    feature_blocks: List[np.ndarray] = []

    for column in manifest.numeric_columns:
        values = _prepare_numeric_feature(df[column], column).to_numpy(dtype=np.float32)
        scaled = _scale_numeric_column(
            values,
            fill_value=manifest.numeric_fill_values[column],
            min_value=manifest.numeric_mins[column],
            max_value=manifest.numeric_maxs[column],
        )
        feature_blocks.append(scaled.reshape(-1, 1))

    for column in manifest.categorical_columns:
        prepared = _prepare_categorical_feature(df[column], column)
        known_levels = set(manifest.categorical_levels[column])
        normalized = prepared.where(
            prepared.astype(str).isin(known_levels),
            other=UNKNOWN_CATEGORY_TOKEN,
        )
        categorical = pd.Categorical(
            normalized,
            categories=manifest.categorical_levels[column],
        )
        encoded = pd.get_dummies(categorical)
        encoded = encoded.reindex(columns=manifest.categorical_levels[column], fill_value=0)
        feature_blocks.append(encoded.to_numpy(dtype=np.float32))

    if not feature_blocks:
        raise ValueError(
            "No feature blocks were generated for sequence {}".format(metadata.sequence_id)
        )

    features = np.concatenate(feature_blocks, axis=1).astype(np.float32)

    return PowerWashSequenceExample(
        metadata=metadata,
        features=features,
        targets=target_array,
        original_length=original_length,
        effective_length=int(features.shape[0]),
    )


def build_powerwash_sequence_examples(
    items: Sequence[PowerWashSequenceMetadata],
    manifest: PowerWashFeatureManifest,
    target_cols: List[str],
    max_seq_length: int = 0,
) -> Tuple[List[PowerWashSequenceExample], Dict[str, int]]:
    """Transform many PowerWash sequence files into model-ready examples."""
    examples: List[PowerWashSequenceExample] = []
    stats = {
        "n_sequence_files": int(len(items)),
        "n_sequences_loaded": 0,
        "n_sequences_skipped_empty": 0,
        "n_sequences_skipped_no_target": 0,
        "n_sequences_truncated": 0,
        "n_rows_original": 0,
        "n_rows_retained": 0,
    }

    for item in items:
        raw_df = read_powerwash_csv(item.path)
        stats["n_rows_original"] += int(len(raw_df))
        if raw_df.empty:
            stats["n_sequences_skipped_empty"] += 1
            continue

        example = transform_powerwash_sequence(
            item,
            manifest=manifest,
            target_cols=target_cols,
            max_seq_length=max_seq_length,
        )
        if example is None:
            stats["n_sequences_skipped_no_target"] += 1
            continue

        if max_seq_length and max_seq_length > 0 and example.original_length > max_seq_length:
            stats["n_sequences_truncated"] += 1

        stats["n_sequences_loaded"] += 1
        stats["n_rows_retained"] += int(example.effective_length)
        examples.append(example)

    if not examples:
        raise ValueError(
            "No model-ready PowerWash sequences were produced for targets {}.".format(target_cols)
        )

    return examples, stats


def powerwash_sequence_collate_fn(batch):
    """Pad a batch of variable-length PowerWash sequences for transformer training."""
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    max_length = max(int(item[2]) for item in batch)
    feature_dim = int(batch[0][0].shape[1])
    target_dim = int(batch[0][1].shape[1])

    features = torch.full(
        (len(batch), max_length, feature_dim),
        fill_value=float(FILL_MISSING_VALUE),
        dtype=torch.float32,
    )
    targets = torch.full(
        (len(batch), max_length, target_dim),
        fill_value=float("nan"),
        dtype=torch.float32,
    )
    lengths = torch.zeros(len(batch), dtype=torch.long)
    sequence_ids: List[str] = []

    for index, (feature_tensor, target_tensor, seq_length, sequence_id) in enumerate(batch):
        seq_length = int(seq_length)
        features[index, :seq_length] = feature_tensor[:seq_length]
        targets[index, :seq_length] = target_tensor[:seq_length]
        lengths[index] = seq_length
        sequence_ids.append(sequence_id)

    return features, targets, lengths, sequence_ids


def estimate_recommended_context_length(examples: Sequence[PowerWashSequenceExample]) -> int:
    """Estimate a reasonable sequence length from observed effective lengths."""
    lengths = np.asarray([example.effective_length for example in examples], dtype=np.int32)
    if lengths.size == 0:
        return 0
    return int(np.percentile(lengths, 75))


def save_feature_manifest(manifest: PowerWashFeatureManifest, output_path: Path) -> None:
    """Write the PowerWash feature manifest to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Training-data cache helpers
# ---------------------------------------------------------------------------


def compute_powerwash_data_cache_key(
    all_items: Sequence[PowerWashSequenceMetadata],
    target_cols: List[str],
    max_seq_length: int,
    val_fraction: float,
    test_fraction: float,
    split_group: str,
    seed: int,
    max_files: Optional[int] = None,
    schema_path: Optional[Path] = None,
) -> str:
    """Compute a deterministic SHA-256 key for the processed training data.

    The key captures every input that determines the processed examples: the
    CSV file contents (via mtime + size), the schema file, the
    feature-engineering hyperparameters, and the split configuration.
    Increment CACHE_VERSION when feature-engineering logic changes.
    """
    file_fingerprints: List[Tuple[str, int, int]] = []
    for item in sorted(all_items, key=lambda x: x.sequence_id):
        try:
            stat = item.path.stat()
            file_fingerprints.append((item.sequence_id, int(stat.st_mtime), int(stat.st_size)))
        except OSError:
            file_fingerprints.append((item.sequence_id, -1, -1))

    schema_fingerprint: List = ["", -1, -1]
    if schema_path is not None:
        try:
            stat = Path(schema_path).stat()
            schema_fingerprint = [str(schema_path), int(stat.st_mtime), int(stat.st_size)]
        except OSError:
            schema_fingerprint = [str(schema_path), -1, -1]

    key_data = json.dumps(
        {
            "cache_version": CACHE_VERSION,
            "target_cols": sorted(target_cols),
            "max_seq_length": int(max_seq_length),
            "val_fraction": float(val_fraction),
            "test_fraction": float(test_fraction),
            "split_group": str(split_group),
            "seed": int(seed),
            "max_files": int(max_files) if max_files is not None else -1,
            "schema": schema_fingerprint,
            "files": file_fingerprints,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(key_data.encode("utf-8")).hexdigest()


def _save_split_to_cache(
    cache_dir: Path,
    examples: List[PowerWashSequenceExample],
    split_name: str,
) -> None:
    """Persist one split's examples as a JSON metadata file and a compressed NPZ."""
    if not examples:
        metadata_list: List[Dict[str, Any]] = []
        features_concat = np.zeros((0, 1), dtype=np.float32)
        targets_concat = np.zeros((0, 1), dtype=np.float32)
        effective_lengths = np.array([], dtype=np.int64)
        original_lengths = np.array([], dtype=np.int64)
    else:
        metadata_list = [
            {
                "sequence_id": ex.metadata.sequence_id,
                "pid": ex.metadata.pid,
                "task_name": ex.metadata.task_name,
                "attempt_id": ex.metadata.attempt_id,
                "relative_path": ex.metadata.relative_path,
                "path": str(ex.metadata.path),
            }
            for ex in examples
        ]
        features_concat = np.concatenate([ex.features for ex in examples], axis=0)
        targets_concat = np.concatenate([ex.targets for ex in examples], axis=0)
        effective_lengths = np.array([ex.effective_length for ex in examples], dtype=np.int64)
        original_lengths = np.array([ex.original_length for ex in examples], dtype=np.int64)

    with (cache_dir / "{}_metadata.json".format(split_name)).open("w", encoding="utf-8") as handle:
        json.dump(metadata_list, handle)
    np.savez_compressed(
        str(cache_dir / "{}_arrays.npz".format(split_name)),
        features_concat=features_concat,
        targets_concat=targets_concat,
        effective_lengths=effective_lengths,
        original_lengths=original_lengths,
    )


def _load_split_from_cache(
    cache_dir: Path,
    split_name: str,
) -> List[PowerWashSequenceExample]:
    """Reconstruct one split's examples from its cache files."""
    with (cache_dir / "{}_metadata.json".format(split_name)).open("r", encoding="utf-8") as handle:
        metadata_list: List[Dict[str, Any]] = json.load(handle)

    data = np.load(str(cache_dir / "{}_arrays.npz".format(split_name)))
    features_concat: np.ndarray = data["features_concat"]
    targets_concat: np.ndarray = data["targets_concat"]
    effective_lengths: List[int] = data["effective_lengths"].tolist()
    original_lengths: List[int] = data["original_lengths"].tolist()

    examples: List[PowerWashSequenceExample] = []
    offset = 0
    for i, meta in enumerate(metadata_list):
        eff_len = int(effective_lengths[i])
        orig_len = int(original_lengths[i])
        features = features_concat[offset : offset + eff_len].copy()
        targets = targets_concat[offset : offset + eff_len].copy()
        offset += eff_len
        seq_metadata = PowerWashSequenceMetadata(
            path=Path(meta["path"]),
            relative_path=meta["relative_path"],
            sequence_id=meta["sequence_id"],
            pid=meta["pid"],
            task_name=meta["task_name"],
            attempt_id=meta["attempt_id"],
        )
        examples.append(
            PowerWashSequenceExample(
                metadata=seq_metadata,
                features=features,
                targets=targets,
                original_length=orig_len,
                effective_length=eff_len,
            )
        )
    return examples


def save_training_data_cache(
    cache_dir: Path,
    cache_key: str,
    manifest: PowerWashFeatureManifest,
    train_examples: List[PowerWashSequenceExample],
    val_examples: List[PowerWashSequenceExample],
    test_examples: List[PowerWashSequenceExample],
    train_stats: Dict[str, int],
    val_stats: Dict[str, int],
    test_stats: Dict[str, int],
) -> None:
    """Write the full processed training data to a cache directory for later reuse."""
    split_dir = Path(cache_dir) / cache_key[:16]
    split_dir.mkdir(parents=True, exist_ok=True)

    with (split_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2)

    for split_name, examples in [
        ("train", train_examples),
        ("val", val_examples),
        ("test", test_examples),
    ]:
        _save_split_to_cache(split_dir, examples, split_name)

    with (split_dir / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "cache_key": cache_key,
                "train_stats": train_stats,
                "val_stats": val_stats,
                "test_stats": test_stats,
            },
            handle,
            indent=2,
        )

    logging.info(
        "Saved training data cache to %s (train=%d val=%d test=%d)",
        split_dir,
        len(train_examples),
        len(val_examples),
        len(test_examples),
    )


def load_training_data_cache(
    cache_dir: Path,
    cache_key: str,
) -> Optional[
    Tuple[
        PowerWashFeatureManifest,
        List[PowerWashSequenceExample],
        List[PowerWashSequenceExample],
        List[PowerWashSequenceExample],
        Dict[str, int],
        Dict[str, int],
        Dict[str, int],
    ]
]:
    """Load cached training data, or return None on cache miss or load error."""
    split_dir = Path(cache_dir) / cache_key[:16]

    required = [
        "manifest.json",
        "stats.json",
        "train_metadata.json",
        "train_arrays.npz",
        "val_metadata.json",
        "val_arrays.npz",
        "test_metadata.json",
        "test_arrays.npz",
    ]
    if not all((split_dir / f).exists() for f in required):
        return None

    try:
        with (split_dir / "manifest.json").open("r", encoding="utf-8") as handle:
            manifest_dict = json.load(handle)
        manifest = PowerWashFeatureManifest(
            base_feature_columns=manifest_dict["base_feature_columns"],
            numeric_columns=manifest_dict["numeric_columns"],
            categorical_columns=manifest_dict["categorical_columns"],
            categorical_levels=manifest_dict["categorical_levels"],
            numeric_fill_values=manifest_dict["numeric_fill_values"],
            numeric_mins=manifest_dict["numeric_mins"],
            numeric_maxs=manifest_dict["numeric_maxs"],
            feature_columns=manifest_dict["feature_columns"],
            schema_path=manifest_dict.get("schema_path"),
        )

        train_examples = _load_split_from_cache(split_dir, "train")
        val_examples = _load_split_from_cache(split_dir, "val")
        test_examples = _load_split_from_cache(split_dir, "test")

        with (split_dir / "stats.json").open("r", encoding="utf-8") as handle:
            stats_data = json.load(handle)

        logging.info(
            "Loaded training data cache from %s (train=%d val=%d test=%d)",
            split_dir,
            len(train_examples),
            len(val_examples),
            len(test_examples),
        )
        return (
            manifest,
            train_examples,
            val_examples,
            test_examples,
            stats_data["train_stats"],
            stats_data["val_stats"],
            stats_data["test_stats"],
        )

    except Exception as exc:  # noqa: BLE001
        logging.warning("Cache load failed (%s); rebuilding from source.", exc)
        return None
