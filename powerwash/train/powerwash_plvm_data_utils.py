#!/usr/bin/env python3
"""Data utilities for the PowerWash PLVM and its embedding-based baselines.

Builds labelled sample candidates, applies each frozen task encoder's saved feature
manifest to the processed sequence CSVs, and computes (and caches) the frozen-encoder
embeddings that the PLVM and its baselines are trained on. With a ``player_label_*``
target every player contributes one sample per cutoff in ``BOTH_PARTIAL_CUTOFF_SECONDS``;
with ``partial_target`` one sample per row of the partial task. Model types are described
in ``train_plvm_powerwash.py``.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from powerwash.train.powerwash_model_training_data_utils import (
    FILENAME_PATTERN,
    PowerWashSequenceMetadata,
    UNKNOWN_CATEGORY_TOKEN,
    PowerWashFeatureManifest,
    _ensure_dataframe_columns,
    _prepare_categorical_feature,
    _prepare_numeric_feature,
    _scale_numeric_column,
    _sort_sequence_dataframe,
    list_powerwash_sequence_metadata,
    read_powerwash_csv,
)
from plvm import artifact_paths
from plvm.model_training_utils import PAD_VALUE, _sample_train_indices_by_id, precompute_embeddings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLVM_EMBEDDING_CACHE_VERSION = 7
TARGET_MODE_PARTIAL_TARGET = "partial_target"
TARGET_MODE_PLAYER_LABEL = "player_label"

MODEL_TYPE_BOTH_PARTIAL = "plvm_both_partial"
# Model types that require both full_task_name and partial_task_name.
DUAL_TASK_MODEL_TYPES = frozenset({"plvm_full_partial", MODEL_TYPE_BOTH_PARTIAL})

# Fixed observation cutoffs (in elapsed seconds) used by plvm_both_partial and
# aligned single_task player-label runs. The default schedule is every 5
# minutes from 5 to 120 minutes inclusive. Each player/event generates one
# candidate per cutoff; all are used in the same run.
BOTH_PARTIAL_CUTOFF_SECONDS: Tuple[int, ...] = tuple(range(5 * 60, 121 * 60, 5 * 60))
TARGET_MODE_PLAYER_LABEL_2_CLASS = "player_label_2_class"
TARGET_MODE_PLAYER_LABEL_3_CLASS = "player_label_3_class"
TARGET_MODE_PLAYER_LABEL_2_CLASS_CLEAN_ZONEHOPPING = "player_label_2_class_clean_zonehopping"
TARGET_MODE_PLAYER_LABEL_2_CLASS_ZONEHOPPING = "player_label_2_class_zonehopping"
TARGET_MODE_PLAYER_LABEL_2_CLASS_CLEAN_ENTROPY = "player_label_2_class_clean_entropy"
TARGET_MODE_PLAYER_LABEL_3_CLASS_ZONEHOPPING = "player_label_3_class_zonehopping"
TARGET_MODE_PLAYER_LABEL_2_CLASS_ENTROPY = "player_label_2_class_entropy"
TARGET_MODE_PLAYER_LABEL_3_CLASS_ENTROPY = "player_label_3_class_entropy"

PLAYER_LABEL_2_CLASS_NAMES = ["Zone Planner", "Zone Hopper"]
PLAYER_LABEL_3_CLASS_NAMES = ["Zone Planner", "Mixed", "Zone Hopper"]

PLAYER_LABEL_2_CLASS_TARGET_MODES = frozenset(
    {
        TARGET_MODE_PLAYER_LABEL_2_CLASS,
        TARGET_MODE_PLAYER_LABEL_2_CLASS_CLEAN_ZONEHOPPING,
        TARGET_MODE_PLAYER_LABEL_2_CLASS_ZONEHOPPING,
        TARGET_MODE_PLAYER_LABEL_2_CLASS_CLEAN_ENTROPY,
        TARGET_MODE_PLAYER_LABEL_2_CLASS_ENTROPY,
    }
)
PLAYER_LABEL_3_CLASS_TARGET_MODES = frozenset(
    {
        TARGET_MODE_PLAYER_LABEL_3_CLASS,
        TARGET_MODE_PLAYER_LABEL_3_CLASS_ZONEHOPPING,
        TARGET_MODE_PLAYER_LABEL_3_CLASS_ENTROPY,
    }
)

ALL_PLAYER_LABEL_MODES = frozenset({TARGET_MODE_PLAYER_LABEL}) | (
    PLAYER_LABEL_2_CLASS_TARGET_MODES | PLAYER_LABEL_3_CLASS_TARGET_MODES
)


def resolve_powerwash_player_label_class_names(target_mode: str) -> Optional[List[str]]:
    """Return fixed class names for supported PowerWash player-label targets.

    ``player_label`` remains free-form and returns ``None`` so callers can use
    ``--player_label_classes`` explicitly. All fixed-schema zone-planning
    variants share the same 2-class or 3-class label names.
    """
    normalized_target_mode = str(target_mode or "").strip().lower()
    if normalized_target_mode in PLAYER_LABEL_2_CLASS_TARGET_MODES:
        return list(PLAYER_LABEL_2_CLASS_NAMES)
    if normalized_target_mode in PLAYER_LABEL_3_CLASS_TARGET_MODES:
        return list(PLAYER_LABEL_3_CLASS_NAMES)
    return None


# Column name in the processed sequence CSVs that carries elapsed time.
ELAPSED_TIME_COL = "time_elapsed_since_job_start"

# Which observed-task attempts a player contributes. "first" keeps only attempt 1 of each
# task: later attempts may have been played after the target task, and keeping one attempt
# gives every player the same weight. "all" uses every attempt file found.
ATTEMPT_POLICY_FIRST = "first"
ATTEMPT_POLICY_ALL = "all"
ATTEMPT_POLICIES = (ATTEMPT_POLICY_FIRST, ATTEMPT_POLICY_ALL)


@dataclass(frozen=True)
class PowerWashPLVMSampleCandidate:
    """One candidate labeled cutoff before any embedding computation."""

    pid: str
    task_name: str
    attempt_number: str
    cutoff_seconds: float
    target_value: float
    source_row_index: int
    source_type: str

    def to_dict(self) -> dict:
        """Serialize the candidate to a JSON-friendly dictionary."""
        return asdict(self)


def canonicalize_dual_task_configuration(
    model_type: str,
    full_task_name: str,
    partial_task_name: str,
) -> Tuple[str, str]:
    """Return canonical full/partial task names for symmetric dual-task runs.

    ``plvm_both_partial`` treats the two tasks as an unordered pair: the lexicographically
    smaller task becomes ``partial_task_name`` and the larger ``full_task_name``.
    ``plvm_full_partial`` is directional and returned unchanged.
    """
    normalized_full = str(full_task_name or "").strip()
    normalized_partial = str(partial_task_name or "").strip()
    if str(model_type or "").strip().lower() != MODEL_TYPE_BOTH_PARTIAL:
        return normalized_full, normalized_partial
    if not normalized_full or not normalized_partial or normalized_full == normalized_partial:
        return normalized_full, normalized_partial
    ordered = sorted([normalized_partial, normalized_full])
    return ordered[1], ordered[0]


def load_player_label_lookup(
    label_csv_path: Path,
    player_id_col: str = "player_id",
    label_col: str = "label",
    allowed_labels: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    """Load a player-level label CSV into a player -> class-index lookup.

    Rows with empty labels are skipped. If an ``excluded`` column is present,
    rows marked true are skipped as well.
    """
    label_csv_path = Path(label_csv_path)
    if not label_csv_path.exists():
        raise FileNotFoundError("Player-label CSV not found: {}".format(label_csv_path))

    df = pd.read_csv(label_csv_path, dtype=object, keep_default_na=False)
    required_cols = {str(player_id_col), str(label_col)}
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "Player-label CSV missing required column(s) {}: {}".format(
                missing_cols,
                label_csv_path,
            )
        )

    ordered_labels = [str(label).strip() for label in (allowed_labels or []) if str(label).strip()]
    if not ordered_labels:
        raise ValueError("allowed_labels must contain at least one non-empty label.")
    label_to_index = {label: idx for idx, label in enumerate(ordered_labels)}
    if len(label_to_index) != len(ordered_labels):
        raise ValueError("allowed_labels contains duplicates: {}".format(ordered_labels))

    excluded_mask = pd.Series(False, index=df.index)
    if "excluded" in df.columns:
        excluded_mask = (
            df["excluded"].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})
        )

    lookup: Dict[str, int] = {}
    skip_reasons: Dict[str, int] = defaultdict(int)
    for _, row in df.iterrows():
        pid = str(row.get(player_id_col, "") or "").strip()
        if not pid:
            skip_reasons["missing_pid"] += 1
            continue

        if bool(excluded_mask.loc[row.name]):
            skip_reasons["excluded"] += 1
            continue

        raw_label = str(row.get(label_col, "") or "").strip()
        if not raw_label:
            skip_reasons["empty_label"] += 1
            continue
        if raw_label not in label_to_index:
            skip_reasons["label_not_allowed:{}".format(raw_label)] += 1
            continue

        label_index = int(label_to_index[raw_label])
        if pid in lookup and lookup[pid] != label_index:
            raise ValueError(
                "Conflicting labels for player '{}': {} vs {}.".format(
                    pid,
                    lookup[pid],
                    label_index,
                )
            )
        lookup[pid] = label_index

    logging.info(
        "Loaded %d player labels from %s with class order=%s",
        len(lookup),
        label_csv_path,
        ordered_labels,
    )
    if skip_reasons:
        top_reasons = sorted(skip_reasons.items(), key=lambda kv: -kv[1])[:8]
        logging.info("Player-label CSV skipped rows: %s", dict(top_reasons))
    return lookup


# ---------------------------------------------------------------------------
# Model spec / loading
# ---------------------------------------------------------------------------


def load_task_model_spec(spec_path: Path, project_root: Path) -> Dict[str, dict]:
    """Load the task-model spec JSON into ``{task_name: {"model_log_dir": Path, "seq_length": int}}``.

    ``tasks`` maps each task either to a run name (resolved under
    ``models/powerwash/task_specific/powerwash_<task>/training_logs/``, as in
    ``models/powerwash/pw_task_specific_next_subtask.json``) or to
    ``{"model_log_dir": ..., "seq_length": ...}``. A top-level ``seq_length``
    (default 500) applies to entries that do not set one.
    """
    spec_path = Path(spec_path)
    if not spec_path.exists():
        raise FileNotFoundError("Task model spec JSON not found: {}".format(spec_path))

    with spec_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    global_seq_length = int(payload.get("seq_length", 500))
    tasks_raw = payload.get("tasks", {})

    resolved: Dict[str, dict] = {}
    for task_name, entry in tasks_raw.items():
        if isinstance(entry, str):
            # Short format: the entry is just the run name.
            model_name = entry
            model_log_dir = (
                Path(project_root)
                / artifact_paths.POWERWASH_TASK_SPECIFIC_MODELS_DIR.relative_to(
                    artifact_paths.ARTIFACT_ROOT
                )
                / "powerwash_{}".format(task_name)
                / "training_logs"
                / model_name
            )
            seq_length = global_seq_length
        elif isinstance(entry, dict):
            raw_dir = entry.get("model_log_dir", "")
            candidate = Path(raw_dir)
            # Explicit entries may carry an absolute path recorded at training time.
            model_log_dir = artifact_paths.rebase_recorded_artifact_path(
                candidate if candidate.is_absolute() else Path(project_root) / candidate
            )
            seq_length = int(entry.get("seq_length", global_seq_length))
        else:
            logging.warning("Unrecognised entry for task '%s' in spec — skipping.", task_name)
            continue

        resolved[task_name] = {
            "model_log_dir": model_log_dir,
            "seq_length": seq_length,
        }

    return resolved


def _load_feature_manifest_from_dir(model_log_dir: Path) -> PowerWashFeatureManifest:
    """Load the saved PowerWash feature manifest from a model log directory."""
    manifest_path = model_log_dir / "feature_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("feature_manifest.json not found in: {}".format(model_log_dir))
    with manifest_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    return PowerWashFeatureManifest(
        base_feature_columns=payload["base_feature_columns"],
        numeric_columns=payload["numeric_columns"],
        categorical_columns=payload["categorical_columns"],
        categorical_levels=payload["categorical_levels"],
        numeric_fill_values=payload["numeric_fill_values"],
        numeric_mins=payload["numeric_mins"],
        numeric_maxs=payload["numeric_maxs"],
        feature_columns=payload["feature_columns"],
        schema_path=payload.get("schema_path"),
    )


def _load_model_pth(model_log_dir: Path) -> torch.nn.Module:
    """Load the saved transformer model from a model log directory."""
    from plvm.model_training_utils import COMPUTE_DEVICE, load_model  # noqa: PLC0415

    pth_files = sorted(model_log_dir.glob("*_model.pth"))
    if not pth_files:
        raise FileNotFoundError("No *_model.pth file found in: {}".format(model_log_dir))
    model = load_model(str(pth_files[0]))
    if model is None:
        raise RuntimeError("load_model returned None for: {}".format(pth_files[0]))
    model = model.to(COMPUTE_DEVICE)
    model.eval()
    return model


def load_task_models_and_manifests(
    task_model_spec: Dict[str, dict],
) -> Dict[str, dict]:
    """Load the task-specific model and feature manifest for every entry in the spec.

    Returns ``{task_name: {"model", "manifest", "seq_length", "hidden_size", "model_log_dir"}}``.
    Tasks whose model fails to load are skipped with a warning.
    """
    loaded: Dict[str, dict] = {}
    for task_name, entry in task_model_spec.items():
        model_log_dir = Path(entry["model_log_dir"])
        if not model_log_dir.is_dir():
            logging.warning(
                "Skipping task '%s': model log dir not found: %s",
                task_name,
                model_log_dir,
            )
            continue
        try:
            model = _load_model_pth(model_log_dir)
            manifest = _load_feature_manifest_from_dir(model_log_dir)
            hidden_size = int(model.hidden_size)
            loaded[task_name] = {
                "model": model,
                "manifest": manifest,
                "seq_length": entry["seq_length"],
                "hidden_size": hidden_size,
                "model_log_dir": model_log_dir,
            }
            logging.info(
                "Loaded task model for '%s': hidden_size=%d, seq_length=%d, dir=%s",
                task_name,
                hidden_size,
                entry["seq_length"],
                model_log_dir,
            )
        except Exception as exc:
            logging.warning("Failed to load model for task '%s': %s", task_name, exc)

    if not loaded:
        raise ValueError(
            "No task models could be loaded from the provided spec. "
            "Check the spec JSON and that model log dirs exist."
        )
    return loaded


# ---------------------------------------------------------------------------
# Sequence file discovery
# ---------------------------------------------------------------------------


def _normalize_attempt_number(attempt_id: str) -> str:
    """Extract the numeric part from an attempt_id string.

    Converts 'attempt1' → '1', 'attempt12' → '12', '1' → '1'.
    """
    s = str(attempt_id).strip().lower()
    if s.startswith("attempt"):
        s = s[len("attempt") :]
    # Strip leading zeros but keep at least one digit.
    return s.lstrip("0") or "0"


def list_sequences_for_attempt_policy(
    data_dir: Path,
    task_names: Optional[Sequence[str]],
    attempt_policy: str = ATTEMPT_POLICY_ALL,
) -> List[PowerWashSequenceMetadata]:
    """List processed sequence CSVs, keeping only the attempts allowed by ``attempt_policy``.

    With ``"first"`` only ``attempt1`` files are returned, so a player without a usable
    attempt 1 for a task has no sequence for that task. Raises ``FileNotFoundError`` when
    nothing remains, like ``list_powerwash_sequence_metadata``.
    """
    if attempt_policy not in ATTEMPT_POLICIES:
        raise ValueError(
            "attempt_policy must be one of {}, got {!r}.".format(ATTEMPT_POLICIES, attempt_policy)
        )
    items = list_powerwash_sequence_metadata(data_dir, task_names=task_names)
    if attempt_policy == ATTEMPT_POLICY_FIRST:
        items = [item for item in items if _normalize_attempt_number(item.attempt_id) == "1"]
        if not items:
            raise FileNotFoundError(
                "No attempt-1 PowerWash CSVs found under {} for tasks {}".format(
                    data_dir, sorted(task_names or [])
                )
            )
    return items


def find_sequence_path(
    data_dir: Path,
    pid: str,
    task_name: str,
    attempt_id: str,
) -> Optional[Path]:
    """Return ``data_dir/<task>/<pid>_<task>_attempt<N>.csv``, or None if not found.

    Falls back to a recursive search of ``data_dir/<task>/`` for a matching filename.
    ``attempt_id`` may be ``attempt1`` or ``1``.
    """
    attempt_number = _normalize_attempt_number(attempt_id)
    task_dir = Path(data_dir) / task_name

    # Direct lookup.
    filename = "{}_{}_attempt{}.csv".format(pid, task_name, attempt_number)
    candidate = task_dir / filename
    if candidate.exists():
        return candidate

    # Recursive fallback for alternative directory layouts.
    if task_dir.is_dir():
        for path in sorted(task_dir.rglob("*.csv")):
            m = FILENAME_PATTERN.match(path.name)
            if m is None:
                continue
            if (
                m.group("pid") == pid
                and m.group("task") == task_name
                and m.group("attempt") == attempt_number
            ):
                return path

    return None


# ---------------------------------------------------------------------------
# Sequence transformation (applying a saved manifest to a DataFrame)
# ---------------------------------------------------------------------------


def _prepare_manifest_dataframe(
    df: pd.DataFrame,
    manifest: PowerWashFeatureManifest,
) -> Optional[pd.DataFrame]:
    """Normalize a raw sequence dataframe before manifest-driven transforms."""
    if df.empty:
        return None

    prepared_df = _ensure_dataframe_columns(df, manifest.base_feature_columns)
    prepared_df = _sort_sequence_dataframe(prepared_df)
    if len(prepared_df) == 0:
        return None
    return prepared_df


def _apply_manifest_to_prepared_df(
    prepared_df: pd.DataFrame,
    manifest: PowerWashFeatureManifest,
) -> Optional[np.ndarray]:
    """Apply a manifest to an already-prepared dataframe."""
    if prepared_df is None or len(prepared_df) == 0:
        return None

    feature_blocks: List[np.ndarray] = []

    # Numeric columns — fill with manifest median, then min-max scale.
    for col in manifest.numeric_columns:
        values = _prepare_numeric_feature(prepared_df[col], col).to_numpy(dtype=np.float32)
        scaled = _scale_numeric_column(
            values,
            fill_value=manifest.numeric_fill_values.get(col, 0.0),
            min_value=manifest.numeric_mins.get(col, 0.0),
            max_value=manifest.numeric_maxs.get(col, 1.0),
        )
        feature_blocks.append(scaled.reshape(-1, 1))

    # Categorical columns — one-hot encode using manifest levels.
    for col in manifest.categorical_columns:
        prepared = _prepare_categorical_feature(prepared_df[col], col)
        known_levels = set(manifest.categorical_levels[col])
        # Map unseen categories to the UNKNOWN token.
        normalized = prepared.where(
            prepared.astype(str).isin(known_levels),
            other=UNKNOWN_CATEGORY_TOKEN,
        )
        categorical = pd.Categorical(
            normalized,
            categories=manifest.categorical_levels[col],
        )
        encoded = pd.get_dummies(categorical)
        encoded = encoded.reindex(columns=manifest.categorical_levels[col], fill_value=0)
        feature_blocks.append(encoded.to_numpy(dtype=np.float32))

    if not feature_blocks:
        return None

    return np.concatenate(feature_blocks, axis=1).astype(np.float32)


def _pad_or_truncate(
    features: np.ndarray,
    seq_length: int,
    pad_value: float = PAD_VALUE,
) -> np.ndarray:
    """Pad or truncate a feature array to shape (seq_length, n_features).

    Padding rows are filled with pad_value (default PAD_VALUE = -20.0), which
    is the sentinel that MultiTaskEmbedModel / precompute_embeddings uses to
    detect and ignore padded positions.
    """
    n_rows, n_features = features.shape
    if n_rows >= seq_length:
        return features[:seq_length]
    pad = np.full((seq_length - n_rows, n_features), pad_value, dtype=np.float32)
    return np.concatenate([features, pad], axis=0)


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------


def _plvm_cache_key(
    spec_path: str,
    target_mode: str,
    target_col: str,
    data_dir: str,
    seq_length: int,
    model_type: str,
    embeddings_to_use: str,
    full_task_name: str,
    partial_task_name: str,
    max_files_per_task: int,
    sampling_seed: int,
    task_names_filter: str,
    split_name: str = "",
    max_samples_per_player_per_split: int = 0,
    candidate_digest: str = "",
    attempt_policy: str = ATTEMPT_POLICY_ALL,
) -> str:
    """Return a short hex digest identifying this PLVM embedding configuration.

    ``attempt_policy`` enters the key only when it is not ``"all"``, so the default
    ``"all"`` and an omitted policy give the same key.
    """
    canonical_full_task_name, canonical_partial_task_name = canonicalize_dual_task_configuration(
        model_type=model_type,
        full_task_name=full_task_name,
        partial_task_name=partial_task_name,
    )
    key_fields = {
        "version": PLVM_EMBEDDING_CACHE_VERSION,
        "spec_path": str(spec_path),
        "target_mode": str(target_mode),
        "target_col": str(target_col),
        "data_dir": str(data_dir),
        "seq_length": seq_length,
        "model_type": model_type,
        "embeddings_to_use": embeddings_to_use,
        "full_task_name": canonical_full_task_name,
        "partial_task_name": canonical_partial_task_name,
        "max_files_per_task": int(max_files_per_task),
        "sampling_seed": int(sampling_seed),
        "task_names_filter": task_names_filter,
        "split_name": str(split_name),
        "max_samples_per_player_per_split": int(max_samples_per_player_per_split),
        "candidate_digest": str(candidate_digest),
    }
    if attempt_policy != ATTEMPT_POLICY_ALL:
        key_fields["attempt_policy"] = str(attempt_policy)
    payload = json.dumps(key_fields, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Candidate indexing and split helpers
# ---------------------------------------------------------------------------


def _resolve_requested_tasks(
    partial_task_name: str,
    task_names: Optional[Sequence[str]],
) -> Optional[List[str]]:
    """Return the partial-task filter: ``partial_task_name`` if set, else ``task_names``."""
    partial_task_name = str(partial_task_name or "").strip()
    if partial_task_name:
        return [partial_task_name]
    if not task_names:
        return None
    resolved = [str(task_name).strip() for task_name in task_names if str(task_name).strip()]
    return resolved or None


def _uses_fixed_time_cutoff_candidates(model_type: str, target_mode: str) -> bool:
    """Return True when candidates should use fixed elapsed-time cutoffs.

    ``plvm_both_partial`` always uses the shared time-cutoff schedule in
    ``BOTH_PARTIAL_CUTOFF_SECONDS``. For player-label comparisons,
    ``single_task`` should use the same elapsed-time cutoffs so the supervision
    granularity matches the both-task variant.
    """
    normalized_model_type = str(model_type or "").strip().lower()
    normalized_target_mode = str(target_mode or "").strip().lower()
    return normalized_model_type == MODEL_TYPE_BOTH_PARTIAL or (
        normalized_model_type == "single_task" and normalized_target_mode in ALL_PLAYER_LABEL_MODES
    )


def _validate_plvm_task_configuration(
    task_models: Dict[str, dict],
    model_type: str,
    full_task_name: str,
    partial_task_name: str,
) -> None:
    """Validate the requested PLVM task configuration against loaded task models."""
    full_task_name = str(full_task_name or "").strip()
    partial_task_name = str(partial_task_name or "").strip()

    if model_type in DUAL_TASK_MODEL_TYPES:
        if not full_task_name or not partial_task_name:
            raise ValueError(
                "{} requires explicit full_task_name and partial_task_name.".format(model_type)
            )
        if full_task_name == partial_task_name:
            raise ValueError(
                "full_task_name and partial_task_name must differ for {}.".format(model_type)
            )
    elif full_task_name:
        raise ValueError(
            "full_task_name is only valid for dual-task model types: {}.".format(
                sorted(DUAL_TASK_MODEL_TYPES)
            )
        )

    configured_task_names = [
        task_name for task_name in [full_task_name, partial_task_name] if task_name
    ]
    missing_configured_tasks = [
        task_name for task_name in configured_task_names if task_name not in task_models
    ]
    if missing_configured_tasks:
        raise ValueError(
            "Configured task(s) missing from task_models: {}".format(
                sorted(missing_configured_tasks)
            )
        )


def _candidate_sequence_key(candidate: PowerWashPLVMSampleCandidate) -> Tuple[str, str, str]:
    """Return the pid/task/attempt key for one candidate."""
    return candidate.pid, candidate.task_name, candidate.attempt_number


def _candidate_cache_identity(candidate: PowerWashPLVMSampleCandidate) -> Tuple[object, ...]:
    """Return the cache-identity tuple for one candidate without its label value."""
    source_type = str(candidate.source_type)
    if source_type in ALL_PLAYER_LABEL_MODES:
        # Player-label cache reuse should depend on the underlying sequence/cutoff
        # identity, not on which downstream label variant produced the class value.
        source_type = TARGET_MODE_PLAYER_LABEL
    return (
        str(candidate.pid),
        str(candidate.task_name),
        str(candidate.attempt_number),
        float(candidate.cutoff_seconds),
        int(candidate.source_row_index),
        source_type,
    )


def _candidate_digest(candidates: Sequence[PowerWashPLVMSampleCandidate]) -> str:
    """Return a stable digest for a candidate list."""
    payload = json.dumps([candidate.to_dict() for candidate in candidates], sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:16]


def _targets_tensor_from_candidates(
    candidates: Sequence[PowerWashPLVMSampleCandidate],
) -> torch.Tensor:
    """Build the target tensor aligned to one candidate list."""
    return torch.tensor(
        [candidate.target_value for candidate in candidates],
        dtype=torch.float32,
    ).unsqueeze(1)


def _load_compatible_player_label_cache(
    *,
    cache_dir: Path,
    candidates: Sequence[PowerWashPLVMSampleCandidate],
    model_type: str,
    embeddings_to_use: str,
    full_task_name: str,
    partial_task_name: str,
    max_files_per_task: int,
    sampling_seed: int,
    task_names_filter_str: str,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, List[PowerWashPLVMSampleCandidate]]]:
    """Best-effort reuse of existing cache files across player-label variants.

    The exact cache key depends on the full candidate payload, including
    ``target_value``, so a run with different labels (another label variant or a
    label-sensitivity run) misses it even though the embeddings are the same. This
    loader scans the cache files for one whose sample identities (player, task,
    attempt, cutoff, row) cover the requested candidates, whatever labels that cache
    was written with, and rebuilds the targets from the current candidate list.
    """
    requested_candidates = list(candidates)
    requested_identities = [
        _candidate_cache_identity(candidate) for candidate in requested_candidates
    ]
    requested_identity_set = set(requested_identities)
    if not requested_identity_set:
        return None

    cache_paths = sorted(Path(cache_dir).glob("pw_plvm_embeddings_*.pt"))
    for cache_path in cache_paths:
        try:
            cached = torch.load(cache_path, map_location="cpu")
        except Exception as exc:
            logging.warning("Failed to inspect PLVM cache %s: %s", cache_path, exc)
            continue

        if str(cached.get("model_type", "")).strip() != str(model_type).strip():
            continue
        if str(cached.get("embeddings_to_use", "")).strip() != str(embeddings_to_use).strip():
            continue
        if str(cached.get("full_task_name", "")).strip() != str(full_task_name).strip():
            continue
        if str(cached.get("partial_task_name", "")).strip() != str(partial_task_name).strip():
            continue
        if int(cached.get("max_files_per_task", -1)) != int(max_files_per_task):
            continue
        if int(cached.get("sampling_seed", -1)) != int(sampling_seed):
            continue
        if str(cached.get("task_names_filter", "")).strip() != str(task_names_filter_str).strip():
            continue

        cached_payloads = list(cached.get("sample_candidates") or [])
        cached_embeddings = cached.get("embeddings")
        if cached_embeddings is None or len(cached_payloads) != len(cached_embeddings):
            continue

        identity_to_index: Dict[Tuple[object, ...], int] = {}
        for row_index, payload in enumerate(cached_payloads):
            cached_candidate = PowerWashPLVMSampleCandidate(**payload)
            identity_to_index[_candidate_cache_identity(cached_candidate)] = row_index

        if not requested_identity_set.issubset(set(identity_to_index.keys())):
            continue

        ordered_embeddings = torch.stack(
            [cached_embeddings[identity_to_index[identity]] for identity in requested_identities],
            dim=0,
        )
        logging.info(
            "Reused compatible PLVM embedding cache %s for %d player-label candidates.",
            cache_path,
            len(requested_candidates),
        )
        return (
            ordered_embeddings,
            _targets_tensor_from_candidates(requested_candidates),
            requested_candidates,
        )

    return None


def _apply_candidate_max_files_cap(
    candidates: Sequence[PowerWashPLVMSampleCandidate],
    max_files_per_task: Optional[int],
    sampling_seed: int,
) -> List[PowerWashPLVMSampleCandidate]:
    """Cap retained unique sequences per task before any sequence materialization."""
    if max_files_per_task is None:
        return list(candidates)
    if max_files_per_task <= 0:
        raise ValueError("max_files_per_task must be > 0 when provided.")

    task_to_sequence_keys: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    sequence_key_order: List[Tuple[str, str, str]] = []
    seen_keys = set()
    for candidate in candidates:
        key = _candidate_sequence_key(candidate)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        sequence_key_order.append(key)
        task_to_sequence_keys[key[1]].append(key)

    if not sequence_key_order:
        return []

    rng = np.random.RandomState(sampling_seed)
    retained_keys = set()
    capped_out_rows = 0
    capped_out_sequences = 0
    sequence_to_count: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for candidate in candidates:
        sequence_to_count[_candidate_sequence_key(candidate)] += 1

    for task_name, task_sequence_keys in task_to_sequence_keys.items():
        if len(task_sequence_keys) <= max_files_per_task:
            retained_keys.update(task_sequence_keys)
            continue
        sampled_indices = np.sort(
            rng.choice(
                len(task_sequence_keys),
                size=max_files_per_task,
                replace=False,
            )
        )
        retained_task_keys = [task_sequence_keys[idx] for idx in sampled_indices.tolist()]
        retained_task_key_set = set(retained_task_keys)
        retained_keys.update(retained_task_keys)
        capped_out_sequences += len(task_sequence_keys) - len(retained_task_keys)
        capped_out_rows += sum(
            sequence_to_count[key] for key in task_sequence_keys if key not in retained_task_key_set
        )
        logging.info(
            "Applied max_files_per_task=%d to task '%s': retained %d/%d unique sequences.",
            max_files_per_task,
            task_name,
            len(retained_task_keys),
            len(task_sequence_keys),
        )

    retained = [
        candidate for candidate in candidates if _candidate_sequence_key(candidate) in retained_keys
    ]
    if capped_out_sequences:
        logging.info(
            "Applied max_files_per_task=%d with sampling_seed=%d: retained %d/%d unique sequences (%d candidate rows excluded).",
            max_files_per_task,
            sampling_seed,
            len(retained_keys),
            len(sequence_key_order),
            capped_out_rows,
        )
    return retained


def _build_full_task_sequence_index(
    full_task_items: Sequence[PowerWashSequenceMetadata],
) -> Dict[str, List[PowerWashSequenceMetadata]]:
    """Index full-task sequence metadata by player with deterministic attempt ordering."""

    def _sort_key(item: PowerWashSequenceMetadata) -> Tuple[int, str, str]:
        attempt_number = _normalize_attempt_number(item.attempt_id)
        try:
            attempt_sort = int(attempt_number)
        except ValueError:
            attempt_sort = 10**9
        return attempt_sort, attempt_number, str(item.path)

    items_by_pid: Dict[str, List[PowerWashSequenceMetadata]] = defaultdict(list)
    for item in full_task_items:
        items_by_pid[item.pid].append(item)
    for pid in items_by_pid:
        items_by_pid[pid] = sorted(items_by_pid[pid], key=_sort_key)
    return items_by_pid


def _resolve_full_task_sequence_metadata(
    items_by_pid: Dict[str, List[PowerWashSequenceMetadata]],
    pid: str,
    preferred_attempt_number: str,
) -> Optional[PowerWashSequenceMetadata]:
    """Return the preferred full-task sequence for one player.

    Prefer the matching attempt number when available; otherwise fall back to the
    numerically closest attempt for that player.
    """
    candidate_items = items_by_pid.get(pid, [])
    if not candidate_items:
        return None

    normalized_preferred_attempt = _normalize_attempt_number(preferred_attempt_number)
    for item in candidate_items:
        if _normalize_attempt_number(item.attempt_id) == normalized_preferred_attempt:
            return item

    try:
        preferred_attempt_value = int(normalized_preferred_attempt)
    except ValueError:
        return candidate_items[0]

    return min(
        candidate_items,
        key=lambda item: (
            abs(int(_normalize_attempt_number(item.attempt_id)) - preferred_attempt_value),
            int(_normalize_attempt_number(item.attempt_id)),
            str(item.path),
        ),
    )


def _filter_candidates_to_pairable_plvm_sequences(
    candidates: Sequence[PowerWashPLVMSampleCandidate],
    data_dir: Path,
    full_task_name: str,
    attempt_policy: str = ATTEMPT_POLICY_ALL,
) -> Tuple[List[PowerWashPLVMSampleCandidate], int, int]:
    """Retain only candidates whose partial attempt has a matching full-task sequence."""
    if not candidates:
        return [], 0, 0

    all_partial_sequence_keys = {_candidate_sequence_key(candidate) for candidate in candidates}

    try:
        full_task_items = list_sequences_for_attempt_policy(
            data_dir, task_names=[full_task_name], attempt_policy=attempt_policy
        )
    except FileNotFoundError:
        return [], len(all_partial_sequence_keys), len(candidates)
    full_task_items_by_pid = _build_full_task_sequence_index(full_task_items)

    retained_partial_sequence_keys = set()
    for candidate in candidates:
        partial_key = _candidate_sequence_key(candidate)
        full_item = _resolve_full_task_sequence_metadata(
            full_task_items_by_pid,
            pid=candidate.pid,
            preferred_attempt_number=candidate.attempt_number,
        )
        if full_item is not None:
            retained_partial_sequence_keys.add(partial_key)

    filtered = [
        candidate
        for candidate in candidates
        if _candidate_sequence_key(candidate) in retained_partial_sequence_keys
    ]
    dropped_sequence_count = len(all_partial_sequence_keys) - len(retained_partial_sequence_keys)
    dropped_row_count = len(candidates) - len(filtered)
    return filtered, dropped_sequence_count, dropped_row_count


def _read_minimal_target_frame(path: Path, target_col: str) -> pd.DataFrame:
    """Read only elapsed time and the requested target column from a sequence CSV."""
    required = {ELAPSED_TIME_COL, str(target_col)}
    try:
        return pd.read_csv(
            path,
            dtype=object,
            keep_default_na=False,
            usecols=lambda col: col in required,
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=list(required))


def _read_minimal_elapsed_frame(path: Path) -> pd.DataFrame:
    """Read only the elapsed-time column from a sequence CSV."""
    try:
        return pd.read_csv(
            path,
            dtype=object,
            keep_default_na=False,
            usecols=lambda col: col == ELAPSED_TIME_COL,
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=[ELAPSED_TIME_COL])


def find_players_with_usable_sequences_for_all_tasks(
    data_dir: Path,
    task_names: Sequence[str],
    attempt_policy: str = ATTEMPT_POLICY_ALL,
) -> set:
    """Return players that have at least one usable processed sequence in every task.

    A task counts as usable for a player if at least one processed sequence CSV
    exists for that player/task and contains a finite elapsed-time value. This
    is stricter than filename presence alone and is intended to keep the
    train/validation player pools aligned across single-task and dual-task
    player-label runs. With ``attempt_policy="first"`` only attempt 1 counts.
    """
    required_task_names = sorted(
        {str(task_name).strip() for task_name in task_names if str(task_name).strip()}
    )
    if not required_task_names:
        return set()

    usable_players_by_task: Dict[str, set] = {}
    for task_name in required_task_names:
        task_items = list_sequences_for_attempt_policy(
            data_dir, task_names=[task_name], attempt_policy=attempt_policy
        )
        usable_players = set()
        for item in task_items:
            minimal_df = _read_minimal_elapsed_frame(item.path)
            if minimal_df.empty or ELAPSED_TIME_COL not in minimal_df.columns:
                continue
            elapsed_series = pd.to_numeric(
                minimal_df[ELAPSED_TIME_COL],
                errors="coerce",
            )
            elapsed_values = elapsed_series.to_numpy(dtype=np.float32)
            if np.isfinite(elapsed_values).any():
                usable_players.add(item.pid)
        usable_players_by_task[task_name] = usable_players
        logging.info(
            "Usable player coverage for task '%s': %d players",
            task_name,
            len(usable_players),
        )

    eligible_players = set.intersection(*usable_players_by_task.values())
    logging.info(
        "Players with usable sequences for all required tasks %s: %d",
        required_task_names,
        len(eligible_players),
    )
    return eligible_players


def build_powerwash_plvm_candidates(
    data_dir: Path,
    task_models: Dict[str, dict],
    target_mode: str,
    target_col: str,
    player_label_lookup: Optional[Dict[str, int]] = None,
    model_type: str = "plvm_full_partial",
    full_task_name: str = "",
    partial_task_name: str = "",
    max_files_per_task: Optional[int] = None,
    sampling_seed: int = 42,
    task_names: Optional[Sequence[str]] = None,
    require_players_with_all_tasks: bool = False,
    attempt_policy: str = ATTEMPT_POLICY_ALL,
) -> List[PowerWashPLVMSampleCandidate]:
    """Build a lightweight candidate index before any PLVM embedding computation.

    ``attempt_policy`` selects which attempts of the observed tasks are indexed (see
    ``ATTEMPT_POLICY_FIRST``); it applies to the partial-target and player-label modes.
    """
    full_task_name, partial_task_name = canonicalize_dual_task_configuration(
        model_type=model_type,
        full_task_name=full_task_name,
        partial_task_name=partial_task_name,
    )
    _validate_plvm_task_configuration(task_models, model_type, full_task_name, partial_task_name)
    requested_tasks = _resolve_requested_tasks(partial_task_name, task_names)
    active_task_names = sorted(
        task_name
        for task_name in (requested_tasks or task_models.keys())
        if task_name in task_models
    )
    if not active_task_names:
        raise ValueError("No active partial-task names resolved for candidate indexing.")

    candidates: List[PowerWashPLVMSampleCandidate] = []
    skip_reasons: Dict[str, int] = defaultdict(int)
    eligible_multi_task_players: Optional[set] = None

    def _skip(reason: str, count: int = 1) -> None:
        skip_reasons[reason] += count

    if target_mode == TARGET_MODE_PARTIAL_TARGET:
        if not target_col:
            raise ValueError("target_col is required when target_mode='partial_target'.")
        items = list_sequences_for_attempt_policy(
            data_dir, task_names=active_task_names, attempt_policy=attempt_policy
        )
        logging.info(
            "Indexing partial-target candidates from %d sequence file(s) for target '%s'.",
            len(items),
            target_col,
        )
        for item in items:
            if item.task_name not in task_models:
                _skip("no_task_model:{}".format(item.task_name))
                continue
            minimal_df = _read_minimal_target_frame(item.path, target_col)
            if minimal_df.empty:
                _skip("empty_csv")
                continue
            if ELAPSED_TIME_COL not in minimal_df.columns or target_col not in minimal_df.columns:
                _skip("missing_target_or_elapsed")
                continue
            sorted_df = _sort_sequence_dataframe(minimal_df)
            elapsed_series = pd.to_numeric(sorted_df[ELAPSED_TIME_COL], errors="coerce")
            target_series = pd.to_numeric(sorted_df[target_col], errors="coerce")
            for row_idx, (elapsed_value, target_value) in enumerate(
                zip(
                    elapsed_series.to_numpy(dtype=np.float32),
                    target_series.to_numpy(dtype=np.float32),
                )
            ):
                if not np.isfinite(elapsed_value):
                    continue
                if not np.isfinite(target_value):
                    continue
                if model_type == MODEL_TYPE_BOTH_PARTIAL:
                    for cutoff_seconds in BOTH_PARTIAL_CUTOFF_SECONDS:
                        candidates.append(
                            PowerWashPLVMSampleCandidate(
                                pid=item.pid,
                                task_name=item.task_name,
                                attempt_number=_normalize_attempt_number(item.attempt_id),
                                cutoff_seconds=float(cutoff_seconds),
                                target_value=float(target_value),
                                source_row_index=int(row_idx),
                                source_type=TARGET_MODE_PARTIAL_TARGET,
                            )
                        )
                else:
                    candidates.append(
                        PowerWashPLVMSampleCandidate(
                            pid=item.pid,
                            task_name=item.task_name,
                            attempt_number=_normalize_attempt_number(item.attempt_id),
                            cutoff_seconds=float(elapsed_value),
                            target_value=float(target_value),
                            source_row_index=int(row_idx),
                            source_type=TARGET_MODE_PARTIAL_TARGET,
                        )
                    )
        logging.info(
            "Built %d partial-target candidates for target '%s' across %d active task(s).",
            len(candidates),
            target_col,
            len(active_task_names),
        )
    elif target_mode in ALL_PLAYER_LABEL_MODES:
        if not player_label_lookup:
            raise ValueError(
                "player_label_lookup is required when target_mode is a player_label variant."
            )
        if require_players_with_all_tasks:
            eligible_multi_task_players = find_players_with_usable_sequences_for_all_tasks(
                data_dir=data_dir,
                task_names=sorted(task_models.keys()),
                attempt_policy=attempt_policy,
            )
        items = list_sequences_for_attempt_policy(
            data_dir, task_names=active_task_names, attempt_policy=attempt_policy
        )
        logging.info(
            "Indexing player-label candidates from %d sequence file(s) across %d labeled player(s).",
            len(items),
            len(player_label_lookup),
        )
        for item in items:
            if item.task_name not in task_models:
                _skip("no_task_model:{}".format(item.task_name))
                continue
            if (
                eligible_multi_task_players is not None
                and item.pid not in eligible_multi_task_players
            ):
                _skip("missing_required_task_coverage")
                continue
            if item.pid not in player_label_lookup:
                _skip("missing_player_label")
                continue
            minimal_df = _read_minimal_elapsed_frame(item.path)
            if minimal_df.empty:
                _skip("empty_csv")
                continue
            if ELAPSED_TIME_COL not in minimal_df.columns:
                _skip("missing_elapsed")
                continue
            class_index = float(player_label_lookup[item.pid])
            if _uses_fixed_time_cutoff_candidates(model_type, target_mode):
                for cutoff_seconds in BOTH_PARTIAL_CUTOFF_SECONDS:
                    candidates.append(
                        PowerWashPLVMSampleCandidate(
                            pid=item.pid,
                            task_name=item.task_name,
                            attempt_number=_normalize_attempt_number(item.attempt_id),
                            cutoff_seconds=float(cutoff_seconds),
                            target_value=class_index,
                            source_row_index=-1,
                            source_type=target_mode,
                        )
                    )
            else:
                sorted_df = _sort_sequence_dataframe(minimal_df)
                elapsed_series = pd.to_numeric(sorted_df[ELAPSED_TIME_COL], errors="coerce")
                for row_idx, elapsed_value in enumerate(elapsed_series.to_numpy(dtype=np.float32)):
                    if not np.isfinite(elapsed_value):
                        continue
                    candidates.append(
                        PowerWashPLVMSampleCandidate(
                            pid=item.pid,
                            task_name=item.task_name,
                            attempt_number=_normalize_attempt_number(item.attempt_id),
                            cutoff_seconds=float(elapsed_value),
                            target_value=class_index,
                            source_row_index=int(row_idx),
                            source_type=target_mode,
                        )
                    )
        logging.info(
            "Built %d player-label candidates across %d active task(s).",
            len(candidates),
            len(active_task_names),
        )
    else:
        raise ValueError("Unsupported target_mode='{}'.".format(target_mode))

    if model_type in DUAL_TASK_MODEL_TYPES:
        candidates, dropped_sequence_count, dropped_row_count = (
            _filter_candidates_to_pairable_plvm_sequences(
                candidates,
                data_dir=data_dir,
                full_task_name=full_task_name,
                attempt_policy=attempt_policy,
            )
        )
        if dropped_row_count:
            _skip("full_sequence_not_found_pre_cap", dropped_row_count)
            retained_sequence_count = len(
                {_candidate_sequence_key(candidate) for candidate in candidates}
            )
            logging.info(
                "Filtered PLVM candidates to pairable full task '%s': retained %d unique sequences after excluding %d/%d unique sequences (%d candidate rows) before max_files_per_task.",
                full_task_name,
                retained_sequence_count,
                dropped_sequence_count,
                dropped_sequence_count + retained_sequence_count,
                dropped_row_count,
            )

    candidates = _apply_candidate_max_files_cap(
        candidates,
        max_files_per_task=max_files_per_task,
        sampling_seed=sampling_seed,
    )
    if skip_reasons:
        top_reasons = sorted(skip_reasons.items(), key=lambda kv: -kv[1])[:8]
        logging.warning("Candidate indexing skipped rows/files: %s", dict(top_reasons))
    if not candidates:
        raise ValueError(
            "No valid PowerWash PLVM candidates built for target_mode='{}'.".format(target_mode)
        )
    return candidates


def split_plvm_candidates_by_player(
    candidates: Sequence[PowerWashPLVMSampleCandidate],
    val_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 42,
) -> Tuple[
    List[PowerWashPLVMSampleCandidate],
    List[PowerWashPLVMSampleCandidate],
    List[PowerWashPLVMSampleCandidate],
]:
    """Split candidate metadata by player before sequence materialization."""
    import random  # noqa: PLC0415

    candidates = list(candidates)
    unique_pids = sorted({candidate.pid for candidate in candidates})
    rng = random.Random(seed)
    rng.shuffle(unique_pids)

    n = len(unique_pids)
    n_test = 0 if test_fraction <= 0 else max(1, int(n * test_fraction))
    n_test = min(n_test, max(0, n - 1))

    if val_fraction <= 0:
        n_val = 0
    else:
        n_val = max(1, int(n * val_fraction))
        n_val = min(n_val, max(0, n - n_test - 1))
        if n_val == 0:
            raise ValueError(
                "Not enough players to create a non-empty validation split "
                "(players={}, test_fraction={}, val_fraction={}).".format(
                    n, test_fraction, val_fraction
                )
            )

    test_pids = set(unique_pids[:n_test])
    val_pids = set(unique_pids[n_test : n_test + n_val])
    train_pids = set(unique_pids[n_test + n_val :])
    if not train_pids:
        raise ValueError(
            "Player split left the training set empty "
            "(players={}, test_fraction={}, val_fraction={}).".format(
                n, test_fraction, val_fraction
            )
        )

    def _select(pid_set):
        return [candidate for candidate in candidates if candidate.pid in pid_set]

    train_candidates = _select(train_pids)
    val_candidates = _select(val_pids)
    test_candidates = _select(test_pids)

    if test_pids:
        logging.info(
            "Candidate split | train=%d (%d players) val=%d (%d players) test=%d (%d players)",
            len(train_candidates),
            len(train_pids),
            len(val_candidates),
            len(val_pids),
            len(test_candidates),
            len(test_pids),
        )
    else:
        logging.info(
            "Candidate split | train=%d (%d players) val=%d (%d players)",
            len(train_candidates),
            len(train_pids),
            len(val_candidates),
            len(val_pids),
        )
    return train_candidates, val_candidates, test_candidates


def sample_plvm_candidates_by_player(
    candidates: Sequence[PowerWashPLVMSampleCandidate],
    max_samples_per_player: int,
    seed: int,
) -> List[PowerWashPLVMSampleCandidate]:
    """Sample at most N candidates per player with deterministic uniform sampling."""
    candidates = list(candidates)
    if max_samples_per_player <= 0:
        return candidates
    sampled_indices = sorted(
        _sample_train_indices_by_id(
            sample_ids=[candidate.pid for candidate in candidates],
            samples_per_id=max_samples_per_player,
            sampling_seed=seed,
        )
    )
    sampled = [candidates[index] for index in sampled_indices]
    logging.info(
        "Applied player-wise candidate cap=%d: retained %d/%d candidates across %d players.",
        max_samples_per_player,
        len(sampled),
        len(candidates),
        len({candidate.pid for candidate in sampled}),
    )
    return sampled


# ---------------------------------------------------------------------------
# Candidate materialization
# ---------------------------------------------------------------------------


def materialize_powerwash_plvm_candidates(
    candidates: Sequence[PowerWashPLVMSampleCandidate],
    data_dir: Path,
    task_models: Dict[str, dict],
    seq_length: int,
    model_type: str = "plvm_full_partial",
    embeddings_to_use: str = "masked_average_all",
    fusion_type: str = "concat",
    pool_kernel_size: int = 1,
    pool_stride: int = 1,
    full_task_name: str = "",
    partial_task_name: str = "",
    cache_dir: Optional[str] = None,
    spec_path: str = "",
    target_mode: str = TARGET_MODE_PLAYER_LABEL,
    target_col: str = "",
    max_files_per_task: Optional[int] = None,
    sampling_seed: int = 42,
    task_names: Optional[Sequence[str]] = None,
    split_name: str = "",
    max_samples_per_player_per_split: int = 0,
    attempt_policy: str = ATTEMPT_POLICY_ALL,
) -> Tuple[torch.Tensor, torch.Tensor, List[PowerWashPLVMSampleCandidate]]:
    """Materialize sampled candidates into aligned embeddings and targets.

    ``attempt_policy`` restricts which full-task attempts a partial attempt can be paired
    with; use the value passed to ``build_powerwash_plvm_candidates``.
    """
    candidates = list(candidates)
    if not candidates:
        raise ValueError("Cannot materialize an empty candidate list.")

    full_task_name, partial_task_name = canonicalize_dual_task_configuration(
        model_type=model_type,
        full_task_name=full_task_name,
        partial_task_name=partial_task_name,
    )
    _validate_plvm_task_configuration(task_models, model_type, full_task_name, partial_task_name)
    requested_tasks = _resolve_requested_tasks(partial_task_name, task_names)
    task_names_filter_str = ",".join(sorted(requested_tasks or []))
    effective_max_files_per_task = int(max_files_per_task) if max_files_per_task is not None else -1
    candidate_digest = _candidate_digest(candidates)

    cache_file: Optional[Path] = None
    if cache_dir:
        cache_key = _plvm_cache_key(
            spec_path=spec_path,
            target_mode=target_mode,
            target_col=target_col,
            data_dir=str(data_dir),
            seq_length=seq_length,
            model_type=model_type,
            embeddings_to_use=embeddings_to_use,
            full_task_name=full_task_name,
            partial_task_name=partial_task_name,
            max_files_per_task=effective_max_files_per_task,
            sampling_seed=sampling_seed,
            task_names_filter=task_names_filter_str,
            split_name=split_name,
            max_samples_per_player_per_split=max_samples_per_player_per_split,
            candidate_digest=candidate_digest,
            attempt_policy=attempt_policy,
        )
        cache_file = Path(cache_dir) / "pw_plvm_embeddings_{}.pt".format(cache_key)
        if cache_file.exists():
            try:
                cached = torch.load(cache_file, map_location="cpu")
                if cached.get("candidate_digest") == candidate_digest:
                    return (
                        cached["embeddings"],
                        cached["targets"],
                        [
                            PowerWashPLVMSampleCandidate(**payload)
                            for payload in cached["sample_candidates"]
                        ],
                    )
            except Exception as exc:
                logging.warning("Failed to load PLVM split cache %s: %s", cache_file, exc)
        if target_mode in ALL_PLAYER_LABEL_MODES:
            compatible_cache = _load_compatible_player_label_cache(
                cache_dir=Path(cache_dir),
                candidates=candidates,
                model_type=model_type,
                embeddings_to_use=embeddings_to_use,
                full_task_name=full_task_name,
                partial_task_name=partial_task_name,
                max_files_per_task=effective_max_files_per_task,
                sampling_seed=sampling_seed,
                task_names_filter_str=task_names_filter_str,
            )
            if compatible_cache is not None:
                return compatible_cache

    task_buffers: Dict[str, List[dict]] = defaultdict(list)
    valid_candidates: List[Optional[PowerWashPLVMSampleCandidate]] = [None] * len(candidates)
    skip_reasons: Dict[str, int] = defaultdict(int)

    def _skip(reason: str, count: int = 1) -> None:
        skip_reasons[reason] += count

    grouped_candidates: Dict[
        Tuple[str, str, str], List[Tuple[int, PowerWashPLVMSampleCandidate]]
    ] = defaultdict(list)
    for candidate_index, candidate in enumerate(candidates):
        grouped_candidates[_candidate_sequence_key(candidate)].append((candidate_index, candidate))

    full_task_items_by_pid: Dict[str, List[PowerWashSequenceMetadata]] = {}
    if model_type in DUAL_TASK_MODEL_TYPES:
        try:
            full_task_items = list_sequences_for_attempt_policy(
                data_dir, task_names=[full_task_name], attempt_policy=attempt_policy
            )
        except FileNotFoundError:
            full_task_items = []
        full_task_items_by_pid = _build_full_task_sequence_index(full_task_items)

    for sequence_idx, ((pid, task_name, attempt_number), sample_requests) in enumerate(
        grouped_candidates.items(),
        start=1,
    ):
        partial_seq_path = find_sequence_path(data_dir, pid, task_name, attempt_number)
        if partial_seq_path is None:
            _skip(
                "sequence_not_found:{}/{}/{}".format(pid, task_name, attempt_number),
                count=len(sample_requests),
            )
            continue

        partial_seq_df = read_powerwash_csv(partial_seq_path)
        if partial_seq_df.empty:
            _skip("empty_csv", count=len(sample_requests))
            continue
        if ELAPSED_TIME_COL not in partial_seq_df.columns:
            _skip("missing_elapsed_time_col", count=len(sample_requests))
            continue

        partial_task_entry = task_models[task_name]
        partial_manifest: PowerWashFeatureManifest = partial_task_entry["manifest"]
        partial_task_seq_length = int(partial_task_entry.get("seq_length", seq_length))

        prepared_partial_df = _prepare_manifest_dataframe(partial_seq_df, partial_manifest)
        if prepared_partial_df is None:
            _skip("manifest_prepare_failed", count=len(sample_requests))
            continue

        partial_full_features = _apply_manifest_to_prepared_df(
            prepared_partial_df, partial_manifest
        )
        if partial_full_features is None:
            _skip("manifest_transform_failed", count=len(sample_requests))
            continue

        sorted_elapsed = pd.to_numeric(
            prepared_partial_df[ELAPSED_TIME_COL], errors="coerce"
        ).to_numpy(dtype=np.float32)
        elapsed_for_search = np.where(
            np.isfinite(sorted_elapsed),
            sorted_elapsed,
            np.inf,
        )

        full_t = None
        shared_full_t = None
        full_tensor_by_prefix_len: Dict[int, torch.Tensor] = {}
        full_elapsed_for_search = None
        if model_type in DUAL_TASK_MODEL_TYPES:
            full_task_entry = task_models[full_task_name]
            full_manifest: PowerWashFeatureManifest = full_task_entry["manifest"]
            full_task_seq_length = int(full_task_entry.get("seq_length", seq_length))

            full_seq_item = _resolve_full_task_sequence_metadata(
                full_task_items_by_pid,
                pid=pid,
                preferred_attempt_number=attempt_number,
            )
            if full_seq_item is None:
                _skip(
                    "full_sequence_not_found:{}/{}".format(pid, full_task_name),
                    count=len(sample_requests),
                )
                continue
            full_seq_path = full_seq_item.path

            full_seq_df = read_powerwash_csv(full_seq_path)
            if full_seq_df.empty:
                _skip("full_empty_csv", count=len(sample_requests))
                continue

            prepared_full_df = _prepare_manifest_dataframe(full_seq_df, full_manifest)
            if prepared_full_df is None:
                _skip("full_manifest_prepare_failed", count=len(sample_requests))
                continue

            full_features = _apply_manifest_to_prepared_df(prepared_full_df, full_manifest)
            if full_features is None:
                _skip("full_manifest_transform_failed", count=len(sample_requests))
                continue

            if model_type == MODEL_TYPE_BOTH_PARTIAL:
                full_sorted_elapsed = pd.to_numeric(
                    prepared_full_df[ELAPSED_TIME_COL], errors="coerce"
                ).to_numpy(dtype=np.float32)
                full_elapsed_for_search = np.where(
                    np.isfinite(full_sorted_elapsed),
                    full_sorted_elapsed,
                    np.inf,
                )

            # For plvm_full_partial: full_t is shared across all candidates in this group.
            # For plvm_both_partial: full_t varies by elapsed-time cutoff and is
            # computed lazily below.
            if model_type != MODEL_TYPE_BOTH_PARTIAL:
                full_padded = _pad_or_truncate(full_features, full_task_seq_length)
                shared_full_t = torch.tensor(full_padded, dtype=torch.float32).unsqueeze(0)

        partial_tensor_cache: Dict[int, torch.Tensor] = {}
        for candidate_index, candidate in sample_requests:
            if model_type == MODEL_TYPE_BOTH_PARTIAL:
                cutoff_seconds = float(candidate.cutoff_seconds)
                if cutoff_seconds <= 0.0:
                    _skip("both_partial_non_positive_cutoff")
                    continue
                full_prefix_len = int(
                    np.searchsorted(
                        full_elapsed_for_search,
                        cutoff_seconds,
                        side="right",
                    )
                )
                if full_prefix_len <= 0:
                    _skip("full_empty:cutoff={:.1f}".format(cutoff_seconds))
                    continue
                effective_prefix_len = int(
                    np.searchsorted(
                        elapsed_for_search,
                        cutoff_seconds,
                        side="right",
                    )
                )
                if effective_prefix_len <= 0:
                    _skip("partial_empty:cutoff={:.1f}".format(cutoff_seconds))
                    continue
                full_t = full_tensor_by_prefix_len.get(full_prefix_len)
                if full_t is None:
                    full_padded = _pad_or_truncate(
                        full_features[:full_prefix_len], full_task_seq_length
                    )
                    full_t = torch.tensor(full_padded, dtype=torch.float32).unsqueeze(0)
                    full_tensor_by_prefix_len[full_prefix_len] = full_t
            else:
                full_t = shared_full_t
                prefix_len = int(
                    np.searchsorted(
                        elapsed_for_search,
                        candidate.cutoff_seconds,
                        side="right",
                    )
                )
                if prefix_len <= 0:
                    _skip("partial_empty:cutoff={:.1f}".format(candidate.cutoff_seconds))
                    continue
                effective_prefix_len = prefix_len
                if effective_prefix_len <= 0:
                    _skip("partial_empty:cutoff={:.1f}".format(candidate.cutoff_seconds))
                    continue

            partial_t = partial_tensor_cache.get(effective_prefix_len)
            if partial_t is None:
                partial_features = partial_full_features[:effective_prefix_len]
                partial_padded = _pad_or_truncate(partial_features, partial_task_seq_length)
                partial_t = torch.tensor(partial_padded, dtype=torch.float32).unsqueeze(0)
                partial_tensor_cache[effective_prefix_len] = partial_t

            task_buffers[task_name].append(
                {
                    "candidate_index": candidate_index,
                    "candidate": candidate,
                    "full_t": full_t,
                    "partial_t": partial_t,
                }
            )
            valid_candidates[candidate_index] = candidate

        if sequence_idx % 1000 == 0:
            logging.info(
                "Materialized %d/%d candidate sequences so far.",
                sequence_idx,
                len(grouped_candidates),
            )

    flat_embeddings: Dict[int, torch.Tensor] = {}
    for task_name, buf_list in task_buffers.items():
        partial_task_entry = task_models[task_name]
        partial_model = partial_task_entry["model"]
        partial_hidden_size = int(partial_task_entry["hidden_size"])
        partial_task_seq_length = int(partial_task_entry.get("seq_length", seq_length))

        if model_type in DUAL_TASK_MODEL_TYPES:
            full_task_entry = task_models[full_task_name]
            full_model = full_task_entry["model"]
            full_hidden_size = int(full_task_entry["hidden_size"])
            full_task_seq_length = int(full_task_entry.get("seq_length", seq_length))
            if partial_hidden_size != full_hidden_size:
                raise ValueError(
                    "PowerWash {} expects matching hidden sizes across configured models. "
                    "partial={} (task={}) full={} (task={})".format(
                        model_type,
                        partial_hidden_size,
                        task_name,
                        full_hidden_size,
                        full_task_name,
                    )
                )
            models_dict = {"full": full_model, "partial": partial_model}
            task_hidden_size = partial_hidden_size
            task_seq_length = max(partial_task_seq_length, full_task_seq_length)
        else:
            models_dict = {"partial": partial_model}
            task_hidden_size = partial_hidden_size
            task_seq_length = partial_task_seq_length

        session_data = []
        for buf in buf_list:
            if model_type in DUAL_TASK_MODEL_TYPES:
                session_data.append([buf["full_t"], buf["partial_t"]])
            else:
                session_data.append([buf["partial_t"]])

        task_embs = precompute_embeddings(
            task_specific_models=models_dict,
            session_task_data_x=session_data,
            max_sequence_len=task_seq_length,
            task_specific_hidden_size=task_hidden_size,
            fusion_type=fusion_type,
            pool_kernel_size=pool_kernel_size,
            pool_stride=pool_stride,
            batch_size=64,
            embeddings_to_use=embeddings_to_use,
        )
        for row_index, buf in enumerate(buf_list):
            flat_embeddings[int(buf["candidate_index"])] = task_embs[row_index]

    ordered_valid_candidates: List[PowerWashPLVMSampleCandidate] = []
    ordered_embeddings: List[torch.Tensor] = []
    ordered_targets: List[float] = []
    for candidate_index, candidate in enumerate(valid_candidates):
        if candidate is None or candidate_index not in flat_embeddings:
            continue
        ordered_valid_candidates.append(candidate)
        ordered_embeddings.append(flat_embeddings[candidate_index])
        ordered_targets.append(candidate.target_value)

    if not ordered_embeddings:
        raise ValueError(
            "No valid PLVM samples remained after candidate materialization. Reasons: {}".format(
                dict(skip_reasons)
            )
        )

    if skip_reasons:
        top_reasons = sorted(skip_reasons.items(), key=lambda kv: -kv[1])[:8]
        logging.warning("Candidate materialization skipped samples: %s", dict(top_reasons))

    embeddings = torch.stack(ordered_embeddings, dim=0)
    targets_tensor = torch.tensor(ordered_targets, dtype=torch.float32).unsqueeze(1)

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(
                {
                    "embeddings": embeddings.cpu(),
                    "targets": targets_tensor.cpu(),
                    "sample_candidates": [
                        candidate.to_dict() for candidate in ordered_valid_candidates
                    ],
                    "candidate_digest": candidate_digest,
                    "model_type": model_type,
                    "target_mode": target_mode,
                    "target_col": target_col,
                    "full_task_name": full_task_name,
                    "partial_task_name": partial_task_name,
                    "embeddings_to_use": embeddings_to_use,
                    "max_files_per_task": effective_max_files_per_task,
                    "sampling_seed": sampling_seed,
                    "task_names_filter": task_names_filter_str,
                },
                cache_file,
            )
            logging.info("Saved PLVM split cache to %s", cache_file)
        except Exception as exc:
            logging.warning("Could not save PLVM split cache: %s", exc)

    return embeddings, targets_tensor, ordered_valid_candidates
