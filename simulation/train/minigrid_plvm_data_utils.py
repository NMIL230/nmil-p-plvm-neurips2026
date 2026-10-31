"""Utilities for paired-task PLVM label classification on minigrid simulations."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from simulation.config import (
    TASK_NAME_CLEANING,
    TASK_NAME_FORAGING,
)
from simulation.train.data_utils import (
    MinigridFeatureManifest,
    build_minigrid_sequence_examples,
    build_player_splits,
    load_simulation_tables,
    load_task_layout,
)
from plvm import artifact_paths
from plvm.model_training_utils import (
    COMPUTE_DEVICE,
    FILL_MISSING_VALUE,
    PAD_VALUE,
    build_padding_mask,
    load_model,
)


DEFAULT_TASK_MODEL_ROOT = artifact_paths.SIMULATION_TASK_SPECIFIC_MODELS_DIR
# Layout of DEFAULT_TASK_MODEL_ROOT below the artifact root, so that a caller passing
# its own project root gets the same relative tree.
_TASK_SPECIFIC_SUBPATH = DEFAULT_TASK_MODEL_ROOT.relative_to(artifact_paths.ARTIFACT_ROOT)
MINIGRID_PLVM_DATASET_CACHE_VERSION = 1
MINIGRID_PLVM_EMBEDDING_CACHE_VERSION = 1
MODEL_TYPE_BOTH_PARTIAL = "plvm_both_partial"
MODEL_TYPE_SINGLE_TASK = "single_task"
PLVM_TASK_NAMES: Tuple[str, str] = (TASK_NAME_FORAGING, TASK_NAME_CLEANING)
DEFAULT_CUTOFF_SCHEDULE_STEPS: Tuple[int, ...] = tuple(range(20, 201, 20))
SUPPORTED_EMBEDDING_POOLERS = frozenset(
    {
        "last_only",
        "average_all",
        "masked_average_all",
        "average_last_10",
        "average_last_20",
        "average_last_30",
        "first_only",
        "first_last_concat",
        "last_only_plus_count",
    }
)


@dataclass(frozen=True)
class MinigridPLVMSampleCandidate:
    """One paired-task player sample for PLVM label classification."""

    player_id: int
    label: str
    class_id: int
    latent_P: float
    latent_O: float
    cutoff_steps: int

    def to_dict(self) -> dict:
        """Serialize the sample metadata to a JSON-friendly dictionary."""
        return asdict(self)


def load_minigrid_feature_manifest(path: Path) -> MinigridFeatureManifest:
    """Load a saved minigrid feature manifest from JSON."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError("feature manifest not found: {}".format(manifest_path))

    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return MinigridFeatureManifest(
        task_name=str(payload["task_name"]),
        grid_size=int(payload["grid_size"]),
        local_patch_radius=int(payload["local_patch_radius"]),
        feature_columns=list(payload["feature_columns"]),
        action_vocab=list(payload["action_vocab"]),
        action_to_index={
            str(key): int(value) for key, value in dict(payload["action_to_index"]).items()
        },
        terrain_levels=[int(value) for value in payload["terrain_levels"]],
        boundary_terrain_code=int(payload["boundary_terrain_code"]),
        target_ids=list(payload["target_ids"]),
        target_rewards={
            str(key): float(value) for key, value in dict(payload["target_rewards"]).items()
        },
        target_saliences={
            str(key): float(value)
            for key, value in dict(payload.get("target_saliences", {})).items()
        },
        target_high_value_flags={
            str(key): int(value) for key, value in dict(payload["target_high_value_flags"]).items()
        },
        max_distance=float(payload["max_distance"]),
        max_reward=float(payload["max_reward"]),
        max_salience=float(payload.get("max_salience", payload["max_reward"])),
        total_targets=int(payload["total_targets"]),
        total_possible_reward=float(payload["total_possible_reward"]),
    )


def _infer_seq_length_from_model_log_dir(model_log_dir: Path) -> int:
    """Best-effort recovery of the task-specific chunk length for logging only."""
    data_splits_path = Path(model_log_dir) / "data_splits.json"
    if not data_splits_path.exists():
        return 0

    try:
        with data_splits_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return 0

    train_stats = payload.get("train_stats", {})
    return int(train_stats.get("max_seq_length", 0) or 0)


def _resolve_model_log_dir(task_name: str, raw_entry, project_root: Path) -> Path:
    """Resolve one task-model spec entry to a task-specific training-log directory."""
    if isinstance(raw_entry, dict):
        raw_dir = str(raw_entry.get("model_log_dir", "") or "").strip()
        raw_name = str(raw_entry.get("model_name", "") or "").strip()
        if raw_dir:
            candidate = Path(raw_dir)
            if candidate.suffix == ".pth":
                candidate = candidate.parent
            # Spec entries may carry an absolute path recorded at training time.
            return artifact_paths.rebase_recorded_artifact_path(
                candidate if candidate.is_absolute() else Path(project_root) / candidate
            )
        if raw_name:
            return (
                Path(project_root) / _TASK_SPECIFIC_SUBPATH / task_name / "training_logs" / raw_name
            )
        raise ValueError(
            "Task spec entry for '{}' is missing model_log_dir/model_name.".format(task_name)
        )

    raw_value = str(raw_entry or "").strip()
    if not raw_value:
        raise ValueError("Task spec entry for '{}' is empty.".format(task_name))

    candidate = Path(raw_value)
    if candidate.suffix == ".pth":
        candidate = candidate.parent

    if candidate.is_absolute() or "/" in raw_value:
        return artifact_paths.rebase_recorded_artifact_path(
            candidate if candidate.is_absolute() else Path(project_root) / candidate
        )

    return Path(project_root) / _TASK_SPECIFIC_SUBPATH / task_name / "training_logs" / raw_value


def _discover_best_task_model_dir(task_name: str, project_root: Path) -> Path:
    """Pick the best completed task-specific model for one minigrid task."""
    training_root = Path(project_root) / _TASK_SPECIFIC_SUBPATH / task_name / "training_logs"
    if not training_root.is_dir():
        raise FileNotFoundError("Task-specific training root not found: {}".format(training_root))

    scored_runs: List[Tuple[float, float, Path]] = []
    for run_dir in sorted(training_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if not list(run_dir.glob("*_model.pth")):
            continue

        score = float("-inf")
        summary_path = run_dir / "training_summary.json"
        val_metrics_path = run_dir / "val_metrics.json"
        if summary_path.exists():
            try:
                with summary_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                score = float(payload.get("best_val_accuracy", float("-inf")))
            except Exception:
                score = float("-inf")
        elif val_metrics_path.exists():
            try:
                with val_metrics_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                score = float(payload.get("accuracy", float("-inf")))
            except Exception:
                score = float("-inf")

        scored_runs.append((score, float(run_dir.stat().st_mtime), run_dir))

    if not scored_runs:
        raise FileNotFoundError(
            "Could not find a completed task-specific model for '{}' under {}".format(
                task_name,
                training_root,
            )
        )

    scored_runs.sort(key=lambda item: (item[0], item[1]))
    best_dir = scored_runs[-1][2]
    logging.info("Auto-selected %s task model: %s", task_name, best_dir)
    return best_dir


def load_minigrid_task_model_spec(
    spec_path: Optional[Path],
    project_root: Path = artifact_paths.ARTIFACT_ROOT,
    required_task_names: Optional[Sequence[str]] = None,
) -> Dict[str, dict]:
    """Load or auto-discover the minigrid task-model specification."""
    required_task_names = list(required_task_names or PLVM_TASK_NAMES)
    if spec_path:
        spec_path = Path(spec_path).expanduser().resolve()
        if not spec_path.exists():
            raise FileNotFoundError("Task-model spec JSON not found: {}".format(spec_path))

        with spec_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        tasks_raw = payload.get("tasks", {})
        if not isinstance(tasks_raw, dict):
            raise ValueError(
                "Task-model spec must contain a top-level 'tasks' object: {}".format(spec_path)
            )

        resolved: Dict[str, dict] = {}
        for task_name in required_task_names:
            if task_name not in tasks_raw:
                raise ValueError(
                    "Task-model spec is missing required task '{}': {}".format(
                        task_name,
                        spec_path,
                    )
                )
            entry = tasks_raw[task_name]
            model_log_dir = _resolve_model_log_dir(task_name, entry, project_root)
            seq_length = 0
            if isinstance(entry, dict):
                seq_length = int(entry.get("seq_length", 0) or 0)
            resolved[task_name] = {
                "model_log_dir": model_log_dir,
                "seq_length": seq_length,
            }
        return resolved

    resolved = {}
    for task_name in required_task_names:
        model_log_dir = _discover_best_task_model_dir(task_name, project_root)
        resolved[task_name] = {
            "model_log_dir": model_log_dir,
            "seq_length": _infer_seq_length_from_model_log_dir(model_log_dir),
        }
    return resolved


def _load_model_pth(model_log_dir: Path) -> torch.nn.Module:
    """Load one saved task-specific model from a training-log directory."""
    model_log_dir = Path(model_log_dir)
    pth_files = sorted(model_log_dir.glob("*_model.pth"))
    if not pth_files:
        raise FileNotFoundError("No *_model.pth file found in: {}".format(model_log_dir))

    model = load_model(str(pth_files[0]))
    if model is None:
        raise RuntimeError("load_model returned None for: {}".format(pth_files[0]))
    if not hasattr(model, "get_embedding"):
        raise TypeError(
            "Task-specific model at {} does not expose get_embedding(); "
            "PLVM training requires transformer-style task models.".format(model_log_dir)
        )
    model = model.to(COMPUTE_DEVICE)
    model.eval()
    return model


def load_task_models_and_manifests(task_model_spec: Dict[str, dict]) -> Dict[str, dict]:
    """Load the paired minigrid task models and feature manifests."""
    loaded: Dict[str, dict] = {}
    for task_name in task_model_spec:
        entry = task_model_spec.get(task_name)
        if entry is None:
            raise KeyError("Task-model spec is missing required task '{}'.".format(task_name))

        model_log_dir = Path(entry["model_log_dir"]).expanduser().resolve()
        if not model_log_dir.is_dir():
            raise FileNotFoundError("Task model log dir not found: {}".format(model_log_dir))

        model = _load_model_pth(model_log_dir)
        manifest = load_minigrid_feature_manifest(model_log_dir / "feature_manifest.json")
        hidden_size = int(
            getattr(model, "hidden_size", 0) or model.hyperparameters.get("hidden_size", 0)
        )
        loaded[task_name] = {
            "model": model,
            "manifest": manifest,
            "seq_length": int(entry.get("seq_length", 0) or 0),
            "hidden_size": hidden_size,
            "model_log_dir": model_log_dir,
            "model_name": model_log_dir.name,
        }
        logging.info(
            "Loaded minigrid task model for '%s': hidden_size=%d dir=%s",
            task_name,
            hidden_size,
            model_log_dir,
        )
    return loaded


def load_player_split_manifest(split_path: Path) -> Dict[str, object]:
    """Load a player split CSV with `player_id` and `split` columns."""
    split_path = Path(split_path).expanduser().resolve()
    if not split_path.exists():
        raise FileNotFoundError("Player split manifest not found: {}".format(split_path))

    split_df = pd.read_csv(split_path)
    required_columns = {"player_id", "split"}
    missing_columns = required_columns.difference(split_df.columns)
    if missing_columns:
        raise ValueError(
            "Player split manifest missing required columns {}: {}".format(
                sorted(missing_columns),
                split_path,
            )
        )

    split_df = split_df.copy()
    split_df["player_id"] = split_df["player_id"].astype(int)
    split_df["split"] = split_df["split"].astype(str).str.strip().str.lower()
    invalid_splits = sorted(set(split_df["split"]) - {"train", "val", "ignored"})
    if invalid_splits:
        raise ValueError("Unsupported split values {} in {}".format(invalid_splits, split_path))

    train_player_ids = sorted(
        split_df.loc[split_df["split"] == "train", "player_id"].astype(int).tolist()
    )
    val_player_ids = sorted(
        split_df.loc[split_df["split"] == "val", "player_id"].astype(int).tolist()
    )
    ignored_player_ids = sorted(
        split_df.loc[split_df["split"] == "ignored", "player_id"].astype(int).tolist()
    )

    return {
        "train_player_ids": train_player_ids,
        "val_player_ids": val_player_ids,
        "ignored_player_ids": ignored_player_ids,
        "split_df": split_df.sort_values(["split", "player_id"], kind="stable").reset_index(
            drop=True
        ),
        "summary": {
            "n_total_players": int(len(split_df)),
            "n_train_val_players": int(len(train_player_ids) + len(val_player_ids)),
            "n_ignored_players": int(len(ignored_player_ids)),
            "n_train_players": int(len(train_player_ids)),
            "n_val_players": int(len(val_player_ids)),
            "source": str(split_path),
        },
    }


def _pad_or_truncate_features(
    features: np.ndarray,
    seq_length: int,
    pad_value: float = PAD_VALUE,
) -> np.ndarray:
    """Pad or truncate one feature sequence to a fixed length."""
    if seq_length <= 0:
        return features.astype(np.float32, copy=False)
    n_rows, n_features = features.shape
    if n_rows >= seq_length:
        return features[:seq_length].astype(np.float32, copy=False)
    padding = np.full((seq_length - n_rows, n_features), pad_value, dtype=np.float32)
    return np.concatenate([features.astype(np.float32, copy=False), padding], axis=0)


def _normalize_cutoff_schedule_steps(cutoff_schedule_steps: Sequence[int]) -> List[int]:
    """Normalize a raw cutoff-step schedule to a sorted positive unique list."""
    normalized = sorted({int(step) for step in cutoff_schedule_steps if int(step) > 0})
    if not normalized:
        raise ValueError("cutoff_schedule_steps must contain at least one positive integer.")
    return normalized


def _sample_cutoff_steps_for_player(
    max_observed_steps: int,
    cutoff_schedule_steps: Sequence[int],
    cutoff_samples_per_player: int,
    rng: np.random.RandomState,
) -> List[int]:
    """Sample cutoff steps for one player from a fixed schedule without replacement."""
    valid_cutoffs = [
        int(cutoff_step)
        for cutoff_step in cutoff_schedule_steps
        if int(cutoff_step) <= int(max_observed_steps)
    ]
    if not valid_cutoffs:
        return []
    if cutoff_samples_per_player <= 0 or cutoff_samples_per_player >= len(valid_cutoffs):
        return valid_cutoffs
    sampled_indices = np.sort(
        rng.choice(
            len(valid_cutoffs),
            size=int(cutoff_samples_per_player),
            replace=False,
        )
    )
    return [valid_cutoffs[int(index)] for index in sampled_indices.tolist()]


def _candidate_digest(candidates: Sequence[MinigridPLVMSampleCandidate]) -> str:
    """Return a stable digest for one ordered minigrid PLVM candidate list."""
    payload = json.dumps([candidate.to_dict() for candidate in candidates], sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]


def _minigrid_plvm_cache_key(
    *,
    cache_version: int,
    data_dir: str,
    model_type: str,
    task_names: Sequence[str],
    task_model_signature: Dict[str, dict],
    embedding_seq_length: int,
    embeddings_to_use: str,
    cutoff_schedule_steps: Sequence[int],
    cutoff_samples_per_player: int,
    max_train_val_players: int,
    val_fraction: float,
    seed: int,
    player_split_manifest: str,
    candidate_digest: str,
) -> str:
    """Return a stable cache key for minigrid PLVM embedding materialization."""
    payload = {
        "cache_version": int(cache_version),
        "data_dir": str(data_dir),
        "model_type": str(model_type),
        "task_names": [str(task_name) for task_name in task_names],
        "task_model_signature": task_model_signature,
        "embedding_seq_length": int(embedding_seq_length),
        "embeddings_to_use": str(embeddings_to_use),
        "cutoff_schedule_steps": [int(step) for step in cutoff_schedule_steps],
        "cutoff_samples_per_player": int(cutoff_samples_per_player),
        "max_train_val_players": int(max_train_val_players),
        "val_fraction": float(val_fraction),
        "seed": int(seed),
        "player_split_manifest": str(player_split_manifest or ""),
        "candidate_digest": str(candidate_digest),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()[:16]


def _path_fingerprint(path: Path) -> Dict[str, object]:
    """Return a stable file fingerprint for cache invalidation."""
    path = Path(path).expanduser().resolve()
    try:
        stat = path.stat()
        return {
            "path": str(path),
            "mtime": int(stat.st_mtime),
            "size": int(stat.st_size),
        }
    except OSError:
        return {
            "path": str(path),
            "mtime": -1,
            "size": -1,
        }


def _minigrid_plvm_dataset_cache_key(
    *,
    data_dir: Path,
    task_names: Sequence[str],
    task_model_signature: Dict[str, dict],
    cutoff_schedule_steps: Sequence[int],
    cutoff_samples_per_player: int,
    max_train_val_players: int,
    val_fraction: float,
    seed: int,
    player_split_manifest: str,
    embedding_seq_length: int,
) -> str:
    """Return a stable cache key for pre-embedding minigrid PLVM dataset materialization."""
    data_dir = Path(data_dir).expanduser().resolve()
    payload = {
        "cache_version": int(MINIGRID_PLVM_DATASET_CACHE_VERSION),
        "data_dir": str(data_dir),
        "task_names": [str(task_name) for task_name in task_names],
        "task_model_signature": task_model_signature,
        "cutoff_schedule_steps": [int(step) for step in cutoff_schedule_steps],
        "cutoff_samples_per_player": int(cutoff_samples_per_player),
        "max_train_val_players": int(max_train_val_players),
        "val_fraction": float(val_fraction),
        "seed": int(seed),
        "player_split_manifest": str(player_split_manifest or ""),
        "embedding_seq_length": int(embedding_seq_length),
        "input_files": {
            "players": _path_fingerprint(data_dir / "players.csv"),
            "events": _path_fingerprint(data_dir / "events.csv"),
            "environment_metadata": _path_fingerprint(data_dir / "environment_metadata.json"),
            "player_split_manifest": (
                _path_fingerprint(Path(player_split_manifest)) if player_split_manifest else None
            ),
        },
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()[:16]


def _save_minigrid_plvm_dataset_cache(cache_file: Path, payload: Dict[str, object]) -> None:
    """Persist a materialized paired-player dataset for fast head-only reruns."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    serializable_payload = dict(payload)
    serializable_payload["samples"] = [sample.to_dict() for sample in payload["samples"]]
    torch.save(serializable_payload, cache_file)


def _load_minigrid_plvm_dataset_cache(cache_file: Path) -> Optional[Dict[str, object]]:
    """Load a cached paired-player dataset, or return None on cache miss/load error."""
    if not cache_file.exists():
        return None
    try:
        cached = torch.load(cache_file, map_location="cpu")
        cached["samples"] = [
            MinigridPLVMSampleCandidate(**sample_payload) for sample_payload in cached["samples"]
        ]
        return cached
    except Exception as exc:
        logging.warning("Failed to load minigrid PLVM dataset cache %s: %s", cache_file, exc)
        return None


def build_minigrid_cutoff_sample_payload(
    players_df: pd.DataFrame,
    *,
    available_player_ids: Sequence[int],
    max_observed_steps_by_player: Dict[int, int],
    cutoff_schedule_steps: Optional[Sequence[int]] = None,
    cutoff_samples_per_player: int,
    max_train_val_players: int,
    val_fraction: float,
    seed: int,
    player_split_manifest: Optional[Path] = None,
) -> Dict[str, object]:
    """Build balanced train/val cutoff samples for eligible minigrid players."""
    if cutoff_samples_per_player == 0:
        raise ValueError("cutoff_samples_per_player cannot be 0.")

    normalized_cutoff_steps = _normalize_cutoff_schedule_steps(
        cutoff_schedule_steps or DEFAULT_CUTOFF_SCHEDULE_STEPS
    )

    if player_split_manifest:
        split_payload = load_player_split_manifest(player_split_manifest)
    else:
        split_payload = build_player_splits(
            players_df,
            max_train_val_players=max_train_val_players,
            val_fraction=val_fraction,
            seed=seed,
        )

    selected_player_ids = sorted(
        set(split_payload["train_player_ids"]) | set(split_payload["val_player_ids"])
    )
    if not selected_player_ids:
        raise ValueError("No train/val players were selected for training.")

    available_player_id_set = {int(player_id) for player_id in available_player_ids}
    available_selected_player_ids = sorted(
        int(player_id)
        for player_id in selected_player_ids
        if int(player_id) in available_player_id_set
    )
    if not available_selected_player_ids:
        raise ValueError("No eligible players remain after availability filtering.")

    missing_selected_ids = sorted(set(selected_player_ids) - available_player_id_set)
    effective_train_ids = [
        int(player_id)
        for player_id in split_payload["train_player_ids"]
        if int(player_id) in available_player_id_set
    ]
    effective_val_ids = [
        int(player_id)
        for player_id in split_payload["val_player_ids"]
        if int(player_id) in available_player_id_set
    ]
    effective_ignored_ids = sorted(
        set(int(player_id) for player_id in split_payload["ignored_player_ids"])
        | set(missing_selected_ids)
    )
    if not effective_train_ids:
        raise ValueError("No train players remain after availability filtering.")
    if not effective_val_ids:
        raise ValueError("No validation players remain after availability filtering.")

    player_rows = (
        players_df.loc[
            :,
            ["player_id", "label", "class_id", "latent_P", "latent_O"],
        ]
        .drop_duplicates(subset=["player_id"])
        .copy()
    )
    player_rows["player_id"] = player_rows["player_id"].astype(int)
    player_index = player_rows.set_index("player_id")

    cutoff_rng = np.random.RandomState(seed)
    samples: List[MinigridPLVMSampleCandidate] = []
    player_ids_with_samples: List[int] = []
    skipped_for_cutoff_player_ids: List[int] = []
    sampled_cutoff_histogram = {int(cutoff_step): 0 for cutoff_step in normalized_cutoff_steps}

    for player_id in available_selected_player_ids:
        if player_id not in player_index.index:
            raise KeyError("Player {} missing from players.csv".format(player_id))
        if player_id not in max_observed_steps_by_player:
            raise KeyError("Player {} missing cutoff-length metadata.".format(player_id))

        row = player_index.loc[player_id]
        sampled_cutoff_steps = _sample_cutoff_steps_for_player(
            max_observed_steps=int(max_observed_steps_by_player[player_id]),
            cutoff_schedule_steps=normalized_cutoff_steps,
            cutoff_samples_per_player=cutoff_samples_per_player,
            rng=cutoff_rng,
        )
        if not sampled_cutoff_steps:
            skipped_for_cutoff_player_ids.append(int(player_id))
            continue

        player_ids_with_samples.append(int(player_id))
        for cutoff_steps in sampled_cutoff_steps:
            samples.append(
                MinigridPLVMSampleCandidate(
                    player_id=int(player_id),
                    label=str(row["label"]),
                    class_id=int(row["class_id"]),
                    latent_P=float(row["latent_P"]),
                    latent_O=float(row["latent_O"]),
                    cutoff_steps=int(cutoff_steps),
                )
            )
            sampled_cutoff_histogram[int(cutoff_steps)] += 1

    player_ids_with_samples = sorted(set(player_ids_with_samples))
    if not player_ids_with_samples:
        raise ValueError(
            "No cutoff samples were available. Check cutoff_schedule_steps={} against sequence lengths.".format(
                normalized_cutoff_steps
            )
        )

    effective_train_ids = [
        int(player_id)
        for player_id in effective_train_ids
        if int(player_id) in player_ids_with_samples
    ]
    effective_val_ids = [
        int(player_id)
        for player_id in effective_val_ids
        if int(player_id) in player_ids_with_samples
    ]
    effective_ignored_ids = sorted(set(effective_ignored_ids) | set(skipped_for_cutoff_player_ids))
    if not effective_train_ids:
        raise ValueError("No train players remain after cutoff-based sample construction.")
    if not effective_val_ids:
        raise ValueError("No validation players remain after cutoff-based sample construction.")

    split_rows = []
    train_set = set(effective_train_ids)
    val_set = set(effective_val_ids)
    ignored_set = set(effective_ignored_ids)
    for _, row in player_rows.sort_values("player_id", kind="stable").iterrows():
        player_id = int(row["player_id"])
        if player_id in train_set:
            split_name = "train"
        elif player_id in val_set:
            split_name = "val"
        elif player_id in ignored_set:
            split_name = "ignored"
        else:
            continue
        split_rows.append(
            {
                "player_id": player_id,
                "label": str(row["label"]),
                "split": split_name,
            }
        )
    effective_split_df = pd.DataFrame(split_rows).sort_values(
        ["split", "player_id"],
        kind="stable",
    )

    return {
        "samples": samples,
        "cutoff_schedule_steps": [int(cutoff_step) for cutoff_step in normalized_cutoff_steps],
        "cutoff_samples_per_player": int(cutoff_samples_per_player),
        "cutoff_histogram": {
            str(cutoff_step): int(count)
            for cutoff_step, count in sampled_cutoff_histogram.items()
            if int(count) > 0
        },
        "split_payload": {
            "train_player_ids": effective_train_ids,
            "val_player_ids": effective_val_ids,
            "ignored_player_ids": effective_ignored_ids,
            "split_df": effective_split_df.reset_index(drop=True),
            "summary": {
                "n_total_players": int(player_rows["player_id"].nunique()),
                "n_train_val_players": int(len(effective_train_ids) + len(effective_val_ids)),
                "n_ignored_players": int(len(effective_ignored_ids)),
                "n_train_players": int(len(effective_train_ids)),
                "n_val_players": int(len(effective_val_ids)),
                "n_missing_selected_players": int(len(missing_selected_ids)),
                "n_players_with_cutoff_samples": int(len(player_ids_with_samples)),
                "n_players_dropped_for_cutoff_schedule": int(len(skipped_for_cutoff_player_ids)),
                "missing_selected_player_ids": [
                    int(player_id) for player_id in missing_selected_ids
                ],
                "cutoff_dropped_player_ids": [
                    int(player_id) for player_id in skipped_for_cutoff_player_ids
                ],
            },
        },
    }


def build_paired_player_dataset(
    data_dir: Path,
    loaded_task_models: Dict[str, dict],
    *,
    selected_task_names: Optional[Sequence[str]] = None,
    cutoff_schedule_steps: Optional[Sequence[int]] = None,
    cutoff_samples_per_player: int,
    max_train_val_players: int,
    val_fraction: float,
    seed: int,
    player_split_manifest: Optional[Path] = None,
    embedding_seq_length: int = 0,
    cache_dir: Optional[str] = None,
) -> Dict[str, object]:
    """Materialize cutoff-conditioned task-prefix samples per eligible player."""
    data_dir = Path(data_dir).expanduser().resolve()
    players_path = data_dir / "players.csv"
    if not players_path.exists():
        raise FileNotFoundError("Missing players.csv under {}".format(data_dir))
    selected_task_names = list(selected_task_names or PLVM_TASK_NAMES)
    if not selected_task_names:
        raise ValueError("selected_task_names must contain at least one task.")
    unsupported_tasks = sorted(set(selected_task_names).difference(PLVM_TASK_NAMES))
    if unsupported_tasks:
        raise ValueError(
            "Unsupported minigrid task(s): {}. Expected subset of {}.".format(
                unsupported_tasks,
                list(PLVM_TASK_NAMES),
            )
        )

    normalized_cutoff_steps = _normalize_cutoff_schedule_steps(
        cutoff_schedule_steps or DEFAULT_CUTOFF_SCHEDULE_STEPS
    )
    task_model_signature = {
        task_name: {
            "model_name": str(loaded_task_models[task_name]["model_name"]),
            "model_log_dir": str(loaded_task_models[task_name]["model_log_dir"]),
            "hidden_size": int(loaded_task_models[task_name]["hidden_size"]),
            "seq_length": int(loaded_task_models[task_name].get("seq_length", 0) or 0),
        }
        for task_name in selected_task_names
    }
    dataset_cache_file: Optional[Path] = None
    if cache_dir:
        dataset_cache_key = _minigrid_plvm_dataset_cache_key(
            data_dir=data_dir,
            task_names=selected_task_names,
            task_model_signature=task_model_signature,
            cutoff_schedule_steps=normalized_cutoff_steps,
            cutoff_samples_per_player=cutoff_samples_per_player,
            max_train_val_players=max_train_val_players,
            val_fraction=val_fraction,
            seed=seed,
            player_split_manifest=str(player_split_manifest or ""),
            embedding_seq_length=embedding_seq_length,
        )
        dataset_cache_file = (
            Path(cache_dir).expanduser().resolve()
            / "datasets"
            / "minigrid_plvm_dataset_{}.pt".format(dataset_cache_key)
        )
        cached_payload = _load_minigrid_plvm_dataset_cache(dataset_cache_file)
        if cached_payload is not None:
            logging.info("Loaded minigrid PLVM dataset cache from %s", dataset_cache_file)
            return cached_payload

    players_df = pd.read_csv(players_path)
    if player_split_manifest:
        split_payload = load_player_split_manifest(player_split_manifest)
        selected_player_ids = sorted(
            set(split_payload["train_player_ids"]) | set(split_payload["val_player_ids"])
        )
    else:
        split_payload = build_player_splits(
            players_df,
            max_train_val_players=max_train_val_players,
            val_fraction=val_fraction,
            seed=seed,
        )
        selected_player_ids = sorted(
            set(split_payload["train_player_ids"]) | set(split_payload["val_player_ids"])
        )
    if not selected_player_ids:
        raise ValueError("No train/val players were selected for PLVM training.")

    example_maps: Dict[str, Dict[int, object]] = {}
    task_sequence_stats: Dict[str, object] = {}
    task_pad_lengths: Dict[str, int] = {}
    for task_name in selected_task_names:
        _, task_events = load_simulation_tables(data_dir, task_name)
        layout = load_task_layout(data_dir, task_name)
        examples, stats = build_minigrid_sequence_examples(
            task_events,
            player_ids=selected_player_ids,
            layout=layout,
            manifest=loaded_task_models[task_name]["manifest"],
            max_seq_length=0,
        )
        example_map: Dict[int, object] = {}
        for example in examples:
            player_id = int(example.metadata.player_id)
            if player_id in example_map:
                raise ValueError(
                    "Expected one full-sequence example per player for task '{}', got duplicate player_id={}.".format(
                        task_name,
                        player_id,
                    )
                )
            example_map[player_id] = example
        if not example_map:
            raise ValueError("No sequence examples were built for task '{}'.".format(task_name))
        example_maps[task_name] = example_map
        task_sequence_stats[task_name] = stats
        task_pad_lengths[task_name] = max(
            (
                int(embedding_seq_length)
                if embedding_seq_length > 0
                else max(int(example.effective_length) for example in example_map.values())
            ),
            max(
                _normalize_cutoff_schedule_steps(
                    cutoff_schedule_steps or DEFAULT_CUTOFF_SCHEDULE_STEPS
                )
            ),
        )

    available_player_ids = set(selected_player_ids)
    for example_map in example_maps.values():
        available_player_ids.intersection_update(example_map.keys())
    available_player_ids = sorted(int(player_id) for player_id in available_player_ids)
    if not available_player_ids:
        raise ValueError("No paired task sequences were available for the selected players.")

    sample_payload = build_minigrid_cutoff_sample_payload(
        players_df,
        available_player_ids=available_player_ids,
        max_observed_steps_by_player={
            int(player_id): min(
                int(example_maps[task_name][int(player_id)].effective_length)
                for task_name in selected_task_names
            )
            for player_id in available_player_ids
        },
        cutoff_schedule_steps=cutoff_schedule_steps,
        cutoff_samples_per_player=cutoff_samples_per_player,
        max_train_val_players=max_train_val_players,
        val_fraction=val_fraction,
        seed=seed,
        player_split_manifest=player_split_manifest,
    )

    samples = list(sample_payload["samples"])
    session_task_data_x: List[List[torch.Tensor]] = []
    for sample in samples:
        task_tensors: List[torch.Tensor] = []
        for task_name in selected_task_names:
            example = example_maps[task_name][int(sample.player_id)]
            padded = _pad_or_truncate_features(
                example.features[: int(sample.cutoff_steps)],
                seq_length=task_pad_lengths[task_name],
                pad_value=PAD_VALUE,
            )
            task_tensors.append(torch.as_tensor(padded, dtype=torch.float32))
        session_task_data_x.append(task_tensors)

    dataset_payload = {
        "samples": samples,
        "session_task_data_x": session_task_data_x,
        "selected_task_names": list(selected_task_names),
        "cutoff_schedule_steps": sample_payload["cutoff_schedule_steps"],
        "cutoff_samples_per_player": sample_payload["cutoff_samples_per_player"],
        "cutoff_histogram": sample_payload["cutoff_histogram"],
        "split_payload": sample_payload["split_payload"],
        "task_sequence_stats": task_sequence_stats,
        "task_pad_lengths": {task: int(length) for task, length in task_pad_lengths.items()},
        "data_dir": str(data_dir),
    }
    if dataset_cache_file is not None:
        try:
            _save_minigrid_plvm_dataset_cache(dataset_cache_file, dataset_payload)
            logging.info("Saved minigrid PLVM dataset cache to %s", dataset_cache_file)
        except Exception as exc:
            logging.warning(
                "Failed to save minigrid PLVM dataset cache %s: %s", dataset_cache_file, exc
            )
    return dataset_payload


def _pool_encoded_sequence(
    encoded: torch.Tensor,
    padding_mask: torch.Tensor,
    embeddings_to_use: str,
) -> torch.Tensor:
    """Pool one encoded task sequence to a single fixed-width embedding."""
    embeddings_to_use = str(embeddings_to_use or "masked_average_all").strip().lower()
    if embeddings_to_use not in SUPPORTED_EMBEDDING_POOLERS:
        raise ValueError(
            "Unsupported embeddings_to_use '{}'. Expected one of {}.".format(
                embeddings_to_use,
                sorted(SUPPORTED_EMBEDDING_POOLERS),
            )
        )

    valid_mask = ~padding_mask
    batch_size, _, hidden_size = encoded.shape
    valid_counts = valid_mask.sum(dim=1)
    has_valid = valid_counts > 0
    batch_indices = torch.arange(batch_size, device=encoded.device)

    if embeddings_to_use == "last_only":
        last_indices = torch.clamp(valid_counts - 1, min=0)
        pooled = encoded[batch_indices, last_indices, :]
        return torch.where(has_valid.unsqueeze(-1), pooled, torch.zeros_like(pooled))

    if embeddings_to_use in {"average_all", "masked_average_all"}:
        mask_float = valid_mask.unsqueeze(-1).to(encoded.dtype)
        denom = mask_float.sum(dim=1).clamp_min(1.0)
        pooled = (encoded * mask_float).sum(dim=1) / denom
        return torch.where(has_valid.unsqueeze(-1), pooled, torch.zeros_like(pooled))

    if embeddings_to_use in {"average_last_10", "average_last_20", "average_last_30"}:
        trailing_count = int(embeddings_to_use.rsplit("_", 1)[-1])
        pooled_rows = []
        for batch_index in range(batch_size):
            current_count = int(valid_counts[batch_index].item())
            if current_count <= 0:
                pooled_rows.append(
                    torch.zeros(hidden_size, device=encoded.device, dtype=encoded.dtype)
                )
                continue
            start_index = max(0, current_count - trailing_count)
            pooled_rows.append(encoded[batch_index, start_index:current_count, :].mean(dim=0))
        return torch.stack(pooled_rows, dim=0)

    if embeddings_to_use == "first_only":
        first_indices = valid_mask.float().argmax(dim=1)
        pooled = encoded[batch_indices, first_indices, :]
        return torch.where(has_valid.unsqueeze(-1), pooled, torch.zeros_like(pooled))

    if embeddings_to_use == "first_last_concat":
        first_indices = valid_mask.float().argmax(dim=1)
        last_indices = torch.clamp(valid_counts - 1, min=0)
        first_tokens = encoded[batch_indices, first_indices, :]
        last_tokens = encoded[batch_indices, last_indices, :]
        first_tokens = torch.where(
            has_valid.unsqueeze(-1),
            first_tokens,
            torch.zeros_like(first_tokens),
        )
        last_tokens = torch.where(
            has_valid.unsqueeze(-1),
            last_tokens,
            torch.zeros_like(last_tokens),
        )
        return torch.cat([first_tokens, last_tokens], dim=-1)

    if embeddings_to_use == "last_only_plus_count":
        last_indices = torch.clamp(valid_counts - 1, min=0)
        last_tokens = encoded[batch_indices, last_indices, :]
        last_tokens = torch.where(
            has_valid.unsqueeze(-1),
            last_tokens,
            torch.zeros_like(last_tokens),
        )
        seq_len = max(valid_mask.shape[1], 1)
        count_feature = (valid_counts.to(encoded.dtype) / float(seq_len)).unsqueeze(-1)
        return torch.cat([last_tokens, count_feature], dim=-1)

    raise AssertionError("Unsupported embeddings_to_use: {}".format(embeddings_to_use))


def precompute_minigrid_paired_embeddings(
    loaded_task_models: Dict[str, dict],
    session_task_data_x: Sequence[Sequence[torch.Tensor]],
    *,
    task_names: Optional[Sequence[str]] = None,
    embeddings_to_use: str,
    batch_size: int,
) -> torch.Tensor:
    """Precompute one concatenated embedding across the requested task branches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not session_task_data_x:
        raise ValueError("No paired task sequences were provided.")
    task_order = list(task_names or PLVM_TASK_NAMES)
    if not task_order:
        raise ValueError("task_names must contain at least one task.")

    all_embeddings: List[torch.Tensor] = []

    with torch.no_grad():
        for start_index in range(0, len(session_task_data_x), batch_size):
            batch_sessions = session_task_data_x[start_index : start_index + batch_size]
            pooled_per_task: List[torch.Tensor] = []

            for task_index, task_name in enumerate(task_order):
                task_model = loaded_task_models[task_name]["model"]
                task_batch = torch.stack(
                    [session[task_index] for session in batch_sessions],
                    dim=0,
                ).to(COMPUTE_DEVICE)
                padding_mask = build_padding_mask(
                    task_batch,
                    fill_value=PAD_VALUE,
                    extra_fill_values=[FILL_MISSING_VALUE],
                )
                encoded = task_model.get_embedding(task_batch, padding_mask=padding_mask)
                pooled = _pool_encoded_sequence(encoded, padding_mask, embeddings_to_use)
                pooled_per_task.append(pooled.detach().cpu())

            all_embeddings.append(torch.cat(pooled_per_task, dim=-1))

    return torch.cat(all_embeddings, dim=0)


def materialize_minigrid_plvm_embeddings(
    loaded_task_models: Dict[str, dict],
    session_task_data_x: Sequence[Sequence[torch.Tensor]],
    sample_candidates: Sequence[MinigridPLVMSampleCandidate],
    *,
    task_names: Optional[Sequence[str]] = None,
    embeddings_to_use: str,
    batch_size: int,
    cache_dir: Optional[str] = None,
    data_dir: str = "",
    model_type: str = "",
    embedding_seq_length: int = 0,
    cutoff_schedule_steps: Optional[Sequence[int]] = None,
    cutoff_samples_per_player: int = 0,
    max_train_val_players: int = 0,
    val_fraction: float = 0.0,
    seed: int = 0,
    player_split_manifest: str = "",
) -> Tuple[torch.Tensor, torch.Tensor, List[MinigridPLVMSampleCandidate]]:
    """Materialize and optionally cache minigrid PLVM embeddings for one sample set."""
    candidates = list(sample_candidates)
    if not candidates:
        raise ValueError("Cannot materialize an empty minigrid PLVM sample list.")

    task_order = list(task_names or PLVM_TASK_NAMES)
    candidate_digest = _candidate_digest(candidates)
    task_model_signature = {
        task_name: {
            "model_name": str(loaded_task_models[task_name]["model_name"]),
            "model_log_dir": str(loaded_task_models[task_name]["model_log_dir"]),
            "hidden_size": int(loaded_task_models[task_name]["hidden_size"]),
            "seq_length": int(loaded_task_models[task_name].get("seq_length", 0) or 0),
        }
        for task_name in task_order
    }

    cache_file: Optional[Path] = None
    if cache_dir:
        cache_key = _minigrid_plvm_cache_key(
            cache_version=MINIGRID_PLVM_EMBEDDING_CACHE_VERSION,
            data_dir=str(data_dir),
            model_type=str(model_type),
            task_names=task_order,
            task_model_signature=task_model_signature,
            embedding_seq_length=int(embedding_seq_length),
            embeddings_to_use=str(embeddings_to_use),
            cutoff_schedule_steps=list(cutoff_schedule_steps or []),
            cutoff_samples_per_player=int(cutoff_samples_per_player),
            max_train_val_players=int(max_train_val_players),
            val_fraction=float(val_fraction),
            seed=int(seed),
            player_split_manifest=str(player_split_manifest or ""),
            candidate_digest=candidate_digest,
        )
        cache_file = Path(
            cache_dir
        ).expanduser().resolve() / "minigrid_plvm_embeddings_{}.pt".format(cache_key)
        if cache_file.exists():
            try:
                cached = torch.load(cache_file, map_location="cpu")
                if cached.get("candidate_digest") == candidate_digest:
                    logging.info("Loaded minigrid PLVM embedding cache from %s", cache_file)
                    return (
                        cached["embeddings"],
                        cached["targets"],
                        [
                            MinigridPLVMSampleCandidate(**payload)
                            for payload in cached["sample_candidates"]
                        ],
                    )
            except Exception as exc:
                logging.warning("Failed to load minigrid PLVM cache %s: %s", cache_file, exc)

    embeddings = precompute_minigrid_paired_embeddings(
        loaded_task_models,
        session_task_data_x,
        task_names=task_order,
        embeddings_to_use=embeddings_to_use,
        batch_size=batch_size,
    )
    targets_tensor = torch.tensor(
        [candidate.class_id for candidate in candidates],
        dtype=torch.long,
    ).unsqueeze(-1)

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(
                {
                    "embeddings": embeddings.cpu(),
                    "targets": targets_tensor.cpu(),
                    "sample_candidates": [candidate.to_dict() for candidate in candidates],
                    "candidate_digest": candidate_digest,
                    "data_dir": str(data_dir),
                    "model_type": str(model_type),
                    "task_names": [str(task_name) for task_name in task_order],
                    "task_model_signature": task_model_signature,
                    "embedding_seq_length": int(embedding_seq_length),
                    "embeddings_to_use": str(embeddings_to_use),
                    "cutoff_schedule_steps": [int(step) for step in (cutoff_schedule_steps or [])],
                    "cutoff_samples_per_player": int(cutoff_samples_per_player),
                    "max_train_val_players": int(max_train_val_players),
                    "val_fraction": float(val_fraction),
                    "seed": int(seed),
                    "player_split_manifest": str(player_split_manifest or ""),
                },
                cache_file,
            )
            logging.info("Saved minigrid PLVM embedding cache to %s", cache_file)
        except Exception as exc:
            logging.warning("Failed to save minigrid PLVM cache %s: %s", cache_file, exc)

    return embeddings, targets_tensor, candidates
