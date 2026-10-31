"""Data utilities for task-specific transformer training on minigrid simulations."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from simulation.config import (
    LABEL_HOPPER,
    LABEL_PLANNER,
    TASK_NAME_CLEANING,
    TASK_NAME_FORAGING,
)
from plvm import artifact_paths
from plvm.model_training_utils import FILL_MISSING_VALUE


DEFAULT_SIMULATION_DATA_DIR = artifact_paths.SIMULATION_EPISODES_DIR
NEXT_ACTION_IGNORE_INDEX = -100

ACTION_VOCAB_BY_TASK = {
    TASK_NAME_FORAGING: ["UP", "DOWN", "LEFT", "RIGHT", "COLLECT"],
    TASK_NAME_CLEANING: ["UP", "DOWN", "LEFT", "RIGHT", "CLEAN"],
}
MOVE_DELTAS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}


@dataclass(frozen=True)
class TaskTargetMetadata:
    """Static metadata for one target in a shared task layout."""

    target_id: str
    x: int
    y: int
    patch_id: int
    target_type: str
    reward: float
    salience: float
    is_high_value: bool

    @property
    def position(self) -> Tuple[int, int]:
        """Return the `(x, y)` target position."""
        return (int(self.x), int(self.y))


@dataclass
class TaskLayoutMetadata:
    """Serialized task layout plus derived shortest-distance maps."""

    task_name: str
    grid_size: int
    start_pos: Tuple[int, int]
    terrain: np.ndarray
    obstacle_labels: Dict[int, str]
    targets: List[TaskTargetMetadata]
    terrain_levels: List[int]
    boundary_terrain_code: int
    distance_maps: Dict[str, np.ndarray]

    @property
    def targets_by_id(self) -> Dict[str, TaskTargetMetadata]:
        """Index targets by id."""
        return {target.target_id: target for target in self.targets}

    @property
    def total_reward(self) -> float:
        """Return the total reward obtainable in the task."""
        return float(sum(target.reward for target in self.targets))

    @property
    def max_reward(self) -> float:
        """Return the maximum per-target reward."""
        if not self.targets:
            return 1.0
        return float(max(target.reward for target in self.targets))

    @property
    def max_distance(self) -> float:
        """Return a conservative normalizer for shortest-path distances."""
        return float(max(1, self.grid_size * self.grid_size))

    def is_passable(self, x: int, y: int) -> bool:
        """Return whether a terrain cell is traversable."""
        return bool(
            0 <= x < self.grid_size and 0 <= y < self.grid_size and int(self.terrain[y, x]) == 0
        )


@dataclass
class MinigridFeatureManifest:
    """Task-specific feature metadata used to build transformer inputs."""

    task_name: str
    grid_size: int
    local_patch_radius: int
    feature_columns: List[str]
    action_vocab: List[str]
    action_to_index: Dict[str, int]
    terrain_levels: List[int]
    boundary_terrain_code: int
    target_ids: List[str]
    target_rewards: Dict[str, float]
    target_saliences: Dict[str, float]
    target_high_value_flags: Dict[str, int]
    max_distance: float
    max_reward: float
    max_salience: float
    total_targets: int
    total_possible_reward: float

    @property
    def feature_dim(self) -> int:
        """Return the input feature dimension."""
        return len(self.feature_columns)

    def to_dict(self) -> Dict[str, object]:
        """Serialize the manifest to a JSON-friendly payload."""
        return {
            "task_name": self.task_name,
            "grid_size": int(self.grid_size),
            "local_patch_radius": int(self.local_patch_radius),
            "feature_columns": list(self.feature_columns),
            "action_vocab": list(self.action_vocab),
            "action_to_index": {
                str(key): int(value) for key, value in self.action_to_index.items()
            },
            "terrain_levels": [int(value) for value in self.terrain_levels],
            "boundary_terrain_code": int(self.boundary_terrain_code),
            "target_ids": list(self.target_ids),
            "target_rewards": {
                str(key): float(value) for key, value in self.target_rewards.items()
            },
            "target_saliences": {
                str(key): float(value) for key, value in self.target_saliences.items()
            },
            "target_high_value_flags": {
                str(key): int(value) for key, value in self.target_high_value_flags.items()
            },
            "max_distance": float(self.max_distance),
            "max_reward": float(self.max_reward),
            "max_salience": float(self.max_salience),
            "total_targets": int(self.total_targets),
            "total_possible_reward": float(self.total_possible_reward),
        }


@dataclass(frozen=True)
class MinigridSequenceMetadata:
    """Metadata describing one player-task chunk."""

    sequence_id: str
    player_id: int
    task_name: str
    label: str
    chunk_index: int


@dataclass
class MinigridSequenceExample:
    """One variable-length transformer example."""

    metadata: MinigridSequenceMetadata
    features: np.ndarray
    targets: np.ndarray
    effective_length: int


class MinigridSequenceDataset(Dataset):
    """Torch dataset over variable-length minigrid sequence examples."""

    def __init__(self, examples: Sequence[MinigridSequenceExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        return (
            torch.as_tensor(example.features, dtype=torch.float32),
            torch.as_tensor(example.targets, dtype=torch.long),
            example.effective_length,
            example.metadata.sequence_id,
        )


def save_feature_manifest(manifest: MinigridFeatureManifest, path: Path) -> None:
    """Persist the fitted feature manifest for later inspection."""
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)


def minigrid_sequence_collate_fn(batch):
    """Pad variable-length feature and label sequences for transformer training."""
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    batch_size = len(batch)
    max_length = max(int(item[2]) for item in batch)
    feature_dim = int(batch[0][0].shape[-1])

    padded_x = torch.full(
        (batch_size, max_length, feature_dim),
        fill_value=float(FILL_MISSING_VALUE),
        dtype=torch.float32,
    )
    padded_y = torch.full(
        (batch_size, max_length),
        fill_value=int(NEXT_ACTION_IGNORE_INDEX),
        dtype=torch.long,
    )
    lengths: List[int] = []
    sequence_ids: List[str] = []

    for row_index, (features, targets, effective_length, sequence_id) in enumerate(batch):
        effective_length = int(effective_length)
        padded_x[row_index, :effective_length] = features[:effective_length]
        padded_y[row_index, :effective_length] = targets[:effective_length]
        lengths.append(effective_length)
        sequence_ids.append(sequence_id)

    return padded_x, padded_y, lengths, sequence_ids


def load_players_table(data_dir: Path) -> pd.DataFrame:
    """Load only the shared `players.csv` table from one simulation output directory."""
    data_dir = Path(data_dir).expanduser().resolve()
    players_path = data_dir / "players.csv"
    if not players_path.exists():
        raise FileNotFoundError(f"Missing players.csv under {data_dir}")
    return pd.read_csv(players_path)


def load_task_events(
    data_dir: Path,
    task_name: str,
    *,
    usecols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load one task's event rows, optionally reading only a minimal column subset."""
    data_dir = Path(data_dir).expanduser().resolve()
    events_path = data_dir / "events.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events.csv under {data_dir}")

    required_columns = ["player_id", "task_name", "event_index"]
    if usecols is None:
        read_columns = None
    else:
        read_columns = list(dict.fromkeys([*required_columns, *[str(col) for col in usecols]]))

    events_df = pd.read_csv(events_path, usecols=read_columns)
    events_df = events_df.loc[events_df["task_name"] == task_name].copy()
    if events_df.empty:
        raise ValueError(f"No events found for task '{task_name}' under {data_dir}")
    events_df = events_df.sort_values(["player_id", "event_index"], kind="stable").reset_index(
        drop=True
    )
    return events_df


def load_simulation_tables(data_dir: Path, task_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load players and task-filtered events from one simulation output directory."""
    players_df = load_players_table(data_dir)
    events_df = load_task_events(data_dir, task_name)
    return players_df, events_df


def load_task_layout(data_dir: Path, task_name: str) -> TaskLayoutMetadata:
    """Load the shared task layout and derive distance maps from every target."""
    metadata_path = Path(data_dir).expanduser().resolve() / "environment_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            "Missing environment_metadata.json under {}. Re-run simulation.simulate to create it.".format(
                metadata_path.parent
            )
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if task_name not in payload:
        raise KeyError(f"Task '{task_name}' not present in {metadata_path}")

    task_payload = payload[task_name]
    terrain = np.asarray(task_payload["terrain"], dtype=np.int16)
    if terrain.ndim != 2 or terrain.shape[0] != terrain.shape[1]:
        raise ValueError(f"Expected square 2-D terrain grid for task '{task_name}'")

    terrain_codes = sorted(int(value) for value in np.unique(terrain).tolist())
    boundary_code = max(terrain_codes) + 1 if terrain_codes else 1
    terrain_levels = terrain_codes + [boundary_code]

    targets = [
        TaskTargetMetadata(
            target_id=str(item["target_id"]),
            x=int(item["x"]),
            y=int(item["y"]),
            patch_id=int(item["patch_id"]),
            target_type=str(item["target_type"]),
            reward=float(item["reward"]),
            salience=float(item.get("salience", item.get("reward", 1.0))),
            is_high_value=bool(item["is_high_value"]),
        )
        for item in sorted(task_payload["targets"], key=lambda item: str(item["target_id"]))
    ]
    if not targets:
        raise ValueError(f"Task '{task_name}' has no targets in {metadata_path}")

    obstacle_labels = {
        int(key): str(value) for key, value in dict(task_payload["obstacle_labels"]).items()
    }
    layout = TaskLayoutMetadata(
        task_name=str(task_payload["task_name"]),
        grid_size=int(task_payload["grid_size"]),
        start_pos=(int(task_payload["start_pos"][0]), int(task_payload["start_pos"][1])),
        terrain=terrain,
        obstacle_labels=obstacle_labels,
        targets=targets,
        terrain_levels=terrain_levels,
        boundary_terrain_code=int(boundary_code),
        distance_maps={},
    )
    for target in layout.targets:
        layout.distance_maps[target.target_id] = _build_distance_map(
            layout.terrain, target.position
        )
    return layout


def build_player_splits(
    players_df: pd.DataFrame,
    *,
    max_train_val_players: int,
    val_fraction: float,
    seed: int,
) -> Dict[str, object]:
    """Split players into train/val using a balanced random Planner/Hopper subset cap."""
    if max_train_val_players <= 0:
        raise ValueError("max_train_val_players must be positive.")
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError("val_fraction must be in (0, 1).")

    required_columns = {"player_id", "label"}
    missing_columns = required_columns.difference(players_df.columns)
    if missing_columns:
        raise ValueError(f"players_df is missing required columns: {sorted(missing_columns)}")

    ordered_players = players_df.loc[:, ["player_id", "label"]].drop_duplicates().copy()
    ordered_players["player_id"] = ordered_players["player_id"].astype(int)
    ordered_players = ordered_players.sort_values("player_id", kind="stable").reset_index(drop=True)
    if len(ordered_players) < 2:
        raise ValueError("Need at least two players to form train/val splits.")

    label_order = [LABEL_PLANNER, LABEL_HOPPER]
    observed_labels = set(ordered_players["label"].astype(str))
    missing_labels = [label for label in label_order if label not in observed_labels]
    if missing_labels:
        raise ValueError(
            "Balanced minigrid splits require both Planner and Hopper labels. Missing: {}".format(
                missing_labels
            )
        )

    requested_balanced_cap = min(int(max_train_val_players), int(len(ordered_players)))
    if requested_balanced_cap < 2:
        raise ValueError("Need at least two eligible players to form balanced splits.")
    effective_balanced_cap = requested_balanced_cap - (requested_balanced_cap % 2)
    if effective_balanced_cap < 2:
        raise ValueError(
            "Balanced minigrid splits require an even max_train_val_players of at least 2."
        )

    per_label_cap = effective_balanced_cap // 2
    rng = np.random.RandomState(seed)
    eligible_chunks: List[pd.DataFrame] = []
    for label in label_order:
        label_df = ordered_players.loc[ordered_players["label"].astype(str) == label].copy()
        if len(label_df) < per_label_cap:
            raise ValueError(
                "Requested max_train_val_players={} requires {} players per label, "
                "but label '{}' only has {} players.".format(
                    max_train_val_players,
                    per_label_cap,
                    label,
                    len(label_df),
                )
            )
        shuffled_indices = rng.permutation(len(label_df))[:per_label_cap]
        sampled_label_df = label_df.iloc[np.sort(shuffled_indices)].copy()
        eligible_chunks.append(sampled_label_df)

    eligible_df = (
        pd.concat(eligible_chunks, ignore_index=True)
        .sort_values("player_id", kind="stable")
        .reset_index(drop=True)
    )
    eligible_player_ids = set(eligible_df["player_id"].astype(int).tolist())
    ignored_df = ordered_players.loc[
        ~ordered_players["player_id"].astype(int).isin(eligible_player_ids)
    ].copy()

    val_player_ids: Set[int] = set()
    train_player_ids: Set[int] = set()

    for _, group in eligible_df.groupby("label", sort=True):
        group_ids = group["player_id"].astype(int).tolist()
        rng.shuffle(group_ids)
        if len(group_ids) == 1:
            train_player_ids.add(int(group_ids[0]))
            continue

        n_val = int(round(len(group_ids) * float(val_fraction)))
        n_val = max(1, n_val)
        n_val = min(len(group_ids) - 1, n_val)
        val_player_ids.update(int(player_id) for player_id in group_ids[:n_val])
        train_player_ids.update(int(player_id) for player_id in group_ids[n_val:])

    if not val_player_ids:
        fallback_player = int(eligible_df.iloc[-1]["player_id"])
        val_player_ids.add(fallback_player)
        train_player_ids.discard(fallback_player)
    if not train_player_ids:
        fallback_player = min(val_player_ids)
        train_player_ids.add(fallback_player)
        val_player_ids.discard(fallback_player)

    split_rows: List[Dict[str, object]] = []
    for _, row in eligible_df.iterrows():
        player_id = int(row["player_id"])
        split_rows.append(
            {
                "player_id": player_id,
                "label": str(row["label"]),
                "split": "val" if player_id in val_player_ids else "train",
            }
        )
    for _, row in ignored_df.iterrows():
        split_rows.append(
            {
                "player_id": int(row["player_id"]),
                "label": str(row["label"]),
                "split": "ignored",
            }
        )

    split_df = pd.DataFrame(split_rows).sort_values(["split", "player_id"], kind="stable")
    return {
        "train_player_ids": sorted(int(player_id) for player_id in train_player_ids),
        "val_player_ids": sorted(int(player_id) for player_id in val_player_ids),
        "ignored_player_ids": ignored_df["player_id"].astype(int).tolist(),
        "split_df": split_df.reset_index(drop=True),
        "summary": {
            "n_total_players": int(len(ordered_players)),
            "n_train_val_players": int(len(eligible_df)),
            "n_ignored_players": int(len(ignored_df)),
            "n_train_players": int(len(train_player_ids)),
            "n_val_players": int(len(val_player_ids)),
            "selected_label_counts": {
                str(label): int(
                    eligible_df.loc[
                        eligible_df["label"].astype(str) == label, "player_id"
                    ].nunique()
                )
                for label in label_order
            },
            "requested_max_train_val_players": int(max_train_val_players),
            "effective_max_train_val_players": int(len(eligible_df)),
            "selected_player_id_min": int(eligible_df["player_id"].min()),
            "selected_player_id_max": int(eligible_df["player_id"].max()),
        },
    }


def fit_minigrid_feature_manifest(
    *,
    task_name: str,
    layout: TaskLayoutMetadata,
    local_patch_radius: int,
) -> MinigridFeatureManifest:
    """Create a deterministic feature manifest for one task layout."""
    if task_name not in ACTION_VOCAB_BY_TASK:
        raise ValueError(f"Unsupported task_name: {task_name}")
    if local_patch_radius < 0:
        raise ValueError("local_patch_radius must be non-negative.")

    action_vocab = list(ACTION_VOCAB_BY_TASK[task_name])
    action_to_index = {action: index for index, action in enumerate(action_vocab)}
    target_ids = [target.target_id for target in layout.targets]
    target_rewards = {target.target_id: float(target.reward) for target in layout.targets}
    target_saliences = {target.target_id: float(target.salience) for target in layout.targets}
    target_high_value_flags = {
        target.target_id: int(bool(target.is_high_value)) for target in layout.targets
    }

    feature_columns = [
        "agent_x_norm",
        "agent_y_norm",
        "previous_x_norm",
        "previous_y_norm",
        "delta_x_norm",
        "delta_y_norm",
        "reward_received_norm",
        "cumulative_reward_norm",
        "remaining_targets_frac",
        "completed_targets_frac",
    ]
    feature_columns.extend(f"current_action__{action}" for action in action_vocab)
    feature_columns.extend(
        [
            "legal_move_up",
            "legal_move_down",
            "legal_move_left",
            "legal_move_right",
        ]
    )

    for dy in range(-local_patch_radius, local_patch_radius + 1):
        for dx in range(-local_patch_radius, local_patch_radius + 1):
            for terrain_code in layout.terrain_levels:
                feature_columns.append(
                    f"local_terrain_dx{dx:+d}_dy{dy:+d}_code_{int(terrain_code)}"
                )

    for target in layout.targets:
        prefix = f"target__{target.target_id}"
        feature_columns.extend(
            [
                f"{prefix}__remaining",
                f"{prefix}__rel_dx_norm",
                f"{prefix}__rel_dy_norm",
                f"{prefix}__distance_norm",
                f"{prefix}__reward_norm",
                f"{prefix}__is_high_value",
            ]
        )
        if task_name == TASK_NAME_CLEANING:
            feature_columns.append(f"{prefix}__salience_norm")

    return MinigridFeatureManifest(
        task_name=task_name,
        grid_size=int(layout.grid_size),
        local_patch_radius=int(local_patch_radius),
        feature_columns=feature_columns,
        action_vocab=action_vocab,
        action_to_index=action_to_index,
        terrain_levels=list(layout.terrain_levels),
        boundary_terrain_code=int(layout.boundary_terrain_code),
        target_ids=target_ids,
        target_rewards=target_rewards,
        target_saliences=target_saliences if task_name == TASK_NAME_CLEANING else {},
        target_high_value_flags=target_high_value_flags,
        max_distance=float(layout.max_distance),
        max_reward=float(layout.max_reward),
        max_salience=float(max((target.salience for target in layout.targets), default=1.0)),
        total_targets=int(len(layout.targets)),
        total_possible_reward=float(layout.total_reward),
    )


def build_minigrid_sequence_examples(
    events_df: pd.DataFrame,
    *,
    player_ids: Sequence[int],
    layout: TaskLayoutMetadata,
    manifest: MinigridFeatureManifest,
    max_seq_length: int,
) -> Tuple[List[MinigridSequenceExample], Dict[str, object]]:
    """Transform task event logs into player-level next-action sequence examples."""
    if max_seq_length < 0:
        raise ValueError("max_seq_length must be >= 0.")

    player_id_set = {int(player_id) for player_id in player_ids}
    filtered_df = events_df.loc[events_df["player_id"].astype(int).isin(player_id_set)].copy()
    if filtered_df.empty:
        raise ValueError("No events remain after filtering by player_ids.")
    filtered_df["player_id"] = filtered_df["player_id"].astype(int)
    filtered_df = filtered_df.sort_values(["player_id", "event_index"], kind="stable").reset_index(
        drop=True
    )

    examples: List[MinigridSequenceExample] = []
    raw_lengths: List[int] = []
    chunk_lengths: List[int] = []
    total_chunks = 0

    for player_id, player_events in filtered_df.groupby("player_id", sort=True):
        ordered = player_events.sort_values("event_index", kind="stable").reset_index(drop=True)
        label = str(ordered.iloc[0]["label"])
        features, labels = _build_player_sequence(
            ordered=ordered,
            layout=layout,
            manifest=manifest,
        )
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Feature/label length mismatch for player {player_id}: {features.shape} vs {labels.shape}"
            )
        raw_lengths.append(int(features.shape[0]))

        chunk_arrays = _chunk_sequence(
            features=features,
            labels=labels,
            max_seq_length=max_seq_length,
        )
        for chunk_index, (chunk_features, chunk_labels) in enumerate(chunk_arrays):
            sequence_id = (
                f"player_{int(player_id):05d}_{manifest.task_name}_chunk_{chunk_index:03d}"
            )
            examples.append(
                MinigridSequenceExample(
                    metadata=MinigridSequenceMetadata(
                        sequence_id=sequence_id,
                        player_id=int(player_id),
                        task_name=manifest.task_name,
                        label=label,
                        chunk_index=int(chunk_index),
                    ),
                    features=chunk_features.astype(np.float32),
                    targets=chunk_labels.astype(np.int64),
                    effective_length=int(chunk_features.shape[0]),
                )
            )
            chunk_lengths.append(int(chunk_features.shape[0]))
            total_chunks += 1

    stats = {
        "n_players": int(len(player_id_set)),
        "n_examples": int(len(examples)),
        "n_chunks": int(total_chunks),
        "mean_raw_sequence_length": float(np.mean(raw_lengths)) if raw_lengths else 0.0,
        "median_raw_sequence_length": float(median(raw_lengths)) if raw_lengths else 0.0,
        "max_raw_sequence_length": int(max(raw_lengths)) if raw_lengths else 0,
        "mean_chunk_length": float(np.mean(chunk_lengths)) if chunk_lengths else 0.0,
        "median_chunk_length": float(median(chunk_lengths)) if chunk_lengths else 0.0,
        "max_chunk_length": int(max(chunk_lengths)) if chunk_lengths else 0,
        "max_seq_length": int(max_seq_length),
    }
    return examples, stats


def _chunk_sequence(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    max_seq_length: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split one full player sequence into contiguous chunks when requested."""
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have the same sequence length.")
    if features.shape[0] == 0:
        return []

    if max_seq_length <= 0 or features.shape[0] <= max_seq_length:
        return [(features, labels)]

    chunks: List[Tuple[np.ndarray, np.ndarray]] = []
    for start in range(0, features.shape[0], max_seq_length):
        end = min(start + max_seq_length, features.shape[0])
        chunks.append((features[start:end], labels[start:end]))
    return chunks


def _build_player_sequence(
    *,
    ordered: pd.DataFrame,
    layout: TaskLayoutMetadata,
    manifest: MinigridFeatureManifest,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build one player-task sequence with a synthetic BOS state."""
    if ordered.empty:
        raise ValueError("Cannot build a sequence from an empty event frame.")

    completed_target_ids: Set[str] = set()
    feature_rows: List[np.ndarray] = []
    label_rows: List[int] = []

    first_action = str(ordered.iloc[0]["action"])
    feature_rows.append(
        _build_feature_vector(
            agent_x=int(layout.start_pos[0]),
            agent_y=int(layout.start_pos[1]),
            previous_x=int(layout.start_pos[0]),
            previous_y=int(layout.start_pos[1]),
            current_action=None,
            reward_received=0.0,
            cumulative_reward=0.0,
            remaining_targets=int(len(layout.targets)),
            num_completed_targets=0,
            completed_target_ids=completed_target_ids,
            layout=layout,
            manifest=manifest,
        )
    )
    label_rows.append(int(manifest.action_to_index[first_action]))

    for row_index in range(len(ordered) - 1):
        row = ordered.iloc[row_index]
        completed_after_event = set(completed_target_ids)
        completed_target_id = row.get("completed_target_id", None)
        if pd.notna(completed_target_id):
            completed_after_event.add(str(completed_target_id))

        expected_completed = int(row["num_completed_targets"])
        expected_remaining = int(row["remaining_targets"])
        if expected_completed != len(completed_after_event):
            raise AssertionError(
                "Completed-target count mismatch for player {} at event {}: row={} derived={}".format(
                    int(row["player_id"]),
                    int(row["event_index"]),
                    expected_completed,
                    len(completed_after_event),
                )
            )
        if expected_remaining != (len(layout.targets) - len(completed_after_event)):
            raise AssertionError(
                "Remaining-target count mismatch for player {} at event {}: row={} derived={}".format(
                    int(row["player_id"]),
                    int(row["event_index"]),
                    expected_remaining,
                    len(layout.targets) - len(completed_after_event),
                )
            )

        feature_rows.append(
            _build_feature_vector(
                agent_x=int(row["agent_x"]),
                agent_y=int(row["agent_y"]),
                previous_x=int(row["previous_x"]),
                previous_y=int(row["previous_y"]),
                current_action=str(row["action"]),
                reward_received=float(row["reward_received"]),
                cumulative_reward=float(row["cumulative_reward"]),
                remaining_targets=expected_remaining,
                num_completed_targets=expected_completed,
                completed_target_ids=completed_after_event,
                layout=layout,
                manifest=manifest,
            )
        )
        next_action = str(ordered.iloc[row_index + 1]["action"])
        label_rows.append(int(manifest.action_to_index[next_action]))
        completed_target_ids = completed_after_event

    return np.vstack(feature_rows).astype(np.float32), np.asarray(label_rows, dtype=np.int64)


def _build_feature_vector(
    *,
    agent_x: int,
    agent_y: int,
    previous_x: int,
    previous_y: int,
    current_action: Optional[str],
    reward_received: float,
    cumulative_reward: float,
    remaining_targets: int,
    num_completed_targets: int,
    completed_target_ids: Set[str],
    layout: TaskLayoutMetadata,
    manifest: MinigridFeatureManifest,
) -> np.ndarray:
    """Construct one per-timestep feature vector."""
    if not layout.is_passable(agent_x, agent_y):
        raise AssertionError(f"Agent occupies blocked cell {(agent_x, agent_y)}")
    if not layout.is_passable(previous_x, previous_y):
        raise AssertionError(f"Previous agent cell is blocked {(previous_x, previous_y)}")

    delta_x = int(agent_x - previous_x)
    delta_y = int(agent_y - previous_y)
    coord_scale = float(max(1, layout.grid_size - 1))
    reward_scale = float(max(manifest.max_reward, 1.0))
    salience_scale = float(max(getattr(manifest, "max_salience", 1.0), 1.0))
    total_reward_scale = float(max(manifest.total_possible_reward, 1.0))
    total_targets_scale = float(max(manifest.total_targets, 1))

    values: List[float] = [
        float(agent_x) / coord_scale,
        float(agent_y) / coord_scale,
        float(previous_x) / coord_scale,
        float(previous_y) / coord_scale,
        float(delta_x) / coord_scale,
        float(delta_y) / coord_scale,
        float(reward_received) / reward_scale,
        float(cumulative_reward) / total_reward_scale,
        float(remaining_targets) / total_targets_scale,
        float(num_completed_targets) / total_targets_scale,
    ]

    values.extend(_encode_current_action(current_action, manifest))
    values.extend(_encode_legal_moves(agent_x, agent_y, layout))
    values.extend(_encode_local_terrain(agent_x, agent_y, layout, manifest))

    for target_id in manifest.target_ids:
        target = layout.targets_by_id[target_id]
        rel_dx = float(target.x - agent_x) / coord_scale
        rel_dy = float(target.y - agent_y) / coord_scale
        distance = _lookup_distance(
            layout.distance_maps[target_id], agent_x, agent_y, manifest.max_distance
        )
        values.extend(
            [
                float(target_id not in completed_target_ids),
                rel_dx,
                rel_dy,
                float(distance) / float(max(manifest.max_distance, 1.0)),
                float(target.reward) / reward_scale,
                float(target.is_high_value),
            ]
        )
        if target_id in manifest.target_saliences:
            values.append(float(target.salience) / salience_scale)

    feature_vector = np.asarray(values, dtype=np.float32)
    if feature_vector.shape[0] != manifest.feature_dim:
        raise AssertionError(
            f"Feature dimension mismatch: expected {manifest.feature_dim}, got {feature_vector.shape[0]}"
        )
    return feature_vector


def _encode_current_action(
    current_action: Optional[str],
    manifest: MinigridFeatureManifest,
) -> List[float]:
    """Encode the current observed action as a one-hot vector."""
    vector = [0.0] * len(manifest.action_vocab)
    if current_action is None:
        return vector
    if current_action not in manifest.action_to_index:
        raise KeyError(f"Unexpected action '{current_action}' for task {manifest.task_name}")
    vector[manifest.action_to_index[current_action]] = 1.0
    return vector


def _encode_legal_moves(agent_x: int, agent_y: int, layout: TaskLayoutMetadata) -> List[float]:
    """Encode which cardinal moves are legal from the current cell."""
    return [
        float(layout.is_passable(agent_x, agent_y - 1)),
        float(layout.is_passable(agent_x, agent_y + 1)),
        float(layout.is_passable(agent_x - 1, agent_y)),
        float(layout.is_passable(agent_x + 1, agent_y)),
    ]


def _encode_local_terrain(
    agent_x: int,
    agent_y: int,
    layout: TaskLayoutMetadata,
    manifest: MinigridFeatureManifest,
) -> List[float]:
    """Encode a small egocentric terrain patch around the agent."""
    values: List[float] = []
    terrain_index = {int(code): idx for idx, code in enumerate(manifest.terrain_levels)}
    vector_size = len(manifest.terrain_levels)

    for dy in range(-manifest.local_patch_radius, manifest.local_patch_radius + 1):
        for dx in range(-manifest.local_patch_radius, manifest.local_patch_radius + 1):
            x = agent_x + dx
            y = agent_y + dy
            if 0 <= x < layout.grid_size and 0 <= y < layout.grid_size:
                terrain_code = int(layout.terrain[y, x])
            else:
                terrain_code = int(manifest.boundary_terrain_code)
            one_hot = [0.0] * vector_size
            one_hot[terrain_index[terrain_code]] = 1.0
            values.extend(one_hot)
    return values


def _lookup_distance(
    distance_map: np.ndarray,
    agent_x: int,
    agent_y: int,
    max_distance: float,
) -> float:
    """Return a finite shortest-path distance for one cell-to-target pair."""
    value = float(distance_map[agent_y, agent_x])
    if not np.isfinite(value):
        return float(max_distance)
    return float(min(value, max_distance))


def _build_distance_map(terrain: np.ndarray, goal: Tuple[int, int]) -> np.ndarray:
    """Run BFS from a goal cell to every passable cell."""
    height, width = terrain.shape
    gx, gy = goal
    if not (0 <= gx < width and 0 <= gy < height):
        raise ValueError(f"Goal {goal} lies outside the terrain grid.")
    if int(terrain[gy, gx]) != 0:
        raise ValueError(f"Goal {goal} lies on blocked terrain code={int(terrain[gy, gx])}.")

    distances = np.full((height, width), np.inf, dtype=np.float32)
    distances[gy, gx] = 0.0
    frontier: deque[Tuple[int, int]] = deque([goal])

    while frontier:
        x, y = frontier.popleft()
        current_distance = float(distances[y, x])
        for nx, ny in _neighbors(terrain, x, y):
            if np.isfinite(distances[ny, nx]):
                continue
            distances[ny, nx] = current_distance + 1.0
            frontier.append((nx, ny))

    return distances


def _neighbors(terrain: np.ndarray, x: int, y: int) -> Iterable[Tuple[int, int]]:
    """Yield passable cardinal neighbors for a terrain cell."""
    height, width = terrain.shape
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        nx = x + dx
        ny = y + dy
        if 0 <= nx < width and 0 <= ny < height and int(terrain[ny, nx]) == 0:
            yield (nx, ny)
