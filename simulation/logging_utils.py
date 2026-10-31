"""Event schema and summary helpers for the paired-task simulator."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from simulation.agents import PlayerProfile
from simulation.config import TASK_NAME_FORAGING
from simulation.environments import TaskEnvironment


PLAYER_COLUMNS = [
    "player_id",
    "latent_P",
    "latent_O",
    "label",
    "class_id",
]

TARGET_COLUMNS = [
    "task_name",
    "target_id",
    "target_x",
    "target_y",
    "target_patch",
    "target_type",
    "target_reward",
    "target_salience",
]

EVENT_COLUMNS = [
    "player_id",
    "task_name",
    "label",
    "class_id",
    "latent_P",
    "latent_O",
    "event_index",
    "decision_index",
    "time_step",
    "event_type",
    "action",
    "agent_x",
    "agent_y",
    "previous_x",
    "previous_y",
    "current_patch",
    "previous_patch",
    "selected_target_id",
    "selected_target_x",
    "selected_target_y",
    "selected_target_patch",
    "selected_target_type",
    "selected_target_reward",
    "selected_target_salience",
    "completed_target_id",
    "completed_target_x",
    "completed_target_y",
    "completed_target_patch",
    "completed_target_type",
    "completed_target_reward",
    "completed_target_salience",
    "completion_rank",
    "shortest_path_distance_to_selected",
    "step_cost",
    "reward_received",
    "cumulative_reward",
    "cumulative_cost",
    "cumulative_net_reward",
    "remaining_targets",
    "num_completed_targets",
    "total_targets",
    "patch_switch_event",
    "patch_revisit_event",
]

SUMMARY_COLUMNS = [
    "player_id",
    "task_name",
    "label",
    "latent_P",
    "latent_O",
    "total_steps",
    "total_targets",
    "total_reward",
    "total_cost",
    "total_net_reward",
    "patch_switch_count",
    "patch_revisit_count",
    "mean_patch_run_length",
    "median_patch_run_length",
    "early_high_value_rate",
    "early_patch_persistence",
    "first_10pct_patch_switch_rate",
    "first_20pct_patch_switch_rate",
    "first_50pct_patch_switch_rate",
]


def players_to_frame(players: Sequence[PlayerProfile]) -> pd.DataFrame:
    """Convert player profiles into a stable tabular representation."""
    return pd.DataFrame(
        [
            {
                "player_id": player.player_id,
                "latent_P": player.latent_P,
                "latent_O": player.latent_O,
                "label": player.label,
                "class_id": player.class_id,
            }
            for player in players
        ],
        columns=PLAYER_COLUMNS,
    ).sort_values("player_id", kind="stable")


def targets_to_frame(environments: Sequence[TaskEnvironment]) -> pd.DataFrame:
    """Convert environment targets into one shared metadata table."""
    rows: List[Dict[str, object]] = []
    for environment in environments:
        for target in environment.targets:
            rows.append(
                {
                    "task_name": environment.task_name,
                    "target_id": target.target_id,
                    "target_x": target.x,
                    "target_y": target.y,
                    "target_patch": target.patch_id,
                    "target_type": target.target_type,
                    "target_reward": target.reward,
                    "target_salience": target.salience,
                }
            )
    return pd.DataFrame(rows, columns=TARGET_COLUMNS)


def build_compact_sequence_record(
    player: PlayerProfile,
    task_name: str,
    event_payload: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    """Build the transformer-ready JSONL record for one player-task."""
    return {
        "player_id": player.player_id,
        "task_name": task_name,
        "label": player.label,
        "latent_P": player.latent_P,
        "latent_O": player.latent_O,
        "events": list(event_payload),
    }


def compute_task_summary(
    player: PlayerProfile,
    environment: TaskEnvironment,
    completion_records: Sequence[Dict[str, object]],
    total_steps: int,
    total_reward: float,
    total_cost: float,
    patch_switch_count: int,
    patch_revisit_count: int,
    early_fraction: float,
) -> Dict[str, object]:
    """Aggregate one player-task trajectory into summary features."""
    patch_sequence = [int(record["patch_id"]) for record in completion_records]
    run_lengths = compute_run_lengths(patch_sequence)

    summary = {
        "player_id": player.player_id,
        "task_name": environment.task_name,
        "label": player.label,
        "latent_P": player.latent_P,
        "latent_O": player.latent_O,
        "total_steps": int(total_steps),
        "total_targets": int(len(completion_records)),
        "total_reward": float(total_reward),
        "total_cost": float(total_cost),
        "total_net_reward": float(total_reward - total_cost),
        "patch_switch_count": int(patch_switch_count),
        "patch_revisit_count": int(patch_revisit_count),
        "mean_patch_run_length": float(np.mean(run_lengths)) if run_lengths else 0.0,
        "median_patch_run_length": float(median(run_lengths)) if run_lengths else 0.0,
        "early_high_value_rate": compute_early_high_value_rate(
            environment.task_name,
            completion_records,
            early_fraction=early_fraction,
        ),
        "early_patch_persistence": float(
            1.0 - compute_prefix_patch_switch_rate(patch_sequence, early_fraction)
        ),
        "first_10pct_patch_switch_rate": float(
            compute_prefix_patch_switch_rate(patch_sequence, 0.10)
        ),
        "first_20pct_patch_switch_rate": float(
            compute_prefix_patch_switch_rate(patch_sequence, 0.20)
        ),
        "first_50pct_patch_switch_rate": float(
            compute_prefix_patch_switch_rate(patch_sequence, 0.50)
        ),
    }
    return summary


def compute_run_lengths(patch_sequence: Sequence[int]) -> List[int]:
    """Return consecutive same-patch run lengths from a completion sequence."""
    if not patch_sequence:
        return []

    run_lengths: List[int] = []
    current_run = 1
    for previous_patch, current_patch in zip(patch_sequence[:-1], patch_sequence[1:]):
        if current_patch == previous_patch:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    run_lengths.append(current_run)
    return run_lengths


def compute_prefix_patch_switch_rate(patch_sequence: Sequence[int], fraction: float) -> float:
    """Compute the patch-switch rate for the first fraction of completions."""
    if len(patch_sequence) <= 1:
        return 0.0
    prefix_length = max(2, int(np.ceil(len(patch_sequence) * float(fraction))))
    prefix = list(patch_sequence[:prefix_length])
    transitions = len(prefix) - 1
    if transitions <= 0:
        return 0.0
    switches = sum(int(current != previous) for previous, current in zip(prefix[:-1], prefix[1:]))
    return float(switches / transitions)


def compute_early_high_value_rate(
    task_name: str,
    completion_records: Sequence[Dict[str, object]],
    early_fraction: float,
) -> float:
    """Return the fraction of high-value foraging targets collected early."""
    if task_name != TASK_NAME_FORAGING:
        return float("nan")

    high_value_ranks = [
        int(record["completion_rank"])
        for record in completion_records
        if bool(record["is_high_value"])
    ]
    if not high_value_ranks:
        return 0.0

    early_cutoff = max(1, int(np.ceil(len(completion_records) * float(early_fraction))))
    early_high_value = sum(int(rank <= early_cutoff) for rank in high_value_ranks)
    return float(early_high_value / len(high_value_ranks))


def build_sanity_summary(
    players_df: pd.DataFrame,
    summaries_df: pd.DataFrame,
    failed_paths: int,
) -> Dict[str, object]:
    """Create a compact sanity summary for printing and saving."""
    grouped_steps = (
        summaries_df.groupby(["label", "task_name"], sort=True)["total_steps"]
        .mean()
        .round(3)
        .to_dict()
    )
    grouped_switches = (
        summaries_df.groupby(["label", "task_name"], sort=True)["patch_switch_count"]
        .mean()
        .round(3)
        .to_dict()
    )
    grouped_run_lengths = (
        summaries_df.groupby(["label", "task_name"], sort=True)["mean_patch_run_length"]
        .mean()
        .round(3)
        .to_dict()
    )

    foraging_df = summaries_df[summaries_df["task_name"] == TASK_NAME_FORAGING]
    high_value_rate = (
        foraging_df.groupby("label", sort=True)["early_high_value_rate"].mean().round(3).to_dict()
    )

    return {
        "n_players": int(len(players_df)),
        "label_counts": {
            str(label): int(count)
            for label, count in players_df["label"].value_counts(sort=False).to_dict().items()
        },
        "mean_total_steps_by_label_and_task": {
            f"{label}|{task_name}": float(value)
            for (label, task_name), value in grouped_steps.items()
        },
        "mean_patch_switches_by_label_and_task": {
            f"{label}|{task_name}": float(value)
            for (label, task_name), value in grouped_switches.items()
        },
        "mean_patch_run_length_by_label_and_task": {
            f"{label}|{task_name}": float(value)
            for (label, task_name), value in grouped_run_lengths.items()
        },
        "foraging_early_high_value_rate_by_label": {
            str(label): float(value) for label, value in high_value_rate.items()
        },
        "average_trajectory_length": float(
            summaries_df["total_steps"].mean() + summaries_df["total_targets"].mean()
        ),
        "failed_paths": int(failed_paths),
    }


def save_json(path: Path, payload: Dict[str, object]) -> None:
    """Save a JSON payload with stable formatting."""
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_jsonl_record(handle, payload: Dict[str, object]) -> None:
    """Write one JSON line to an already-open file handle."""
    handle.write(json.dumps(payload))
    handle.write("\n")


def environments_to_metadata(environments: Sequence[TaskEnvironment]) -> Dict[str, object]:
    """Serialize shared task layouts for downstream model training."""
    payload: Dict[str, object] = {}
    for environment in environments:
        payload[environment.task_name] = {
            "task_name": environment.task_name,
            "grid_size": int(environment.grid_size),
            "start_pos": [int(environment.start_pos[0]), int(environment.start_pos[1])],
            "terrain": environment.terrain.astype(int).tolist(),
            "obstacle_labels": {
                str(key): str(value) for key, value in environment.obstacle_labels.items()
            },
            "targets": [
                {
                    "target_id": target.target_id,
                    "x": int(target.x),
                    "y": int(target.y),
                    "patch_id": int(target.patch_id),
                    "target_type": str(target.target_type),
                    "reward": float(target.reward),
                    "salience": float(target.salience),
                    "is_high_value": bool(target.is_high_value),
                }
                for target in environment.targets
            ],
        }
    return payload
