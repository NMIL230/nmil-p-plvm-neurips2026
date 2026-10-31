#!/usr/bin/env python3
"""Pure outcome-observation baseline for minigrid player-label classification.

It ignores process-level path structure and uses only cumulative completion
summaries (completed targets and task-completion percent per task) observed up to
a fixed step cutoff. The same kind of outcome baseline is used for PowerWash in
powerwash/train/.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from simulation.config import (
    TASK_NAME_CLEANING,
    TASK_NAME_FORAGING,
)
from simulation.train.data_utils import (
    TaskLayoutMetadata,
)
from simulation.train.minigrid_plvm_data_utils import (
    MinigridPLVMSampleCandidate,
)


OUTCOME_OBSERVATION_BASELINE_MODEL_TYPE = "outcome_observation_baseline"
OUTCOME_OBSERVATION_BASELINE_FAMILY = "OutcomeObservationBaseline"


@dataclass
class OutcomeObservationDataset:
    """Outcome-only train/validation matrices for the minigrid baseline."""

    feature_names: List[str]
    train_features: np.ndarray
    val_features: np.ndarray
    train_targets: np.ndarray
    val_targets: np.ndarray
    metadata: Dict[str, object]


@dataclass
class OutcomeObservationFitResult:
    """Fitted classifier probabilities plus serializable estimator metadata."""

    train_probabilities: np.ndarray
    val_probabilities: np.ndarray
    estimator: object
    model_artifacts: Dict[str, object]


def build_outcome_observation_dataset(
    *,
    task_names: Sequence[str],
    train_candidates: Sequence[MinigridPLVMSampleCandidate],
    val_candidates: Sequence[MinigridPLVMSampleCandidate],
    task_events_by_player: Dict[str, Dict[int, pd.DataFrame]],
    task_layouts: Dict[str, TaskLayoutMetadata],
) -> OutcomeObservationDataset:
    """Build fixed outcome-only features for train and validation candidates."""
    if not train_candidates:
        raise ValueError("OutcomeObservationBaseline requires at least one training candidate.")
    if not val_candidates:
        raise ValueError("OutcomeObservationBaseline requires at least one validation candidate.")

    selected_task_names = [
        str(task_name).strip().lower() for task_name in task_names if str(task_name).strip()
    ]
    if not selected_task_names:
        raise ValueError("task_names must contain at least one minigrid task.")

    task_feature_names: Dict[str, List[str]] = {}
    per_task_feature_dims: Dict[str, int] = {}

    train_player_ids = sorted({int(candidate.player_id) for candidate in train_candidates})
    val_player_ids = sorted({int(candidate.player_id) for candidate in val_candidates})

    for task_name in selected_task_names:
        if task_name not in task_events_by_player:
            raise KeyError("Missing preloaded event groups for task '{}'.".format(task_name))
        if task_name not in task_layouts:
            raise KeyError("Missing task layout metadata for task '{}'.".format(task_name))
        task_feature_names[task_name] = _task_feature_names(task_name)
        per_task_feature_dims[task_name] = len(task_feature_names[task_name])

    feature_names: List[str] = []
    for task_name in selected_task_names:
        feature_names.extend(task_feature_names[task_name])

    train_features = _build_feature_matrix(
        train_candidates,
        selected_task_names,
        task_events_by_player,
        task_layouts,
    )
    val_features = _build_feature_matrix(
        val_candidates,
        selected_task_names,
        task_events_by_player,
        task_layouts,
    )
    train_targets = np.asarray(
        [int(candidate.class_id) for candidate in train_candidates],
        dtype=np.int64,
    )
    val_targets = np.asarray(
        [int(candidate.class_id) for candidate in val_candidates],
        dtype=np.int64,
    )

    return OutcomeObservationDataset(
        feature_names=feature_names,
        train_features=train_features,
        val_features=val_features,
        train_targets=train_targets,
        val_targets=val_targets,
        metadata={
            "task_names": list(selected_task_names),
            "task_feature_dims": per_task_feature_dims,
            "n_train_examples": int(len(train_candidates)),
            "n_val_examples": int(len(val_candidates)),
            "n_train_players": int(len(train_player_ids)),
            "n_val_players": int(len(val_player_ids)),
            "cutoff_schedule_steps_train": sorted(
                {int(candidate.cutoff_steps) for candidate in train_candidates}
            ),
            "cutoff_schedule_steps_val": sorted(
                {int(candidate.cutoff_steps) for candidate in val_candidates}
            ),
        },
    )


def fit_outcome_observation_baseline(
    dataset: OutcomeObservationDataset,
    *,
    seed: int = 42,
    n_estimators: int = 200,
) -> OutcomeObservationFitResult:
    """Fit a random-forest classifier on the outcome features."""
    if dataset.train_features.shape[0] == 0:
        raise ValueError("No training rows were provided.")
    unique_classes = np.unique(dataset.train_targets)
    if unique_classes.size < 2:
        raise ValueError(
            "OutcomeObservationBaseline requires at least two training classes, got {}.".format(
                unique_classes.tolist()
            )
        )

    from sklearn.ensemble import RandomForestClassifier

    estimator = RandomForestClassifier(
        n_estimators=int(n_estimators),
        random_state=int(seed),
        n_jobs=-1,
    )
    estimator.fit(dataset.train_features, dataset.train_targets)
    train_probabilities = estimator.predict_proba(dataset.train_features).astype(np.float64)
    val_probabilities = estimator.predict_proba(dataset.val_features).astype(np.float64)

    return OutcomeObservationFitResult(
        train_probabilities=train_probabilities,
        val_probabilities=val_probabilities,
        estimator=estimator,
        model_artifacts={
            "variant": "random_forest",
            "feature_names": list(dataset.feature_names),
            "estimator": {
                "class_name": estimator.__class__.__name__,
                "classes": np.asarray(estimator.classes_, dtype=np.int64).tolist(),
                "n_estimators": int(estimator.n_estimators),
                "feature_importances": np.asarray(
                    estimator.feature_importances_,
                    dtype=np.float64,
                ).tolist(),
            },
            "metadata": dict(dataset.metadata),
        },
    )


def _build_feature_matrix(
    candidates: Sequence[MinigridPLVMSampleCandidate],
    task_names: Sequence[str],
    task_events_by_player: Dict[str, Dict[int, pd.DataFrame]],
    task_layouts: Dict[str, TaskLayoutMetadata],
) -> np.ndarray:
    """Concatenate per-task outcome observations into one dense feature matrix."""
    feature_rows: List[np.ndarray] = []
    for candidate in candidates:
        task_features: List[np.ndarray] = []
        for task_name in task_names:
            player_events = task_events_by_player[task_name].get(int(candidate.player_id))
            if player_events is None:
                raise KeyError(
                    "Missing events for player {} task '{}'.".format(
                        int(candidate.player_id),
                        task_name,
                    )
                )
            prefix_df = player_events.iloc[: int(candidate.cutoff_steps)].copy()
            task_features.append(
                _extract_task_outcome_features(
                    task_name=task_name,
                    prefix_df=prefix_df,
                    task_layout=task_layouts[task_name],
                )
            )
        feature_rows.append(np.concatenate(task_features, axis=0))
    return np.vstack(feature_rows).astype(np.float64)


def _task_feature_names(task_name: str) -> List[str]:
    """Return the deterministic feature names for one task branch."""
    prefix = str(task_name).strip().lower()
    if prefix in {TASK_NAME_FORAGING, TASK_NAME_CLEANING}:
        return [
            f"{prefix}__completed_count",
            f"{prefix}__task_completion_percent",
        ]
    raise ValueError("Unsupported task_name for outcome observations: {}".format(task_name))


def _extract_task_outcome_features(
    *,
    task_name: str,
    prefix_df: pd.DataFrame,
    task_layout: TaskLayoutMetadata,
) -> np.ndarray:
    """Extract pure completion-only outcome features from one task prefix."""
    completed_df = prefix_df.loc[prefix_df["completed_target_id"].notna()].copy()
    num_completed = int(len(completed_df))
    total_targets = max(1, int(len(task_layout.targets)))
    task_completion_percent = 100.0 * float(num_completed) / float(total_targets)

    if task_name in {TASK_NAME_FORAGING, TASK_NAME_CLEANING}:
        return np.asarray(
            [
                float(num_completed),
                task_completion_percent,
            ],
            dtype=np.float64,
        )

    raise ValueError("Unsupported task_name for outcome observations: {}".format(task_name))
