#!/usr/bin/env python3
"""Train a pure outcome-observation baseline for minigrid label classification."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from simulation.config import (
    LABEL_HOPPER,
    LABEL_PLANNER,
)
from simulation.train.data_utils import (
    DEFAULT_SIMULATION_DATA_DIR,
    load_players_table,
    load_task_events,
    load_task_layout,
)
from simulation.train.minigrid_plvm_data_utils import (
    DEFAULT_CUTOFF_SCHEDULE_STEPS,
    MODEL_TYPE_BOTH_PARTIAL,
    MODEL_TYPE_SINGLE_TASK,
    MinigridPLVMSampleCandidate,
    build_minigrid_cutoff_sample_payload,
)
from simulation.train.outcome_observation_baseline import (
    OUTCOME_OBSERVATION_BASELINE_FAMILY,
    OUTCOME_OBSERVATION_BASELINE_MODEL_TYPE,
    build_outcome_observation_dataset,
    fit_outcome_observation_baseline,
)
from simulation.train.train_plvm_label_minigrid import (
    _parse_cutoff_schedule_steps,
    _resolve_task_configuration,
    _save_classification_metrics_by_cutoff,
    _save_val_predictions,
)
from plvm import artifact_paths
from plvm.global_utils import set_seed
from plvm.model_training_utils import (
    compute_classification_metrics,
    generate_classification_visualizations,
    generate_model_name,
    update_hierarchical_model_performance_csv,
)


DEFAULT_OUTPUT_DIR = artifact_paths.SIMULATION_MODELS_DIR / "outcome_observation_baseline"
DEFAULT_CLASS_NAMES = [LABEL_PLANNER, LABEL_HOPPER]
DEFAULT_CUTOFF_SCHEDULE_STEPS_STR = ",".join(
    str(cutoff_step) for cutoff_step in DEFAULT_CUTOFF_SCHEDULE_STEPS
)
OUTCOME_BASELINE_EVENT_COLUMNS = [
    "player_id",
    "task_name",
    "event_index",
    "completed_target_id",
]


def make_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the minigrid outcome baseline."""
    parser = argparse.ArgumentParser(
        description="Train a pure outcome-observation baseline for minigrid player-label classification."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DEFAULT_SIMULATION_DATA_DIR),
        help="Simulation output directory containing players.csv, events.csv, and environment_metadata.json.",
    )
    parser.add_argument(
        "--player_split_manifest",
        type=str,
        default="",
        help="Optional player_splits.csv to reuse an existing train/val split.",
    )
    parser.add_argument(
        "--max_train_val_players",
        type=int,
        default=8000,
        help="Use a balanced Planner/Hopper cap of this many players total for train/validation.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="Validation fraction applied within the selected player subset.",
    )
    parser.add_argument(
        "--cutoff_schedule_steps",
        type=str,
        default=DEFAULT_CUTOFF_SCHEDULE_STEPS_STR,
        help="Comma-separated fixed step cutoffs used to build partial-observation samples.",
    )
    parser.add_argument(
        "--cutoff_samples_per_player",
        type=int,
        default=3,
        help="Number of cutoff samples to retain per player. Use <=0 to keep all valid cutoffs.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=MODEL_TYPE_BOTH_PARTIAL,
        choices=[MODEL_TYPE_BOTH_PARTIAL, MODEL_TYPE_SINGLE_TASK],
        help="plvm_both_partial: use both task outcome vectors. single_task: use one selected task outcome vector.",
    )
    parser.add_argument(
        "--full_task_name",
        type=str,
        default="",
        help="Second task branch used by plvm_both_partial.",
    )
    parser.add_argument(
        "--partial_task_name",
        type=str,
        default="",
        help="Primary task branch used by single_task and one branch of plvm_both_partial.",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=200,
        help="Number of trees for the random forest outcome baseline.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--run_mode",
        type=str,
        default="run",
        choices=["debug", "run"],
        help="Execution mode.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="default-model-0001",
        help="Model name. Use the default to auto-generate one.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Base output directory for trained minigrid outcome baselines.",
    )
    return parser


def _save_json(path: Path, payload: Dict[str, object]) -> None:
    """Save a JSON payload with stable formatting."""
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _to_pseudo_logits(probabilities: np.ndarray) -> torch.Tensor:
    """Convert probability rows into logits with equivalent softmax outputs."""
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-7, None)
    return torch.tensor(np.log(probs), dtype=torch.float32)


def _compute_metrics_from_probabilities(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    num_classes: int,
) -> Dict[str, object]:
    """Compute classification metrics and cross-entropy from probability rows."""
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-7, None)
    target_np = np.asarray(targets, dtype=np.int64).reshape(-1)
    pred_np = np.argmax(probs, axis=1).astype(np.int64)
    metrics = compute_classification_metrics(
        torch.tensor(target_np, dtype=torch.long),
        torch.tensor(pred_np, dtype=torch.long),
        num_classes=num_classes,
    )
    metrics["cross_entropy"] = float(
        -np.mean(np.log(probs[np.arange(target_np.shape[0]), target_np]))
    )
    return metrics


def _compute_majority_baseline_metrics(
    train_targets: np.ndarray,
    eval_targets: np.ndarray,
    *,
    num_classes: int,
) -> Dict[str, object]:
    """Evaluate a majority-class classifier derived from the training split."""
    train_np = np.asarray(train_targets, dtype=np.int64).reshape(-1)
    eval_np = np.asarray(eval_targets, dtype=np.int64).reshape(-1)
    if train_np.size == 0 or eval_np.size == 0:
        return {"majority_class": None, "metrics": {"accuracy": float("nan")}}
    majority_class = int(np.bincount(train_np, minlength=num_classes).argmax())
    preds = np.full(eval_np.shape[0], fill_value=majority_class, dtype=np.int64)
    return {
        "majority_class": majority_class,
        "metrics": compute_classification_metrics(
            torch.tensor(eval_np, dtype=torch.long),
            torch.tensor(preds, dtype=torch.long),
            num_classes=num_classes,
        ),
    }


def _compute_random_baseline_metrics(
    eval_targets: np.ndarray,
    *,
    num_classes: int,
    seed: int,
) -> Dict[str, object]:
    """Evaluate a reproducible uniform-random baseline."""
    eval_np = np.asarray(eval_targets, dtype=np.int64).reshape(-1)
    rng = np.random.RandomState(seed)
    preds = rng.randint(0, num_classes, size=eval_np.shape[0]).astype(np.int64)
    return {
        "seed": int(seed),
        "metrics": compute_classification_metrics(
            torch.tensor(eval_np, dtype=torch.long),
            torch.tensor(preds, dtype=torch.long),
            num_classes=num_classes,
        ),
    }


def _save_candidate_split_manifest(
    log_dir: Path,
    args: argparse.Namespace,
    train_candidates: Sequence[MinigridPLVMSampleCandidate],
    val_candidates: Sequence[MinigridPLVMSampleCandidate],
    sample_payload: Dict[str, object],
) -> None:
    """Persist the exact cutoff-conditioned samples used by this baseline run."""
    payload = {
        "model_name": args.model_name,
        "baseline_type": OUTCOME_OBSERVATION_BASELINE_MODEL_TYPE,
        "model_type": args.model_type,
        "tasks": list(args.tasks),
        "seed": int(args.seed),
        "val_fraction": float(args.val_fraction),
        "cutoff_schedule_steps": [int(value) for value in sample_payload["cutoff_schedule_steps"]],
        "cutoff_samples_per_player": int(sample_payload["cutoff_samples_per_player"]),
        "train_size": int(len(train_candidates)),
        "val_size": int(len(val_candidates)),
        "train": [candidate.to_dict() for candidate in train_candidates],
        "val": [candidate.to_dict() for candidate in val_candidates],
    }
    _save_json(Path(log_dir) / "candidate_splits.json", payload)


def _save_fit_artifacts(
    model_log_dir: Path,
    model_name: str,
    fit_result,
) -> None:
    """Save the fitted random forest and its serializable metadata."""
    artifact_path = model_log_dir / "{}_baseline_artifacts.pkl".format(model_name)
    with artifact_path.open("wb") as handle:
        pickle.dump(fit_result.model_artifacts, handle)

    baseline_dir = model_log_dir / "baseline_models"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    with (baseline_dir / "random_forest_model.pkl").open("wb") as handle:
        pickle.dump(fit_result.estimator, handle)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the minigrid outcome-observation baseline end to end."""
    parser = make_arg_parser()
    args = parser.parse_args(argv)

    if args.max_train_val_players <= 0:
        raise ValueError("max_train_val_players must be positive.")
    if args.cutoff_samples_per_player == 0:
        raise ValueError("cutoff_samples_per_player cannot be 0.")
    if args.n_estimators <= 0:
        raise ValueError("n_estimators must be positive.")

    args.full_task_name = str(args.full_task_name or "").strip()
    args.partial_task_name = str(args.partial_task_name or "").strip()
    args.cutoff_schedule_steps = _parse_cutoff_schedule_steps(args.cutoff_schedule_steps)
    selected_tasks = _resolve_task_configuration(args)

    set_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.model_name == "default-model-0001":
        args.model_name = generate_model_name()
    if args.run_mode == "debug":
        args.max_train_val_players = min(args.max_train_val_players, 64)
        if not args.model_name.startswith("debug-"):
            args.model_name = "debug-{}".format(args.model_name)

    output_root = Path(args.output_dir).expanduser().resolve()
    log_root = output_root / ("debug_logs" if args.run_mode == "debug" else "training_logs")
    model_log_dir = log_root / args.model_name
    model_log_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir = str(model_log_dir)

    logging.basicConfig(
        filename=str(model_log_dir / "training.logs"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    # Fixed run metadata read by the shared model_performance.csv registry and the
    # saved hyperparameters; most of these fields have no effect on the baseline.
    args.task_type = "classification"
    args.target_mode = "player_label"
    args.target_name = "player_label"
    args.target_type = "classification"
    args.loss_type = "classification"
    args.num_classes = len(DEFAULT_CLASS_NAMES)
    args.class_names = list(DEFAULT_CLASS_NAMES)
    args.tasks = list(selected_tasks)
    args.predicted_task = (
        ",".join(sorted(selected_tasks))
        if args.model_type == MODEL_TYPE_BOTH_PARTIAL
        else str(args.partial_task_name)
    )
    args.predicted_next_task = args.predicted_task
    args.parameters_to_predict = "minigrid_player_label"
    args.prediction_head_type = "rf"
    args.input_embeddings_source = "minigrid_outcome_observation"
    args.feature_mode = "pure_outcome_observation"
    args.model_family = OUTCOME_OBSERVATION_BASELINE_FAMILY
    args.data_file = str(Path(args.data_dir).expanduser().resolve())
    args.seq_length = 0
    args.max_norm = 0.0
    args.use_early_stopping = False

    logging.info("Minigrid outcome-observation baseline configuration:")
    for key, value in sorted(vars(args).items()):
        logging.info("%s: %s", key, value)

    data_dir = Path(args.data_dir).expanduser().resolve()
    players_df = load_players_table(data_dir)

    task_event_groups: Dict[str, Dict[int, pd.DataFrame]] = {}
    task_layouts = {}
    available_player_ids: Optional[set] = None
    max_observed_steps_by_player: Dict[int, int] = {}
    for task_name in selected_tasks:
        task_layouts[task_name] = load_task_layout(data_dir, task_name)
        task_events = load_task_events(
            data_dir,
            task_name,
            usecols=OUTCOME_BASELINE_EVENT_COLUMNS,
        )
        grouped = {
            int(player_id): group.reset_index(drop=True)
            for player_id, group in task_events.groupby(
                task_events["player_id"].astype(int), sort=True
            )
        }
        task_event_groups[task_name] = grouped
        task_player_ids = set(grouped.keys())
        available_player_ids = (
            task_player_ids
            if available_player_ids is None
            else available_player_ids & task_player_ids
        )

    available_player_ids = available_player_ids or set()
    if not available_player_ids:
        raise ValueError(
            "No overlapping player-task sequences are available for the selected tasks."
        )

    for player_id in sorted(available_player_ids):
        max_observed_steps_by_player[int(player_id)] = min(
            int(len(task_event_groups[task_name][int(player_id)])) for task_name in selected_tasks
        )

    sample_payload = build_minigrid_cutoff_sample_payload(
        players_df,
        available_player_ids=sorted(available_player_ids),
        max_observed_steps_by_player=max_observed_steps_by_player,
        cutoff_schedule_steps=args.cutoff_schedule_steps,
        cutoff_samples_per_player=args.cutoff_samples_per_player,
        max_train_val_players=args.max_train_val_players,
        val_fraction=args.val_fraction,
        seed=args.seed,
        player_split_manifest=(
            Path(args.player_split_manifest).expanduser().resolve()
            if args.player_split_manifest
            else None
        ),
    )

    train_player_id_set = set(sample_payload["split_payload"]["train_player_ids"])
    val_player_id_set = set(sample_payload["split_payload"]["val_player_ids"])
    train_candidates = [
        candidate
        for candidate in sample_payload["samples"]
        if int(candidate.player_id) in train_player_id_set
    ]
    val_candidates = [
        candidate
        for candidate in sample_payload["samples"]
        if int(candidate.player_id) in val_player_id_set
    ]
    if not train_candidates:
        raise ValueError("No train samples remained after cutoff construction.")
    if not val_candidates:
        raise ValueError("No validation samples remained after cutoff construction.")

    dataset = build_outcome_observation_dataset(
        task_names=selected_tasks,
        train_candidates=train_candidates,
        val_candidates=val_candidates,
        task_events_by_player=task_event_groups,
        task_layouts=task_layouts,
    )
    fit_result = fit_outcome_observation_baseline(
        dataset,
        seed=args.seed,
        n_estimators=args.n_estimators,
    )

    train_metrics = _compute_metrics_from_probabilities(
        fit_result.train_probabilities,
        dataset.train_targets,
        num_classes=args.num_classes,
    )
    val_metrics = _compute_metrics_from_probabilities(
        fit_result.val_probabilities,
        dataset.val_targets,
        num_classes=args.num_classes,
    )
    majority_train = _compute_majority_baseline_metrics(
        dataset.train_targets,
        dataset.train_targets,
        num_classes=args.num_classes,
    )
    majority_val = _compute_majority_baseline_metrics(
        dataset.train_targets,
        dataset.val_targets,
        num_classes=args.num_classes,
    )
    random_train = _compute_random_baseline_metrics(
        dataset.train_targets,
        num_classes=args.num_classes,
        seed=args.seed,
    )
    random_val = _compute_random_baseline_metrics(
        dataset.val_targets,
        num_classes=args.num_classes,
        seed=args.seed + 1,
    )

    pseudo_train_logits = _to_pseudo_logits(fit_result.train_probabilities)
    pseudo_val_logits = _to_pseudo_logits(fit_result.val_probabilities)
    train_target_tensor = torch.tensor(dataset.train_targets, dtype=torch.long).unsqueeze(-1)
    val_target_tensor = torch.tensor(dataset.val_targets, dtype=torch.long).unsqueeze(-1)

    sample_payload["split_payload"]["split_df"].to_csv(
        model_log_dir / "player_splits.csv", index=False
    )
    _save_json(model_log_dir / "split_summary.json", sample_payload["split_payload"]["summary"])
    _save_json(
        model_log_dir / "outcome_dataset_summary.json",
        {
            "task_names": list(selected_tasks),
            "feature_names": list(dataset.feature_names),
            "feature_metadata": dict(dataset.metadata),
            "cutoff_schedule_steps": sample_payload["cutoff_schedule_steps"],
            "cutoff_samples_per_player": sample_payload["cutoff_samples_per_player"],
            "cutoff_histogram": sample_payload["cutoff_histogram"],
            "n_total_samples": int(len(sample_payload["samples"])),
            "n_train_samples": int(len(train_candidates)),
            "n_val_samples": int(len(val_candidates)),
            "n_train_unique_players": int(
                len(set(candidate.player_id for candidate in train_candidates))
            ),
            "n_val_unique_players": int(
                len(set(candidate.player_id for candidate in val_candidates))
            ),
        },
    )
    _save_candidate_split_manifest(
        model_log_dir,
        args,
        train_candidates,
        val_candidates,
        sample_payload,
    )
    _save_fit_artifacts(model_log_dir, args.model_name, fit_result)

    generate_classification_visualizations(
        true_labels=np.asarray(dataset.train_targets, dtype=np.int64),
        predicted_probs=np.asarray(fit_result.train_probabilities, dtype=np.float64),
        output_dir=str(model_log_dir / "classification" / "train"),
        class_names=args.class_names,
        title_prefix="Train Outcome Baseline",
    )
    generate_classification_visualizations(
        true_labels=np.asarray(dataset.val_targets, dtype=np.int64),
        predicted_probs=np.asarray(fit_result.val_probabilities, dtype=np.float64),
        output_dir=str(model_log_dir / "classification" / "val"),
        class_names=args.class_names,
        title_prefix="Val Outcome Baseline",
    )

    _save_val_predictions(
        model_log_dir,
        args.model_name,
        pseudo_val_logits,
        val_target_tensor,
        val_candidates,
        args.class_names,
    )
    _save_classification_metrics_by_cutoff(
        model_log_dir,
        args.model_name,
        "train",
        pseudo_train_logits,
        train_target_tensor,
        train_candidates,
        args.class_names,
    )
    _save_classification_metrics_by_cutoff(
        model_log_dir,
        args.model_name,
        "val",
        pseudo_val_logits,
        val_target_tensor,
        val_candidates,
        args.class_names,
    )

    baseline_metrics = {
        "majority_train": majority_train,
        "majority_val": majority_val,
        "random_train": random_train,
        "random_val": random_val,
    }

    update_hierarchical_model_performance_csv(
        csv_file=str(model_log_dir.parent / "model_performance.csv"),
        args=args,
        train_loss=float(train_metrics["cross_entropy"]),
        val_loss=float(val_metrics["cross_entropy"]),
        train_size=len(train_candidates),
        val_size=len(val_candidates),
        naive_train_accuracy=float(majority_train["metrics"]["accuracy"]),
        naive_val_accuracy=float(majority_val["metrics"]["accuracy"]),
        train_accuracy=float(train_metrics["accuracy"]),
        val_accuracy=float(val_metrics["accuracy"]),
        heldout_session_ids="",
    )

    hparams = {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, (str, int, float, bool, list))
    }
    hparams["train_metrics"] = train_metrics
    hparams["val_metrics"] = val_metrics
    hparams["baseline_metrics"] = baseline_metrics
    hparams["feature_names"] = list(dataset.feature_names)
    hparams["feature_metadata"] = dict(dataset.metadata)
    hparams["estimator_artifacts"] = fit_result.model_artifacts
    hparams["n_train_samples"] = int(len(train_candidates))
    hparams["n_val_samples"] = int(len(val_candidates))

    hparams_path = model_log_dir / "{}_hyperparameters.json".format(args.model_name)
    with hparams_path.open("w", encoding="utf-8") as handle:
        json.dump(hparams, handle, indent=2, sort_keys=True)

    _save_json(model_log_dir / "train_metrics.json", train_metrics)
    _save_json(model_log_dir / "val_metrics.json", val_metrics)
    _save_json(model_log_dir / "baseline_metrics.json", baseline_metrics)
    _save_json(
        model_log_dir / "training_summary.json",
        {
            "model_name": args.model_name,
            "baseline_type": OUTCOME_OBSERVATION_BASELINE_MODEL_TYPE,
            "model_family": OUTCOME_OBSERVATION_BASELINE_FAMILY,
            "model_type": args.model_type,
            "tasks": list(selected_tasks),
            "feature_names": list(dataset.feature_names),
            "split_summary": sample_payload["split_payload"]["summary"],
            "cutoff_schedule_steps": sample_payload["cutoff_schedule_steps"],
            "cutoff_samples_per_player": sample_payload["cutoff_samples_per_player"],
            "cutoff_histogram": sample_payload["cutoff_histogram"],
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "baseline_metrics": baseline_metrics,
            "output_dir": str(output_root),
            "log_dir": str(model_log_dir),
        },
    )

    logging.info(
        "OutcomeObservationBaseline tasks=%s train_acc=%.4f val_acc=%.4f",
        ",".join(selected_tasks),
        float(train_metrics["accuracy"]),
        float(val_metrics["accuracy"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
