"""Train a PLVM player-label classifier on frozen minigrid task embeddings.

Each sample pairs one frozen task-specific encoder embedding per task for the same
player, observed up to a fixed step cutoff, and predicts the player label
(Planner vs Hopper). The prediction head, embedding standardization and training
loop are shared with the PowerWash PLVM trainer (powerwash/train/train_plvm_powerwash.py).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from powerwash.train.train_plvm_powerwash import (
    _fit_and_apply_dim_reduction,
    _fit_and_apply_embedding_standardization,
    _log_probability_tensor,
    _parse_dim_reduction_method,
    _predict_logistic_probabilities,
    _save_dim_reduction_artifacts,
    _save_embedding_standardization_scaler,
)
from simulation.config import (
    LABEL_HOPPER,
    LABEL_PLANNER,
)
from simulation.train.data_utils import (
    DEFAULT_SIMULATION_DATA_DIR,
)
from simulation.train.minigrid_plvm_data_utils import (
    DEFAULT_CUTOFF_SCHEDULE_STEPS,
    DEFAULT_TASK_MODEL_ROOT,
    MODEL_TYPE_BOTH_PARTIAL,
    MODEL_TYPE_SINGLE_TASK,
    PLVM_TASK_NAMES,
    SUPPORTED_EMBEDDING_POOLERS,
    MinigridPLVMSampleCandidate,
    build_paired_player_dataset,
    load_minigrid_task_model_spec,
    load_task_models_and_manifests,
    materialize_minigrid_plvm_embeddings,
)
from plvm import artifact_paths
from plvm.global_utils import set_seed
from plvm.model_training_utils import (
    COMPUTE_DEVICE,
    EmbeddingHierarchicalModel,
    EmbeddingMLPModel,
    compute_classification_metrics,
    compute_naive_baseline_losses,
    generate_classification_visualizations,
    generate_model_name,
    plot_loss_curves,
    save_session_specific_plots,
    train_embedding_hierarchical_model,
    update_hierarchical_model_performance_csv,
)


DEFAULT_OUTPUT_DIR = artifact_paths.SIMULATION_MODELS_DIR / "plvm_label"
DEFAULT_CACHE_DIR = DEFAULT_OUTPUT_DIR / "embedding_cache"
DEFAULT_CLASS_NAMES = [LABEL_PLANNER, LABEL_HOPPER]
DEFAULT_MODEL_TYPE = MODEL_TYPE_BOTH_PARTIAL
DEFAULT_CUTOFF_SCHEDULE_STEPS_STR = ",".join(
    str(cutoff_step) for cutoff_step in DEFAULT_CUTOFF_SCHEDULE_STEPS
)


def _resolved_output_root(output_root: Path, args: argparse.Namespace) -> Path:
    """Route probe-style runs to sibling family folders next to ``output_root``.

    Logistic heads go to ``<root>_concat_logistic_probe`` or
    ``<root>_single_task_logistic_probe``, and the two-task MLP head to
    ``<root>_concat_mlp_probe``; the hierarchical and single-task MLP runs stay in
    ``output_root``.
    """
    root = Path(output_root).expanduser().resolve()
    prediction_head_type = str(getattr(args, "prediction_head_type", "")).strip().lower()
    model_type = str(getattr(args, "model_type", "")).strip().lower()
    if prediction_head_type == "logistic":
        family_suffix = (
            "_single_task_logistic_probe"
            if model_type == MODEL_TYPE_SINGLE_TASK
            else "_concat_logistic_probe"
        )
    elif prediction_head_type == "mlp" and model_type == MODEL_TYPE_BOTH_PARTIAL:
        family_suffix = "_concat_mlp_probe"
    else:
        return root
    if root.name.endswith(family_suffix):
        return root
    return root.parent / "{}{}".format(root.name, family_suffix)


def make_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for minigrid PLVM label training."""
    parser = argparse.ArgumentParser(
        description="Train a paired-task PLVM player-label classifier on frozen minigrid task embeddings."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DEFAULT_SIMULATION_DATA_DIR),
        help="Simulation output directory containing players.csv, events.csv, and environment_metadata.json.",
    )
    parser.add_argument(
        "--task_model_spec",
        type=str,
        default=str(artifact_paths.SIMULATION_TASK_SPECIFIC_SPEC_JSON),
        help=(
            "JSON spec naming the frozen task-specific encoder per task (default: the "
            "encoders included in the artifact bundle). Pass an empty string to auto-select "
            "the completed foraging/cleaning models with the best validation accuracy under {}."
        ).format(DEFAULT_TASK_MODEL_ROOT),
    )
    parser.add_argument(
        "--player_split_manifest",
        type=str,
        default="",
        help="Optional player_splits.csv to reuse an existing train/val split.",
    )
    parser.add_argument(
        "--embedding_seq_length",
        type=int,
        default=200,
        help=(
            "Pad/truncate each task sequence to this length before embedding extraction. "
            "Use 0 to keep the full max length observed within each task."
        ),
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=64,
        help="Batch size used while extracting frozen task embeddings.",
    )
    parser.add_argument(
        "--embeddings_to_use",
        type=str,
        default="masked_average_all",
        choices=sorted(SUPPORTED_EMBEDDING_POOLERS),
        help="How to pool each frozen task-specific sequence model into a single embedding.",
    )
    parser.add_argument(
        "--max_train_val_players",
        type=int,
        default=8000,
        help="Randomly sample a balanced Planner/Hopper cap of this many players total for PLVM train/validation.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="Validation fraction applied within the sampled balanced player subset.",
    )
    parser.add_argument(
        "--cutoff_schedule_steps",
        type=str,
        default=DEFAULT_CUTOFF_SCHEDULE_STEPS_STR,
        help="Comma-separated fixed step cutoffs used to build partial-observation PLVM samples.",
    )
    parser.add_argument(
        "--cutoff_samples_per_player",
        type=int,
        default=3,
        help="Number of cutoff samples to retain per player from the fixed schedule. Use <=0 to keep all valid cutoffs.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=DEFAULT_MODEL_TYPE,
        choices=[MODEL_TYPE_BOTH_PARTIAL, MODEL_TYPE_SINGLE_TASK],
        help="plvm_both_partial: use both task embeddings. single_task: use one selected task embedding only.",
    )
    parser.add_argument(
        "--prediction_head_type",
        type=str,
        default="auto",
        choices=["auto", "hierarchical", "mlp", "logistic"],
        help="Prediction head. auto: hierarchical for plvm_both_partial, mlp for single_task.",
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
        "--latent_dim",
        type=int,
        default=2,
        help="Latent bottleneck size for hierarchical PLVM heads.",
    )
    parser.add_argument(
        "--encoder_hidden_size",
        type=int,
        default=256,
        help="Encoder hidden size for hierarchical PLVM heads.",
    )
    parser.add_argument(
        "--decoder_hidden_dim",
        type=int,
        default=32,
        help="Decoder hidden size for the label classifier.",
    )
    parser.add_argument(
        "--decoder_num_layers", type=int, default=2, help="Number of decoder MLP layers."
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.2, help="Dropout rate for encoder/decoder layers."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of PLVM training epochs."
    )
    parser.add_argument("--batch_size", type=int, default=128, help="PLVM classifier batch size.")
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="PLVM classifier learning rate."
    )
    parser.add_argument(
        "--lambda_l2", type=float, default=0.05, help="L2 penalty weight for the PLVM classifier."
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm for PLVM training.",
    )
    parser.add_argument(
        "--embedding_standardization",
        type=str,
        default="standardize",
        choices=["standardize", "center", "none"],
        help="Train-only embedding standardization applied before PLVM fitting.",
    )
    parser.add_argument(
        "--dim_reduction_method",
        type=str,
        default="none",
        help="Optional train-only embedding dimensionality reduction, e.g. none or pca_components_64.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        default="run",
        choices=["debug", "run"],
        help="Execution mode.",
    )
    parser.add_argument(
        "--use_early_stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Stop training once the validation loss has not improved for 10%% of --num_epochs "
            "(default: on). --no-use_early_stopping trains for the full --num_epochs."
        ),
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
        help="Base output directory for trained minigrid PLVM label models.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=str(DEFAULT_CACHE_DIR),
        help="Optional shared cache directory for precomputed minigrid PLVM embeddings. Pass an empty string to disable caching.",
    )
    return parser


def run_with_captured_stdout(func, *args, **kwargs):
    """Capture stdout/stderr from shared helpers and mirror it to the logger."""
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        result = func(*args, **kwargs)

    for stream_value in [stdout_buffer.getvalue(), stderr_buffer.getvalue()]:
        stream_value = stream_value.strip()
        if not stream_value:
            continue
        for line in stream_value.splitlines():
            logging.info("%s", line)
    return result


def _save_json(path: Path, payload: Dict[str, object]) -> None:
    """Save a JSON payload with stable formatting."""
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _build_model(embedding_dim: int, args: argparse.Namespace) -> torch.nn.Module:
    """Build the PLVM classification head."""
    if args.prediction_head_type == "mlp":
        hidden_sizes = [args.encoder_hidden_size]
        hidden_sizes.extend([args.decoder_hidden_dim] * max(args.decoder_num_layers - 2, 0))
        model = EmbeddingMLPModel(
            embedding_dim=embedding_dim,
            num_parameters=1,
            dropout_rate=args.dropout_rate,
            positive_only=False,
            task_type="classification",
            num_layers=args.decoder_num_layers,
            hidden_dim=args.decoder_hidden_dim,
            hidden_sizes=hidden_sizes,
            num_classes=args.num_classes,
        )
    else:
        model = EmbeddingHierarchicalModel(
            embedding_dim=embedding_dim,
            low_dim_size=args.latent_dim,
            num_output_parameters=1,
            encoder_hidden_size=args.encoder_hidden_size,
            decoder_hidden_dim=args.decoder_hidden_dim,
            decoder_num_layers=args.decoder_num_layers,
            positive_only=False,
            encoder_dropout_rate=args.dropout_rate,
            decoder_dropout_rate=args.dropout_rate,
            task_type="classification",
            num_classes=args.num_classes,
        )
    return model.to(COMPUTE_DEVICE)


def _resolve_prediction_head_type(args: argparse.Namespace) -> str:
    """Resolve ``auto`` to a concrete head; single-task runs never use the hierarchical head."""
    requested = str(getattr(args, "prediction_head_type", "auto") or "auto").strip().lower()
    model_type = str(getattr(args, "model_type", "") or "").strip().lower()
    if requested == "auto":
        return "mlp" if model_type == MODEL_TYPE_SINGLE_TASK else "hierarchical"
    if requested not in {"hierarchical", "mlp", "logistic"}:
        raise ValueError(
            "Unsupported prediction_head_type={}. Expected one of auto, hierarchical, mlp, logistic.".format(
                requested
            )
        )
    if model_type == MODEL_TYPE_SINGLE_TASK and requested == "hierarchical":
        return "mlp"
    return requested


def _parse_cutoff_schedule_steps(raw_cutoff_schedule_steps: str) -> List[int]:
    """Parse a comma-separated cutoff-step schedule."""
    values = []
    for raw_value in str(raw_cutoff_schedule_steps or "").split(","):
        raw_value = raw_value.strip()
        if not raw_value:
            continue
        values.append(int(raw_value))
    if not values:
        raise ValueError("--cutoff_schedule_steps must contain at least one integer cutoff.")
    return values


def _resolve_task_configuration(args: argparse.Namespace) -> List[str]:
    """Resolve the task branches from --model_type, --partial_task_name and --full_task_name.

    plvm_both_partial needs both task flags or neither (then cleaning and foraging),
    and stores the pair in sorted order; single_task uses --partial_task_name. The
    resolved names are written back to ``args`` (including ``args.task_names``) for
    the run metadata.
    """
    model_type = str(args.model_type).strip().lower()
    full_task_name = str(args.full_task_name or "").strip().lower()
    partial_task_name = str(args.partial_task_name or "").strip().lower()
    allowed_tasks = set(PLVM_TASK_NAMES)

    if model_type == MODEL_TYPE_BOTH_PARTIAL:
        if not full_task_name and not partial_task_name:
            ordered_tasks = sorted(PLVM_TASK_NAMES)
            partial_task_name = ordered_tasks[0]
            full_task_name = ordered_tasks[1]
        elif not full_task_name or not partial_task_name:
            raise ValueError(
                "{} requires both --full_task_name and --partial_task_name.".format(model_type)
            )
        else:
            ordered_tasks = sorted([partial_task_name, full_task_name])
            partial_task_name = ordered_tasks[0]
            full_task_name = ordered_tasks[1]

        if partial_task_name == full_task_name:
            raise ValueError(
                "--full_task_name and --partial_task_name must differ for {}.".format(model_type)
            )
        invalid_tasks = sorted({partial_task_name, full_task_name}.difference(allowed_tasks))
        if invalid_tasks:
            raise ValueError(
                "Unsupported minigrid task(s) {}. Expected subset of {}.".format(
                    invalid_tasks,
                    sorted(allowed_tasks),
                )
            )
        args.partial_task_name = partial_task_name
        args.full_task_name = full_task_name
        args.task_names = ",".join([partial_task_name, full_task_name])
        return [partial_task_name, full_task_name]

    if full_task_name:
        raise ValueError("--full_task_name is only valid for {}.".format(MODEL_TYPE_BOTH_PARTIAL))

    if not partial_task_name:
        raise ValueError("{} requires --partial_task_name.".format(model_type))

    if partial_task_name not in allowed_tasks:
        raise ValueError(
            "Unsupported minigrid task '{}' for single_task. Expected one of {}.".format(
                partial_task_name,
                sorted(allowed_tasks),
            )
        )
    args.partial_task_name = partial_task_name
    args.full_task_name = ""
    args.task_names = partial_task_name
    return [partial_task_name]


def _compute_classification_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> dict:
    """Compute classification metrics and cross-entropy from logits."""
    if logits.numel() == 0:
        return {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "f1_per_class": {},
            "confusion_matrix": [],
            "support": 0,
            "cross_entropy": float("nan"),
        }

    target_tensor = targets.detach().cpu().view(-1).long()
    metrics = compute_classification_metrics(
        target_tensor,
        logits.detach().cpu(),
        num_classes=num_classes,
    )
    metrics["cross_entropy"] = float(
        torch.nn.functional.cross_entropy(
            logits.detach().cpu(),
            target_tensor,
        ).item()
    )
    return metrics


def _compute_majority_baseline_classification(
    train_targets: torch.Tensor,
    eval_targets: torch.Tensor,
    num_classes: int,
) -> dict:
    """Evaluate a majority-class baseline."""
    train_np = train_targets.detach().cpu().view(-1).numpy().astype(np.int64)
    eval_np = eval_targets.detach().cpu().view(-1).numpy().astype(np.int64)
    if train_np.size == 0 or eval_np.size == 0:
        return {
            "majority_class": None,
            "metrics": {
                "accuracy": float("nan"),
                "macro_f1": float("nan"),
                "f1_per_class": {},
                "confusion_matrix": [],
                "support": 0,
            },
        }

    majority_class = int(np.bincount(train_np, minlength=num_classes).argmax())
    preds = torch.full_like(
        eval_targets.detach().cpu().view(-1).long(),
        fill_value=majority_class,
    )
    return {
        "majority_class": majority_class,
        "metrics": compute_classification_metrics(
            eval_targets.detach().cpu().view(-1).long(),
            preds,
            num_classes=num_classes,
        ),
    }


def _compute_random_baseline_classification(
    eval_targets: torch.Tensor,
    num_classes: int,
    seed: int,
) -> dict:
    """Evaluate a reproducible uniform-random classification baseline."""
    target_tensor = eval_targets.detach().cpu().view(-1).long()
    if target_tensor.numel() == 0:
        return {
            "seed": seed,
            "metrics": {
                "accuracy": float("nan"),
                "macro_f1": float("nan"),
                "f1_per_class": {},
                "confusion_matrix": [],
                "support": 0,
            },
        }

    rng = np.random.RandomState(seed)
    preds = torch.tensor(
        rng.randint(0, num_classes, size=target_tensor.shape[0]),
        dtype=torch.long,
    )
    return {
        "seed": seed,
        "metrics": compute_classification_metrics(
            target_tensor,
            preds,
            num_classes=num_classes,
        ),
    }


def _save_candidate_split_manifest(
    log_dir: Path,
    args: argparse.Namespace,
    train_candidates: Sequence[MinigridPLVMSampleCandidate],
    val_candidates: Sequence[MinigridPLVMSampleCandidate],
    task_model_spec: Dict[str, dict],
    dataset_payload: Dict[str, object],
) -> None:
    """Persist the exact player-level train/validation samples used by this run."""
    payload = {
        "model_name": args.model_name,
        "model_type": args.model_type,
        "target_mode": args.target_mode,
        "target_name": args.target_name,
        "tasks": list(args.tasks),
        "seed": int(args.seed),
        "val_fraction": float(args.val_fraction),
        "embedding_seq_length": int(args.embedding_seq_length),
        "cutoff_schedule_steps": [int(value) for value in dataset_payload["cutoff_schedule_steps"]],
        "cutoff_samples_per_player": int(dataset_payload["cutoff_samples_per_player"]),
        "embeddings_to_use": str(args.embeddings_to_use),
        "train_size": int(len(train_candidates)),
        "val_size": int(len(val_candidates)),
        "task_model_spec": {
            task_name: {
                "model_log_dir": str(entry["model_log_dir"]),
                "seq_length": int(entry.get("seq_length", 0) or 0),
            }
            for task_name, entry in task_model_spec.items()
        },
        "task_pad_lengths": dataset_payload["task_pad_lengths"],
        "task_sequence_stats": dataset_payload["task_sequence_stats"],
        "train": [candidate.to_dict() for candidate in train_candidates],
        "val": [candidate.to_dict() for candidate in val_candidates],
    }
    _save_json(Path(log_dir) / "candidate_splits.json", payload)


def _save_val_predictions(
    log_dir: Path,
    model_name: str,
    val_logits: torch.Tensor,
    val_targets: torch.Tensor,
    val_candidates: Sequence[MinigridPLVMSampleCandidate],
    class_names: Sequence[str],
) -> None:
    """Save per-player validation predictions to CSV."""
    logits_np = val_logits.detach().cpu().numpy()
    exp_logits = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
    prob_matrix = exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), 1e-7, None)
    targets_flat = val_targets.detach().cpu().view(-1).long().tolist()
    pred_ids = np.argmax(logits_np, axis=1).tolist()
    pred_probs = prob_matrix.max(axis=1).tolist()

    out_path = Path(log_dir) / "val_predictions.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "player_id",
                "label",
                "class_id",
                "latent_P",
                "latent_O",
                "cutoff_steps",
                "target_label",
                "prediction_class_id",
                "prediction_label",
                "prediction_probability",
                "prediction_probabilities_json",
                "model_name",
            ],
        )
        writer.writeheader()
        for candidate, target_id, pred_id, pred_prob, prob_row in zip(
            val_candidates,
            targets_flat,
            pred_ids,
            pred_probs,
            prob_matrix,
        ):
            writer.writerow(
                {
                    "player_id": int(candidate.player_id),
                    "label": str(candidate.label),
                    "class_id": int(candidate.class_id),
                    "latent_P": float(candidate.latent_P),
                    "latent_O": float(candidate.latent_O),
                    "cutoff_steps": int(candidate.cutoff_steps),
                    "target_label": class_names[int(target_id)],
                    "prediction_class_id": int(pred_id),
                    "prediction_label": class_names[int(pred_id)],
                    "prediction_probability": float(pred_prob),
                    "prediction_probabilities_json": json.dumps(
                        [float(value) for value in prob_row.tolist()]
                    ),
                    "model_name": model_name,
                }
            )
    logging.info("Saved validation predictions to %s", out_path)


def _save_classification_plots(
    args: argparse.Namespace,
    trained_model: torch.nn.Module,
    train_embeddings: torch.Tensor,
    val_embeddings: torch.Tensor,
    train_targets: torch.Tensor,
    val_targets: torch.Tensor,
    train_ids: Sequence[int],
    val_ids: Sequence[int],
) -> None:
    """Save latent-space and prediction diagnostics using the shared plotting helper."""
    visualize_latent = bool(
        getattr(trained_model, "produces_latent", True) and getattr(args, "latent_dim", 4) < 3
    )
    for emb, tgt, ids, folder in [
        (train_embeddings, train_targets, train_ids, "train"),
        (val_embeddings, val_targets, val_ids, "val"),
    ]:
        save_session_specific_plots(
            trained_model,
            trained_model,
            emb,
            tgt,
            args=args,
            folder=folder,
            id_mapping=[str(player_id) for player_id in ids],
            visualize_latent_space=visualize_latent,
            parameter_labels=None,
            color_by_parameters=None,
            color_parameter_labels=None,
            model_name=args.model_name,
        )


def _to_probability_matrix(logits: torch.Tensor) -> np.ndarray:
    """Convert logits to a dense probability matrix on CPU."""
    probabilities = torch.softmax(logits.detach().cpu(), dim=1)
    return probabilities.numpy().astype(np.float32)


def _save_classification_metrics_by_cutoff(
    log_dir: Path,
    model_name: str,
    split_name: str,
    split_logits: torch.Tensor,
    split_targets: torch.Tensor,
    split_candidates: Sequence[MinigridPLVMSampleCandidate],
    class_names: Sequence[str],
) -> None:
    """Save split-specific classification metrics grouped by cutoff steps."""
    logits_cpu = split_logits.detach().cpu()
    targets_cpu = split_targets.detach().cpu().view(-1).long()
    if logits_cpu.numel() == 0 or targets_cpu.numel() == 0 or not split_candidates:
        return

    cutoff_dir = Path(log_dir) / "plots" / str(split_name).strip().lower() / "cutoff_metrics"
    cutoff_dir.mkdir(parents=True, exist_ok=True)

    grouped_indices: Dict[int, List[int]] = {}
    for index, candidate in enumerate(split_candidates):
        grouped_indices.setdefault(int(candidate.cutoff_steps), []).append(index)

    rows = []
    sorted_cutoffs = sorted(grouped_indices)
    accuracy_values = []
    support_values = []
    for cutoff_steps in sorted_cutoffs:
        indices = grouped_indices[cutoff_steps]
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        cutoff_logits = logits_cpu.index_select(0, idx_tensor)
        cutoff_targets = targets_cpu.index_select(0, idx_tensor)
        metrics = _compute_classification_metrics_from_logits(
            cutoff_logits,
            cutoff_targets,
            num_classes=int(cutoff_logits.shape[1]),
        )
        pred_ids = torch.argmax(cutoff_logits, dim=1).tolist()
        target_ids = cutoff_targets.tolist()
        rows.append(
            {
                "cutoff_kind": "observation_steps",
                "cutoff_value": int(cutoff_steps),
                "support": int(len(indices)),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "cross_entropy": float(metrics["cross_entropy"]),
                "target_class_counts_json": json.dumps(
                    [
                        int(target_ids.count(class_id))
                        for class_id in range(int(cutoff_logits.shape[1]))
                    ]
                ),
                "predicted_class_counts_json": json.dumps(
                    [
                        int(pred_ids.count(class_id))
                        for class_id in range(int(cutoff_logits.shape[1]))
                    ]
                ),
                "class_names_json": json.dumps(list(class_names)),
            }
        )
        accuracy_values.append(float(metrics["accuracy"]))
        support_values.append(int(len(indices)))

    csv_path = cutoff_dir / "metrics_by_cutoff.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cutoff_kind",
                "cutoff_value",
                "support",
                "accuracy",
                "macro_f1",
                "cross_entropy",
                "target_class_counts_json",
                "predicted_class_counts_json",
                "class_names_json",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    logging.info("Saved %s classification metrics by cutoff to %s", split_name, csv_path)

    try:
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        ax1.plot(sorted_cutoffs, accuracy_values, marker="o", linewidth=2.0, color="#1f77b4")
        ax1.set_xlabel("Observation steps used")
        ax1.set_ylabel("{} accuracy".format(str(split_name).capitalize()), color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax1.set_ylim(0.0, 1.0)
        ax1.grid(True, axis="y", alpha=0.25)

        ax2 = ax1.twinx()
        ax2.bar(sorted_cutoffs, support_values, alpha=0.18, color="#7f7f7f")
        ax2.set_ylabel("{} samples".format(str(split_name).capitalize()), color="#555555")
        ax2.tick_params(axis="y", labelcolor="#555555")

        fig.suptitle("{} Accuracy by Cutoff".format(str(split_name).capitalize()))
        ax1.set_title(model_name)
        fig.tight_layout()

        pdf_path = cutoff_dir / "accuracy_by_cutoff.pdf"
        fig.savefig(pdf_path)
        plt.close(fig)
        logging.info("Saved %s accuracy-by-cutoff plot to %s", split_name, pdf_path)
    except Exception as exc:
        logging.warning("Could not save %s accuracy-by-cutoff plot: %s", split_name, exc)


def _run_logistic_probe(
    args: argparse.Namespace,
    train_emb: torch.Tensor,
    val_emb: torch.Tensor,
    train_y: torch.Tensor,
    val_y: torch.Tensor,
    train_candidates: Sequence[MinigridPLVMSampleCandidate],
    val_candidates: Sequence[MinigridPLVMSampleCandidate],
    report_naive_train_loss: torch.Tensor,
    report_naive_val_loss: torch.Tensor,
    embedding_dim: int,
    embedding_standardization_metadata: Dict[str, object],
    dim_reduction_metadata: Dict[str, object],
    dim_reduction_artifacts: Dict[str, object],
    embedding_standardization_scaler,
) -> None:
    """Fit and save a frozen-embedding logistic probe for minigrid labels."""
    train_x = train_emb.detach().cpu().numpy()
    val_x = val_emb.detach().cpu().numpy()
    train_labels = train_y.detach().cpu().view(-1).long().numpy()

    logistic_model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        multi_class="auto",
        random_state=int(args.seed),
    )
    logistic_model.fit(train_x, train_labels)

    train_logits_cpu = _log_probability_tensor(
        _predict_logistic_probabilities(logistic_model, train_x, args.num_classes)
    )
    val_logits_cpu = _log_probability_tensor(
        _predict_logistic_probabilities(logistic_model, val_x, args.num_classes)
    )
    train_y_cpu = train_y.detach().cpu()
    val_y_cpu = val_y.detach().cpu()

    train_metrics = _compute_classification_metrics_from_logits(
        train_logits_cpu,
        train_y_cpu,
        num_classes=args.num_classes,
    )
    val_metrics = _compute_classification_metrics_from_logits(
        val_logits_cpu,
        val_y_cpu,
        num_classes=args.num_classes,
    )
    majority_baseline_metrics_train = _compute_majority_baseline_classification(
        train_y_cpu,
        train_y_cpu,
        num_classes=args.num_classes,
    )
    majority_baseline_metrics_val = _compute_majority_baseline_classification(
        train_y_cpu,
        val_y_cpu,
        num_classes=args.num_classes,
    )

    logging.info(
        "Frozen-embedding logistic probe train metrics [player label]: accuracy=%.4f macro_f1=%.4f cross_entropy=%.4f support=%d",
        train_metrics["accuracy"],
        train_metrics["macro_f1"],
        train_metrics["cross_entropy"],
        train_metrics["support"],
    )
    logging.info(
        "Frozen-embedding logistic probe val metrics [player label]: accuracy=%.4f macro_f1=%.4f cross_entropy=%.4f support=%d",
        val_metrics["accuracy"],
        val_metrics["macro_f1"],
        val_metrics["cross_entropy"],
        val_metrics["support"],
    )

    _save_val_predictions(
        log_dir=Path(args.log_dir),
        model_name=args.model_name,
        val_logits=val_logits_cpu,
        val_targets=val_y_cpu,
        val_candidates=val_candidates,
        class_names=args.class_names,
    )
    _save_classification_metrics_by_cutoff(
        log_dir=Path(args.log_dir),
        model_name=args.model_name,
        split_name="train",
        split_logits=train_logits_cpu,
        split_targets=train_y_cpu,
        split_candidates=train_candidates,
        class_names=args.class_names,
    )
    _save_classification_metrics_by_cutoff(
        log_dir=Path(args.log_dir),
        model_name=args.model_name,
        split_name="val",
        split_logits=val_logits_cpu,
        split_targets=val_y_cpu,
        split_candidates=val_candidates,
        class_names=args.class_names,
    )

    model_save_path = Path(args.log_dir) / "{}_model.pkl".format(args.model_name)
    with model_save_path.open("wb") as handle:
        pickle.dump(logistic_model, handle)
    logging.info("Saved logistic probe model to %s", model_save_path)

    _save_embedding_standardization_scaler(args.log_dir, embedding_standardization_scaler)
    _save_dim_reduction_artifacts(args.log_dir, dim_reduction_artifacts)

    update_hierarchical_model_performance_csv(
        csv_file=str(Path(args.log_dir).parent / "model_performance.csv"),
        args=args,
        train_loss=train_metrics["cross_entropy"],
        val_loss=val_metrics["cross_entropy"],
        train_size=len(train_emb),
        val_size=len(val_emb),
        naive_train_loss=float(report_naive_train_loss.item()),
        naive_val_loss=float(report_naive_val_loss.item()),
        train_accuracy=train_metrics["accuracy"],
        val_accuracy=val_metrics["accuracy"],
        naive_train_accuracy=majority_baseline_metrics_train["metrics"].get(
            "accuracy", float("nan")
        ),
        naive_val_accuracy=majority_baseline_metrics_val["metrics"].get("accuracy", float("nan")),
        train_mse=float("nan"),
        val_mse=float("nan"),
        heldout_session_ids="",
    )

    hparams = {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, (str, int, float, bool, list))
    }
    hparams["embedding_dim"] = int(embedding_dim)
    hparams["logistic_model_class"] = "sklearn.linear_model.LogisticRegression"
    hparams.update(embedding_standardization_metadata)
    hparams.update(dim_reduction_metadata)
    hparams["cache_dir"] = str(args.cache_dir or "")
    hparam_path = Path(args.log_dir) / "{}_hyperparameters.json".format(args.model_name)
    with hparam_path.open("w", encoding="utf-8") as handle:
        json.dump(hparams, handle, indent=2, sort_keys=True)
    logging.info("Saved logistic probe hyperparameters to %s", hparam_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run minigrid PLVM label training end to end."""
    parser = make_arg_parser()
    args = parser.parse_args(argv)

    if args.embedding_batch_size <= 0:
        raise ValueError("embedding_batch_size must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")
    if args.latent_dim <= 0:
        raise ValueError("latent_dim must be positive.")
    if args.encoder_hidden_size <= 0:
        raise ValueError("encoder_hidden_size must be positive.")
    if args.decoder_hidden_dim <= 0:
        raise ValueError("decoder_hidden_dim must be positive.")
    if args.decoder_num_layers <= 0:
        raise ValueError("decoder_num_layers must be positive.")
    if args.embedding_seq_length < 0:
        raise ValueError("embedding_seq_length must be >= 0.")
    if args.cutoff_samples_per_player == 0:
        raise ValueError("cutoff_samples_per_player cannot be 0.")

    args.full_task_name = str(args.full_task_name or "").strip()
    args.partial_task_name = str(args.partial_task_name or "").strip()
    args.cutoff_schedule_steps = _parse_cutoff_schedule_steps(args.cutoff_schedule_steps)
    selected_tasks = _resolve_task_configuration(args)
    args.prediction_head_type = _resolve_prediction_head_type(args)

    set_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.model_name == "default-model-0001":
        args.model_name = generate_model_name()
    if args.run_mode == "debug":
        args.num_epochs = min(args.num_epochs, 5)
        args.batch_size = min(args.batch_size, 16)
        args.embedding_batch_size = min(args.embedding_batch_size, 16)
        args.max_train_val_players = min(args.max_train_val_players, 64)
        if not args.model_name.startswith("debug-"):
            args.model_name = "debug-{}".format(args.model_name)

    output_root = _resolved_output_root(Path(args.output_dir), args)
    args.output_dir = str(output_root)
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

    # Fixed run metadata. The fields from max_kld_loss to fusion_type are not used by
    # the label heads; they are set because the shared trainer, plotting helpers and
    # the model_performance.csv registry read them.
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
    args.input_embeddings_source = (
        "minigrid_task_specific_transformers_all_tasks"
        if args.model_type == MODEL_TYPE_BOTH_PARTIAL
        else "minigrid_task_specific_transformers_single_task"
    )
    args.feature_mode = str(args.embeddings_to_use)
    args.model_family = (
        "PLVMBothPartial" if args.model_type == MODEL_TYPE_BOTH_PARTIAL else "SingleTask"
    )
    args.max_kld_loss = 1000.0
    args.aggregate_classification_by_player = False
    args.prediction_horizon = 1
    args.num_slices_per_task = 1
    args.max_trials_per_task = 1
    args.pool_kernel_size = 0
    args.pool_stride = 0
    args.fusion_type = "concat"
    args.seq_length = int(args.embedding_seq_length)
    args.max_norm = float(args.grad_clip_norm)
    args.data_file = str(Path(args.data_dir).expanduser().resolve())

    logging.info("Minigrid PLVM classification configuration:")
    for key, value in sorted(vars(args).items()):
        logging.info("%s: %s", key, value)

    data_dir = Path(args.data_dir).expanduser().resolve()
    task_model_spec = load_minigrid_task_model_spec(
        Path(args.task_model_spec).expanduser().resolve() if args.task_model_spec else None,
        required_task_names=selected_tasks,
    )
    loaded_task_models = load_task_models_and_manifests(task_model_spec)
    dataset_payload = build_paired_player_dataset(
        data_dir,
        loaded_task_models,
        selected_task_names=selected_tasks,
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
        embedding_seq_length=args.embedding_seq_length,
        cache_dir=args.cache_dir or None,
    )

    paired_embeddings, paired_targets, materialized_candidates = (
        materialize_minigrid_plvm_embeddings(
            loaded_task_models,
            dataset_payload["session_task_data_x"],
            dataset_payload["samples"],
            task_names=selected_tasks,
            embeddings_to_use=args.embeddings_to_use,
            batch_size=args.embedding_batch_size,
            cache_dir=args.cache_dir or None,
            data_dir=str(data_dir),
            model_type=args.model_type,
            embedding_seq_length=args.embedding_seq_length,
            cutoff_schedule_steps=dataset_payload["cutoff_schedule_steps"],
            cutoff_samples_per_player=dataset_payload["cutoff_samples_per_player"],
            max_train_val_players=args.max_train_val_players,
            val_fraction=args.val_fraction,
            seed=args.seed,
            player_split_manifest=str(args.player_split_manifest or ""),
        )
    )
    player_ids = [int(candidate.player_id) for candidate in materialized_candidates]

    train_player_id_set = set(dataset_payload["split_payload"]["train_player_ids"])
    val_player_id_set = set(dataset_payload["split_payload"]["val_player_ids"])
    train_indices = [
        index for index, player_id in enumerate(player_ids) if player_id in train_player_id_set
    ]
    val_indices = [
        index for index, player_id in enumerate(player_ids) if player_id in val_player_id_set
    ]
    if not train_indices:
        raise ValueError("No train samples remained after pairing task sequences.")
    if not val_indices:
        raise ValueError("No validation samples remained after pairing task sequences.")

    train_emb = paired_embeddings[train_indices].to(COMPUTE_DEVICE)
    val_emb = paired_embeddings[val_indices].to(COMPUTE_DEVICE)
    train_y = paired_targets[train_indices].to(COMPUTE_DEVICE)
    val_y = paired_targets[val_indices].to(COMPUTE_DEVICE)
    train_candidates = [materialized_candidates[index] for index in train_indices]
    val_candidates = [materialized_candidates[index] for index in val_indices]
    train_ids = [candidate.player_id for candidate in train_candidates]
    val_ids = [candidate.player_id for candidate in val_candidates]

    _save_json(
        model_log_dir / "resolved_task_model_spec.json",
        {
            task_name: {
                "model_name": loaded_task_models[task_name]["model_name"],
                "model_log_dir": str(loaded_task_models[task_name]["model_log_dir"]),
                "hidden_size": int(loaded_task_models[task_name]["hidden_size"]),
                "seq_length": int(task_model_spec[task_name].get("seq_length", 0) or 0),
            }
            for task_name in selected_tasks
        },
    )
    dataset_payload["split_payload"]["split_df"].to_csv(
        model_log_dir / "player_splits.csv", index=False
    )
    _save_json(model_log_dir / "split_summary.json", dataset_payload["split_payload"]["summary"])
    _save_json(
        model_log_dir / "embedding_dataset_summary.json",
        {
            "cache_dir": str(args.cache_dir or ""),
            "task_pad_lengths": dataset_payload["task_pad_lengths"],
            "task_sequence_stats": dataset_payload["task_sequence_stats"],
            "cutoff_schedule_steps": dataset_payload["cutoff_schedule_steps"],
            "cutoff_samples_per_player": dataset_payload["cutoff_samples_per_player"],
            "cutoff_histogram": dataset_payload["cutoff_histogram"],
            "n_total_samples": int(len(dataset_payload["samples"])),
            "n_train_samples": int(len(train_candidates)),
            "n_val_samples": int(len(val_candidates)),
            "n_train_unique_players": int(len(set(train_ids))),
            "n_val_unique_players": int(len(set(val_ids))),
            "train_player_ids": [int(player_id) for player_id in sorted(set(train_ids))],
            "val_player_ids": [int(player_id) for player_id in sorted(set(val_ids))],
        },
    )
    _save_candidate_split_manifest(
        model_log_dir,
        args,
        train_candidates,
        val_candidates,
        task_model_spec,
        dataset_payload,
    )

    dim_reduction_config = _parse_dim_reduction_method(args.dim_reduction_method)
    train_emb, val_emb, embedding_standardization_scaler, embedding_standardization_metadata = (
        _fit_and_apply_embedding_standardization(
            train_emb,
            val_emb,
            args.embedding_standardization,
        )
    )
    train_emb, val_emb, dim_reduction_artifacts, dim_reduction_metadata = (
        _fit_and_apply_dim_reduction(
            train_emb,
            val_emb,
            dim_reduction_config,
            args.seed,
        )
    )

    n_tasks = len(selected_tasks)
    train_item_level = torch.zeros(len(train_emb), n_tasks, 50, device=COMPUTE_DEVICE)
    val_item_level = torch.zeros(len(val_emb), n_tasks, 50, device=COMPUTE_DEVICE)

    naive_train_loss, naive_val_loss, _ = compute_naive_baseline_losses(
        train_y,
        val_y,
        train_item_level,
        val_item_level,
        metric="classification",
        args=args,
    )
    logging.info(
        "Naive baseline cross-entropy loss — train: %.4f | val: %.4f",
        float(naive_train_loss.item()),
        float(naive_val_loss.item()),
    )

    embedding_dim = int(train_emb.shape[1])
    args.input_feature_dim = embedding_dim
    args.task_specific_model_ids = ",".join(
        "{}={}".format(task_name, loaded_task_models[task_name]["model_name"])
        for task_name in selected_tasks
    )
    if args.prediction_head_type == "logistic":
        _run_logistic_probe(
            args=args,
            train_emb=train_emb,
            val_emb=val_emb,
            train_y=train_y,
            val_y=val_y,
            train_candidates=train_candidates,
            val_candidates=val_candidates,
            report_naive_train_loss=naive_train_loss,
            report_naive_val_loss=naive_val_loss,
            embedding_dim=embedding_dim,
            embedding_standardization_metadata=embedding_standardization_metadata,
            dim_reduction_metadata=dim_reduction_metadata,
            dim_reduction_artifacts=dim_reduction_artifacts,
            embedding_standardization_scaler=embedding_standardization_scaler,
        )
        return 0
    model = _build_model(embedding_dim, args)

    patience = int(args.num_epochs * 0.1) if args.use_early_stopping else args.num_epochs
    eval_interval = 2
    trained_model, train_losses, validation_losses = train_embedding_hierarchical_model(
        model=model,
        train_embeddings=train_emb,
        train_y=train_y,
        train_item_level_data=train_item_level,
        val_embeddings=val_emb,
        val_y=val_y,
        val_item_level_data=val_item_level,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        args=args,
        patience=patience,
        eval_interval=eval_interval,
        train_sample_ids=train_ids,
        val_sample_ids=val_ids,
        train_samples_per_id_per_epoch=0,
        sampling_seed=args.seed,
    )

    run_with_captured_stdout(
        plot_loss_curves,
        train_losses,
        validation_losses,
        [],
        args.num_epochs,
        args,
        eval_interval,
        [0, max(float(naive_val_loss.item()) * 4, 1.0)],
    )

    trained_model.eval()
    with torch.no_grad():
        _, train_logits = trained_model(train_emb)
        _, val_logits = trained_model(val_emb)

    train_logits_cpu = train_logits.detach().cpu()
    val_logits_cpu = val_logits.detach().cpu()
    train_y_cpu = train_y.detach().cpu()
    val_y_cpu = val_y.detach().cpu()

    train_metrics = _compute_classification_metrics_from_logits(
        train_logits_cpu,
        train_y_cpu,
        num_classes=args.num_classes,
    )
    val_metrics = _compute_classification_metrics_from_logits(
        val_logits_cpu,
        val_y_cpu,
        num_classes=args.num_classes,
    )
    majority_baseline_metrics_train = _compute_majority_baseline_classification(
        train_y_cpu,
        train_y_cpu,
        num_classes=args.num_classes,
    )
    majority_baseline_metrics_val = _compute_majority_baseline_classification(
        train_y_cpu,
        val_y_cpu,
        num_classes=args.num_classes,
    )
    random_baseline_metrics_train = _compute_random_baseline_classification(
        train_y_cpu,
        num_classes=args.num_classes,
        seed=args.seed,
    )
    random_baseline_metrics_val = _compute_random_baseline_classification(
        val_y_cpu,
        num_classes=args.num_classes,
        seed=args.seed + 1,
    )

    logging.info(
        "Train metrics [player label]: accuracy=%.4f macro_f1=%.4f cross_entropy=%.4f support=%d",
        train_metrics["accuracy"],
        train_metrics["macro_f1"],
        train_metrics["cross_entropy"],
        train_metrics["support"],
    )
    logging.info(
        "Val   metrics [player label]: accuracy=%.4f macro_f1=%.4f cross_entropy=%.4f support=%d",
        val_metrics["accuracy"],
        val_metrics["macro_f1"],
        val_metrics["cross_entropy"],
        val_metrics["support"],
    )
    logging.info("Train majority baseline metrics: %s", majority_baseline_metrics_train)
    logging.info("Val majority baseline metrics: %s", majority_baseline_metrics_val)
    logging.info("Train random baseline metrics: %s", random_baseline_metrics_train)
    logging.info("Val random baseline metrics: %s", random_baseline_metrics_val)

    train_probabilities = _to_probability_matrix(train_logits_cpu)
    val_probabilities = _to_probability_matrix(val_logits_cpu)
    run_with_captured_stdout(
        generate_classification_visualizations,
        train_y_cpu.view(-1).numpy().astype(np.int64),
        train_probabilities,
        str(model_log_dir / "classification" / "train"),
        list(args.class_names),
        "{} train".format(args.target_name),
    )
    run_with_captured_stdout(
        generate_classification_visualizations,
        val_y_cpu.view(-1).numpy().astype(np.int64),
        val_probabilities,
        str(model_log_dir / "classification" / "val"),
        list(args.class_names),
        "{} val".format(args.target_name),
    )

    try:
        _save_classification_plots(
            args,
            trained_model,
            train_emb,
            val_emb,
            train_y,
            val_y,
            train_ids,
            val_ids,
        )
    except Exception as exc:
        logging.warning("Could not save classification plots: %s", exc)

    model_save_path = model_log_dir / "{}_model.pth".format(args.model_name)
    torch.save(trained_model.state_dict(), model_save_path)
    logging.info("Saved model to %s", model_save_path)

    _save_embedding_standardization_scaler(args.log_dir, embedding_standardization_scaler)
    _save_dim_reduction_artifacts(args.log_dir, dim_reduction_artifacts)
    _save_val_predictions(
        model_log_dir,
        args.model_name,
        val_logits_cpu,
        val_y_cpu,
        val_candidates,
        args.class_names,
    )
    _save_classification_metrics_by_cutoff(
        model_log_dir,
        args.model_name,
        "train",
        train_logits_cpu,
        train_y_cpu,
        train_candidates,
        args.class_names,
    )
    _save_classification_metrics_by_cutoff(
        model_log_dir,
        args.model_name,
        "val",
        val_logits_cpu,
        val_y_cpu,
        val_candidates,
        args.class_names,
    )

    update_hierarchical_model_performance_csv(
        csv_file=str(model_log_dir.parent / "model_performance.csv"),
        args=args,
        train_loss=float(
            getattr(
                trained_model,
                "last_epoch_train_loss",
                train_losses[-1] if train_losses else float("nan"),
            )
        ),
        val_loss=float(
            getattr(
                trained_model,
                "last_epoch_val_loss",
                validation_losses[-1] if validation_losses else float("nan"),
            )
        ),
        train_size=len(train_emb),
        val_size=len(val_emb),
        naive_train_loss=float(naive_train_loss.item()),
        naive_val_loss=float(naive_val_loss.item()),
        train_accuracy=train_metrics["accuracy"],
        val_accuracy=val_metrics["accuracy"],
        naive_train_accuracy=majority_baseline_metrics_train["metrics"].get(
            "accuracy", float("nan")
        ),
        naive_val_accuracy=majority_baseline_metrics_val["metrics"].get("accuracy", float("nan")),
        train_mse=float("nan"),
        val_mse=float("nan"),
        heldout_session_ids="",
    )

    best_validation_loss = float(
        getattr(
            trained_model,
            "best_validation_loss",
            min(validation_losses) if validation_losses else float("nan"),
        )
    )
    best_validation_epoch = int(getattr(trained_model, "best_validation_epoch", 0))
    hparams = {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, (str, int, float, bool, list))
    }
    hparams["embedding_dim"] = embedding_dim
    hparams["task_model_dirs"] = {
        task_name: str(loaded_task_models[task_name]["model_log_dir"])
        for task_name in selected_tasks
    }
    hparams["task_model_names"] = {
        task_name: loaded_task_models[task_name]["model_name"] for task_name in selected_tasks
    }
    hparams["cache_dir"] = str(args.cache_dir or "")
    hparams["task_pad_lengths"] = dataset_payload["task_pad_lengths"]
    hparams["task_sequence_stats"] = dataset_payload["task_sequence_stats"]
    hparams["cutoff_schedule_steps"] = dataset_payload["cutoff_schedule_steps"]
    hparams["cutoff_samples_per_player"] = dataset_payload["cutoff_samples_per_player"]
    hparams["cutoff_histogram"] = dataset_payload["cutoff_histogram"]
    hparams.update(embedding_standardization_metadata)
    hparams.update(dim_reduction_metadata)
    hparams["n_train_samples"] = int(len(train_emb))
    hparams["n_val_samples"] = int(len(val_emb))
    hparams["n_train_players"] = int(len(set(train_ids)))
    hparams["n_val_players"] = int(len(set(val_ids)))
    hparams["train_metrics"] = train_metrics
    hparams["val_metrics"] = val_metrics
    hparams["train_majority_baseline_metrics"] = majority_baseline_metrics_train
    hparams["val_majority_baseline_metrics"] = majority_baseline_metrics_val
    hparams["train_random_baseline_metrics"] = random_baseline_metrics_train
    hparams["val_random_baseline_metrics"] = random_baseline_metrics_val
    hparams["naive_train_loss"] = float(naive_train_loss.item())
    hparams["naive_val_loss"] = float(naive_val_loss.item())
    hparams["best_validation_loss"] = best_validation_loss
    hparams["best_validation_epoch"] = best_validation_epoch
    hparams["validation_player_ids"] = [int(player_id) for player_id in sorted(set(val_ids))]
    hparams["player_split_summary"] = dataset_payload["split_payload"]["summary"]
    hparams_path = model_log_dir / "{}_hyperparameters.json".format(args.model_name)
    with hparams_path.open("w", encoding="utf-8") as handle:
        json.dump(hparams, handle, indent=2, default=str)

    _save_json(model_log_dir / "train_metrics.json", train_metrics)
    _save_json(model_log_dir / "val_metrics.json", val_metrics)
    _save_json(
        model_log_dir / "baseline_metrics.json",
        {
            "majority_train": majority_baseline_metrics_train,
            "majority_val": majority_baseline_metrics_val,
            "random_train": random_baseline_metrics_train,
            "random_val": random_baseline_metrics_val,
        },
    )
    _save_json(
        model_log_dir / "training_summary.json",
        {
            "model_name": args.model_name,
            "task_names": list(selected_tasks),
            "device": str(COMPUTE_DEVICE),
            "embedding_dim": embedding_dim,
            "cache_dir": str(args.cache_dir or ""),
            "best_validation_loss": best_validation_loss,
            "best_validation_epoch": best_validation_epoch,
            "cutoff_schedule_steps": dataset_payload["cutoff_schedule_steps"],
            "cutoff_samples_per_player": dataset_payload["cutoff_samples_per_player"],
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "split_summary": dataset_payload["split_payload"]["summary"],
        },
    )
    logging.info(
        "Minigrid PLVM training complete. best_validation_epoch=%d best_validation_loss=%.4f",
        best_validation_epoch,
        best_validation_loss,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
