#!/usr/bin/env python3
"""Train a PowerWash task-specific transformer for future target prediction.

These are the frozen task encoders of the PLVM: one per observed task (Back Garden, Bungalow),
trained on the processed per-attempt CSVs to predict the x/y/z displacement to the next
subtask. Runs are written to ``models/powerwash/task_specific/powerwash_<task>/training_logs/``.
The defaults train the encoders the PLVM uses; run once per task, e.g.::

    python powerwash/train/train_task_specific_behavior_model_powerwash.py --task_name back_garden

The PLVM trainer reads the selected runs from
``models/powerwash/pw_task_specific_next_subtask.json``.
"""

import argparse
import contextlib
import io
import json
import logging
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from powerwash.train.powerwash_model_training_data_utils import (
    DEFAULT_DATA_DIR,
    PowerWashSequenceDataset,
    build_powerwash_sequence_examples,
    compute_powerwash_data_cache_key,
    estimate_recommended_context_length,
    filter_model_ready_sequence_metadata,
    fit_powerwash_feature_manifest,
    list_powerwash_sequence_metadata,
    load_training_data_cache,
    load_training_schema,
    parse_csv_items,
    powerwash_sequence_collate_fn,
    save_feature_manifest,
    save_training_data_cache,
    split_sequence_metadata,
)
from powerwash.train.powerwash_target_utils import (
    TARGET_CHOICES,
    is_binary_target,
)
from plvm import artifact_paths
from plvm.global_utils import set_seed
from plvm.model_training_utils import (
    COMPUTE_DEVICE,
    build_padding_mask,
    generate_model_name,
    initialize_model_and_optimizer,
    log_task_specific_model_performance,
    save_model,
    train_task_specific_model,
    update_task_specific_model_reference_file,
)
from plvm.visualization_utils import plot_and_save_training_curve

# Default targets: x/y/z displacement to the next subtask.
DEFAULT_TARGET_COLS = (
    "delta_x_to_next_subtask",
    "delta_y_to_next_subtask",
    "delta_z_to_next_subtask",
)
# Default file cap; debug runs use DEBUG_MAX_FILES instead.
DEFAULT_MAX_FILES = 6500
DEBUG_MAX_FILES = 256

def fit_target_stds(examples: List, n_targets: int) -> np.ndarray:
    """Compute per-target std from all finite training values for loss normalisation."""
    stds = np.ones(n_targets, dtype=np.float32)
    for i in range(n_targets):
        values = np.concatenate([ex.targets[:, i] for ex in examples])
        finite = values[np.isfinite(values)]
        if finite.size > 1:
            stds[i] = float(np.std(finite))
    return stds


def compute_joint_normalized_mse_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    target_stds: torch.Tensor,
    loss_weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted sum of per-target MSE, each normalised by its training-set std."""
    output_fp32 = output.to(torch.float32)
    target_fp32 = target.to(torch.float32)
    total_loss = output_fp32.sum() * 0.0
    for i in range(output_fp32.shape[-1]):
        out_i = output_fp32[..., i]
        tgt_i = target_fp32[..., i]
        valid_mask = torch.isfinite(out_i) & torch.isfinite(tgt_i)
        if not bool(valid_mask.any().item()):
            continue
        diff = (out_i[valid_mask] - tgt_i[valid_mask]) / target_stds[i].clamp(min=1e-6)
        total_loss = total_loss + loss_weights[i] * torch.mean(diff**2)
    return total_loss


def compute_masked_mse_loss_torch(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute MSE over finite target entries only."""
    if output.shape != target.shape:
        raise ValueError(
            "Shape mismatch for masked loss: output={} target={}".format(
                tuple(output.shape), tuple(target.shape)
            )
        )

    output_fp32 = output.to(torch.float32)
    target_fp32 = target.to(torch.float32)
    valid_mask = torch.isfinite(output_fp32) & torch.isfinite(target_fp32)
    if not bool(valid_mask.any().item()):
        return output_fp32.sum() * 0.0

    diff = output_fp32[valid_mask] - target_fp32[valid_mask]
    return torch.mean(torch.square(diff))


def compute_masked_bce_loss_torch(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute BCE-with-logits over finite target entries only."""
    if output.shape != target.shape:
        raise ValueError(
            "Shape mismatch for masked loss: output={} target={}".format(
                tuple(output.shape), tuple(target.shape)
            )
        )

    output_fp32 = output.to(torch.float32)
    target_fp32 = target.to(torch.float32)
    valid_mask = torch.isfinite(output_fp32) & torch.isfinite(target_fp32)
    if not bool(valid_mask.any().item()):
        return output_fp32.sum() * 0.0

    return torch.nn.functional.binary_cross_entropy_with_logits(
        output_fp32[valid_mask], target_fp32[valid_mask]
    )


def infer_horizon_seconds(target_col: str) -> int:
    """Infer the horizon length encoded in target columns such as progress_next_60s."""
    match = re.search(r"_(\d+)s$", str(target_col))
    if match is None:
        raise ValueError("Could not infer horizon seconds from target_col='{}'".format(target_col))
    return int(match.group(1))


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    n_targets: int = 1,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Collect per-target flattened predictions and targets across all valid timesteps.

    Returns two lists of length ``n_targets``, each entry being a 1-D array of
    finite prediction/target pairs for that target column.
    """
    model.eval()
    pred_chunks: List[List[np.ndarray]] = [[] for _ in range(n_targets)]
    target_chunks: List[List[np.ndarray]] = [[] for _ in range(n_targets)]

    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            batch_x = batch_x.to(COMPUTE_DEVICE, non_blocking=True)
            padding_mask = build_padding_mask(batch_x)
            output = model(batch_x, padding_mask=padding_mask).detach().cpu().numpy()
            target = batch_y.detach().cpu().numpy()
            for i in range(n_targets):
                valid_mask = np.isfinite(target[..., i]) & np.isfinite(output[..., i])
                if not np.any(valid_mask):
                    continue
                pred_chunks[i].append(output[..., i][valid_mask].astype(np.float32))
                target_chunks[i].append(target[..., i][valid_mask].astype(np.float32))

    preds = [
        np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float32)
        for chunks in pred_chunks
    ]
    targets = [
        np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float32)
        for chunks in target_chunks
    ]
    return preds, targets


def compute_regression_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute scalar regression metrics for flattened predictions/targets."""
    if pred.shape != target.shape:
        raise ValueError(
            "Prediction/target shape mismatch: pred={} target={}".format(pred.shape, target.shape)
        )
    if pred.size == 0:
        return {
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
        }

    diff = pred.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(np.square(diff)))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(max(mse, 0.0)))

    target_mean = float(np.mean(target))
    total_sum_squares = float(np.sum(np.square(target - target_mean)))
    residual_sum_squares = float(np.sum(np.square(diff)))
    if total_sum_squares <= 0.0:
        r2 = float("nan")
    else:
        r2 = float(1.0 - (residual_sum_squares / total_sum_squares))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def flatten_targets_from_examples(
    examples: Sequence,
    n_targets: int,
) -> List[np.ndarray]:
    """Flatten targets from many examples into one array per target column."""
    flattened: List[np.ndarray] = []
    for target_index in range(n_targets):
        if not examples:
            flattened.append(np.asarray([], dtype=np.float32))
            continue
        flattened.append(
            np.concatenate(
                [example.targets[:, target_index] for example in examples], axis=0
            ).astype(np.float32)
        )
    return flattened


def build_zero_baseline_predictions(targets: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Create a zero-valued prediction for each flattened target array."""
    return [np.zeros_like(target, dtype=np.float32) for target in targets]


def build_mean_baseline_predictions(
    train_targets: Sequence[np.ndarray],
    eval_targets: Sequence[np.ndarray],
) -> List[np.ndarray]:
    """Create a per-target constant-mean baseline using train split target means."""
    preds: List[np.ndarray] = []
    for train_target, eval_target in zip(train_targets, eval_targets):
        finite_train = np.asarray(train_target, dtype=np.float32)
        finite_train = finite_train[np.isfinite(finite_train)]
        train_mean = float(np.mean(finite_train)) if finite_train.size > 0 else 0.0
        preds.append(np.full_like(eval_target, fill_value=train_mean, dtype=np.float32))
    return preds


def collect_persistence_baseline_predictions(
    examples: Sequence,
    n_targets: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Collect last-value persistence predictions for each target separately."""
    pred_chunks: List[List[np.ndarray]] = [[] for _ in range(n_targets)]
    target_chunks: List[List[np.ndarray]] = [[] for _ in range(n_targets)]

    for example in examples:
        if example.targets.shape[0] < 2:
            continue
        for target_index in range(n_targets):
            prev = example.targets[:-1, target_index].astype(np.float32)
            curr = example.targets[1:, target_index].astype(np.float32)
            valid_mask = np.isfinite(prev) & np.isfinite(curr)
            if not np.any(valid_mask):
                continue
            pred_chunks[target_index].append(prev[valid_mask])
            target_chunks[target_index].append(curr[valid_mask])

    preds = [
        np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float32)
        for chunks in pred_chunks
    ]
    targets = [
        np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float32)
        for chunks in target_chunks
    ]
    return preds, targets


def compute_joint_loss_breakdown(
    preds: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    target_cols: Sequence[str],
    target_stds: Sequence[float],
    loss_weights: Sequence[float],
) -> Dict[str, object]:
    """Compute total joint loss plus per-target contributions on flattened arrays."""
    if not (
        len(preds) == len(targets) == len(target_cols) == len(target_stds) == len(loss_weights)
    ):
        raise ValueError(
            "Joint loss inputs must have the same length: preds={} targets={} target_cols={} target_stds={} loss_weights={}".format(
                len(preds), len(targets), len(target_cols), len(target_stds), len(loss_weights)
            )
        )

    joint_loss = 0.0
    has_valid_component = False
    per_target: Dict[str, Dict[str, float]] = {}

    for target_col, pred, target, target_std, loss_weight in zip(
        target_cols, preds, targets, target_stds, loss_weights
    ):
        pred_arr = np.asarray(pred, dtype=np.float32)
        target_arr = np.asarray(target, dtype=np.float32)
        if pred_arr.shape != target_arr.shape:
            raise ValueError(
                "Prediction/target shape mismatch for '{}': pred={} target={}".format(
                    target_col, pred_arr.shape, target_arr.shape
                )
            )

        valid_mask = np.isfinite(pred_arr) & np.isfinite(target_arr)
        if np.any(valid_mask):
            diff = pred_arr[valid_mask].astype(np.float64) - target_arr[valid_mask].astype(
                np.float64
            )
            raw_mse = float(np.mean(np.square(diff)))
            std_value = max(float(target_std), 1e-6)
            normalized_mse = float(raw_mse / (std_value**2))
            weighted_loss = float(float(loss_weight) * normalized_mse)
            n_valid = int(np.sum(valid_mask))
            joint_loss += weighted_loss
            has_valid_component = True
        else:
            raw_mse = float("nan")
            normalized_mse = float("nan")
            weighted_loss = float("nan")
            n_valid = 0

        per_target[str(target_col)] = {
            "raw_mse": raw_mse,
            "normalized_mse": normalized_mse,
            "weighted_loss": weighted_loss,
            "weight": float(loss_weight),
            "target_std": float(target_std),
            "n_valid": int(n_valid),
        }

    return {
        "joint_loss": float(joint_loss) if has_valid_component else float("nan"),
        "per_target": per_target,
    }


def compute_zero_baseline_metrics(target: np.ndarray) -> Dict[str, float]:
    """Evaluate a zero-progress baseline against flattened targets."""
    baseline_pred = np.zeros_like(target, dtype=np.float32)
    return compute_regression_metrics(baseline_pred, target)


def compute_mean_baseline_metrics(
    train_target: np.ndarray, eval_target: np.ndarray
) -> Dict[str, float]:
    """Evaluate a constant-mean baseline (mean of train targets) against eval targets."""
    train_mean = float(np.nanmean(train_target)) if train_target.size > 0 else 0.0
    baseline_pred = np.full_like(eval_target, fill_value=train_mean, dtype=np.float32)
    return compute_regression_metrics(baseline_pred, eval_target)


def compute_persistence_baseline_metrics(examples: List) -> Dict[str, float]:
    """Evaluate a last-value persistence baseline: predict target[t-1] for target[t].

    Only the first (primary) target column is scored; per-target persistence
    losses for joint runs come from :func:`collect_persistence_baseline_predictions`.
    The first timestep of each sequence is excluded since there is no prior value.
    Sequence boundaries are respected so adjacent sequences are never conflated.
    """
    pred_chunks: List[np.ndarray] = []
    target_chunks: List[np.ndarray] = []
    for example in examples:
        targets = example.targets[:, 0]
        if len(targets) < 2:
            continue
        pred_chunks.append(targets[:-1].astype(np.float32))
        target_chunks.append(targets[1:].astype(np.float32))
    if not pred_chunks:
        return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    return compute_regression_metrics(np.concatenate(pred_chunks), np.concatenate(target_chunks))


def compute_binary_metrics(logits: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute AUC, accuracy, F1, and BCE from raw logits and binary targets."""
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    if logits.size == 0:
        return {
            "accuracy": float("nan"),
            "auc": float("nan"),
            "f1": float("nan"),
            "bce": float("nan"),
        }

    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    hard_preds = (probs >= 0.5).astype(np.int32)
    binary_target = target.astype(np.int32)

    bce = float(
        np.mean(
            -(binary_target * np.log(probs + 1e-7) + (1 - binary_target) * np.log(1 - probs + 1e-7))
        )
    )
    accuracy = float(accuracy_score(binary_target, hard_preds))
    f1 = float(f1_score(binary_target, hard_preds, zero_division=0))
    try:
        auc = float(roc_auc_score(binary_target, probs))
    except ValueError:
        auc = float("nan")

    return {"accuracy": accuracy, "auc": auc, "f1": f1, "bce": bce}


def compute_majority_baseline_metrics(target: np.ndarray) -> Dict[str, float]:
    """Evaluate a majority-class baseline against binary targets."""
    if target.size == 0:
        return {
            "accuracy": float("nan"),
            "auc": float("nan"),
            "f1": float("nan"),
            "bce": float("nan"),
        }

    positive_rate = float(np.mean(target))
    majority_label = 1 if positive_rate >= 0.5 else 0
    hard_preds = np.full_like(target, majority_label, dtype=np.int32)
    binary_target = target.astype(np.int32)

    accuracy = float(np.mean(hard_preds == binary_target))
    # Majority class has constant probability — AUC is 0.5 by definition
    auc = 0.5
    from sklearn.metrics import f1_score

    f1 = float(f1_score(binary_target, hard_preds, zero_division=0))
    # BCE for a calibrated constant predictor: binary entropy H(positive_rate)
    p = np.clip(positive_rate, 1e-7, 1.0 - 1e-7)
    bce = float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))

    return {"accuracy": accuracy, "auc": auc, "f1": f1, "bce": bce}


def compute_binary_persistence_baseline_metrics(examples: List) -> Dict[str, float]:
    """Evaluate a last-value persistence baseline for binary targets.

    Uses empirical transition probabilities P(y=1 | prev=0) and P(y=1 | prev=1)
    as calibrated predictions, so BCE is on the same scale as the majority baseline.
    Sequence boundaries are respected.
    """
    prev_chunks: List[np.ndarray] = []
    target_chunks: List[np.ndarray] = []
    for example in examples:
        targets = example.targets[:, 0]
        if len(targets) < 2:
            continue
        prev = targets[:-1].astype(np.float32)
        curr = targets[1:].astype(np.float32)
        finite = np.isfinite(prev) & np.isfinite(curr)
        if not finite.any():
            continue
        prev_chunks.append(prev[finite])
        target_chunks.append(curr[finite])
    if not prev_chunks:
        return {
            "accuracy": float("nan"),
            "auc": float("nan"),
            "f1": float("nan"),
            "bce": float("nan"),
        }

    all_prev = np.concatenate(prev_chunks)
    all_target = np.concatenate(target_chunks)

    # Calibrated soft probability: empirical P(y=1 | prev_label)
    p_pos_given_prev1 = (
        float(np.mean(all_target[all_prev == 1.0])) if (all_prev == 1.0).any() else 0.5
    )
    p_pos_given_prev0 = (
        float(np.mean(all_target[all_prev == 0.0])) if (all_prev == 0.0).any() else 0.5
    )

    probs = np.where(all_prev == 1.0, p_pos_given_prev1, p_pos_given_prev0)
    probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    logits = np.log(probs / (1.0 - probs)).astype(np.float32)
    return compute_binary_metrics(logits, all_target)


def run_with_captured_stdout(func, *args, **kwargs):
    """Run a helper that prints to stdout and mirror that output into the file logger."""
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        result = func(*args, **kwargs)

    captured_lines = []
    captured_stdout = stdout_buffer.getvalue().strip()
    captured_stderr = stderr_buffer.getvalue().strip()
    if captured_stdout:
        captured_lines.extend(captured_stdout.splitlines())
    if captured_stderr:
        captured_lines.extend(captured_stderr.splitlines())

    for line in captured_lines:
        logging.info("%s", line)
    return result


def make_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for PowerWash task-specific training."""
    parser = argparse.ArgumentParser(
        description="Train a PowerWash task-specific transformer for future target prediction."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing processed per-attempt PowerWash CSV files.",
    )
    parser.add_argument(
        "--schema_path",
        type=str,
        default="",
        help="Optional path to transformer_feature_schema.json.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="",
        help="Optional task filter. Accepts one task or a comma-separated list.",
    )
    parser.add_argument(
        "--target_cols",
        type=str,
        default=",".join(DEFAULT_TARGET_COLS),
        help=(
            "Comma-separated target column(s) to predict. "
            "Single target: standard MSE (or BCE for binary). "
            "Multiple targets: joint normalised-MSE with optional --loss_weights. "
            "Choices per target: {}".format(", ".join(TARGET_CHOICES))
        ),
    )
    parser.add_argument(
        "--loss_weights",
        type=str,
        default="",
        help=(
            "Comma-separated floats, one per --target_cols entry. Defaults to 1.0 for every target."
        ),
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=120,
        help="Maximum context length. Use 0 to keep each file at full length.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help=(
            "Cap on the number of sequence files used before splitting (a seeded "
            "random subset). Defaults to {} ({} with --run_mode debug); "
            "0 disables the cap.".format(DEFAULT_MAX_FILES, DEBUG_MAX_FILES)
        ),
    )

    parser.add_argument("--model_name", type=str, default="default-model-0001", help="Model name.")
    parser.add_argument(
        "--run_mode", type=str, default="run", choices=["debug", "run"], help="Execution mode."
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--max_norm", type=float, default=0.5, help="Gradient clipping max norm.")
    parser.add_argument("--hidden_size", type=int, default=32, help="Transformer hidden size.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers.")
    parser.add_argument("--nhead", type=int, default=4, help="Number of transformer heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="transformer",
        choices=["transformer", "lstm"],
        help="Sequence model architecture.",
    )
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--disable_amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run float32-only training. Pass --no-disable_amp to use CUDA mixed precision.",
    )

    parser.add_argument(
        "--val_fraction", type=float, default=0.08, help="Validation split fraction."
    )
    parser.add_argument("--test_fraction", type=float, default=0.02, help="Test split fraction.")
    parser.add_argument(
        "--split_group",
        type=str,
        default="player",
        choices=["player", "sequence"],
        help="Split by player id to avoid cross-attempt leakage, or by sequence file.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(artifact_paths.POWERWASH_TASK_SPECIFIC_MODELS_DIR),
        help="Base directory for PowerWash model outputs.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=str(artifact_paths.POWERWASH_TASK_SPECIFIC_CACHE_DIR),
        help="Directory for caching processed training data (manifest + split arrays). "
        "Use --no_cache to disable.",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable loading and saving of the training data cache.",
    )
    return parser


def _build_task_label(task_names: Sequence[str]) -> str:
    """Create a stable task label for logs and output folders."""
    if not task_names:
        return "powerwash_all_tasks"
    normalized = [
        str(task_name).strip().lower() for task_name in task_names if str(task_name).strip()
    ]
    if not normalized:
        return "powerwash_all_tasks"
    if len(normalized) == 1:
        return "powerwash_{}".format(normalized[0])
    return "powerwash_" + "_".join(sorted(normalized))


def _build_target_identifier(target_cols: Sequence[str]) -> str:
    """Create a stable filename-safe suffix for one or more target columns."""
    normalized = [str(col).strip().lower() for col in target_cols if str(col).strip()]
    if not normalized:
        return "unknown_target"
    if len(normalized) == 1:
        return normalized[0]
    return "joint__" + "__".join(normalized)


def _serialize_split(items, data_root: Path) -> List[Dict[str, str]]:
    """Create a compact JSON-friendly split manifest."""
    payload = []
    data_root = data_root.expanduser().resolve()
    parent_root = data_root.parent
    for item in items:
        try:
            path_string = str(item.path.relative_to(parent_root))
        except ValueError:
            path_string = str(item.path)
        payload.append(
            {
                "sequence_id": item.sequence_id,
                "pid": item.pid,
                "task_name": item.task_name,
                "attempt_id": item.attempt_id,
                "relative_path": item.relative_path,
                "path": path_string,
            }
        )
    return payload


def main() -> None:
    """Run PowerWash task-specific transformer training end to end."""
    parser = make_arg_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    requested_tasks = parse_csv_items(args.task_name)
    target_cols = parse_csv_items(args.target_cols)
    if not target_cols:
        target_cols = ["progress_next_60s"]
    for col in target_cols:
        if col not in TARGET_CHOICES:
            raise ValueError("Unknown target column '{}'. Choices: {}".format(col, TARGET_CHOICES))
    multi_target_binary_cols = [col for col in target_cols if is_binary_target(col)]
    if len(target_cols) > 1 and multi_target_binary_cols:
        raise ValueError(
            "Joint multi-target training supports regression targets only. "
            "Binary targets are not supported in --target_cols: {}".format(multi_target_binary_cols)
        )

    raw_weights = parse_csv_items(args.loss_weights)
    if raw_weights:
        loss_weights = [float(w) for w in raw_weights]
        if len(loss_weights) != len(target_cols):
            raise ValueError(
                "--loss_weights has {} entries but --target_cols has {}.".format(
                    len(loss_weights), len(target_cols)
                )
            )
    else:
        loss_weights = [1.0] * len(target_cols)

    args.target_cols = target_cols
    args.target_col = target_cols[0] if len(target_cols) == 1 else ""
    args.loss_weights = loss_weights
    args.loss_type = ",".join(target_cols)
    args.prediction_horizon = infer_horizon_seconds(target_cols[0]) if len(target_cols) == 1 else 0
    args.feature_encoding = "powerwash_schema_one_hot"
    args.fold_number = None

    if args.model_name == "default-model-0001":
        args.model_name = generate_model_name()

    if args.run_mode == "debug":
        args.num_epochs = min(args.num_epochs, 5)
        args.batch_size = min(args.batch_size, 32)
        if args.max_files is None:
            args.max_files = DEBUG_MAX_FILES
        if not args.model_name.startswith("debug-"):
            args.model_name = "debug-{}".format(args.model_name)
    if args.max_files is None:
        args.max_files = DEFAULT_MAX_FILES

    data_dir = Path(args.data_dir).expanduser().resolve()
    schema = load_training_schema(data_dir, schema_path=args.schema_path)
    all_items = list_powerwash_sequence_metadata(data_dir, task_names=requested_tasks)
    all_items, prefilter_stats = filter_model_ready_sequence_metadata(
        all_items,
        target_cols=target_cols,
    )

    if args.max_files is not None and args.max_files > 0 and len(all_items) > args.max_files:
        rng = np.random.RandomState(args.seed)
        sampled_indices = np.sort(rng.choice(len(all_items), size=args.max_files, replace=False))
        all_items = [all_items[index] for index in sampled_indices.tolist()]
        logging.info("Applied max_files cap: retained %d sequences", len(all_items))

    observed_tasks = sorted({item.task_name for item in all_items})
    args.task_label = _build_task_label(requested_tasks if requested_tasks else observed_tasks)
    output_root = Path(args.output_dir).expanduser().resolve() / args.task_label
    args.log_dir = str(
        output_root / ("debug_logs" if args.run_mode == "debug" else "training_logs")
    )
    model_log_dir = Path(args.log_dir) / args.model_name
    model_log_dir.mkdir(parents=True, exist_ok=True)

    log_file = model_log_dir / "training.logs"
    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    logging.info("PowerWash task-specific training configuration:")
    for key, value in sorted(vars(args).items()):
        logging.info("%s: %s", key, value)

    logging.info(
        "Discovered %d model-ready PowerWash sequences across tasks=%s and players=%d",
        len(all_items),
        observed_tasks,
        len({item.pid for item in all_items}),
    )
    logging.info("Prefilter stats: %s", prefilter_stats)

    # ------------------------------------------------------------------
    # Training data cache: skip expensive CSV reads on repeated runs when
    # hyperparameters and source files have not changed.
    # ------------------------------------------------------------------
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_key = compute_powerwash_data_cache_key(
        all_items,
        target_cols=target_cols,
        max_seq_length=args.seq_length,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        split_group=args.split_group,
        seed=args.seed,
        max_files=args.max_files,
        schema_path=schema.schema_path,
    )
    logging.info("Training data cache key: %s (dir: %s)", cache_key[:16], cache_dir)

    cached = None
    if not args.no_cache:
        cached = load_training_data_cache(cache_dir, cache_key)

    if cached is not None:
        (
            manifest,
            train_examples,
            val_examples,
            test_examples,
            train_stats,
            val_stats,
            test_stats,
        ) = cached
        train_items = [ex.metadata for ex in train_examples]
        val_items = [ex.metadata for ex in val_examples]
        test_items = [ex.metadata for ex in test_examples]
        logging.info(
            "Cache hit — skipped data loading (train=%d val=%d test=%d)",
            len(train_examples),
            len(val_examples),
            len(test_examples),
        )
    else:
        train_items, val_items, test_items = split_sequence_metadata(
            all_items,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
            split_group=args.split_group,
        )
        logging.info(
            "Split sizes | train=%d val=%d test=%d",
            len(train_items),
            len(val_items),
            len(test_items),
        )

        manifest = fit_powerwash_feature_manifest(
            train_items=train_items,
            schema=schema,
            target_cols=target_cols,
        )

        train_examples, train_stats = build_powerwash_sequence_examples(
            train_items,
            manifest=manifest,
            target_cols=target_cols,
            max_seq_length=args.seq_length,
        )
        val_examples, val_stats = build_powerwash_sequence_examples(
            val_items,
            manifest=manifest,
            target_cols=target_cols,
            max_seq_length=args.seq_length,
        )
        test_examples, test_stats = build_powerwash_sequence_examples(
            test_items,
            manifest=manifest,
            target_cols=target_cols,
            max_seq_length=args.seq_length,
        )

        if not args.no_cache:
            save_training_data_cache(
                cache_dir,
                cache_key,
                manifest,
                train_examples,
                val_examples,
                test_examples,
                train_stats,
                val_stats,
                test_stats,
            )

    # Always write the per-run manifest artifact regardless of cache state.
    save_feature_manifest(manifest, model_log_dir / "feature_manifest.json")

    recommended_context = estimate_recommended_context_length(train_examples)
    logging.info(
        "Model-ready examples | train=%d val=%d test=%d | recommended_context_length=%d",
        len(train_examples),
        len(val_examples),
        len(test_examples),
        recommended_context,
    )
    logging.info("Train stats: %s", train_stats)
    logging.info("Val stats: %s", val_stats)
    logging.info("Test stats: %s", test_stats)

    split_manifest = {
        "target_cols": target_cols,
        "split_group": args.split_group,
        "seed": int(args.seed),
        "requested_tasks": requested_tasks,
        "observed_tasks": observed_tasks,
        "train": _serialize_split(train_items, data_dir),
        "val": _serialize_split(val_items, data_dir),
        "test": _serialize_split(test_items, data_dir),
        "train_stats": train_stats,
        "val_stats": val_stats,
        "test_stats": test_stats,
    }
    with (model_log_dir / "data_splits.json").open("w", encoding="utf-8") as handle:
        json.dump(split_manifest, handle, indent=2)

    train_dataset = PowerWashSequenceDataset(train_examples)
    val_dataset = PowerWashSequenceDataset(val_examples)
    test_dataset = PowerWashSequenceDataset(test_examples)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": powerwash_sequence_collate_fn,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    train_eval_loader = DataLoader(train_dataset, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    n_targets = len(target_cols)
    args.output_size = n_targets
    model, optimizer, scheduler, max_norm = initialize_model_and_optimizer(
        args,
        input_size=manifest.feature_dim,
        output_size=n_targets,
        num_epochs=args.num_epochs,
    )

    binary = n_targets == 1 and is_binary_target(target_cols[0])
    target_identifier = _build_target_identifier(target_cols)
    joint_loss_model_breakdown = {}
    baseline_joint_loss_payload = {}

    if binary:
        train_target_flat = np.concatenate(
            [example.targets[:, 0] for example in train_examples],
            axis=0,
        ).astype(np.float32)
        val_target_flat = np.concatenate(
            [example.targets[:, 0] for example in val_examples],
            axis=0,
        ).astype(np.float32)
        majority_baseline_metrics_val = compute_majority_baseline_metrics(val_target_flat)
        persistence_baseline_metrics_val = compute_binary_persistence_baseline_metrics(val_examples)
        logging.info("Validation majority baseline metrics: %s", majority_baseline_metrics_val)
        logging.info(
            "Validation persistence baseline metrics: %s", persistence_baseline_metrics_val
        )
        loss_fn = compute_masked_bce_loss_torch
        baseline_metrics_payload = {
            "baseline_majority_val_metrics": majority_baseline_metrics_val,
        }
    elif n_targets == 1:
        train_target_flat = np.concatenate(
            [example.targets[:, 0] for example in train_examples],
            axis=0,
        ).astype(np.float32)
        val_target_flat = np.concatenate(
            [example.targets[:, 0] for example in val_examples],
            axis=0,
        ).astype(np.float32)
        zero_baseline_metrics_val = compute_zero_baseline_metrics(val_target_flat)
        mean_baseline_metrics_val = compute_mean_baseline_metrics(
            train_target_flat, val_target_flat
        )
        persistence_baseline_metrics_val = compute_persistence_baseline_metrics(val_examples)
        logging.info("Validation zero baseline metrics: %s", zero_baseline_metrics_val)
        logging.info("Validation mean baseline metrics: %s", mean_baseline_metrics_val)
        logging.info(
            "Validation persistence baseline metrics: %s", persistence_baseline_metrics_val
        )
        loss_fn = compute_masked_mse_loss_torch
        baseline_metrics_payload = {
            "baseline_zero_val_metrics": zero_baseline_metrics_val,
            "baseline_mean_val_metrics": mean_baseline_metrics_val,
        }
    else:
        target_stds = fit_target_stds(train_examples, n_targets)
        logging.info(
            "Target stds for normalisation: %s", dict(zip(target_cols, target_stds.tolist()))
        )
        target_stds_t = torch.tensor(target_stds, dtype=torch.float32).to(COMPUTE_DEVICE)
        loss_weights_t = torch.tensor(loss_weights, dtype=torch.float32).to(COMPUTE_DEVICE)
        loss_fn = lambda out, tgt: compute_joint_normalized_mse_loss(
            out, tgt, target_stds_t, loss_weights_t
        )
        train_target_arrays = flatten_targets_from_examples(train_examples, n_targets)
        val_target_arrays = flatten_targets_from_examples(val_examples, n_targets)
        zero_baseline_val_breakdown = compute_joint_loss_breakdown(
            build_zero_baseline_predictions(val_target_arrays),
            val_target_arrays,
            target_cols,
            target_stds,
            loss_weights,
        )
        mean_baseline_val_breakdown = compute_joint_loss_breakdown(
            build_mean_baseline_predictions(train_target_arrays, val_target_arrays),
            val_target_arrays,
            target_cols,
            target_stds,
            loss_weights,
        )
        persistence_val_preds, persistence_val_targets = collect_persistence_baseline_predictions(
            val_examples,
            n_targets,
        )
        persistence_baseline_val_breakdown = compute_joint_loss_breakdown(
            persistence_val_preds,
            persistence_val_targets,
            target_cols,
            target_stds,
            loss_weights,
        )
        # Scalar persistence metrics for the primary target only (saved as
        # baseline_persistence_val_metrics); the breakdown above covers every target.
        persistence_baseline_metrics_val = compute_persistence_baseline_metrics(val_examples)
        logging.info(
            "Validation persistence baseline metrics (primary target '%s'): %s",
            target_cols[0],
            persistence_baseline_metrics_val,
        )
        logging.info(
            "Validation zero baseline joint loss breakdown: %s", zero_baseline_val_breakdown
        )
        logging.info(
            "Validation mean baseline joint loss breakdown: %s", mean_baseline_val_breakdown
        )
        logging.info(
            "Validation persistence baseline joint loss breakdown: %s",
            persistence_baseline_val_breakdown,
        )
        baseline_metrics_payload = {}
        baseline_joint_loss_payload = {
            "baseline_zero_val_joint_loss_breakdown": zero_baseline_val_breakdown,
            "baseline_mean_val_joint_loss_breakdown": mean_baseline_val_breakdown,
            "baseline_persistence_val_joint_loss_breakdown": persistence_baseline_val_breakdown,
        }

    losses, val_losses = train_task_specific_model(
        args,
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        scheduler,
        max_norm,
        patience=max(20, args.num_epochs // 3),
        validation_interval=max(1, args.num_epochs // 10),
        use_amp=not args.disable_amp,
    )

    train_preds, train_targets = collect_predictions(model, train_eval_loader, n_targets)
    val_preds, val_targets = collect_predictions(model, val_loader, n_targets)
    test_preds, test_targets = collect_predictions(model, test_loader, n_targets)

    # Primary metrics use target index 0 for scalar logging / reference files.
    if binary:
        train_metrics = compute_binary_metrics(train_preds[0], train_targets[0])
        val_metrics = compute_binary_metrics(val_preds[0], val_targets[0])
        test_metrics = compute_binary_metrics(test_preds[0], test_targets[0])
        train_metrics_by_target = {target_cols[0]: train_metrics}
        val_metrics_by_target = {target_cols[0]: val_metrics}
        test_metrics_by_target = {target_cols[0]: test_metrics}
        majority_baseline_metrics_test = compute_majority_baseline_metrics(test_targets[0])
        persistence_baseline_metrics_test = compute_binary_persistence_baseline_metrics(
            test_examples
        )
        logging.info("Train metrics: %s", train_metrics)
        logging.info("Validation metrics: %s", val_metrics)
        logging.info("Test metrics: %s", test_metrics)
        logging.info("Test majority baseline metrics: %s", majority_baseline_metrics_test)
        logging.info("Test persistence baseline metrics: %s", persistence_baseline_metrics_test)
        baseline_metrics_payload["baseline_majority_test_metrics"] = majority_baseline_metrics_test
    else:
        if n_targets > 1:
            joint_loss_model_breakdown = {
                "train": compute_joint_loss_breakdown(
                    train_preds,
                    train_targets,
                    target_cols,
                    target_stds,
                    loss_weights,
                ),
                "val": compute_joint_loss_breakdown(
                    val_preds,
                    val_targets,
                    target_cols,
                    target_stds,
                    loss_weights,
                ),
                "test": compute_joint_loss_breakdown(
                    test_preds,
                    test_targets,
                    target_cols,
                    target_stds,
                    loss_weights,
                ),
            }
        train_metrics_by_target = {
            col: compute_regression_metrics(train_preds[i], train_targets[i])
            for i, col in enumerate(target_cols)
        }
        val_metrics_by_target = {
            col: compute_regression_metrics(val_preds[i], val_targets[i])
            for i, col in enumerate(target_cols)
        }
        test_metrics_by_target = {
            col: compute_regression_metrics(test_preds[i], test_targets[i])
            for i, col in enumerate(target_cols)
        }
        train_metrics = train_metrics_by_target[target_cols[0]]
        val_metrics = val_metrics_by_target[target_cols[0]]
        test_metrics = test_metrics_by_target[target_cols[0]]
        persistence_baseline_metrics_test = compute_persistence_baseline_metrics(test_examples)
        logging.info("Train metrics (primary target '%s'): %s", target_cols[0], train_metrics)
        logging.info("Validation metrics (primary target '%s'): %s", target_cols[0], val_metrics)
        logging.info("Test metrics (primary target '%s'): %s", target_cols[0], test_metrics)
        logging.info(
            "Test persistence baseline metrics (primary target '%s'): %s",
            target_cols[0],
            persistence_baseline_metrics_test,
        )
        if n_targets == 1:
            test_target_flat = test_targets[0]
            zero_baseline_metrics_test = compute_zero_baseline_metrics(test_target_flat)
            mean_baseline_metrics_test = compute_mean_baseline_metrics(
                train_target_flat, test_target_flat
            )
            logging.info("Test zero baseline metrics: %s", zero_baseline_metrics_test)
            logging.info("Test mean baseline metrics: %s", mean_baseline_metrics_test)
            baseline_metrics_payload.update(
                {
                    "baseline_zero_test_metrics": zero_baseline_metrics_test,
                    "baseline_mean_test_metrics": mean_baseline_metrics_test,
                }
            )
        if n_targets > 1:
            test_target_arrays = flatten_targets_from_examples(test_examples, n_targets)
            zero_baseline_test_breakdown = compute_joint_loss_breakdown(
                build_zero_baseline_predictions(test_target_arrays),
                test_target_arrays,
                target_cols,
                target_stds,
                loss_weights,
            )
            mean_baseline_test_breakdown = compute_joint_loss_breakdown(
                build_mean_baseline_predictions(train_target_arrays, test_target_arrays),
                test_target_arrays,
                target_cols,
                target_stds,
                loss_weights,
            )
            persistence_test_preds, persistence_test_targets = (
                collect_persistence_baseline_predictions(
                    test_examples,
                    n_targets,
                )
            )
            persistence_baseline_test_breakdown = compute_joint_loss_breakdown(
                persistence_test_preds,
                persistence_test_targets,
                target_cols,
                target_stds,
                loss_weights,
            )
            baseline_joint_loss_payload.update(
                {
                    "baseline_zero_test_joint_loss_breakdown": zero_baseline_test_breakdown,
                    "baseline_mean_test_joint_loss_breakdown": mean_baseline_test_breakdown,
                    "baseline_persistence_test_joint_loss_breakdown": persistence_baseline_test_breakdown,
                }
            )
            logging.info("Train joint loss breakdown: %s", joint_loss_model_breakdown["train"])
            logging.info("Validation joint loss breakdown: %s", joint_loss_model_breakdown["val"])
            logging.info("Test joint loss breakdown: %s", joint_loss_model_breakdown["test"])
            logging.info(
                "Test zero baseline joint loss breakdown: %s", zero_baseline_test_breakdown
            )
            logging.info(
                "Test mean baseline joint loss breakdown: %s", mean_baseline_test_breakdown
            )
            logging.info(
                "Test persistence baseline joint loss breakdown: %s",
                persistence_baseline_test_breakdown,
            )
            logging.info("Per-target train metrics: %s", train_metrics_by_target)
            logging.info("Per-target validation metrics: %s", val_metrics_by_target)
            logging.info("Per-target test metrics: %s", test_metrics_by_target)
    _ex = train_dataset.examples[0]
    logging.info(
        "Dataset sizes: Train=%d, Val=%d, Test=%d | feature_dim=%d, seq_len=%d",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
        _ex.features.shape[-1],
        _ex.effective_length,
    )

    model_path = model_log_dir / "{}_model.pth".format(args.model_name)
    run_with_captured_stdout(save_model, model, str(model_path))
    finite_loss_values = [
        float(loss_value)
        for loss_value in list(losses) + list(val_losses)
        if np.isfinite(loss_value)
    ]
    plot_ylim_upper = min(finite_loss_values) * 2 if finite_loss_values else 0.1
    if plot_ylim_upper <= 0:
        plot_ylim_upper = 0.1

    run_with_captured_stdout(
        plot_and_save_training_curve,
        args.log_dir,
        args.model_name,
        losses,
        val_losses,
        args.num_epochs,
        args,
        plot_ylims=[0, plot_ylim_upper],
    )

    final_train_loss = float(losses[-1]) if losses else float("nan")
    final_val_loss = float(val_losses[-1]) if val_losses else float("nan")
    log_task_specific_model_performance(
        X_train_size=len(train_dataset),
        X_val_size=len(val_dataset),
        train_loss=final_train_loss,
        val_loss=final_val_loss,
        args=args,
        log_csv_path=os.path.join(args.log_dir, "model_performance.csv"),
    )

    reference_dir = output_root / "references"
    reference_dir.mkdir(parents=True, exist_ok=True)
    reference_file_path = reference_dir / "best_powerwash_task_model_ids_{}_{}.json".format(
        target_identifier,
        datetime.now().strftime("%Y%m%d"),
    )
    update_task_specific_model_reference_file(
        args,
        reference_file_path=str(reference_file_path),
        validation_ids=[example.metadata.sequence_id for example in val_examples],
    )

    metrics_payload = {
        "model_name": args.model_name,
        "timestamp": datetime.now().isoformat(),
        "device": str(COMPUTE_DEVICE),
        "task_label": args.task_label,
        "target_cols": target_cols,
        "target_identifier": target_identifier,
        "loss_weights": [float(weight) for weight in loss_weights],
        "task_names": observed_tasks,
        "split_group": args.split_group,
        "n_features": int(manifest.feature_dim),
        "n_numeric_features": int(len(manifest.numeric_columns)),
        "n_categorical_features": int(len(manifest.categorical_columns)),
        "context_length": int(args.seq_length),
        "recommended_context_length": int(recommended_context),
        "n_sequences_train": int(len(train_examples)),
        "n_sequences_val": int(len(val_examples)),
        "n_sequences_test": int(len(test_examples)),
        "target_type": "binary" if binary else "regression",
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_metrics_by_target": train_metrics_by_target,
        "val_metrics_by_target": val_metrics_by_target,
        "test_metrics_by_target": test_metrics_by_target,
        "joint_loss_breakdown": joint_loss_model_breakdown,
        "baseline_persistence_val_metrics": persistence_baseline_metrics_val,
        "baseline_persistence_test_metrics": persistence_baseline_metrics_test,
        **baseline_metrics_payload,
        **baseline_joint_loss_payload,
        "train_stats": train_stats,
        "val_stats": val_stats,
        "test_stats": test_stats,
        "feature_columns": manifest.feature_columns,
        "numeric_columns": manifest.numeric_columns,
        "categorical_columns": manifest.categorical_columns,
    }
    with (model_log_dir / "evaluation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    logging.info("Saved PowerWash artifacts to %s", model_log_dir)
    logging.info("Model dir: %s", model_log_dir)
    if binary:
        logging.info("Validation AUC: %.6f", val_metrics["auc"])
        logging.info("Test AUC: %.6f", test_metrics["auc"])
    else:
        logging.info("Validation RMSE (%s): %.6f", target_cols[0], val_metrics["rmse"])
        logging.info("Test RMSE (%s): %.6f", target_cols[0], test_metrics["rmse"])
    logging.info("Done")


if __name__ == "__main__":
    main()
