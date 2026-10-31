"""Train a task-specific transformer to predict next action on minigrid simulations."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from simulation.config import TASK_NAME_CLEANING, TASK_NAME_FORAGING
from simulation.train.data_utils import (
    ACTION_VOCAB_BY_TASK,
    DEFAULT_SIMULATION_DATA_DIR,
    NEXT_ACTION_IGNORE_INDEX,
    MinigridSequenceDataset,
    build_minigrid_sequence_examples,
    build_player_splits,
    fit_minigrid_feature_manifest,
    load_simulation_tables,
    load_task_layout,
    minigrid_sequence_collate_fn,
    save_feature_manifest,
)
from plvm import artifact_paths
from plvm.global_utils import set_seed
from plvm.model_training_utils import (
    COMPUTE_DEVICE,
    build_padding_mask,
    compute_classification_metrics,
    generate_classification_visualizations,
    generate_model_name,
    initialize_model_and_optimizer,
    log_task_specific_model_performance,
    save_model,
    update_task_specific_model_reference_file,
)
from plvm.visualization_utils import plot_and_save_training_curve

DEFAULT_OUTPUT_DIR = artifact_paths.SIMULATION_TASK_SPECIFIC_MODELS_DIR


def make_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for task-specific next-action training."""
    parser = argparse.ArgumentParser(
        description="Train a task-specific transformer to predict next action on minigrid simulations."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DEFAULT_SIMULATION_DATA_DIR),
        help="Simulation output directory containing players.csv, events.csv, and environment_metadata.json.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        choices=[TASK_NAME_FORAGING, TASK_NAME_CLEANING],
        help="Task-specific sequence model to train.",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=200,
        help="Maximum context length per chunk. Use 0 to keep full player-task sequences.",
    )
    parser.add_argument(
        "--max_train_val_players",
        type=int,
        default=4000,
        help="Randomly sample a balanced Planner/Hopper cap of this many players total for train/val.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="Validation fraction applied within the sampled balanced player subset.",
    )
    parser.add_argument(
        "--local_patch_radius",
        type=int,
        default=2,
        help="Radius for the egocentric terrain patch encoded at each timestep.",
    )
    parser.add_argument("--model_name", type=str, default="default-model-0001", help="Model name.")
    parser.add_argument(
        "--run_mode", type=str, default="run", choices=["debug", "run"], help="Execution mode."
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--max_norm", type=float, default=0.5, help="Gradient clipping max norm.")
    parser.add_argument("--hidden_size", type=int, default=256, help="Transformer hidden size.")
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Base output directory for trained next-action models.",
    )
    parser.add_argument(
        "--disable_amp",
        action="store_true",
        help="Disable CUDA mixed precision and run float32-only training.",
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


def compute_majority_baseline_metrics(
    train_labels: np.ndarray,
    eval_labels: np.ndarray,
    num_classes: int,
) -> Dict[str, object]:
    """Evaluate a majority-class baseline on flattened next-action labels."""
    if train_labels.size == 0 or eval_labels.size == 0:
        raise ValueError("Majority baseline requires non-empty train and eval labels.")
    counts = np.bincount(train_labels.astype(np.int64), minlength=num_classes)
    majority_class = int(np.argmax(counts))
    predictions = np.full_like(eval_labels, fill_value=majority_class, dtype=np.int64)
    metrics = compute_classification_metrics(eval_labels, predictions, num_classes=num_classes)
    metrics["majority_class"] = majority_class
    return metrics


def compute_next_action_loss(
    logits: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, int, int]:
    """Compute masked cross-entropy and accuracy counts for padded sequences."""
    if logits.dim() != 3:
        raise ValueError(
            f"Expected logits with shape (batch, seq, classes), got {tuple(logits.shape)}"
        )
    if labels.dim() != 2:
        raise ValueError(f"Expected labels with shape (batch, seq), got {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"Logit/label sequence shape mismatch: logits={tuple(logits.shape)} labels={tuple(labels.shape)}"
        )

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    valid_mask = flat_labels != NEXT_ACTION_IGNORE_INDEX
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return logits.sum() * 0.0, 0, 0

    valid_logits = flat_logits[valid_mask]
    valid_labels = flat_labels[valid_mask]
    loss = F.cross_entropy(valid_logits, valid_labels, reduction="sum")
    predictions = valid_logits.argmax(dim=1)
    correct = int((predictions == valid_labels).sum().item())
    return loss, correct, valid_count


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    amp_enabled: bool,
) -> Dict[str, float]:
    """Evaluate loss and accuracy over one loader."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            batch_x = batch_x.to(COMPUTE_DEVICE, non_blocking=True)
            batch_y = batch_y.to(COMPUTE_DEVICE, non_blocking=True)
            padding_mask = build_padding_mask(batch_x)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(batch_x, padding_mask=padding_mask)
                loss_sum, batch_correct, batch_count = compute_next_action_loss(logits, batch_y)
            total_loss += float(loss_sum.detach().item())
            total_correct += int(batch_correct)
            total_count += int(batch_count)

    mean_loss = float(total_loss / max(1, total_count))
    accuracy = float(total_correct / max(1, total_count))
    return {
        "loss": mean_loss,
        "accuracy": accuracy,
        "support": int(total_count),
    }


def collect_probabilities_and_labels(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    amp_enabled: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect valid softmax probabilities and labels across one dataset split."""
    model.eval()
    all_probabilities: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            batch_x = batch_x.to(COMPUTE_DEVICE, non_blocking=True)
            batch_y = batch_y.to(COMPUTE_DEVICE, non_blocking=True)
            padding_mask = build_padding_mask(batch_x)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(batch_x, padding_mask=padding_mask)
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_labels = batch_y.reshape(-1)
            valid_mask = flat_labels != NEXT_ACTION_IGNORE_INDEX
            if not bool(valid_mask.any().item()):
                continue
            valid_logits = flat_logits[valid_mask]
            valid_labels = flat_labels[valid_mask]
            probs = torch.softmax(valid_logits, dim=1).detach().cpu().numpy().astype(np.float32)
            labels = valid_labels.detach().cpu().numpy().astype(np.int64)
            all_probabilities.append(probs)
            all_labels.append(labels)

    if not all_probabilities:
        num_classes = int(getattr(model, "output_size", 1))
        return np.zeros((0, num_classes), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.vstack(all_probabilities), np.concatenate(all_labels)


def train_next_action_model(
    args: argparse.Namespace,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    max_norm: float,
) -> Dict[str, object]:
    """Train a next-action classifier and keep the best checkpoint by validation accuracy."""
    amp_enabled = torch.cuda.is_available() and not args.disable_amp
    scaler = GradScaler(enabled=amp_enabled)

    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accuracies: List[float] = []
    val_accuracies: List[float] = []
    history_rows: List[Dict[str, object]] = []

    best_state_dict = deepcopy(model.state_dict())
    best_epoch = -1
    best_val_accuracy = float("-inf")
    best_val_loss = float("inf")

    for epoch_index in range(args.num_epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_count = 0

        for batch_x, batch_y, _, _ in train_loader:
            batch_x = batch_x.to(COMPUTE_DEVICE, non_blocking=True)
            batch_y = batch_y.to(COMPUTE_DEVICE, non_blocking=True)
            padding_mask = build_padding_mask(batch_x)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(batch_x, padding_mask=padding_mask)
                loss_sum, batch_correct, batch_count = compute_next_action_loss(logits, batch_y)
                if batch_count == 0:
                    continue
                loss = loss_sum / batch_count

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss_sum += float(loss_sum.detach().item())
            epoch_correct += int(batch_correct)
            epoch_count += int(batch_count)

        if scheduler is not None:
            scheduler.step()

        train_loss = float(epoch_loss_sum / max(1, epoch_count))
        train_accuracy = float(epoch_correct / max(1, epoch_count))
        val_metrics = evaluate_model(model, val_loader, amp_enabled=amp_enabled)
        val_loss = float(val_metrics["loss"])
        val_accuracy = float(val_metrics["accuracy"])

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        history_rows.append(
            {
                "epoch": int(epoch_index + 1),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
            }
        )

        logging.info(
            "Epoch [%d/%d] train_loss=%.4f val_loss=%.4f train_acc=%.4f val_acc=%.4f",
            epoch_index + 1,
            args.num_epochs,
            train_loss,
            val_loss,
            train_accuracy,
            val_accuracy,
        )

        if val_accuracy > best_val_accuracy or (
            np.isclose(val_accuracy, best_val_accuracy) and val_loss < best_val_loss
        ):
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_epoch = int(epoch_index + 1)
            best_state_dict = deepcopy(model.state_dict())

    model.load_state_dict(best_state_dict)
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "history_rows": history_rows,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,
    }


def _build_task_label(task_name: str) -> str:
    """Create a stable label for logs and reference files."""
    return f"minigrid_{task_name}_next_action"


def _save_json(path: Path, payload: Dict[str, object]) -> None:
    """Save a JSON payload with stable formatting."""
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run task-specific next-action training end to end."""
    parser = make_arg_parser()
    args = parser.parse_args(argv)

    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")
    if args.hidden_size <= 0:
        raise ValueError("hidden_size must be positive.")
    if args.num_layers <= 0:
        raise ValueError("num_layers must be positive.")
    if args.nhead <= 0:
        raise ValueError("nhead must be positive.")
    if args.model_type == "transformer" and (args.hidden_size % args.nhead) != 0:
        raise ValueError("hidden_size must be divisible by nhead for transformer models.")

    set_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.model_name == "default-model-0001":
        args.model_name = generate_model_name()
    if args.run_mode == "debug":
        args.num_epochs = min(args.num_epochs, 5)
        args.batch_size = min(args.batch_size, 16)
        args.max_train_val_players = min(args.max_train_val_players, 32)
        if not args.model_name.startswith("debug-"):
            args.model_name = f"debug-{args.model_name}"

    args.task_label = _build_task_label(args.task_name)
    args.prediction_horizon = 1
    args.loss_type = "next_action_classification"
    args.feature_encoding = f"env_metadata_patch_r{args.local_patch_radius}"
    args.fold_number = None

    output_root = Path(args.output_dir).expanduser().resolve() / args.task_name
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

    logging.info("Minigrid task-specific training configuration:")
    for key, value in sorted(vars(args).items()):
        logging.info("%s: %s", key, value)

    data_dir = Path(args.data_dir).expanduser().resolve()
    players_df, events_df = load_simulation_tables(data_dir, args.task_name)
    layout = load_task_layout(data_dir, args.task_name)
    split_payload = build_player_splits(
        players_df,
        max_train_val_players=args.max_train_val_players,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    manifest = fit_minigrid_feature_manifest(
        task_name=args.task_name,
        layout=layout,
        local_patch_radius=args.local_patch_radius,
    )

    save_feature_manifest(manifest, model_log_dir / "feature_manifest.json")
    split_payload["split_df"].to_csv(model_log_dir / "player_splits.csv", index=False)
    _save_json(model_log_dir / "split_summary.json", split_payload["summary"])

    train_examples, train_stats = build_minigrid_sequence_examples(
        events_df,
        player_ids=split_payload["train_player_ids"],
        layout=layout,
        manifest=manifest,
        max_seq_length=args.seq_length,
    )
    val_examples, val_stats = build_minigrid_sequence_examples(
        events_df,
        player_ids=split_payload["val_player_ids"],
        layout=layout,
        manifest=manifest,
        max_seq_length=args.seq_length,
    )
    if not train_examples:
        raise ValueError("No train examples were generated.")
    if not val_examples:
        raise ValueError("No validation examples were generated.")

    _save_json(
        model_log_dir / "data_splits.json",
        {
            "task_name": args.task_name,
            "max_train_val_players": int(args.max_train_val_players),
            "ignored_player_ids": [
                int(player_id) for player_id in split_payload["ignored_player_ids"]
            ],
            "train_player_ids": [int(player_id) for player_id in split_payload["train_player_ids"]],
            "val_player_ids": [int(player_id) for player_id in split_payload["val_player_ids"]],
            "train_sequence_ids": [example.metadata.sequence_id for example in train_examples],
            "val_sequence_ids": [example.metadata.sequence_id for example in val_examples],
            "train_stats": train_stats,
            "val_stats": val_stats,
        },
    )

    train_dataset = MinigridSequenceDataset(train_examples)
    val_dataset = MinigridSequenceDataset(val_examples)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": minigrid_sequence_collate_fn,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    train_eval_loader = DataLoader(train_dataset, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    args.output_size = len(ACTION_VOCAB_BY_TASK[args.task_name])
    args.num_classes = args.output_size
    model, optimizer, scheduler, max_norm = initialize_model_and_optimizer(
        args,
        input_size=manifest.feature_dim,
        output_size=args.output_size,
        num_epochs=max(args.num_epochs, 4),
    )

    training_result = train_next_action_model(
        args=args,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        max_norm=max_norm,
    )

    train_probabilities, train_labels = collect_probabilities_and_labels(
        model,
        train_eval_loader,
        amp_enabled=torch.cuda.is_available() and not args.disable_amp,
    )
    val_probabilities, val_labels = collect_probabilities_and_labels(
        model,
        val_loader,
        amp_enabled=torch.cuda.is_available() and not args.disable_amp,
    )

    train_predictions = np.argmax(train_probabilities, axis=1)
    val_predictions = np.argmax(val_probabilities, axis=1)
    train_metrics = compute_classification_metrics(
        train_labels,
        train_predictions,
        num_classes=args.output_size,
    )
    val_metrics = compute_classification_metrics(
        val_labels,
        val_predictions,
        num_classes=args.output_size,
    )
    baseline_train_metrics = compute_majority_baseline_metrics(
        train_labels,
        train_labels,
        num_classes=args.output_size,
    )
    baseline_val_metrics = compute_majority_baseline_metrics(
        train_labels,
        val_labels,
        num_classes=args.output_size,
    )

    logging.info("Train metrics: %s", train_metrics)
    logging.info("Validation metrics: %s", val_metrics)
    logging.info("Majority baseline train metrics: %s", baseline_train_metrics)
    logging.info("Majority baseline val metrics: %s", baseline_val_metrics)

    run_with_captured_stdout(
        generate_classification_visualizations,
        train_labels,
        train_probabilities,
        str(model_log_dir / "classification" / "train"),
        ACTION_VOCAB_BY_TASK[args.task_name],
        f"{args.task_label} train",
    )
    run_with_captured_stdout(
        generate_classification_visualizations,
        val_labels,
        val_probabilities,
        str(model_log_dir / "classification" / "val"),
        ACTION_VOCAB_BY_TASK[args.task_name],
        f"{args.task_label} val",
    )

    model_path = model_log_dir / f"{args.model_name}_model.pth"
    run_with_captured_stdout(save_model, model, str(model_path))

    history_df = np.asarray(
        [
            [
                row["epoch"],
                row["train_loss"],
                row["val_loss"],
                row["train_accuracy"],
                row["val_accuracy"],
            ]
            for row in training_result["history_rows"]
        ],
        dtype=np.float64,
    )
    if history_df.size:
        import pandas as pd

        pd.DataFrame(
            history_df,
            columns=["epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy"],
        ).to_csv(model_log_dir / "training_history.csv", index=False)

    finite_loss_values = [
        float(value)
        for value in training_result["train_losses"] + training_result["val_losses"]
        if np.isfinite(value)
    ]
    plot_ylim_upper = max(finite_loss_values) * 1.1 if finite_loss_values else 1.0
    plot_ylim_upper = max(plot_ylim_upper, 0.1)
    run_with_captured_stdout(
        plot_and_save_training_curve,
        args.log_dir,
        args.model_name,
        training_result["train_losses"],
        training_result["val_losses"],
        args.num_epochs,
        args,
        plot_ylims=[0.0, plot_ylim_upper],
    )

    log_task_specific_model_performance(
        X_train_size=len(train_dataset),
        X_val_size=len(val_dataset),
        train_loss=float(training_result["train_losses"][-1]),
        val_loss=float(training_result["val_losses"][-1]),
        args=args,
        log_csv_path=str(Path(args.log_dir) / "model_performance.csv"),
    )

    reference_dir = output_root / "references"
    reference_dir.mkdir(parents=True, exist_ok=True)
    reference_file_path = reference_dir / (
        f"best_minigrid_task_model_ids_{args.task_name}_{datetime.now().strftime('%Y%m%d')}.json"
    )
    update_task_specific_model_reference_file(
        args,
        reference_file_path=str(reference_file_path),
        validation_ids=[example.metadata.sequence_id for example in val_examples],
    )

    _save_json(model_log_dir / "train_metrics.json", train_metrics)
    _save_json(model_log_dir / "val_metrics.json", val_metrics)
    _save_json(
        model_log_dir / "baseline_metrics.json",
        {
            "train_majority": baseline_train_metrics,
            "val_majority": baseline_val_metrics,
        },
    )
    _save_json(
        model_log_dir / "training_summary.json",
        {
            "model_name": args.model_name,
            "timestamp": datetime.now().isoformat(),
            "task_name": args.task_name,
            "task_label": args.task_label,
            "device": str(COMPUTE_DEVICE),
            "feature_dim": int(manifest.feature_dim),
            "action_vocab": ACTION_VOCAB_BY_TASK[args.task_name],
            "best_epoch": int(training_result["best_epoch"]),
            "best_val_accuracy": float(training_result["best_val_accuracy"]),
            "best_val_loss": float(training_result["best_val_loss"]),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "train_stats": train_stats,
            "val_stats": val_stats,
            "split_summary": split_payload["summary"],
        },
    )

    logging.info(
        "Training complete. best_epoch=%d best_val_accuracy=%.4f best_val_loss=%.4f",
        int(training_result["best_epoch"]),
        float(training_result["best_val_accuracy"]),
        float(training_result["best_val_loss"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
