#!/usr/bin/env python3
"""Train the PowerWash PLVM and its embedding-based baselines on frozen task encoders.

Predicts a player's navigation-style label on a later task from the Back Garden and Bungalow
traces truncated at each observation cutoff (one sample per player per cutoff). Runs are
written under ``models/powerwash/hierarchical/training_logs/<family>/`` in the artifact root.

``--model_type`` x ``--prediction_head_type`` -> model:
  - ``plvm_both_partial`` (default) x ``hierarchical`` (default): PLVM (PLVM-2 at the default ``--latent_dim 2``)
  - ``plvm_both_partial`` x ``logistic``: linear probe on the concatenated embeddings
  - ``plvm_both_partial`` x ``mlp``: two-task MLP
  - ``single_task`` (with ``--partial_task_name``) x ``mlp`` (default): single-task MLP
  - ``single_task`` x ``logistic``: single-task linear probe

``--attempt_policy first`` (default) uses only attempt 1 of each observed task.

Usage (PLVM-2, fold 00, 2-class Fire Station label; from the repository root):
    python powerwash/train/train_plvm_powerwash.py \\
        --player_label_csv powerwash/outputs/firestation_zone_labels/fold_00/firestation_zone_labels_2_class_clean.csv \\
        --player_split_manifest powerwash/outputs/firestation_zone_labels/fold_00/player_split_manifest.csv
"""

import argparse
import csv
import json
import logging
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from plvm import artifact_paths
from plvm.global_utils import set_seed
from plvm.model_training_utils import (
    COMPUTE_DEVICE,
    EmbeddingHierarchicalModel,
    EmbeddingMLPModel,
    _sample_train_indices_by_id,
    aggregate_classification_logits_by_id,
    aggregate_classification_targets_by_id,
    compute_classification_metrics,
    compute_naive_baseline_losses,
    generate_model_name,
    plot_loss_curves,
    save_session_specific_plots,
    train_embedding_hierarchical_model,
    update_hierarchical_model_performance_csv,
)
from powerwash.train.powerwash_plvm_data_utils import (
    ALL_PLAYER_LABEL_MODES,
    ATTEMPT_POLICIES,
    ATTEMPT_POLICY_FIRST,
    DUAL_TASK_MODEL_TYPES,
    MODEL_TYPE_BOTH_PARTIAL,
    TARGET_MODE_PARTIAL_TARGET,
    TARGET_MODE_PLAYER_LABEL,
    TARGET_MODE_PLAYER_LABEL_2_CLASS,
    TARGET_MODE_PLAYER_LABEL_2_CLASS_CLEAN_ENTROPY,
    TARGET_MODE_PLAYER_LABEL_2_CLASS_CLEAN_ZONEHOPPING,
    TARGET_MODE_PLAYER_LABEL_2_CLASS_ENTROPY,
    TARGET_MODE_PLAYER_LABEL_2_CLASS_ZONEHOPPING,
    TARGET_MODE_PLAYER_LABEL_3_CLASS,
    TARGET_MODE_PLAYER_LABEL_3_CLASS_ENTROPY,
    TARGET_MODE_PLAYER_LABEL_3_CLASS_ZONEHOPPING,
    build_powerwash_plvm_candidates,
    canonicalize_dual_task_configuration,
    load_player_label_lookup,
    load_task_model_spec,
    load_task_models_and_manifests,
    materialize_powerwash_plvm_candidates,
    resolve_powerwash_player_label_class_names,
    sample_plvm_candidates_by_player,
    split_plvm_candidates_by_player,
)
from powerwash.train.powerwash_hierarchical_log_utils import (
    FIRESTATION_SENSITIVITY_FAMILY_DIRS,
    resolve_powerwash_hierarchical_log_base,
)
from powerwash.train.powerwash_target_utils import (
    TARGET_CHOICES,
    is_binary_target,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIM_REDUCTION_METHOD_NONE = "none"
DIM_REDUCTION_SCALER_FILENAME = "embedding_dim_reduction_scaler.pkl"
DIM_REDUCTION_MODEL_FILENAME = "embedding_dim_reduction_model.pkl"
# Regression targets are never transformed; the value is still recorded in run outputs.
TARGET_TRANSFORM_NONE = "none"

EMBEDDING_STANDARDIZATION_SCALER_FILENAME = "embedding_standardization_scaler.pkl"
STORYBOOK_TASK_TOKEN = "storybook_house"
STORYBOOK_LOG_SUFFIX = "_storybook"
FIRESTATION_SENSITIVITY_LABEL_TOKEN = "firestation_zone_labels_sensitivity"
# Task pair used when --full_task_name / --partial_task_name are not given.
DEFAULT_FULL_TASK_NAME = "bungalow"
DEFAULT_PARTIAL_TASK_NAME = "back_garden"


# ---------------------------------------------------------------------------
# Dimensionality reduction helpers
# ---------------------------------------------------------------------------


def _parse_dim_reduction_method(method: str) -> dict:
    """Parse a dim-reduction method string into a structured config."""
    normalized = str(method or DIM_REDUCTION_METHOD_NONE).strip().lower()
    if not normalized or normalized == DIM_REDUCTION_METHOD_NONE:
        return {"type": "none", "method": DIM_REDUCTION_METHOD_NONE}

    m = re.fullmatch(r"pca_components_(\d+)", normalized)
    if m:
        n_components = int(m.group(1))
        if n_components < 1:
            raise ValueError("PCA component count must be >= 1.")
        return {"type": "pca", "method": normalized, "n_components": n_components}

    raise ValueError(
        "Unsupported --dim_reduction_method '{}'. Expected 'none' or 'pca_components_<N>'.".format(
            method
        )
    )


def _fit_and_apply_dim_reduction(
    train_emb: torch.Tensor,
    val_emb: torch.Tensor,
    method_config: dict,
    seed: int,
):
    """Fit train-only dimensionality reduction and apply to both splits."""
    pre_dim = int(train_emb.shape[1])
    metadata = {
        "dim_reduction_method": method_config["method"],
        "dim_reduction_type": method_config["type"],
        "uses_dim_reduction": method_config["type"] != "none",
        "pre_reduction_embedding_dim": pre_dim,
        "post_reduction_embedding_dim": pre_dim,
    }

    if method_config["type"] == "none":
        return train_emb, val_emb, None, metadata

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    train_dev = train_emb.device
    val_dev = val_emb.device
    train_np = train_emb.detach().cpu().numpy()
    val_np = val_emb.detach().cpu().numpy()

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_np)
    val_scaled = scaler.transform(val_np)

    requested_n = int(method_config["n_components"])
    effective_n = min(requested_n, train_scaled.shape[0], train_scaled.shape[1])
    if effective_n < 1:
        raise ValueError("PCA needs at least 1 train sample/feature.")

    reducer = PCA(n_components=effective_n, random_state=seed)
    train_reduced = reducer.fit_transform(train_scaled)
    val_reduced = reducer.transform(val_scaled)

    metadata.update(
        {
            "post_reduction_embedding_dim": int(train_reduced.shape[1]),
            "pca_requested_components": requested_n,
            "pca_n_components": int(effective_n),
            "pca_explained_variance_ratio": [
                float(x) for x in reducer.explained_variance_ratio_.tolist()
            ],
            "pca_explained_variance_total": float(np.sum(reducer.explained_variance_ratio_)),
        }
    )

    return (
        torch.tensor(train_reduced, dtype=torch.float32, device=train_dev),
        torch.tensor(val_reduced, dtype=torch.float32, device=val_dev),
        {"scaler": scaler, "model": reducer},
        metadata,
    )


def _save_dim_reduction_artifacts(log_dir: str, artifacts: Optional[dict]) -> None:
    if not artifacts:
        return
    for fname, key in [
        (DIM_REDUCTION_SCALER_FILENAME, "scaler"),
        (DIM_REDUCTION_MODEL_FILENAME, "model"),
    ]:
        path = os.path.join(log_dir, fname)
        with open(path, "wb") as handle:
            pickle.dump(artifacts[key], handle)
    logging.info(
        "Saved dim-reduction artifacts to %s",
        log_dir,
    )


def _fit_and_apply_embedding_standardization(
    train_emb: torch.Tensor,
    val_emb: torch.Tensor,
    method: str,
):
    """Fit train-only embedding standardization and apply to both splits.

    ``method`` is ``standardize`` (per-dimension z-score), ``center`` (mean only) or
    ``none``. Returns ``(train_emb, val_emb, scaler_or_None, metadata)``.
    """
    from sklearn.preprocessing import StandardScaler

    method = str(method or "none").strip().lower()
    if method not in {"standardize", "center", "none"}:
        raise ValueError(
            "Unsupported --embedding_standardization '{}'. "
            "Expected one of: standardize, center, none.".format(method)
        )

    metadata = {"embedding_standardization": method}

    if method == "none":
        return train_emb, val_emb, None, metadata

    train_dev = train_emb.device
    val_dev = val_emb.device
    train_np = train_emb.detach().cpu().numpy()
    val_np = val_emb.detach().cpu().numpy()

    with_std = method == "standardize"
    scaler = StandardScaler(with_std=with_std)
    train_scaled = scaler.fit_transform(train_np)
    val_scaled = scaler.transform(val_np)

    mean_norm = float(np.linalg.norm(scaler.mean_))
    if scaler.scale_ is not None:
        scale_min = float(np.min(scaler.scale_))
        scale_max = float(np.max(scaler.scale_))
    else:
        scale_min = scale_max = 1.0
    logging.info(
        "Embedding standardization [%s] — mean_norm=%.4f, per-dim scale range=[%.4f, %.4f]",
        method,
        mean_norm,
        scale_min,
        scale_max,
    )

    return (
        torch.tensor(train_scaled, dtype=torch.float32, device=train_dev),
        torch.tensor(val_scaled, dtype=torch.float32, device=val_dev),
        scaler,
        metadata,
    )


def _save_embedding_standardization_scaler(log_dir: str, scaler) -> None:
    if scaler is None:
        return
    path = os.path.join(log_dir, EMBEDDING_STANDARDIZATION_SCALER_FILENAME)
    with open(path, "wb") as handle:
        pickle.dump(scaler, handle)
    logging.info("Saved embedding standardization scaler to %s", path)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def _resolve_prediction_head_type(args: argparse.Namespace) -> str:
    """Return the effective prediction head for ``--prediction_head_type``/``--model_type``.

    ``auto`` selects ``mlp`` for single_task and ``hierarchical`` otherwise. A single-task
    run has no latent fusion, so ``hierarchical`` falls back to ``mlp``; ``logistic`` is kept
    (single-task linear probe).
    """
    requested = str(getattr(args, "prediction_head_type", "auto") or "auto").strip().lower()
    model_type = str(getattr(args, "model_type", "") or "").strip().lower()
    if requested == "auto":
        return "mlp" if model_type == "single_task" else "hierarchical"
    if requested not in {"hierarchical", "mlp", "logistic"}:
        raise ValueError(
            "Unsupported prediction_head_type={}. "
            "Expected one of auto, hierarchical, mlp, logistic.".format(requested)
        )
    if model_type == "single_task" and requested == "hierarchical":
        return "mlp"
    return requested


def _resolve_log_family(model_type: str, prediction_head_type: str) -> str:
    """Return the model-family token used to route a run's log directory.

    Logistic heads go to ``single_task_logistic_probe`` (single_task) or
    ``concat_logistic_probe`` (two-task); the two-task MLP head goes to
    ``concat_mlp_probe``; every other run uses its model type (``plvm_models`` /
    ``single_task``).
    """
    model_type = str(model_type or "").strip().lower()
    if prediction_head_type == "logistic":
        return (
            "single_task_logistic_probe" if model_type == "single_task" else "concat_logistic_probe"
        )
    if prediction_head_type == "mlp" and model_type == MODEL_TYPE_BOTH_PARTIAL:
        return "concat_mlp_probe"
    return model_type


def _should_use_storybook_log_suffix(player_label_csv: str) -> bool:
    """Return True when the player-label CSV clearly targets Storybook House."""
    return STORYBOOK_TASK_TOKEN in str(player_label_csv or "").strip().lower()


def _apply_storybook_log_suffix(base_log: Path, enabled: bool) -> Path:
    """Route Storybook PLVM runs into a sibling log-family folder with a suffix."""
    if not enabled:
        return base_log
    if base_log.name.endswith(STORYBOOK_LOG_SUFFIX):
        return base_log
    return base_log.parent / "{}{}".format(base_log.name, STORYBOOK_LOG_SUFFIX)


def _should_use_firestation_sensitivity_family(player_label_csv: str) -> bool:
    """Return True when the player-label CSV comes from Fire Station sensitivity labels."""
    return FIRESTATION_SENSITIVITY_LABEL_TOKEN in str(player_label_csv or "").strip().lower()


def _apply_firestation_sensitivity_family(
    base_log: Path,
    *,
    enabled: bool,
) -> Path:
    """Route Fire Station sensitivity runs into dedicated model-family folders."""
    if not enabled:
        return base_log
    sensitivity_family = FIRESTATION_SENSITIVITY_FAMILY_DIRS.get(base_log.name, "")
    if not sensitivity_family:
        return base_log
    return base_log.parent / sensitivity_family


def _build_hierarchical_model(
    embedding_dim: int,
    args: argparse.Namespace,
) -> torch.nn.Module:
    """Build the embedding predictor for PowerWash target prediction."""
    if args.prediction_head_type == "mlp":
        hidden_sizes = [args.encoder_hidden_size]
        hidden_sizes.extend([args.decoder_hidden_dim] * max(args.decoder_num_layers - 2, 0))
        model = EmbeddingMLPModel(
            embedding_dim=embedding_dim,
            num_parameters=1,
            dropout_rate=args.dropout_rate,
            positive_only=False,
            task_type=args.task_type,
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
            task_type=args.task_type,
            num_classes=args.num_classes,
        )
    return model.to(COMPUTE_DEVICE)


# ---------------------------------------------------------------------------
# Per-sample train subsampling
# ---------------------------------------------------------------------------


def _filter_outlier_targets(emb, y, candidates, lo, hi):
    """Remove rows where the target falls outside [lo, hi]."""
    mask = (y.squeeze(-1) >= lo) & (y.squeeze(-1) <= hi)
    idx = mask.nonzero(as_tuple=True)[0]
    return emb[idx], y[idx], [candidates[i] for i in idx.tolist()]


def _apply_target_outlier_filter(
    train_emb, train_y, train_candidates, val_emb, val_y, val_candidates, n_std=2.0
):
    """Filter train and val by mean ± n_std of the training targets."""
    mean = train_y.mean().item()
    std = train_y.std().item()
    lo, hi = mean - n_std * std, mean + n_std * std
    n_train_before = len(train_y)
    n_val_before = len(val_y)
    train_emb, train_y, train_candidates = _filter_outlier_targets(
        train_emb, train_y, train_candidates, lo, hi
    )
    val_emb, val_y, val_candidates = _filter_outlier_targets(val_emb, val_y, val_candidates, lo, hi)
    logging.info(
        "Outlier filter (mean=%.3f, std=%.3f, range=[%.3f, %.3f]): "
        "train %d→%d (removed %d), val %d→%d (removed %d)",
        mean,
        std,
        lo,
        hi,
        n_train_before,
        len(train_y),
        n_train_before - len(train_y),
        n_val_before,
        len(val_y),
        n_val_before - len(val_y),
    )
    return train_emb, train_y, train_candidates, val_emb, val_y, val_candidates


def _apply_fixed_train_subsampling(
    train_emb: torch.Tensor,
    train_y: torch.Tensor,
    train_ids,
    train_cutoffs,
    train_candidates,
    samples_per_player: int,
    seed: int,
):
    """Sample a fixed subset of train rows, at most N per player."""
    if samples_per_player <= 0:
        return train_emb, train_y, train_ids, train_cutoffs, train_candidates

    sampled_indices = sorted(
        _sample_train_indices_by_id(
            sample_ids=train_ids,
            samples_per_id=samples_per_player,
            sampling_seed=seed,
        )
    )
    idx_t = torch.tensor(sampled_indices, dtype=torch.long, device=train_emb.device)
    return (
        train_emb.index_select(0, idx_t),
        train_y.index_select(0, idx_t),
        [train_ids[i] for i in sampled_indices],
        [train_cutoffs[i] for i in sampled_indices],
        [train_candidates[i] for i in sampled_indices],
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _compute_regression_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """Return MSE, RMSE, MAE and optionally R² for scalar regression."""
    p = preds.detach().cpu().squeeze(-1).numpy()
    t = targets.detach().cpu().squeeze(-1).numpy()
    mse = float(np.mean((p - t) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(p - t)))
    metrics = {"mse": mse, "rmse": rmse, "mae": mae}
    ss_res = np.sum((p - t) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    if ss_tot > 0:
        metrics["r2"] = float(1.0 - ss_res / ss_tot)
    return metrics


def _parse_player_label_classes(raw_classes: str) -> list:
    """Parse a comma-separated player-label class list into canonical order."""
    labels = [part.strip() for part in str(raw_classes or "").split(",") if part.strip()]
    if len(labels) < 2:
        raise ValueError(
            "--player_label_classes must contain at least two comma-separated class names."
        )
    if len(set(labels)) != len(labels):
        raise ValueError("--player_label_classes contains duplicates: {}".format(labels))
    return labels


def _classification_class_names(args: argparse.Namespace) -> list:
    """Return the class-name list for any classification run."""
    class_names = getattr(args, "class_names", None)
    if class_names:
        return list(class_names)
    return ["Class {}".format(idx) for idx in range(int(getattr(args, "num_classes", 2)))]


def _compute_classification_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> dict:
    """Compute multiclass metrics and cross-entropy from classification logits."""
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
    cross_entropy = float(
        torch.nn.functional.cross_entropy(
            logits.detach().cpu(),
            target_tensor,
        ).item()
    )
    metrics["cross_entropy"] = cross_entropy
    return metrics


def _aggregate_classification_metrics_by_player(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_ids: list,
) -> tuple:
    """Average classification logits per player and return grouped tensors."""
    grouped_logits, grouped_targets, grouped_ids = aggregate_classification_logits_by_id(
        logits.detach().cpu(),
        targets.detach().cpu(),
        sample_ids,
    )
    return grouped_logits.detach().cpu(), grouped_targets.detach().cpu(), grouped_ids


def _compute_majority_baseline_classification(
    train_targets: torch.Tensor,
    eval_targets: torch.Tensor,
    num_classes: int,
) -> dict:
    """Evaluate a majority-class predictor on a classification target."""
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
    majority_class = int(np.bincount(train_np).argmax())
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
    """Evaluate a reproducible uniform-random multiclass predictor."""
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


def _target_display_name(args: argparse.Namespace) -> str:
    """Return the user-facing target label for logs and outputs."""
    target_mode = str(getattr(args, "target_mode", "") or "").strip().lower()
    if target_mode == TARGET_MODE_PARTIAL_TARGET:
        return str(getattr(args, "target_col", "") or "").strip()
    if target_mode in ALL_PLAYER_LABEL_MODES:
        if resolve_powerwash_player_label_class_names(target_mode) is not None:
            return target_mode
        return str(getattr(args, "player_label_col", "") or "player_label").strip()
    raise ValueError("Unsupported target_mode={!r}.".format(target_mode))


def _resolve_powerwash_prediction_metadata(args: argparse.Namespace):
    """Return canonical PowerWash task/target metadata for logging output."""
    model_type = str(getattr(args, "model_type", "") or "").strip().lower()
    full_task_name = str(getattr(args, "full_task_name", "") or "").strip().lower()
    partial_task_name = str(getattr(args, "partial_task_name", "") or "").strip().lower()
    if model_type == MODEL_TYPE_BOTH_PARTIAL and full_task_name and partial_task_name:
        predicted_task = ",".join(sorted([partial_task_name, full_task_name]))
    elif partial_task_name:
        predicted_task = partial_task_name
    else:
        requested_tasks = [
            task_name.strip().lower()
            for task_name in str(getattr(args, "task_names", "") or "").split(",")
            if task_name.strip()
        ]
        if not requested_tasks:
            raise ValueError(
                "Could not resolve PowerWash predicted_task: "
                "expected partial_task_name or task_names."
            )
        predicted_task = ",".join(requested_tasks)

    target_mode = str(getattr(args, "target_mode", "") or "").strip().lower()
    if target_mode == TARGET_MODE_PARTIAL_TARGET:
        target_col = str(getattr(args, "target_col", "") or "").strip().lower()
        if not target_col:
            raise ValueError(
                "Could not resolve PowerWash parameters_to_predict: target_col is empty."
            )
        parameters_to_predict = "partial_target_{}".format(target_col)
    elif target_mode in ALL_PLAYER_LABEL_MODES:
        player_label_col = str(getattr(args, "player_label_col", "") or "").strip().lower()
        if not player_label_col:
            raise ValueError(
                "Could not resolve PowerWash parameters_to_predict: player_label_col is empty."
            )
        parameters_to_predict = "{}_{}".format(target_mode, player_label_col)
    else:
        raise ValueError("Unsupported target_mode={!r}.".format(target_mode))
    return predicted_task, parameters_to_predict


def _save_val_predictions(
    log_dir: str,
    model_name: str,
    model_type: str,
    prediction_head_type: str,
    target_mode: str,
    target_name: str,
    target_type: str,
    val_preds: torch.Tensor,
    val_targets: torch.Tensor,
    val_candidates,
    class_names: Optional[list] = None,
) -> None:
    """Save per-sample validation predictions to a CSV."""
    class_names = list(class_names or [])
    if target_type == "classification":
        logits_np = val_preds.detach().cpu().numpy()
        exp_logits = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
        pred_prob_matrix = exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), 1e-7, None)
        targets_flat = val_targets.detach().cpu().view(-1).long().tolist()
        pred_probs = pred_prob_matrix.max(axis=1).tolist()
        pred_labels = np.argmax(logits_np, axis=1).tolist()
        target_labels = [
            class_names[int(target)] if class_names and 0 <= int(target) < len(class_names) else ""
            for target in targets_flat
        ]
        prediction_labels = [
            class_names[int(pred_label)]
            if class_names and 0 <= int(pred_label) < len(class_names)
            else str(pred_label)
            for pred_label in pred_labels
        ]
        prediction_probabilities_json = [
            json.dumps([float(prob) for prob in row.tolist()]) for row in pred_prob_matrix
        ]
        sq_errors = [float("nan")] * len(pred_labels)
    else:
        targets_flat = val_targets.detach().cpu().squeeze(-1).tolist()
        pred_probs = val_preds.detach().cpu().squeeze(-1).tolist()
        pred_labels = [""] * len(pred_probs)
        target_labels = [""] * len(pred_probs)
        prediction_labels = [""] * len(pred_probs)
        prediction_probabilities_json = [""] * len(pred_probs)
        sq_errors = [(pred - tgt) ** 2 for pred, tgt in zip(pred_probs, targets_flat)]

    out_path = os.path.join(log_dir, "val_predictions.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_mode",
                "target_name",
                "target_type",
                "player_id",
                "task_name",
                "attempt_id",
                "source_type",
                "source_row_index",
                "observation_cutoff_seconds",
                "target",
                "target_label",
                "prediction",
                "prediction_class_id",
                "prediction_label",
                "prediction_probabilities_json",
                "sq_error",
                "model_type",
                "prediction_head_type",
                "model_name",
            ],
        )
        writer.writeheader()
        for (
            candidate,
            pred,
            pred_label,
            pred_label_name,
            pred_probs_json,
            tgt,
            tgt_label,
            sq_error,
        ) in zip(
            val_candidates,
            pred_probs,
            pred_labels,
            prediction_labels,
            prediction_probabilities_json,
            targets_flat,
            target_labels,
            sq_errors,
        ):
            writer.writerow(
                {
                    "target_mode": target_mode,
                    "target_name": target_name,
                    "target_type": target_type,
                    "player_id": candidate.pid,
                    "task_name": candidate.task_name,
                    "attempt_id": candidate.attempt_number,
                    "source_type": candidate.source_type,
                    "source_row_index": candidate.source_row_index,
                    "observation_cutoff_seconds": candidate.cutoff_seconds,
                    "target": tgt,
                    "target_label": tgt_label,
                    "prediction": pred,
                    "prediction_class_id": pred_label if target_type == "classification" else "",
                    "prediction_label": pred_label_name
                    if target_type == "classification"
                    else pred_label,
                    "prediction_probabilities_json": pred_probs_json,
                    "sq_error": sq_error,
                    "model_type": model_type,
                    "prediction_head_type": prediction_head_type,
                    "model_name": model_name,
                }
            )
    logging.info("Saved per-sample val predictions to %s", out_path)


def _save_classification_metrics_by_cutoff(
    log_dir: str,
    model_name: str,
    split_name: str,
    split_logits: torch.Tensor,
    split_targets: torch.Tensor,
    split_candidates,
    class_names: Optional[list] = None,
) -> None:
    """Save split-specific classification metrics grouped by observation cutoff.

    ``candidate.cutoff_seconds`` carries the requested observation cutoff in
    elapsed seconds (shown in minutes when every cutoff is a whole minute). When
    all cutoffs are at most 60 they are treated as row counts instead
    (``cutoff_kind`` = ``row_count``).
    """
    class_names = list(class_names or [])
    split_name = str(split_name or "split").strip().lower()
    logits_cpu = split_logits.detach().cpu()
    targets_cpu = split_targets.detach().cpu().view(-1).long()
    if logits_cpu.numel() == 0 or targets_cpu.numel() == 0 or not split_candidates:
        return

    raw_cutoff_values = [float(candidate.cutoff_seconds) for candidate in split_candidates]
    use_row_count_axis = bool(raw_cutoff_values) and max(raw_cutoff_values) <= 60.0
    cutoff_kind = "row_count" if use_row_count_axis else "observation_cutoff_seconds"
    use_minutes_axis = not use_row_count_axis and all(
        abs((value / 60.0) - round(value / 60.0)) < 1e-6 for value in raw_cutoff_values
    )
    cutoff_label = (
        "Observation rows used"
        if use_row_count_axis
        else (
            "Observation cutoff (minutes)" if use_minutes_axis else "Observation cutoff (seconds)"
        )
    )
    cutoff_dir = os.path.join(log_dir, "plots", split_name, "cutoff_metrics")
    os.makedirs(cutoff_dir, exist_ok=True)

    grouped_indices = {}
    for index, candidate in enumerate(split_candidates):
        if use_row_count_axis:
            cutoff_value = int(candidate.cutoff_seconds)
        else:
            cutoff_value = float(candidate.cutoff_seconds)
        grouped_indices.setdefault(cutoff_value, []).append(index)

    rows = []
    sorted_cutoffs = sorted(grouped_indices)
    plot_cutoffs = (
        [float(cutoff_value) / 60.0 for cutoff_value in sorted_cutoffs]
        if use_minutes_axis
        else list(sorted_cutoffs)
    )
    accuracy_values = []
    support_values = []
    for cutoff_value in sorted_cutoffs:
        indices = grouped_indices[cutoff_value]
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
                "cutoff_kind": cutoff_kind,
                "cutoff_value": cutoff_value,
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
                "class_names_json": json.dumps(class_names),
            }
        )
        accuracy_values.append(float(metrics["accuracy"]))
        support_values.append(int(len(indices)))

    csv_path = os.path.join(cutoff_dir, "metrics_by_cutoff.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
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
    logging.info(
        "Saved %s classification metrics by cutoff to %s",
        split_name,
        csv_path,
    )
    try:
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        ax1.plot(plot_cutoffs, accuracy_values, marker="o", linewidth=2.0, color="#1f77b4")
        ax1.set_xlabel(cutoff_label)
        ax1.set_ylabel("{} accuracy".format(split_name.capitalize()), color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax1.set_ylim(0.0, 1.0)
        ax1.grid(True, axis="y", alpha=0.25)

        ax2 = ax1.twinx()
        ax2.bar(plot_cutoffs, support_values, alpha=0.18, color="#7f7f7f")
        ax2.set_ylabel("{} samples".format(split_name.capitalize()), color="#555555")
        ax2.tick_params(axis="y", labelcolor="#555555")

        fig.suptitle("{} Accuracy by Cutoff".format(split_name.capitalize()))
        ax1.set_title(model_name)
        fig.tight_layout()

        pdf_path = os.path.join(cutoff_dir, "accuracy_by_cutoff.pdf")
        fig.savefig(pdf_path)
        plt.close(fig)
        logging.info(
            "Saved %s accuracy-by-cutoff plot to %s",
            split_name,
            pdf_path,
        )
    except Exception as exc:
        logging.warning(
            "Could not save %s accuracy-by-cutoff plot: %s",
            split_name,
            exc,
        )


def _log_probability_tensor(probabilities: np.ndarray) -> torch.Tensor:
    """Return a float tensor of log-probabilities usable as classification logits."""
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-8, 1.0)
    return torch.tensor(np.log(clipped), dtype=torch.float32)


def _predict_logistic_probabilities(
    logistic_model: LogisticRegression,
    features: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Return class-probability columns aligned to ``range(num_classes)``."""
    raw_probabilities = logistic_model.predict_proba(features)
    probabilities = np.zeros((raw_probabilities.shape[0], int(num_classes)), dtype=np.float64)
    for source_column, class_id in enumerate(logistic_model.classes_):
        class_index = int(class_id)
        if 0 <= class_index < int(num_classes):
            probabilities[:, class_index] = raw_probabilities[:, source_column]
    row_sums = probabilities.sum(axis=1, keepdims=True)
    return probabilities / np.clip(row_sums, 1e-8, None)


def _run_concat_logistic_probe(
    args: argparse.Namespace,
    train_emb: torch.Tensor,
    val_emb: torch.Tensor,
    train_y: torch.Tensor,
    val_y: torch.Tensor,
    train_candidates,
    val_candidates,
    train_ids: list,
    val_ids: list,
    report_naive_train_loss: torch.Tensor,
    report_naive_val_loss: torch.Tensor,
    embedding_dim: int,
    embedding_standardization_metadata: dict,
    dim_reduction_metadata: dict,
    dim_reduction_artifacts: dict,
    embedding_standardization_scaler,
    aggregate_classification_by_player: bool,
) -> None:
    """Fit and save a linear logistic probe on frozen PowerWash embeddings.

    Used for both logistic heads: the two-task probe on concatenated embeddings and
    the single-task linear probe on one task's embedding.
    """
    if args.task_type != "classification":
        raise ValueError("prediction_head_type=logistic only supports classification targets.")

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

    train_metrics_sample_level = _compute_classification_metrics_from_logits(
        train_logits_cpu,
        train_y_cpu,
        num_classes=args.num_classes,
    )
    val_metrics_sample_level = _compute_classification_metrics_from_logits(
        val_logits_cpu,
        val_y_cpu,
        num_classes=args.num_classes,
    )
    train_metrics = train_metrics_sample_level
    val_metrics = val_metrics_sample_level
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
    if aggregate_classification_by_player:
        train_player_logits_cpu, train_player_targets_cpu, _ = (
            _aggregate_classification_metrics_by_player(
                train_logits_cpu,
                train_y_cpu,
                train_ids,
            )
        )
        val_player_logits_cpu, val_player_targets_cpu, _ = (
            _aggregate_classification_metrics_by_player(
                val_logits_cpu,
                val_y_cpu,
                val_ids,
            )
        )
        train_metrics = _compute_classification_metrics_from_logits(
            train_player_logits_cpu,
            train_player_targets_cpu,
            num_classes=args.num_classes,
        )
        val_metrics = _compute_classification_metrics_from_logits(
            val_player_logits_cpu,
            val_player_targets_cpu,
            num_classes=args.num_classes,
        )
        majority_baseline_metrics_train = _compute_majority_baseline_classification(
            train_player_targets_cpu,
            train_player_targets_cpu,
            num_classes=args.num_classes,
        )
        majority_baseline_metrics_val = _compute_majority_baseline_classification(
            train_player_targets_cpu,
            val_player_targets_cpu,
            num_classes=args.num_classes,
        )

    logging.info(
        "Concat logistic probe train metrics [%s]: accuracy=%.4f macro_f1=%.4f cross_entropy=%.4f support=%d",
        args.target_name,
        train_metrics["accuracy"],
        train_metrics["macro_f1"],
        train_metrics["cross_entropy"],
        train_metrics["support"],
    )
    logging.info(
        "Concat logistic probe val metrics [%s]: accuracy=%.4f macro_f1=%.4f cross_entropy=%.4f support=%d",
        args.target_name,
        val_metrics["accuracy"],
        val_metrics["macro_f1"],
        val_metrics["cross_entropy"],
        val_metrics["support"],
    )

    _save_val_predictions(
        log_dir=args.log_dir,
        model_name=args.model_name,
        model_type=args.model_type,
        prediction_head_type=args.prediction_head_type,
        target_mode=args.target_mode,
        target_name=args.target_name,
        target_type=args.target_type,
        val_preds=val_logits_cpu,
        val_targets=val_y_cpu,
        val_candidates=val_candidates,
        class_names=_classification_class_names(args),
    )
    _save_classification_metrics_by_cutoff(
        log_dir=args.log_dir,
        model_name=args.model_name,
        split_name="train",
        split_logits=train_logits_cpu,
        split_targets=train_y_cpu,
        split_candidates=train_candidates,
        class_names=_classification_class_names(args),
    )
    _save_classification_metrics_by_cutoff(
        log_dir=args.log_dir,
        model_name=args.model_name,
        split_name="val",
        split_logits=val_logits_cpu,
        split_targets=val_y_cpu,
        split_candidates=val_candidates,
        class_names=_classification_class_names(args),
    )

    model_save_path = os.path.join(args.log_dir, "{}_model.pkl".format(args.model_name))
    with open(model_save_path, "wb") as handle:
        pickle.dump(logistic_model, handle)
    logging.info("Saved concat logistic probe model to %s", model_save_path)

    _save_dim_reduction_artifacts(args.log_dir, dim_reduction_artifacts)
    _save_embedding_standardization_scaler(args.log_dir, embedding_standardization_scaler)

    update_hierarchical_model_performance_csv(
        csv_file=os.path.join(args.log_dir, "..", "model_performance.csv"),
        args=args,
        train_loss=train_metrics["cross_entropy"],
        val_loss=val_metrics["cross_entropy"],
        train_size=len(train_emb),
        val_size=len(val_emb),
        naive_train_loss=float(report_naive_train_loss.item()),
        naive_val_loss=float(report_naive_val_loss.item()),
        train_mse=float("nan"),
        val_mse=float("nan"),
        train_accuracy=train_metrics.get("accuracy", float("nan")),
        val_accuracy=val_metrics.get("accuracy", float("nan")),
        naive_train_accuracy=majority_baseline_metrics_train["metrics"].get(
            "accuracy",
            float("nan"),
        ),
        naive_val_accuracy=majority_baseline_metrics_val["metrics"].get(
            "accuracy",
            float("nan"),
        ),
        heldout_session_ids="",
    )

    hparams = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, list))}
    hparams["embedding_dim"] = embedding_dim
    hparams["logistic_model_class"] = "sklearn.linear_model.LogisticRegression"
    hparams.update(embedding_standardization_metadata)
    hparams.update(dim_reduction_metadata)
    hparams["class_names"] = _classification_class_names(args)
    hparams["classification_metric_unit"] = (
        "player" if aggregate_classification_by_player else "sample"
    )
    hparam_path = os.path.join(args.log_dir, "{}_hyperparameters.json".format(args.model_name))
    with open(hparam_path, "w", encoding="utf-8") as handle:
        json.dump(hparams, handle, indent=2, sort_keys=True)
    logging.info("Saved concat logistic probe hyperparameters to %s", hparam_path)


def _save_regression_plots(
    args: argparse.Namespace,
    trained_model,
    train_embeddings: torch.Tensor,
    val_embeddings: torch.Tensor,
    train_targets: torch.Tensor,
    val_targets: torch.Tensor,
    train_ids,
    val_ids,
) -> None:
    """Save regression diagnostic plots using the shared helper."""
    parameter_labels = [_target_display_name(args)]
    visualize_latent = bool(
        getattr(trained_model, "produces_latent", True) and getattr(args, "latent_dim", 4) < 3
    )
    plot_model = trained_model

    def _prep(t):
        return t.unsqueeze(1) if t.dim() == 1 else t

    for emb, tgt, ids, folder in [
        (train_embeddings, _prep(train_targets), train_ids, "train"),
        (val_embeddings, _prep(val_targets), val_ids, "val"),
    ]:
        try:
            save_session_specific_plots(
                plot_model,
                plot_model,
                emb,
                tgt,
                args=args,
                folder=folder,
                id_mapping=ids,
                visualize_latent_space=visualize_latent,
                parameter_labels=parameter_labels,
                color_by_parameters=None,
                color_parameter_labels=None,
                model_name=args.model_name,
            )
        except Exception as exc:
            logging.warning("Could not save %s plots: %s", folder, exc)


def _save_classification_plots(
    args: argparse.Namespace,
    trained_model,
    train_embeddings: torch.Tensor,
    val_embeddings: torch.Tensor,
    train_targets: torch.Tensor,
    val_targets: torch.Tensor,
    train_ids,
    val_ids,
) -> None:
    """Save classification diagnostics using the shared plotting helper."""
    visualize_latent = bool(
        getattr(trained_model, "produces_latent", True) and getattr(args, "latent_dim", 4) < 3
    )

    for emb, tgt, ids, folder in [
        (train_embeddings, train_targets, train_ids, "train"),
        (val_embeddings, val_targets, val_ids, "val"),
    ]:
        try:
            save_session_specific_plots(
                trained_model,
                trained_model,
                emb,
                tgt,
                args=args,
                folder=folder,
                id_mapping=ids,
                visualize_latent_space=visualize_latent,
                parameter_labels=None,
                color_by_parameters=None,
                color_parameter_labels=None,
                model_name=args.model_name,
            )
        except Exception as exc:
            logging.warning("Could not save %s classification plots: %s", folder, exc)


def _save_candidate_split_manifest(
    log_dir: str,
    args: argparse.Namespace,
    train_candidates,
    val_candidates,
) -> None:
    """Persist the exact train/validation candidate rows used by this run."""
    payload = {
        "model_name": args.model_name,
        "model_type": args.model_type,
        "model_family": getattr(args, "model_family", ""),
        "target_mode": args.target_mode,
        "target_name": args.target_name,
        "target_col": args.target_col,
        "target_transform": args.target_transform,
        "player_label_csv": getattr(args, "player_label_csv", ""),
        "player_label_col": getattr(args, "player_label_col", ""),
        "player_label_classes": getattr(args, "class_names", []),
        "partial_task_name": args.partial_task_name,
        "full_task_name": args.full_task_name,
        "both_partial_task_names": (
            [args.partial_task_name, args.full_task_name]
            if args.model_type == MODEL_TYPE_BOTH_PARTIAL
            else []
        ),
        "seed": int(args.seed),
        "val_fraction": float(args.val_fraction),
        "train_size": int(len(train_candidates)),
        "val_size": int(len(val_candidates)),
        "train": [candidate.to_dict() for candidate in train_candidates],
        "val": [candidate.to_dict() for candidate in val_candidates],
    }
    manifest_path = os.path.join(log_dir, "candidate_splits.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logging.info("Saved candidate split manifest to %s", manifest_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Parse the CLI, build the samples and embeddings, train the head, and save the run."""
    parser = argparse.ArgumentParser(
        description="Train the PowerWash PLVM and its embedding-based baselines on frozen "
        "task encoders. The defaults train PLVM-2 on the 2-class clean Fire Station label "
        "from Back Garden and Bungalow; see the module docstring for an example command."
    )

    # ---- Data ----
    # Relative --data_dir, --task_model_spec, --player_label_csv and
    # --player_split_manifest paths are resolved against the artifact root.
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(
            artifact_paths.PLAYER_SEQUENCE_FEATURES_DIR.relative_to(artifact_paths.ARTIFACT_ROOT)
        ),
        help="Root directory of processed per-attempt sequence CSVs, absolute or relative "
        "to the artifact root (default: %(default)s).",
    )
    parser.add_argument(
        "--task_model_spec",
        type=str,
        default=str(
            artifact_paths.POWERWASH_TASK_SPECIFIC_SPEC_JSON.relative_to(
                artifact_paths.ARTIFACT_ROOT
            )
        ),
        help="JSON file mapping task names to the frozen task-encoder runs, absolute or "
        "relative to the artifact root (default: %(default)s).",
    )
    parser.add_argument(
        "--target_mode",
        type=str,
        default=TARGET_MODE_PLAYER_LABEL_2_CLASS_CLEAN_ZONEHOPPING,
        choices=[
            TARGET_MODE_PARTIAL_TARGET,
            TARGET_MODE_PLAYER_LABEL,
            TARGET_MODE_PLAYER_LABEL_2_CLASS,
            TARGET_MODE_PLAYER_LABEL_2_CLASS_CLEAN_ZONEHOPPING,
            TARGET_MODE_PLAYER_LABEL_2_CLASS_ZONEHOPPING,
            TARGET_MODE_PLAYER_LABEL_2_CLASS_CLEAN_ENTROPY,
            TARGET_MODE_PLAYER_LABEL_2_CLASS_ENTROPY,
            TARGET_MODE_PLAYER_LABEL_3_CLASS,
            TARGET_MODE_PLAYER_LABEL_3_CLASS_ZONEHOPPING,
            TARGET_MODE_PLAYER_LABEL_3_CLASS_ENTROPY,
        ],
        help=(
            "Prediction target: a player-level label from --player_label_csv (default: "
            "%(default)s; use player_label_3_class_zonehopping for 3-class labels) or a "
            "row-level partial-task target (partial_target, with --target_col). "
            "All player_label_2_class* variants use a fixed 2-class scheme "
            "(Zone Planner / Zone Hopper); all player_label_3_class* variants use "
            "a fixed 3-class scheme (Zone Planner / Mixed / Zone Hopper); player_label "
            "takes its classes from --player_label_classes."
        ),
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="",
        help="PowerWash partial-task target column to predict when target_mode=partial_target.",
    )
    parser.add_argument(
        "--player_label_csv",
        type=str,
        default="",
        help="CSV of player-level labels; required for the player_label* target modes.",
    )
    parser.add_argument(
        "--player_id_col",
        type=str,
        default="player_id",
        help="Player-id column in --player_label_csv.",
    )
    parser.add_argument(
        "--player_label_col",
        type=str,
        default="label",
        help="Label column in --player_label_csv.",
    )
    parser.add_argument(
        "--player_label_classes",
        type=str,
        default="",
        help=(
            "Comma-separated class names in canonical order for target_mode=player_label, "
            "for example 'Zone Planner,Mixed,Zone Hopper'."
        ),
    )
    parser.add_argument(
        "--full_task_name",
        type=str,
        default=None,
        help="Second task of two-task runs (default: {} for two-task model types; not "
        "allowed for single_task).".format(DEFAULT_FULL_TASK_NAME),
    )
    parser.add_argument(
        "--partial_task_name",
        type=str,
        default=None,
        help="Partial (truncated) task; the only task for single_task runs (default: {}, "
        "unless a single_task run gives --task_names).".format(DEFAULT_PARTIAL_TASK_NAME),
    )
    parser.add_argument(
        "--task_names",
        type=str,
        default="",
        help="single_task runs only: comma-separated tasks whose sequences are pooled as "
        "the partial task when --partial_task_name is not given.",
    )

    # ---- Model type ----
    parser.add_argument(
        "--model_type",
        type=str,
        default=MODEL_TYPE_BOTH_PARTIAL,
        choices=["plvm_full_partial", MODEL_TYPE_BOTH_PARTIAL, "single_task"],
        help="plvm_both_partial (default): both tasks truncated at the same elapsed-time cutoff. "
        "plvm_full_partial: full attempt of --full_task_name + truncated --partial_task_name. "
        "single_task: --partial_task_name only.",
    )
    parser.add_argument(
        "--prediction_head_type",
        type=str,
        default="auto",
        choices=["auto", "hierarchical", "mlp", "logistic"],
        help=(
            "Prediction head. auto selects hierarchical for PLVM and mlp for "
            "single_task; logistic fits a linear probe on the frozen embeddings "
            "(concatenated for two-task runs: concat_logistic_probe; one task for "
            "single_task: single_task_logistic_probe)."
        ),
    )

    # ---- Sequence / embedding ----
    parser.add_argument(
        "--seq_length",
        type=int,
        default=120,
        help="Padded sequence length of the task encoders, recorded in the run metadata and "
        "the embedding-cache key (default: %(default)s). Each encoder is padded to the "
        "length given in the task-model spec.",
    )
    parser.add_argument(
        "--task_specific_hidden_size",
        type=int,
        default=32,
        help="Hidden size of the task encoders, recorded in the run metadata (default: "
        "%(default)s). The size used for the embeddings is read from each loaded encoder.",
    )
    parser.add_argument(
        "--embeddings_to_use",
        type=str,
        default="masked_average_all",
        choices=[
            "last_only",
            "average_all",
            "masked_average_all",
            "average_last_10",
            "average_last_20",
            "average_last_30",
            "first_only",
            "first_last_concat",
            "last_only_plus_count",
        ],
        help="Embedding extraction strategy.",
    )
    parser.add_argument(
        "--fusion_type",
        type=str,
        default="concat",
        choices=["concat", "avg"],
        help="Multi-task embedding fusion strategy.",
    )
    parser.add_argument(
        "--pool_kernel_size", type=int, default=1, help="Temporal pooling kernel (1 = no pooling)."
    )
    parser.add_argument(
        "--pool_stride", type=int, default=1, help="Temporal pooling stride (1 = no pooling)."
    )
    parser.add_argument(
        "--dim_reduction_method",
        type=str,
        default=DIM_REDUCTION_METHOD_NONE,
        help="Embedding dimensionality reduction. Supported: none, pca_components_<N>.",
    )
    parser.add_argument(
        "--embedding_standardization",
        type=str,
        default="standardize",
        choices=["standardize", "center", "none"],
        help=(
            "Fit-on-train embedding standardization applied after cache load and before "
            "the PLVM head. standardize: z-score per dimension (default); "
            "center: mean-center only; none: no transform."
        ),
    )

    # ---- Hierarchical model ----
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=2,
        help="PLVM latent size (default: %(default)s, i.e. PLVM-2).",
    )
    parser.add_argument(
        "--encoder_hidden_size",
        type=int,
        default=16,
        help="Fuser hidden size (default: %(default)s).",
    )
    parser.add_argument(
        "--decoder_hidden_dim",
        type=int,
        default=16,
        help="Decoder hidden size (default: %(default)s).",
    )
    parser.add_argument(
        "--decoder_num_layers", type=int, default=2, help="Decoder layers (default: %(default)s)."
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.2, help="Dropout (default: %(default)s)."
    )

    # ---- Training ----
    parser.add_argument("--num_epochs", type=int, default=50, help="Epochs (default: %(default)s).")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size (default: %(default)s)."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="Learning rate (default: %(default)s)."
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=1.0,
        help="Gradient-norm clip (default: %(default)s).",
    )
    parser.add_argument(
        "--lambda_l2", type=float, default=0.05, help="L2 penalty weight (default: %(default)s)."
    )
    parser.add_argument(
        "--max_files_per_task",
        type=int,
        default=2500,
        help="Cap on unique pid/task/attempt sequence files used per partial task before "
        "embedding computation (default: %(default)s; 0 disables the cap).",
    )
    parser.add_argument(
        "--attempt_policy",
        choices=list(ATTEMPT_POLICIES),
        default=ATTEMPT_POLICY_FIRST,
        help="Observed-task attempts each player contributes: 'first' (default) keeps only "
        "attempt 1 of every task, so players without a usable attempt 1 are excluded; "
        "'all' uses every attempt file (repeat attempts become extra samples).",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default=None,
        choices=["mse", "classification"],
        help="classification for classification targets (always used for player labels); "
        "mse for regression targets. Default: chosen from the target.",
    )
    parser.add_argument(
        "--use_early_stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop training once the validation loss has not improved for 10%% of "
        "--num_epochs (default: on, for every model type and head). "
        "--no-use_early_stopping trains for the full --num_epochs.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fraction of players reserved for validation when no --player_split_manifest "
        "is given (default: %(default)s).",
    )
    parser.add_argument(
        "--train_samples_per_player_per_epoch",
        type=int,
        default=0,
        help="If >0, sample this many train rows per player per epoch.",
    )
    parser.add_argument(
        "--train_fixed_samples_per_player",
        type=int,
        default=0,
        help="If >0, sample this many train rows per player once for the whole run.",
    )
    parser.add_argument(
        "--max_samples_per_player_per_split",
        type=int,
        default=0,
        help="If >0, cap each split to at most this many sampled candidates per player before embedding computation.",
    )
    parser.add_argument(
        "--require_players_with_all_tasks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For player-label runs, restrict the candidate pool to players with at least "
            "one usable processed sequence in every task from the task-model spec "
            "(default: on). This keeps single-task and dual-task runs on the same player "
            "pool and therefore the same player-level train/validation split for a fixed seed."
        ),
    )
    parser.add_argument(
        "--aggregate_classification_by_player",
        dest="aggregate_classification_by_player",
        action="store_true",
        default=None,
        help=(
            "Aggregate all cutoff samples from the same player into one "
            "training/evaluation unit by averaging logits per player. "
            "Defaults to enabled for player_label target modes."
        ),
    )
    parser.add_argument(
        "--no_aggregate_classification_by_player",
        dest="aggregate_classification_by_player",
        action="store_false",
        help=(
            "Disable player-level aggregation and keep classification "
            "training/evaluation at the row/sample level."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: %(default)s).")

    # ---- Caching ----
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=str(artifact_paths.POWERWASH_EMBEDDING_CACHE_DIR),
        help="Directory for caching precomputed embeddings (default: "
        "models/powerwash/hierarchical/embedding_cache under the artifact root). "
        "Pass an empty string to disable.",
    )

    # ---- Misc ----
    parser.add_argument(
        "--run_mode",
        type=str,
        default="run",
        choices=["debug", "run"],
        help="run (default): full training, runs go to the training_logs tree; debug: at "
        "most 10 epochs, runs go to the debug_logs tree.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help=(
            "Base directory for model outputs. If set to the shared PowerWash "
            "training_logs/debug_logs root, runs are routed into the appropriate "
            "model-family subfolder automatically."
        ),
    )
    parser.add_argument(
        "--player_split_manifest",
        type=str,
        default="",
        help="Path to a player_split_manifest.csv produced by K-fold CV labeling. "
        "Columns: player_id, split (train/val), fold. "
        "When provided the manifest's train/val assignment is used directly "
        "and --val_fraction is ignored.",
    )

    args = parser.parse_args()
    args.target_transform = TARGET_TRANSFORM_NONE
    set_seed(args.seed)

    args.player_split_manifest = args.player_split_manifest.strip()

    # ---- Validate ----
    if not args.player_split_manifest and not 0 < args.val_fraction < 1:
        raise ValueError("--val_fraction must be between 0 and 1.")
    if args.target_col and args.target_col not in TARGET_CHOICES:
        raise ValueError("--target_col must be one of {}.".format(", ".join(TARGET_CHOICES)))
    if args.max_files_per_task < 0:
        raise ValueError("--max_files_per_task must be >= 0.")
    if args.max_files_per_task == 0:
        args.max_files_per_task = None
    if args.max_samples_per_player_per_split < 0:
        raise ValueError("--max_samples_per_player_per_split must be >= 0.")
    args.task_names = args.task_names.strip()
    # Fill in the default task pair when the names are not given explicitly.
    if args.full_task_name is None:
        args.full_task_name = (
            DEFAULT_FULL_TASK_NAME if args.model_type in DUAL_TASK_MODEL_TYPES else ""
        )
    if args.partial_task_name is None:
        uses_task_names = args.model_type not in DUAL_TASK_MODEL_TYPES and args.task_names
        args.partial_task_name = "" if uses_task_names else DEFAULT_PARTIAL_TASK_NAME
    args.full_task_name = args.full_task_name.strip()
    args.partial_task_name = args.partial_task_name.strip()
    args.target_col = args.target_col.strip()
    args.player_label_csv = args.player_label_csv.strip()
    args.player_id_col = args.player_id_col.strip()
    args.player_label_col = args.player_label_col.strip()
    args.player_label_classes = args.player_label_classes.strip()
    args.full_task_name, args.partial_task_name = canonicalize_dual_task_configuration(
        model_type=args.model_type,
        full_task_name=args.full_task_name,
        partial_task_name=args.partial_task_name,
    )
    if args.train_fixed_samples_per_player > 0 and args.train_samples_per_player_per_epoch > 0:
        raise ValueError(
            "--train_fixed_samples_per_player and "
            "--train_samples_per_player_per_epoch are mutually exclusive."
        )
    if args.max_samples_per_player_per_split > 0 and (
        args.train_fixed_samples_per_player > 0 or args.train_samples_per_player_per_epoch > 0
    ):
        raise ValueError(
            "--max_samples_per_player_per_split cannot be combined with the train-only "
            "subsampling flags --train_fixed_samples_per_player and "
            "--train_samples_per_player_per_epoch."
        )
    if args.target_mode == TARGET_MODE_PARTIAL_TARGET and not args.target_col:
        raise ValueError("--target_col is required when --target_mode=partial_target.")
    if args.target_mode in ALL_PLAYER_LABEL_MODES:
        if not args.player_label_csv:
            raise ValueError(
                "--player_label_csv is required when --target_mode is a player_label variant."
            )
        if not args.player_label_col:
            raise ValueError(
                "--player_label_col is required when --target_mode is a player_label variant."
            )
        if not args.player_id_col:
            raise ValueError(
                "--player_id_col is required when --target_mode is a player_label variant."
            )
    if args.model_type in DUAL_TASK_MODEL_TYPES:
        if not args.full_task_name or not args.partial_task_name:
            raise ValueError(
                "{} requires both --full_task_name and --partial_task_name.".format(args.model_type)
            )
        if args.full_task_name == args.partial_task_name:
            raise ValueError(
                "--full_task_name and --partial_task_name must differ for {}.".format(
                    args.model_type
                )
            )
    else:
        if args.full_task_name:
            raise ValueError(
                "--full_task_name is only valid for dual-task model types: {}.".format(
                    sorted(DUAL_TASK_MODEL_TYPES)
                )
            )
        if not args.partial_task_name and not args.task_names:
            raise ValueError(
                "{} requires --partial_task_name (or --task_names).".format(args.model_type)
            )

    fixed_class_names = resolve_powerwash_player_label_class_names(args.target_mode)
    if fixed_class_names is not None:
        args.class_names = fixed_class_names
    elif args.target_mode == TARGET_MODE_PLAYER_LABEL:
        args.class_names = _parse_player_label_classes(args.player_label_classes)
    else:
        args.class_names = []

    dim_reduction_config = _parse_dim_reduction_method(args.dim_reduction_method)
    args.dim_reduction_method = dim_reduction_config["method"]
    args.prediction_head_type = _resolve_prediction_head_type(args)

    # ---- Resolve log directory ----
    base_log = Path(
        resolve_powerwash_hierarchical_log_base(
            Path(args.log_dir) if args.log_dir else None,
            _resolve_log_family(args.model_type, args.prediction_head_type),
            run_mode=args.run_mode,
        )
    )
    base_log = _apply_storybook_log_suffix(
        base_log,
        _should_use_storybook_log_suffix(args.player_label_csv),
    )
    base_log = _apply_firestation_sensitivity_family(
        base_log,
        enabled=_should_use_firestation_sensitivity_family(args.player_label_csv),
    )
    args.model_name = generate_model_name()
    if args.run_mode == "debug":
        args.num_epochs = min(args.num_epochs, 10)
        args.model_name = "debug-{}".format(args.model_name)

    model_log_dir = os.path.join(str(base_log), args.model_name)
    os.makedirs(model_log_dir, exist_ok=True)
    args.log_dir = model_log_dir

    # ---- Logging ----
    log_file = os.path.join(model_log_dir, "training.logs")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("=== PowerWash PLVM target prediction ===")
    logging.info("Training configuration:")
    for key, value in sorted(vars(args).items()):
        logging.info("  %s: %s", key, value)

    # Fields required by shared utilities (update_hierarchical_model_performance_csv etc.)
    args.target_name = _target_display_name(args)
    is_classification_target = args.target_mode in ALL_PLAYER_LABEL_MODES or (
        args.target_mode == TARGET_MODE_PARTIAL_TARGET and is_binary_target(args.target_col)
    )
    args.target_type = "classification" if is_classification_target else "regression"
    args.task_type = args.target_type
    args.num_classes = (
        len(args.class_names)
        if args.target_mode in ALL_PLAYER_LABEL_MODES
        else (2 if args.task_type == "classification" else 1)
    )
    if args.aggregate_classification_by_player is None:
        args.aggregate_classification_by_player = bool(args.target_mode in ALL_PLAYER_LABEL_MODES)
    if args.task_type == "classification" or args.loss_type is None:
        args.loss_type = "classification" if args.task_type == "classification" else "mse"
    args.encoder_dropout_rate = args.dropout_rate
    args.decoder_dropout_rate = args.dropout_rate
    if args.model_type == MODEL_TYPE_BOTH_PARTIAL:
        args.tasks = ["task_a", "task_b"]
    elif args.model_type in DUAL_TASK_MODEL_TYPES:
        args.tasks = ["full", "partial"]
    else:
        args.tasks = ["partial"]
    args.predicted_task, args.parameters_to_predict = _resolve_powerwash_prediction_metadata(args)
    args.predicted_next_task = args.predicted_task
    _model_family_map = {
        "plvm_full_partial": "PLVMFullPartial",
        MODEL_TYPE_BOTH_PARTIAL: "PLVMBothPartial",
    }
    args.model_family = _model_family_map.get(args.model_type, "SingleTask")
    args.assumed_distribution_type = "normal"
    args.max_kld_loss = 10000.0
    args.pdlvm_log_prob_loss_weight = 0.5
    args.diversity_loss_weight = 0.0
    args.diversity_loss_type = "distribution"
    args.latent_vector_penalty_loss_weight = 0.001
    args.contrastive_loss_weight = 0.0
    args.contrastive_temperature = 1.0
    args.contrastive_similarity_threshold = 0.8
    args.use_task_specific_heads = False
    args.task_output_dim_map = {}
    args.multitask_head_mode = "shared"
    args.task_loss_weight_mode = "none"
    args.task_loss_weights = None
    args.latent_fusion_type = "none"
    args.add_noise = False
    args.log_data = False
    args.max_trials_per_task = -1
    if args.target_mode == TARGET_MODE_PARTIAL_TARGET:
        horizon_match = re.search(r"_(\d+)s$", args.target_col)
        args.prediction_horizon = int(horizon_match.group(1)) if horizon_match else 1
    else:
        args.prediction_horizon = 1
    args.input_embeddings_source = "all_tasks"
    args.fold_number = -1

    # ---- Task filter from --task_names (single_task runs without --partial_task_name) ----
    task_names_filter = None
    if not args.partial_task_name and args.task_names:
        task_names_filter = [t.strip() for t in args.task_names.split(",") if t.strip()] or None

    # ------------------------------------------------------------------ #
    # 1. Load task-specific models                                        #
    # ------------------------------------------------------------------ #
    spec_path = Path(args.task_model_spec)
    if not spec_path.is_absolute():
        spec_path = artifact_paths.ARTIFACT_ROOT / spec_path

    logging.info("Loading task model spec from %s", spec_path)
    task_model_spec = load_task_model_spec(spec_path, artifact_paths.ARTIFACT_ROOT)
    task_models = load_task_models_and_manifests(task_model_spec)
    logging.info(
        "Loaded models for tasks: %s",
        sorted(task_models.keys()),
    )

    # ------------------------------------------------------------------ #
    # 2. Load data and compute embeddings                                 #
    # ------------------------------------------------------------------ #
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = artifact_paths.ARTIFACT_ROOT / data_dir
    player_label_lookup = None
    if args.target_mode in ALL_PLAYER_LABEL_MODES:
        player_label_csv = Path(args.player_label_csv)
        if not player_label_csv.is_absolute():
            player_label_csv = artifact_paths.ARTIFACT_ROOT / player_label_csv
        args.player_label_csv = str(player_label_csv)
        player_label_lookup = load_player_label_lookup(
            label_csv_path=player_label_csv,
            player_id_col=args.player_id_col,
            label_col=args.player_label_col,
            allowed_labels=args.class_names,
        )
        logging.info(
            "Player-label classification: %d labeled players across classes=%s",
            len(player_label_lookup),
            args.class_names,
        )

    logging.info(
        "Target mode=%s | target=%s | model_type=%s | embeddings=%s",
        args.target_mode,
        args.target_name,
        args.model_type,
        args.embeddings_to_use,
    )

    all_candidates = build_powerwash_plvm_candidates(
        data_dir=data_dir,
        task_models=task_models,
        target_mode=args.target_mode,
        target_col=args.target_col,
        player_label_lookup=player_label_lookup,
        model_type=args.model_type,
        full_task_name=args.full_task_name,
        partial_task_name=args.partial_task_name,
        max_files_per_task=args.max_files_per_task,
        sampling_seed=args.seed,
        task_names=task_names_filter,
        require_players_with_all_tasks=args.require_players_with_all_tasks,
        attempt_policy=args.attempt_policy,
    )
    if args.player_split_manifest:
        manifest_path = Path(args.player_split_manifest)
        if not manifest_path.is_absolute():
            manifest_path = artifact_paths.ARTIFACT_ROOT / manifest_path
        manifest_df = pd.read_csv(manifest_path)
        train_pids = set(manifest_df.loc[manifest_df["split"] == "train", "player_id"].astype(str))
        val_pids = set(manifest_df.loc[manifest_df["split"] == "val", "player_id"].astype(str))
        train_candidates = [c for c in all_candidates if str(c.pid) in train_pids]
        val_candidates = [c for c in all_candidates if str(c.pid) in val_pids]
        # Extract fold number from the manifest's parent directory name (e.g. fold_03 → 3).
        _fold_match = re.search(r"fold_(\d+)", manifest_path.parent.name)
        if _fold_match:
            args.fold_number = int(_fold_match.group(1))
        logging.info(
            "Player split from manifest %s | fold=%s | train=%d (%d players) val=%d (%d players)",
            manifest_path,
            args.fold_number,
            len(train_candidates),
            len(train_pids),
            len(val_candidates),
            len(val_pids),
        )
    else:
        train_candidates, val_candidates, _unused_test_candidates = split_plvm_candidates_by_player(
            all_candidates,
            val_fraction=args.val_fraction,
            test_fraction=0.0,
            seed=args.seed,
        )
    if args.max_samples_per_player_per_split > 0:
        train_candidates = sample_plvm_candidates_by_player(
            train_candidates,
            max_samples_per_player=args.max_samples_per_player_per_split,
            seed=args.seed,
        )
        val_candidates = sample_plvm_candidates_by_player(
            val_candidates,
            max_samples_per_player=args.max_samples_per_player_per_split,
            seed=args.seed + 1,
        )

    train_emb, train_y, train_candidates = materialize_powerwash_plvm_candidates(
        candidates=train_candidates,
        data_dir=data_dir,
        task_models=task_models,
        seq_length=args.seq_length,
        model_type=args.model_type,
        embeddings_to_use=args.embeddings_to_use,
        fusion_type=args.fusion_type,
        pool_kernel_size=args.pool_kernel_size,
        pool_stride=args.pool_stride,
        full_task_name=args.full_task_name,
        partial_task_name=args.partial_task_name,
        cache_dir=args.cache_dir or None,
        spec_path=str(spec_path),
        target_mode=args.target_mode,
        target_col=args.target_col,
        max_files_per_task=args.max_files_per_task,
        sampling_seed=args.seed,
        task_names=task_names_filter,
        split_name="train",
        max_samples_per_player_per_split=args.max_samples_per_player_per_split,
        attempt_policy=args.attempt_policy,
    )
    val_emb, val_y, val_candidates = materialize_powerwash_plvm_candidates(
        candidates=val_candidates,
        data_dir=data_dir,
        task_models=task_models,
        seq_length=args.seq_length,
        model_type=args.model_type,
        embeddings_to_use=args.embeddings_to_use,
        fusion_type=args.fusion_type,
        pool_kernel_size=args.pool_kernel_size,
        pool_stride=args.pool_stride,
        full_task_name=args.full_task_name,
        partial_task_name=args.partial_task_name,
        cache_dir=args.cache_dir or None,
        spec_path=str(spec_path),
        target_mode=args.target_mode,
        target_col=args.target_col,
        max_files_per_task=args.max_files_per_task,
        sampling_seed=args.seed,
        task_names=task_names_filter,
        split_name="val",
        max_samples_per_player_per_split=args.max_samples_per_player_per_split,
        attempt_policy=args.attempt_policy,
    )

    if args.target_mode == TARGET_MODE_PARTIAL_TARGET and args.task_type != "classification":
        train_emb, train_y, train_candidates, val_emb, val_y, val_candidates = (
            _apply_target_outlier_filter(
                train_emb,
                train_y,
                train_candidates,
                val_emb,
                val_y,
                val_candidates,
            )
        )

    train_emb = train_emb.to(COMPUTE_DEVICE)
    val_emb = val_emb.to(COMPUTE_DEVICE)
    train_y = train_y.to(COMPUTE_DEVICE)
    val_y = val_y.to(COMPUTE_DEVICE)
    if args.task_type == "classification":
        train_y = train_y.long()
        val_y = val_y.long()
    train_ids = [candidate.pid for candidate in train_candidates]
    val_ids = [candidate.pid for candidate in val_candidates]
    train_cutoffs = [candidate.cutoff_seconds for candidate in train_candidates]

    logging.info(
        "Embedding dim=%d | Train=%d (%d players) | Val=%d (%d players)",
        train_emb.shape[1],
        len(train_emb),
        len(set(train_ids)),
        len(val_emb),
        len(set(val_ids)),
    )
    logging.info("Validation players: %s", sorted(set(val_ids)))

    # ---- Fixed train subsampling ----
    train_emb, train_y, train_ids, train_cutoffs, train_candidates = _apply_fixed_train_subsampling(
        train_emb=train_emb,
        train_y=train_y,
        train_ids=train_ids,
        train_cutoffs=train_cutoffs,
        train_candidates=train_candidates,
        samples_per_player=args.train_fixed_samples_per_player,
        seed=args.seed,
    )
    if args.train_fixed_samples_per_player > 0:
        logging.info(
            "Fixed train subsampling: %d rows across %d players.",
            len(train_emb),
            len(set(train_ids)),
        )

    # Keep untouched copies of the materialized targets for metrics and plots.
    raw_train_y = train_y.clone()
    raw_val_y = val_y.clone()

    _save_candidate_split_manifest(
        log_dir=args.log_dir,
        args=args,
        train_candidates=train_candidates,
        val_candidates=val_candidates,
    )

    # ---- Embedding standardization (fit on train only) ----
    train_emb, val_emb, embedding_standardization_scaler, embedding_standardization_metadata = (
        _fit_and_apply_embedding_standardization(train_emb, val_emb, args.embedding_standardization)
    )

    # ---- Dimensionality reduction ----
    train_emb, val_emb, dim_reduction_artifacts, dim_reduction_metadata = (
        _fit_and_apply_dim_reduction(train_emb, val_emb, dim_reduction_config, args.seed)
    )

    logging.info(
        "Dim reduction [%s] — pre_dim=%d | post_dim=%d",
        dim_reduction_metadata["dim_reduction_method"],
        dim_reduction_metadata["pre_reduction_embedding_dim"],
        dim_reduction_metadata["post_reduction_embedding_dim"],
    )

    # Dummy item-level tensors required by train_embedding_hierarchical_model.
    n_tasks = len(args.tasks)
    train_item_level = torch.zeros(len(train_emb), n_tasks, 50, device=COMPUTE_DEVICE)
    val_item_level = torch.zeros(len(val_emb), n_tasks, 50, device=COMPUTE_DEVICE)
    aggregate_classification_by_player = bool(
        args.task_type == "classification" and args.aggregate_classification_by_player
    )

    # ---- Naive baseline ----
    baseline_metric = "classification" if args.task_type == "classification" else "mse"
    optimization_baseline_train_y = train_y
    optimization_baseline_val_y = val_y
    optimization_baseline_train_item_level = train_item_level
    optimization_baseline_val_item_level = val_item_level
    aggregated_train_player_ids = sorted(set(train_ids))
    aggregated_val_player_ids = sorted(set(val_ids))
    if aggregate_classification_by_player:
        optimization_baseline_train_y, aggregated_train_player_ids = (
            aggregate_classification_targets_by_id(train_y, train_ids)
        )
        optimization_baseline_val_y, aggregated_val_player_ids = (
            aggregate_classification_targets_by_id(val_y, val_ids)
        )
        optimization_baseline_train_item_level = torch.zeros(
            len(optimization_baseline_train_y),
            n_tasks,
            50,
            device=COMPUTE_DEVICE,
        )
        optimization_baseline_val_item_level = torch.zeros(
            len(optimization_baseline_val_y),
            n_tasks,
            50,
            device=COMPUTE_DEVICE,
        )
    optimization_naive_train_loss, optimization_naive_val_loss, _ = compute_naive_baseline_losses(
        optimization_baseline_train_y,
        optimization_baseline_val_y,
        optimization_baseline_train_item_level,
        optimization_baseline_val_item_level,
        metric=baseline_metric,
        args=args,
    )
    report_naive_train_loss = optimization_naive_train_loss
    report_naive_val_loss = optimization_naive_val_loss
    logging.info(
        "Naive baseline loss — train: %.4f | val: %.4f",
        float(optimization_naive_train_loss.item()),
        float(optimization_naive_val_loss.item()),
    )

    if args.prediction_head_type == "logistic":
        _run_concat_logistic_probe(
            args=args,
            train_emb=train_emb,
            val_emb=val_emb,
            train_y=train_y,
            val_y=val_y,
            train_candidates=train_candidates,
            val_candidates=val_candidates,
            train_ids=train_ids,
            val_ids=val_ids,
            report_naive_train_loss=report_naive_train_loss,
            report_naive_val_loss=report_naive_val_loss,
            embedding_dim=int(train_emb.shape[1]),
            embedding_standardization_metadata=embedding_standardization_metadata,
            dim_reduction_metadata=dim_reduction_metadata,
            dim_reduction_artifacts=dim_reduction_artifacts,
            embedding_standardization_scaler=embedding_standardization_scaler,
            aggregate_classification_by_player=aggregate_classification_by_player,
        )
        return

    # ------------------------------------------------------------------ #
    # 4. Build hierarchical model                                         #
    # ------------------------------------------------------------------ #
    embedding_dim = int(train_emb.shape[1])
    hierarchical_model = _build_hierarchical_model(embedding_dim, args)

    # ------------------------------------------------------------------ #
    # 5. Train                                                            #
    # ------------------------------------------------------------------ #
    patience = int(args.num_epochs * 0.1) if args.use_early_stopping else args.num_epochs
    eval_interval = 2

    trained_model, train_losses, validation_losses = train_embedding_hierarchical_model(
        model=hierarchical_model,
        train_embeddings=train_emb,
        train_y=train_y,
        train_item_level_data=train_item_level,
        val_embeddings=val_emb,
        val_y=val_y,
        val_item_level_data=val_item_level,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=patience,
        args=args,
        eval_interval=eval_interval,
        train_sample_ids=train_ids,
        val_sample_ids=val_ids,
        train_samples_per_id_per_epoch=args.train_samples_per_player_per_epoch,
        sampling_seed=args.seed,
    )

    # Save the loss curve.
    try:
        plot_loss_curves(
            train_losses,
            validation_losses,
            training_contrastive_loss=[],
            num_epochs=args.num_epochs,
            args=args,
            eval_interval=eval_interval,
            y_lim=(
                [0, float(optimization_naive_val_loss.item()) * 4 + 1e-6]
                if args.task_type != "classification"
                else None
            ),
        )
    except Exception as exc:
        logging.warning("Could not save loss curve: %s", exc)

    trained_model.eval()

    # ------------------------------------------------------------------ #
    # 6. Evaluate and save                                                #
    # ------------------------------------------------------------------ #
    optimization_train_loss_value = float(
        getattr(
            trained_model,
            "last_epoch_train_loss",
            train_losses[-1] if train_losses else float("nan"),
        )
    )
    optimization_val_loss_value = float(
        getattr(
            trained_model,
            "last_epoch_val_loss",
            validation_losses[-1] if validation_losses else float("nan"),
        )
    )

    with torch.no_grad():
        _, train_preds = trained_model(train_emb)
        _, val_preds = trained_model(val_emb)

    train_preds_cpu = train_preds.detach().cpu()
    val_preds_cpu = val_preds.detach().cpu()
    train_y_cpu = train_y.detach().cpu()
    val_y_cpu = val_y.detach().cpu()
    raw_train_y_cpu = raw_train_y.detach().cpu()
    raw_val_y_cpu = raw_val_y.detach().cpu()

    if args.task_type == "classification":
        train_metrics_sample_level = _compute_classification_metrics_from_logits(
            train_preds_cpu,
            train_y_cpu,
            num_classes=args.num_classes,
        )
        val_metrics_sample_level = _compute_classification_metrics_from_logits(
            val_preds_cpu,
            val_y_cpu,
            num_classes=args.num_classes,
        )
        sample_majority_baseline_metrics_train = _compute_majority_baseline_classification(
            train_y_cpu,
            train_y_cpu,
            num_classes=args.num_classes,
        )
        sample_majority_baseline_metrics_val = _compute_majority_baseline_classification(
            train_y_cpu,
            val_y_cpu,
            num_classes=args.num_classes,
        )
        sample_random_baseline_metrics_train = _compute_random_baseline_classification(
            train_y_cpu,
            num_classes=args.num_classes,
            seed=args.seed,
        )
        sample_random_baseline_metrics_val = _compute_random_baseline_classification(
            val_y_cpu,
            num_classes=args.num_classes,
            seed=args.seed + 1,
        )
        train_metrics = train_metrics_sample_level
        val_metrics = val_metrics_sample_level
        majority_baseline_metrics_train = sample_majority_baseline_metrics_train
        majority_baseline_metrics_val = sample_majority_baseline_metrics_val
        random_baseline_metrics_train = sample_random_baseline_metrics_train
        random_baseline_metrics_val = sample_random_baseline_metrics_val
        metric_unit_label = "sample"
        train_metric_player_ids = None
        val_metric_player_ids = None

        logging.info(
            "Train sample-level metrics [%s]: accuracy=%.4f macro_f1=%.4f cross_entropy=%.4f support=%d",
            args.target_name,
            train_metrics_sample_level["accuracy"],
            train_metrics_sample_level["macro_f1"],
            train_metrics_sample_level["cross_entropy"],
            train_metrics_sample_level["support"],
        )
        logging.info(
            "Val   sample-level metrics [%s]: accuracy=%.4f macro_f1=%.4f cross_entropy=%.4f support=%d",
            args.target_name,
            val_metrics_sample_level["accuracy"],
            val_metrics_sample_level["macro_f1"],
            val_metrics_sample_level["cross_entropy"],
            val_metrics_sample_level["support"],
        )

        if aggregate_classification_by_player:
            train_player_logits_cpu, train_player_targets_cpu, train_metric_player_ids = (
                _aggregate_classification_metrics_by_player(
                    train_preds_cpu,
                    train_y_cpu,
                    train_ids,
                )
            )
            val_player_logits_cpu, val_player_targets_cpu, val_metric_player_ids = (
                _aggregate_classification_metrics_by_player(
                    val_preds_cpu,
                    val_y_cpu,
                    val_ids,
                )
            )
            train_metrics = _compute_classification_metrics_from_logits(
                train_player_logits_cpu,
                train_player_targets_cpu,
                num_classes=args.num_classes,
            )
            val_metrics = _compute_classification_metrics_from_logits(
                val_player_logits_cpu,
                val_player_targets_cpu,
                num_classes=args.num_classes,
            )
            majority_baseline_metrics_train = _compute_majority_baseline_classification(
                train_player_targets_cpu,
                train_player_targets_cpu,
                num_classes=args.num_classes,
            )
            majority_baseline_metrics_val = _compute_majority_baseline_classification(
                train_player_targets_cpu,
                val_player_targets_cpu,
                num_classes=args.num_classes,
            )
            random_baseline_metrics_train = _compute_random_baseline_classification(
                train_player_targets_cpu,
                num_classes=args.num_classes,
                seed=args.seed,
            )
            random_baseline_metrics_val = _compute_random_baseline_classification(
                val_player_targets_cpu,
                num_classes=args.num_classes,
                seed=args.seed + 1,
            )
            metric_unit_label = "player"
            logging.info(
                "Train player-level metrics [%s]: accuracy=%.4f macro_f1=%.4f cross_entropy=%.4f support=%d",
                args.target_name,
                train_metrics["accuracy"],
                train_metrics["macro_f1"],
                train_metrics["cross_entropy"],
                train_metrics["support"],
            )
            logging.info(
                "Val   player-level metrics [%s]: accuracy=%.4f macro_f1=%.4f cross_entropy=%.4f support=%d",
                args.target_name,
                val_metrics["accuracy"],
                val_metrics["macro_f1"],
                val_metrics["cross_entropy"],
                val_metrics["support"],
            )
            logging.info(
                "Player-level aggregation active: %d train players | %d val players",
                len(train_metric_player_ids),
                len(val_metric_player_ids),
            )

        logging.info(
            "%s-level train majority baseline metrics: %s",
            metric_unit_label.title(),
            majority_baseline_metrics_train,
        )
        logging.info(
            "%s-level val majority baseline metrics: %s",
            metric_unit_label.title(),
            majority_baseline_metrics_val,
        )
        logging.info(
            "%s-level train random baseline metrics: %s",
            metric_unit_label.title(),
            random_baseline_metrics_train,
        )
        logging.info(
            "%s-level val random baseline metrics: %s",
            metric_unit_label.title(),
            random_baseline_metrics_val,
        )
    else:
        train_metrics = _compute_regression_metrics(train_preds_cpu, raw_train_y_cpu)
        val_metrics = _compute_regression_metrics(val_preds_cpu, raw_val_y_cpu)
        logging.info(
            "Train metrics [%s | original scale]: MSE=%.4f RMSE=%.4f MAE=%.4f%s",
            args.target_name,
            train_metrics["mse"],
            train_metrics["rmse"],
            train_metrics["mae"],
            " R²={:.4f}".format(train_metrics["r2"]) if "r2" in train_metrics else "",
        )
        logging.info(
            "Val   metrics [%s | original scale]: MSE=%.4f RMSE=%.4f MAE=%.4f%s",
            args.target_name,
            val_metrics["mse"],
            val_metrics["rmse"],
            val_metrics["mae"],
            " R²={:.4f}".format(val_metrics["r2"]) if "r2" in val_metrics else "",
        )
        train_loss_value = train_metrics["mse"]
        val_loss_value = val_metrics["mse"]
    if args.task_type == "classification":
        train_loss_value = optimization_train_loss_value
        val_loss_value = optimization_val_loss_value

    best_validation_loss = float(
        getattr(
            trained_model,
            "best_validation_loss",
            min(validation_losses) if validation_losses else float("nan"),
        )
    )
    best_validation_epoch = int(getattr(trained_model, "best_validation_epoch", 0))
    logging.info(
        "Best val checkpoint — epoch: %d | optimization loss: %.4f",
        best_validation_epoch,
        best_validation_loss,
    )

    # Save per-sample validation predictions.
    _save_val_predictions(
        log_dir=args.log_dir,
        model_name=args.model_name,
        model_type=args.model_type,
        prediction_head_type=args.prediction_head_type,
        target_mode=args.target_mode,
        target_name=args.target_name,
        target_type=args.target_type,
        val_preds=(val_preds_cpu),
        val_targets=(raw_val_y_cpu if args.task_type != "classification" else val_y_cpu),
        val_candidates=val_candidates,
        class_names=_classification_class_names(args),
    )
    if args.task_type == "classification":
        _save_classification_metrics_by_cutoff(
            log_dir=args.log_dir,
            model_name=args.model_name,
            split_name="train",
            split_logits=train_preds_cpu,
            split_targets=train_y_cpu,
            split_candidates=train_candidates,
            class_names=_classification_class_names(args),
        )
        _save_classification_metrics_by_cutoff(
            log_dir=args.log_dir,
            model_name=args.model_name,
            split_name="val",
            split_logits=val_preds_cpu,
            split_targets=val_y_cpu,
            split_candidates=val_candidates,
            class_names=_classification_class_names(args),
        )

    # ---- Plots ----
    if args.task_type == "classification":
        try:
            _save_classification_plots(
                args=args,
                trained_model=trained_model,
                train_embeddings=train_emb,
                val_embeddings=val_emb,
                train_targets=train_y,
                val_targets=val_y,
                train_ids=train_ids,
                val_ids=val_ids,
            )
        except Exception as exc:
            logging.warning("Could not save classification plots: %s", exc)
    else:
        try:
            _save_regression_plots(
                args=args,
                trained_model=trained_model,
                train_embeddings=train_emb,
                val_embeddings=val_emb,
                train_targets=raw_train_y,
                val_targets=raw_val_y,
                train_ids=train_ids,
                val_ids=val_ids,
            )
        except Exception as exc:
            logging.warning("Could not save regression plots: %s", exc)

    # Save model weights.
    model_save_path = os.path.join(args.log_dir, "{}_model.pth".format(args.model_name))
    torch.save(trained_model.state_dict(), model_save_path)
    logging.info("Saved model to %s", model_save_path)

    _save_dim_reduction_artifacts(args.log_dir, dim_reduction_artifacts)
    _save_embedding_standardization_scaler(args.log_dir, embedding_standardization_scaler)

    # Log to shared performance CSV.
    update_hierarchical_model_performance_csv(
        csv_file=os.path.join(args.log_dir, "..", "model_performance.csv"),
        args=args,
        train_loss=train_loss_value,
        val_loss=val_loss_value,
        train_size=len(train_emb),
        val_size=len(val_emb),
        naive_train_loss=float(report_naive_train_loss.item()),
        naive_val_loss=float(report_naive_val_loss.item()),
        train_mse=train_metrics.get("mse", float("nan")),
        val_mse=val_metrics.get("mse", float("nan")),
        train_accuracy=train_metrics.get("accuracy", float("nan")),
        val_accuracy=val_metrics.get("accuracy", float("nan")),
        naive_train_accuracy=(
            majority_baseline_metrics_train["metrics"].get("accuracy", float("nan"))
            if args.task_type == "classification"
            else float("nan")
        ),
        naive_val_accuracy=(
            majority_baseline_metrics_val["metrics"].get("accuracy", float("nan"))
            if args.task_type == "classification"
            else float("nan")
        ),
        heldout_session_ids="",
    )

    # Save hyperparameters JSON.
    hparams = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, list))}
    hparams["embedding_dim"] = embedding_dim
    hparams.update(embedding_standardization_metadata)
    hparams.update(dim_reduction_metadata)
    hparams["n_train_samples"] = len(train_emb)
    hparams["n_val_samples"] = len(val_emb)
    hparams["last_epoch_train_loss"] = optimization_train_loss_value
    hparams["last_epoch_val_loss"] = optimization_val_loss_value
    hparams["reported_train_loss"] = train_loss_value
    hparams["reported_val_loss"] = val_loss_value
    hparams["train_metrics"] = train_metrics
    hparams["val_metrics"] = val_metrics
    if args.task_type == "classification":
        hparams["class_names"] = _classification_class_names(args)
        hparams["train_majority_baseline_metrics"] = majority_baseline_metrics_train
        hparams["val_majority_baseline_metrics"] = majority_baseline_metrics_val
        hparams["train_random_baseline_metrics"] = random_baseline_metrics_train
        hparams["val_random_baseline_metrics"] = random_baseline_metrics_val
        hparams["classification_metric_unit"] = (
            "player" if aggregate_classification_by_player else "sample"
        )
        hparams["n_train_players"] = len(aggregated_train_player_ids)
        hparams["n_val_players"] = len(aggregated_val_player_ids)
        if aggregate_classification_by_player:
            hparams["train_metrics_player_level"] = train_metrics
            hparams["val_metrics_player_level"] = val_metrics
            hparams["train_metrics_sample_level"] = train_metrics_sample_level
            hparams["val_metrics_sample_level"] = val_metrics_sample_level
            hparams["train_majority_baseline_metrics_sample_level"] = (
                sample_majority_baseline_metrics_train
            )
            hparams["val_majority_baseline_metrics_sample_level"] = (
                sample_majority_baseline_metrics_val
            )
            hparams["train_random_baseline_metrics_sample_level"] = (
                sample_random_baseline_metrics_train
            )
            hparams["val_random_baseline_metrics_sample_level"] = sample_random_baseline_metrics_val
    hparams["naive_train_loss"] = float(report_naive_train_loss.item())
    hparams["naive_val_loss"] = float(report_naive_val_loss.item())
    hparams["validation_player_ids"] = sorted(set(val_ids))
    hparams["best_validation_loss"] = best_validation_loss
    hparams["best_validation_epoch"] = best_validation_epoch
    hparams["target_mode"] = args.target_mode
    hparams["target_col"] = args.target_col
    hparams["target_type"] = args.target_type
    hparams["target_name"] = args.target_name
    hparams["max_samples_per_player_per_split"] = args.max_samples_per_player_per_split

    hparams_path = os.path.join(args.log_dir, "{}_hyperparameters.json".format(args.model_name))
    with open(hparams_path, "w") as fp:
        json.dump(hparams, fp, indent=2, default=str)
    logging.info("Saved hyperparameters to %s", hparams_path)
    logging.info("=== PowerWash PLVM training complete ===")


if __name__ == "__main__":
    main()
