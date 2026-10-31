#!/usr/bin/env python3
"""Generates the Table 9 matched-head-gain column: two-task MLP minus the best single-task MLP.

Settings: the complementary, redundant and weak-shared-structure simulations at 200
steps, and PowerWash Fire Station (2-class) at 120 min. Simulation runs are the best
by validation accuracy at the 200-step cutoff, the two-task MLP coming from the
``*_concat_mlp_probe`` root; a setting without such runs reads "not computed". PowerWash uses the best run per fold from the main ``single_task``
and ``concat_mlp_probe`` families only (Storybook and label-sensitivity families
share the model types and are excluded), compared on folds all three conditions share.

``build_simulation_matched_head_values`` is reused by ``table09_simulation_complementarity.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper.figures.fig06_fig08_fig09_fig10_powerwash_accuracy_curves import (  # noqa: E402
    TRAINING_LOGS_DIR as POWERWASH_TRAINING_LOGS_DIR,
    build_fold_long_frame,
)
from paper.output_paths import resolve_tables_dir  # noqa: E402
from paper.table_output_utils import write_latex_table  # noqa: E402
from plvm import artifact_paths  # noqa: E402
from powerwash.train.powerwash_hierarchical_log_utils import (  # noqa: E402
    load_powerwash_hierarchical_performance_rows,
)


DEFAULT_MINIGRID_ROOT = artifact_paths.SIMULATION_MODELS_DIR
DEFAULT_POWERWASH_TARGET_MODE = "player_label_2_class_clean_zonehopping"
DEFAULT_OUTPUT_DIR = resolve_tables_dir()
DEFAULT_OUTPUT_STEM = "cross_domain_combination_gain_analysis"

MINIGRID_CUTOFF_STEPS = 200
POWERWASH_CUTOFF_MINUTES = 120
POWERWASH_SEQ_LENGTH = 120
MINIGRID_SEQ_LENGTH = 200
MINIGRID_MAX_TRAIN_VAL_PLAYERS = 8000
MINIGRID_RUN_MODE = "run"

# Main Fire Station families holding the PowerWash matched-head runs (single-task MLP and
# two-task concat MLP). Loading every family would also pick Storybook House / label-sensitivity
# runs with the same model type and target name.
POWERWASH_MATCHED_HEAD_MODEL_TYPES = ("single_task", "concat_mlp_probe")
NOT_COMPUTED = "not computed"

MINIGRID_SINGLE_FORAGING = "foraging"
MINIGRID_SINGLE_CLEANING = "cleaning"
MINIGRID_BOTH = "cleaning,foraging"

POWERWASH_TASK_A = "back_garden"
POWERWASH_TASK_B = "bungalow"
POWERWASH_BOTH = "back_garden,bungalow"


@dataclass(frozen=True)
class SimulationSetting:
    """Filesystem metadata for one simulation relationship condition."""

    key: str
    display_label: str
    single_root: Path
    concat_root: Path


@dataclass(frozen=True)
class SimulationRunRecord:
    """Minimal metadata needed to score one MiniGrid MLP run."""

    setting_key: str
    setting_label: str
    model_name: str
    model_type: str
    predicted_next_task: str
    val_accuracy: float
    train_accuracy: float
    seq_length: int
    max_train_val_players: int
    run_mode: str
    logs_dir: Path
    all_arguments_json: Dict[str, object]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--complementary_single_root",
        type=Path,
        default=DEFAULT_MINIGRID_ROOT / "plvm_label",
        help="MiniGrid complementary single-task MLP root.",
    )
    parser.add_argument(
        "--complementary_concat_root",
        type=Path,
        default=DEFAULT_MINIGRID_ROOT / "plvm_label_concat_mlp_probe",
        help="MiniGrid complementary concatenated MLP root.",
    )
    parser.add_argument(
        "--redundant_single_root",
        type=Path,
        default=DEFAULT_MINIGRID_ROOT / "plvm_label_redundant",
        help="MiniGrid redundant single-task MLP root.",
    )
    parser.add_argument(
        "--redundant_concat_root",
        type=Path,
        default=DEFAULT_MINIGRID_ROOT / "plvm_label_redundant_concat_mlp_probe",
        help="MiniGrid redundant concatenated MLP root.",
    )
    parser.add_argument(
        "--weak_structure_single_root",
        type=Path,
        default=DEFAULT_MINIGRID_ROOT / "plvm_label_weak_structure",
        help="MiniGrid weak-shared-structure single-task MLP root.",
    )
    parser.add_argument(
        "--weak_structure_concat_root",
        type=Path,
        default=DEFAULT_MINIGRID_ROOT / "plvm_label_weak_structure_concat_mlp_probe",
        help=(
            "MiniGrid weak-shared-structure concatenated MLP root. When it has no "
            "model_performance.csv, the matched-head gain is reported as not computed."
        ),
    )
    parser.add_argument(
        "--powerwash_model_performance_csv",
        type=Path,
        default=POWERWASH_TRAINING_LOGS_DIR,
        help="PowerWash hierarchical training_logs root or performance CSV source.",
    )
    parser.add_argument(
        "--powerwash_training_logs_dir",
        type=Path,
        default=POWERWASH_TRAINING_LOGS_DIR,
        help="PowerWash training_logs directory containing per-run cutoff metrics.",
    )
    parser.add_argument(
        "--powerwash_target_mode",
        type=str,
        default=DEFAULT_POWERWASH_TARGET_MODE,
        help="PowerWash target mode used for the comparison.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the CSV, markdown, and TeX outputs will be written.",
    )
    return parser.parse_args(argv)


def _safe_float(value: object, default: float = float("nan")) -> float:
    """Convert one scalar to float with a fallback."""
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: object, default: int = -1) -> int:
    """Convert one scalar to int with a fallback."""
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return int(float(value))
    except Exception:
        return default


def _parse_json_dict(raw_payload: object) -> Dict[str, object]:
    """Parse one JSON payload into a dictionary."""
    text = str(raw_payload or "").strip()
    if not text or text.lower() == "nan":
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize_task_scope(raw_value: object) -> str:
    """Normalize task names to a stable comma-separated token."""
    tokens = [token.strip().lower() for token in str(raw_value or "").split(",") if token.strip()]
    if not tokens:
        return ""
    unique_tokens = sorted(dict.fromkeys(tokens))
    return ",".join(unique_tokens)


def _format_percent(value: float, *, decimals: int = 1) -> str:
    """Format one scalar accuracy as a percent string."""
    return "{:.{decimals}f}".format(float(value) * 100.0, decimals=decimals)


def _format_mean_sd(mean_value: float, sd_value: float, *, decimals: int = 1) -> str:
    """Format one mean and standard deviation as a percentage summary."""
    return "{} +/- {}".format(
        _format_percent(mean_value, decimals=decimals),
        _format_percent(sd_value, decimals=decimals),
    )


def _format_gain(value: float, *, decimals: int = 1) -> str:
    """Format one signed percentage-point gain."""
    return "{:+.{decimals}f}".format(float(value) * 100.0, decimals=decimals)


def _format_pvalue(value: float) -> str:
    """Format one p-value for compact table display."""
    if not np.isfinite(float(value)):
        return ""
    return "p={:.3f}".format(float(value))


def _resolve_simulation_settings(args: argparse.Namespace) -> List[SimulationSetting]:
    """Return the fixed simulation settings to analyze."""
    return [
        SimulationSetting(
            key="complementary",
            display_label="Complementary simulation",
            single_root=Path(args.complementary_single_root).expanduser().resolve(),
            concat_root=Path(args.complementary_concat_root).expanduser().resolve(),
        ),
        SimulationSetting(
            key="redundant",
            display_label="Redundant simulation",
            single_root=Path(args.redundant_single_root).expanduser().resolve(),
            concat_root=Path(args.redundant_concat_root).expanduser().resolve(),
        ),
        SimulationSetting(
            key="weak_structure",
            display_label="Weak shared structure simulation",
            single_root=Path(args.weak_structure_single_root).expanduser().resolve(),
            concat_root=Path(args.weak_structure_concat_root).expanduser().resolve(),
        ),
    ]


def _load_simulation_rows_for_root(
    root: Path,
    *,
    setting_key: str,
    setting_label: str,
) -> List[SimulationRunRecord]:
    """Load one MiniGrid MLP family root into run records."""
    csv_path = Path(root).expanduser().resolve() / "training_logs" / "model_performance.csv"
    if not csv_path.exists():
        raise FileNotFoundError("Missing MiniGrid MLP performance CSV: {}".format(csv_path))

    frame = pd.read_csv(csv_path)
    rows: List[SimulationRunRecord] = []
    for _, row in frame.iterrows():
        payload = _parse_json_dict(row.get("all_arguments_json"))
        model_type = str(row.get("model_type") or payload.get("model_type") or "").strip().lower()
        predicted_next_task = _normalize_task_scope(
            row.get("predicted_next_task") or payload.get("predicted_next_task")
        )
        if model_type not in {"single_task", "plvm_both_partial"}:
            continue
        if predicted_next_task not in {
            MINIGRID_SINGLE_FORAGING,
            MINIGRID_SINGLE_CLEANING,
            MINIGRID_BOTH,
        }:
            continue
        if str(
            row.get("parameters_to_predict") or payload.get("parameters_to_predict") or ""
        ).strip() != ("minigrid_player_label"):
            continue
        if str(
            row.get("prediction_head_type") or payload.get("prediction_head_type") or ""
        ).strip().lower() != ("mlp"):
            continue

        rows.append(
            SimulationRunRecord(
                setting_key=setting_key,
                setting_label=setting_label,
                model_name=str(row.get("model_name") or "").strip(),
                model_type=model_type,
                predicted_next_task=predicted_next_task,
                val_accuracy=_safe_float(row.get("val_accuracy")),
                train_accuracy=_safe_float(row.get("train_accuracy")),
                seq_length=_safe_int(
                    row.get(
                        "seq_length", payload.get("seq_length", payload.get("embedding_seq_length"))
                    ),
                ),
                max_train_val_players=_safe_int(
                    payload.get("max_train_val_players", row.get("max_train_val_players"))
                ),
                run_mode=str(row.get("run_mode") or payload.get("run_mode") or "").strip().lower(),
                logs_dir=Path(root).expanduser().resolve() / "training_logs",
                all_arguments_json=payload,
            )
        )
    return rows


def _load_simulation_cutoff_accuracy(
    run_record: SimulationRunRecord,
    *,
    cutoff_value: int,
) -> float:
    """Load one MiniGrid run's validation accuracy at the requested cutoff."""
    metrics_path = (
        run_record.logs_dir
        / run_record.model_name
        / "plots"
        / "val"
        / "cutoff_metrics"
        / "metrics_by_cutoff.csv"
    )
    if not metrics_path.exists():
        raise FileNotFoundError("Missing MiniGrid cutoff metrics: {}".format(metrics_path))
    frame = pd.read_csv(metrics_path)
    matches = frame.loc[
        pd.to_numeric(frame["cutoff_value"], errors="coerce") == float(cutoff_value)
    ]
    if matches.empty:
        raise ValueError("No cutoff_value={} found in {}".format(int(cutoff_value), metrics_path))
    return float(pd.to_numeric(matches.iloc[-1]["accuracy"], errors="coerce"))


def _select_best_simulation_run(
    runs: Iterable[SimulationRunRecord],
    *,
    cutoff_value: int,
) -> Tuple[SimulationRunRecord, float]:
    """Select the best MiniGrid run by validation accuracy at one cutoff."""
    scored_runs: List[Tuple[float, float, float, SimulationRunRecord]] = []
    for run_record in runs:
        cutoff_accuracy = _load_simulation_cutoff_accuracy(run_record, cutoff_value=cutoff_value)
        scored_runs.append(
            (
                float(cutoff_accuracy),
                float(run_record.val_accuracy),
                float(run_record.train_accuracy),
                run_record,
            )
        )
    if not scored_runs:
        raise ValueError("No MiniGrid runs were available for cutoff selection.")
    best_cutoff_accuracy, _, _, best_run = max(
        scored_runs, key=lambda item: (item[0], item[1], item[2])
    )
    return best_run, float(best_cutoff_accuracy)


def _selected_run_row(
    setting: SimulationSetting, task_scope: str, run: SimulationRunRecord, accuracy: float
) -> Dict[str, object]:
    """Provenance row for one selected MiniGrid run."""
    return {
        "setting": setting.key,
        "setting_label": setting.display_label,
        "task_scope": task_scope,
        "model_name": run.model_name,
        "model_type": run.model_type,
        "cutoff_value": MINIGRID_CUTOFF_STEPS,
        "cutoff_accuracy": float(accuracy),
        "overall_val_accuracy": float(run.val_accuracy),
        "logs_dir": artifact_paths.display_path(run.logs_dir),
    }


def build_simulation_matched_head_values(
    settings: Sequence[SimulationSetting],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Return unrounded matched-head values per simulation setting plus the selected-run rows.

    Each value row holds the best single-task MLP accuracies (foraging, cleaning), the two-task
    concat-MLP accuracy and the matched-head gain (two-task minus best single), all at 200
    steps. A setting without a ``*_concat_mlp_probe`` root gets NaN for
    the two-task accuracy and gain, and an explicit ``note``.
    """
    value_rows: List[Dict[str, object]] = []
    selected_run_rows: List[Dict[str, object]] = []

    for setting in settings:
        all_rows = _load_simulation_rows_for_root(
            setting.single_root,
            setting_key=setting.key,
            setting_label=setting.display_label,
        )
        concat_csv = setting.concat_root / "training_logs" / "model_performance.csv"
        if concat_csv.exists():
            all_rows += _load_simulation_rows_for_root(
                setting.concat_root,
                setting_key=setting.key,
                setting_label=setting.display_label,
            )
        filtered_rows = [
            row
            for row in all_rows
            if row.seq_length == MINIGRID_SEQ_LENGTH
            and row.max_train_val_players == MINIGRID_MAX_TRAIN_VAL_PLAYERS
            and row.run_mode == MINIGRID_RUN_MODE
        ]
        by_scope = {
            scope: [row for row in filtered_rows if row.predicted_next_task == scope]
            for scope in (MINIGRID_SINGLE_FORAGING, MINIGRID_SINGLE_CLEANING, MINIGRID_BOTH)
        }
        required = [MINIGRID_SINGLE_FORAGING, MINIGRID_SINGLE_CLEANING] + (
            [MINIGRID_BOTH] if concat_csv.exists() else []
        )
        for scope_name in required:
            if not by_scope[scope_name]:
                raise ValueError(
                    "Missing MiniGrid MLP run type '{}' for '{}'.".format(
                        scope_name, setting.display_label
                    )
                )

        best_foraging, foraging_acc = _select_best_simulation_run(
            by_scope[MINIGRID_SINGLE_FORAGING],
            cutoff_value=MINIGRID_CUTOFF_STEPS,
        )
        best_cleaning, cleaning_acc = _select_best_simulation_run(
            by_scope[MINIGRID_SINGLE_CLEANING],
            cutoff_value=MINIGRID_CUTOFF_STEPS,
        )
        best_single_acc = max(float(foraging_acc), float(cleaning_acc))
        best_single_source = (
            "foraging" if float(foraging_acc) >= float(cleaning_acc) else "cleaning"
        )
        selected_run_rows.append(
            _selected_run_row(setting, "foraging", best_foraging, foraging_acc)
        )
        selected_run_rows.append(
            _selected_run_row(setting, "cleaning", best_cleaning, cleaning_acc)
        )

        if by_scope[MINIGRID_BOTH]:
            best_both, both_acc = _select_best_simulation_run(
                by_scope[MINIGRID_BOTH],
                cutoff_value=MINIGRID_CUTOFF_STEPS,
            )
            both_acc = float(both_acc)
            delta_comp = both_acc - best_single_acc
            note = ""
            selected_run_rows.append(
                _selected_run_row(setting, "cleaning,foraging", best_both, both_acc)
            )
        else:
            both_acc = float("nan")
            delta_comp = float("nan")
            note = "{}: no two-task concat-MLP runs were saved ({} is missing)".format(
                NOT_COMPUTED,
                Path(setting.concat_root.name) / "training_logs" / "model_performance.csv",
            )
        selected_run_rows.append(
            {
                "setting": setting.key,
                "setting_label": setting.display_label,
                "task_scope": "summary",
                "model_name": "",
                "model_type": "",
                "cutoff_value": MINIGRID_CUTOFF_STEPS,
                "cutoff_accuracy": float(delta_comp),
                "overall_val_accuracy": float(best_single_acc),
                "logs_dir": best_single_source,
            }
        )
        value_rows.append(
            {
                "setting": setting.key,
                "setting_label": setting.display_label,
                "cutoff_steps": MINIGRID_CUTOFF_STEPS,
                "foraging_mlp_accuracy": float(foraging_acc),
                "cleaning_mlp_accuracy": float(cleaning_acc),
                "best_single_mlp_accuracy": float(best_single_acc),
                "best_single_source": best_single_source,
                "two_task_mlp_accuracy": both_acc,
                "matched_head_gain": delta_comp,
                "note": note,
            }
        )
    return value_rows, selected_run_rows


def _build_simulation_summary_rows(
    settings: Sequence[SimulationSetting],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Build display rows plus supporting selected-run rows for MiniGrid settings."""
    value_rows, selected_run_rows = build_simulation_matched_head_values(settings)
    display_rows = [
        {
            "Setting": row["setting_label"],
            "Eval point": "{} steps".format(int(row["cutoff_steps"])),
            "Task A": _format_percent(row["foraging_mlp_accuracy"]),
            "Task B": _format_percent(row["cleaning_mlp_accuracy"]),
            "A+B MLP": NOT_COMPUTED
            if row["note"]
            else _format_percent(row["two_task_mlp_accuracy"]),
            "Delta_comp": NOT_COMPUTED if row["note"] else _format_gain(row["matched_head_gain"]),
        }
        for row in value_rows
    ]
    return display_rows, selected_run_rows


def _load_powerwash_mlp_rows(
    log_source: Path,
    *,
    seq_length: int,
    target_mode: str,
) -> List[Dict[str, object]]:
    """Load filtered PowerWash MLP rows."""
    rows: List[Dict[str, object]] = []
    for row in load_powerwash_hierarchical_performance_rows(
        Path(log_source),
        model_types=POWERWASH_MATCHED_HEAD_MODEL_TYPES,
    ):
        payload = _parse_json_dict(row.get("all_arguments_json"))
        effective_target_mode = str(
            row.get("target_mode") or payload.get("target_mode") or ""
        ).strip()
        if effective_target_mode != str(target_mode):
            continue
        effective_seq_length = _safe_int(
            row.get("seq_length", payload.get("seq_length")),
            default=-1,
        )
        if effective_seq_length != int(seq_length):
            continue
        if str(
            row.get("prediction_head_type") or payload.get("prediction_head_type") or ""
        ).strip().lower() != ("mlp"):
            continue
        fold_number = row.get("fold_number") or payload.get("fold_number")
        if fold_number is None:
            continue
        rows.append(
            {
                "model_name": str(row.get("model_name") or "").strip(),
                "model_type": str(row.get("model_type") or payload.get("model_type") or "")
                .strip()
                .lower(),
                "predicted_next_task": _normalize_task_scope(
                    row.get("predicted_next_task") or payload.get("predicted_next_task")
                ),
                "prediction_head_type": "mlp",
                "val_accuracy": _safe_float(row.get("val_accuracy")),
                "naive_val_accuracy": _safe_float(row.get("naive_val_accuracy")),
                "fold_id": "fold_{:02d}".format(int(fold_number)),
                "num_epochs": _safe_int(
                    row.get("num_epochs", payload.get("num_epochs")), default=-1
                ),
                "max_files_per_task": _safe_int(payload.get("max_files_per_task"), default=-1),
                "dim_reduction_method": str(
                    row.get("dim_reduction_method") or payload.get("dim_reduction_method") or ""
                ).strip(),
                "latent_dim": str(row.get("latent_dim") or payload.get("latent_dim") or "").strip(),
                "target_mode": effective_target_mode,
                "condition_label": "",
            }
        )
    return rows


def _load_powerwash_cutoff_frame(
    log_source: Path,
    training_logs_dir: Path,
    target_mode: str,
) -> pd.DataFrame:
    """Load shared-fold PowerWash cutoff accuracies for the three MLP conditions."""
    mlp_rows = _load_powerwash_mlp_rows(
        Path(log_source),
        seq_length=POWERWASH_SEQ_LENGTH,
        target_mode=target_mode,
    )
    condition_rows: Dict[str, Dict[str, Dict[str, object]]] = {
        "Task A": {},
        "Task B": {},
        "A+B MLP": {},
    }
    scope_map = {
        ("single_task", POWERWASH_TASK_A): "Task A",
        ("single_task", POWERWASH_TASK_B): "Task B",
        ("plvm_both_partial", POWERWASH_BOTH): "A+B MLP",
    }
    for row in mlp_rows:
        condition_name = scope_map.get((str(row["model_type"]), str(row["predicted_next_task"])))
        if condition_name is None:
            continue
        fold_id = str(row["fold_id"])
        current = condition_rows[condition_name].get(fold_id)
        if current is None or float(row["val_accuracy"]) > float(current["val_accuracy"]):
            selected = dict(row)
            selected["condition"] = condition_name
            condition_rows[condition_name][fold_id] = selected

    for condition_name, fold_rows in condition_rows.items():
        if not fold_rows:
            raise ValueError("Missing PowerWash MLP rows for '{}'.".format(condition_name))

    fold_frame = build_fold_long_frame(Path(training_logs_dir), condition_rows)
    cutoff_seconds = float(POWERWASH_CUTOFF_MINUTES) * 60.0
    subset = fold_frame.loc[
        (fold_frame["split"] == "val")
        & (fold_frame["condition"].isin(["Task A", "Task B", "A+B MLP"]))
        & np.isclose(fold_frame["cutoff_value"].astype(float).to_numpy(), cutoff_seconds)
    ].copy()
    pivot = subset.pivot_table(
        index="fold_id",
        columns="condition",
        values="accuracy",
        aggfunc="mean",
    ).dropna(subset=["Task A", "Task B", "A+B MLP"])
    if pivot.empty:
        raise ValueError(
            "No shared PowerWash MLP folds were available at {} minutes.".format(
                int(POWERWASH_CUTOFF_MINUTES)
            )
        )
    return pivot.reset_index()


def _build_powerwash_summary_rows(
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Build the PowerWash display row plus supporting fold-level table."""
    shared = _load_powerwash_cutoff_frame(
        args.powerwash_model_performance_csv,
        args.powerwash_training_logs_dir,
        target_mode=str(args.powerwash_target_mode),
    )
    task_a = shared["Task A"].astype(float)
    task_b = shared["Task B"].astype(float)
    both = shared["A+B MLP"].astype(float)

    task_a_mean = float(task_a.mean())
    task_b_mean = float(task_b.mean())
    both_mean = float(both.mean())
    task_a_sd = float(task_a.std(ddof=1)) if len(task_a) > 1 else 0.0
    task_b_sd = float(task_b.std(ddof=1)) if len(task_b) > 1 else 0.0
    both_sd = float(both.std(ddof=1)) if len(both) > 1 else 0.0

    if task_a_mean >= task_b_mean:
        best_single_name = "Task A"
        best_single = task_a
        best_single_mean = task_a_mean
    else:
        best_single_name = "Task B"
        best_single = task_b
        best_single_mean = task_b_mean

    delta_comp = both_mean - best_single_mean
    wilcoxon_p = float("nan")
    deltas = both.to_numpy() - best_single.to_numpy()
    if deltas.size > 1 and not np.allclose(deltas, 0.0):
        try:
            wilcoxon_p = float(stats.wilcoxon(deltas, zero_method="wilcox").pvalue)
        except ValueError:
            wilcoxon_p = float("nan")

    display_row = {
        "Setting": "PowerWash",
        "Eval point": "120 min",
        "Task A": _format_mean_sd(task_a_mean, task_a_sd),
        "Task B": _format_mean_sd(task_b_mean, task_b_sd),
        "A+B MLP": _format_mean_sd(both_mean, both_sd),
        "Delta_comp": "{} ({})".format(
            _format_gain(delta_comp), _format_pvalue(wilcoxon_p)
        ).strip(),
    }

    fold_details = shared.copy()
    fold_details["best_single_name"] = best_single_name
    fold_details["best_single_accuracy"] = best_single.to_numpy()
    fold_details["delta_comp"] = deltas
    fold_details["target_mode"] = str(args.powerwash_target_mode)
    fold_details["cutoff_minutes"] = int(POWERWASH_CUTOFF_MINUTES)
    return display_row, fold_details


def _build_markdown_table(table_frame: pd.DataFrame) -> str:
    """Render ``table_frame`` as a pipe-delimited markdown table."""
    columns = [str(column) for column in table_frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body_lines = []
    for _, row in table_frame.iterrows():
        body_lines.append(
            "| " + " | ".join(str(row[column]) for column in table_frame.columns) + " |"
        )
    return "\n".join([header, separator] + body_lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the cross-domain combination-gain analysis."""
    args = parse_args(argv)

    simulation_rows, simulation_selected_runs = _build_simulation_summary_rows(
        _resolve_simulation_settings(args)
    )
    powerwash_row, powerwash_fold_details = _build_powerwash_summary_rows(args)

    table_frame = pd.DataFrame(simulation_rows + [powerwash_row])
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "{}.csv".format(DEFAULT_OUTPUT_STEM)
    md_path = output_dir / "{}.md".format(DEFAULT_OUTPUT_STEM)
    tex_path = output_dir / "{}.tex".format(DEFAULT_OUTPUT_STEM)
    selected_runs_csv_path = output_dir / "{}_selected_runs.csv".format(DEFAULT_OUTPUT_STEM)
    fold_details_csv_path = output_dir / "{}_powerwash_fold_details.csv".format(DEFAULT_OUTPUT_STEM)

    table_frame.to_csv(csv_path, index=False)
    md_path.write_text(_build_markdown_table(table_frame) + "\n", encoding="utf-8")
    write_latex_table(table_frame, tex_path, leading_left_columns=2)
    pd.DataFrame(simulation_selected_runs).to_csv(selected_runs_csv_path, index=False)
    powerwash_fold_details.to_csv(fold_details_csv_path, index=False)

    print("Wrote {}".format(csv_path))
    print("Wrote {}".format(md_path))
    print("Wrote {}".format(tex_path))
    print("Wrote {}".format(selected_runs_csv_path))
    print("Wrote {}".format(fold_details_csv_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
