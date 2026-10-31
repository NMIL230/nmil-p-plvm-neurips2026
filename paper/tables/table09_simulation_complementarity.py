#!/usr/bin/env python3
"""Generates Table 9: simulation complementarity (best single, PLVM, PLVM gain, matched-head gain).

For the complementary, redundant and weak-shared-structure simulations, picks the
best foraging, cleaning and ``plvm_both_partial`` run by validation accuracy at the
reported cutoff (200 steps by default). The matched-head gain comes from
``table09_matched_head_gain.build_simulation_matched_head_values``; a condition
without two-task MLP runs reads "not computed".
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from paper.output_paths import resolve_tables_dir
from paper.tables.table09_matched_head_gain import (
    NOT_COMPUTED,
    SimulationSetting,
    build_simulation_matched_head_values,
)
from paper.table_output_utils import write_latex_table
from plvm import artifact_paths


DEFAULT_MODEL_ROOT = artifact_paths.SIMULATION_MODELS_DIR
DEFAULT_ANALYSIS_OUTPUT_ROOT = (
    artifact_paths.MODEL_TRAINING_ANALYSIS_DIR / "minigrid_foraging_cleaning_condition_comparison"
)
DEFAULT_TABLES_DIR = resolve_tables_dir()
DEFAULT_MANUSCRIPT_TABLE_STEM = "Table_9"
CONCAT_MLP_ROOT_SUFFIX = "_concat_mlp_probe"

TASK_SCOPE_SINGLE_FORAGING = "foraging"
TASK_SCOPE_SINGLE_CLEANING = "cleaning"
TASK_SCOPE_BOTH = "cleaning,foraging"
MODEL_TYPE_SINGLE_TASK = "single_task"
MODEL_TYPE_PLVM_BOTH_PARTIAL = "plvm_both_partial"


@dataclass(frozen=True)
class ConditionSpec:
    """Filesystem and display metadata for one simulation condition."""

    key: str
    display_label: str
    plvm_root: Path

    @property
    def model_performance_csv(self) -> Path:
        """Run registry (``model_performance.csv``) of this condition's PLVM root."""
        return self.plvm_root / "training_logs" / "model_performance.csv"

    @property
    def training_logs_dir(self) -> Path:
        """Directory holding this condition's saved PLVM runs."""
        return self.plvm_root / "training_logs"


@dataclass(frozen=True)
class RunRecord:
    """Minimal metadata needed to select and report one run."""

    condition_key: str
    condition_label: str
    model_name: str
    model_type: str
    predicted_next_task: str
    val_accuracy: float
    train_accuracy: float
    naive_val_accuracy: float
    seq_length: int
    max_train_val_players: int
    run_mode: str
    logs_dir: Path
    all_arguments_json: Dict[str, object]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--complementary_plvm_root",
        type=Path,
        default=DEFAULT_MODEL_ROOT / "plvm_label",
        help="PLVM output root for the complementary condition.",
    )
    parser.add_argument(
        "--redundant_plvm_root",
        type=Path,
        default=DEFAULT_MODEL_ROOT / "plvm_label_redundant",
        help="PLVM output root for the redundant condition.",
    )
    parser.add_argument(
        "--weak_structure_plvm_root",
        type=Path,
        default=DEFAULT_MODEL_ROOT / "plvm_label_weak_structure",
        help="PLVM output root for the weak-structure condition.",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=200,
        help="Filter PLVM runs by sequence length.",
    )
    parser.add_argument(
        "--cutoff_value",
        type=int,
        default=200,
        help="Observed-step cutoff used for the reported comparison.",
    )
    parser.add_argument(
        "--max_train_val_players",
        type=int,
        default=8000,
        help="Balanced player cap used to filter runs. Use <=0 to disable the filter.",
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        default="run",
        choices=["run", "debug"],
        help="Filter runs by run_mode.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for raw analysis exports.",
    )
    parser.add_argument(
        "--tables_dir",
        type=Path,
        default=DEFAULT_TABLES_DIR,
        help="Directory for the formatted table outputs.",
    )
    return parser.parse_args(argv)


def _default_output_dir(seq_length: int, cutoff_value: int, max_train_val_players: int) -> Path:
    player_token = (
        "all_players"
        if int(max_train_val_players) <= 0
        else "players{}".format(int(max_train_val_players))
    )
    return DEFAULT_ANALYSIS_OUTPUT_ROOT / "seq{}_cutoff{}_{}".format(
        int(seq_length),
        int(cutoff_value),
        player_token,
    )


def resolve_output_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path]:
    """Resolve analysis CSVs plus the Table 9 output paths."""
    output_dir = args.output_dir or _default_output_dir(
        args.seq_length,
        args.cutoff_value,
        args.max_train_val_players,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = "minigrid_condition_comparison_seq{}_cutoff{}".format(
        int(args.seq_length),
        int(args.cutoff_value),
    )
    summary_csv = output_dir / "{}.csv".format(stem)
    selected_runs_csv = output_dir / "{}_selected_runs.csv".format(stem)
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    manuscript_csv = tables_dir / "{}.csv".format(DEFAULT_MANUSCRIPT_TABLE_STEM)
    manuscript_tex = tables_dir / "{}.tex".format(DEFAULT_MANUSCRIPT_TABLE_STEM)
    return summary_csv, selected_runs_csv, manuscript_csv, manuscript_tex


def build_condition_specs(args: argparse.Namespace) -> List[ConditionSpec]:
    """Build the three fixed condition specs in Table 9 row order."""
    return [
        ConditionSpec(
            key="complementary",
            display_label="Complementary tasks",
            plvm_root=Path(args.complementary_plvm_root).expanduser().resolve(),
        ),
        ConditionSpec(
            key="redundant",
            display_label="Redundant tasks",
            plvm_root=Path(args.redundant_plvm_root).expanduser().resolve(),
        ),
        ConditionSpec(
            key="weak_structure",
            display_label="Weak shared structure",
            plvm_root=Path(args.weak_structure_plvm_root).expanduser().resolve(),
        ),
    ]


def _parse_json_dict(raw_payload: object) -> Dict[str, object]:
    """Best-effort parser for `all_arguments_json`."""
    text = str(raw_payload or "").strip()
    if not text or text.lower() == "nan":
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _normalize_task_scope(raw_value: object) -> str:
    """Normalize task names to a stable comma-separated token."""
    tokens = [token.strip().lower() for token in str(raw_value or "").split(",") if token.strip()]
    if not tokens:
        return ""
    unique_tokens = sorted(dict.fromkeys(tokens))
    return ",".join(unique_tokens)


def load_condition_runs(
    spec: ConditionSpec,
    *,
    seq_length: int,
    max_train_val_players: int,
    run_mode: str,
) -> List[RunRecord]:
    """Load filtered model-performance rows for one condition."""
    csv_path = spec.model_performance_csv
    if not csv_path.exists():
        raise FileNotFoundError("Missing PLVM performance CSV: {}".format(csv_path))

    df = pd.read_csv(csv_path)
    rows: List[RunRecord] = []
    for _, row in df.iterrows():
        payload = _parse_json_dict(row.get("all_arguments_json"))
        model_type = str(row.get("model_type") or payload.get("model_type") or "").strip().lower()
        if model_type not in {MODEL_TYPE_SINGLE_TASK, MODEL_TYPE_PLVM_BOTH_PARTIAL}:
            continue

        parameters_to_predict = str(
            row.get("parameters_to_predict") or payload.get("parameters_to_predict") or ""
        ).strip()
        if parameters_to_predict != "minigrid_player_label":
            continue

        effective_run_mode = (
            str(row.get("run_mode") or payload.get("run_mode") or "").strip().lower()
        )
        if effective_run_mode != str(run_mode).strip().lower():
            continue

        effective_max_players = _safe_int(
            payload.get("max_train_val_players", row.get("max_train_val_players")),
            default=-1,
        )
        if int(max_train_val_players) > 0 and effective_max_players != int(max_train_val_players):
            continue

        effective_seq_length = _safe_int(
            row.get("seq_length", payload.get("seq_length", payload.get("embedding_seq_length"))),
            default=-1,
        )
        if effective_seq_length != int(seq_length):
            continue

        predicted_next_task = _normalize_task_scope(
            row.get("predicted_next_task") or payload.get("predicted_next_task")
        )
        if not predicted_next_task:
            continue

        rows.append(
            RunRecord(
                condition_key=spec.key,
                condition_label=spec.display_label,
                model_name=str(row.get("model_name") or "").strip(),
                model_type=model_type,
                predicted_next_task=predicted_next_task,
                val_accuracy=_safe_float(row.get("val_accuracy")),
                train_accuracy=_safe_float(row.get("train_accuracy")),
                naive_val_accuracy=_safe_float(row.get("naive_val_accuracy")),
                seq_length=effective_seq_length,
                max_train_val_players=effective_max_players,
                run_mode=effective_run_mode,
                logs_dir=spec.training_logs_dir,
                all_arguments_json=payload,
            )
        )

    if not rows:
        raise ValueError(
            "No filtered PLVM rows found for condition '{}' in {}.".format(
                spec.key,
                csv_path,
            )
        )
    return rows


def load_cutoff_accuracy(
    run_record: RunRecord,
    *,
    cutoff_value: int,
    split_name: str = "val",
) -> float:
    """Load one run's accuracy at the requested cutoff value."""
    metrics_path = (
        run_record.logs_dir
        / run_record.model_name
        / "plots"
        / str(split_name).strip().lower()
        / "cutoff_metrics"
        / "metrics_by_cutoff.csv"
    )
    if not metrics_path.exists():
        raise FileNotFoundError("Missing cutoff metrics: {}".format(metrics_path))

    frame = pd.read_csv(metrics_path)
    if "cutoff_value" not in frame.columns or "accuracy" not in frame.columns:
        raise ValueError("Missing cutoff_value/accuracy columns in {}".format(metrics_path))

    matches = frame.loc[
        pd.to_numeric(frame["cutoff_value"], errors="coerce") == float(cutoff_value)
    ]
    if matches.empty:
        raise ValueError(
            "No cutoff_value={} found in {}".format(
                int(cutoff_value),
                metrics_path,
            )
        )
    return float(pd.to_numeric(matches.iloc[-1]["accuracy"], errors="coerce"))


def select_best_run_at_cutoff(
    runs: Iterable[RunRecord],
    *,
    cutoff_value: int,
) -> Tuple[RunRecord, float]:
    """Select the best run by validation accuracy at one cutoff value."""
    scored_candidates: List[Tuple[float, float, float, RunRecord]] = []
    for run_record in runs:
        cutoff_accuracy = load_cutoff_accuracy(
            run_record,
            cutoff_value=cutoff_value,
            split_name="val",
        )
        scored_candidates.append(
            (
                cutoff_accuracy,
                float(run_record.val_accuracy),
                float(run_record.train_accuracy),
                run_record,
            )
        )
    if not scored_candidates:
        raise ValueError("No candidate runs were available for cutoff selection.")

    best_cutoff_accuracy, _, _, best_run = max(
        scored_candidates,
        key=lambda item: (item[0], item[1], item[2]),
    )
    return best_run, float(best_cutoff_accuracy)


def build_condition_summary(
    spec: ConditionSpec,
    *,
    seq_length: int,
    cutoff_value: int,
    max_train_val_players: int,
    run_mode: str,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Build one Table 9 row and the supporting selected-run detail rows."""
    runs = load_condition_runs(
        spec,
        seq_length=seq_length,
        max_train_val_players=max_train_val_players,
        run_mode=run_mode,
    )

    by_scope = {
        TASK_SCOPE_SINGLE_FORAGING: [
            run_record
            for run_record in runs
            if run_record.model_type == MODEL_TYPE_SINGLE_TASK
            and run_record.predicted_next_task == TASK_SCOPE_SINGLE_FORAGING
        ],
        TASK_SCOPE_SINGLE_CLEANING: [
            run_record
            for run_record in runs
            if run_record.model_type == MODEL_TYPE_SINGLE_TASK
            and run_record.predicted_next_task == TASK_SCOPE_SINGLE_CLEANING
        ],
        TASK_SCOPE_BOTH: [
            run_record
            for run_record in runs
            if run_record.model_type == MODEL_TYPE_PLVM_BOTH_PARTIAL
            and run_record.predicted_next_task == TASK_SCOPE_BOTH
        ],
    }
    for scope_name, scope_runs in by_scope.items():
        if not scope_runs:
            raise ValueError(
                "Missing required run type '{}' for condition '{}'.".format(
                    scope_name,
                    spec.key,
                )
            )

    selected_details: List[Dict[str, object]] = []
    selected_by_scope: Dict[str, Tuple[RunRecord, float]] = {}
    for scope_name, scope_runs in by_scope.items():
        best_run, cutoff_accuracy = select_best_run_at_cutoff(
            scope_runs,
            cutoff_value=cutoff_value,
        )
        selected_by_scope[scope_name] = (best_run, cutoff_accuracy)
        selected_details.append(
            {
                "condition_key": spec.key,
                "condition_label": spec.display_label,
                "task_scope": scope_name,
                "model_name": best_run.model_name,
                "model_type": best_run.model_type,
                "cutoff_value": int(cutoff_value),
                "cutoff_val_accuracy": float(cutoff_accuracy),
                "overall_val_accuracy": float(best_run.val_accuracy),
                "train_accuracy": float(best_run.train_accuracy),
                "seq_length": int(best_run.seq_length),
                "max_train_val_players": int(best_run.max_train_val_players),
                "run_mode": str(best_run.run_mode),
                "logs_dir": artifact_paths.display_path(best_run.logs_dir),
            }
        )

    foraging_val = float(selected_by_scope[TASK_SCOPE_SINGLE_FORAGING][1])
    cleaning_val = float(selected_by_scope[TASK_SCOPE_SINGLE_CLEANING][1])
    plvm_val = float(selected_by_scope[TASK_SCOPE_BOTH][1])
    best_single_task = max(foraging_val, cleaning_val)
    best_single_task_source = (
        TASK_SCOPE_SINGLE_FORAGING if foraging_val >= cleaning_val else TASK_SCOPE_SINGLE_CLEANING
    )

    row = {
        "Task relationship": spec.display_label,
        "Best single-task": best_single_task,
        "PLVM": plvm_val,
        "PLVM gain": plvm_val - best_single_task,
        "Best single-task source": best_single_task_source,
        "Best single-task model": selected_by_scope[best_single_task_source][0].model_name,
        "PLVM model": selected_by_scope[TASK_SCOPE_BOTH][0].model_name,
        "cutoff_value": int(cutoff_value),
        "seq_length": int(seq_length),
        "max_train_val_players": int(max_train_val_players),
        "run_mode": str(run_mode),
    }
    return row, selected_details


def _format_percent(value: float, *, decimals: int = 1) -> str:
    """Format one accuracy value as a percent string."""
    return "{:.{decimals}f}".format(float(value) * 100.0, decimals=decimals)


def _format_gain(value: float, *, decimals: int = 1) -> str:
    """Format one gain value as a signed percent-point string."""
    return "{:+.{decimals}f}".format(float(value) * 100.0, decimals=decimals)


def build_matched_head_values(specs: Sequence[ConditionSpec]) -> List[Dict[str, object]]:
    """Matched-head values per condition (single-task MLPs from the PLVM root, two-task MLP from its concat root)."""
    settings = [
        SimulationSetting(
            key=spec.key,
            display_label=spec.display_label,
            single_root=spec.plvm_root,
            concat_root=spec.plvm_root.parent / (spec.plvm_root.name + CONCAT_MLP_ROOT_SUFFIX),
        )
        for spec in specs
    ]
    value_rows, _selected = build_simulation_matched_head_values(settings)
    return value_rows


def build_manuscript_table_frame(
    summary_rows: Sequence[Dict[str, object]],
    matched_head_rows: Sequence[Dict[str, object]],
) -> pd.DataFrame:
    """Build the formatted Table 9 frame (one row per task relationship)."""
    rows = []
    for row, matched in zip(summary_rows, matched_head_rows):
        rows.append(
            {
                "Task relationship": row["Task relationship"],
                "Best single-task": _format_percent(row["Best single-task"]),
                "PLVM": _format_percent(row["PLVM"]),
                "PLVM gain": _format_gain(row["PLVM gain"]),
                "Matched-head gain": NOT_COMPUTED
                if matched["note"]
                else _format_gain(matched["matched_head_gain"]),
            }
        )
    return pd.DataFrame(rows)


def build_values_frame(
    summary_rows: Sequence[Dict[str, object]],
    matched_head_rows: Sequence[Dict[str, object]],
) -> pd.DataFrame:
    """Unrounded Table 9 values (accuracies as fractions) with the matched-head provenance note."""
    rows = []
    for row, matched in zip(summary_rows, matched_head_rows):
        rows.append(
            {
                "task_relationship": row["Task relationship"],
                "best_single_task_accuracy": float(row["Best single-task"]),
                "best_single_task_source": row["Best single-task source"],
                "plvm_accuracy": float(row["PLVM"]),
                "plvm_gain": float(row["PLVM gain"]),
                "matched_head_best_single_mlp_accuracy": float(matched["best_single_mlp_accuracy"]),
                "matched_head_two_task_mlp_accuracy": float(matched["two_task_mlp_accuracy"]),
                "matched_head_gain": float(matched["matched_head_gain"]),
                "matched_head_note": matched["note"],
                "cutoff_value": int(row["cutoff_value"]),
            }
        )
    return pd.DataFrame(rows)


def render_markdown(frame: pd.DataFrame, values: pd.DataFrame) -> str:
    """Markdown table plus a note for every cell that could not be computed."""
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines += [
        "| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)
    ]
    notes = [
        (r.task_relationship, r.matched_head_note)
        for r in values.itertuples()
        if r.matched_head_note
    ]
    if notes:
        lines.append("")
        lines += ["Matched-head gain, {}: {}.".format(name, note) for name, note in notes]
    return "\n".join(lines) + "\n"


def write_selected_runs_csv(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    """Write the selected-run detail rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition_key",
        "condition_label",
        "task_scope",
        "model_name",
        "model_type",
        "cutoff_value",
        "cutoff_val_accuracy",
        "overall_val_accuracy",
        "train_accuracy",
        "seq_length",
        "max_train_val_players",
        "run_mode",
        "logs_dir",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the condition-comparison table workflow."""
    args = parse_args(argv)
    summary_csv, selected_runs_csv, manuscript_csv, manuscript_tex = resolve_output_paths(args)

    summary_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []
    specs = build_condition_specs(args)
    for spec in specs:
        condition_summary, condition_selected = build_condition_summary(
            spec,
            seq_length=args.seq_length,
            cutoff_value=args.cutoff_value,
            max_train_val_players=args.max_train_val_players,
            run_mode=args.run_mode,
        )
        summary_rows.append(condition_summary)
        selected_rows.extend(condition_selected)

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(summary_csv, index=False)
    write_selected_runs_csv(selected_rows, selected_runs_csv)

    matched_head_rows = build_matched_head_values(specs)
    manuscript_frame = build_manuscript_table_frame(summary_rows, matched_head_rows)
    manuscript_frame.to_csv(manuscript_csv, index=False)
    write_latex_table(manuscript_frame, manuscript_tex, column_alignment="lcccc")
    values_frame = build_values_frame(summary_rows, matched_head_rows)
    values_csv = manuscript_csv.with_name("{}_values.csv".format(DEFAULT_MANUSCRIPT_TABLE_STEM))
    values_frame.to_csv(values_csv, index=False)
    manuscript_md = manuscript_csv.with_suffix(".md")
    manuscript_md.write_text(render_markdown(manuscript_frame, values_frame), encoding="utf-8")

    print("Wrote {}".format(summary_csv))
    print("Wrote {}".format(selected_runs_csv))
    print("Wrote {}".format(manuscript_csv))
    print("Wrote {}".format(manuscript_tex))
    print("Wrote {}".format(manuscript_md))
    print("Wrote {}".format(values_csv))
    for note in values_frame["matched_head_note"]:
        if note:
            print("NOTE matched-head gain: {}".format(note))
    for row in summary_rows:
        print(
            "{}: best_single_task={:.4f} plvm={:.4f} gain={:+.4f}".format(
                row["Task relationship"],
                float(row["Best single-task"]),
                float(row["PLVM"]),
                float(row["PLVM gain"]),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
