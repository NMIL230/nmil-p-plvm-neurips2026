#!/usr/bin/env python3
"""Generates Table 8: frozen task-encoder quality.

PowerWash: test-split RMSE of the next-subtask displacement targets for the encoders
named in ``models/powerwash/pw_task_specific_next_subtask.json``, from
``evaluation_metrics.json`` (falling back to the ``training.logs`` metrics line).
Simulation: next-action accuracy of the encoders named in
``pw_task_specific_models_best_256_new.json``; no test split was saved, so the table
prints the range over the validation and training splits.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper.output_paths import resolve_tables_dir  # noqa: E402
from paper.table_output_utils import resolve_table_output_paths, write_latex_table  # noqa: E402
from plvm import artifact_paths  # noqa: E402

POWERWASH_ROOT = artifact_paths.POWERWASH_MODELS_DIR
POWERWASH_SPEC = artifact_paths.POWERWASH_TASK_SPECIFIC_SPEC_JSON
POWERWASH_TASK_DIRS = {"back_garden": "powerwash_back_garden", "bungalow": "powerwash_bungalow"}
POWERWASH_TASK_LABELS = {"back_garden": "Back Garden", "bungalow": "Bungalow"}
POWERWASH_TARGETS = (
    ("delta_x_to_next_subtask", "Delta x"),
    ("delta_y_to_next_subtask", "Delta y"),
    ("delta_z_to_next_subtask", "Delta z"),
)
SIMULATION_ROOT = artifact_paths.SIMULATION_MODELS_DIR
SIMULATION_SPEC = artifact_paths.SIMULATION_TASK_SPECIFIC_SPEC_JSON
SIMULATION_SPLITS = ("val", "train")
DEFAULT_TABLE_STEM = "Table_8"

# "Per-target <split> metrics: {...}" lines written to training.logs by the encoder trainer.
_PER_TARGET_LINE = re.compile(r"Per-target (train|validation|test) metrics: (\{.*\})")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--powerwash_spec",
        type=Path,
        default=POWERWASH_SPEC,
        help="JSON naming the frozen PowerWash encoder runs.",
    )
    parser.add_argument(
        "--powerwash_root",
        type=Path,
        default=POWERWASH_ROOT / "task_specific",
        help="Directory holding the PowerWash encoder runs.",
    )
    parser.add_argument(
        "--simulation_spec",
        type=Path,
        default=SIMULATION_SPEC,
        help="JSON naming the frozen simulation encoder runs.",
    )
    parser.add_argument(
        "--simulation_root",
        type=Path,
        default=SIMULATION_ROOT / "task_specific",
        help="Directory holding the simulation encoder runs.",
    )
    parser.add_argument(
        "--tables_dir", type=Path, default=resolve_tables_dir(), help="Output directory."
    )
    parser.add_argument(
        "--table_stem", type=str, default=DEFAULT_TABLE_STEM, help="Output file stem."
    )
    return parser.parse_args(argv)


def read_test_metrics_by_target(run_dir: Path) -> Dict[str, Dict[str, float]]:
    """Return ``{target: {"rmse": ..., ...}}`` for the test split of one encoder run."""
    metrics_path = run_dir / "evaluation_metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())["test_metrics_by_target"]
    log_path = run_dir / "training.logs"
    for match in _PER_TARGET_LINE.finditer(log_path.read_text()):
        if match.group(1) == "test":
            return ast.literal_eval(match.group(2))
    raise FileNotFoundError("No test metrics in {} or {}".format(metrics_path, log_path))


def _display_path(path: Path) -> str:
    """Repo-relative path when possible (keeps outputs free of machine-specific roots)."""
    return artifact_paths.display_path(path)


def build_table8_values(args: argparse.Namespace) -> pd.DataFrame:
    """Return one unrounded row per Table 8 quantity, with split and source file."""
    rows: List[Dict[str, object]] = []
    pw_runs = json.loads(Path(args.powerwash_spec).read_text())["tasks"]
    for task, task_dir in POWERWASH_TASK_DIRS.items():
        run_dir = Path(args.powerwash_root) / task_dir / "training_logs" / pw_runs[task]
        by_target = read_test_metrics_by_target(run_dir)
        for target, _label in POWERWASH_TARGETS:
            rows.append(
                {
                    "domain": "powerwash",
                    "task": task,
                    "run": pw_runs[task],
                    "split": "test",
                    "quantity": "rmse",
                    "target": target,
                    "value": float(by_target[target]["rmse"]),
                    "source": _display_path(run_dir),
                }
            )
    sim_runs = json.loads(Path(args.simulation_spec).read_text())["tasks"]
    for task in sorted(sim_runs):
        run_dir = Path(args.simulation_root) / task / "training_logs" / sim_runs[task]
        for split in SIMULATION_SPLITS:
            metrics = json.loads((run_dir / "{}_metrics.json".format(split)).read_text())
            rows.append(
                {
                    "domain": "simulation",
                    "task": task,
                    "run": sim_runs[task],
                    "split": split,
                    "quantity": "accuracy",
                    "target": "next_action",
                    "value": float(metrics["accuracy"]),
                    "source": _display_path(run_dir),
                }
            )
    return pd.DataFrame(rows)


def simulation_accuracy_range(values: pd.DataFrame) -> str:
    """Range of the simulation encoder accuracies over tasks and splits, in whole percent."""
    sim = values.loc[values["domain"] == "simulation", "value"] * 100.0
    return "{:.0f}-{:.0f}%".format(sim.min(), sim.max())


def simulation_accuracy_detail(values: pd.DataFrame) -> str:
    """Per task and split simulation accuracies (one decimal), for the table note."""
    sim = values.loc[values["domain"] == "simulation"]
    return "; ".join(
        "{} {} {:.1f}%".format(r.task, r.split, 100.0 * r.value) for r in sim.itertuples()
    )


def build_table8_frame(values: pd.DataFrame) -> pd.DataFrame:
    """Format Table 8: PowerWash test RMSE rows plus the simulation next-action accuracy row."""
    columns = ["Encoder"] + [c for _, c in POWERWASH_TARGETS] + ["Next-action accuracy"]
    rows = []
    for task, label in POWERWASH_TASK_LABELS.items():
        cells = values.loc[(values["domain"] == "powerwash") & (values["task"] == task)].set_index(
            "target"
        )
        row = {"Encoder": "PowerWash {}".format(label), "Next-action accuracy": "--"}
        for target, column in POWERWASH_TARGETS:
            row[column] = "{:.2f}".format(cells.loc[target, "value"])
        rows.append(row)
    sim_row = {column: "--" for _, column in POWERWASH_TARGETS}
    sim_row.update(
        {
            "Encoder": "Simulation (foraging, cleaning)",
            "Next-action accuracy": simulation_accuracy_range(values),
        }
    )
    rows.append(sim_row)
    return pd.DataFrame(rows, columns=columns)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Build and write Table 8."""
    args = parse_args(argv)
    values = build_table8_values(args)
    frame = build_table8_frame(values)
    csv_path, tex_path = resolve_table_output_paths(args.table_stem, Path(args.tables_dir))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    write_latex_table(frame, tex_path, column_alignment="lcccc")
    md_lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    md_lines += [
        "| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)
    ]
    md_path = csv_path.with_suffix(".md")
    md_lines += [
        "",
        "PowerWash: test-split RMSE. Simulation: next-action accuracy over the validation and training "
        "splits (no test split was saved): {}.".format(simulation_accuracy_detail(values)),
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    values_path = csv_path.with_name("{}_values.csv".format(args.table_stem))
    values.to_csv(values_path, index=False)
    for path in (csv_path, tex_path, md_path, values_path):
        print("Wrote {}".format(path))
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
