#!/usr/bin/env python3
"""Generates Table 2: the Fire Station 2-class prediction summary and majority-class rates.

Reports validation accuracy (mean +/- SD over folds) for Outcome, both single tasks
and PLVM-2 at 20/60/120 min plus the normalised accuracy-curve area up to 120 min,
and PLVM deltas with paired Wilcoxon p-values. Runs are the best per condition and
fold by validation accuracy (``select_all_rows_by_fold``), and the paired statistics
come from ``table10_robustness.build_paired_fold_tests``, so Tables 2 and 10 share
every delta. Also writes ``Table_2_values.csv`` (unrounded, one row per cell).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper.figures.fig06_fig08_fig09_fig10_powerwash_accuracy_curves import (  # noqa: E402
    DEFAULT_PERF_CSV,
    SUMMARY_TIMEPOINT_MINUTES,
    TRAINING_LOGS_DIR,
    build_fold_long_frame,
    load_candidate_rows,
    select_all_rows_by_fold,
)
from paper.output_paths import resolve_tables_dir  # noqa: E402
from paper.tables.table10_robustness import (  # noqa: E402
    TABLE10_COMPARISONS,
    TABLE10_PLVM_CONDITION,
    TABLE10_TARGET_MODE,
    _auc_values_by_fold,
    _fmt_p_value,
    _resolve_val_predictions_path,
    build_paired_fold_tests,
)
from paper.table_output_utils import (  # noqa: E402
    resolve_table_output_paths,
    write_latex_table,
)

DEFAULT_MANUSCRIPT_TABLE_STEM = "Table_2"
# Metric key (shared with table10_robustness) of the normalised mean accuracy over 5-120 min.
MEAN_METRIC = "AUC at 120 min"
# (Table 2 row label, condition) in table row order.
ACCURACY_ROWS = (
    ("Outcome", "RF Both"),
    ("Single task (Bungalow)", "Bungalow Alone"),
    ("Single task (Back Garden)", "Back Garden Alone"),
    ("PLVM", TABLE10_PLVM_CONDITION),
)
COLUMN_LABELS = ["{} min".format(int(m)) for m in SUMMARY_TIMEPOINT_MINUTES] + [
    "Mean acc. (5-120 min)"
]
METRICS = ["{} min accuracy".format(int(m)) for m in SUMMARY_TIMEPOINT_MINUTES] + [MEAN_METRIC]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model_performance_csv",
        type=Path,
        default=DEFAULT_PERF_CSV,
        help="PowerWash training_logs root (or one model_performance.csv) used for run selection.",
    )
    parser.add_argument(
        "--training_logs_dir",
        type=Path,
        default=TRAINING_LOGS_DIR,
        help="Directory containing the per-run cutoff metrics and validation predictions.",
    )
    parser.add_argument(
        "--seq_length", type=int, default=120, help="Sequence length of the runs to summarize."
    )
    parser.add_argument(
        "--tables_dir",
        type=Path,
        default=resolve_tables_dir(),
        help="Directory for the formatted table outputs.",
    )
    parser.add_argument(
        "--table_stem", type=str, default=DEFAULT_MANUSCRIPT_TABLE_STEM, help="Output file stem."
    )
    return parser.parse_args(argv)


def _accuracy_cells(fold_frame: pd.DataFrame, condition: str) -> List[Dict[str, object]]:
    """Mean/SD over folds of one condition's validation accuracy at each cutoff and for the 5-120 min mean."""
    frame = fold_frame.loc[(fold_frame["split"] == "val") & (fold_frame["condition"] == condition)]
    cells = []
    for minutes, metric in zip(SUMMARY_TIMEPOINT_MINUTES, METRICS):
        values = frame.loc[
            np.isclose(frame["cutoff_value"].astype(float), float(minutes) * 60.0), "accuracy"
        ]
        cells.append({"metric": metric, "values": values.astype(float).to_numpy()})
    cells.append(
        {
            "metric": MEAN_METRIC,
            "values": np.asarray(list(_auc_values_by_fold(frame).values()), dtype=float),
        }
    )
    return [
        {
            "metric": c["metric"],
            "mean": float(c["values"].mean()),
            "sd": float(c["values"].std(ddof=1)),
            "n_folds": int(c["values"].size),
        }
        for c in cells
    ]


def majority_class_rates(
    training_logs_dir: Path, fold_rows: Dict[str, dict]
) -> List[Dict[str, object]]:
    """Majority-class share of the held-out players per fold and cutoff (mean/SD over folds).

    Uses the selected PLVM runs' ``val_predictions.csv`` so the rate refers to exactly the
    players the accuracies are computed on. The PLVM writes one row per observed task, so
    rows are first collapsed to one per (player, cutoff).
    """
    by_cutoff: Dict[float, List[float]] = {float(m): [] for m in SUMMARY_TIMEPOINT_MINUTES}
    for _fold_id, row in sorted(fold_rows.items()):
        frame = pd.read_csv(
            _resolve_val_predictions_path(Path(training_logs_dir), str(row["model_name"])),
            usecols=["player_id", "observation_cutoff_seconds", "target"],
        )
        frame = frame.drop_duplicates(subset=["player_id", "observation_cutoff_seconds"])
        for minutes in by_cutoff:
            at_cutoff = frame.loc[
                np.isclose(frame["observation_cutoff_seconds"].astype(float), minutes * 60.0)
            ]
            by_cutoff[minutes].append(float(at_cutoff["target"].value_counts(normalize=True).max()))
    return [
        {
            "metric": "{} min accuracy".format(int(m)),
            "mean": float(np.mean(v)),
            "sd": float(np.std(v, ddof=1)),
            "n_folds": len(v),
        }
        for m, v in by_cutoff.items()
    ]


def build_table2_values(
    model_performance_csv: Path, training_logs_dir: Path, seq_length: int
) -> pd.DataFrame:
    """Return one unrounded row per Table 2 cell (accuracy, delta and majority rows)."""
    rows = load_candidate_rows(
        Path(model_performance_csv), seq_length=int(seq_length), target_mode=TABLE10_TARGET_MODE
    )
    if not rows:
        raise ValueError("No candidate runs for target_mode='{}'.".format(TABLE10_TARGET_MODE))
    _, rows_by_condition_fold = select_all_rows_by_fold(rows)
    needed = [condition for _, condition in ACCURACY_ROWS]
    missing = [c for c in needed if c not in rows_by_condition_fold]
    if missing:
        raise ValueError("Missing Table 2 conditions: {}".format(missing))
    fold_frame = build_fold_long_frame(
        Path(training_logs_dir), {c: rows_by_condition_fold[c] for c in needed}
    )

    records: List[Dict[str, object]] = []
    for label, condition in ACCURACY_ROWS:
        for cell in _accuracy_cells(fold_frame, condition):
            records.append(dict(cell, row=label, kind="accuracy", condition=condition))
    for comparison_label, baseline in TABLE10_COMPARISONS:
        for paired in build_paired_fold_tests(
            fold_frame, TABLE10_TARGET_MODE, TABLE10_PLVM_CONDITION, baseline
        ):
            records.append(
                {
                    "row": "Delta PLVM - {}".format(comparison_label),
                    "kind": "delta",
                    "condition": baseline,
                    "metric": paired["metric"],
                    "mean": float(paired["mean_delta"]),
                    "sd": float("nan"),
                    "n_folds": int(paired["n_folds"]),
                    "wilcoxon_p": float(paired["wilcoxon_p"]),
                }
            )
    for cell in majority_class_rates(
        Path(training_logs_dir), rows_by_condition_fold[TABLE10_PLVM_CONDITION]
    ):
        records.append(
            dict(cell, row="Majority class", kind="majority", condition=TABLE10_PLVM_CONDITION)
        )
    values = pd.DataFrame(records)
    values.insert(0, "target_mode", TABLE10_TARGET_MODE)
    return values[
        ["target_mode", "row", "kind", "condition", "metric", "mean", "sd", "n_folds", "wilcoxon_p"]
    ]


def _format_cell(record: pd.Series) -> str:
    """Format one cell as shown in Table 2 (percent, one decimal)."""
    if record["kind"] == "accuracy":
        return "{:.1f} ± {:.1f}".format(100.0 * record["mean"], 100.0 * record["sd"])
    if record["kind"] == "delta":
        return "{:+.1f} (p={})".format(100.0 * record["mean"], _fmt_p_value(record["wilcoxon_p"]))
    return "{:.1f}".format(100.0 * record["mean"])


def build_table2_frame(values: pd.DataFrame) -> pd.DataFrame:
    """Pivot the cell values into Table 2 (one row per model / delta, one column per cutoff)."""
    rows = []
    for label in dict.fromkeys(values["row"]):
        cells = values.loc[values["row"] == label].set_index("metric")
        row = {"Model": label}
        for column, metric in zip(COLUMN_LABELS, METRICS):
            row[column] = _format_cell(cells.loc[metric]) if metric in cells.index else "--"
        rows.append(row)
    return pd.DataFrame(rows, columns=["Model"] + COLUMN_LABELS)


def render_markdown(frame: pd.DataFrame) -> str:
    """Render the formatted table as GitHub markdown."""
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines += [
        "| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Build and write Table 2."""
    args = parse_args(argv)
    values = build_table2_values(
        Path(args.model_performance_csv), Path(args.training_logs_dir), int(args.seq_length)
    )
    frame = build_table2_frame(values)

    csv_path, tex_path = resolve_table_output_paths(args.table_stem, Path(args.tables_dir))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    write_latex_table(frame, tex_path, column_alignment="lcccc")
    md_path = csv_path.with_suffix(".md")
    md_path.write_text(render_markdown(frame), encoding="utf-8")
    values_path = csv_path.with_name("{}_values.csv".format(args.table_stem))
    values.to_csv(values_path, index=False)
    for path in (csv_path, tex_path, md_path, values_path):
        print("Wrote {}".format(path))
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
