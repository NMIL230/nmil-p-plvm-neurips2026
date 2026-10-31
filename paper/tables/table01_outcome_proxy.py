#!/usr/bin/env python3
"""Generates Table 1 from saved PowerWash player-sequence artifacts.

Table 1 summarizes how much simple observed-task outcome features overlap
between players later labeled as ``Zone Planner`` versus ``Zone Hopper``.
By default it uses the fold-00 clean 2-class labels, the train split,
and a 20-minute observation cutoff for both ``back_garden`` and ``bungalow``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from paper.output_paths import resolve_tables_dir
from paper.table_output_utils import (
    resolve_table_output_paths,
    write_latex_table,
)
from powerwash.analysis.outcome_proxy_features import (
    DEFAULT_CUTOFF_SECONDS,
    DEFAULT_INPUT_DIR,
    build_feature_frame,
    build_summary_table_rows,
    filter_invalid_subtask_counts,
    load_zone_labels,
    resolve_default_labels_csv,
    resolve_default_split_manifest,
    resolve_split_player_ids,
)


DEFAULT_MANUSCRIPT_TABLE_STEM = "Table_1"
DEFAULT_SOURCE_TASKS = ("back_garden", "bungalow")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold_number",
        type=int,
        default=0,
        help="Fire Station fold used to resolve the default labels and split manifest.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=("train", "val", "all"),
        help="Player split to summarize from the resolved fold manifest.",
    )
    parser.add_argument(
        "--source_tasks",
        type=str,
        default=",".join(DEFAULT_SOURCE_TASKS),
        help="Comma-separated PowerWash source tasks to include in Table 1.",
    )
    parser.add_argument(
        "--cutoff_seconds",
        type=float,
        default=DEFAULT_CUTOFF_SECONDS,
        help="Observation cutoff in seconds. Default: 1200 seconds (20 min).",
    )
    parser.add_argument(
        "--labels_csv",
        type=Path,
        default=None,
        help="Override planner/hopper labels CSV.",
    )
    parser.add_argument(
        "--player_split_manifest",
        type=Path,
        default=None,
        help="Override player_split_manifest.csv. Ignored for --split all.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Root directory containing the per-task PowerWash player-sequence CSVs.",
    )
    parser.add_argument(
        "--tables_dir",
        type=Path,
        default=resolve_tables_dir(),
        help="Directory for the formatted table outputs.",
    )
    parser.add_argument(
        "--table_stem",
        type=str,
        default=DEFAULT_MANUSCRIPT_TABLE_STEM,
        help="Filename stem for the Table 1 outputs.",
    )
    return parser.parse_args(argv)


def resolve_source_tasks(raw_value: str) -> List[str]:
    """Resolve one comma-separated task list while preserving order."""
    ordered_tasks: List[str] = []
    seen = set()
    for token in str(raw_value or "").split(","):
        task_name = token.strip()
        if not task_name or task_name in seen:
            continue
        ordered_tasks.append(task_name)
        seen.add(task_name)
    if not ordered_tasks:
        raise ValueError("At least one source task is required.")
    return ordered_tasks


def _metric_column_name(cutoff_seconds: float) -> str:
    """Return the Table 1 metric column name for one cutoff."""
    cutoff_minutes = float(cutoff_seconds) / 60.0
    if abs(cutoff_minutes - round(cutoff_minutes)) < 1e-6:
        return "{}-min observed-task outcome metric".format(int(round(cutoff_minutes)))
    return "{:.1f}-min observed-task outcome metric".format(cutoff_minutes)


def build_table_rows(
    *,
    source_tasks: Sequence[str],
    input_dir: Path,
    labels_df: pd.DataFrame,
    cutoff_seconds: float,
) -> List[Dict[str, object]]:
    """Build the combined Table 1 rows across the requested source tasks."""
    rows: List[Dict[str, object]] = []
    for source_task in source_tasks:
        task_dir = (Path(input_dir) / str(source_task)).resolve()
        if not task_dir.is_dir():
            task_dir = Path(input_dir).resolve()
        feature_frame = build_feature_frame(
            source_task=source_task,
            task_dir=task_dir,
            labels_df=labels_df,
            cutoff_seconds=float(cutoff_seconds),
        )
        feature_frame = filter_invalid_subtask_counts(feature_frame, source_task)
        if feature_frame.empty:
            raise ValueError(
                "No feature rows were available for source_task='{}'.".format(source_task)
            )
        rows.extend(
            build_summary_table_rows(
                feature_frame=feature_frame,
                source_task=source_task,
                cutoff_seconds=float(cutoff_seconds),
            )
        )
    return rows


def build_manuscript_table_frame(
    rows: Sequence[Dict[str, object]],
    *,
    cutoff_seconds: float,
) -> pd.DataFrame:
    """Convert the collected summary rows into the formatted Table 1 frame."""
    table_frame = pd.DataFrame(list(rows))
    first_column = "20-min observed-task outcome metric"
    desired_column = _metric_column_name(cutoff_seconds)
    if first_column in table_frame.columns and first_column != desired_column:
        table_frame = table_frame.rename(columns={first_column: desired_column})
    return table_frame


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Generate and write Table 1."""
    args = parse_args(argv)
    source_tasks = resolve_source_tasks(args.source_tasks)
    labels_csv = (
        Path(args.labels_csv)
        if args.labels_csv is not None
        else resolve_default_labels_csv(args.fold_number)
    )
    player_split_manifest = (
        None
        if args.split == "all"
        else (
            Path(args.player_split_manifest)
            if args.player_split_manifest is not None
            else resolve_default_split_manifest(args.fold_number)
        )
    )
    selected_player_ids, split_label = resolve_split_player_ids(
        player_split_manifest,
        args.split,
    )
    labels_df = load_zone_labels(labels_csv, selected_player_ids=selected_player_ids)
    if labels_df.empty:
        raise ValueError("No planner/hopper labels were available after filtering.")

    rows = build_table_rows(
        source_tasks=source_tasks,
        input_dir=Path(args.input_dir),
        labels_df=labels_df,
        cutoff_seconds=float(args.cutoff_seconds),
    )
    manuscript_frame = build_manuscript_table_frame(rows, cutoff_seconds=float(args.cutoff_seconds))

    manuscript_csv, manuscript_tex = resolve_table_output_paths(
        args.table_stem,
        Path(args.tables_dir),
    )
    manuscript_csv.parent.mkdir(parents=True, exist_ok=True)
    manuscript_frame.to_csv(manuscript_csv, index=False)
    write_latex_table(manuscript_frame, manuscript_tex, column_alignment="lcccc")

    print("Split:", split_label)
    print("Source tasks:", ", ".join(source_tasks))
    print("Players:", int(labels_df.shape[0]))
    print("Wrote {}".format(manuscript_csv))
    print("Wrote {}".format(manuscript_tex))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
