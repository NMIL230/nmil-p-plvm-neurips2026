#!/usr/bin/env python3
"""Generates Table 13: Storybook House as the prediction target (2-class and 3-class blocks).

The Table 2 comparison plus a Linear fusion row, read from the ``*_storybook`` training-log
families: Outcome, Bungalow and Back Garden single task, Linear fusion and PLVM-2 at
20/60/120 min (mean +/- SD over shared folds), with PLVM deltas and paired Wilcoxon
p-values. Every number comes from one call of
``powerwash_domain_summary_common.build_domain_summary_rows`` with ``Back Garden Alone``
as an extra single-task condition. ``powerwash_summary_storybook.csv`` holds the
unrounded values.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper.output_paths import resolve_tables_dir  # noqa: E402
from paper.powerwash_domain_summary_common import (  # noqa: E402
    DEFAULT_TARGET_MODES,
    TRAINING_LOGS_DIR,
    build_domain_summary_rows,
    extra_condition_prefix,
    render_latex_table,
    render_markdown_table,
)
from paper.table_output_utils import resolve_table_output_paths, write_latex_table  # noqa: E402

FAMILY_SUFFIX = "_storybook"
OUTPUT_STEM = "powerwash_summary_storybook"
DEFAULT_TABLE_STEM = "Table_13"
BACK_GARDEN_CONDITION = "Back Garden Alone"
_BG = extra_condition_prefix(BACK_GARDEN_CONDITION)
# (row label, mean column, SD column) and (row label, delta column, p column), in table order.
ACCURACY_ROWS = (
    ("Outcome", "outcome_mean_accuracy", "outcome_sd_accuracy"),
    ("Single task (Bungalow)", "single_task_mean_accuracy", "single_task_sd_accuracy"),
    ("Single task (Back Garden)", "{}_mean_accuracy".format(_BG), "{}_sd_accuracy".format(_BG)),
    ("Linear fusion", "logistic_mean_accuracy", "logistic_sd_accuracy"),
    ("PLVM", "plvm_mean_accuracy", "plvm_sd_accuracy"),
)
DELTA_ROWS = (
    ("Delta PLVM - Bungalow", "plvm_minus_single_task", "plvm_minus_single_task_wilcoxon_p"),
    (
        "Delta PLVM - Back Garden",
        "plvm_minus_{}".format(_BG),
        "plvm_minus_{}_wilcoxon_p".format(_BG),
    ),
    ("Delta PLVM - Linear fusion", "plvm_minus_logistic", "plvm_minus_logistic_wilcoxon_p"),
    ("Delta PLVM - Outcome", "plvm_minus_outcome", "plvm_minus_outcome_wilcoxon_p"),
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--training_logs_dir",
        type=Path,
        default=TRAINING_LOGS_DIR,
        help="PowerWash hierarchical training-log root (the *_storybook families are read).",
    )
    parser.add_argument(
        "--seq_length", type=int, default=120, help="Sequence length of the runs to summarize."
    )
    parser.add_argument(
        "--target_mode",
        action="append",
        default=None,
        help="Target mode(s) to summarize; repeatable (default: the 2-class clean and "
        "3-class zone-hopping labels).",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=resolve_tables_dir(), help="Output directory."
    )
    return parser.parse_args(argv)


def _target_modes(args: argparse.Namespace) -> List[str]:
    return list(args.target_mode or DEFAULT_TARGET_MODES)


def _format_p(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return "<0.001" if float(value) < 1e-3 else "{:.3f}".format(float(value))


def build_paper_layout_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Lay out Table 13: one block per label target, cutoffs as columns."""
    cutoffs = sorted(int(c) for c in frame["cutoff_minutes"].unique())
    columns = ["Target", "Model"] + ["{} min".format(c) for c in cutoffs]
    rows = []
    for target_label in sorted(frame["target_label"].unique()):
        block = frame.loc[frame["target_label"] == target_label].set_index("cutoff_minutes")
        for label, mean_col, sd_col in ACCURACY_ROWS:
            cells = [
                "N/A"
                if pd.isna(block.loc[c, mean_col])
                else "{:.1f} ± {:.1f}".format(
                    100.0 * block.loc[c, mean_col], 100.0 * block.loc[c, sd_col]
                )
                for c in cutoffs
            ]
            rows.append([target_label, label] + cells)
        for label, delta_col, p_col in DELTA_ROWS:
            cells = [
                "N/A"
                if pd.isna(block.loc[c, delta_col])
                else "{:+.1f} (p={})".format(
                    100.0 * block.loc[c, delta_col], _format_p(block.loc[c, p_col])
                )
                for c in cutoffs
            ]
            rows.append([target_label, label] + cells)
    return pd.DataFrame(rows, columns=columns)


def render_paper_markdown(table: pd.DataFrame) -> str:
    """Render the Table 13 layout as GitHub markdown."""
    lines = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    lines += [
        "| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Build and save Table 13 and the Storybook House summary tables."""
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = (
        build_domain_summary_rows(
            training_logs_root=Path(args.training_logs_dir),
            family_suffix=FAMILY_SUFFIX,
            seq_length=int(args.seq_length),
            target_modes=_target_modes(args),
            extra_single_conditions=(BACK_GARDEN_CONDITION,),
        )
        .sort_values(["target_label", "cutoff_minutes"])
        .reset_index(drop=True)
    )

    csv_path = output_dir / "{}.csv".format(OUTPUT_STEM)
    md_path = output_dir / "{}.md".format(OUTPUT_STEM)
    frame.to_csv(csv_path, index=False)
    md_path.write_text(render_markdown_table(frame), encoding="utf-8")
    print("Wrote", csv_path)
    print("Wrote", md_path)

    for target_mode in frame["target_mode"].unique():
        target_frame = frame.loc[frame["target_mode"] == target_mode]
        target_label = target_frame["target_label"].iloc[0]
        tex_path = output_dir / "{}_{}.tex".format(OUTPUT_STEM, target_label.replace("-", "_"))
        tex_path.write_text(
            render_latex_table(target_frame),
            encoding="utf-8",
        )
        print("Wrote", tex_path)

    table = build_paper_layout_frame(frame)
    table_csv, table_tex = resolve_table_output_paths(DEFAULT_TABLE_STEM, output_dir)
    table.to_csv(table_csv, index=False)
    write_latex_table(table, table_tex, leading_left_columns=2)
    table_md = table_csv.with_suffix(".md")
    table_md.write_text(render_paper_markdown(table), encoding="utf-8")
    for path in (table_csv, table_tex, table_md):
        print("Wrote", path)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
