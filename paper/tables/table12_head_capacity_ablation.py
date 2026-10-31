#!/usr/bin/env python3
"""Generates Table 12: fusion head capacity (linear vs nonlinear) x number of source tasks.

Reports each cell at an early and a late cutoff per domain, the gain from adding
the second task under each head, and their interaction. Cells are read off the
Figure 5 curves and selection (``fig05_combined_overview``): single/both x
logistic/MLP head, where both + nonlinear is PLVM-2. PowerWash cells are fold mean
+/- SD, and derived rows carry Wilcoxon p-values over the shared folds (the same
test as Table 2). The simulation has one run per condition, so
its cells have no spread and its derived rows no p-value.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper.figures.fig05_combined_overview import (  # noqa: E402
    build_powerwash_curves,
    build_powerwash_fold_frame,
    build_simulation_curves,
)
from paper.output_paths import resolve_tables_dir  # noqa: E402
from paper.powerwash_domain_summary_common import (  # noqa: E402
    _wilcoxon_pvalue as wilcoxon_pvalue,
)
from paper.table_output_utils import resolve_table_output_paths  # noqa: E402

DEFAULT_TABLE_STEM = "head_capacity_ablation"
DEFAULT_POWERWASH_CUTOFF_MINUTES = (60.0, 120.0)
DEFAULT_SIMULATION_CUTOFF_STEPS = (100.0, 200.0)
# Requested cutoffs must be points of the saved cutoff grid (60 s steps for PowerWash,
# 20 steps for the simulation); this tolerance only absorbs float rounding.
CUTOFF_MATCH_TOLERANCE = 1e-6

# (source tasks, head) -> curve key produced by the Figure 5 overview builders.
CELL_CURVE_KEYS = {
    ("Single", "Linear"): "single_linear",
    ("Single", "Nonlinear"): "single_nonlinear",
    ("Both (PLVM)", "Linear"): "both_linear",
    ("Both (PLVM)", "Nonlinear"): "both_nonlinear",
}
CELL_ORDER = (
    ("Single", "Linear"),
    ("Single", "Nonlinear"),
    ("Both (PLVM)", "Linear"),
    ("Both (PLVM)", "Nonlinear"),
)
SINGLE_LABEL = "Single"
BOTH_LABEL = "Both (PLVM)"
# Only PowerWash repeats each condition over a shared fold axis, so it is the
# only domain where the derived rows can be tested for significance.
PAIRED_TEST_DOMAIN = "Fire Station"
REQUIRED_TEST_KEYS = ("single_linear", "single_nonlinear", "both_linear", "both_nonlinear")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--powerwash_cutoff_minutes",
        type=float,
        nargs="+",
        default=list(DEFAULT_POWERWASH_CUTOFF_MINUTES),
        help="PowerWash (Fire Station) observation cutoffs to report, in minutes.",
    )
    parser.add_argument(
        "--simulation_cutoff_steps",
        type=float,
        nargs="+",
        default=list(DEFAULT_SIMULATION_CUTOFF_STEPS),
        help="Simulation observation cutoffs to report, in observation steps.",
    )
    parser.add_argument(
        "--tables_dir",
        type=Path,
        default=resolve_tables_dir(),
        help="Directory where the CSV, markdown, and LaTeX tables will be written.",
    )
    parser.add_argument(
        "--table_stem",
        type=str,
        default=DEFAULT_TABLE_STEM,
        help="File stem shared by the generated table artifacts.",
    )
    return parser.parse_args(argv)


def _value_at_cutoff(
    curve: Tuple[np.ndarray, np.ndarray, np.ndarray],
    cutoff_value: float,
    *,
    domain_label: str,
    curve_key: str,
) -> Tuple[float, float]:
    """Return (accuracy_percent, error_percent) for one curve at one cutoff."""
    x_values, y_values, yerr_values = curve
    matches = np.flatnonzero(
        np.abs(np.asarray(x_values, dtype=float) - float(cutoff_value)) <= CUTOFF_MATCH_TOLERANCE
    )
    if matches.size == 0:
        raise ValueError(
            "{} curve '{}' has no saved cutoff at {:g}; available cutoffs: {}.".format(
                domain_label,
                curve_key,
                cutoff_value,
                ", ".join("{:g}".format(value) for value in np.asarray(x_values, dtype=float)),
            )
        )
    index = int(matches[0])
    return float(y_values[index]), float(yerr_values[index])


def build_cell_rows(
    *,
    powerwash_cutoff_minutes: Sequence[float],
    simulation_cutoff_steps: Sequence[float],
) -> Tuple[pd.DataFrame, str]:
    """Return the tidy per-cell accuracies for both domains.

    The returned frame has one row per (domain, cutoff, source tasks, head)
    cell. ``error_kind`` records what the error column means, which differs by
    domain: a cross-fold SD for PowerWash, a within-run binomial SE for the
    simulation (which has no fold or seed axis).
    """
    powerwash_curves = build_powerwash_curves()
    simulation_curves, simulation_best_task = build_simulation_curves()

    domain_specs = (
        ("Fire Station", powerwash_curves, powerwash_cutoff_minutes, "min", "fold SD"),
        ("Simulation", simulation_curves, simulation_cutoff_steps, "steps", "binomial SE"),
    )

    records: List[dict] = []
    for domain_label, curves, cutoff_values, cutoff_unit, error_kind in domain_specs:
        for cutoff_value in cutoff_values:
            for source_tasks, head in CELL_ORDER:
                curve_key = CELL_CURVE_KEYS[(source_tasks, head)]
                accuracy_percent, error_percent = _value_at_cutoff(
                    curves[curve_key],
                    float(cutoff_value),
                    domain_label=domain_label,
                    curve_key=curve_key,
                )
                records.append(
                    {
                        "domain": domain_label,
                        "cutoff_value": float(cutoff_value),
                        "cutoff_unit": cutoff_unit,
                        "column_label": "{:g} {}".format(float(cutoff_value), cutoff_unit),
                        "source_tasks": source_tasks,
                        "head": head,
                        "curve_key": curve_key,
                        "accuracy_percent": accuracy_percent,
                        "error_percent": error_percent,
                        "error_kind": error_kind,
                    }
                )
    return pd.DataFrame.from_records(records), simulation_best_task


def _column_keys(cell_frame: pd.DataFrame) -> List[Tuple[str, str]]:
    """Return the ordered (domain, column label) pairs spanning the table."""
    ordered = cell_frame.drop_duplicates(subset=["domain", "column_label"])
    return [(str(row["domain"]), str(row["column_label"])) for _, row in ordered.iterrows()]


def _accuracy_lookup(cell_frame: pd.DataFrame) -> Dict[Tuple[str, str, str, str], float]:
    """Index unrounded accuracies by (domain, column label, source tasks, head)."""
    return {
        (
            str(row["domain"]),
            str(row["column_label"]),
            str(row["source_tasks"]),
            str(row["head"]),
        ): float(row["accuracy_percent"])
        for _, row in cell_frame.iterrows()
    }


def build_derived_rows(cell_frame: pd.DataFrame, significance_frame: pd.DataFrame) -> pd.DataFrame:
    """Return the second-task gains and their interaction for every column.

    Gains are computed from unrounded accuracies and rounded once at render
    time, so a reported gain can differ by 0.1 from the difference of the
    rounded cells above it. Paired p-values are attached where the domain has a
    fold axis to test over, and left as NaN elsewhere.
    """
    accuracies = _accuracy_lookup(cell_frame)
    records: List[dict] = []
    for domain, column_label in _column_keys(cell_frame):
        gains = {
            head: accuracies[(domain, column_label, BOTH_LABEL, head)]
            - accuracies[(domain, column_label, SINGLE_LABEL, head)]
            for head in ("Linear", "Nonlinear")
        }
        records.append(
            {
                "domain": domain,
                "column_label": column_label,
                "linear_gain": gains["Linear"],
                "nonlinear_gain": gains["Nonlinear"],
                "interaction": gains["Nonlinear"] - gains["Linear"],
            }
        )
    derived_frame = pd.DataFrame.from_records(records)
    # Left join: domains without a fold axis keep NaN p-values and print bare gains.
    return derived_frame.merge(significance_frame, on=["domain", "column_label"], how="left")


def build_powerwash_significance(cutoff_minutes: Sequence[float]) -> pd.DataFrame:
    """Return paired across-fold p-values for PowerWash's derived rows.

    Each fold gives one paired observation of the linear gain, the nonlinear gain
    and their interaction; each is tested against zero with the Wilcoxon helper of
    ``powerwash_domain_summary_common``.
    """
    fold_frame, key_to_condition = build_powerwash_fold_frame()
    val_frame = fold_frame.loc[fold_frame["split"] == "val"]

    records: List[dict] = []
    for cutoff_value in cutoff_minutes:
        cutoff_seconds = float(cutoff_value) * 60.0
        at_cutoff = val_frame.loc[
            np.abs(val_frame["cutoff_value"].astype(float) - cutoff_seconds)
            <= CUTOFF_MATCH_TOLERANCE
        ]
        # One row per fold, one column per condition, so gains pair within a fold.
        wide = (
            at_cutoff.pivot_table(index="fold_id", columns="condition", values="accuracy") * 100.0
        )
        cells = {
            key: wide[condition]
            for key, condition in key_to_condition.items()
            if key in REQUIRED_TEST_KEYS and condition in wide.columns
        }
        missing = [key for key in REQUIRED_TEST_KEYS if key not in cells]
        if missing:
            raise ValueError(
                "PowerWash cutoff {:g} min is missing fold data for {}.".format(
                    float(cutoff_value), missing
                )
            )
        # Drop any fold that is not scored for all four cells, so every test
        # below is run on one common set of complete folds.
        paired = pd.DataFrame(cells).dropna()
        if paired.empty:
            raise ValueError(
                "PowerWash cutoff {:g} min has no complete folds.".format(float(cutoff_value))
            )
        linear_gain = paired["both_linear"] - paired["single_linear"]
        nonlinear_gain = paired["both_nonlinear"] - paired["single_nonlinear"]
        records.append(
            {
                "domain": PAIRED_TEST_DOMAIN,
                "column_label": "{:g} min".format(float(cutoff_value)),
                "n_folds": int(len(paired)),
                "linear_gain_p": wilcoxon_pvalue(paired["both_linear"], paired["single_linear"]),
                "nonlinear_gain_p": wilcoxon_pvalue(
                    paired["both_nonlinear"], paired["single_nonlinear"]
                ),
                "interaction_p": wilcoxon_pvalue(nonlinear_gain, linear_gain),
            }
        )
    return pd.DataFrame.from_records(records)


def _format_cell(accuracy_percent: float, error_percent: float, *, with_error: bool) -> str:
    """Format one accuracy cell, with or without its spread."""
    if not with_error or not np.isfinite(error_percent):
        return "{:.1f}".format(accuracy_percent)
    return "{:.1f} ± {:.1f}".format(accuracy_percent, error_percent)


def _format_p_value(p_value: float) -> str:
    """Format one two-sided p-value, flooring the display at three decimals."""
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "p<0.001"
    return "p={:.3f}".format(p_value)


def _format_delta(value: float, p_value: float = float("nan")) -> str:
    """Format one signed gain or interaction term, with its p-value when tested."""
    p_text = _format_p_value(p_value)
    if not p_text:
        return "{:+.1f}".format(value)
    return "{:+.1f} ({})".format(value, p_text)


def build_table_frame(cell_frame: pd.DataFrame, derived_frame: pd.DataFrame) -> pd.DataFrame:
    """Return the rendered table frame, one row per printed table row."""
    column_keys = _column_keys(cell_frame)
    column_names = ["Source tasks", "Head"] + [
        "{} {}".format(domain, column_label) for domain, column_label in column_keys
    ]

    records: List[dict] = []
    for source_tasks, head in CELL_ORDER:
        record = {"Source tasks": source_tasks, "Head": head}
        for (domain, column_label), column_name in zip(column_keys, column_names[2:]):
            match = cell_frame.loc[
                (cell_frame["domain"] == domain)
                & (cell_frame["column_label"] == column_label)
                & (cell_frame["source_tasks"] == source_tasks)
                & (cell_frame["head"] == head)
            ].iloc[0]
            # Only PowerWash has a fold axis; simulation cells print bare means.
            record[column_name] = _format_cell(
                float(match["accuracy_percent"]),
                float(match["error_percent"]),
                with_error=str(match["error_kind"]) == "fold SD",
            )
        records.append(record)

    derived_labels = (
        ("Δ from second task, linear head", "linear_gain", "linear_gain_p"),
        ("Δ from second task, nonlinear head", "nonlinear_gain", "nonlinear_gain_p"),
        ("Interaction (nonlinear − linear gain)", "interaction", "interaction_p"),
    )
    for row_label, derived_column, p_column in derived_labels:
        record = {"Source tasks": row_label, "Head": ""}
        for (domain, column_label), column_name in zip(column_keys, column_names[2:]):
            match = derived_frame.loc[
                (derived_frame["domain"] == domain)
                & (derived_frame["column_label"] == column_label)
            ].iloc[0]
            record[column_name] = _format_delta(
                float(match[derived_column]), float(match.get(p_column, float("nan")))
            )
        records.append(record)

    return pd.DataFrame.from_records(records, columns=column_names)


def _latex_delta_cell(cell_text: str) -> str:
    """Render one derived-row cell, math-wrapping the gain and p-value separately."""
    if " (" not in cell_text:
        return "${}$".format(cell_text)
    value_text, _, p_text = cell_text.partition(" (")
    p_text = p_text.rstrip(")")
    relation = "<" if "<" in p_text else "="
    _, _, p_number = p_text.partition(relation)
    return "${}$ ($p{{{}}}{}$)".format(value_text, relation, p_number)


def render_latex_table(cell_frame: pd.DataFrame, table_frame: pd.DataFrame) -> str:
    """Render the full LaTeX table environment with grouped domain headers."""
    column_keys = _column_keys(cell_frame)
    domains: List[str] = []
    for domain, _column_label in column_keys:
        if domain not in domains:
            domains.append(domain)

    # Group the value columns under one \multicolumn span per domain.
    header_cells: List[str] = []
    cmidrules: List[str] = []
    first_value_column = 3
    for domain in domains:
        span = sum(1 for candidate_domain, _ in column_keys if candidate_domain == domain)
        header_cells.append("\\multicolumn{{{}}}{{c}}{{{}}}".format(span, domain))
        cmidrules.append(
            "\\cmidrule(lr){{{}-{}}}".format(first_value_column, first_value_column + span - 1)
        )
        first_value_column += span

    value_columns = list(table_frame.columns)[2:]
    cutoff_header = " & ".join(column_label for _domain, column_label in column_keys)

    lines = [
        "\\begin{{tabular}}{{ll{}}}".format("c" * len(value_columns)),
        "\\toprule",
        "& & {} \\\\".format(" & ".join(header_cells)),
        "".join(cmidrules),
        "Source tasks & Head & {} \\\\".format(cutoff_header),
        "\\midrule",
    ]

    accuracy_rows = table_frame.iloc[: len(CELL_ORDER)]
    for position, (_, row) in enumerate(accuracy_rows.iterrows()):
        # Each source-task block spans its two head rows via \multirow.
        if position % 2 == 0:
            lines.append("\\multirow{{2}}{{*}}{{{}}}".format(row["Source tasks"]))
        values = " & ".join(
            "${}$".format(str(row[column]).replace("±", "\\pm")) for column in value_columns
        )
        lines.append(" & {:<9} & {} \\\\".format(row["Head"], values))

    lines.append("\\midrule")
    for _, row in table_frame.iloc[len(CELL_ORDER) :].iterrows():
        label = str(row["Source tasks"]).replace("Δ", "$\\Delta$").replace("−", "$-$")
        values = " & ".join(_latex_delta_cell(str(row[column])) for column in value_columns)
        lines.append("\\multicolumn{{2}}{{l}}{{{}}} & {} \\\\".format(label, values))

    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Build and save the head-capacity ablation table artifacts."""
    args = parse_args(argv)

    cell_frame, simulation_best_task = build_cell_rows(
        powerwash_cutoff_minutes=args.powerwash_cutoff_minutes,
        simulation_cutoff_steps=args.simulation_cutoff_steps,
    )
    significance_frame = build_powerwash_significance(args.powerwash_cutoff_minutes)
    derived_frame = build_derived_rows(cell_frame, significance_frame)
    table_frame = build_table_frame(cell_frame, derived_frame)

    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path, tex_path = resolve_table_output_paths(args.table_stem, tables_dir)
    md_path = tables_dir / "{}.md".format(args.table_stem)
    cells_csv_path = tables_dir / "{}_cells.csv".format(args.table_stem)
    significance_csv_path = tables_dir / "{}_significance.csv".format(args.table_stem)

    table_frame.to_csv(csv_path, index=False)
    cell_frame.to_csv(cells_csv_path, index=False)
    derived_frame.to_csv(significance_csv_path, index=False)
    # disable_numparse keeps "+1.6" / "53.0" as written instead of being
    # re-parsed as numbers and reformatted by tabulate.
    md_path.write_text(
        table_frame.to_markdown(index=False, disable_numparse=True) + "\n",
        encoding="utf-8",
    )
    tex_path.write_text(render_latex_table(cell_frame, table_frame), encoding="utf-8")

    print("Simulation single-task arm:", simulation_best_task)
    print(table_frame.to_string(index=False))
    for path in (csv_path, cells_csv_path, significance_csv_path, md_path, tex_path):
        print("Wrote {}".format(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
