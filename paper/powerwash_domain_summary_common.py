#!/usr/bin/env python3
"""Shared builder for the four-method PowerWash prediction-target summary table.

Compares Single-task / Logistic Fuser / Best Outcome / PLVM (mean +/- SD at
20/60/120 minutes, plus pairwise Wilcoxon deltas) for either target: Fire Station
reads ``plvm_models``, ``single_task``, ``rf_baselines`` and
``concat_logistic_probe``; Storybook reads the matching ``*_storybook`` families.
All statistics at a cutoff use only the folds where every compared condition has
a run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper.figures.fig06_fig08_fig09_fig10_powerwash_accuracy_curves import (  # noqa: E402
    SUMMARY_TIMEPOINT_MINUTES,
    TRAINING_LOGS_DIR,  # noqa: F401  (re-exported for the Table 2 / Table 13 scripts)
    build_fold_long_frame,
    load_candidate_rows,
    load_logistic_rows as _load_logistic_rows,
    select_all_rows_by_fold,
    select_best_logistic_condition,
)
from paper.tables.table10_robustness import _best_condition_by_family  # noqa: E402

DEFAULT_TARGET_MODES = (
    "player_label_2_class_clean_zonehopping",
    "player_label_3_class_zonehopping",
)
FIXED_SUMMARY_CONDITIONS = {
    "player_label_2_class_clean_zonehopping": {
        "PLVM": "PLVM-2",
        "Single-task process": "Bungalow Alone",
    },
    "player_label_3_class_zonehopping": {
        "PLVM": "PLVM-2",
        "Single-task process": "Bungalow Alone",
    },
}
# (base family directory name) -> (model type it contains)
BASE_FAMILY_DIRS = {
    "plvm_models": "plvm_both_partial",
    "single_task": "single_task",
    "rf_baselines": "rf_baseline",
}
# Logistic-head runs live in their own family directories and are optional.
# Only the two-task concat logistic probe ("Logistic Fuser") is part of this
# summary; the single-task logistic probe family is deliberately not searched.
LOGISTIC_FAMILY_DIRS = ("concat_logistic_probe",)


def _target_label(target_mode: str) -> str:
    """Return a compact human-readable label for one target mode."""
    labels = {
        "player_label_2_class_clean_zonehopping": "2-class",
        "player_label_3_class_zonehopping": "3-class",
    }
    return labels.get(str(target_mode), str(target_mode))


def _load_base_rows(
    training_logs_root: Path,
    family_suffix: str,
    seq_length: int,
    target_mode: str,
) -> List[dict]:
    """Load process/outcome/single-task rows from one target's family directories."""
    rows: List[dict] = []
    for base_family_dir in BASE_FAMILY_DIRS:
        csv_path = (
            Path(training_logs_root)
            / "{}{}".format(base_family_dir, family_suffix)
            / "model_performance.csv"
        )
        if not csv_path.exists():
            continue
        rows.extend(
            load_candidate_rows(csv_path, seq_length=int(seq_length), target_mode=str(target_mode))
        )
    return rows


def _load_logistic_rows_for_target(
    training_logs_root: Path,
    family_suffix: str,
    seq_length: int,
    target_mode: str,
) -> List[dict]:
    """Load logistic-head rows from one target's logistic family directories, if any exist."""
    rows: List[dict] = []
    for family_dir in LOGISTIC_FAMILY_DIRS:
        csv_path = (
            Path(training_logs_root)
            / "{}{}".format(family_dir, family_suffix)
            / "model_performance.csv"
        )
        if not csv_path.exists():
            continue
        rows.extend(
            _load_logistic_rows(csv_path, seq_length=int(seq_length), target_mode=str(target_mode))
        )
    return rows


def _wilcoxon_pvalue(values_a: pd.Series, values_b: pd.Series) -> float:
    """Return the Wilcoxon signed-rank p-value for one shared-fold comparison."""
    deltas = values_a.astype(float).to_numpy() - values_b.astype(float).to_numpy()
    if deltas.size <= 1 or np.allclose(deltas, 0.0):
        return float("nan")
    try:
        return float(stats.wilcoxon(deltas, zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def _shared_cutoff_rows(
    fold_frame: pd.DataFrame,
    conditions: Sequence[str],
    cutoff_minutes: float,
) -> pd.DataFrame:
    """Return one shared-fold validation row per condition at the requested cutoff."""
    cutoff_seconds = float(cutoff_minutes) * 60.0
    subset = fold_frame.loc[
        (fold_frame["split"] == "val")
        & (fold_frame["condition"].isin(list(conditions)))
        & np.isclose(fold_frame["cutoff_value"].astype(float).to_numpy(), cutoff_seconds)
    ].copy()
    pivot = subset.pivot_table(
        index="fold_id",
        columns="condition",
        values="accuracy",
        aggfunc="mean",
    ).dropna(subset=list(conditions))
    return pivot.reset_index()


def extra_condition_prefix(condition: str) -> str:
    """Column prefix for an extra single-task condition, e.g. 'Back Garden Alone' -> 'back_garden_single_task'."""
    return str(condition).replace(" Alone", "").strip().lower().replace(" ", "_") + "_single_task"


def build_domain_summary_rows(
    training_logs_root: Path,
    family_suffix: str,
    seq_length: int,
    target_modes: Sequence[str] = DEFAULT_TARGET_MODES,
    extra_single_conditions: Sequence[str] = (),
) -> pd.DataFrame:
    """Build the four-method (or three, if no logistic runs exist) summary table.

    Each ``extra_single_conditions`` entry (a ``select_all_rows_by_fold`` condition such as
    ``"Back Garden Alone"``) joins the shared-fold comparison and adds the columns
    ``<prefix>_condition``, ``<prefix>_mean_accuracy``, ``<prefix>_sd_accuracy``,
    ``plvm_minus_<prefix>`` and ``plvm_minus_<prefix>_wilcoxon_p`` (prefix from
    ``extra_condition_prefix``).
    """
    training_logs_root = Path(training_logs_root)
    output_rows: List[Dict[str, object]] = []

    for target_mode in target_modes:
        base_rows = _load_base_rows(training_logs_root, family_suffix, seq_length, target_mode)
        if not base_rows:
            raise ValueError(
                "No candidate runs found for target_mode='{}' with family_suffix='{}'.".format(
                    target_mode, family_suffix
                )
            )
        condition_meta, rows_by_condition_fold = select_all_rows_by_fold(base_rows)
        fold_frame = build_fold_long_frame(training_logs_root, rows_by_condition_fold)
        selected = _best_condition_by_family(fold_frame, condition_meta)
        selected.update(FIXED_SUMMARY_CONDITIONS.get(str(target_mode), {}))

        plvm_condition = selected["PLVM"]
        single_condition = selected["Single-task process"]
        outcome_condition = selected.get("Outcome baseline")
        if outcome_condition is None:
            raise ValueError(
                "No Outcome-baseline (rf_baseline) runs found for target_mode='{}' "
                "with family_suffix='{}'.".format(target_mode, family_suffix)
            )

        logistic_rows = _load_logistic_rows_for_target(
            training_logs_root, family_suffix, seq_length, target_mode
        )
        logistic_selection = select_best_logistic_condition(training_logs_root, logistic_rows)
        logistic_condition = logistic_selection[0] if logistic_selection else None

        extra_conditions = [c for c in extra_single_conditions if c != single_condition]
        missing_extra = [c for c in extra_conditions if c not in rows_by_condition_fold]
        if missing_extra:
            raise ValueError(
                "No runs for extra single-task condition(s) {} (target_mode='{}', family_suffix='{}').".format(
                    missing_extra, target_mode, family_suffix
                )
            )
        selected_rows = {
            plvm_condition: rows_by_condition_fold[plvm_condition],
            single_condition: rows_by_condition_fold[single_condition],
            outcome_condition: rows_by_condition_fold[outcome_condition],
        }
        for condition in extra_conditions:
            selected_rows[condition] = rows_by_condition_fold[condition]
        if logistic_selection is not None:
            selected_rows[logistic_condition] = logistic_selection[1]
        comparison_frame = build_fold_long_frame(training_logs_root, selected_rows)

        shared_conditions = [plvm_condition, single_condition, outcome_condition]
        if logistic_condition is not None:
            shared_conditions.append(logistic_condition)
        shared_conditions.extend(extra_conditions)

        for cutoff_minutes in SUMMARY_TIMEPOINT_MINUTES:
            shared = _shared_cutoff_rows(
                comparison_frame, conditions=shared_conditions, cutoff_minutes=float(cutoff_minutes)
            )
            if shared.empty:
                continue

            row: Dict[str, object] = {
                "target_mode": str(target_mode),
                "target_label": _target_label(str(target_mode)),
                "cutoff_minutes": int(cutoff_minutes),
                "n_shared_folds": int(len(shared)),
                "plvm_condition": plvm_condition,
                "single_task_condition": single_condition,
                "outcome_condition": outcome_condition,
                "logistic_condition": logistic_condition if logistic_condition is not None else "",
                "plvm_mean_accuracy": float(shared[plvm_condition].mean()),
                "plvm_sd_accuracy": float(shared[plvm_condition].std(ddof=1)),
                "single_task_mean_accuracy": float(shared[single_condition].mean()),
                "single_task_sd_accuracy": float(shared[single_condition].std(ddof=1)),
                "outcome_mean_accuracy": float(shared[outcome_condition].mean()),
                "outcome_sd_accuracy": float(shared[outcome_condition].std(ddof=1)),
                "plvm_minus_single_task": float(
                    shared[plvm_condition].mean() - shared[single_condition].mean()
                ),
                "plvm_minus_single_task_wilcoxon_p": _wilcoxon_pvalue(
                    shared[plvm_condition], shared[single_condition]
                ),
                "plvm_minus_outcome": float(
                    shared[plvm_condition].mean() - shared[outcome_condition].mean()
                ),
                "plvm_minus_outcome_wilcoxon_p": _wilcoxon_pvalue(
                    shared[plvm_condition], shared[outcome_condition]
                ),
                "logistic_minus_single_task": float("nan"),
                "logistic_minus_single_task_wilcoxon_p": float("nan"),
                "plvm_minus_logistic": float("nan"),
                "plvm_minus_logistic_wilcoxon_p": float("nan"),
            }
            if logistic_condition is not None:
                row["logistic_mean_accuracy"] = float(shared[logistic_condition].mean())
                row["logistic_sd_accuracy"] = float(shared[logistic_condition].std(ddof=1))
                row["plvm_minus_logistic"] = float(
                    shared[plvm_condition].mean() - shared[logistic_condition].mean()
                )
                row["plvm_minus_logistic_wilcoxon_p"] = _wilcoxon_pvalue(
                    shared[plvm_condition], shared[logistic_condition]
                )
                row["logistic_minus_single_task"] = float(
                    shared[logistic_condition].mean() - shared[single_condition].mean()
                )
                row["logistic_minus_single_task_wilcoxon_p"] = _wilcoxon_pvalue(
                    shared[logistic_condition], shared[single_condition]
                )
            else:
                row["logistic_mean_accuracy"] = float("nan")
                row["logistic_sd_accuracy"] = float("nan")

            row["shared_folds"] = ",".join(shared["fold_id"].astype(str).tolist())
            for condition in extra_conditions:
                prefix = extra_condition_prefix(condition)
                row["{}_condition".format(prefix)] = condition
                row["{}_mean_accuracy".format(prefix)] = float(shared[condition].mean())
                row["{}_sd_accuracy".format(prefix)] = float(shared[condition].std(ddof=1))
                row["plvm_minus_{}".format(prefix)] = float(
                    shared[plvm_condition].mean() - shared[condition].mean()
                )
                row["plvm_minus_{}_wilcoxon_p".format(prefix)] = _wilcoxon_pvalue(
                    shared[plvm_condition], shared[condition]
                )
            output_rows.append(row)

    if not output_rows:
        raise ValueError("No shared-fold summary rows could be built.")
    return pd.DataFrame(output_rows)


def _format_percent(value: float) -> str:
    """Format one accuracy-like value as a percent with one decimal place, or N/A."""
    if value is None or pd.isna(value):
        return "N/A"
    return "{:.1f}".format(float(value) * 100.0)


def _format_mean_sd(mean_value: float, sd_value: float) -> str:
    """Format one mean/sd accuracy pair as percent text, or N/A when missing."""
    if mean_value is None or pd.isna(mean_value):
        return "N/A"
    return "{} +/- {}".format(
        _format_percent(mean_value), _format_percent(0.0 if pd.isna(sd_value) else sd_value)
    )


def _format_pvalue(value: float) -> str:
    """Format one p-value compactly, or N/A when missing."""
    if value is None or pd.isna(value):
        return "N/A"
    if float(value) < 1e-3:
        return "<0.001"
    return "{:.3f}".format(float(value))


def _format_diff_with_pvalue(diff_value: float, pvalue: float) -> str:
    """Format one mean difference together with its Wilcoxon p-value, or N/A when missing."""
    if diff_value is None or pd.isna(diff_value):
        return "N/A"
    return "{} (p={})".format(_format_percent(diff_value), _format_pvalue(pvalue))


def render_markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact markdown table without external dependencies."""
    headers = [
        "Target",
        "Cutoff (min)",
        "Shared folds",
        "Single-task",
        "Logistic",
        "Best Outcome",
        "PLVM",
        "PLVM-Single",
        "PLVM-Logistic",
        "PLVM-Outcome",
        "Logistic-Single",
    ]
    rows = [headers]
    for _, row in frame.iterrows():
        rows.append(
            [
                str(row["target_label"]),
                str(int(row["cutoff_minutes"])),
                str(int(row["n_shared_folds"])),
                _format_mean_sd(row["single_task_mean_accuracy"], row["single_task_sd_accuracy"]),
                _format_mean_sd(row["logistic_mean_accuracy"], row["logistic_sd_accuracy"]),
                _format_mean_sd(row["outcome_mean_accuracy"], row["outcome_sd_accuracy"]),
                _format_mean_sd(row["plvm_mean_accuracy"], row["plvm_sd_accuracy"]),
                _format_diff_with_pvalue(
                    row["plvm_minus_single_task"], row["plvm_minus_single_task_wilcoxon_p"]
                ),
                _format_diff_with_pvalue(
                    row["plvm_minus_logistic"], row["plvm_minus_logistic_wilcoxon_p"]
                ),
                _format_diff_with_pvalue(
                    row["plvm_minus_outcome"], row["plvm_minus_outcome_wilcoxon_p"]
                ),
                _format_diff_with_pvalue(
                    row["logistic_minus_single_task"], row["logistic_minus_single_task_wilcoxon_p"]
                ),
            ]
        )

    widths = [max(len(item) for item in column) for column in zip(*rows)]

    def render_line(items: Sequence[str]) -> str:
        cells = [
            " {item:<{width}} ".format(item=item, width=width) for item, width in zip(items, widths)
        ]
        return "|" + "|".join(cells) + "|"

    separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    lines = [render_line(rows[0]), separator]
    lines.extend(render_line(items) for items in rows[1:])
    return "\n".join(lines) + "\n"


def render_latex_table(frame: pd.DataFrame) -> str:
    """Render one target's summary as a \\multirow LaTeX tabular.

    Only the tabular is written, without a caption or label.
    """
    cutoffs = sorted(frame["cutoff_minutes"].unique())
    if len(cutoffs) != 3:
        raise ValueError("Expected exactly 3 cutoffs, found {}.".format(cutoffs))
    by_cutoff = {int(c): frame.loc[frame["cutoff_minutes"] == c].iloc[0] for c in cutoffs}
    c0, c1, c2 = cutoffs

    def acc_line(method: str, mean_col: str, sd_col: str) -> str:
        cells = " & ".join(
            _format_mean_sd(by_cutoff[int(c)][mean_col], by_cutoff[int(c)][sd_col]) for c in cutoffs
        )
        return "  & {}  & {} \\\\".format(method, cells)

    def delta_cell(diff_value: float, pvalue: float) -> str:
        if diff_value is None or pd.isna(diff_value):
            return "N/A"
        return "{}\\,($p{{=}}{}$)".format(_format_percent(diff_value), _format_pvalue(pvalue))

    def delta_line(label_text: str, diff_col: str, p_col: str) -> str:
        cells = " & ".join(
            delta_cell(by_cutoff[int(c)][diff_col], by_cutoff[int(c)][p_col]) for c in cutoffs
        )
        return "  & {}         & {} \\\\".format(label_text, cells)

    lines = [
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Metric & Method & {} min & {} min & {} min \\\\".format(int(c0), int(c1), int(c2)),
        "\\midrule",
        "\\multirow{4}{*}{Accuracy (\\%)}",
        acc_line("Single-task (Bungalow)", "single_task_mean_accuracy", "single_task_sd_accuracy"),
        acc_line("Logistic Fuser", "logistic_mean_accuracy", "logistic_sd_accuracy"),
        acc_line("Best Outcome", "outcome_mean_accuracy", "outcome_sd_accuracy"),
        acc_line("PLVM", "plvm_mean_accuracy", "plvm_sd_accuracy"),
        "\\midrule",
        "\\multirow{4}{*}{$\\Delta$ Acc. (pp)}",
        delta_line(
            "PLVM $-$ Single", "plvm_minus_single_task", "plvm_minus_single_task_wilcoxon_p"
        ),
        delta_line("PLVM $-$ Logistic", "plvm_minus_logistic", "plvm_minus_logistic_wilcoxon_p"),
        delta_line("PLVM $-$ Best Outcome", "plvm_minus_outcome", "plvm_minus_outcome_wilcoxon_p"),
        delta_line(
            "Logistic $-$ Single",
            "logistic_minus_single_task",
            "logistic_minus_single_task_wilcoxon_p",
        ),
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines) + "\n"
