#!/usr/bin/env python3
"""Collect every in-text (non-table) number into one JSON/Markdown file.

Computes nothing new: it reads the outputs of the generators that own each number
and records the reported (rounded) value, the unrounded value, ``source_script`` and
``source_file``. Run it last (``make_paper.py`` does); missing inputs are listed in
the output and make the script exit with status 1.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper.output_paths import resolve_tables_dir  # noqa: E402
from plvm import artifact_paths  # noqa: E402

# Table outputs live in the repository; the aggregated curve CSVs live in the artifact tree
# (``$PLVM_ARTIFACT_ROOT`` when set).
TABLES = resolve_tables_dir()
ANALYSIS = artifact_paths.MODEL_TRAINING_ANALYSIS_DIR
FIG06_CURVES = (
    ANALYSIS
    / "powerwash_hierarchical_performance"
    / "player_label_2_class_clean_zonehopping"
    / "zone_planning_seq120"
    / "detailed_figures"
    / "player_label_2_class_clean_zonehopping__best_curves.csv"
)
FIG07_CURVES = (
    ANALYSIS
    / "minigrid_foraging_cleaning_performance"
    / "label_prediction_seq200_players8000"
    / "minigrid_label_prediction_seq200_cutoff_metrics.csv"
)

# (source script, input file) per number group.
SOURCES = {
    "ladder_full": (
        "paper/tables/table14_feature_ladder.py",
        TABLES / "table14_feature_ladder" / "navstyle_ladder_by_fold_full.csv",
    ),
    "ladder_shuffled": (
        "paper/tables/table14_feature_ladder.py",
        TABLES / "table14_feature_ladder" / "navstyle_ladder_by_fold_shuffled.csv",
    ),
    "ladder_tests": (
        "paper/tables/table14_feature_ladder.py",
        TABLES / "table14_feature_ladder" / "navstyle_ladder_tests_full.csv",
    ),
    "icc": (
        "paper/tables/appD6_label_stability.py",
        TABLES / "appD6_label_stability" / "navstyle_3level_icc_median_cohort2187.csv",
    ),
    "agreement": (
        "paper/tables/appD6_label_stability.py",
        TABLES / "appD6_label_stability" / "navstyle_3level_label_agreement_median_cohort2187.csv",
    ),
    "carry_overall": (
        "paper/tables/appD6_carry_forward.py",
        TABLES / "appD6_carry_forward" / "navstyle_gated_overall_120min.csv",
    ),
    "carry_tests": (
        "paper/tables/appD6_carry_forward.py",
        TABLES / "appD6_carry_forward" / "navstyle_gated_tests_120min.csv",
    ),
    "latent": (
        "paper/tables/table10_robustness.py",
        TABLES / "table10_robustness" / "figure_5c_latent_space_metrics.csv",
    ),
    "table2": ("paper/tables/table02_prediction_summary.py", TABLES / "Table_2_values.csv"),
    "table8": ("paper/tables/table08_encoder_metrics.py", TABLES / "Table_8_values.csv"),
    "table9": (
        "paper/tables/table09_simulation_complementarity.py",
        TABLES / "Table_9_values.csv",
    ),
    "matched_head_pw": (
        "paper/tables/table09_matched_head_gain.py",
        TABLES / "cross_domain_combination_gain_analysis.csv",
    ),
    "table12_cells": (
        "paper/tables/table12_head_capacity_ablation.py",
        TABLES / "head_capacity_ablation_cells.csv",
    ),
    "table12_sig": (
        "paper/tables/table12_head_capacity_ablation.py",
        TABLES / "head_capacity_ablation_significance.csv",
    ),
    "fig06_curves": (
        "paper/figures/fig06_fig08_fig09_fig10_powerwash_accuracy_curves.py",
        FIG06_CURVES,
    ),
    "fig07_curves": ("paper/figures/fig07_simulation_accuracy_curves.py", FIG07_CURVES),
}
NOT_GENERATED = [
    {
        "section": "App. B.1 / Table 5",
        "quantity": "encoder players 6,012 (Back Garden) / 6,061 (Bungalow)",
        "reason": "not regenerated: count/hyperparameter table",
    },
    {
        "section": "App. A / Table 3 text",
        "quantity": "52 Fire Station completers lacked an earlier task; >98%",
        "reason": "not regenerated: count/hyperparameter table",
    },
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output_dir", type=Path, default=resolve_tables_dir(), help="Output directory."
    )
    return parser.parse_args(argv)


def _pct(value: float, decimals: int = 1, sign: bool = False) -> str:
    return ("{:+.{d}f}" if sign else "{:.{d}f}").format(float(value), d=decimals)


def _mean_sd(values: pd.Series) -> str:
    values = values.astype(float) * 100.0
    return "{:.1f} ± {:.1f}".format(values.mean(), values.std(ddof=1))


class Collector:
    """Accumulates number records; each group reads one generator output."""

    def __init__(self):
        self.numbers: List[Dict[str, object]] = []
        self.missing: List[Dict[str, str]] = []

    def group(self, key: str, build: Callable[[pd.DataFrame], List[Dict[str, object]]]) -> None:
        """Read the input of ``key`` and append the records returned by ``build``."""
        script, path = SOURCES[key]
        source_file = artifact_paths.display_path(path)
        if not path.exists():
            self.missing.append({"source_script": script, "source_file": source_file})
            return
        for record in build(pd.read_csv(path)):
            record.setdefault("note", "")
            self.numbers.append(dict(record, source_script=script, source_file=source_file))


def _num(
    nid: str, section: str, quantity: str, value: str, raw, note: str = ""
) -> Dict[str, object]:
    return {
        "id": nid,
        "section": section,
        "quantity": quantity,
        "value": value,
        "raw": raw,
        "note": note,
    }


def _ladder_at(df: pd.DataFrame, column: str, minutes: int) -> pd.Series:
    return df.loc[df["cutoff_seconds"] == minutes * 60, column].dropna()


def collect() -> Collector:
    """Build every in-text number from the generator outputs listed in ``SOURCES``."""
    c = Collector()

    # ---- Sec. 5.1: hand-crafted navigation statistics (Table 14 ladder) -------------------
    c.group(
        "ladder_full",
        lambda df: [
            _num(
                "sec5_1_nav_stats_lr_120min",
                "Sec. 5.1",
                "logistic regression on navigation statistics, 120 min (%)",
                _mean_sd(_ladder_at(df, "acc_P-order", 120)),
                float(_ladder_at(df, "acc_P-order", 120).mean()),
            )
        ],
    )
    c.group(
        "ladder_tests",
        lambda df: [
            _num(
                "sec5_1_nav_stats_vs_plvm_p_120min",
                "Sec. 5.1",
                "paired Wilcoxon p, navigation statistics vs PLVM, 120 min",
                "{:.2f}".format(r.wilcoxon_p),
                float(r.wilcoxon_p),
            )
            for r in df.itertuples()
            if r.lower == "P-order" and r.higher == "P-learned" and int(r.cutoff_seconds) == 7200
        ],
    )
    c.group(
        "ladder_shuffled",
        lambda df: [
            _num(
                "sec5_1_nav_stats_shuffled_120min",
                "Sec. 5.1",
                "navigation statistics with shuffled prefixes, 120 min (%)",
                _pct(100 * _ladder_at(df, "acc_P-order", 120).mean()),
                float(_ladder_at(df, "acc_P-order", 120).mean()),
            )
        ],
    )

    # ---- Sec. 5.2 / App. D.7: head capacity (Table 12) -------------------------------------
    def table12_cells(df):
        sim = df.loc[(df["domain"] == "Simulation") & (df["cutoff_value"].astype(float) == 200.0)]
        acc = {(r.source_tasks, r.head): float(r.accuracy_percent) for r in sim.itertuples()}
        single_gap = acc[("Single", "Nonlinear")] - acc[("Single", "Linear")]
        both_gap = acc[("Both (PLVM)", "Nonlinear")] - acc[("Both (PLVM)", "Linear")]
        return [
            _num(
                "sec5_2_linear_two_task_accuracy_sim200",
                "Sec. 5.2",
                "two-task linear-head accuracy, simulation 200 steps (%)",
                _pct(acc[("Both (PLVM)", "Linear")]),
                acc[("Both (PLVM)", "Linear")] / 100.0,
            ),
            _num(
                "appD7_linear_to_nonlinear_single_sim200",
                "App. D.7",
                "linear -> nonlinear head gain, stronger single task, simulation 200 steps (pp, approx.)",
                "{:.0f}".format(single_gap),
                single_gap,
            ),
            _num(
                "appD7_linear_to_nonlinear_two_task_sim200",
                "App. D.7",
                "linear -> nonlinear head gain, two-task model, simulation 200 steps (pp, approx.)",
                "{:.0f}".format(both_gap),
                both_gap,
            ),
        ]

    c.group("table12_cells", table12_cells)

    # ``sec5_2_powerwash_nonlinear_second_task_gain_120min`` is Table 12's nonlinear-head
    # second-task gain (PLVM-2 minus the Bungalow single-task MLP), reported in Section 5.2
    # as the PowerWash matched-head gain. It is a different quantity from the Table 9
    # matched-head gain recorded under ``matched_head_pw`` below (two-task concat MLP minus
    # the best single-task MLP).
    def table12_sig(df):
        sim = df.loc[(df["domain"] == "Simulation") & (df["column_label"] == "200 steps")].iloc[0]
        pw = df.loc[(df["domain"] == "Fire Station") & (df["column_label"] == "120 min")].iloc[0]
        return [
            _num(
                "sec5_2_linear_two_task_gain_sim200",
                "Sec. 5.2",
                "gain from the second task, linear head, simulation 200 steps (pp)",
                _pct(sim.linear_gain, sign=True),
                float(sim.linear_gain),
            ),
            _num(
                "sec5_2_nonlinear_two_task_gain_sim200",
                "Sec. 5.2",
                "gain from the second task, nonlinear head, simulation 200 steps (pp)",
                _pct(sim.nonlinear_gain, sign=True),
                float(sim.nonlinear_gain),
            ),
            _num(
                "sec5_2_powerwash_interaction_120min",
                "Sec. 5.2",
                "PowerWash head x source interaction, Fire Station 120 min (pp)",
                _pct(pw.interaction, sign=True),
                float(pw.interaction),
                "Wilcoxon p = {:.3f}".format(pw.interaction_p),
            ),
            _num(
                "sec5_2_powerwash_nonlinear_second_task_gain_120min",
                "Sec. 5.2",
                "PowerWash gain from the second task, nonlinear heads (PLVM-2 - single task Bungalow), 120 min (pp)",
                _pct(pw.nonlinear_gain, sign=True),
                float(pw.nonlinear_gain),
                "Wilcoxon p = {:.3f}; the number the text quotes as the PowerWash matched-head gain".format(
                    pw.nonlinear_gain_p
                ),
            ),
        ]

    c.group("table12_sig", table12_sig)

    # ---- Sec. 5.2: matched-head gains (Table 9 column 4 and the PowerWash concat MLP) ------
    def table9(df):
        rows = df.set_index("task_relationship")
        comp, red, weak = (
            rows.loc["Complementary tasks"],
            rows.loc["Redundant tasks"],
            rows.loc["Weak shared structure"],
        )
        out = [
            _num(
                "sec5_2_matched_head_gain_complementary",
                "Sec. 5.2",
                "matched-head gain, complementary tasks (pp)",
                _pct(100 * comp.matched_head_gain, sign=True),
                float(comp.matched_head_gain),
            ),
            _num(
                "sec5_2_matched_head_single_complementary",
                "Sec. 5.2",
                "best single-task MLP accuracy, complementary tasks (%)",
                _pct(100 * comp.matched_head_best_single_mlp_accuracy),
                float(comp.matched_head_best_single_mlp_accuracy),
            ),
            _num(
                "sec5_2_matched_head_two_task_complementary",
                "Sec. 5.2",
                "two-task concat-MLP accuracy, complementary tasks (%)",
                _pct(100 * comp.matched_head_two_task_mlp_accuracy),
                float(comp.matched_head_two_task_mlp_accuracy),
            ),
            _num(
                "sec5_2_matched_head_gain_redundant",
                "Sec. 5.2",
                "matched-head gain, redundant tasks (pp)",
                _pct(100 * red.matched_head_gain, sign=True),
                float(red.matched_head_gain),
            ),
            _num(
                "sec5_2_matched_head_single_redundant",
                "Sec. 5.2",
                "best single-task MLP accuracy, redundant tasks (%)",
                _pct(100 * red.matched_head_best_single_mlp_accuracy),
                float(red.matched_head_best_single_mlp_accuracy),
            ),
        ]
        weak_note = weak.matched_head_note if isinstance(weak.matched_head_note, str) else ""
        out.append(
            _num(
                "sec5_2_matched_head_gain_weak",
                "Sec. 5.2",
                "matched-head gain, weak shared structure (pp)",
                "not computed" if weak_note else _pct(100 * weak.matched_head_gain, sign=True),
                None if weak_note else float(weak.matched_head_gain),
                weak_note
                + (
                    "; the PLVM gain for this condition is {:+.1f}".format(100 * weak.plvm_gain)
                    if weak_note
                    else ""
                ),
            )
        )
        return out

    c.group("table9", table9)

    # Table 9 matched-head definition applied to PowerWash: two-task concat MLP minus the
    # best single-task MLP (``table09_matched_head_gain.py``).
    def matched_head_pw(df):
        row = df.loc[df["Setting"] == "PowerWash"].iloc[0]
        gain, p = re.match(r"([+\-][\d.]+) \(p=([\d.]+)\)", str(row["Delta_comp"])).groups()
        return [
            _num(
                "sec5_2_powerwash_concat_mlp_matched_head_gain_120min",
                "Sec. 5.2",
                "PowerWash two-task concat-MLP - best single-task MLP, Fire Station 120 min (pp)",
                gain,
                float(gain),
                "Wilcoxon p = {}; same matched-head definition as Table 9 column 4".format(p),
            )
        ]

    c.group("matched_head_pw", matched_head_pw)

    # ---- Sec. 5.2: simulation, PLVM best from N steps onward (Figure 7 curves) -------------
    def fig07(df):
        val = df.loc[df["split"] == "val"].pivot_table(
            index="cutoff_value", columns="condition", values="accuracy"
        )
        leader = val.idxmax(axis=1)
        cutoffs = sorted(val.index)
        first = next(
            cut
            for cut in cutoffs
            if all(leader[cc] == "Process: PLVM-2" for cc in cutoffs if cc >= cut)
        )
        return [
            _num(
                "sec5_2_sim_plvm_best_from_steps",
                "Sec. 5.2",
                "simulation: PLVM best of all models from this many observed steps onward",
                str(int(first)),
                int(first),
            )
        ]

    c.group("fig07_curves", fig07)

    # ---- App. C.2: encoder accuracy range (Table 8) ----------------------------------------
    def table8(df):
        sim = df.loc[df["domain"] == "simulation"]
        detail = ", ".join(
            "{} {} {:.1f}%".format(r.task, r.split, 100 * r.value) for r in sim.itertuples()
        )
        return [
            _num(
                "appC2_simulation_encoder_accuracy_range",
                "App. C.2 / Table 8",
                "simulation encoder next-action accuracy range (val and train splits)",
                "{:.0f}-{:.0f}%".format(100 * sim.value.min(), 100 * sim.value.max()),
                {"min": float(sim.value.min()), "max": float(sim.value.max())},
                detail,
            )
        ]

    c.group("table8", table8)

    # ---- App. D.2: qualitative curve statements (Figure 6 curves) --------------------------
    def fig06(df):
        val = (
            df.loc[df["split"] == "val"]
            .groupby(["cutoff_value", "condition"])["accuracy"]
            .mean()
            .unstack()
        )
        early = [c for c in val.index if c <= 1800]
        bg_ahead = [
            int(c // 60)
            for c in early
            if val.loc[c, "Back Garden Alone"] > val.loc[c, "Bungalow Alone"]
        ]
        bung_ahead = [int(c // 60) for c in early if int(c // 60) not in bg_ahead]
        process_ahead = (
            val[["Bungalow Alone", "Back Garden Alone", "PLVM-2"]].min(axis=1) > val["RF Both"]
        )
        cutoffs = sorted(val.index)
        onward = next(c for c in cutoffs if all(process_ahead[cc] for cc in cutoffs if cc >= c))
        return [
            _num(
                "appD2_back_garden_vs_bungalow_early",
                "App. D.2",
                "Back Garden single task ahead of Bungalow below ~30 min",
                "Back Garden ahead at {} min; Bungalow ahead or tied at {} min".format(
                    bg_ahead, bung_ahead
                ),
                {"back_garden_ahead_min": bg_ahead, "bungalow_ahead_or_tied_min": bung_ahead},
            ),
            _num(
                "appD2_process_beats_outcome_from_min",
                "App. D.2",
                "every process model above the outcome baseline (RF Both) from this cutoff onward (min)",
                str(int(onward // 60)),
                int(onward // 60),
            ),
        ]

    c.group("fig06_curves", fig06)

    # ---- Figure 5c caption ------------------------------------------------------------------
    c.group(
        "latent",
        lambda df: [
            _num(
                "fig5c_linear_probe_auroc",
                "Figure 5c",
                "held-out linear-probe AUROC in the 2-D PLVM latent space (30 min)",
                "{:.3f}".format(df.iloc[0].linear_probe_cv_auc),
                float(df.iloc[0].linear_probe_cv_auc),
                "run {}".format(df.iloc[0].selected_model_name),
            ),
            _num(
                "fig5c_silhouette",
                "Figure 5c",
                "silhouette score of the two classes in the latent space (30 min)",
                "{:.3f}".format(df.iloc[0].silhouette_score),
                float(df.iloc[0].silhouette_score),
            ),
        ],
    )

    # ---- Table 2 text: majority-class rates --------------------------------------------------
    c.group(
        "table2",
        lambda df: [
            _num(
                "table2_majority_rate_{}min".format(int(r.metric.split()[0])),
                "Table 2 / Sec. 5.1",
                "majority-class rate of the held-out players, {} (%)".format(
                    r.metric.replace(" accuracy", "")
                ),
                _pct(100 * r.mean),
                float(r.mean),
            )
            for r in df.itertuples()
            if r.kind == "majority"
        ],
    )

    # ---- App. D.6: cross-level stability (n = 2,187 cohort) and carry-forward ----------------
    def icc(df):
        r = df.iloc[0]
        return [
            _num(
                "appD6_icc31",
                "App. D.6",
                "ICC(3,1) of the navigation-style score across the three levels",
                "{:.3f}".format(r.ICC3_1_consistency_single),
                float(r.ICC3_1_consistency_single),
            ),
            _num(
                "appD6_icc31_ci95",
                "App. D.6",
                "ICC(3,1) 95% CI",
                "[{:.3f}, {:.3f}]".format(r.ICC3_1_ci_lo, r.ICC3_1_ci_hi),
                [float(r.ICC3_1_ci_lo), float(r.ICC3_1_ci_hi)],
            ),
            _num(
                "appD6_cohort_n",
                "App. D.6",
                "players in the stability analysis (final PLVM cohort)",
                "{:,}".format(int(r.n_players)),
                int(r.n_players),
            ),
        ]

    c.group("icc", icc)

    def agreement(df):
        r = df.iloc[0]
        return [
            _num(
                "appD6_fleiss_kappa",
                "App. D.6",
                "Fleiss kappa of the per-level median-split labels",
                "{:.3f}".format(r.fleiss_kappa),
                float(r.fleiss_kappa),
            ),
            _num(
                "appD6_same_label_all_levels_pct",
                "App. D.6",
                "players with the same label on all three levels (%)",
                _pct(r.unanimous_3_0_pct),
                float(r.unanimous_3_0_pct),
            ),
            _num(
                "appD6_chance_same_label_pct",
                "App. D.6",
                "chance rate of the same label on all three levels (%)",
                "{:.0f}".format(r.chance_unanimous_pct),
                float(r.chance_unanimous_pct),
            ),
        ]

    c.group("agreement", agreement)

    def carry_overall(df):
        rows = df.set_index("predictor")
        bg, plvm = rows.loc["Observed-label transfer (Back Garden)"], rows.loc["PLVM"]
        return [
            _num(
                "appD6_carry_forward_back_garden_120min",
                "App. D.6",
                "carry-forward of the Back Garden label, 120 min (%)",
                "{:.1f} ± {:.1f}".format(100 * bg.acc_mean, 100 * bg.acc_sd),
                float(bg.acc_mean),
            ),
            _num(
                "appD6_plvm_same_folds_120min",
                "App. D.6",
                "PLVM on the same folds, 120 min (%)",
                "{:.1f} ± {:.1f}".format(100 * plvm.acc_mean, 100 * plvm.acc_sd),
                float(plvm.acc_mean),
            ),
        ]

    c.group("carry_overall", carry_overall)
    c.group(
        "carry_tests",
        lambda df: [
            _num(
                "appD6_carry_forward_vs_plvm_p",
                "App. D.6",
                "paired Wilcoxon p, PLVM vs Back Garden carry-forward",
                "{:.2f}".format(r.wilcoxon_p),
                float(r.wilcoxon_p),
            )
            for r in df.itertuples()
            if r.comparison == "PLVM vs transfer (Back Garden)"
        ],
    )
    return c


def render_markdown(numbers: List[Dict[str, object]], missing: List[Dict[str, str]]) -> str:
    """Render the collected numbers as a Markdown table."""
    lines = [
        "# In-text numbers",
        "",
        "Generated by `paper/tables/in_text_numbers.py` from the generator outputs.",
        "",
        "| Section | Quantity | Value | Source script | Note |",
        "|---|---|---|---|---|",
    ]
    for n in numbers:
        lines.append(
            "| {} | {} | {} | `{}` | {} |".format(
                n["section"], n["quantity"], n["value"], n["source_script"], n["note"]
            )
        )
    lines += ["", "Not generated:", ""]
    lines += [
        "- {}: {} ({})".format(n["section"], n["quantity"], n["reason"]) for n in NOT_GENERATED
    ]
    if missing:
        lines += ["", "Missing inputs (run their generators first):", ""]
        lines += ["- `{}` from `{}`".format(m["source_file"], m["source_script"]) for m in missing]
    return "\n".join(lines) + "\n"


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(type(value))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Collect the numbers and write in_text_numbers.json / .md."""
    args = parse_args(argv)
    collected = collect()
    payload = {
        "generator": "paper/tables/in_text_numbers.py",
        "numbers": collected.numbers,
        "not_generated": NOT_GENERATED,
        "missing_inputs": collected.missing,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "in_text_numbers.json"
    md_path = output_dir / "in_text_numbers.md"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(collected.numbers, collected.missing), encoding="utf-8")
    print("Wrote {}".format(json_path))
    print("Wrote {}".format(md_path))
    for n in collected.numbers:
        print("{:<55s} {}".format(n["id"], n["value"]))
    for m in collected.missing:
        print(
            "MISSING input {} (from {})".format(m["source_file"], m["source_script"]),
            file=sys.stderr,
        )
    return 1 if collected.missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
