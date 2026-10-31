#!/usr/bin/env python3
"""
appD6_carry_forward.py
======================
Generates the App. D.6 carry-forward baseline: PLVM-2 vs carrying the player's
Back Garden navigation-style label forward as the Fire Station prediction.

Uses each fold's selected PLVM-2 run (``val_predictions.csv`` at the chosen
cutoff, 120 min by default), restricted to players labelled on both Back Garden
and Bungalow. Reports per-fold accuracy, mean +/- SD and a paired per-fold
Wilcoxon test. Writes to ``manuscript_figures/tables/appD6_carry_forward/``.

Prerequisite: ``navstyle_per_player_scores_3tasks.csv`` from
``paper/tables/appD6_navigation_style_scores.py``, which needs the Bungalow labels::

    python powerwash/data/create_firestation_zone_planning_labels.py \
        --task bungalow --n_folds 1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper.figures.fig06_fig08_fig09_fig10_powerwash_accuracy_curves import (  # noqa: E402
    DEFAULT_PERF_CSV,
    TRAINING_LOGS_DIR,
    load_candidate_rows,
    select_all_rows_by_fold,
)
from paper.output_paths import resolve_tables_dir  # noqa: E402
from plvm import artifact_paths  # noqa: E402

TARGET_MODE = "player_label_2_class_clean_zonehopping"
PLVM_CONDITION = "PLVM-2"
DEFAULT_SCORES = artifact_paths.NAVSTYLE_PER_PLAYER_SCORES_CSV
DEFAULT_OUT = resolve_tables_dir() / "appD6_carry_forward"


def paired_p(a: pd.Series, b: pd.Series) -> float:
    """Paired Wilcoxon signed-rank p-value; NaN when the test is undefined."""
    try:
        return float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        return float("nan")


def build(cutoff_minutes: float, scores_csv: Path) -> pd.DataFrame:
    """Return one row per (fold, player) with the PLVM and carry-forward predictions."""
    if not scores_csv.exists():
        raise SystemExit(
            f"missing {scores_csv}\nRun paper/tables/appD6_navigation_style_scores.py first "
            "(see the module docstring for the bungalow-label prerequisite)."
        )
    sc = pd.read_csv(scores_csv)
    sc["player_id"] = sc["player_id"].astype(str)
    lab = pd.DataFrame(
        {
            "player_id": sc["player_id"],
            "lab_bg": sc["label_2_class_back_garden"].astype(str),
            "lab_bu": sc["label_2_class_bungalow"].astype(str),
        }
    )

    rows = load_candidate_rows(DEFAULT_PERF_CSV, seq_length=120, target_mode=TARGET_MODE)
    _meta, by_cond_fold = select_all_rows_by_fold(rows)
    if PLVM_CONDITION not in by_cond_fold:
        raise SystemExit(f"condition {PLVM_CONDITION} not found in training logs")

    cutoff_s = float(cutoff_minutes) * 60.0
    frames = []
    for fold_id, run in sorted(by_cond_fold[PLVM_CONDITION].items()):
        pred = Path(TRAINING_LOGS_DIR) / "plvm_models" / run["model_name"] / "val_predictions.csv"
        if not pred.exists():
            continue
        p = pd.read_csv(pred)
        p = p[np.isclose(p["observation_cutoff_seconds"].astype(float), cutoff_s)].copy()
        if p.empty:
            continue
        p["player_id"] = p["player_id"].astype(str)
        m = p.merge(lab, on="player_id", how="inner")
        # Population: players labelled on both observed levels (Bungalow is not a
        # predictor here, but this filter defines the reported cohort).
        m = m[m["lab_bg"].ne("nan") & m["lab_bu"].ne("nan")].copy()
        if m.empty:
            continue
        m["fold"] = fold_id
        m["pred_plvm"] = m["prediction_label"]
        m["pred_transfer_bg"] = m["lab_bg"]
        frames.append(m[["fold", "player_id", "target_label", "pred_plvm", "pred_transfer_bg"]])
    if not frames:
        raise SystemExit("no PLVM predictions found at the requested cutoff")
    return pd.concat(frames, ignore_index=True)


PREDICTORS = [("pred_transfer_bg", "Observed-label transfer (Back Garden)"), ("pred_plvm", "PLVM")]


def main(argv=None):
    """Compute per-fold carry-forward vs PLVM accuracy and write the CSV outputs."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cutoff_minutes",
        type=float,
        default=120.0,
        help="PLVM observation cutoff in minutes. Also used as the output tag.",
    )
    ap.add_argument(
        "--scores_csv",
        type=Path,
        default=DEFAULT_SCORES,
        help="Per-player navigation-style labels (from appD6_navigation_style_scores.py).",
    )
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT, help="Output directory.")
    args = ap.parse_args(argv)

    d = build(args.cutoff_minutes, args.scores_csv)
    tgt = d["target_label"].to_numpy()
    for col, _ in PREDICTORS:
        d[f"ok_{col}"] = d[col].to_numpy() == tgt

    # ---------- per-fold ----------------------------------------------------
    agg = {f"ok_{c}": (f"ok_{c}", "mean") for c, _ in PREDICTORS}
    by_fold = d.groupby("fold").agg(n=("player_id", "size"), **agg).reset_index()

    # ---------- overall (mean +- SD over folds) + paired test ---------------
    overall = []
    for col, label in PREDICTORS:
        v = by_fold[f"ok_{col}"]
        overall.append(
            {"predictor": label, "acc_mean": float(v.mean()), "acc_sd": float(v.std(ddof=1))}
        )
    overall = pd.DataFrame(overall)

    va, vb = by_fold["ok_pred_plvm"], by_fold["ok_pred_transfer_bg"]
    tests_df = pd.DataFrame(
        [
            {
                "comparison": "PLVM vs transfer (Back Garden)",
                "n_folds": int(len(by_fold)),
                "a_mean": float(va.mean()),
                "b_mean": float(vb.mean()),
                "diff_pp": float(100 * (va - vb).mean()),
                "wilcoxon_p": paired_p(va, vb),
            }
        ]
    )

    # ---------- write artifacts --------------------------------------------
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{int(args.cutoff_minutes)}min"
    by_fold.to_csv(out / f"navstyle_gated_by_fold_{tag}.csv", index=False)
    overall.to_csv(out / f"navstyle_gated_overall_{tag}.csv", index=False)
    tests_df.to_csv(out / f"navstyle_gated_tests_{tag}.csv", index=False)

    # ---------- console -----------------------------------------------------
    print("=" * 78)
    print(
        f"CARRY-FORWARD ANALYSIS | cutoff {args.cutoff_minutes:.0f} min | "
        f"{len(by_fold)} folds | {len(d)} player-fold rows"
    )
    print("=" * 78)
    print("\n-- overall (mean over folds) --")
    print(
        overall.assign(
            acc_mean=(100 * overall.acc_mean).round(1), acc_sd=(100 * overall.acc_sd).round(1)
        ).to_string(index=False)
    )
    print("\n-- paired per-fold Wilcoxon --")
    print(
        tests_df.assign(
            a_mean=(100 * tests_df.a_mean).round(1),
            b_mean=(100 * tests_df.b_mean).round(1),
            diff_pp=tests_df.diff_pp.round(1),
            wilcoxon_p=tests_df.wilcoxon_p.round(4),
        ).to_string(index=False)
    )
    print(f"Wrote 3 CSVs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
