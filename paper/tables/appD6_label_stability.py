#!/usr/bin/env python3
"""
appD6_label_stability.py
========================
Generates the App. D.6 cross-level stability table: how consistent ZoneHoppingScore
and its 2-class label are across Back Garden, Bungalow and Fire Station.

Treats the three levels as repeated measurements per player: ICCs, Cronbach's alpha,
variance decomposition and Spearman-Brown for the score; Fleiss' and Light's kappa
and unanimous-vs-split concordance for the label. A low ICC bounds how well any model
can predict one level from the others, but cannot separate a level-specific trait
from a noisy single-level measurement.

Reads ``navstyle_per_player_scores_3tasks.csv`` (from ``appD6_navigation_style_scores.py``)
and writes to ``manuscript_figures/tables/appD6_label_stability/``. The default cohort
is every player scored on all three levels; App. D.6 reports the PLVM cohort
(players in the selected 3-class PLVM-2 runs' validation predictions, all folds)::

    python paper/tables/appD6_label_stability.py --cohort plvm_3class --tag median_cohort2187
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper.output_paths import resolve_tables_dir  # noqa: E402
from plvm import artifact_paths  # noqa: E402

DEFAULT_IN = artifact_paths.NAVSTYLE_PER_PLAYER_SCORES_CSV
DEFAULT_OUT = resolve_tables_dir() / "appD6_label_stability"
LEVELS = ("back_garden", "bungalow", "fire_station")
COHORT_TARGET_MODES = {"plvm_3class": "player_label_3_class_zonehopping"}


def plvm_cohort_players(target_mode: str) -> set:
    """Players in the validation predictions of the selected PLVM-2 runs (union over folds).

    Uses the Figure 6 / Table 2 run selection (``select_all_rows_by_fold``), so the cohort is
    exactly the set of players the PLVM was evaluated on for ``target_mode``.
    """
    from paper.figures.fig06_fig08_fig09_fig10_powerwash_accuracy_curves import (
        DEFAULT_PERF_CSV,
        TRAINING_LOGS_DIR,
        load_candidate_rows,
        select_all_rows_by_fold,
    )
    from powerwash.train.powerwash_hierarchical_log_utils import (
        resolve_powerwash_hierarchical_run_dir,
    )

    rows = load_candidate_rows(Path(DEFAULT_PERF_CSV), seq_length=120, target_mode=target_mode)
    _, rows_by_condition_fold = select_all_rows_by_fold(rows)
    players = set()
    for _fold_id, row in sorted(rows_by_condition_fold["PLVM-2"].items()):
        run_dir = resolve_powerwash_hierarchical_run_dir(
            Path(TRAINING_LOGS_DIR), str(row["model_name"])
        )
        predictions = pd.read_csv(run_dir / "val_predictions.csv", usecols=["player_id"])
        players |= set(predictions["player_id"].astype(str))
    return players


def icc_table(x: np.ndarray) -> dict:
    """Two-way ICCs from the subjects x raters matrix `x` (n players, k levels)."""
    n, k = x.shape
    grand = x.mean()
    ms_rows = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)  # between players
    ms_cols = n * ((x.mean(axis=0) - grand) ** 2).sum() / (k - 1)  # between levels
    resid = x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + grand
    ms_err = (resid**2).sum() / ((n - 1) * (k - 1))

    icc31 = (ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err)
    icc3k = (ms_rows - ms_err) / ms_rows
    icc21 = (ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n)
    icc2k = (ms_rows - ms_err) / (ms_rows + (ms_cols - ms_err) / n)

    # F test / CI for the consistency ICC(3,1)
    f_val = ms_rows / ms_err
    df1, df2 = n - 1, (n - 1) * (k - 1)
    f_lo = f_val / stats.f.ppf(0.975, df1, df2)
    f_hi = f_val / stats.f.ppf(0.025, df1, df2)
    ci_lo = (f_lo - 1) / (f_lo + (k - 1))
    ci_hi = (f_hi - 1) / (f_hi + (k - 1))

    return {
        "n_players": int(n),
        "n_levels": int(k),
        "ICC3_1_consistency_single": float(icc31),
        "ICC3_1_ci_lo": float(ci_lo),
        "ICC3_1_ci_hi": float(ci_hi),
        "ICC3_k_consistency_average": float(icc3k),
        "ICC2_1_agreement_single": float(icc21),
        "ICC2_k_agreement_average": float(icc2k),
        "F": float(f_val),
        "df1": int(df1),
        "df2": int(df2),
        "p": float(stats.f.sf(f_val, df1, df2)),
        "ms_between_players": float(ms_rows),
        "ms_between_levels": float(ms_cols),
        "ms_error": float(ms_err),
    }


def fleiss_kappa(labels: np.ndarray) -> float:
    """Fleiss' kappa for an n x k matrix of categorical labels (k raters)."""
    cats = np.unique(labels)
    n, k = labels.shape
    counts = np.stack([(labels == c).sum(axis=1) for c in cats], axis=1)
    p_i = ((counts**2).sum(axis=1) - k) / (k * (k - 1))
    p_bar = p_i.mean()
    p_e = ((counts.sum(axis=0) / (n * k)) ** 2).sum()
    return float((p_bar - p_e) / (1 - p_e))


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa between two label vectors."""
    cats = np.unique(np.concatenate([a, b]))
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return float((po - pe) / (1 - pe))


def build_level_labels(d: pd.DataFrame, variant: str, lower_q: float, upper_q: float):
    """Label each level independently and return the (n players x 3) matrix.

    ``median`` splits every player at the per-level median (the 2-class label).
    ``clean`` matches the ``*_2_class_clean_*`` PLVM target: the middle band between
    ``lower_q`` and ``upper_q`` is discarded, and a player is kept only when all three
    levels place them outside it.
    """
    if variant == "median":
        lab = np.stack(
            [
                np.where(
                    d[f"zone_hopping_score_{t}"] >= d[f"zone_hopping_score_{t}"].median(), "H", "P"
                )
                for t in LEVELS
            ],
            axis=1,
        )
        return lab, np.ones(len(d), dtype=bool)

    if variant != "clean":
        raise ValueError(f"unknown label_variant {variant!r}")

    cols = []
    keep = np.ones(len(d), dtype=bool)
    for t in LEVELS:
        s = d[f"zone_hopping_score_{t}"]
        lo, hi = s.quantile(lower_q), s.quantile(upper_q)
        col = np.where(s >= hi, "H", np.where(s <= lo, "P", ""))
        keep &= col != ""
        cols.append(col)
    lab = np.stack(cols, axis=1)[keep]
    return lab, keep


def chance_unanimous_pct(labels: np.ndarray) -> float:
    """Unanimity expected if the three levels were independent.

    Computed from each level's own marginals rather than assumed to be 25%:
    the clean variant's retained subset need not be balanced, so a fixed
    constant would misstate the baseline it is compared against.
    """
    cats = np.unique(labels)
    p = 0.0
    for c in cats:
        p += float(np.prod([(labels[:, j] == c).mean() for j in range(labels.shape[1])]))
    return 100.0 * p


def spearman_brown_needed(r1: float, target: float) -> float:
    """Number of parallel levels needed to reach `target` reliability."""
    if r1 <= 0 or r1 >= 1:
        return float("nan")
    return float(target * (1 - r1) / (r1 * (1 - target)))


def main(argv=None):
    """Compute the cross-level stability statistics and write them to ``--out_dir``."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in_csv",
        type=Path,
        default=DEFAULT_IN,
        help="Per-player, per-level navigation-style scores and labels.",
    )
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT, help="Output directory.")
    ap.add_argument(
        "--label_variant",
        choices=["median", "clean"],
        default="median",
        help="median = 2-class label; clean = the 2-class-clean target label "
        "(ambiguous middle band dropped).",
    )
    ap.add_argument(
        "--lower_quantile",
        type=float,
        default=0.35,
        help="Lower edge of the discarded middle band (clean variant only).",
    )
    ap.add_argument(
        "--upper_quantile",
        type=float,
        default=0.65,
        help="Upper edge of the discarded middle band (clean variant only).",
    )
    ap.add_argument("--tag", default="", help="Suffix for output filenames.")
    ap.add_argument(
        "--cohort",
        choices=["all"] + sorted(COHORT_TARGET_MODES),
        default="all",
        help="all = every player scored on the three levels; plvm_3class = the "
        "PLVM cohort (players evaluated by the 3-class PLVM-2 runs), "
        "the population reported in App. D.6.",
    )
    args = ap.parse_args(argv)

    d = pd.read_csv(args.in_csv)
    if args.cohort != "all":
        # Restrict before any statistic (including the per-level median split) is computed.
        cohort = plvm_cohort_players(COHORT_TARGET_MODES[args.cohort])
        d = d.loc[d["player_id"].astype(str).isin(cohort)].reset_index(drop=True)
    score_cols = [f"zone_hopping_score_{t}" for t in LEVELS]
    d = d.dropna(subset=score_cols)
    x = d[score_cols].to_numpy(dtype=float)

    icc = icc_table(x)
    # Cronbach's alpha of the 3-level composite
    k = x.shape[1]
    alpha = float(k / (k - 1) * (1 - x.var(axis=0, ddof=1).sum() / x.sum(axis=1).var(ddof=1)))
    icc["cronbach_alpha_3_level_composite"] = alpha

    # variance decomposition (between-player vs within-player across levels)
    between = x.mean(axis=1).var(ddof=1)
    within = x.var(axis=1, ddof=1).mean()
    icc["var_between_players"] = float(between)
    icc["var_within_player_across_levels"] = float(within)
    icc["pct_variance_between_players"] = float(100 * between / (between + within))

    r1 = icc["ICC3_1_consistency_single"]
    icc["levels_needed_for_reliability_0.70"] = spearman_brown_needed(r1, 0.70)
    icc["levels_needed_for_reliability_0.80"] = spearman_brown_needed(r1, 0.80)

    # pairwise correlations for context
    pair_rows = []
    for a, b in itertools.combinations(LEVELS, 2):
        xa, xb = d[f"zone_hopping_score_{a}"], d[f"zone_hopping_score_{b}"]
        pr, pp = stats.pearsonr(xa, xb)
        sr, sp = stats.spearmanr(xa, xb)
        pair_rows.append(
            {
                "level_a": a,
                "level_b": b,
                "n": len(d),
                "pearson_r": pr,
                "pearson_p": pp,
                "spearman_rho": sr,
                "spearman_p": sp,
            }
        )
    pairs = pd.DataFrame(pair_rows)
    icc["mean_pairwise_pearson_r"] = float(pairs.pearson_r.mean())

    # ---- label-level stability -------------------------------------------
    lab, _ = build_level_labels(d, args.label_variant, args.lower_quantile, args.upper_quantile)
    n_all = lab.shape[0]
    n_unanimous = int(((lab[:, 0] == lab[:, 1]) & (lab[:, 1] == lab[:, 2])).sum())
    lab_stats = {
        "label_variant": args.label_variant,
        "n_players": n_all,
        "n_players_dropped_ambiguous": int(len(d) - n_all),
        "fleiss_kappa": fleiss_kappa(lab),
        "lights_kappa_mean_pairwise": float(
            np.mean(
                [cohen_kappa(lab[:, i], lab[:, j]) for i, j in itertools.combinations(range(3), 2)]
            )
        ),
        "unanimous_3_0_count": n_unanimous,
        "unanimous_3_0_pct": float(100 * n_unanimous / n_all),
        "split_2_1_pct": float(100 * (n_all - n_unanimous) / n_all),
        "chance_unanimous_pct": chance_unanimous_pct(lab),
    }
    for i, j in itertools.combinations(range(3), 2):
        lab_stats[f"cohen_kappa_{LEVELS[i]}_vs_{LEVELS[j]}"] = cohen_kappa(lab[:, i], lab[:, j])

    tag = f"_{args.tag}" if args.tag else ""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.out_dir / f"navstyle_3level_pairwise{tag}.csv", index=False)
    pd.DataFrame([icc]).to_csv(args.out_dir / f"navstyle_3level_icc{tag}.csv", index=False)
    pd.DataFrame([lab_stats]).to_csv(
        args.out_dir / f"navstyle_3level_label_agreement{tag}.csv", index=False
    )
    with (args.out_dir / f"navstyle_3level_stability{tag}.json").open("w") as fh:
        json.dump({"score_icc": icc, "label_agreement": lab_stats}, fh, indent=2)

    # ---- console report ---------------------------------------------------
    print("=" * 72)
    print(f"THREE-LEVEL STABILITY OF NAVIGATION STYLE   (n = {icc['n_players']} players)")
    print("=" * 72)
    print("\n-- continuous ZoneHoppingScore --")
    print(f"  mean pairwise Pearson r            : {icc['mean_pairwise_pearson_r']:.3f}")
    print(
        f"  ICC(3,1) single level  [95% CI]    : {r1:.3f} "
        f"[{icc['ICC3_1_ci_lo']:.3f}, {icc['ICC3_1_ci_hi']:.3f}]"
    )
    print(f"  ICC(3,k) 3-level average           : {icc['ICC3_k_consistency_average']:.3f}")
    print(f"  Cronbach alpha (3-level composite) : {alpha:.3f}")
    print(f"  variance between players           : {icc['pct_variance_between_players']:.1f}%")
    print(f"  levels needed for reliability 0.80 : {icc['levels_needed_for_reliability_0.80']:.1f}")
    print(
        f"\n-- 2-class label ({args.label_variant} split per level, n={lab_stats['n_players']}) --"
    )
    print(f"  Fleiss kappa                       : {lab_stats['fleiss_kappa']:.3f}")
    print(f"  Light kappa (mean pairwise Cohen)  : {lab_stats['lights_kappa_mean_pairwise']:.3f}")
    print(
        f"  unanimous across all 3 levels      : {lab_stats['unanimous_3_0_pct']:.1f}% "
        f"(chance {lab_stats['chance_unanimous_pct']:.1f}%)"
    )
    print(f"  split 2-1                          : {lab_stats['split_2_1_pct']:.1f}%")
    print("\nCaveat: cross-level ICC cannot separate 'trait varies by level' from")
    print("'one level measures the trait noisily'; both depress it equally. A")
    print("within-level split-half reliability would be needed to disentangle them.")
    print(f"\nWrote artifacts to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
