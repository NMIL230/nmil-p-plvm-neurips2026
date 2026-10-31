#!/usr/bin/env python3
"""
appD6_navigation_style_scores.py
================================
Generates the App. D.6 input ``navstyle_per_player_scores_3tasks.csv`` (per-player,
per-level navigation-style scores and labels, read by ``appD6_label_stability.py`` and
``appD6_carry_forward.py``), plus cross-level correlations, Back Garden / Bungalow
label carry-over to Fire Station, and Fire Station fold invariance against fold_00.

ZoneHoppingScore is not recomputed: it is the persisted ``zone_hopping_score`` from
``powerwash/data/create_firestation_zone_planning_labels.py`` (which defines it), on
full, non-truncated traces. Writes to ``artifacts/powerwash/outputs/navigation_style/``.

Bungalow labels must exist first (written to ``bungalow_zone_labels/`` by default)::

    python powerwash/data/create_firestation_zone_planning_labels.py \
        --task bungalow --n_folds 1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plvm import artifact_paths  # noqa: E402

FS_LABEL_ROOT = artifact_paths.FIRESTATION_LABELS_DIR
BUNGALOW_LABEL_DIR = artifact_paths.BUNGALOW_LABELS_DIR
NAVSTYLE_OUT_DIR = artifact_paths.NAVIGATION_STYLE_DIR
BG_LABEL_CSV = artifact_paths.BACK_GARDEN_LABELS_CSV

# Quantiles of the 3-class / clean split (defaults of create_firestation_zone_planning_labels.py).
LOWER_Q, UPPER_Q = 0.35, 0.65

SCORE_FORMULA = (
    "ZoneHoppingScore = +robust_z(zone_switch_rate) +robust_z(return_rate) "
    "-robust_z(mean_zone_run_length) +robust_z(mean_zone_jump_distance)"
)


# ---------------------------------------------------------------- loading ---
def load_task_scores(csv_path: Path, task: str) -> pd.DataFrame:
    """Load one task's per-player score table, dropping excluded players."""
    df = pd.read_csv(csv_path)
    df["player_id"] = df["player_id"].astype(str)
    if "excluded" in df.columns:
        df = df.loc[~df["excluded"].astype(bool)]
    df = df.loc[df["zone_hopping_score"].notna()]
    keep = ["player_id", "zone_hopping_score"]
    for col in ("label", "label_2_class", "label_2_class_clean"):
        if col in df.columns:
            keep.append(col)
    out = df[keep].copy()
    out = out.drop_duplicates(subset="player_id", keep="first")
    return out.rename(columns={c: f"{c}_{task}" for c in keep if c != "player_id"})


def derive_two_class(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    Add `label_2_class_*` / `label_2_class_clean_*` when a task's CSV lacks them,
    using the zone-label rules on that task's own score distribution:
      - 2_class       : Zone Hopper iff score >= median
      - 2_class_clean : 3-class label with the Mixed middle band dropped
                        (bottom 35% Planner / top 35% Hopper)
    """
    s = df[f"zone_hopping_score_{task}"]
    if f"label_2_class_{task}" not in df.columns:
        df[f"label_2_class_{task}"] = np.where(s >= s.median(), "Zone Hopper", "Zone Planner")
    if f"label_2_class_clean_{task}" not in df.columns:
        lo, hi = s.quantile(LOWER_Q), s.quantile(UPPER_Q)
        df[f"label_2_class_clean_{task}"] = np.where(
            s <= lo, "Zone Planner", np.where(s >= hi, "Zone Hopper", "")
        )
    return df


# ------------------------------------------------------------ statistics ---
def fisher_ci(r: float, n: int, alpha: float = 0.05):
    """Fisher-z 95% CI for a correlation coefficient."""
    if n < 4 or not np.isfinite(r) or abs(r) >= 1.0:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    crit = stats.norm.ppf(1 - alpha / 2)
    return tuple(np.tanh([z - crit * se, z + crit * se]))


def correlate(x: pd.Series, y: pd.Series, task_a: str, task_b: str, subset: str):
    """Pearson and Spearman correlation of two score columns, with Fisher-z 95% CIs."""
    n = len(x)
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    plo, phi = fisher_ci(pr, n)
    slo, shi = fisher_ci(sr, n)
    return {
        "subset": subset,
        "task_a": task_a,
        "task_b": task_b,
        "n": n,
        "pearson_r": pr,
        "pearson_p": pp,
        "pearson_ci_lo": plo,
        "pearson_ci_hi": phi,
        "pearson_r2": pr**2,
        "spearman_rho": sr,
        "spearman_p": sp,
        "spearman_ci_lo": slo,
        "spearman_ci_hi": shi,
    }


def carryover(src: pd.Series, tgt: pd.Series, src_task: str, variant: str, fold: str):
    """Accuracy / kappa from predicting the fire_station label = source label."""
    mask = src.astype(str).ne("") & tgt.astype(str).ne("")
    s, t = src[mask].astype(str), tgt[mask].astype(str)
    n = len(s)
    if n == 0:
        return None
    acc = float((s == t).mean())
    kappa = float(cohen_kappa_score(s, t))
    # majority-class baseline on the fire_station target
    base = float(t.value_counts(normalize=True).max())
    # Wilson 95% CI on accuracy
    lo, hi = stats.binomtest(int(round(acc * n)), n).proportion_ci(method="wilson")
    return {
        "variant": variant,
        "fold": fold,
        "source_task": src_task,
        "target_task": "fire_station",
        "n": n,
        "carryover_accuracy": acc,
        "acc_ci_lo": float(lo),
        "acc_ci_hi": float(hi),
        "cohens_kappa": kappa,
        "target_majority_baseline": base,
    }


# ------------------------------------------------------------------ main ---
def main(argv=None):
    """Load the per-level labels, compute correlations and carry-over checks, write outputs."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bungalow_dir",
        type=Path,
        default=BUNGALOW_LABEL_DIR,
        help="Directory holding bungalow_zone_labels*.csv",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=NAVSTYLE_OUT_DIR,
        help="Output directory (navstyle_per_player_scores_3tasks.csv and the reports).",
    )
    args = ap.parse_args(argv)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- fold-invariance check on the saved fire_station labels ----
    fold_dirs = sorted(FS_LABEL_ROOT.glob("fold_[0-9][0-9]"))
    fs_by_fold = {
        d.name: load_task_scores(d / "firestation_zone_labels_3_class.csv", "fire_station")
        for d in fold_dirs
    }
    ref_name = sorted(fs_by_fold)[0]
    ref = fs_by_fold[ref_name].set_index("player_id")
    fold_rows = []
    for name, df in fs_by_fold.items():
        o = df.set_index("player_id").reindex(ref.index)
        fold_rows.append(
            {
                "fold": name,
                "n": int(len(o)),
                "max_abs_score_diff_vs_" + ref_name: float(
                    (ref["zone_hopping_score_fire_station"] - o["zone_hopping_score_fire_station"])
                    .abs()
                    .max()
                ),
                "n_3class_label_mismatch_vs_" + ref_name: int(
                    (ref["label_fire_station"] != o["label_fire_station"]).sum()
                ),
            }
        )
    fold_df = pd.DataFrame(fold_rows)

    # ---- assemble the 3-task per-player table (fold_00 fire_station) ----
    fs = derive_two_class(fs_by_fold[ref_name].copy(), "fire_station")
    bg = derive_two_class(load_task_scores(BG_LABEL_CSV, "back_garden"), "back_garden")
    bun_csv = args.bungalow_dir / "bungalow_zone_labels_3_class.csv"
    if not bun_csv.exists():
        bun_csv = args.bungalow_dir / "bungalow_zone_labels.csv"
    bun = derive_two_class(load_task_scores(bun_csv, "bungalow"), "bungalow")

    per_task_n = {"back_garden": len(bg), "bungalow": len(bun), "fire_station": len(fs)}
    merged = bg.merge(bun, on="player_id", how="inner").merge(fs, on="player_id", how="inner")
    n_all = len(merged)

    # ---- (1) correlations ----
    pairs = [
        ("back_garden", "bungalow"),
        ("back_garden", "fire_station"),
        ("bungalow", "fire_station"),
    ]
    corr_rows = []
    for a, b in pairs:
        corr_rows.append(
            correlate(
                merged[f"zone_hopping_score_{a}"],
                merged[f"zone_hopping_score_{b}"],
                a,
                b,
                subset="all_players_with_all_3_tasks",
            )
        )
    # clean-filter subset: players non-Mixed on BOTH tasks of the pair
    for a, b in pairs:
        m = merged[
            (merged[f"label_2_class_clean_{a}"] != "") & (merged[f"label_2_class_clean_{b}"] != "")
        ]
        corr_rows.append(
            correlate(
                m[f"zone_hopping_score_{a}"],
                m[f"zone_hopping_score_{b}"],
                a,
                b,
                subset="clean_filter_both_tasks",
            )
        )
    corr_df = pd.DataFrame(corr_rows)

    # ---- (2) label carry-over ceiling check, all folds x variants ----
    #
    # Three evaluation protocols:
    #   symmetric_2_class        : median-split label on both sides, all players.
    #   symmetric_2_class_clean  : Mixed dropped on BOTH sides (over-optimistic;
    #                              drops the ambiguous middle from the source too).
    #   paper_matched            : matched to the PLVM evaluation;
    #                              target = fire_station `2_class_clean` (the PLVM
    #                              target), source = observed-task median-split
    #                              `2_class` label, which is always defined (Back
    #                              Garden / Bungalow enter the PLVM as process
    #                              traces, not as filtered labels).
    co_rows = []
    for fold_name in sorted(fs_by_fold):
        fs_f = derive_two_class(fs_by_fold[fold_name].copy(), "fire_station")
        m = bg.merge(bun, on="player_id", how="inner").merge(fs_f, on="player_id", how="inner")

        # combined Back Garden + Bungalow source: mean of the two per-task scores,
        # median-split (simplest 2-task analogue of what the PLVM observes)
        combo = m[["zone_hopping_score_back_garden", "zone_hopping_score_bungalow"]].mean(axis=1)
        m["label_2_class_bg_plus_bungalow"] = np.where(
            combo >= combo.median(), "Zone Hopper", "Zone Planner"
        )

        for variant in ("label_2_class", "label_2_class_clean"):
            for src in ("back_garden", "bungalow"):
                r = carryover(
                    m[f"{variant}_{src}"],
                    m[f"{variant}_fire_station"],
                    src,
                    "symmetric_" + variant,
                    fold_name,
                )
                if r:
                    co_rows.append(r)
        # paper-matched protocol, scored on ALL eligible players
        for src in ("back_garden", "bungalow", "bg_plus_bungalow"):
            r = carryover(
                m[f"label_2_class_{src}"],
                m["label_2_class_clean_fire_station"],
                src,
                "paper_matched",
                fold_name,
            )
            if r:
                co_rows.append(r)

        # paper-matched protocol, scored on this fold's held-out val players only,
        # so the mean +/- SD over folds is comparable to the PLVM's fold accuracies
        manifest_path = FS_LABEL_ROOT / fold_name / "player_split_manifest.csv"
        if manifest_path.exists():
            man = pd.read_csv(manifest_path)
            val_ids = set(man.loc[man["split"] == "val", "player_id"].astype(str))
            mv = m[m["player_id"].isin(val_ids)]
            for src in ("back_garden", "bungalow", "bg_plus_bungalow"):
                r = carryover(
                    mv[f"label_2_class_{src}"],
                    mv["label_2_class_clean_fire_station"],
                    src,
                    "paper_matched_val_fold",
                    fold_name,
                )
                if r:
                    co_rows.append(r)
    co_df = pd.DataFrame(co_rows)
    co_summary = (
        co_df.groupby(["variant", "source_task"])
        .agg(
            n_mean=("n", "mean"),
            acc_mean=("carryover_accuracy", "mean"),
            acc_std=("carryover_accuracy", "std"),
            kappa_mean=("cohens_kappa", "mean"),
            kappa_std=("cohens_kappa", "std"),
            baseline_mean=("target_majority_baseline", "mean"),
        )
        .reset_index()
    )

    # ---- write artifacts ----
    fold_df.to_csv(out_dir / "navstyle_firestation_fold_invariance.csv", index=False)
    corr_df.to_csv(out_dir / "navstyle_cross_task_correlations.csv", index=False)
    co_df.to_csv(out_dir / "navstyle_label_carryover_by_fold.csv", index=False)
    co_summary.to_csv(out_dir / "navstyle_label_carryover_summary.csv", index=False)
    merged.to_csv(out_dir / "navstyle_per_player_scores_3tasks.csv", index=False)

    meta = {
        "score_formula": SCORE_FORMULA,
        "score_source": "persisted `zone_hopping_score` column (NOT recomputed)",
        "traces": "full, non-truncated (full-task subtask completion ordering)",
        "fire_station_labels": artifact_paths.display_path(FS_LABEL_ROOT / ref_name),
        "back_garden_labels": artifact_paths.display_path(BG_LABEL_CSV),
        "bungalow_labels": artifact_paths.display_path(bun_csv),
        "n_eligible_per_task": per_task_n,
        "n_players_with_all_three_tasks": n_all,
        "n_dropped_relative_to_each_task": {k: v - n_all for k, v in per_task_n.items()},
    }
    (out_dir / "navstyle_cross_task_metadata.json").write_text(json.dumps(meta, indent=2))

    # ---- markdown report ----
    def fmt(df_, cols, hdr):
        lines = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
        for _, r in df_.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    md = [
        "# Cross-task stability of the navigation-style score",
        "",
        f"Score: `{SCORE_FORMULA}` (persisted, not recomputed; full non-truncated traces).",
        "",
        f"Eligible players per task: {per_task_n}. Players with all three tasks: **{n_all}**.",
        "",
        "## 1. Continuous score correlations",
        "",
        fmt(
            corr_df,
            [
                "subset",
                "task_a",
                "task_b",
                "n",
                "pearson_r",
                "pearson_ci_lo",
                "pearson_ci_hi",
                "pearson_r2",
                "spearman_rho",
                "pearson_p",
            ],
            [
                "subset",
                "task A",
                "task B",
                "n",
                "Pearson r",
                "CI lo",
                "CI hi",
                "r^2",
                "Spearman rho",
                "Pearson p",
            ],
        ),
        "",
        "## 2. Label carry-over ceiling check (mean over 10 fire-station label folds)",
        "",
        "`paper_matched` is matched to the PLVM evaluation: target = fire_station "
        "`2_class_clean` (the PLVM target), source = the observed task's "
        "always-defined median-split label.",
        "",
        fmt(
            co_summary,
            [
                "variant",
                "source_task",
                "n_mean",
                "acc_mean",
                "acc_std",
                "kappa_mean",
                "baseline_mean",
            ],
            [
                "label variant",
                "source task",
                "n",
                "accuracy",
                "sd",
                "Cohen's kappa",
                "majority baseline",
            ],
        ),
        "",
        "## 3. Fire-station label fold-invariance",
        "",
        fmt(fold_df, list(fold_df.columns), list(fold_df.columns)),
        "",
    ]
    (out_dir / "navstyle_cross_task_correlation.md").write_text("\n".join(md))

    print("\n".join(md))
    print(f"\n[saved] {out_dir}")


if __name__ == "__main__":
    main()
