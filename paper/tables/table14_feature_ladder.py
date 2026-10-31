#!/usr/bin/env python3
"""
table14_feature_ladder.py
=========================
Generates the Table 14 feature ladder: the PLVM vs hand-crafted navigation features on
the same observation window, per fold and cutoff. The first three rungs are logistic
regressions on standardised features computed per observed task and concatenated:

  O          Outcome: n_completed_subtasks and level_completion.
  P-static   O + zone-visit proportions over the K zones (order-free).
  P-order    P-static + the four order-dependent ZoneHoppingScore statistics
             (zone_switch_rate, mean_zone_run_length, return_rate,
             mean_zone_jump_distance).
  P-learned  The PLVM, read from its saved validation predictions.

The window is reconstructed and verified by ``navigation_style_observed_sequences.py``.
Table 14 needs two runs; the second is the ordering ablation (each prefix permuted)::

    python paper/tables/table14_feature_ladder.py --tag full
    python paper/tables/table14_feature_ladder.py --shuffle_seed 0 --tag shuffled

The ablation is repeated for ``--n_shuffle_seeds`` seeds (default 10). Its per-fold
accuracies are averaged over the seeds, and the per-seed rows are written to
``navstyle_ladder_by_seed_<tag>.csv``.
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper.navigation_style_observed_sequences import load_task_sequences  # noqa: E402
from powerwash.data.create_firestation_zone_planning_labels import (  # noqa: E402
    build_subtask_zones,
)
from paper.output_paths import resolve_tables_dir  # noqa: E402
from plvm import artifact_paths  # noqa: E402

OBSERVED_TASKS = ("back_garden", "bungalow")
TARGET_MODE = "player_label_2_class_clean_zonehopping"
FS_LABEL_ROOT = artifact_paths.FIRESTATION_LABELS_DIR
SUBTASK_MAP = artifact_paths.SUBTASK_TASK_MAP_POINTS_CSV
TRAINING_LOGS = artifact_paths.POWERWASH_TRAINING_LOGS_DIR
DEFAULT_OUT = resolve_tables_dir() / "table14_feature_ladder"
CUTOFFS_SECONDS = tuple(range(5 * 60, 121 * 60, 5 * 60))
LADDER = ("O", "P-static", "P-order")


# --------------------------------------------------------------------------
# zone statistics on a prefix
# --------------------------------------------------------------------------


def zone_stats(raw_zones: np.ndarray, centroids: np.ndarray) -> Tuple[float, float, float, float]:
    """The four navigation-style statistics, matching compute_zone_features().

    Returns (zone_switch_rate, mean_zone_run_length, return_rate,
    mean_zone_jump_distance) for one zone sequence.
    """
    n_completed = int(len(raw_zones))
    if n_completed == 0:
        return (0.0, 0.0, 0.0, 0.0)

    # collapsed = consecutive repeats removed
    keep = np.ones(n_completed, dtype=bool)
    if n_completed > 1:
        keep[1:] = raw_zones[1:] != raw_zones[:-1]
    collapsed = raw_zones[keep]
    n_visits = int(len(collapsed))

    switch_rate = (n_visits - 1) / max(n_completed - 1, 1)

    # run lengths in the raw sequence
    boundaries = np.flatnonzero(keep)
    run_lengths = np.diff(np.append(boundaries, n_completed))
    mean_run_length = float(run_lengths.mean()) if len(run_lengths) else 0.0

    # returns: a visit to a zone already seen and left
    seen = set()
    n_returns = 0
    for i, z in enumerate(collapsed):
        if i > 0 and z in seen:
            n_returns += 1
        seen.add(z)
    return_rate = n_returns / max(n_visits, 1)

    if n_visits >= 2:
        pts = centroids[collapsed]
        mean_jump = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).mean())
    else:
        mean_jump = 0.0

    return (float(switch_rate), float(mean_run_length), float(return_rate), mean_jump)


def zone_proportions(raw_zones: np.ndarray, n_zones: int) -> np.ndarray:
    """Order-free zone-visit distribution (proportions, never raw counts)."""
    if len(raw_zones) == 0:
        return np.zeros(n_zones, dtype=float)
    counts = np.bincount(raw_zones, minlength=n_zones).astype(float)
    return counts / counts.sum()


# --------------------------------------------------------------------------
# feature construction
# --------------------------------------------------------------------------


def build_player_features(
    sequences: Dict[str, object],
    zone_of_subtask: Dict[str, int],
    centroids: np.ndarray,
    n_zones: int,
    cutoffs: Tuple[int, ...],
    shuffle_seed: Optional[int] = None,
    task_name: str = "",
) -> Dict[str, Dict[int, dict]]:
    """For one task, compute every feature block at every cutoff, per player.

    ``shuffle_seed`` permutes each observed prefix before the order-dependent
    statistics are computed -- the ordering ablation. It does not affect the
    outcome or zone-proportion blocks, which are permutation invariant by
    construction. Each prefix draws from its own generator, seeded by
    (seed, task, player, cutoff), so a permutation does not depend on which
    other players or cutoffs are evaluated.
    """
    out: Dict[str, Dict[int, dict]] = {}

    for pid, seq in sequences.items():
        # map subtasks to zones once; unmapped subtasks are skipped, matching
        # build_player_zone_sequences()
        zone_ids = np.array([zone_of_subtask.get(name, -1) for name in seq.subtask], dtype=int)
        mapped = zone_ids >= 0
        per_cutoff: Dict[int, dict] = {}
        for cutoff in cutoffs:
            k = seq.prefix_len(cutoff)
            if k <= 0:
                per_cutoff[cutoff] = None
                continue
            sel = mapped[:k]
            zs = zone_ids[:k][sel]
            if shuffle_seed is not None and len(zs) > 1:
                rng = np.random.default_rng(
                    [
                        int(shuffle_seed),
                        zlib.crc32(task_name.encode()),
                        zlib.crc32(str(pid).encode()),
                        int(cutoff),
                    ]
                )
                zs = zs[rng.permutation(len(zs))]
            switch, runlen, ret, jump = zone_stats(zs, centroids)
            per_cutoff[cutoff] = {
                "outcome": np.array([float(k), float(seq.completion[k - 1])], dtype=float),
                "static": zone_proportions(zs, n_zones),
                "order": np.array([switch, runlen, ret, jump], dtype=float),
            }
        out[pid] = per_cutoff
    return out


def assemble_matrix(
    player_ids: List[str],
    cutoff: int,
    features_by_task: Dict[str, Dict[str, Dict[int, dict]]],
    rung: str,
) -> Optional[np.ndarray]:
    """Stack the feature blocks for one rung of the ladder, across both tasks."""
    blocks = {
        "O": ("outcome",),
        "P-static": ("outcome", "static"),
        "P-order": ("outcome", "static", "order"),
    }[rung]
    rows = []
    for pid in player_ids:
        parts = []
        for task in OBSERVED_TASKS:
            cell = features_by_task[task].get(pid, {}).get(cutoff)
            if cell is None:
                parts = None
                break
            parts.extend(cell[b] for b in blocks)
        if parts is None:
            return None
        rows.append(np.concatenate(parts))
    return np.vstack(rows) if rows else None


def eligible_players(player_ids, cutoff: int, features_by_task) -> List[str]:
    """Players with a non-empty observed prefix in BOTH tasks at this cutoff."""
    ok = []
    for pid in player_ids:
        if all(features_by_task[t].get(pid, {}).get(cutoff) is not None for t in OBSERVED_TASKS):
            ok.append(pid)
    return ok


# --------------------------------------------------------------------------
# PLVM reference predictions
# --------------------------------------------------------------------------


def load_plvm_predictions(condition: str = "PLVM-2") -> pd.DataFrame:
    """Per-fold, per-player PLVM correctness from its own saved predictions."""
    from paper.figures.fig06_fig08_fig09_fig10_powerwash_accuracy_curves import (
        DEFAULT_PERF_CSV,
        load_candidate_rows,
        select_all_rows_by_fold,
    )

    rows = load_candidate_rows(DEFAULT_PERF_CSV, seq_length=120, target_mode=TARGET_MODE)
    _meta, by_cond_fold = select_all_rows_by_fold(rows)
    if condition not in by_cond_fold:
        raise SystemExit(f"condition {condition!r} not in {sorted(by_cond_fold)}")

    frames = []
    for fold_id, row in sorted(by_cond_fold[condition].items()):
        path = TRAINING_LOGS / "plvm_models" / row["model_name"] / "val_predictions.csv"
        if not path.exists():
            continue
        p = pd.read_csv(
            path,
            usecols=["player_id", "observation_cutoff_seconds", "target_label", "prediction_label"],
        )
        p["player_id"] = p["player_id"].astype(str)
        # both_partial emits one row per observed task; collapse to one per player
        p = p.drop_duplicates(subset=["player_id", "observation_cutoff_seconds"])
        p["fold"] = str(fold_id)
        frames.append(p)
    if not frames:
        raise SystemExit("no PLVM val_predictions.csv found")
    out = pd.concat(frames, ignore_index=True)
    out["plvm_correct"] = out["target_label"].values == out["prediction_label"].values
    return out


# --------------------------------------------------------------------------
# ladder fit
# --------------------------------------------------------------------------


def fit_ladder(
    folds: List[str],
    manifests: Dict[str, pd.DataFrame],
    plvm: pd.DataFrame,
    label_by_player: Dict[str, str],
    cutoffs: Tuple[int, ...],
    features_by_task: Dict[str, Dict[str, Dict[int, dict]]],
    save_predictions: bool = False,
) -> Tuple[List[dict], List[dict]]:
    """Fit every rung per fold and cutoff; return per-fold accuracies and per-player predictions."""
    records = []
    pred_rows = []
    for fold in folds:
        man = manifests.get(fold)
        if man is None:
            continue
        tr_all = [p for p in man.loc[man["split"] == "train", "player_id"] if p in label_by_player]
        plvm_fold = plvm[plvm["fold"] == fold]

        for cutoff in cutoffs:
            pv = plvm_fold[
                np.isclose(plvm_fold["observation_cutoff_seconds"].astype(float), float(cutoff))
            ]
            if pv.empty:
                continue
            va = eligible_players(
                [p for p in pv["player_id"] if p in label_by_player], cutoff, features_by_task
            )
            tr = eligible_players(tr_all, cutoff, features_by_task)
            if len(va) < 20 or len(tr) < 50:
                continue

            y_tr = np.array([label_by_player[p] for p in tr])
            y_va = np.array([label_by_player[p] for p in va])
            if len(np.unique(y_tr)) < 2:
                continue

            base = {"fold": fold, "cutoff_seconds": cutoff, "n_train": len(tr), "n_val": len(va)}

            # PLVM on exactly this evaluation set
            pv_sel = pv.set_index("player_id").reindex(va)
            base["acc_P-learned"] = float(pv_sel["plvm_correct"].mean())
            base["acc_majority"] = float(pd.Series(y_va).value_counts(normalize=True).max())

            for rung in LADDER:
                X_tr = assemble_matrix(tr, cutoff, features_by_task, rung)
                X_va = assemble_matrix(va, cutoff, features_by_task, rung)
                if X_tr is None or X_va is None:
                    base[f"acc_{rung}"] = np.nan
                    continue
                scaler = StandardScaler().fit(X_tr)
                clf = LogisticRegression(max_iter=2000, C=1.0)
                clf.fit(scaler.transform(X_tr), y_tr)
                pred = clf.predict(scaler.transform(X_va))
                base[f"acc_{rung}"] = float((pred == y_va).mean())
                if save_predictions:
                    # per-player predictions, needed for subgroup analyses that
                    # partition the same held-out players after the fact
                    # Probability of the last class in ``clf.classes_`` (sorted labels).
                    proba = clf.predict_proba(scaler.transform(X_va))[:, -1]
                    for i, pid in enumerate(va):
                        pred_rows.append(
                            {
                                "player_id": pid,
                                "fold": fold,
                                "cutoff_seconds": cutoff,
                                "rung": rung,
                                "target_label": y_va[i],
                                "prediction_label": pred[i],
                                "prediction_prob": float(proba[i]),
                                "prob_class": str(clf.classes_[-1]),
                            }
                        )
            records.append(base)
        print(f"[fold] {fold} done", flush=True)

    return records, pred_rows


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    """Fit the feature ladder per fold and cutoff and write the ladder tables."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n_zones", type=int, default=6, help="Number of spatial zones per level.")
    ap.add_argument(
        "--primary_cutoff_minutes",
        type=float,
        default=120.0,
        help="Cutoff at which adjacent rungs are tested against each other.",
    )
    ap.add_argument(
        "--condition", default="PLVM-2", help="PLVM condition used as the learned rung."
    )
    ap.add_argument(
        "--shuffle_seed",
        type=int,
        default=None,
        help="Permute each observed prefix (ordering ablation); first of the seeds used.",
    )
    ap.add_argument(
        "--n_shuffle_seeds",
        type=int,
        default=10,
        help="Number of shuffle seeds averaged in the ordering ablation.",
    )
    ap.add_argument(
        "--max_players", type=int, default=0, help="Debug: cap players per task (0 = all)."
    )
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT, help="Output directory.")
    ap.add_argument("--tag", default="", help="Suffix for output filenames.")
    ap.add_argument(
        "--save_predictions",
        action="store_true",
        help="Persist per-player validation predictions for each rung.",
    )
    ap.add_argument(
        "--cutoffs_minutes",
        default="",
        help="Comma-separated cutoffs to evaluate (default: the full schedule).",
    )
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cutoffs = (
        tuple(int(float(c) * 60) for c in args.cutoffs_minutes.split(",") if c.strip())
        if args.cutoffs_minutes
        else CUTOFFS_SECONDS
    )
    primary = int(args.primary_cutoff_minutes * 60)

    # ---- PLVM reference -------------------------------------------------
    print("[plvm] loading saved predictions ...", flush=True)
    plvm = load_plvm_predictions(args.condition)
    needed = sorted(set(plvm["player_id"]))
    folds = sorted(plvm["fold"].unique())
    print(f"[plvm] {len(needed)} val players across {len(folds)} folds")

    # training players come from the fold manifests
    manifests = {}
    train_ids = set()
    for fold in folds:
        man = FS_LABEL_ROOT / fold / "player_split_manifest.csv"
        if not man.exists():
            continue
        m = pd.read_csv(man)
        m["player_id"] = m["player_id"].astype(str)
        manifests[fold] = m
        train_ids |= set(m.loc[m["split"] == "train", "player_id"])
    all_needed = sorted(set(needed) | train_ids)
    if args.max_players:
        all_needed = all_needed[: args.max_players]
    print(f"[data] {len(all_needed)} players needed (train + val)")

    # ---- observed sequences + zone features -----------------------------
    seqs_by_task: Dict[str, Dict[str, object]] = {}
    zones_by_task: Dict[str, Tuple[Dict[str, int], np.ndarray]] = {}
    coverage = {}
    for task in OBSERVED_TASKS:
        zone_of_subtask, centroids, _df = build_subtask_zones(
            SUBTASK_MAP, n_zones=args.n_zones, task_name=task
        )
        print(f"[seq] reconstructing observed windows for {task} ...", flush=True)
        seqs, stats_ = load_task_sequences(task, all_needed, progress_every=2000)
        # The dropped IDs feed the exclusion-balance check written to the coverage JSON.
        coverage[task] = stats_
        print(
            f"[seq] {task}: {stats_['n_verified']}/{stats_['n_candidates']} verified "
            f"({stats_['pct_verified']:.1f}%)"
        )
        seqs_by_task[task] = seqs
        zones_by_task[task] = (zone_of_subtask, centroids)

    # ---- labels ---------------------------------------------------------
    label_by_player: Dict[str, str] = {}
    for fold in folds:
        csv_path = FS_LABEL_ROOT / fold / "firestation_zone_labels_2_class_clean.csv"
        if not csv_path.exists():
            continue
        d = pd.read_csv(csv_path, usecols=["player_id", "label"])
        d = d[d["label"].notna()]
        for pid, lab in zip(d["player_id"].astype(str), d["label"].astype(str)):
            label_by_player.setdefault(pid, lab)
    print(f"[labels] {len(label_by_player)} players carry a 2-class-clean label")

    # ---- fit, once per shuffle seed -------------------------------------
    seeds: List[Optional[int]] = (
        [None]
        if args.shuffle_seed is None
        else [args.shuffle_seed + i for i in range(args.n_shuffle_seeds)]
    )
    records: List[dict] = []
    pred_rows: List[dict] = []
    for seed in seeds:
        if seed is not None:
            print(f"[shuffle] seed {seed}", flush=True)
        features_by_task = {
            task: build_player_features(
                seqs_by_task[task],
                *zones_by_task[task],
                args.n_zones,
                cutoffs,
                shuffle_seed=seed,
                task_name=task,
            )
            for task in OBSERVED_TASKS
        }
        seed_records, seed_preds = fit_ladder(
            folds,
            manifests,
            plvm,
            label_by_player,
            cutoffs,
            features_by_task,
            save_predictions=args.save_predictions,
        )
        for row in seed_records + seed_preds:
            row["shuffle_seed"] = seed
        records += seed_records
        pred_rows += seed_preds

    by_seed = pd.DataFrame(records)
    if by_seed.empty:
        raise SystemExit("no results produced")
    # One row per fold and cutoff: the ablation's accuracies are averaged over seeds. The
    # player sets do not depend on the seed, so n_train and n_val are the same in every seed.
    by_fold = by_seed.drop(columns="shuffle_seed")
    if len(seeds) > 1:
        by_fold = by_fold.groupby(
            ["fold", "cutoff_seconds", "n_train", "n_val"], as_index=False, sort=False
        ).mean()

    acc_cols = [f"acc_{r}" for r in LADDER] + ["acc_P-learned", "acc_majority"]
    pooled = by_fold.groupby("cutoff_seconds")[acc_cols].agg(["mean", "std", "count"]).reset_index()

    # ---- paired tests between adjacent rungs at the primary cutoff ------
    tests = []
    prim = by_fold[by_fold["cutoff_seconds"] == primary]
    ladder_full = list(LADDER) + ["P-learned"]
    pairs = [("acc_majority", "acc_O")] + [
        (f"acc_{a}", f"acc_{b}") for a, b in zip(ladder_full, ladder_full[1:])
    ]
    for lo, hi in pairs:
        sub = prim.dropna(subset=[lo, hi])
        if len(sub) < 3:
            continue
        try:
            p = float(stats.wilcoxon(sub[hi], sub[lo]).pvalue)
        except ValueError:
            p = float("nan")
        tests.append(
            {
                "cutoff_seconds": primary,
                "lower": lo.replace("acc_", ""),
                "higher": hi.replace("acc_", ""),
                "n_folds": len(sub),
                "lower_mean": float(sub[lo].mean()),
                "higher_mean": float(sub[hi].mean()),
                "gain_pp": float(100 * (sub[hi] - sub[lo]).mean()),
                "wilcoxon_p": p,
            }
        )
    tst = pd.DataFrame(tests)

    tag = f"_{args.tag}" if args.tag else ""
    by_fold.to_csv(args.out_dir / f"navstyle_ladder_by_fold{tag}.csv", index=False)
    if args.shuffle_seed is not None:
        by_seed.to_csv(args.out_dir / f"navstyle_ladder_by_seed{tag}.csv", index=False)
    pooled.to_csv(args.out_dir / f"navstyle_ladder_pooled{tag}.csv", index=False)
    tst.to_csv(args.out_dir / f"navstyle_ladder_tests{tag}.csv", index=False)
    if args.save_predictions and pred_rows:
        pd.DataFrame(pred_rows).to_csv(
            args.out_dir / f"navstyle_ladder_val_predictions{tag}.csv", index=False
        )
    # Exclusion balance: Zone Hopper share (and Fisher exact p) of the labelled players
    # dropped by window verification versus those retained.
    balance = {}
    for task, cov in coverage.items():
        dropped = [p for p in cov.get("dropped_player_ids", []) if p in label_by_player]
        kept = [
            p
            for p in all_needed
            if p not in set(cov.get("dropped_player_ids", [])) and p in label_by_player
        ]

        def _hopper(ids):
            if not ids:
                return float("nan")
            return float(np.mean([label_by_player[p] == "Zone Hopper" for p in ids]))

        n_d, n_k = len(dropped), len(kept)
        p_val = float("nan")
        if n_d >= 5 and n_k >= 5:
            table = [
                [sum(label_by_player[p] == "Zone Hopper" for p in dropped), n_d],
                [sum(label_by_player[p] == "Zone Hopper" for p in kept), n_k],
            ]
            p_val = float(
                stats.fisher_exact(
                    [[table[0][0], n_d - table[0][0]], [table[1][0], n_k - table[1][0]]]
                )[1]
            )
        balance[task] = {
            "n_dropped_labeled": n_d,
            "n_kept_labeled": n_k,
            "pct_hopper_dropped": 100 * _hopper(dropped),
            "pct_hopper_kept": 100 * _hopper(kept),
            "fisher_p": p_val,
        }

    with (args.out_dir / f"navstyle_ladder_coverage{tag}.json").open("w") as fh:
        json.dump(
            {
                "coverage": coverage,
                "n_zones": args.n_zones,
                "shuffle_seeds": [s for s in seeds if s is not None],
                "primary_cutoff_seconds": primary,
                "exclusion_balance": balance,
            },
            fh,
            indent=2,
        )

    # ---- console report --------------------------------------------------
    print("\n" + "=" * 78)
    print(
        f"PROCESS FEATURE LADDER   target = fire-station 2-class-clean   "
        f"cutoff = {primary // 60} min"
    )
    print("=" * 78)
    for name in ["acc_majority"] + [f"acc_{r}" for r in LADDER] + ["acc_P-learned"]:
        if name not in prim:
            continue
        v = prim[name].dropna()
        if v.empty:
            continue
        print(
            f"  {name.replace('acc_', ''):12s} {100 * v.mean():5.1f} +/- {100 * v.std(ddof=1):4.1f} "
            f"   (folds={len(v)}, mean n_val={prim['n_val'].mean():.0f})"
        )
    print("\n  -- adjacent-rung paired Wilcoxon --")
    for r in tst.itertuples():
        print(f"   {r.lower:10s} -> {r.higher:10s} {r.gain_pp:+5.1f} pp   p={r.wilcoxon_p:.4f}")
    print(f"\nWrote artifacts to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
