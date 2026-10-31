#!/usr/bin/env python3
"""Generates Table 10 (PLVM robustness tests) and the Figure 5c latent-space metrics.

Table 10 compares PLVM-2 with Bungalow Alone, Back Garden Alone and RF Both at
20/60/120 min and the 5-120 min mean, using the Figure 6 run selection: fold-level
deltas, bootstrap CIs, paired t-tests and per-player exact McNemar tests (which need
each selected run's ``val_predictions.csv``, RF runs included). Also writes held-out
linear-probe AUROC and silhouette for the Figure 5c latent space. Writes to
``manuscript_figures/tables/table10_robustness/``.
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper.figures.fig05c_latent_space import (  # noqa: E402
    DEFAULT_PARAMETERS_TO_PREDICT,
    filter_frame_to_cutoff,
    load_latent_plot_frame,
    load_model_performance_rows,
    resolve_run_dir,
    select_best_run,
)
from paper.figures.fig06_fig08_fig09_fig10_powerwash_accuracy_curves import (  # noqa: E402
    DEFAULT_PERF_CSV,
    SUMMARY_AUC_MAX_MINUTES,
    SUMMARY_TIMEPOINT_MINUTES,
    TRAINING_LOGS_DIR,
    _condition_average_accuracy_scores,
    _summary_family_for_model_type,
    build_fold_long_frame,
    load_candidate_rows,
    select_all_rows_by_fold,
)
from paper.output_paths import resolve_tables_dir  # noqa: E402
from plvm import artifact_paths  # noqa: E402
from powerwash.train.powerwash_hierarchical_log_utils import (  # noqa: E402
    resolve_powerwash_hierarchical_run_dir,
)

DEFAULT_OUTPUT_DIR = resolve_tables_dir() / "table10_robustness"
DEFAULT_TARGET_MODES = (
    "player_label_2_class_clean_zonehopping",
    "player_label_3_class_zonehopping",
)
DEFAULT_LATENT_TRAINING_LOGS_DIR = artifact_paths.powerwash_family_logs_dir("plvm_models")
DEFAULT_LATENT_MODEL_PERFORMANCE_CSV = DEFAULT_LATENT_TRAINING_LOGS_DIR / "model_performance.csv"
DEFAULT_CUTOFF_SECONDS = 1800.0
BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 20260724
FIXED_TABLE10_CONDITIONS = {
    "player_label_2_class_clean_zonehopping": {
        "PLVM": "PLVM-2",
        "Single-task process": "Bungalow Alone",
    },
    "player_label_3_class_zonehopping": {
        "PLVM": "PLVM-2",
        "Single-task process": "Bungalow Alone",
    },
}
# Table 10 (robustness): PLVM-2 against each single-condition baseline on the
# main 2-class target. Labels are the Table 10 row names; values are conditions.
TABLE10_TARGET_MODE = "player_label_2_class_clean_zonehopping"
TABLE10_PLVM_CONDITION = "PLVM-2"
TABLE10_COMPARISONS = (
    ("Bungalow", "Bungalow Alone"),
    ("Back Garden", "Back Garden Alone"),
    ("Outcome", "RF Both"),
)
TABLE10_CSV_NAME = "table10_plvm_robustness.csv"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_performance_csv",
        type=Path,
        default=DEFAULT_PERF_CSV,
        help="PowerWash training-log root or model_performance.csv for curve summaries.",
    )
    parser.add_argument(
        "--training_logs_dir",
        type=Path,
        default=TRAINING_LOGS_DIR,
        help="Training-log root containing selected cutoff-metric artifacts.",
    )
    parser.add_argument(
        "--target_mode",
        action="append",
        default=None,
        help="Target mode to summarize. Repeat to include multiple targets.",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=120,
        help="Sequence length of the runs to compare.",
    )
    parser.add_argument(
        "--latent_training_logs_dir",
        type=Path,
        default=DEFAULT_LATENT_TRAINING_LOGS_DIR,
        help="PLVM run directory root containing latent-space plot CSVs.",
    )
    parser.add_argument(
        "--latent_model_performance_csv",
        type=Path,
        default=DEFAULT_LATENT_MODEL_PERFORMANCE_CSV,
        help="PLVM model_performance.csv used to select the run shown in Figure 5c.",
    )
    parser.add_argument(
        "--cutoff_seconds",
        type=float,
        default=DEFAULT_CUTOFF_SECONDS,
        help="Cutoff used for latent-space separation metrics.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Table 10 / Figure 5c CSV outputs.",
    )
    return parser.parse_args(argv)


def _target_modes(args: argparse.Namespace) -> List[str]:
    return list(args.target_mode or DEFAULT_TARGET_MODES)


def _best_condition_by_family(fold_frame: pd.DataFrame, condition_meta: list) -> Dict[str, str]:
    """Return the condition with the best mean validation accuracy per model family."""
    condition_types = {
        str(row["condition"]): str(row.get("model_type", "")) for row in condition_meta
    }
    condition_scores = _condition_average_accuracy_scores(fold_frame, "val")
    best: Dict[str, Dict[str, object]] = {}
    for condition, model_type in sorted(condition_types.items()):
        family = _summary_family_for_model_type(model_type)
        if family is None or condition not in condition_scores:
            continue
        score = float(condition_scores[condition])
        current = best.get(family)
        if current is None or score > float(current["score"]):
            best[family] = {"condition": condition, "score": score}
    return {family: str(payload["condition"]) for family, payload in best.items()}


def _paired_values_at_cutoff(
    fold_frame: pd.DataFrame,
    condition_a: str,
    condition_b: str,
    cutoff_seconds: float,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return fold-aligned validation accuracies for two conditions at one cutoff."""
    subset = fold_frame.loc[
        (fold_frame["split"] == "val")
        & (np.isclose(fold_frame["cutoff_value"].astype(float), float(cutoff_seconds)))
        & (fold_frame["condition"].isin([condition_a, condition_b]))
    ].copy()
    pivot = subset.pivot_table(
        index="fold_id",
        columns="condition",
        values="accuracy",
        aggfunc="mean",
    ).dropna(subset=[condition_a, condition_b])
    fold_ids = [str(idx) for idx in pivot.index.tolist()]
    return (
        pivot[condition_a].astype(float).to_numpy(),
        pivot[condition_b].astype(float).to_numpy(),
        fold_ids,
    )


def _auc_values_by_fold(condition_frame: pd.DataFrame) -> Dict[str, float]:
    """Compute normalized accuracy AUC for each fold in one condition frame."""
    values: Dict[str, float] = {}
    cutoff_limit_seconds = float(SUMMARY_AUC_MAX_MINUTES) * 60.0
    subset = condition_frame.loc[
        pd.to_numeric(condition_frame["cutoff_value"], errors="coerce") <= cutoff_limit_seconds
    ].copy()
    integrate_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    for fold_id, fold_subset in subset.groupby("fold_id"):
        ordered = fold_subset.sort_values("cutoff_value")
        x_minutes = ordered["cutoff_value"].astype(float).to_numpy() / 60.0
        y_accuracy = ordered["accuracy"].astype(float).to_numpy()
        if len(x_minutes) < 2:
            continue
        span = float(x_minutes[-1] - x_minutes[0])
        if span <= 0.0:
            continue
        values[str(fold_id)] = float(integrate_fn(y_accuracy, x_minutes) / span)
    return values


def _paired_auc_values(
    fold_frame: pd.DataFrame,
    condition_a: str,
    condition_b: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    a_values = _auc_values_by_fold(
        fold_frame.loc[(fold_frame["split"] == "val") & (fold_frame["condition"] == condition_a)]
    )
    b_values = _auc_values_by_fold(
        fold_frame.loc[(fold_frame["split"] == "val") & (fold_frame["condition"] == condition_b)]
    )
    fold_ids = sorted(set(a_values).intersection(b_values))
    return (
        np.asarray([a_values[fold_id] for fold_id in fold_ids], dtype=float),
        np.asarray([b_values[fold_id] for fold_id in fold_ids], dtype=float),
        fold_ids,
    )


def _bootstrap_mean_ci(values: np.ndarray) -> Tuple[float, float]:
    """Return a deterministic percentile CI for the paired mean delta."""
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True)
    means = samples.mean(axis=1)
    return tuple(float(v) for v in np.percentile(means, [2.5, 97.5]))


def _paired_test_row(
    target_mode: str,
    metric: str,
    condition_a: str,
    condition_b: str,
    values_a: np.ndarray,
    values_b: np.ndarray,
    fold_ids: Iterable[str],
) -> Dict[str, object]:
    """Summarize one paired PLVM-minus-single-task comparison."""
    deltas = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    ci_low, ci_high = _bootstrap_mean_ci(deltas)
    if len(deltas) > 1 and not np.allclose(deltas, deltas[0]):
        t_p = float(stats.ttest_rel(values_a, values_b).pvalue)
        try:
            wilcoxon_p = float(stats.wilcoxon(deltas, zero_method="wilcox").pvalue)
        except ValueError:
            wilcoxon_p = float("nan")
    else:
        t_p = float("nan")
        wilcoxon_p = float("nan")
    return {
        "target_mode": target_mode,
        "metric": metric,
        "plvm_condition": condition_a,
        "single_task_condition": condition_b,
        "n_folds": len(deltas),
        "plvm_mean": float(np.mean(values_a)) if len(values_a) else float("nan"),
        "plvm_sd": float(np.std(values_a, ddof=1)) if len(values_a) > 1 else float("nan"),
        "single_task_mean": float(np.mean(values_b)) if len(values_b) else float("nan"),
        "single_task_sd": float(np.std(values_b, ddof=1)) if len(values_b) > 1 else float("nan"),
        "mean_delta": float(np.mean(deltas)) if len(deltas) else float("nan"),
        "delta_ci95_low": ci_low,
        "delta_ci95_high": ci_high,
        "paired_t_p": t_p,
        "wilcoxon_p": wilcoxon_p,
        "plvm_better_folds": int(np.sum(deltas > 0.0)),
        "shared_fold_ids": ",".join(str(fold_id) for fold_id in fold_ids),
    }


def build_paired_fold_tests(
    fold_frame: pd.DataFrame,
    target_mode: str,
    plvm_condition: str,
    baseline_condition: str,
) -> List[Dict[str, object]]:
    """Return the fold-level paired PLVM-minus-baseline rows used by Tables 2 and 10.

    One row per accuracy cutoff in ``SUMMARY_TIMEPOINT_MINUTES`` (metric
    ``"<m> min accuracy"``) plus the 5-120 min mean-accuracy row (metric
    ``"AUC at 120 min"``, the trapezoid-normalised accuracy curve). Each row
    carries the mean delta, bootstrap CI, paired t-test and Wilcoxon p-values
    of ``_paired_test_row``.
    """
    paired_rows: List[Dict[str, object]] = []
    for minutes in SUMMARY_TIMEPOINT_MINUTES:
        values_a, values_b, fold_ids = _paired_values_at_cutoff(
            fold_frame,
            plvm_condition,
            baseline_condition,
            cutoff_seconds=float(minutes) * 60.0,
        )
        paired_rows.append(
            _paired_test_row(
                target_mode,
                "{} min accuracy".format(int(minutes)),
                plvm_condition,
                baseline_condition,
                values_a,
                values_b,
                fold_ids,
            )
        )
    auc_a, auc_b, auc_fold_ids = _paired_auc_values(fold_frame, plvm_condition, baseline_condition)
    paired_rows.append(
        _paired_test_row(
            target_mode,
            "AUC at 120 min",
            plvm_condition,
            baseline_condition,
            auc_a,
            auc_b,
            auc_fold_ids,
        )
    )
    return paired_rows


def _resolve_val_predictions_path(training_logs_dir: Path, model_name: str) -> Path:
    """Resolve the saved validation-prediction CSV for one selected run."""
    run_dir = resolve_powerwash_hierarchical_run_dir(Path(training_logs_dir), str(model_name))
    path = run_dir / "val_predictions.csv"
    if not path.exists():
        raise FileNotFoundError("Missing validation predictions: {}".format(path))
    return path


def _exact_mcnemar_pvalue(plvm_only_correct: int, single_only_correct: int) -> float:
    """Return the exact two-sided McNemar/binomial p-value for discordant pairs."""
    discordant = int(plvm_only_correct) + int(single_only_correct)
    if discordant <= 0:
        return float("nan")
    try:
        return float(
            stats.binomtest(
                min(int(plvm_only_correct), int(single_only_correct)),
                discordant,
                0.5,
                alternative="two-sided",
            ).pvalue
        )
    except AttributeError:  # scipy<1.7
        return float(
            stats.binom_test(
                min(int(plvm_only_correct), int(single_only_correct)),
                discordant,
                0.5,
                alternative="two-sided",
            )
        )


def build_per_player_mcnemar_rows(
    target_mode: str,
    training_logs_dir: Path,
    rows_by_condition_fold: dict,
    plvm_condition: str,
    single_condition: str,
    timepoints_minutes: Tuple[float, ...] = SUMMARY_TIMEPOINT_MINUTES,
) -> List[Dict[str, object]]:
    """Build per-player McNemar tests from selected validation predictions."""
    pieces: List[pd.DataFrame] = []
    for condition in (single_condition, plvm_condition):
        fold_rows = rows_by_condition_fold.get(condition, {})
        for fold_id, selected_row in sorted(fold_rows.items()):
            path = _resolve_val_predictions_path(
                Path(training_logs_dir),
                str(selected_row["model_name"]),
            )
            frame = pd.read_csv(path)
            if "target_mode" in frame.columns:
                frame = frame.loc[frame["target_mode"].astype(str) == str(target_mode)].copy()
            required_columns = [
                "player_id",
                "observation_cutoff_seconds",
                "target",
                "prediction_class_id",
            ]
            missing = [column for column in required_columns if column not in frame.columns]
            if missing:
                raise ValueError("Missing columns {} in {}".format(missing, path))
            frame = frame[required_columns].copy()
            frame["condition"] = condition
            frame["fold_id"] = str(fold_id)
            frame["model_name"] = str(selected_row["model_name"])
            pieces.append(frame)

    if not pieces:
        return []

    predictions = pd.concat(pieces, ignore_index=True)
    output_rows: List[Dict[str, object]] = []
    for minute_value in timepoints_minutes:
        cutoff_seconds = float(minute_value) * 60.0
        subset = predictions.loc[
            np.isclose(
                predictions["observation_cutoff_seconds"].astype(float).to_numpy(),
                cutoff_seconds,
            )
        ].copy()
        if subset.empty:
            continue
        pivot = subset.pivot_table(
            index=["fold_id", "player_id", "target"],
            columns="condition",
            values="prediction_class_id",
            aggfunc="first",
        ).dropna(subset=[single_condition, plvm_condition])
        if pivot.empty:
            continue

        indexed = pivot.reset_index()
        target_values = indexed["target"].astype(int).to_numpy()
        single_predictions = indexed[single_condition].astype(int).to_numpy()
        plvm_predictions = indexed[plvm_condition].astype(int).to_numpy()
        single_correct = single_predictions == target_values
        plvm_correct = plvm_predictions == target_values
        deltas = plvm_correct.astype(float) - single_correct.astype(float)
        ci_low, ci_high = _bootstrap_mean_ci(deltas)

        plvm_only_correct = int(np.sum(plvm_correct & ~single_correct))
        single_only_correct = int(np.sum(single_correct & ~plvm_correct))
        both_correct = int(np.sum(plvm_correct & single_correct))
        both_wrong = int(np.sum(~plvm_correct & ~single_correct))
        output_rows.append(
            {
                "target_mode": target_mode,
                "metric": "{} min per-player accuracy".format(int(minute_value)),
                "plvm_condition": plvm_condition,
                "single_task_condition": single_condition,
                "cutoff_minutes": float(minute_value),
                "n_observations": int(len(indexed)),
                "n_folds": int(indexed["fold_id"].nunique()),
                "plvm_accuracy": float(np.mean(plvm_correct)),
                "single_task_accuracy": float(np.mean(single_correct)),
                "mean_delta": float(np.mean(deltas)),
                "delta_ci95_low": ci_low,
                "delta_ci95_high": ci_high,
                "both_correct": both_correct,
                "plvm_only_correct": plvm_only_correct,
                "single_task_only_correct": single_only_correct,
                "both_wrong": both_wrong,
                "discordant_pairs": plvm_only_correct + single_only_correct,
                "mcnemar_exact_p": _exact_mcnemar_pvalue(
                    plvm_only_correct,
                    single_only_correct,
                ),
            }
        )
    return output_rows


def build_target_analysis(
    target_mode: str,
    model_performance_csv: Path,
    training_logs_dir: Path,
    seq_length: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Build the paired fold-level test rows and per-player McNemar rows for one target."""
    rows = load_candidate_rows(
        Path(model_performance_csv),
        seq_length=int(seq_length),
        target_mode=str(target_mode),
    )
    if not rows:
        raise ValueError("No candidate rows found for target_mode='{}'.".format(target_mode))
    condition_meta, rows_by_condition_fold = select_all_rows_by_fold(rows)
    fold_frame = build_fold_long_frame(Path(training_logs_dir), rows_by_condition_fold)
    selected = _best_condition_by_family(fold_frame, condition_meta)
    selected.update(FIXED_TABLE10_CONDITIONS.get(str(target_mode), {}))
    plvm_condition = selected["PLVM"]
    single_condition = selected["Single-task process"]

    paired_rows = build_paired_fold_tests(fold_frame, target_mode, plvm_condition, single_condition)
    mcnemar_rows = build_per_player_mcnemar_rows(
        target_mode=target_mode,
        training_logs_dir=Path(training_logs_dir),
        rows_by_condition_fold=rows_by_condition_fold,
        plvm_condition=plvm_condition,
        single_condition=single_condition,
    )
    return paired_rows, mcnemar_rows


def _fmt_p_value(value: float) -> str:
    """Format a p-value for Tables 2 and 10 (three decimals, '<0.001' floor)."""
    if not math.isfinite(float(value)):
        return ""
    return "<0.001" if float(value) < 0.001 else "{:.3f}".format(float(value))


def build_table10_rows(
    model_performance_csv: Path,
    training_logs_dir: Path,
    seq_length: int,
    target_mode: str = TABLE10_TARGET_MODE,
    plvm_condition: str = TABLE10_PLVM_CONDITION,
    comparisons: Sequence[Tuple[str, str]] = TABLE10_COMPARISONS,
) -> List[Dict[str, object]]:
    """Build the Table 10 rows: PLVM minus each baseline at 20/60/120 min and the 5-120 min mean.

    Uses the same fold selection as Table 2, the fold-level paired deltas,
    bootstrap CIs and paired t-tests of ``_paired_test_row``, and the
    per-player exact McNemar test of ``build_per_player_mcnemar_rows`` (pooled
    over folds; not defined for the 5-120 min mean-accuracy column, metric key
    ``"AUC at 120 min"``). Deltas and CIs are in percentage points.
    """
    rows = load_candidate_rows(
        Path(model_performance_csv),
        seq_length=int(seq_length),
        target_mode=str(target_mode),
    )
    if not rows:
        raise ValueError("No candidate rows found for target_mode='{}'.".format(target_mode))
    _, rows_by_condition_fold = select_all_rows_by_fold(rows)
    fold_frame = build_fold_long_frame(Path(training_logs_dir), rows_by_condition_fold)

    output_rows: List[Dict[str, object]] = []
    for comparison_label, baseline_condition in comparisons:
        paired_rows = build_paired_fold_tests(
            fold_frame, target_mode, plvm_condition, baseline_condition
        )
        try:
            mcnemar_rows = build_per_player_mcnemar_rows(
                target_mode=target_mode,
                training_logs_dir=Path(training_logs_dir),
                rows_by_condition_fold=rows_by_condition_fold,
                plvm_condition=plvm_condition,
                single_condition=baseline_condition,
            )
        except FileNotFoundError as exc:
            # A selected run without val_predictions.csv leaves only the McNemar
            # cells of that comparison empty.
            print(
                "WARNING: Table 10 McNemar skipped for {}: {}".format(baseline_condition, exc),
                file=sys.stderr,
            )
            mcnemar_rows = []
        mcnemar_by_minutes = {float(row["cutoff_minutes"]): row for row in mcnemar_rows}
        for paired in paired_rows:
            metric = str(paired["metric"])
            minutes = None if metric == "AUC at 120 min" else float(metric.split()[0])
            mcnemar = mcnemar_by_minutes.get(minutes, {}) if minutes is not None else {}
            mcnemar_p = float(mcnemar.get("mcnemar_exact_p", float("nan")))
            output_rows.append(
                {
                    "target_mode": target_mode,
                    "comparison": "PLVM-{}".format(comparison_label.replace(" ", "")),
                    "plvm_condition": plvm_condition,
                    "baseline_condition": baseline_condition,
                    "metric": metric,
                    "n_folds": int(paired["n_folds"]),
                    "mean_delta_pp": 100.0 * float(paired["mean_delta"]),
                    "delta_ci95_low_pp": 100.0 * float(paired["delta_ci95_low"]),
                    "delta_ci95_high_pp": 100.0 * float(paired["delta_ci95_high"]),
                    "paired_t_p": float(paired["paired_t_p"]),
                    "mcnemar_exact_p": mcnemar_p,
                    "mcnemar_n_observations": mcnemar.get("n_observations", ""),
                    "mcnemar_plvm_only_correct": mcnemar.get("plvm_only_correct", ""),
                    "mcnemar_baseline_only_correct": mcnemar.get("single_task_only_correct", ""),
                    "delta_formatted": "{:+.2f}".format(100.0 * float(paired["mean_delta"])),
                    "ci95_formatted": "[{:.2f}, {:.2f}]".format(
                        100.0 * float(paired["delta_ci95_low"]),
                        100.0 * float(paired["delta_ci95_high"]),
                    ),
                    "paired_t_p_formatted": _fmt_p_value(float(paired["paired_t_p"])),
                    "mcnemar_p_formatted": _fmt_p_value(mcnemar_p),
                }
            )
    return output_rows


def _cv_linear_probe_auc(frame: pd.DataFrame) -> float:
    """Return 5-fold CV ROC AUC for a linear probe on 2-D latent coordinates."""
    labels = pd.Categorical(frame["true_class_label"]).codes
    features = frame[["latent_x", "latent_y"]].astype(float).to_numpy()
    min_class_count = int(pd.Series(labels).value_counts().min())
    if len(set(labels)) != 2 or min_class_count < 2:
        return float("nan")
    n_splits = min(5, min_class_count)
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver="lbfgs"),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=BOOTSTRAP_SEED)
    scores = cross_val_score(estimator, features, labels, scoring="roc_auc", cv=cv)
    return float(np.mean(scores))


def build_latent_space_metrics(
    training_logs_dir: Path,
    model_performance_csv: Path,
    cutoff_seconds: float,
) -> Dict[str, object]:
    """Quantify class separation in the 2-D latent space shown in Figure 5c."""
    rows = load_model_performance_rows(Path(model_performance_csv))
    selected_row = select_best_run(
        rows=rows,
        training_logs_dir=Path(training_logs_dir),
        parameters_to_predict=DEFAULT_PARAMETERS_TO_PREDICT,
        model_type="plvm_both_partial",
        latent_dim=2,
    )
    run_dir = resolve_run_dir(Path(training_logs_dir), selected_row["model_name"])
    frame, _ = load_latent_plot_frame(run_dir)
    frame = filter_frame_to_cutoff(frame, float(cutoff_seconds))
    labels = pd.Categorical(frame["true_class_label"]).codes
    features = frame[["latent_x", "latent_y"]].astype(float).to_numpy()
    silhouette = (
        float(silhouette_score(features, labels))
        if len(set(labels)) > 1 and len(frame) > len(set(labels))
        else float("nan")
    )
    return {
        "selected_model_name": selected_row["model_name"],
        "cutoff_seconds": float(cutoff_seconds),
        "n_points": int(len(frame)),
        "class_counts": "; ".join(
            "{}={}".format(label, count)
            for label, count in frame["true_class_label"].value_counts().sort_index().items()
        ),
        "silhouette_score": silhouette,
        "linear_probe_cv_auc": _cv_linear_probe_auc(frame),
        "selected_val_accuracy": float(selected_row["val_accuracy"]),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the Table 10 and Figure 5c metrics workflow."""
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paired_rows: List[Dict[str, object]] = []
    mcnemar_rows: List[Dict[str, object]] = []
    for target_mode in _target_modes(args):
        target_paired, target_mcnemar = build_target_analysis(
            target_mode=target_mode,
            model_performance_csv=Path(args.model_performance_csv),
            training_logs_dir=Path(args.training_logs_dir),
            seq_length=int(args.seq_length),
        )
        paired_rows.extend(target_paired)
        mcnemar_rows.extend(target_mcnemar)

    table10_rows: List[Dict[str, object]] = []
    if TABLE10_TARGET_MODE in _target_modes(args):
        table10_rows = build_table10_rows(
            model_performance_csv=Path(args.model_performance_csv),
            training_logs_dir=Path(args.training_logs_dir),
            seq_length=int(args.seq_length),
        )

    latent_metrics = build_latent_space_metrics(
        training_logs_dir=Path(args.latent_training_logs_dir),
        model_performance_csv=Path(args.latent_model_performance_csv),
        cutoff_seconds=float(args.cutoff_seconds),
    )

    paired_csv = output_dir / "paired_plvm_vs_single_task_tests.csv"
    mcnemar_csv = output_dir / "per_player_mcnemar_tests.csv"
    latent_csv = output_dir / "figure_5c_latent_space_metrics.csv"

    pd.DataFrame(paired_rows).to_csv(paired_csv, index=False)
    pd.DataFrame(mcnemar_rows).to_csv(mcnemar_csv, index=False)
    pd.DataFrame([latent_metrics]).to_csv(latent_csv, index=False)
    table10_csv = output_dir / TABLE10_CSV_NAME
    if table10_rows:
        pd.DataFrame(table10_rows).to_csv(table10_csv, index=False)

    print("Wrote {}".format(paired_csv))
    print("Wrote {}".format(mcnemar_csv))
    print("Wrote {}".format(latent_csv))
    if table10_rows:
        print("Wrote {}".format(table10_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
