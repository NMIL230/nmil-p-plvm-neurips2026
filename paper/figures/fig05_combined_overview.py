#!/usr/bin/env python3
"""Generate Figure 5: (a) PowerWash and (b) simulation accuracy over observation time.

Each panel draws single-task process (nonlinear head), both-task process
(PLVM-2) and outcome curves. Panel (a) is the Fire Station 2-class zone-hopping
target at ``seq_length=120``, fold mean +/-1 SD, with Bungalow as the single
task. Panel (b) is the simulation at ``max_train_val_players=8000``, one best run
per condition, with whichever of Foraging/Cleaning has higher validation accuracy
as the single task.

The ``build_*`` helpers also return linear-head curves (``single_linear``,
``both_linear``), used by ``paper/tables/table12_head_capacity_ablation.py``.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper.plot_style import maybe_set_plot_style  # noqa: E402
from paper.output_paths import resolve_figures_dir  # noqa: E402

from paper.figures.fig06_fig08_fig09_fig10_powerwash_accuracy_curves import (  # noqa: E402
    DEFAULT_PERF_CSV as PW_DEFAULT_PERF_CSV,
    TRAINING_LOGS_DIR as PW_TRAINING_LOGS_DIR,
    aggregate_folds as pw_aggregate_folds,
    build_fold_long_frame as pw_build_fold_long_frame,
    load_candidate_rows as pw_load_candidate_rows,
    load_logistic_rows as pw_load_logistic_rows,
    select_all_rows_by_fold as pw_select_all_rows_by_fold,
    select_best_logistic_condition as pw_select_best_logistic_condition,
    select_single_task_logistic_conditions as pw_select_single_task_logistic_conditions,
)
from paper.figures.fig07_simulation_accuracy_curves import (  # noqa: E402
    DEFAULT_LOGISTIC_LOGS_DIR as SIM_LOGISTIC_LOGS_DIR,
    DEFAULT_LOGISTIC_PERF_CSV as SIM_LOGISTIC_PERF_CSV,
    DEFAULT_OUTCOME_LOGS_DIR as SIM_OUTCOME_LOGS_DIR,
    DEFAULT_OUTCOME_PERF_CSV as SIM_OUTCOME_PERF_CSV,
    DEFAULT_PLVM_LOGS_DIR as SIM_PLVM_LOGS_DIR,
    DEFAULT_PLVM_PERF_CSV as SIM_PLVM_PERF_CSV,
    DEFAULT_SINGLE_TASK_LOGISTIC_LOGS_DIR as SIM_SINGLE_TASK_LOGISTIC_LOGS_DIR,
    DEFAULT_SINGLE_TASK_LOGISTIC_PERF_CSV as SIM_SINGLE_TASK_LOGISTIC_PERF_CSV,
    load_candidate_rows as sim_load_candidate_rows,
    load_cutoff_frame as sim_load_cutoff_frame,
    select_best_rows as sim_select_best_rows,
)

PW_TARGET_MODE = "player_label_2_class_clean_zonehopping"
PW_SEQ_LENGTH = 120
SIM_MAX_TRAIN_VAL_PLAYERS = 8000
SIM_SEQ_LENGTH = 200
SIM_RUN_MODE = "run"

OUTCOME_COLOR = "#999999"
SINGLE_TASK_NONLINEAR_COLOR = "#c9602c"
SINGLE_TASK_LINEAR_COLOR = "#f2b880"
BOTH_TASK_NONLINEAR_COLOR = "#1f6f8b"

CURVE_ORDER = ("single_nonlinear", "both_nonlinear", "outcome")
CURVE_STYLE = {
    "outcome": {"label": "Outcome", "color": OUTCOME_COLOR, "linestyle": "-"},
    "single_nonlinear": {
        "label": "Process: Single Task",
        "color": SINGLE_TASK_NONLINEAR_COLOR,
        "linestyle": "-",
    },
    "both_nonlinear": {
        "label": "Process: PLVM",
        "color": BOTH_TASK_NONLINEAR_COLOR,
        "linestyle": "-",
    },
}
# Saved at exactly this size (no bbox_inches="tight"), so the PDF needs no rescaling.
FIGURE_WIDTH_INCHES = 4.3
FIGURE_HEIGHT_INCHES = 2.2  # matches Figure_5c.pdf's height
DEFAULT_MANUSCRIPT_FIGURES_DIR = resolve_figures_dir()


def build_powerwash_fold_frame():
    """Return (fold_frame, {curve_key: condition_name}) for the PowerWash panel.

    The per-fold accuracies behind :func:`build_powerwash_curves`, so that conditions
    can be paired within a CV fold.
    """
    base_rows = pw_load_candidate_rows(
        PW_DEFAULT_PERF_CSV, seq_length=PW_SEQ_LENGTH, target_mode=PW_TARGET_MODE
    )
    if not base_rows:
        raise ValueError("No PowerWash candidate rows found.")
    _condition_meta, rows_by_condition_fold = pw_select_all_rows_by_fold(base_rows)

    logistic_csv = Path(PW_TRAINING_LOGS_DIR) / "concat_logistic_probe" / "model_performance.csv"
    logistic_rows = pw_load_logistic_rows(logistic_csv, PW_SEQ_LENGTH, PW_TARGET_MODE)
    logistic_selection = pw_select_best_logistic_condition(PW_TRAINING_LOGS_DIR, logistic_rows)
    if logistic_selection is None:
        raise ValueError("No PowerWash Logistic Fuser condition found.")
    logistic_condition, logistic_fold_rows = logistic_selection

    single_task_logistic_csv = (
        Path(PW_TRAINING_LOGS_DIR) / "single_task_logistic_probe" / "model_performance.csv"
    )
    single_task_logistic_rows = pw_load_logistic_rows(
        single_task_logistic_csv, PW_SEQ_LENGTH, PW_TARGET_MODE
    )
    bungalow_logistic = dict(
        pw_select_single_task_logistic_conditions(
            single_task_logistic_rows,
            task_specs=[("Bungalow Logistic", "bungalow", SINGLE_TASK_LINEAR_COLOR)],
        )
    )
    if "Bungalow Logistic" not in bungalow_logistic:
        raise ValueError("No PowerWash Bungalow (linear head) condition found.")

    selected = {
        "outcome": ("RF Both", rows_by_condition_fold["RF Both"]),
        "single_nonlinear": ("Bungalow Alone", rows_by_condition_fold["Bungalow Alone"]),
        "single_linear": ("Bungalow Logistic", bungalow_logistic["Bungalow Logistic"]),
        "both_nonlinear": ("PLVM-2", rows_by_condition_fold["PLVM-2"]),
        "both_linear": (logistic_condition, logistic_fold_rows),
    }

    rows_by_condition_fold_subset = {name: fold_rows for name, fold_rows in selected.values()}
    fold_frame = pw_build_fold_long_frame(PW_TRAINING_LOGS_DIR, rows_by_condition_fold_subset)
    return fold_frame, {key: name for key, (name, _fold_rows) in selected.items()}


def build_powerwash_curves():
    """Return {curve_key: (x_minutes, y_percent, sd_percent)} for the PowerWash panel."""
    fold_frame, key_to_condition = build_powerwash_fold_frame()
    agg_frame = pw_aggregate_folds(fold_frame)
    val_agg = agg_frame.loc[agg_frame["split"] == "val"]

    curves = {}
    for key, condition_name in key_to_condition.items():
        condition_df = val_agg.loc[val_agg["condition"] == condition_name].sort_values(
            "cutoff_value"
        )
        x_minutes = condition_df["cutoff_value"].astype(float) / 60.0
        y_percent = condition_df["accuracy_mean"].astype(float) * 100.0
        yerr_percent = condition_df["accuracy_std"].astype(float) * 100.0
        curves[key] = (x_minutes.to_numpy(), y_percent.to_numpy(), yerr_percent.to_numpy())
    return curves


def build_simulation_curves():
    """Return ({curve_key: (x_steps, y_percent, se_percent)}, best_single_task) for panel (b)."""
    rows = []
    rows.extend(
        sim_load_candidate_rows(
            SIM_PLVM_PERF_CSV,
            source_family="process",
            logs_dir=SIM_PLVM_LOGS_DIR,
            run_mode=SIM_RUN_MODE,
            seq_length=SIM_SEQ_LENGTH,
            max_train_val_players=SIM_MAX_TRAIN_VAL_PLAYERS,
        )
    )
    rows.extend(
        sim_load_candidate_rows(
            SIM_OUTCOME_PERF_CSV,
            source_family="outcome",
            logs_dir=SIM_OUTCOME_LOGS_DIR,
            run_mode=SIM_RUN_MODE,
            seq_length=SIM_SEQ_LENGTH,
            max_train_val_players=SIM_MAX_TRAIN_VAL_PLAYERS,
        )
    )
    rows.extend(
        sim_load_candidate_rows(
            SIM_LOGISTIC_PERF_CSV,
            source_family="logistic",
            logs_dir=SIM_LOGISTIC_LOGS_DIR,
            run_mode=SIM_RUN_MODE,
            seq_length=SIM_SEQ_LENGTH,
            max_train_val_players=SIM_MAX_TRAIN_VAL_PLAYERS,
        )
    )
    rows.extend(
        sim_load_candidate_rows(
            SIM_SINGLE_TASK_LOGISTIC_PERF_CSV,
            source_family="single_task_logistic",
            logs_dir=SIM_SINGLE_TASK_LOGISTIC_LOGS_DIR,
            run_mode=SIM_RUN_MODE,
            seq_length=SIM_SEQ_LENGTH,
            max_train_val_players=SIM_MAX_TRAIN_VAL_PLAYERS,
        )
    )
    if not rows:
        raise ValueError("No simulation candidate rows found.")

    _condition_meta, selected_rows = sim_select_best_rows(rows)

    foraging_acc = float(selected_rows["Process: Foraging"]["val_accuracy"])
    cleaning_acc = float(selected_rows["Process: Cleaning"]["val_accuracy"])
    best_task = "Cleaning" if cleaning_acc >= foraging_acc else "Foraging"

    selected = {
        "outcome": selected_rows["Outcome: Both"],
        "single_nonlinear": selected_rows["Process: {}".format(best_task)],
        "single_linear": selected_rows["Process: {} (Logistic)".format(best_task)],
        "both_nonlinear": selected_rows["Process: PLVM-2"],
        "both_linear": selected_rows["Process: Logistic Fuser"],
    }

    curves = {}
    for key, run_row in selected.items():
        condition_df = sim_load_cutoff_frame(run_row, "val").sort_values("cutoff_value")
        accuracy = condition_df["accuracy"].astype(float).to_numpy()
        support = condition_df["support"].astype(float).to_numpy()
        # One run per condition, so the spread is the binomial standard error.
        binomial_se_percent = (
            np.sqrt(accuracy * (1.0 - accuracy) / np.maximum(support, 1.0)) * 100.0
        )
        curves[key] = (
            condition_df["cutoff_value"].astype(float).to_numpy(),
            accuracy * 100.0,
            binomial_se_percent,
        )
    return curves, best_task


def _plot_panel(ax, curves, x_label, reference_percent, show_band=True):
    """Draw the ``CURVE_ORDER`` curves for one panel, optionally with a shaded +/-1 SD/SE band.

    The panel caption is drawn below the axes by the caller.
    """
    handles = []
    for key in CURVE_ORDER:
        style = CURVE_STYLE[key]
        x, y, yerr = curves[key]
        if show_band:
            ax.fill_between(
                x,
                y - yerr,
                y + yerr,
                color=style["color"],
                alpha=0.15,
                linewidth=0,
                zorder=1,
            )
        (line,) = ax.plot(
            x,
            y,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.3,
            label=style["label"],
            zorder=3,
        )
        handles.append(line)

    ax.axhline(reference_percent, color="#bbbbbb", linestyle=":", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(x_label, labelpad=1.5)
    ax.set_ylabel("Accuracy (%)", labelpad=1.5)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.4)
    ax.tick_params(width=0.5, length=2.5, pad=1.5)
    return handles


def main() -> int:
    """Build and save the combined Figure 5 overview (panels a and b)."""
    maybe_set_plot_style()
    # Font sizes are set for the final FIGURE_WIDTH_INCHES output size.
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 7.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 6.8,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
            "legend.borderpad": 0.3,
            "legend.labelspacing": 0.25,
            "legend.handlelength": 3.0,
            "legend.handletextpad": 0.4,
            "legend.columnspacing": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.5,
        }
    )

    pw_curves = build_powerwash_curves()
    sim_curves, sim_best_task = build_simulation_curves()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))

    _plot_panel(ax_a, pw_curves, x_label="Observation time (min)", reference_percent=50.0)
    handles = _plot_panel(
        ax_b,
        sim_curves,
        x_label="Observed player steps",
        reference_percent=50.0,
        show_band=False,
    )

    labels = [CURVE_STYLE[key]["label"] for key in CURVE_ORDER]
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 1.0),
    )
    legend.get_frame().set_linewidth(0.5)

    # Fixed margins keep the legend and captions inside the canvas at the saved size.
    fig.subplots_adjust(left=0.095, right=0.99, top=0.84, bottom=0.275, wspace=0.32)

    caption_y = 0.03
    for ax, caption in (
        (ax_a, "(a) PowerWash: Future\nNavigation Style Prediction"),
        (ax_b, "(b) Simulation:\nAgent Type Prediction"),
    ):
        pos = ax.get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2.0,
            caption_y,
            caption,
            ha="center",
            va="bottom",
            fontsize=7.0,
            fontweight="bold",
        )

    output_path = DEFAULT_MANUSCRIPT_FIGURES_DIR / "Figure_5.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print("Wrote {} ({}in x {}in)".format(output_path, FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
