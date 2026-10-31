"""Tests for the simulation PLVM performance plotting script."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from paper.figures.fig07_simulation_accuracy_curves import (
    _resolve_named_manuscript_curve_path,
    main as plot_simulation_main,
)


def _write_cutoff_metrics(run_dir: Path, split_name: str, values) -> None:
    cutoff_dir = run_dir / "plots" / split_name / "cutoff_metrics"
    cutoff_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "cutoff_kind": "observation_steps",
                "cutoff_value": cutoff_value,
                "support": 100,
                "accuracy": accuracy,
                "macro_f1": accuracy,
                "cross_entropy": 0.6,
                "target_class_counts_json": "[50, 50]",
                "predicted_class_counts_json": "[50, 50]",
                "class_names_json": '["Planner", "Hopper"]',
            }
            for cutoff_value, accuracy in values
        ]
    ).to_csv(cutoff_dir / "metrics_by_cutoff.csv", index=False)


def test_plot_simulation_curves_smoke(tmp_path):
    process_logs = tmp_path / "plvm_label" / "training_logs"
    outcome_logs = tmp_path / "outcome" / "training_logs"
    process_logs.mkdir(parents=True, exist_ok=True)
    outcome_logs.mkdir(parents=True, exist_ok=True)

    process_rows = [
        {
            "model_name": "proc-foraging",
            "model_type": "single_task",
            "predicted_next_task": "foraging",
            "prediction_head_type": "mlp",
            "val_accuracy": 0.71,
            "train_accuracy": 0.79,
            "naive_val_accuracy": 0.5,
            "naive_train_accuracy": 0.5,
            "seq_length": 200,
            "latent_dim": 1,
            "parameters_to_predict": "minigrid_player_label",
            "run_mode": "run",
            "all_arguments_json": json.dumps(
                {
                    "max_train_val_players": 4000,
                    "run_mode": "run",
                    "seq_length": 200,
                }
            ),
        },
        {
            "model_name": "proc-cleaning",
            "model_type": "single_task",
            "predicted_next_task": "cleaning",
            "prediction_head_type": "mlp",
            "val_accuracy": 0.68,
            "train_accuracy": 0.77,
            "naive_val_accuracy": 0.5,
            "naive_train_accuracy": 0.5,
            "seq_length": 200,
            "latent_dim": 1,
            "parameters_to_predict": "minigrid_player_label",
            "run_mode": "run",
            "all_arguments_json": json.dumps(
                {
                    "max_train_val_players": 4000,
                    "run_mode": "run",
                    "seq_length": 200,
                }
            ),
        },
        {
            "model_name": "proc-plvm2",
            "model_type": "plvm_both_partial",
            "predicted_next_task": "cleaning,foraging",
            "prediction_head_type": "hierarchical",
            "val_accuracy": 0.82,
            "train_accuracy": 0.9,
            "naive_val_accuracy": 0.5,
            "naive_train_accuracy": 0.5,
            "seq_length": 200,
            "latent_dim": 2,
            "parameters_to_predict": "minigrid_player_label",
            "run_mode": "run",
            "all_arguments_json": json.dumps(
                {
                    "max_train_val_players": 4000,
                    "run_mode": "run",
                    "seq_length": 200,
                    "latent_dim": 2,
                }
            ),
        },
    ]
    outcome_rows = [
        {
            "model_name": "out-foraging",
            "model_type": "single_task",
            "predicted_next_task": "foraging",
            "prediction_head_type": "logistic_regression",
            "val_accuracy": 0.61,
            "train_accuracy": 0.64,
            "naive_val_accuracy": 0.5,
            "naive_train_accuracy": 0.5,
            "seq_length": 0,
            "latent_dim": 0,
            "parameters_to_predict": "minigrid_player_label",
            "run_mode": "run",
            "all_arguments_json": json.dumps(
                {
                    "max_train_val_players": 4000,
                    "run_mode": "run",
                }
            ),
        },
        {
            "model_name": "out-cleaning",
            "model_type": "single_task",
            "predicted_next_task": "cleaning",
            "prediction_head_type": "logistic_regression",
            "val_accuracy": 0.63,
            "train_accuracy": 0.66,
            "naive_val_accuracy": 0.5,
            "naive_train_accuracy": 0.5,
            "seq_length": 0,
            "latent_dim": 0,
            "parameters_to_predict": "minigrid_player_label",
            "run_mode": "run",
            "all_arguments_json": json.dumps(
                {
                    "max_train_val_players": 4000,
                    "run_mode": "run",
                }
            ),
        },
        {
            "model_name": "out-both",
            "model_type": "plvm_both_partial",
            "predicted_next_task": "cleaning,foraging",
            "prediction_head_type": "logistic_regression",
            "val_accuracy": 0.67,
            "train_accuracy": 0.7,
            "naive_val_accuracy": 0.5,
            "naive_train_accuracy": 0.5,
            "seq_length": 0,
            "latent_dim": 0,
            "parameters_to_predict": "minigrid_player_label",
            "run_mode": "run",
            "all_arguments_json": json.dumps(
                {
                    "max_train_val_players": 4000,
                    "run_mode": "run",
                }
            ),
        },
    ]

    pd.DataFrame(process_rows).to_csv(process_logs / "model_performance.csv", index=False)
    pd.DataFrame(outcome_rows).to_csv(outcome_logs / "model_performance.csv", index=False)

    for run_name in ["proc-foraging", "proc-cleaning", "proc-plvm2"]:
        run_dir = process_logs / run_name
        _write_cutoff_metrics(run_dir, "train", [(20, 0.58), (40, 0.65), (60, 0.72)])
        _write_cutoff_metrics(run_dir, "val", [(20, 0.54), (40, 0.62), (60, 0.69)])

    for run_name in ["out-foraging", "out-cleaning", "out-both"]:
        run_dir = outcome_logs / run_name
        _write_cutoff_metrics(run_dir, "train", [(20, 0.52), (40, 0.57), (60, 0.63)])
        _write_cutoff_metrics(run_dir, "val", [(20, 0.5), (40, 0.56), (60, 0.61)])

    output_dir = tmp_path / "analysis_outputs"
    exit_code = plot_simulation_main(
        [
            "--plvm_model_performance_csv",
            str(process_logs / "model_performance.csv"),
            "--outcome_model_performance_csv",
            str(outcome_logs / "model_performance.csv"),
            "--plvm_training_logs_dir",
            str(process_logs),
            "--outcome_training_logs_dir",
            str(outcome_logs),
            "--seq_length",
            "200",
            "--max_train_val_players",
            "4000",
            "--output_dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0

    curve_pdf = output_dir / "minigrid_label_prediction_seq200_val_curves.pdf"
    bar_pdf = output_dir / "minigrid_label_prediction_seq200_val_bar_summary.pdf"
    long_csv = output_dir / "minigrid_label_prediction_seq200_cutoff_metrics.csv"
    runs_csv = output_dir / "minigrid_label_prediction_seq200_selected_runs.csv"
    for path in [curve_pdf, bar_pdf, long_csv, runs_csv]:
        assert path.exists(), str(path)

    selected_runs = pd.read_csv(runs_csv)
    assert set(selected_runs["condition_label"]) == {
        "Process: Foraging",
        "Process: Cleaning",
        "Process: PLVM-2",
        "Outcome: Foraging",
        "Outcome: Cleaning",
        "Outcome: Both",
    }

    manuscript_dir = tmp_path / "paper" / "figures"
    manuscript_output_dir = tmp_path / "analysis_outputs_manuscript"
    exit_code = plot_simulation_main(
        [
            "--plvm_model_performance_csv",
            str(process_logs / "model_performance.csv"),
            "--outcome_model_performance_csv",
            str(outcome_logs / "model_performance.csv"),
            "--plvm_training_logs_dir",
            str(process_logs),
            "--outcome_training_logs_dir",
            str(outcome_logs),
            "--seq_length",
            "200",
            "--max_train_val_players",
            "4000",
            "--output_dir",
            str(manuscript_output_dir),
            "--manuscript_figure",
            "--manuscript_figure_dir",
            str(manuscript_dir),
        ]
    )
    assert exit_code == 0
    assert (manuscript_output_dir / "minigrid_label_prediction_seq200_val_curves.pdf").exists()
    assert (manuscript_output_dir / "minigrid_label_prediction_seq200_val_bar_summary.pdf").exists()
    assert (manuscript_output_dir / "minigrid_label_prediction_seq200_cutoff_metrics.csv").exists()
    assert (manuscript_output_dir / "minigrid_label_prediction_seq200_selected_runs.csv").exists()
    assert not any(manuscript_dir.iterdir()) if manuscript_dir.exists() else True


def test_resolve_named_curve_path_only_targets_8000_players():
    assert _resolve_named_manuscript_curve_path(8000).name == "Figure_7.pdf"
    assert _resolve_named_manuscript_curve_path(4000) is None
