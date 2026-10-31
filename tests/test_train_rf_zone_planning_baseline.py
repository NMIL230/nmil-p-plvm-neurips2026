#!/usr/bin/env python3
"""Smoke test for the PowerWash RF outcome baseline trainer.

Trains on a tiny synthetic player_sequences tree (a few players, few trees) and
checks that ``val_predictions.csv`` is written in the PLVM schema and agrees
with the saved ``metrics_by_cutoff.csv``.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from powerwash.train import train_rf_zone_planning_baseline as rf_baseline  # noqa: E402

TASKS = ("back_garden", "bungalow")
SEQUENCE_COLUMNS = [
    "event_type",
    "time_elapsed_since_job_start_raw",
    "LevelProgressionAmount",
    "CurrentPosition",
    "LastSubtaskCompleted",
]


def _write_attempt(path: Path, rng: np.random.Generator, hopper: bool) -> None:
    """Write one synthetic attempt: progress, positions and subtask completions over 2 h."""
    rows = []
    for step in range(1, 25):
        elapsed = step * 300.0
        rows.append(
            {
                "event_type": "subtask_completed" if step % (2 if hopper else 3) == 0 else "tick",
                "time_elapsed_since_job_start_raw": elapsed,
                "LevelProgressionAmount": min(1.0, step / 24.0),
                "CurrentPosition": "({:.2f}, 0.0, {:.2f})".format(
                    rng.normal(scale=10.0 if hopper else 2.0), rng.normal()
                ),
                "LastSubtaskCompleted": "Job.Zone.Subtask{}".format(step),
            }
        )
    pd.DataFrame(rows, columns=SEQUENCE_COLUMNS).to_csv(path, index=False)


class TestTrainRfZonePlanningBaselineSmoke(unittest.TestCase):
    """Run the trainer end to end on synthetic data."""

    def test_val_predictions_written_and_match_cutoff_metrics(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "player_sequences"
            players = ["p{:03d}".format(i) for i in range(12)]
            labels = ["Zone Hopper" if i % 2 else "Zone Planner" for i in range(len(players))]
            for task in TASKS:
                (data_dir / task).mkdir(parents=True)
                for pid, label in zip(players, labels):
                    _write_attempt(
                        data_dir / task / "{}_{}_attempt1.csv".format(pid, task),
                        rng,
                        hopper=(label == "Zone Hopper"),
                    )
            label_csv = root / "labels.csv"
            pd.DataFrame({"player_id": players, "label": labels}).to_csv(label_csv, index=False)
            manifest_csv = root / "player_split_manifest.csv"
            pd.DataFrame(
                {
                    "player_id": players,
                    "split": ["val" if i < 4 else "train" for i in range(len(players))],
                    "fold": "fold_00",
                }
            ).to_csv(manifest_csv, index=False)
            log_base = root / "rf_baselines"

            rf_baseline.main(
                [
                    "--player_label_csv",
                    str(label_csv),
                    "--player_split_manifest",
                    str(manifest_csv),
                    "--data_dir",
                    str(data_dir),
                    "--task_names",
                    ",".join(TASKS),
                    "--target_mode",
                    "player_label_2_class_clean_zonehopping",
                    "--n_estimators",
                    "5",
                    "--log_dir_base",
                    str(log_base),
                    "--model_performance_csv",
                    str(log_base / "model_performance.csv"),
                ]
            )

            run_dirs = [p for p in log_base.iterdir() if p.is_dir()]
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            predictions_path = run_dir / rf_baseline.VAL_PREDICTIONS_FILENAME
            self.assertTrue(predictions_path.exists())
            predictions = pd.read_csv(predictions_path)
            metrics = pd.read_csv(
                run_dir / "plots" / "val" / "cutoff_metrics" / "metrics_by_cutoff.csv"
            )

            self.assertEqual(list(predictions.columns), rf_baseline.VAL_PREDICTIONS_FIELDS)
            self.assertEqual(len(predictions), int(metrics["support"].sum()))
            self.assertEqual(set(predictions["player_id"]), set(players[:4]))
            self.assertEqual(set(predictions["model_name"]), {run_dir.name})
            # Every validation player appears once per cutoff.
            self.assertEqual(
                len(predictions),
                4 * len(rf_baseline.BOTH_PARTIAL_CUTOFF_SECONDS),
            )

            # Accuracy recomputed from per-player predictions equals the saved cutoff metric.
            correct = predictions["prediction_class_id"] == predictions["target"]
            recomputed = correct.groupby(predictions["observation_cutoff_seconds"]).mean()
            saved = metrics.set_index("cutoff_value")["accuracy"]
            pd.testing.assert_series_equal(
                recomputed.sort_index(),
                saved.sort_index(),
                check_names=False,
                check_index_type=False,
            )

            # The predicted class is the most probable class, and ``prediction`` is its probability.
            probabilities = np.array(
                [json.loads(value) for value in predictions["prediction_probabilities_json"]]
            )
            np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
            np.testing.assert_allclose(predictions["prediction"], probabilities.max(axis=1))
            np.testing.assert_array_equal(
                probabilities.argmax(axis=1),
                predictions["prediction_class_id"].to_numpy(),
            )


if __name__ == "__main__":
    unittest.main()
