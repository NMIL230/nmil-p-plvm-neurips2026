#!/usr/bin/env python3
"""Tests for ``--attempt_policy``: which observed-task attempt each player contributes.

Covers the PLVM data utilities (sequence listing, all-task coverage, cache key) and the
RF baseline's per-player attempt selection, on tiny synthetic sequence folders.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from powerwash.train import powerwash_plvm_data_utils as plvm_data  # noqa: E402
from powerwash.train import train_rf_zone_planning_baseline as rf_baseline  # noqa: E402


def _write_sequence(path: Path, final_progress: float) -> None:
    """Write a minimal processed sequence: two subtask events ending at ``final_progress``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "event_type": ["subtask_completed", "subtask_completed"],
            "time_elapsed_since_job_start": [60.0, 120.0],
            "time_elapsed_since_job_start_raw": [60.0, 120.0],
            "LevelProgressionAmount": [final_progress / 2, final_progress],
            "CurrentPosition": ["(0.00, 0.00, 0.00)", "(1.00, 0.00, 1.00)"],
            "LastSubtaskCompleted": ["Subtask_0", "Subtask_1"],
        }
    ).to_csv(path, index=False)


def _build_tree(root: Path) -> None:
    """p1: attempts 1 and 2 of both tasks; p2: Back Garden attempt 2 only; p3: attempt 1 only."""
    for task in ("back_garden", "bungalow"):
        _write_sequence(root / task / "p1_{}_attempt1.csv".format(task), 0.5)
        _write_sequence(root / task / "p1_{}_attempt2.csv".format(task), 1.0)
        _write_sequence(root / task / "p3_{}_attempt1.csv".format(task), 1.0)
    _write_sequence(root / "back_garden" / "p2_back_garden_attempt2.csv", 1.0)
    _write_sequence(root / "bungalow" / "p2_bungalow_attempt1.csv", 1.0)


class TestPlvmAttemptPolicy(unittest.TestCase):
    """Sequence listing and coverage under the two PLVM attempt policies."""

    def test_first_policy_lists_attempt_one_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_tree(root)
            all_items = plvm_data.list_sequences_for_attempt_policy(root, ["back_garden"], "all")
            first_items = plvm_data.list_sequences_for_attempt_policy(
                root, ["back_garden"], "first"
            )
            self.assertEqual(len(all_items), 4)
            self.assertEqual(
                sorted((item.pid, item.attempt_id) for item in first_items),
                [("p1", "1"), ("p3", "1")],
            )

    def test_all_task_coverage_requires_attempt_one_in_every_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_tree(root)
            tasks = ["back_garden", "bungalow"]
            self.assertEqual(
                plvm_data.find_players_with_usable_sequences_for_all_tasks(root, tasks, "all"),
                {"p1", "p2", "p3"},
            )
            # p2 has no Back Garden attempt 1, so it drops out.
            self.assertEqual(
                plvm_data.find_players_with_usable_sequences_for_all_tasks(root, tasks, "first"),
                {"p1", "p3"},
            )

    def test_unknown_policy_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_tree(root)
            with self.assertRaises(ValueError):
                plvm_data.list_sequences_for_attempt_policy(root, ["back_garden"], "latest")

    def test_cache_key_default_equals_all_and_differs_for_first(self):
        kwargs = dict(
            spec_path="spec.json",
            target_mode="player_label_3_class_zonehopping",
            target_col="",
            data_dir="data",
            seq_length=120,
            model_type="plvm_both_partial",
            embeddings_to_use="masked_average_all",
            full_task_name="bungalow",
            partial_task_name="back_garden",
            max_files_per_task=2500,
            sampling_seed=42,
            task_names_filter="back_garden",
        )
        default_key = plvm_data._plvm_cache_key(**kwargs)
        self.assertEqual(default_key, plvm_data._plvm_cache_key(attempt_policy="all", **kwargs))
        self.assertNotEqual(
            default_key, plvm_data._plvm_cache_key(attempt_policy="first", **kwargs)
        )


class TestRfAttemptPolicy(unittest.TestCase):
    """The RF baseline picks attempt 1 or the highest-completion attempt."""

    def test_first_and_best_policies(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "back_garden"
            _build_tree(Path(tmp))
            players = {"p1", "p2", "p3"}
            best = rf_baseline._load_selected_attempt_events(task_dir, "back_garden", players, "best")
            first = rf_baseline._load_selected_attempt_events(task_dir, "back_garden", players, "first")
            # p1's attempt 2 reaches completion 1.0, attempt 1 only 0.5.
            self.assertEqual(max(o.completion for o in best["p1"]), 1.0)
            self.assertEqual(max(o.completion for o in first["p1"]), 0.5)
            # p2 has only attempt 2: kept by "best", absent under "first".
            self.assertIn("p2", best)
            self.assertNotIn("p2", first)
            self.assertIn("p3", first)

    def test_cli_default_is_first(self):
        args = rf_baseline.parse_args(
            [
                "--player_label_csv",
                "labels.csv",
                "--player_split_manifest",
                "manifest.csv",
                "--data_dir",
                "data",
                "--task_names",
                "back_garden",
                "--target_mode",
                "player_label_3_class_zonehopping",
            ]
        )
        self.assertEqual(args.attempt_policy, "first")


class TestPlvmEarlyStoppingFlag(unittest.TestCase):
    """Early stopping is on by default in the PLVM trainer and can be switched off."""

    def test_help_lists_both_forms(self):
        import subprocess

        script = Path(__file__).resolve().parents[1] / "powerwash/train/train_plvm_powerwash.py"
        out = subprocess.run(
            [sys.executable, str(script), "--help"], capture_output=True, text=True, check=True
        ).stdout
        self.assertIn("--use_early_stopping, --no-use_early_stopping", out)
        self.assertIn("default: on", " ".join(out.split()))


if __name__ == "__main__":
    unittest.main()
