#!/usr/bin/env python3
"""Focused tests for the PowerWash performance-curve helpers (Figures 6 and 8-10)."""

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from paper.output_paths import (  # noqa: E402
    copy_figure_to_path,
    resolve_tables_dir,
)
from paper.figures.fig06_fig08_fig09_fig10_powerwash_accuracy_curves import (  # noqa: E402
    _resolve_named_manuscript_curve_path,
    _resolve_named_manuscript_multi_plvm_curve_path,
    build_family_summary_rows,
    mean_naive_baseline_accuracy,
    write_family_summary_table,
)


class TestPowerWashPlvmPerformanceCurves(unittest.TestCase):
    """Exercise the shared naive-baseline helper used by curves and bars."""

    def test_mean_naive_baseline_accuracy_pools_all_fold_values(self):
        naive_accs = {
            "A": [0.40, 0.50],
            "B": [0.60],
        }

        baseline = mean_naive_baseline_accuracy(naive_accs)

        self.assertAlmostEqual(baseline, 0.50)

    def test_mean_naive_baseline_accuracy_returns_nan_when_empty(self):
        baseline = mean_naive_baseline_accuracy({})

        self.assertTrue(math.isnan(baseline))

    def test_copy_figure_to_path_writes_requested_filename(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_figure = tmp_path / "analysis" / "paper_figures" / "source.pdf"
            source_figure.parent.mkdir(parents=True, exist_ok=True)
            source_figure.write_bytes(b"%PDF-1.4 renamed copy test\n")

            destination_figure = tmp_path / "paper" / "figures" / "Figure_5a.pdf"
            copied_figure = copy_figure_to_path(source_figure, destination_figure)

            self.assertEqual(copied_figure, destination_figure)
            self.assertTrue(copied_figure.exists())
            self.assertEqual(copied_figure.read_bytes(), source_figure.read_bytes())

    def test_resolve_tables_dir_honors_override(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            override = Path(tmp_dir) / "tables"

            resolved = resolve_tables_dir(override)

            self.assertEqual(resolved, override)

    def test_resolve_named_curve_path_matches_requested_targets(self):
        self.assertEqual(
            _resolve_named_manuscript_curve_path("player_label_2_class_clean_zonehopping").name,
            "Figure_6.pdf",
        )
        self.assertEqual(
            _resolve_named_manuscript_curve_path("player_label_3_class_zonehopping").name,
            "Figure_8.pdf",
        )
        self.assertIsNone(
            _resolve_named_manuscript_curve_path("player_label_2_class_clean_entropy")
        )

    def test_resolve_named_multi_plvm_curve_path_matches_requested_targets(self):
        self.assertEqual(
            _resolve_named_manuscript_multi_plvm_curve_path(
                "player_label_2_class_clean_zonehopping"
            ).name,
            "Figure_9.pdf",
        )
        self.assertEqual(
            _resolve_named_manuscript_multi_plvm_curve_path(
                "player_label_3_class_zonehopping"
            ).name,
            "Figure_10.pdf",
        )
        self.assertIsNone(
            _resolve_named_manuscript_multi_plvm_curve_path("player_label_2_class_clean_entropy")
        )

    def test_build_family_summary_rows_selects_best_condition_per_family(self):
        condition_meta = [
            {"condition": "RF Back Garden", "model_type": "rf_baseline", "num_classes": 2},
            {"condition": "RF Both", "model_type": "rf_baseline", "num_classes": 2},
            {"condition": "Back Garden Alone", "model_type": "single_task", "num_classes": 2},
            {"condition": "Bungalow Alone", "model_type": "single_task", "num_classes": 2},
            {"condition": "PLVM-1", "model_type": "plvm_both_partial", "num_classes": 2},
            {"condition": "PLVM-2", "model_type": "plvm_both_partial", "num_classes": 2},
        ]
        rows = []
        fold_accuracies = {
            "RF Back Garden": {
                "fold_00": [0.50, 0.51, 0.52],
                "fold_01": [0.51, 0.52, 0.53],
            },
            "RF Both": {
                "fold_00": [0.54, 0.55, 0.56],
                "fold_01": [0.56, 0.57, 0.58],
            },
            "Back Garden Alone": {
                "fold_00": [0.58, 0.60, 0.61],
                "fold_01": [0.60, 0.62, 0.63],
            },
            "Bungalow Alone": {
                "fold_00": [0.59, 0.61, 0.62],
                "fold_01": [0.61, 0.63, 0.64],
            },
            "PLVM-1": {
                "fold_00": [0.60, 0.62, 0.63],
                "fold_01": [0.61, 0.63, 0.64],
            },
            "PLVM-2": {
                "fold_00": [0.62, 0.64, 0.66],
                "fold_01": [0.64, 0.66, 0.68],
            },
        }
        cutoff_values = [1200.0, 3600.0, 7200.0]
        for condition, folds in fold_accuracies.items():
            for fold_id, values in folds.items():
                for cutoff_value, accuracy in zip(cutoff_values, values):
                    rows.append(
                        {
                            "split": "val",
                            "condition": condition,
                            "fold_id": fold_id,
                            "cutoff_value": cutoff_value,
                            "accuracy": accuracy,
                        }
                    )

        fold_frame = pd.DataFrame(rows)

        summary_rows = build_family_summary_rows(fold_frame, condition_meta)

        self.assertEqual(
            [row["Model family"] for row in summary_rows],
            ["Chance baseline", "Outcome baseline", "Single-task process", "PLVM"],
        )
        self.assertEqual(summary_rows[0]["20 min Acc."], "50.0")
        self.assertEqual(summary_rows[0]["AUC at 120 min"], "0.50")
        self.assertEqual(summary_rows[1]["120 min Acc."], "57.0 ± 1.4")
        self.assertEqual(summary_rows[2]["60 min Acc."], "62.0 ± 1.4")
        self.assertEqual(summary_rows[3]["20 min Acc."], "63.0 ± 1.4")
        self.assertEqual(summary_rows[3]["AUC at 120 min"], "0.65 ± 0.01")

    def test_write_family_summary_table_persists_rows(self):
        rows = [
            {
                "Model family": "Chance baseline",
                "Inputs": "Class prior",
                "20 min Acc.": "50.0",
                "60 min Acc.": "50.0",
                "120 min Acc.": "50.0",
                "AUC at 120 min": "0.50",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_csv = Path(tmp_dir) / "tables" / "summary.csv"

            write_family_summary_table(rows, output_csv)

            self.assertTrue(output_csv.exists())
            written = pd.read_csv(output_csv, dtype=str)
            self.assertEqual(written.loc[0, "Model family"], "Chance baseline")
            self.assertEqual(written.loc[0, "AUC at 120 min"], "0.50")


if __name__ == "__main__":
    unittest.main()
