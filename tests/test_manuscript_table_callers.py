#!/usr/bin/env python3
"""Focused tests for the Table 1 and Table 2 helpers and the shared table output helper."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from paper.tables.table01_outcome_proxy import (  # noqa: E402
    build_manuscript_table_frame as build_table_1_frame,
    resolve_source_tasks,
)
from paper.tables.table02_prediction_summary import (  # noqa: E402
    build_table2_frame,
)
from paper.table_output_utils import write_latex_table  # noqa: E402


class TestTableOutputUtils(unittest.TestCase):
    """Exercise the shared numbered table output helpers."""

    def test_write_latex_table_escapes_percent_and_plus_minus(self):
        frame = pd.DataFrame(
            [
                {
                    "Metric (%)": "Back Garden completion (%)",
                    "Future Planner": "41.5 ± 19.2",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "Table_1.tex"

            write_latex_table(frame, output_path, column_alignment="lc")

            text = output_path.read_text(encoding="utf-8")
            self.assertIn(r"Metric (\%)", text)
            self.assertIn(r"Back Garden completion (\%)", text)
            self.assertIn(r"41.5 $\pm$ 19.2", text)


class TestGenerateTable1Helpers(unittest.TestCase):
    """Exercise the light table-specific helpers for Table 1."""

    def test_resolve_source_tasks_preserves_order_and_deduplicates(self):
        resolved = resolve_source_tasks("back_garden,bungalow,back_garden")

        self.assertEqual(resolved, ["back_garden", "bungalow"])

    def test_build_table_1_frame_renames_metric_column_for_cutoff(self):
        rows = [
            {
                "20-min observed-task outcome metric": "Back Garden completion (%)",
                "Future Planner": "41.5 ± 19.2",
                "Future Hopper": "43.3 ± 23.1",
                "Std. diff.": "0.08",
                "Single-metric AUC": "0.51",
            }
        ]

        table_frame = build_table_1_frame(rows, cutoff_seconds=1800.0)

        self.assertEqual(
            list(table_frame.columns),
            [
                "30-min observed-task outcome metric",
                "Future Planner",
                "Future Hopper",
                "Std. diff.",
                "Single-metric AUC",
            ],
        )

    def test_outcome_proxy_features_module_imports_cleanly(self):
        from powerwash.analysis import outcome_proxy_features as module

        self.assertEqual(module.LABEL_COLORS["Zone Planner"], "#377eb8")


class TestGenerateTable2Helpers(unittest.TestCase):
    """Exercise the PDF-layout formatting of Table 2."""

    def test_build_table2_frame_preserves_row_order(self):
        metrics = ["20 min accuracy", "60 min accuracy", "120 min accuracy", "AUC at 120 min"]
        records = []
        for metric in metrics:
            records.append(
                {
                    "row": "PLVM",
                    "kind": "accuracy",
                    "metric": metric,
                    "mean": 0.616,
                    "sd": 0.046,
                    "wilcoxon_p": float("nan"),
                }
            )
        for metric in metrics:
            records.append(
                {
                    "row": "Delta PLVM - Bungalow",
                    "kind": "delta",
                    "metric": metric,
                    "mean": 0.0377,
                    "sd": float("nan"),
                    "wilcoxon_p": 0.01953125,
                }
            )
        for metric in metrics[:3]:
            records.append(
                {
                    "row": "Majority class",
                    "kind": "majority",
                    "metric": metric,
                    "mean": 0.5236,
                    "sd": 0.017,
                    "wilcoxon_p": float("nan"),
                }
            )

        table_frame = build_table2_frame(pd.DataFrame(records))

        self.assertEqual(
            table_frame["Model"].tolist(), ["PLVM", "Delta PLVM - Bungalow", "Majority class"]
        )
        self.assertEqual(table_frame.loc[0, "20 min"], "61.6 ± 4.6")
        self.assertEqual(table_frame.loc[1, "120 min"], "+3.8 (p=0.020)")
        self.assertEqual(table_frame.loc[2, "60 min"], "52.4")
        self.assertEqual(table_frame.loc[2, "Mean acc. (5-120 min)"], "--")


if __name__ == "__main__":
    unittest.main()
