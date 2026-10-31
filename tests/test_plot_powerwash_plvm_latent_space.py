#!/usr/bin/env python3
"""Focused tests for the saved PowerWash latent-space plotting script."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from paper.figures.fig05c_latent_space import (  # noqa: E402
    _axis_limits,
    attach_run_metadata,
    filter_frame_to_cutoff,
    load_latent_plot_frame,
    manuscript_figure_size_inches,
    resolve_manuscript_output_pdf,
    resolve_output_paths,
    sample_frame_for_scatter,
    select_best_run,
)


class TestPlotPowerWashPlvmLatentSpace(unittest.TestCase):
    """Exercise the artifact-driven latent-space helpers on tiny synthetic data."""

    def test_select_best_run_prefers_highest_val_accuracy_with_latent_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            good_dir = root / "good-run" / "plots"
            good_dir.mkdir(parents=True)
            (good_dir / "train_classification_latent_space_plot_data.csv").write_text(
                "x\n1\n", encoding="utf-8"
            )
            bad_dir = root / "bad-run"
            bad_dir.mkdir(parents=True)
            rows = [
                {
                    "model_name": "bad-run",
                    "model_type": "plvm_both_partial",
                    "parameters_to_predict": "player_label_2_class_clean_zonehopping_label",
                    "latent_dim": "2",
                    "val_accuracy": "0.90",
                },
                {
                    "model_name": "good-run",
                    "model_type": "plvm_both_partial",
                    "parameters_to_predict": "player_label_2_class_clean_zonehopping_label",
                    "latent_dim": "2",
                    "val_accuracy": "0.71",
                },
            ]

            selected = select_best_run(
                rows=rows,
                training_logs_dir=root,
                parameters_to_predict="player_label_2_class_clean_zonehopping_label",
                model_type="plvm_both_partial",
                latent_dim=2,
            )

            self.assertEqual(selected["model_name"], "good-run")

    def test_load_latent_plot_frame_joins_candidate_metadata_by_sample_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            plots_dir = run_dir / "plots"
            plots_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "sample_index": 0,
                        "split": "train",
                        "model_name": "demo",
                        "latent_x": 0.1,
                        "latent_y": 0.2,
                        "true_class_id": 0,
                        "true_class_label": "Zone Planner",
                        "predicted_class_id": 1,
                        "predicted_class_label": "Zone Hopper",
                        "predicted_prob_0": 0.4,
                        "predicted_prob_1": 0.6,
                    },
                    {
                        "sample_index": 1,
                        "split": "train",
                        "model_name": "demo",
                        "latent_x": 0.3,
                        "latent_y": 0.4,
                        "true_class_id": 1,
                        "true_class_label": "Zone Hopper",
                        "predicted_class_id": 1,
                        "predicted_class_label": "Zone Hopper",
                        "predicted_prob_0": 0.2,
                        "predicted_prob_1": 0.8,
                    },
                ]
            ).to_csv(plots_dir / "train_classification_latent_space_plot_data.csv", index=False)
            (plots_dir / "train_classification_latent_space_plot_metadata.json").write_text(
                json.dumps(
                    {
                        "split": "train",
                        "model_name": "demo",
                        "latent_dim": 2,
                        "class_names": ["Zone Planner", "Zone Hopper"],
                        "probability_columns": ["predicted_prob_0", "predicted_prob_1"],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "candidate_splits.json").write_text(
                json.dumps(
                    {
                        "train": [
                            {
                                "pid": "p1",
                                "task_name": "back_garden",
                                "cutoff_seconds": 300.0,
                                "target_value": 0.0,
                            },
                            {
                                "pid": "p2",
                                "task_name": "back_garden",
                                "cutoff_seconds": 600.0,
                                "target_value": 1.0,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            frame, metadata = load_latent_plot_frame(run_dir)

            self.assertEqual(metadata["split"], "train")
            self.assertEqual(frame["pid"].tolist(), ["p1", "p2"])
            self.assertEqual(frame["cutoff_seconds"].tolist(), [300.0, 600.0])

    def test_filter_frame_to_cutoff_keeps_one_row_per_player(self):
        frame = pd.DataFrame(
            [
                {"pid": "p1", "cutoff_seconds": 1800.0, "latent_x": 0.1},
                {"pid": "p1", "cutoff_seconds": 2100.0, "latent_x": 0.2},
                {"pid": "p2", "cutoff_seconds": 1800.0, "latent_x": 0.3},
            ]
        )

        filtered = filter_frame_to_cutoff(frame, 1800.0)

        self.assertEqual(filtered["pid"].tolist(), ["p1", "p2"])

    def test_sample_frame_for_scatter_preserves_both_labels(self):
        frame = pd.DataFrame(
            [
                {
                    "sample_index": idx,
                    "true_class_label": "Zone Planner" if idx < 8 else "Zone Hopper",
                }
                for idx in range(12)
            ]
        )

        sampled = sample_frame_for_scatter(frame, max_points=4, random_seed=7)

        self.assertLessEqual(len(sampled), 4)
        self.assertIn("Zone Planner", set(sampled["true_class_label"]))
        self.assertIn("Zone Hopper", set(sampled["true_class_label"]))

    def test_resolve_output_paths_defaults_to_best_ld2_stem(self):
        pdf_path, png_path, csv_path = resolve_output_paths(
            output_dir=Path("/tmp/out"),
            cutoff_seconds=1800.0,
            output_pdf=None,
            output_png=None,
            output_csv=None,
        )

        self.assertEqual(pdf_path.name, "powerwash_plvm_best_ld2_latent_space_1800s.pdf")
        self.assertEqual(png_path.name, "powerwash_plvm_best_ld2_latent_space_1800s.png")
        self.assertEqual(csv_path.name, "powerwash_plvm_best_ld2_latent_space_1800s.csv")

    def test_resolve_output_pdf_defaults_to_figure_5c(self):
        manuscript_pdf = resolve_manuscript_output_pdf(None)

        self.assertEqual(manuscript_pdf.name, "Figure_5c.pdf")
        self.assertEqual(manuscript_pdf.parent.name, "figures")

    def test_figure_size_matches_requested_dimensions(self):
        width, height = manuscript_figure_size_inches()

        # Figure 5c in the paper is 2.13 x 2.2 in.
        self.assertAlmostEqual(width, 2.13)
        self.assertAlmostEqual(height, 2.2)

    def test_axis_limits_use_shared_square_extent(self):
        frame = pd.DataFrame(
            [
                {"latent_x": -2.0, "latent_y": -1.0},
                {"latent_x": 3.0, "latent_y": 1.0},
            ]
        )

        x_min, x_max, y_min, y_max = _axis_limits(frame)

        self.assertAlmostEqual(x_max - x_min, y_max - y_min)

    def test_attach_run_metadata_adds_selected_run_fields(self):
        frame = pd.DataFrame([{"sample_index": 0, "latent_x": 0.1}])

        enriched = attach_run_metadata(
            frame,
            {
                "model_name": "demo-run",
                "model_type": "plvm_both_partial",
                "parameters_to_predict": "player_label_2_class_clean_zonehopping_label",
                "val_accuracy": "0.715",
            },
        )

        self.assertEqual(enriched.loc[0, "selected_model_name"], "demo-run")
        self.assertAlmostEqual(enriched.loc[0, "selected_val_accuracy"], 0.715)


if __name__ == "__main__":
    unittest.main()
