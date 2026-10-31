"""Smoke tests for the Figure 3a and 3c scripts (Fire Station navigation-style panels)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from paper.figures import fig03a_fig03c_firestation_label_panels as firestation_panels  # noqa: E402


class TestPlotNavigationStylePanels(unittest.TestCase):
    """Exercise the Figure 3a and 3c panel scripts on minimal local artifacts."""

    def test_firestation_resolve_output_paths_uses_figure_numbers(self):
        output_dir = Path("/tmp/manuscript-figures")

        zones_path, score_path = firestation_panels.resolve_fire_station_output_paths(
            output_dir=output_dir,
            figure_3a_pdf=None,
            figure_3c_pdf=None,
        )

        self.assertEqual(zones_path.name, "Figure_3a.pdf")
        self.assertEqual(score_path.name, "Figure_3c.pdf")

    def test_firestation_main_writes_requested_figure_pdfs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_dir = root / "firestation"
            output_dir = root / "paper" / "figures"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {
                        "player_id": "planner_a",
                        "label": "Zone Planner",
                        "label_2_class": "Zone Planner",
                        "label_2_class_clean": "Zone Planner",
                        "zone_hopping_score": -1.4,
                    },
                    {
                        "player_id": "mixed_a",
                        "label": "Mixed",
                        "label_2_class": "Zone Hopper",
                        "label_2_class_clean": "",
                        "zone_hopping_score": 0.1,
                    },
                    {
                        "player_id": "hopper_a",
                        "label": "Zone Hopper",
                        "label_2_class": "Zone Hopper",
                        "label_2_class_clean": "Zone Hopper",
                        "zone_hopping_score": 1.6,
                    },
                ]
            ).to_csv(artifact_dir / "firestation_zone_labels.csv", index=False)
            pd.DataFrame(
                [
                    {"subtask_id": 1, "subtask_name": "alpha", "x": -1.0, "z": -1.0, "zone_id": 0},
                    {"subtask_id": 2, "subtask_name": "beta", "x": -0.7, "z": -1.2, "zone_id": 0},
                    {"subtask_id": 3, "subtask_name": "gamma", "x": 1.0, "z": 1.1, "zone_id": 1},
                    {"subtask_id": 4, "subtask_name": "delta", "x": 1.3, "z": 0.9, "zone_id": 1},
                ]
            ).to_csv(artifact_dir / "firestation_subtask_zone_map.csv", index=False)
            pd.DataFrame(
                [
                    {"zone_id": 0, "centroid_x": -0.85, "centroid_z": -1.1},
                    {"zone_id": 1, "centroid_x": 1.15, "centroid_z": 1.0},
                ]
            ).to_csv(artifact_dir / "firestation_zone_centroids.csv", index=False)

            exit_code = firestation_panels.main(
                [
                    "--fire_station_artifact_dir",
                    str(artifact_dir),
                    "--output_dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "Figure_3a.pdf").exists())
            self.assertTrue((output_dir / "Figure_3c.pdf").exists())


class TestScoreDistributionThresholds(unittest.TestCase):
    """Figure 3c draws the class thresholds when label_thresholds.json is present."""

    def test_load_label_thresholds(self):
        import json

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.assertIsNone(firestation_panels.load_label_thresholds(root))
            (root / "label_thresholds.json").write_text(
                json.dumps({"lower_threshold": -0.75, "upper_threshold": 1.14})
            )
            self.assertEqual(firestation_panels.load_label_thresholds(root), (-0.75, 1.14))


if __name__ == "__main__":
    unittest.main()
