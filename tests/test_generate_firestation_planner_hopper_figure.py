"""Smoke tests for the Figure 3b script (example Zone Planner and Zone Hopper)."""

import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from paper.figures import fig03b_planner_hopper_examples as script  # noqa: E402


class TestGenerateFirestationPlannerHopperFigure(unittest.TestCase):
    """Exercise the Figure 3b entrypoint."""

    def test_resolve_output_path_uses_figure_default_name(self):
        output_path = script.resolve_output_path(Path("/tmp/manuscript-figures"), None)
        self.assertEqual(output_path.name, "Figure_3b.pdf")

    def test_main_writes_requested_figure_pdf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_dir = root / "firestation"
            output_dir = root / "paper" / "figures"
            cache_path = root / "firestation_player_records.pkl"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {"player_id": "planner_a", "label": "Zone Planner", "zone_hopping_score": -8.6},
                    {"player_id": "hopper_a", "label": "Zone Hopper", "zone_hopping_score": 12.1},
                ]
            ).to_csv(artifact_dir / "firestation_zone_labels.csv", index=False)
            pd.DataFrame(
                [
                    {"subtask_id": 1, "subtask_name": "A", "x": 0.0, "z": 0.0, "zone_id": 0},
                    {"subtask_id": 2, "subtask_name": "B", "x": 0.2, "z": 0.2, "zone_id": 0},
                    {"subtask_id": 3, "subtask_name": "C", "x": 2.0, "z": 0.0, "zone_id": 1},
                    {"subtask_id": 4, "subtask_name": "D", "x": 2.2, "z": 0.2, "zone_id": 1},
                    {"subtask_id": 5, "subtask_name": "E", "x": 0.0, "z": 2.0, "zone_id": 2},
                ]
            ).to_csv(artifact_dir / "firestation_subtask_zone_map.csv", index=False)

            player_records = {
                "planner_a": {
                    "subtask_sequence": [
                        ("2023-01-01 00:00:00", "A", 0.0, 0.0),
                        ("2023-01-01 00:00:01", "B", 0.2, 0.2),
                        ("2023-01-01 00:00:02", "C", 2.0, 0.0),
                        ("2023-01-01 00:00:03", "D", 2.2, 0.2),
                        ("2023-01-01 00:00:04", "E", 0.0, 2.0),
                    ]
                },
                "hopper_a": {
                    "subtask_sequence": [
                        ("2023-01-01 00:00:00", "A", 0.0, 0.0),
                        ("2023-01-01 00:00:01", "C", 2.0, 0.0),
                        ("2023-01-01 00:00:02", "E", 0.0, 2.0),
                        ("2023-01-01 00:00:03", "C", 2.0, 0.0),
                    ]
                },
            }
            with cache_path.open("wb") as handle:
                pickle.dump(player_records, handle)

            exit_code = script.main(
                [
                    "--artifact_dir",
                    str(artifact_dir),
                    "--player_records_cache",
                    str(cache_path),
                    "--output_dir",
                    str(output_dir),
                    "--max_annotations",
                    "4",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "Figure_3b.pdf").exists())


class TestSelectExamplePlayers(unittest.TestCase):
    """Choice of the planner and hopper drawn in Figure 3b."""

    def setUp(self):
        self.labels = pd.DataFrame(
            [
                {"player_id": "planner_a", "label": "Zone Planner", "zone_hopping_score": -8.6},
                {"player_id": "planner_b", "label": "Zone Planner", "zone_hopping_score": -5.0},
                {"player_id": "hopper_a", "label": "Zone Hopper", "zone_hopping_score": 14.0},
                {
                    "player_id": script.DEFAULT_HOPPER_ID,
                    "label": "Zone Hopper",
                    "zone_hopping_score": 12.1,
                },
            ]
        )

    def test_default_hopper_used_when_labelled(self):
        self.assertEqual(
            script.select_example_players(self.labels),
            [("Zone Planner", "planner_a"), ("Zone Hopper", script.DEFAULT_HOPPER_ID)],
        )

    def test_falls_back_to_most_extreme_hopper(self):
        labels = self.labels[self.labels.player_id != script.DEFAULT_HOPPER_ID]
        self.assertEqual(script.select_example_players(labels)[1], ("Zone Hopper", "hopper_a"))

    def test_explicit_ids(self):
        self.assertEqual(
            script.select_example_players(self.labels, "planner_b", "hopper_a"),
            [("Zone Planner", "planner_b"), ("Zone Hopper", "hopper_a")],
        )
        with self.assertRaises(ValueError):
            script.select_example_players(self.labels, hopper_id="planner_b")


if __name__ == "__main__":
    unittest.main()
