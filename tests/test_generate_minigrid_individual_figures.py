"""Smoke tests for the Figure 4a-4c scripts (MiniGrid tasks and simulated agents)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from paper.figures import fig04a_minigrid_foraging_task as foraging_script  # noqa: E402
from paper.figures import fig04b_minigrid_cleaning_task as cleaning_script  # noqa: E402
from paper.figures import fig04c_minigrid_simulated_agents as agents_script  # noqa: E402


class TestGenerateMinigridIndividualFigures(unittest.TestCase):
    """Exercise the Figure 4a-4c entrypoints."""

    def test_resolve_output_names_match_expected_files(self):
        base_dir = Path("/tmp/manuscript-figures")
        self.assertEqual(
            foraging_script.resolve_output_path(
                base_dir, None, foraging_script.DEFAULT_OUTPUT_FILENAME
            ).name,
            "Figure_4a.pdf",
        )
        self.assertEqual(
            cleaning_script.resolve_output_path(
                base_dir, None, cleaning_script.DEFAULT_OUTPUT_FILENAME
            ).name,
            "Figure_4b.pdf",
        )
        self.assertEqual(
            agents_script.resolve_output_path(
                base_dir, None, agents_script.DEFAULT_OUTPUT_FILENAME
            ).name,
            "Figure_4c.pdf",
        )

    def test_scripts_write_requested_figure_pdfs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "paper" / "figures"
            common_args = [
                "--output_dir",
                str(output_dir),
                "--minigrid_n_players",
                "8",
                "--minigrid_seed",
                "7",
            ]

            self.assertEqual(foraging_script.main(common_args), 0)
            self.assertEqual(cleaning_script.main(common_args), 0)
            self.assertEqual(agents_script.main(common_args), 0)

            self.assertTrue((output_dir / "Figure_4a.pdf").exists())
            self.assertTrue((output_dir / "Figure_4b.pdf").exists())
            self.assertTrue((output_dir / "Figure_4c.pdf").exists())


if __name__ == "__main__":
    unittest.main()
