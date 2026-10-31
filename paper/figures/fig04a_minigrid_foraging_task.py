#!/usr/bin/env python3
"""Generate Figure 4a for the Minigrid foraging task."""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper.minigrid_figure_common import (
    DEFAULT_MINIGRID_N_PLAYERS,
    DEFAULT_MINIGRID_RELATIONSHIP_CONDITION,
    DEFAULT_MINIGRID_SEED,
    SUPPORTED_RELATIONSHIP_CONDITIONS,
    build_minigrid_publication_inputs,
    resolve_output_path,
)
from simulation.visualization import plot_environment


DEFAULT_OUTPUT_FILENAME = "Figure_4a.pdf"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for Figure 4a (Foraging Task)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (default: manuscript_figures/figures).",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Explicit output PDF path (overrides --output_dir).",
    )
    parser.add_argument(
        "--minigrid_seed",
        type=int,
        default=DEFAULT_MINIGRID_SEED,
        help="Simulator seed for the environments and sampled players (default: 123).",
    )
    parser.add_argument(
        "--minigrid_n_players",
        type=int,
        default=DEFAULT_MINIGRID_N_PLAYERS,
        help="Number of sampled players (default: 2000).",
    )
    parser.add_argument(
        "--minigrid_relationship_condition",
        type=str,
        default=DEFAULT_MINIGRID_RELATIONSHIP_CONDITION,
        choices=list(SUPPORTED_RELATIONSHIP_CONDITIONS),
        help="Task-relationship condition preset (default: complementary).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Generate Figure 4a (Foraging Task)."""
    args = parse_args(argv)
    output_path = resolve_output_path(args.output_dir, args.output_path, DEFAULT_OUTPUT_FILENAME)
    foraging_environment, _cleaning_environment, _players_df = build_minigrid_publication_inputs(
        seed=args.minigrid_seed,
        n_players=args.minigrid_n_players,
        relationship_condition=args.minigrid_relationship_condition,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_environment(foraging_environment, output_path)
    print("Wrote {}".format(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
