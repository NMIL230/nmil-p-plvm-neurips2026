"""Shared helpers for the Minigrid figure scripts (Figures 4a-4c)."""

import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper.output_paths import resolve_figures_dir
from simulation.agents import sample_players
from simulation.config import (
    RELATIONSHIP_CONDITION_COMPLEMENTARY,
    SUPPORTED_RELATIONSHIP_CONDITIONS,  # noqa: F401  (re-exported for the Figure 4 scripts)
    SimulationConfig,
)
from simulation.environments import (
    TaskEnvironment,
    build_cleaning_environment,
    build_foraging_environment,
)
from simulation.logging_utils import players_to_frame


DEFAULT_MINIGRID_SEED = 123
DEFAULT_MINIGRID_N_PLAYERS = 2000
DEFAULT_MINIGRID_RELATIONSHIP_CONDITION = RELATIONSHIP_CONDITION_COMPLEMENTARY


def resolve_output_path(
    output_dir: Optional[Path],
    output_path: Optional[Path],
    default_filename: str,
) -> Path:
    """Resolve one figure output path, honoring an explicit override."""
    if output_path is not None:
        return Path(output_path)
    return resolve_figures_dir(output_dir) / str(default_filename)


def build_minigrid_publication_inputs(
    *,
    seed: int,
    n_players: int,
    relationship_condition: str,
) -> Tuple[TaskEnvironment, TaskEnvironment, pd.DataFrame]:
    """Build deterministic Minigrid environments plus a sampled player table."""
    config = SimulationConfig(
        n_players=int(n_players),
        seed=int(seed),
        # Placeholder: this builder only samples environments and players and writes nothing.
        out_dir="unused/paper",
        relationship_condition=str(relationship_condition),
        n_example_plots=0,
    ).resolve_condition_preset()
    config.validate()

    root_rng = np.random.default_rng(config.seed)
    players = sample_players(
        config,
        rng=np.random.default_rng(int(root_rng.integers(0, 2**32 - 1))),
    )
    foraging_environment = build_foraging_environment(
        config,
        np.random.default_rng(int(root_rng.integers(0, 2**32 - 1))),
    )
    cleaning_environment = build_cleaning_environment(
        config,
        np.random.default_rng(int(root_rng.integers(0, 2**32 - 1))),
    )
    players_df = players_to_frame(players)
    return foraging_environment, cleaning_environment, players_df
