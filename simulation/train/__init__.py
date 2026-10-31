"""Model-training utilities for the minigrid foraging/cleaning simulator."""

import os

# Default SciPy / OpenBLAS to a single thread unless the caller has already
# set an explicit threading policy.
for _env_var in [
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(_env_var, "1")

from simulation.train.data_utils import (
    DEFAULT_SIMULATION_DATA_DIR,
    NEXT_ACTION_IGNORE_INDEX,
)
from simulation.train.minigrid_plvm_data_utils import (
    DEFAULT_TASK_MODEL_ROOT,
    PLVM_TASK_NAMES,
)

__all__ = [
    "DEFAULT_SIMULATION_DATA_DIR",
    "DEFAULT_TASK_MODEL_ROOT",
    "NEXT_ACTION_IGNORE_INDEX",
    "PLVM_TASK_NAMES",
]
