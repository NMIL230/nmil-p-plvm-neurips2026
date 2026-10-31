"""Global seeding and device selection.

Importing this module seeds Python, NumPy and PyTorch with 42 (the module-level
``set_seed()`` call below), so any randomness drawn at import time is reproducible;
training scripts call ``set_seed(args.seed)`` again after parsing their arguments.

``COMPUTE_DEVICE`` here is a device string; :mod:`plvm.model_training_utils` defines
its own ``COMPUTE_DEVICE`` as a ``torch.device`` for the same device.
"""

import random
import numpy as np
import torch


def set_seed(seed=42):
    """Seed Python, NumPy and PyTorch (CPU and CUDA) and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Seed at import time; see the module docstring.
set_seed()

COMPUTE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
