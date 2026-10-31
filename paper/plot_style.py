#!/usr/bin/env python3
"""
plot_style.py
=============
Shared matplotlib style for the PowerWash and simulation figures.

Importing this module also selects the ``Agg`` backend and puts the repository
root and ``paper/`` at the front of ``sys.path``, so ``paper.*``, ``plvm.*``,
``powerwash.*`` and ``simulation.*`` resolve when a generator runs as a script.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))


def maybe_set_plot_style():
    """Apply a plotting style consistent with the other PowerWash figures."""
    for candidate in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        if candidate in plt.style.available:
            plt.style.use(candidate)
            break
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
        }
    )
