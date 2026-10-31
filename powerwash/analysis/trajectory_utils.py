"""Shared trajectory feature helpers for PowerWash analysis scripts."""

import numpy as np


def compute_zigzag_score(positions):
    """Count heading changes > 90° between consecutive steps of an (x, z) trajectory.

    Step pairs involving a zero-length step are not counted.
    """
    if len(positions) < 3:
        return 0
    pts = np.asarray(positions, dtype="float64")
    diffs = np.diff(pts, axis=0)
    norms = np.linalg.norm(diffs, axis=1)
    nonzero = norms > 0
    dirs = np.where(norms[:, None] > 0, diffs / np.where(norms[:, None] > 0, norms[:, None], 1), 0)
    dots = (dirs[:-1] * dirs[1:]).sum(axis=1)
    valid = nonzero[:-1] & nonzero[1:]
    return int((dots[valid] < 0).sum())


def compute_path_length(positions):
    """Total Euclidean distance traveled along a (x, z) position list."""
    if len(positions) < 2:
        return 0.0
    pts = np.asarray(positions, dtype="float64")
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
