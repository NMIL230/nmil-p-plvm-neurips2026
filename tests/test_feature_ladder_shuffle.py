"""Unit tests for the ordering ablation of the Table 14 feature ladder."""

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from paper.navigation_style_observed_sequences import ObservedSequence  # noqa: E402
from paper.tables.table14_feature_ladder import build_player_features  # noqa: E402

N_ZONES = 3
CENTROIDS = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
ZONE_OF_SUBTASK = {"s{}".format(i): i % N_ZONES for i in range(30)}


def _sequences(player_ids):
    """Synthetic players: 30 subtasks one minute apart, visited zone by zone."""
    order = sorted(ZONE_OF_SUBTASK, key=lambda name: ZONE_OF_SUBTASK[name])
    return {
        pid: ObservedSequence(
            player_id=pid,
            task="bungalow",
            elapsed=np.arange(1, 31, dtype=float) * 60.0,
            subtask=np.array(order),
            completion=np.linspace(0.03, 1.0, 30),
        )
        for pid in player_ids
    }


def _features(player_ids, cutoffs, seed):
    return build_player_features(
        _sequences(player_ids),
        ZONE_OF_SUBTASK,
        CENTROIDS,
        N_ZONES,
        cutoffs,
        shuffle_seed=seed,
        task_name="bungalow",
    )


class TestFeatureLadderShuffle(unittest.TestCase):
    """The shuffled features depend only on (seed, task, player, cutoff)."""

    def test_permutation_does_not_depend_on_cutoffs_or_players(self):
        a = _features(["p1", "p2", "p3"], (600, 1200, 1800), seed=0)
        b = _features(["p3", "p1"], (1800,), seed=0)
        for pid in ("p1", "p3"):
            np.testing.assert_array_equal(a[pid][1800]["order"], b[pid][1800]["order"])

    def test_seeds_give_different_permutations(self):
        a = _features(["p1"], (1800,), seed=0)["p1"][1800]["order"]
        b = _features(["p1"], (1800,), seed=1)["p1"][1800]["order"]
        self.assertFalse(np.array_equal(a, b))

    def test_shuffle_leaves_order_free_blocks_unchanged(self):
        ordered = _features(["p1"], (1800,), seed=None)["p1"][1800]
        shuffled = _features(["p1"], (1800,), seed=0)["p1"][1800]
        np.testing.assert_array_equal(ordered["outcome"], shuffled["outcome"])
        np.testing.assert_allclose(ordered["static"], shuffled["static"])
        # zone by zone is the least hopping order, so shuffling raises the switch rate
        self.assertGreater(shuffled["order"][0], ordered["order"][0])


if __name__ == "__main__":
    unittest.main()
