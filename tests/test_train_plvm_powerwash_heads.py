#!/usr/bin/env python3
"""Prediction-head resolution and log-family routing of the PowerWash PLVM trainer.

The single-task linear probe (``--model_type single_task --prediction_head_type logistic``)
keeps its logistic head and is written to the ``single_task_logistic_probe`` family; the
other combinations go to their own families.
"""

import argparse
import os
import sys
import unittest
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from powerwash.train import train_plvm_powerwash as trainer  # noqa: E402
from powerwash.train.powerwash_hierarchical_log_utils import (  # noqa: E402
    resolve_powerwash_hierarchical_log_base,
)


def _head(model_type, requested):
    return trainer._resolve_prediction_head_type(
        argparse.Namespace(model_type=model_type, prediction_head_type=requested)
    )


def _family_dir(model_type, requested, label_csv=""):
    """Run the same routing as ``main``: head -> family -> log base -> Storybook suffix."""
    head = _head(model_type, requested)
    base = resolve_powerwash_hierarchical_log_base(
        Path("/logs/training_logs"),
        trainer._resolve_log_family(model_type, head),
        project_root=Path("/repo"),
    )
    base = trainer._apply_storybook_log_suffix(
        base, trainer._should_use_storybook_log_suffix(label_csv)
    )
    return base.name


class TestPredictionHeadResolution(unittest.TestCase):
    """``_resolve_prediction_head_type`` policy."""

    def test_single_task_keeps_logistic(self):
        self.assertEqual(_head("single_task", "logistic"), "logistic")

    def test_single_task_auto_and_hierarchical_fall_back_to_mlp(self):
        self.assertEqual(_head("single_task", "auto"), "mlp")
        self.assertEqual(_head("single_task", "hierarchical"), "mlp")
        self.assertEqual(_head("single_task", "mlp"), "mlp")

    def test_two_task_heads(self):
        self.assertEqual(_head("plvm_both_partial", "auto"), "hierarchical")
        self.assertEqual(_head("plvm_both_partial", "logistic"), "logistic")
        self.assertEqual(_head("plvm_both_partial", "mlp"), "mlp")

    def test_unknown_head_raises(self):
        with self.assertRaises(ValueError):
            _head("single_task", "svm")


class TestLogFamilyRouting(unittest.TestCase):
    """Each (model type, head) pair is written to its model-family folder."""

    def test_families(self):
        cases = {
            ("single_task", "logistic"): "single_task_logistic_probe",
            ("single_task", "mlp"): "single_task",
            ("single_task", "auto"): "single_task",
            ("plvm_both_partial", "logistic"): "concat_logistic_probe",
            ("plvm_both_partial", "mlp"): "concat_mlp_probe",
            ("plvm_both_partial", "hierarchical"): "plvm_models",
            ("plvm_full_partial", "hierarchical"): "plvm_models",
        }
        for (model_type, requested), family in cases.items():
            with self.subTest(model_type=model_type, head=requested):
                self.assertEqual(_family_dir(model_type, requested), family)

    def test_storybook_single_task_logistic_gets_suffix(self):
        label_csv = "powerwash/outputs/storybook_house_zone_labels/fold_00/x.csv"
        self.assertEqual(
            _family_dir("single_task", "logistic", label_csv),
            "single_task_logistic_probe_storybook",
        )


if __name__ == "__main__":
    unittest.main()
