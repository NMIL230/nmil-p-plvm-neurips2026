#!/usr/bin/env python3
"""Tests for the data-pipeline flags that control how the sequence and feature sets are built.

- ``extract_player_sequences.py --implicit_gap_threshold_seconds`` (300 s default; 0/inf off),
  including the double subtraction of explicit pauses;
- ``derive_features_for_transformer.py --fallback_input_folder`` / ``--clean_output``;
- ``generate_simplified_data.py --output_dir`` (default: the folder extraction reads).

All inputs are small synthetic files, so the tests do not need the data artifacts.
"""

import contextlib
import csv
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from plvm import artifact_paths  # noqa: E402
from powerwash.data import derive_features_for_transformer as derive  # noqa: E402
from powerwash.data import extract_player_sequences as extract  # noqa: E402
from powerwash.data import generate_simplified_data as simplify  # noqa: E402
from powerwash.data.powerwash_subtask_cleaning import (  # noqa: E402
    EXPECTED_NORMALIZED_SUBTASKS_BY_TASK,
)


def _row(seconds, elapsed, event_type="subtask_completed", subtask=""):
    """One extraction output row at 2023-01-01 10:00:00 + seconds with the given active elapsed time."""
    row = [None] * len(extract.OUTPUT_COLS)
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    row[extract.TIME_UTC_IDX] = "2023-01-01 {:02d}:{:02d}:{:02d}".format(10 + hours, minutes, secs)
    row[extract.OUTPUT_COLS.index("event_type")] = event_type
    row[extract.ELAPSED_IDX] = extract.format_elapsed_seconds(float(elapsed))
    row[extract.ELAPSED_RAW_IDX] = extract.format_elapsed_seconds(float(seconds))
    row[extract.OUTPUT_COLS.index("LastSubtaskCompleted")] = subtask
    return (extract.parse_time_utc(row[extract.TIME_UTC_IDX]), row)


def _elapsed(rows):
    return [str(row[extract.ELAPSED_IDX]) for _, row in rows]


class TestImplicitGapThreshold(unittest.TestCase):
    """``apply_implicit_gap_correction`` / ``write_attempt_csv`` with the gap threshold."""

    def setUp(self):
        # A 400 s gap between the 2nd and 3rd row, no explicit pause.
        self.rows = [_row(0, 0), _row(10, 10), _row(410, 410), _row(420, 420)]

    def test_default_subtracts_gaps_longer_than_300s(self):
        corrected = extract.apply_implicit_gap_correction(self.rows)
        self.assertEqual(_elapsed(corrected), ["0", "10", "10", "20"])

    def test_zero_and_inf_disable_the_correction(self):
        for threshold in (0, float("inf"), None):
            with self.subTest(threshold=threshold):
                out = extract.apply_implicit_gap_correction(self.rows, threshold)
                self.assertEqual(_elapsed(out), ["0", "10", "410", "420"])

    def test_explicit_pause_is_subtracted_twice(self):
        # build_output_row already removed a 400 s exit -> resume pause from the active clock;
        # the gap test subtracts it again and the clock is clamped at 0.
        rows = [_row(0, 0), _row(10, 10), _row(410, 10), _row(420, 20)]
        corrected = extract.apply_implicit_gap_correction(rows)
        self.assertEqual(_elapsed(corrected), ["0", "10", "0", "0"])

    def test_write_attempt_csv_threshold_argument(self):
        attempt = {"pid": "p1", "task": "back_garden", "attempt_id": "attempt1", "rows": self.rows}
        with tempfile.TemporaryDirectory() as tmp:
            extract.write_attempt_csv(attempt, os.path.join(tmp, "on"))
            extract.write_attempt_csv(attempt, os.path.join(tmp, "off"), 0)
            read = {}
            for key in ("on", "off"):
                path = os.path.join(tmp, key, "back_garden", "p1_back_garden_attempt1.csv")
                with open(path, newline="") as fh:
                    read[key] = list(csv.DictReader(fh))
        self.assertEqual(
            [r["time_elapsed_since_job_start"] for r in read["on"]], ["0", "10", "10", "20"]
        )
        self.assertEqual(
            [r["time_elapsed_since_job_start"] for r in read["off"]], ["0", "10", "410", "420"]
        )

    def test_default_output_follows_the_threshold(self):
        with mock.patch.object(sys, "argv", ["extract"]):
            on = extract.parse_args()
        with mock.patch.object(sys, "argv", ["extract", "--implicit_gap_threshold_seconds", "0"]):
            off = extract.parse_args()
        self.assertEqual(on.implicit_gap_threshold_seconds, 300.0)
        self.assertTrue(on.output.endswith(os.path.join("data", "player_sequences")))
        self.assertTrue(
            off.output.endswith(os.path.join("data", "player_sequences_no_gap_correction"))
        )


def _write_attempt_file(path, elapsed_values, task="bungalow"):
    """A synthetic attempt completing every expected subtask of ``task`` once, 60 s apart."""
    subtasks = sorted(EXPECTED_NORMALIZED_SUBTASKS_BY_TASK[task])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(extract.OUTPUT_COLS)
        for i, (subtask, elapsed) in enumerate(zip(subtasks, elapsed_values)):
            _, row = _row(60 * i, elapsed, subtask=subtask)
            row[extract.OUTPUT_COLS.index("pid")] = "p1"
            row[extract.OUTPUT_COLS.index("attempt_id")] = "attempt1"
            row[extract.OUTPUT_COLS.index("task")] = task
            row[extract.OUTPUT_COLS.index("LevelProgressionAmount")] = "{:.4f}".format(
                i / len(subtasks)
            )
            row[extract.OUTPUT_COLS.index("CurrentPosition")] = "({}.0, 0.0, {}.0)".format(i, i % 5)
            writer.writerow(["" if v is None else v for v in row])


class TestDeriveFallback(unittest.TestCase):
    """``--fallback_input_folder`` and ``--clean_output`` of the feature derivation."""

    NAME = Path("bungalow") / "p1_bungalow_attempt1.csv"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        n = len(EXPECTED_NORMALIZED_SUBTASKS_BY_TASK["bungalow"])
        monotonic = [60.0 * i for i in range(n)]
        broken = list(monotonic)
        broken[n // 2 :] = [0.0] * (n - n // 2)  # clock drops to 0, as after a double subtraction
        _write_attempt_file(self.root / "corrected" / self.NAME, broken)
        _write_attempt_file(self.root / "uncorrected" / self.NAME, monotonic)

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, input_name, output_name, *extra):
        argv = [
            "derive",
            str(self.root / input_name),
            str(self.root / output_name),
            "--tasks",
            "bungalow",
        ]
        stderr = io.StringIO()
        # Canonical subtask coordinates only feed target columns; keep the test independent of the map artifact.
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(sys, "argv", argv + list(extra)))
            stack.enter_context(
                mock.patch.object(derive, "load_canonical_subtask_coordinates", return_value={})
            )
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(stderr))
            rc = derive.main()
        return rc, stderr.getvalue()

    def test_non_monotonic_file_uses_the_fallback(self):
        rc, _ = self._run(
            "corrected", "out", "--fallback_input_folder", str(self.root / "uncorrected")
        )
        self.assertEqual(rc, 0)
        rc, _ = self._run("uncorrected", "direct", "--fallback_input_folder", "none")
        self.assertEqual(rc, 0)
        self.assertEqual(
            (self.root / "out" / self.NAME).read_bytes(),
            (self.root / "direct" / self.NAME).read_bytes(),
        )

    def test_without_fallback_the_file_is_skipped(self):
        rc, _ = self._run("corrected", "out", "--fallback_input_folder", "none")
        self.assertEqual(rc, 0)
        self.assertFalse((self.root / "out" / self.NAME).exists())

    def test_stale_outputs_are_reported_or_cleaned(self):
        stale = self.root / "out" / "bungalow" / "p9_bungalow_attempt1.csv"
        stale.parent.mkdir(parents=True)
        stale.write_text("old\n")
        rc, err = self._run("uncorrected", "out", "--fallback_input_folder", "none")
        self.assertEqual(rc, 0)
        self.assertTrue(stale.exists())
        self.assertIn("stale", err)
        rc, err = self._run(
            "uncorrected", "out", "--fallback_input_folder", "none", "--clean_output"
        )
        self.assertEqual(rc, 0)
        self.assertFalse(stale.exists())
        self.assertTrue((self.root / "out" / self.NAME).exists())
        self.assertNotIn("stale", err)

    def test_fallback_default_and_disable(self):
        preferred = self.root / "player_sequences_no_gap_correction"
        secondary = self.root / "player_sequencesy"
        candidates = (preferred, secondary)
        self.assertIsNone(derive.resolve_fallback_input_folder(None, candidates))
        secondary.mkdir(parents=True)
        self.assertEqual(derive.resolve_fallback_input_folder(None, candidates), secondary)
        preferred.mkdir(parents=True)
        self.assertEqual(derive.resolve_fallback_input_folder(None, candidates), preferred)
        self.assertIsNone(derive.resolve_fallback_input_folder("none", candidates))

    def test_fallback_defaults_come_from_the_artifact_resolver(self):
        self.assertEqual(
            derive.DEFAULT_FALLBACK_INPUT_DIRS,
            artifact_paths.DEFAULT_FALLBACK_SEQUENCE_DIRS,
        )


class TestSimplifiedDataOutputDir(unittest.TestCase):
    """The simplified-data generator writes where extraction reads by default."""

    def test_default_output_dir_is_the_extraction_input(self):
        # Both ends of the hand-off come from the resolver, so the assertion is on the
        # shared artifact location rather than on where either script file happens to live.
        expected = str(artifact_paths.SIMPLIFIED_DATA_DIR)
        args = simplify.parse_args(["/raw"])
        self.assertEqual(os.path.abspath(args.output_dir), expected)
        self.assertEqual(
            os.path.dirname(extract.resolve_default_input("x.csv")),
            expected,
        )

    def test_output_dir_override(self):
        self.assertEqual(
            simplify.parse_args(["/raw", "--output_dir", "/tmp/x"]).output_dir, "/tmp/x"
        )


if __name__ == "__main__":
    unittest.main()
