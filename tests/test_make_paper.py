#!/usr/bin/env python3
"""Tests for the paper orchestrator (paper/make_paper.py) and its manifest."""

import os
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from paper.make_paper import (  # noqa: E402
    SCRIPT_DIR,
    PaperJob,
    format_job_command,
    load_manifest,
    run_jobs,
    select_jobs,
)


class TestMakePaper(unittest.TestCase):
    """Exercise the manifest and the orchestration helpers without running generators."""

    def setUp(self):
        self.jobs = load_manifest()

    def test_manifest_scripts_exist(self):
        for job in self.jobs:
            if job.is_static:
                self.assertEqual(job.command, [])
                continue
            self.assertTrue((SCRIPT_DIR / job.command[0]).is_file(), job.command[0])

    def test_manifest_covers_every_paper_item_in_paper_order(self):
        items = [item for job in self.jobs for item in job.paper_items]
        expected = [
            "Fig {}".format(n)
            for n in (
                "1",
                "2",
                "3a",
                "3b",
                "3c",
                "4a",
                "4b",
                "4c",
                "5a",
                "5b",
                "5c",
                "6",
                "7",
                "8",
                "9",
                "10",
            )
        ]
        expected += ["Table {}".format(n) for n in range(1, 16)] + ["App D.6", "In-text numbers"]
        for item in expected:
            self.assertIn(item, items)
        self.assertEqual(self.jobs[0].job_id, "fig01")
        self.assertEqual(self.jobs[-1].job_id, "in_text")
        first_table = next(i for i, job in enumerate(self.jobs) if job.kind == "table")
        self.assertTrue(all(job.kind in ("figure", "static") for job in self.jobs[:first_table]))

    def test_select_jobs_keeps_manifest_order_and_rejects_unknown_ids(self):
        selected = select_jobs(self.jobs, ["tab09", "tab02"])
        self.assertEqual([job.job_id for job in selected], ["tab02", "tab09"])
        self.assertEqual(len(select_jobs(self.jobs, None)), len(self.jobs))
        with self.assertRaises(ValueError):
            select_jobs(self.jobs, ["no_such_job"])

    def test_format_job_command_includes_interpreter_script_and_args(self):
        job = next(job for job in self.jobs if job.job_id == "fig07")
        rendered = format_job_command(Path("/usr/bin/python3.11"), job)
        self.assertIn("/usr/bin/python3.11", rendered)
        self.assertIn("fig07_simulation_accuracy_curves.py", rendered)
        self.assertIn("--manuscript_figure", rendered)

    def test_run_jobs_dry_run_does_not_execute_subprocesses(self):
        jobs = [
            PaperJob(
                job_id="fake",
                paper_items=["Fig 0"],
                kind="figure",
                command=["does_not_exist.py"],
                note="",
            ),
            PaperJob(
                job_id="static",
                paper_items=["Fig 1"],
                kind="static",
                command=[],
                note="static asset",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            exit_code = run_jobs(
                jobs, python_executable=Path(tmp_dir) / "missing_python", dry_run=True
            )
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
