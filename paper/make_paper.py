#!/usr/bin/env python3
"""Generate every figure, table and in-text number from the saved artifacts.

Jobs, their arguments and the items they produce are listed in
``paper/paper_manifest.tsv``, in the order the items appear in the paper; each runs
one generator under ``paper/`` from the repository root. Static items (Figure 1;
Tables 3-7 and 15) are listed but not run.

Example (a subset by job id)::

    python paper/make_paper.py --only tab02,tab09
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MANIFEST = SCRIPT_DIR / "paper_manifest.tsv"
STATIC_KIND = "static"


@dataclass(frozen=True)
class PaperJob:
    """One manifest row: a generator invocation (or a static paper item)."""

    job_id: str
    paper_items: List[str]
    kind: str
    command: List[str]
    note: str

    @property
    def is_static(self) -> bool:
        """True for manifest rows that are listed but have no generator to run."""
        return self.kind == STATIC_KIND


def load_manifest(path: Path = DEFAULT_MANIFEST) -> List[PaperJob]:
    """Read the manifest (comment lines start with '#') and validate it."""
    lines = [
        line
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    jobs = []
    for row in csv.DictReader(lines, delimiter="\t"):
        command = [] if row["command"].strip() == "-" else shlex.split(row["command"])
        jobs.append(
            PaperJob(
                job_id=row["job_id"].strip(),
                paper_items=[
                    item.strip() for item in row["paper_items"].split(";") if item.strip()
                ],
                kind=row["kind"].strip(),
                command=command,
                note=row.get("note", "").strip(),
            )
        )
    ids = [job.job_id for job in jobs]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError("Duplicate job ids in {}: {}".format(path, duplicates))
    for job in jobs:
        if not job.is_static and not job.command:
            raise ValueError("Job '{}' has no command.".format(job.job_id))
    return jobs


def select_jobs(jobs: Sequence[PaperJob], only: Optional[Sequence[str]] = None) -> List[PaperJob]:
    """Return the jobs to run, in manifest order; ``only`` restricts them to the given job ids."""
    if not only:
        return list(jobs)
    wanted = [token.strip() for token in only if token.strip()]
    unknown = sorted(set(wanted) - {job.job_id for job in jobs})
    if unknown:
        raise ValueError(
            "Unknown job id(s): {}. Use --list to see the manifest.".format(", ".join(unknown))
        )
    return [job for job in jobs if job.job_id in wanted]


def job_argv(python_executable: Path, job: PaperJob) -> List[str]:
    """The child process argv for one job."""
    return [str(python_executable), str(SCRIPT_DIR / job.command[0]), *job.command[1:]]


def format_job_command(python_executable: Path, job: PaperJob) -> str:
    """Shell-quoted preview of one job's command."""
    return shlex.join(job_argv(python_executable, job))


def run_jobs(
    jobs: Sequence[PaperJob],
    *,
    python_executable: Path,
    dry_run: bool = False,
    keep_going: bool = False,
) -> int:
    """Run the jobs in order and return a process-style exit code (static jobs are only reported)."""
    python_executable = Path(python_executable)
    if not dry_run and not python_executable.exists():
        raise FileNotFoundError("Python interpreter not found: {}".format(python_executable))
    failures = []
    for index, job in enumerate(jobs, start=1):
        header = "[{}/{}] {} ({}: {})".format(
            index, len(jobs), job.job_id, job.kind, ", ".join(job.paper_items)
        )
        print(header, flush=True)
        if job.is_static:
            print("  skipped: {}".format(job.note), flush=True)
            continue
        print("  {}".format(format_job_command(python_executable, job)), flush=True)
        if dry_run:
            continue
        completed = subprocess.run(
            job_argv(python_executable, job), cwd=str(PROJECT_ROOT), check=False
        )
        if completed.returncode == 0:
            continue
        failures.append((job.job_id, completed.returncode))
        print(
            "Step failed: {} exited with code {}.".format(job.job_id, completed.returncode),
            file=sys.stderr,
            flush=True,
        )
        if not keep_going:
            return int(completed.returncode) or 1
    for job_id, returncode in failures:
        print("Failed: {} (exit code {})".format(job_id, returncode), file=sys.stderr, flush=True)
    if failures:
        return 1
    print(
        "Dry run only: no generators were executed."
        if dry_run
        else "Completed {} paper job(s).".format(sum(not job.is_static for job in jobs)),
        flush=True,
    )
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST, help="Paper manifest (TSV)."
    )
    parser.add_argument(
        "--python_executable",
        type=Path,
        default=Path(sys.executable),
        help="Interpreter used for every generator.",
    )
    parser.add_argument(
        "--only",
        type=lambda s: s.split(","),
        default=None,
        help="Comma-separated job ids to run (default: all, in manifest order).",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Print the commands without running them."
    )
    parser.add_argument("--keep_going", action="store_true", help="Continue after a failing job.")
    parser.add_argument("--list", action="store_true", help="Print the manifest and exit.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the requested paper jobs."""
    args = parse_args(argv)
    jobs = load_manifest(args.manifest)
    if args.list:
        for job in jobs:
            print(
                "{:<26s} {:<8s} {:<40s} {}".format(
                    job.job_id, job.kind, "; ".join(job.paper_items), job.note
                )
            )
        return 0
    try:
        selected = select_jobs(jobs, args.only)
    except ValueError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    return run_jobs(
        selected,
        python_executable=args.python_executable,
        dry_run=bool(args.dry_run),
        keep_going=bool(args.keep_going),
    )


if __name__ == "__main__":
    raise SystemExit(main())
