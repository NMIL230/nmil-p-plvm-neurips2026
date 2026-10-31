#!/usr/bin/env python3
"""
Generate the simplified_data/ folder from the raw PowerWash Simulator CSV exports.

Filters the raw exports (downloaded and unzipped from https://osf.io/wpeh6/overview)
to the four study jobs and writes ``raw_*.csv`` files, each with a ``task`` label
and a wall-clock ``time_elapsed_since_start`` (pauses included), to
``powerwash/data/simplified_data/``.

Usage:
    python powerwash/data/generate_simplified_data.py /path/to/unzipped/data

Caveats:
    - ``update_current_state`` has no task column, so it is kept by time window
      (job_started/job_resumed to the next job_completed/job_exited); events from
      other game modes started mid-job can slip in.
    - ``subtask_completed`` is matched by its qualified ``WASH_<Task>`` prefix.
      From 2023-01-31 the game logs bare labels, so later rows are dropped and the
      output ends on 2023-02-02.
"""

import argparse
import csv
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from plvm import artifact_paths  # noqa: E402

# ─── Configuration ───────────────────────────────────────────────────────────

# Task definitions: task_label -> job_name
TASKS = {
    "back_garden": "RESIDENTIALSMALL_BACKYARD",
    "bungalow": "RESIDENTIALSMALL_BUNGALOW",
    "fire_station": "FIRESTATION_FIRESTATION",
    "storybook_house": "NATIONALPARK_STORYBOOKHOUSE",
}

# Reverse lookup: job_name -> task_label
JOB_TO_TASK = {v: k for k, v in TASKS.items()}

# task_completed records the finished task under its own identifier.
LAST_TASK_TO_TASK = {
    "WASH_Backyard": "back_garden",
    "WASH_Bungalow": "bungalow",
    "WASH_FireStation": "fire_station",
    "WASH_StoryBookCottage": "storybook_house",
}

# subtask_completed records "<task_id>.<subtask>", so the prefix identifies the task.
SUBTASK_PREFIX_TO_TASK = dict(LAST_TASK_TO_TASK)

# Default output folder (the simplified_data/ folder in the artifact tree).
DEFAULT_OUTPUT_DIR = str(artifact_paths.SIMPLIFIED_DATA_DIR)

# ─── Helpers ─────────────────────────────────────────────────────────────────


def parse_time(s):
    """Parse a timestamp string to datetime."""
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None


def calc_elapsed(first_start_dt, time_str):
    """Compute seconds elapsed since first job start."""
    t = parse_time(time_str)
    if t and first_start_dt:
        return (t - first_start_dt).total_seconds()
    return 0.0


def build_first_starts(source_dir):
    """
    For each (pid, task), find the earliest job_started time.
    Returns: dict of (pid, task_label) -> datetime
    """
    first_start = {}
    with open(os.path.join(source_dir, "job_started.csv")) as f:
        for row in csv.DictReader(f):
            job = row["CurrentJobName"]
            if job not in JOB_TO_TASK:
                continue
            task = JOB_TO_TASK[job]
            pid = row["pid"]
            key = (pid, task)
            t = parse_time(row["Time"])
            if t and (key not in first_start or t < first_start[key]):
                first_start[key] = t
    return first_start


def build_time_windows(source_dir):
    """
    Build per-player time windows for each task.

    A window opens at each job_started or job_resumed event and closes at the next
    job_completed or job_exited for that task.

    Returns: dict of (pid, task_label) -> list of (start_str, end_str)
    """
    # Collect window starts from both job_started and job_resumed
    starts = defaultdict(list)  # (pid, task) -> [time_str, ...]
    for source_file in ("job_started.csv", "job_resumed.csv"):
        with open(os.path.join(source_dir, source_file)) as f:
            for row in csv.DictReader(f):
                job = row.get("CurrentJobName", "")
                if job in JOB_TO_TASK:
                    key = (row["pid"], JOB_TO_TASK[job])
                    starts[key].append(row["Time"])

    # Collect ends from both job_completed and job_exited
    ends = defaultdict(list)  # (pid, task) -> [time_str, ...]

    with open(os.path.join(source_dir, "job_completed.csv")) as f:
        for row in csv.DictReader(f):
            job = row.get("LastJobCompleted", "")
            if job in JOB_TO_TASK:
                key = (row["pid"], JOB_TO_TASK[job])
                ends[key].append(row["Time"])

    with open(os.path.join(source_dir, "job_exited.csv")) as f:
        for row in csv.DictReader(f):
            job = row.get("CurrentJobName", "")
            if job in JOB_TO_TASK:
                key = (row["pid"], JOB_TO_TASK[job])
                ends[key].append(row["Time"])

    # Pair each start/resume with the next end at or after it
    windows = defaultdict(list)  # (pid, task) -> [(start, end), ...]
    for key in starts:
        s_list = sorted(starts[key])
        e_list = sorted(ends.get(key, []))
        ei = 0
        for s in s_list:
            while ei < len(e_list) and e_list[ei] < s:
                ei += 1
            e = e_list[ei] if ei < len(e_list) else "9999-12-31 23:59:59"
            windows[key].append((s, e))

    return windows


def find_task_by_window(pid, time_str, windows_by_pid):
    """
    Check if a (pid, time) falls within any task's time window.
    Returns task_label or None.
    """
    if pid not in windows_by_pid:
        return None
    for task, window_list in windows_by_pid[pid].items():
        for start, end in window_list:
            if start <= time_str <= end:
                return task
    return None


# ─── Processing functions ────────────────────────────────────────────────────


def _write_filtered(source_path, output_path, first_start, classify):
    """
    Stream ``source_path`` to ``output_path``, keeping rows that ``classify``
    assigns a task to and appending the 'task' / 'time_elapsed_since_start' columns.

    ``classify`` takes a row dict and returns a task label, or None to drop the row.
    """
    with open(source_path) as f:
        source_cols = csv.DictReader(f).fieldnames

    out_cols = source_cols + ["task", "time_elapsed_since_start"]
    count = 0

    with open(output_path, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=out_cols)
        writer.writeheader()
        with open(source_path) as f:
            for row in csv.DictReader(f):
                task = classify(row)
                if task is None:
                    continue
                row["task"] = task
                fs = first_start.get((row["pid"], task))
                row["time_elapsed_since_start"] = calc_elapsed(fs, row["Time"])
                writer.writerow(row)
                count += 1

    return count


def process_job_name_file(source_path, output_path, job_col, first_start):
    """Filter a file that carries a job name column (CurrentJobName/LastJobCompleted)."""
    return _write_filtered(
        source_path,
        output_path,
        first_start,
        lambda row: JOB_TO_TASK.get(row.get(job_col, "")),
    )


def process_last_task_file(source_path, output_path, task_col, first_start):
    """Filter a file that records the completed task under its own identifier."""
    return _write_filtered(
        source_path,
        output_path,
        first_start,
        lambda row: LAST_TASK_TO_TASK.get(row.get(task_col, "")),
    )


def process_subtask_file(source_path, output_path, subtask_col, first_start):
    """Filter a file whose subtask identifier is prefixed with the task id."""

    def classify(row):
        subtask_id = row.get(subtask_col, "") or ""
        for prefix, task_label in SUBTASK_PREFIX_TO_TASK.items():
            if subtask_id.startswith(prefix):
                return task_label
        return None

    return _write_filtered(source_path, output_path, first_start, classify)


def process_time_window_file(source_path, output_path, first_start, windows_by_pid):
    """Filter a file with no task identifier, by per-player job time windows."""
    return _write_filtered(
        source_path,
        output_path,
        first_start,
        lambda row: find_task_by_window(row["pid"], row["Time"], windows_by_pid),
    )


def copy_file(source_path, output_path):
    """Copy a file as-is (no task filtering)."""
    shutil.copy2(source_path, output_path)
    with open(source_path) as f:
        return sum(1 for _ in f) - 1  # row count excluding header


# ─── Main ────────────────────────────────────────────────────────────────────


def parse_args(argv=None):
    """Parse the raw-data folder and the output folder."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "source_dir",
        nargs="?",
        default=None,
        help="Unzipped OSF data folder (contains job_started.csv, ...).",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Folder for the raw_*.csv outputs (default: the artifact simplified_data/ folder).",
    )
    return parser.parse_args(argv)


def main():
    """Filter every raw CSV in ``source_dir`` into ``--output_dir``."""
    args = parse_args()
    output_dir = args.output_dir
    if args.source_dir is None:
        print(
            "ERROR: No data path provided.\n"
            "\n"
            "Usage:\n"
            "    python powerwash/data/generate_simplified_data.py /path/to/unzipped/data\n"
            "\n"
            "Download the PowerWash Simulator dataset from:\n"
            "    https://osf.io/wpeh6/overview\n"
            "Then unzip it and pass the resulting folder as the first argument.\n"
            "That folder should contain the raw CSV files (job_started.csv, etc.)."
        )
        sys.exit(1)

    source_dir = os.path.abspath(args.source_dir)

    if not os.path.isdir(source_dir):
        print("ERROR: '{}' is not a valid directory.".format(source_dir))
        sys.exit(1)

    # Verify the expected CSV files are present
    required = ["job_started.csv", "job_completed.csv", "subtask_completed.csv"]
    for fn in required:
        if not os.path.exists(os.path.join(source_dir, fn)):
            print(
                "ERROR: {} not found in '{}'.\n"
                "Make sure you are pointing to the unzipped folder downloaded from:\n"
                "    https://osf.io/wpeh6/overview".format(fn, source_dir)
            )
            sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Build lookup structures
    print("Building first-start times...")
    first_start = build_first_starts(source_dir)
    print("  {} (player, task) pairs".format(len(first_start)))

    print("Building time windows...")
    windows = build_time_windows(source_dir)

    # Reorganise windows for fast lookup: pid -> {task -> [(start, end), ...]}
    windows_by_pid = defaultdict(dict)
    for (pid, task), wlist in windows.items():
        windows_by_pid[pid][task] = wlist
    print("  {} players with windows".format(len(windows_by_pid)))

    # Step 2: Process each source file
    print("\nProcessing files...\n")

    # Files filtered by job name column
    job_name_files = [
        ("job_started.csv", "CurrentJobName"),
        ("job_completed.csv", "LastJobCompleted"),
        ("job_exited.csv", "CurrentJobName"),
        ("job_resumed.csv", "CurrentJobName"),
        ("exited_game.csv", "CurrentJobName"),
        ("game_saved.csv", "CurrentJobName"),
        ("item_purchased.csv", "CurrentJobName"),
        ("study_prompt_answered.csv", "CurrentJobName"),
        ("study_reward_claimed.csv", "CurrentJobName"),
        ("study_reward_unlocked.csv", "CurrentJobName"),
    ]

    for source_file, job_col in job_name_files:
        source_path = os.path.join(source_dir, source_file)
        output_path = os.path.join(output_dir, "raw_" + source_file)
        count = process_job_name_file(source_path, output_path, job_col, first_start)
        print("  {:<30} -> {:>10,} rows".format(source_file, count))

    # Files filtered directly by the completed task identifier
    for source_file, task_col in [("task_completed.csv", "LastTaskCompleted")]:
        source_path = os.path.join(source_dir, source_file)
        output_path = os.path.join(output_dir, "raw_" + source_file)
        count = process_last_task_file(source_path, output_path, task_col, first_start)
        print("  {:<30} -> {:>10,} rows".format(source_file, count))

    # Files filtered by the subtask identifier's task prefix
    for source_file, subtask_col in [("subtask_completed.csv", "LastSubtaskCompleted")]:
        source_path = os.path.join(source_dir, source_file)
        output_path = os.path.join(output_dir, "raw_" + source_file)
        print("  {:<30} -> processing (large file)...".format(source_file), end="", flush=True)
        count = process_subtask_file(source_path, output_path, subtask_col, first_start)
        print("\r  {:<30} -> {:>10,} rows".format(source_file, count))

    # Files filtered by time window (no task identifier at all)
    for source_file in ["update_current_state.csv"]:
        source_path = os.path.join(source_dir, source_file)
        output_path = os.path.join(output_dir, "raw_" + source_file)
        print("  {:<30} -> processing (large file)...".format(source_file), end="", flush=True)
        count = process_time_window_file(source_path, output_path, first_start, windows_by_pid)
        print("\r  {:<30} -> {:>10,} rows".format(source_file, count))

    # Files copied as-is (no task filtering)
    copy_files = ["demographics.csv", "player_logged_in.csv"]
    for source_file in copy_files:
        source_path = os.path.join(source_dir, source_file)
        output_path = os.path.join(output_dir, "raw_" + source_file)
        count = copy_file(source_path, output_path)
        print("  {:<30} -> {:>10,} rows (copied as-is)".format(source_file, count))

    print("\nDone! Output written to {}/".format(output_dir))

    # Step 3: Summary
    print("\n=== Summary ===")
    for task_label in TASKS:
        players = sum(1 for k in first_start if k[1] == task_label)
        print("  {}: {} players".format(task_label, players))


if __name__ == "__main__":
    main()
