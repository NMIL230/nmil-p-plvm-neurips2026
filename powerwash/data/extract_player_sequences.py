#!/usr/bin/env python3
"""
Extract one event-sequence CSV per completed task attempt.

An attempt runs from ``job_started`` to the next ``job_completed`` for the same
player and job; ``job_resumed`` never starts a new attempt. Rows from
``raw_subtask_completed.csv`` and ``raw_update_current_state.csv`` in
``powerwash/data/simplified_data/`` are written per task folder.
``time_elapsed_since_job_start`` excludes explicit ``job_exited`` ->
``job_resumed`` pauses and, by default, any gap over 300 s between rows;
``time_elapsed_since_job_start_raw`` is wall-clock.

The default run writes two sets under ``powerwash/data/``: gap-corrected
``player_sequences/`` (used downstream) and uncorrected
``player_sequences_no_gap_correction/``, which
``derive_features_for_transformer.py`` falls back to when a corrected clock is
not monotonic.

Known limitation: an explicit pause longer than the gap threshold is subtracted
twice, so the active clock can drop (clamped at 0). The derived features are
built from this output; ``derive_features_for_transformer.py`` uses the
uncorrected copy of an attempt instead when its corrected clock decreases.
"""

import argparse
import csv
import heapq
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from plvm import artifact_paths  # noqa: E402

from powerwash.data.powerwash_subtask_cleaning import (  # noqa: E402
    EXPECTED_NORMALIZED_SUBTASKS_BY_TASK,
    normalize_subtask_label,
)

# Default locations used when no command-line arguments are given.
DEFAULT_DATA_DIRS = (artifact_paths.SIMPLIFIED_DATA_DIR,)
DEFAULT_OUTPUT_DIR = artifact_paths.PLAYER_SEQUENCES_DIR
DEFAULT_NO_GAP_CORRECTION_OUTPUT_DIR = artifact_paths.PLAYER_SEQUENCES_NO_GAP_CORRECTION_DIR

JOB_CODE = {
    "back_garden": "RESIDENTIALSMALL_BACKYARD",
    "bungalow": "RESIDENTIALSMALL_BUNGALOW",
    "fire_station": "FIRESTATION_FIRESTATION",
    "storybook_house": "NATIONALPARK_STORYBOOKHOUSE",
}
TASK_BY_JOB_CODE = {job_code: task for task, job_code in JOB_CODE.items()}
TARGET_JOB_CODES = set(TASK_BY_JOB_CODE)

LIFECYCLE_EVENT_PRIORITY = {
    "job_exited": 0,
    "job_started": 1,
    "job_resumed": 2,
    "job_completed": 3,
}

# Output columns for every per-attempt CSV.
OUTPUT_COLS = [
    "pid",
    "attempt_id",
    "task",  # back_garden | bungalow | fire_station | storybook_house
    "event_type",  # subtask_completed | update_current_state
    "Time_utc",
    "job_start_utc",
    "job_end_utc",
    "time_elapsed_since_job_start",  # wall-clock elapsed minus paired exit/resume pauses
    "time_elapsed_since_job_start_raw",  # seconds since the initial job_started
    "CurrentGameMode",
    "CurrentJobName",
    "LevelProgressionAmount",
    "CurrentWasher",
    "CurrentNozzle",
    "CurrentExtension",
    "CurrentPosition",
    "CrouchState",
    "CampaignProgressionAmount",
    "CurrentCreditAmount",
    "CurrentSessionLength",
    "CumulativeTimeSpentInLevel",
    "IsIdleInGame",
    "IsInMenu",
    "LastSubtaskCompleted",
]

TIME_UTC_IDX = OUTPUT_COLS.index("Time_utc")
ELAPSED_IDX = OUTPUT_COLS.index("time_elapsed_since_job_start")
ELAPSED_RAW_IDX = OUTPUT_COLS.index("time_elapsed_since_job_start_raw")
EVENT_TYPE_IDX = OUTPUT_COLS.index("event_type")

# Gaps longer than this between consecutive rows are treated as implicit (unlogged) pauses.
# Default of --implicit_gap_threshold_seconds (the player_sequences/ set).
IMPLICIT_GAP_THRESHOLD_SECONDS = 300  # 5 minutes

PID_SORT_CHUNK_ROWS = 25000


def resolve_default_input(filename):
    """Resolve a default input file path under ``DEFAULT_DATA_DIRS``."""
    candidates = [str(data_dir / filename) for data_dir in DEFAULT_DATA_DIRS]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def resolve_optional_input(filename):
    """Resolve an optional input file path or return an empty string."""
    candidates = [str(data_dir / filename) for data_dir in DEFAULT_DATA_DIRS]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return ""


def parse_args():
    """Parse CLI arguments for lifecycle inputs, event inputs, and output."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--started",
        default=resolve_default_input("raw_job_started.csv"),
        help="Path to raw_job_started.csv",
    )
    parser.add_argument(
        "--exited",
        default=resolve_optional_input("raw_job_exited.csv"),
        help="Optional path to raw_job_exited.csv for pause-interval correction",
    )
    parser.add_argument(
        "--resumed",
        default=resolve_default_input("raw_job_resumed.csv"),
        help="Path to raw_job_resumed.csv",
    )
    parser.add_argument(
        "--completed",
        default=resolve_default_input("raw_job_completed.csv"),
        help="Path to raw_job_completed.csv",
    )
    parser.add_argument(
        "--subtask",
        default=resolve_default_input("raw_subtask_completed.csv"),
        help="Path to raw_subtask_completed.csv",
    )
    parser.add_argument(
        "--state",
        default=resolve_default_input("raw_update_current_state.csv"),
        help="Path to raw_update_current_state.csv",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output folder for per-attempt CSVs (default: {} with the "
            "implicit-gap correction, {} without it)".format(
                artifact_paths.display_path(DEFAULT_OUTPUT_DIR),
                artifact_paths.display_path(DEFAULT_NO_GAP_CORRECTION_OUTPUT_DIR),
            )
        ),
    )
    parser.add_argument(
        "--uncorrected_output",
        default=None,
        help=(
            "Also write the uncorrected (no implicit-gap correction) copy of every "
            "attempt here, in the same pass. derive_features_for_transformer.py needs "
            "this set as its fallback, so producing both at once avoids a second full "
            "run over the inputs. Default: {} when --output is the default, otherwise "
            "'<output>_no_gap_correction'. Pass 'none' to disable, or omit it when the "
            "correction is already off (the primary output is then the uncorrected "
            "set).".format(artifact_paths.display_path(DEFAULT_NO_GAP_CORRECTION_OUTPUT_DIR))
        ),
    )
    parser.add_argument(
        "--implicit_gap_threshold_seconds",
        type=float,
        default=float(IMPLICIT_GAP_THRESHOLD_SECONDS),
        help=(
            "Gaps between consecutive rows longer than this many seconds are subtracted "
            "from time_elapsed_since_job_start as unlogged pauses (default: 300, "
            "the player_sequences/ set). 0 or inf disables the correction (the "
            "uncorrected set). Note: an explicit exit->resume pause longer than the "
            "threshold is subtracted twice (see the module docstring)."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(JOB_CODE),
        choices=sorted(JOB_CODE),
        help="Task names to extract. Defaults to all configured tasks.",
    )
    args = parser.parse_args()
    output_was_default = args.output is None
    if args.output is None:
        args.output = str(
            DEFAULT_OUTPUT_DIR
            if implicit_gap_correction_enabled(args.implicit_gap_threshold_seconds)
            else DEFAULT_NO_GAP_CORRECTION_OUTPUT_DIR
        )

    # Resolve the second (uncorrected) output folder. It is only meaningful while the
    # correction is on: with the correction off the primary output already is that set.
    if not implicit_gap_correction_enabled(args.implicit_gap_threshold_seconds):
        args.uncorrected_output = None
    elif args.uncorrected_output is None:
        args.uncorrected_output = str(
            DEFAULT_NO_GAP_CORRECTION_OUTPUT_DIR
            if output_was_default
            else args.output.rstrip(os.sep) + "_no_gap_correction"
        )
    elif str(args.uncorrected_output).strip().lower() in ("none", ""):
        args.uncorrected_output = None

    if args.uncorrected_output and os.path.abspath(args.uncorrected_output) == os.path.abspath(
        args.output
    ):
        raise SystemExit("ERROR: --uncorrected_output must differ from --output.")
    return args


def implicit_gap_correction_enabled(threshold_seconds):
    """Return True when ``threshold_seconds`` turns the implicit-gap correction on.

    ``None``, 0 (or negative) and ``inf`` disable it.
    """
    if threshold_seconds is None:
        return False
    threshold_seconds = float(threshold_seconds)
    return 0 < threshold_seconds < float("inf")


def parse_time_utc(value):
    """Parse a Time_utc string into a naive UTC datetime or return None."""
    text = (value or "").strip()
    if not text:
        return None

    offset_minutes = 0
    if text.endswith("Z"):
        text = text[:-1].strip()
    else:
        match = re.search(r"([+-])(\d{2}):?(\d{2})$", text)
        if match:
            sign = 1 if match.group(1) == "+" else -1
            offset_minutes = sign * (int(match.group(2)) * 60 + int(match.group(3)))
            text = text[: match.start()].strip()

    text = text.replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(text, fmt)
            if offset_minutes:
                parsed -= timedelta(minutes=offset_minutes)
            return parsed
        except ValueError:
            continue
    return None


def canonical_job_code(row):
    """Return the target CurrentJobName for a row or an empty string.

    Only falls back to the ``task`` column when ``CurrentJobName`` is absent.
    If ``CurrentJobName`` is explicitly set to a non-target code (e.g. PWVAN),
    the row is rejected rather than silently reclassified via the fallback.
    """
    job_code = (row.get("CurrentJobName") or "").strip()
    if job_code in TARGET_JOB_CODES:
        return job_code
    if job_code:
        # Explicitly set to a non-target job — reject without fallback.
        return ""

    task = (row.get("task") or "").strip()
    return JOB_CODE.get(task, "")


def format_elapsed_seconds(seconds):
    """Format elapsed seconds compactly for CSV output."""
    if seconds is None:
        return None
    if seconds.is_integer():
        return int(seconds)
    return round(seconds, 6)


def paused_seconds_before_time(pause_intervals, parsed_time):
    """Return cumulative paused seconds before ``parsed_time``."""
    paused_seconds = 0.0
    for exit_time, resume_time in pause_intervals:
        if parsed_time <= exit_time:
            break
        if parsed_time >= resume_time:
            paused_seconds += (resume_time - exit_time).total_seconds()
            continue
        paused_seconds += (parsed_time - exit_time).total_seconds()
        break
    return paused_seconds


def load_lifecycle_events(path, event_type, events_by_pid, stats, allowed_job_codes):
    """Load lifecycle rows for the target jobs and group them by player."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stats["lifecycle_rows_seen"] += 1

            pid = (row.get("pid") or "").strip()
            if not pid:
                stats["skipped_lifecycle_missing_pid"] += 1
                continue

            job_code = canonical_job_code(row)
            if job_code not in allowed_job_codes:
                stats["skipped_lifecycle_non_target_job"] += 1
                continue

            timestamp = parse_time_utc(row.get("Time_utc"))
            if timestamp is None:
                stats["skipped_lifecycle_bad_time"] += 1
                continue

            events_by_pid[pid].append(
                {
                    "event_type": event_type,
                    "job_code": job_code,
                    "task": TASK_BY_JOB_CODE[job_code],
                    "Time_utc": row.get("Time_utc", ""),
                    "timestamp": timestamp,
                }
            )
            stats["lifecycle_rows_loaded"] += 1


def build_completed_attempts(events_by_pid, stats, exited_csv_present=False):
    """
    Convert lifecycle events into completed attempts.

    Only ``job_started`` opens an attempt and the next ``job_completed`` closes
    it; a new ``job_started`` before completion replaces the incomplete one.
    ``job_exited`` -> ``job_resumed`` pairs become pause intervals. When
    ``exited_csv_present`` is True, a ``job_resumed`` with no preceding
    ``job_exited`` marks the attempt malformed and it is dropped.
    """
    attempts = []

    for pid, events in sorted(events_by_pid.items()):
        open_attempt_by_job = {}
        attempt_counter_by_task = defaultdict(int)

        sorted_events = sorted(
            events,
            key=lambda event: (
                event["timestamp"],
                LIFECYCLE_EVENT_PRIORITY[event["event_type"]],
            ),
        )

        for event in sorted_events:
            job_code = event["job_code"]
            task = event["task"]
            event_type = event["event_type"]
            open_attempt = open_attempt_by_job.get(job_code)

            if event_type == "job_started":
                if open_attempt is not None:
                    stats["replaced_incomplete_starts"] += 1
                open_attempt_by_job[job_code] = {
                    "pid": pid,
                    "task": task,
                    "job_code": job_code,
                    "start_time": event["timestamp"],
                    "job_start_utc": event["Time_utc"],
                    "pause_intervals": [],
                    "pending_exit_time": None,
                    "resume_count": 0,
                    "malformed": False,
                }
                continue

            if event_type == "job_exited":
                if open_attempt is None:
                    stats["orphan_job_exited"] += 1
                    continue
                if event["timestamp"] < open_attempt["start_time"]:
                    stats["out_of_order_job_exited"] += 1
                    continue
                if open_attempt["pending_exit_time"] is not None:
                    stats["replaced_pending_job_exits"] += 1
                open_attempt["pending_exit_time"] = event["timestamp"]
                continue

            if event_type == "job_resumed":
                if open_attempt is None:
                    stats["orphan_job_resumed"] += 1
                    continue
                if event["timestamp"] >= open_attempt["start_time"]:
                    pending_exit_time = open_attempt["pending_exit_time"]
                    if pending_exit_time is not None:
                        if event["timestamp"] >= pending_exit_time:
                            open_attempt["pause_intervals"].append(
                                (pending_exit_time, event["timestamp"])
                            )
                            stats["completed_pause_intervals"] += 1
                        else:
                            stats["out_of_order_pause_resume"] += 1
                        open_attempt["pending_exit_time"] = None
                    elif exited_csv_present:
                        open_attempt["malformed"] = True
                        stats["resume_without_exit"] += 1
                    open_attempt["resume_count"] += 1
                else:
                    stats["out_of_order_job_resumed"] += 1
                continue

            if open_attempt is None:
                stats["orphan_job_completed"] += 1
                continue

            if event["timestamp"] < open_attempt["start_time"]:
                stats["out_of_order_job_completed"] += 1
                continue

            if open_attempt["malformed"]:
                stats["malformed_attempts_dropped"] += 1
                del open_attempt_by_job[job_code]
                continue

            attempt_counter_by_task[task] += 1
            attempt_number = attempt_counter_by_task[task]
            attempts.append(
                {
                    "pid": pid,
                    "task": task,
                    "job_code": job_code,
                    "attempt_number": attempt_number,
                    "attempt_id": "attempt{}".format(attempt_number),
                    "job_start_utc": open_attempt["job_start_utc"],
                    "job_end_utc": event["Time_utc"],
                    "start_time": open_attempt["start_time"],
                    "end_time": event["timestamp"],
                    "pause_intervals": list(open_attempt["pause_intervals"]),
                    "resume_count": open_attempt["resume_count"],
                    "rows": [],
                }
            )
            if open_attempt["pending_exit_time"] is not None:
                stats["dangling_job_exited_before_completion"] += 1
            stats["completed_attempts"] += 1
            del open_attempt_by_job[job_code]

        stats["incomplete_attempts_ignored"] += len(open_attempt_by_job)

    return attempts


def index_attempts(attempts):
    """Index attempts by player for per-player event extraction."""
    attempts_by_pid = defaultdict(list)

    for attempt in attempts:
        attempts_by_pid[attempt["pid"]].append(attempt)

    for pid_attempts in attempts_by_pid.values():
        pid_attempts.sort(key=lambda attempt: (attempt["start_time"], attempt["task"]))

    return attempts_by_pid


def build_output_row(src_row, event_type, attempt, parsed_time):
    """Map a raw event row to the merged per-attempt output schema."""
    raw_seconds = (parsed_time - attempt["start_time"]).total_seconds()
    active_seconds = raw_seconds - paused_seconds_before_time(
        attempt.get("pause_intervals", []),
        parsed_time,
    )
    output = []
    for col in OUTPUT_COLS:
        if col == "attempt_id":
            output.append(attempt["attempt_id"])
        elif col == "event_type":
            output.append(event_type)
        elif col == "CurrentJobName":
            output.append(attempt["job_code"])
        elif col == "job_start_utc":
            output.append(attempt["job_start_utc"])
        elif col == "job_end_utc":
            output.append(attempt["job_end_utc"])
        elif col == "time_elapsed_since_job_start":
            output.append(format_elapsed_seconds(max(active_seconds, 0.0)))
        elif col == "time_elapsed_since_job_start_raw":
            output.append(format_elapsed_seconds(raw_seconds))
        else:
            output.append(src_row.get(col, None))
    return output


def find_matching_attempt(job_attempts, parsed_time):
    """Return the completed attempt covering parsed_time, if any."""
    for attempt in job_attempts:
        if parsed_time < attempt["start_time"]:
            break
        if attempt["start_time"] <= parsed_time <= attempt["end_time"]:
            return attempt
    return None


def iter_rows_grouped_by_pid(path, stats):
    """
    Yield consecutive CSV rows grouped by player id.

    The simplified input tables are expected to be grouped by ``pid``. This
    lets the extractor keep only one player's rows in memory at a time.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        current_pid = None
        current_rows = []
        previous_pid = None

        for row in reader:
            stats["event_rows_seen"] += 1

            pid = (row.get("pid") or "").strip()
            if not pid:
                stats["skipped_event_missing_pid"] += 1
                continue

            if previous_pid is not None and pid < previous_pid:
                raise ValueError(
                    "Input '{}' is not grouped by pid; low-memory extraction "
                    "requires pid-sorted event rows.".format(path)
                )

            if current_pid is None:
                current_pid = pid

            if pid != current_pid:
                yield current_pid, current_rows
                current_pid = pid
                current_rows = []

            current_rows.append(row)
            previous_pid = pid

        if current_pid is not None:
            yield current_pid, current_rows


def input_rows_grouped_by_pid(path):
    """Return True when a CSV's non-empty pid rows are already grouped."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        previous_pid = None
        for row in reader:
            pid = (row.get("pid") or "").strip()
            if not pid:
                continue
            if previous_pid is not None and pid < previous_pid:
                return False
            previous_pid = pid
    return True


def write_pid_sorted_chunk(rows, fieldnames, temp_dir, chunk_index):
    """Write one pid-sorted chunk CSV and return its temporary path."""
    rows.sort(key=lambda row: (row.get("pid") or "").strip())
    chunk_path = os.path.join(
        temp_dir,
        "pid_sorted_chunk_{:06d}.csv".format(chunk_index),
    )
    with open(chunk_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    return chunk_path


def iter_rows_grouped_from_pid_sorted_chunks(chunk_paths):
    """Merge pid-sorted chunk CSVs and yield rows grouped by player id."""
    readers = []
    handles = []
    heap = []
    sequence_counter = 0

    try:
        for chunk_index, chunk_path in enumerate(chunk_paths):
            handle = open(chunk_path, newline="", encoding="utf-8")
            reader = csv.DictReader(handle)
            row = next(reader, None)
            if row is None:
                handle.close()
                continue

            pid = (row.get("pid") or "").strip()
            handles.append(handle)
            readers.append(reader)
            heapq.heappush(heap, (pid, chunk_index, sequence_counter, row))
            sequence_counter += 1

        current_pid = None
        current_rows = []
        while heap:
            pid, chunk_index, _, row = heapq.heappop(heap)

            if current_pid is None:
                current_pid = pid
            elif pid != current_pid:
                yield current_pid, current_rows
                current_pid = pid
                current_rows = []

            current_rows.append(row)

            next_row = next(readers[chunk_index], None)
            if next_row is not None:
                next_pid = (next_row.get("pid") or "").strip()
                heapq.heappush(
                    heap,
                    (next_pid, chunk_index, sequence_counter, next_row),
                )
                sequence_counter += 1

        if current_pid is not None:
            yield current_pid, current_rows
    finally:
        for handle in handles:
            handle.close()


def iter_rows_reordered_by_pid(path, stats):
    """
    Yield CSV rows grouped by pid, reordering unsorted inputs on disk.

    This fallback keeps memory bounded for large raw event tables while still
    letting the extractor process inputs whose pid rows are interleaved.
    """
    stats["event_inputs_reordered_by_pid"] += 1

    with tempfile.TemporaryDirectory(prefix="extract_player_sequences_") as temp_dir:
        chunk_paths = []
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = [
                fieldname for fieldname in (reader.fieldnames or []) if fieldname is not None
            ]
            chunk_rows = []
            chunk_index = 0

            for row in reader:
                stats["event_rows_seen"] += 1

                pid = (row.get("pid") or "").strip()
                if not pid:
                    stats["skipped_event_missing_pid"] += 1
                    continue

                chunk_rows.append(row)
                if len(chunk_rows) >= PID_SORT_CHUNK_ROWS:
                    chunk_paths.append(
                        write_pid_sorted_chunk(
                            chunk_rows,
                            fieldnames,
                            temp_dir,
                            chunk_index,
                        )
                    )
                    chunk_rows = []
                    chunk_index += 1

            if chunk_rows:
                chunk_paths.append(
                    write_pid_sorted_chunk(
                        chunk_rows,
                        fieldnames,
                        temp_dir,
                        chunk_index,
                    )
                )

        yield from iter_rows_grouped_from_pid_sorted_chunks(chunk_paths)


def iter_rows_by_pid(path, stats):
    """Yield CSV rows grouped by pid, reordering unsorted inputs if needed."""
    if input_rows_grouped_by_pid(path):
        yield from iter_rows_grouped_by_pid(path, stats)
        return
    yield from iter_rows_reordered_by_pid(path, stats)


def next_group(group_iter):
    """Return the next ``(pid, rows)`` group or None when exhausted."""
    try:
        return next(group_iter)
    except StopIteration:
        return None


def consume_group_for_pid(current_group, group_iter, target_pid, stats):
    """Advance a grouped-row iterator to target_pid and return that group's rows."""
    while current_group is not None and current_group[0] < target_pid:
        stats["skipped_event_pid_without_attempt"] += len(current_group[1])
        current_group = next_group(group_iter)

    if current_group is not None and current_group[0] == target_pid:
        matched_rows = current_group[1]
        current_group = next_group(group_iter)
        return matched_rows, current_group

    return [], current_group


def drain_remaining_groups(current_group, group_iter, stats):
    """Count and discard any remaining grouped rows after all target players."""
    while current_group is not None:
        stats["skipped_event_pid_without_attempt"] += len(current_group[1])
        current_group = next_group(group_iter)


def route_player_event_rows(rows, event_type, job_attempts_by_code, stats, allowed_job_codes):
    """Route one player's extracted rows into that player's completed attempts."""
    matched_key = "{}_rows_matched".format(event_type)
    for row in rows:
        job_code = canonical_job_code(row)
        if job_code not in allowed_job_codes:
            stats["skipped_event_non_target_job"] += 1
            continue

        job_attempts = job_attempts_by_code.get(job_code)
        if not job_attempts:
            stats["skipped_event_job_without_attempt"] += 1
            continue

        parsed_time = parse_time_utc(row.get("Time_utc"))
        if parsed_time is None:
            stats["skipped_event_bad_time"] += 1
            continue

        attempt = find_matching_attempt(job_attempts, parsed_time)
        if attempt is None:
            stats["skipped_event_outside_attempt_window"] += 1
            continue

        if event_type == "subtask_completed":
            raw_label = (row.get("LastSubtaskCompleted") or "").strip()
            normalized = normalize_subtask_label(raw_label)
            expected = EXPECTED_NORMALIZED_SUBTASKS_BY_TASK.get(attempt["task"], frozenset())
            if normalized and normalized not in expected:
                stats["skipped_subtask_wrong_task_vocabulary"] += 1
                continue
            # Rewrite the label with its normalized suffix so downstream files
            # never contain fully-qualified names.
            row = dict(row)
            row["LastSubtaskCompleted"] = normalized

        output_row = build_output_row(row, event_type, attempt, parsed_time)
        attempt["rows"].append((parsed_time, output_row))
        stats["event_rows_matched"] += 1
        stats[matched_key] += 1


def attempt_filename(attempt):
    """Build the output filename for a completed attempt."""
    return "{}_{}_{}.csv".format(
        attempt["pid"],
        attempt["task"],
        attempt["attempt_id"],
    )


def apply_implicit_gap_correction(rows, threshold_seconds=IMPLICIT_GAP_THRESHOLD_SECONDS):
    """Subtract unlogged pauses detected as gaps > ``threshold_seconds`` from active time.

    ``rows`` are chronological ``(sort_key, output_row)`` pairs. Each gap over the
    threshold is subtracted from ``time_elapsed_since_job_start`` of that row and
    all later rows (clamped at 0). The gap test does not skip explicit pauses that
    ``build_output_row`` already removed, so those are subtracted twice. This is a
    known limitation: the derived features depend on it, and
    ``derive_features_for_transformer.py`` falls back to the uncorrected copy when
    the corrected clock decreases.
    """
    if not implicit_gap_correction_enabled(threshold_seconds):
        return list(rows)
    threshold_seconds = float(threshold_seconds)
    cumulative_implicit_pause = 0.0
    prev_ts = None
    corrected = []
    for sort_key, row in rows:
        ts = parse_time_utc(row[TIME_UTC_IDX])
        if prev_ts is not None and ts is not None:
            gap = (ts - prev_ts).total_seconds()
            if gap > threshold_seconds:
                cumulative_implicit_pause += gap
        if ts is not None:
            prev_ts = ts
        if cumulative_implicit_pause > 0 and row[ELAPSED_IDX] not in (None, ""):
            try:
                row = list(row)
                row[ELAPSED_IDX] = format_elapsed_seconds(
                    max(float(row[ELAPSED_IDX]) - cumulative_implicit_pause, 0.0)
                )
            except (TypeError, ValueError):
                pass
        corrected.append((sort_key, row))
    return corrected


def write_attempt_csv(
    attempt, out_dir, implicit_gap_threshold_seconds=IMPLICIT_GAP_THRESHOLD_SECONDS
):
    """Write one CSV for a completed attempt under its task-specific folder.

    Rows are sorted chronologically, then the implicit-gap correction is applied
    with ``implicit_gap_threshold_seconds`` (0/None/inf: no correction).
    """
    rows = sorted(
        attempt["rows"],
        key=lambda item: (item[0], item[1][TIME_UTC_IDX] or "", item[1][EVENT_TYPE_IDX] or ""),
    )
    rows = apply_implicit_gap_correction(rows, implicit_gap_threshold_seconds)
    task_dir = os.path.join(out_dir, attempt["task"])
    os.makedirs(task_dir, exist_ok=True)
    path = os.path.join(task_dir, attempt_filename(attempt))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(OUTPUT_COLS)
        writer.writerows(row for _, row in rows)
    return len(rows)


def main():
    """Run attempt extraction and write one CSV per completed attempt."""
    args = parse_args()

    required_inputs = [
        (args.started, "job_started CSV"),
        (args.completed, "job_completed CSV"),
        (args.subtask, "subtask CSV"),
        (args.state, "state CSV"),
    ]
    if args.exited:
        required_inputs.append((args.exited, "job_exited CSV"))
    required_inputs.append((args.resumed, "job_resumed CSV"))
    for path, name in required_inputs:
        if not os.path.exists(path):
            sys.exit(
                "ERROR: {} not found at '{}'\n"
                "Provide the path explicitly or place the file under "
                "{}/.".format(name, path, artifact_paths.display_path(DEFAULT_DATA_DIRS[0]))
            )

    os.makedirs(args.output, exist_ok=True)
    if args.uncorrected_output:
        os.makedirs(args.uncorrected_output, exist_ok=True)

    print("job_started CSV  : {}".format(args.started))
    print("job_exited CSV   : {}".format(args.exited or "<not provided>"))
    print("job_resumed CSV  : {}".format(args.resumed))
    print("job_completed CSV: {}".format(args.completed))
    print("subtask CSV      : {}".format(args.subtask))
    print("state CSV        : {}".format(args.state))
    print("Tasks            : {}".format(", ".join(args.tasks)))
    print(
        "Implicit-gap correction: {}".format(
            "> {:g} s".format(args.implicit_gap_threshold_seconds)
            if implicit_gap_correction_enabled(args.implicit_gap_threshold_seconds)
            else "off"
        )
    )
    print("Output dir       : {}/".format(args.output))
    if args.uncorrected_output:
        print("Uncorrected dir  : {}/".format(args.uncorrected_output))
    t0 = time.time()

    stats = Counter()
    lifecycle_events_by_pid = defaultdict(list)
    allowed_job_codes = {JOB_CODE[task_name] for task_name in args.tasks}

    print("\nStep 1: loading lifecycle events...", flush=True)
    load_lifecycle_events(
        args.started,
        "job_started",
        lifecycle_events_by_pid,
        stats,
        allowed_job_codes,
    )
    if args.exited:
        load_lifecycle_events(
            args.exited,
            "job_exited",
            lifecycle_events_by_pid,
            stats,
            allowed_job_codes,
        )
    load_lifecycle_events(
        args.resumed,
        "job_resumed",
        lifecycle_events_by_pid,
        stats,
        allowed_job_codes,
    )
    load_lifecycle_events(
        args.completed,
        "job_completed",
        lifecycle_events_by_pid,
        stats,
        allowed_job_codes,
    )
    print(
        "  -> {:,} lifecycle rows loaded for {:,} players".format(
            stats["lifecycle_rows_loaded"],
            len(lifecycle_events_by_pid),
        )
    )

    print("Step 2: building completed attempts...", flush=True)
    attempts = build_completed_attempts(
        lifecycle_events_by_pid,
        stats,
        exited_csv_present=bool(args.exited),
    )
    attempts_by_pid = index_attempts(attempts)
    print(
        "  -> {:,} completed attempts across {:,} players".format(
            len(attempts),
            len(attempts_by_pid),
        )
    )

    print("Step 3: streaming extracted rows by player and writing attempts...", flush=True)
    subtask_iter = iter_rows_by_pid(args.subtask, stats)
    state_iter = iter_rows_by_pid(args.state, stats)
    current_subtask_group = next_group(subtask_iter)
    current_state_group = next_group(state_iter)

    empty_attempts = 0
    written_rows = 0
    written_files = 0
    for index, pid in enumerate(
        sorted(attempts_by_pid),
        start=1,
    ):
        pid_attempts = attempts_by_pid[pid]
        job_attempts_by_code = defaultdict(list)
        for attempt in pid_attempts:
            attempt["rows"] = []
            job_attempts_by_code[attempt["job_code"]].append(attempt)

        subtask_rows, current_subtask_group = consume_group_for_pid(
            current_subtask_group,
            subtask_iter,
            pid,
            stats,
        )
        route_player_event_rows(
            subtask_rows,
            "subtask_completed",
            job_attempts_by_code,
            stats,
            allowed_job_codes,
        )

        state_rows, current_state_group = consume_group_for_pid(
            current_state_group,
            state_iter,
            pid,
            stats,
        )
        route_player_event_rows(
            state_rows,
            "update_current_state",
            job_attempts_by_code,
            stats,
            allowed_job_codes,
        )

        for attempt in pid_attempts:
            row_count = write_attempt_csv(attempt, args.output, args.implicit_gap_threshold_seconds)
            if args.uncorrected_output:
                # Same rows, correction off. apply_implicit_gap_correction copies each
                # row before editing it, so writing the attempt twice is safe.
                write_attempt_csv(attempt, args.uncorrected_output, 0)
            written_rows += row_count
            written_files += 1
            if row_count == 0:
                empty_attempts += 1
            attempt["rows"] = []

        if index % 1000 == 0:
            print("  {:,}/{:,} players".format(index, len(attempts_by_pid)), flush=True)

    drain_remaining_groups(current_subtask_group, subtask_iter, stats)
    drain_remaining_groups(current_state_group, state_iter, stats)

    elapsed = time.time() - t0
    print("\n{}".format("-" * 50))
    print("Done in {:.0f}s".format(elapsed))
    print("  Completed attempts          : {:,}".format(len(attempts)))
    print("  Output files written        : {:,}".format(written_files))
    print("  Matched subtask rows        : {:,}".format(stats["subtask_completed_rows_matched"]))
    print("  Matched state rows          : {:,}".format(stats["update_current_state_rows_matched"]))
    print("  Matched event rows written  : {:,}".format(written_rows))
    print("  Empty completed attempts    : {:,}".format(empty_attempts))
    print("  Pause intervals recorded    : {:,}".format(stats["completed_pause_intervals"]))
    print("  Malformed attempts dropped  : {:,}".format(stats["malformed_attempts_dropped"]))
    print("  Resume rows without exit    : {:,}".format(stats["resume_without_exit"]))
    print("  Replaced incomplete starts  : {:,}".format(stats["replaced_incomplete_starts"]))
    print("  Ignored incomplete attempts : {:,}".format(stats["incomplete_attempts_ignored"]))
    print("  Skipped bad lifecycle times : {:,}".format(stats["skipped_lifecycle_bad_time"]))
    print("  Skipped bad event times     : {:,}".format(stats["skipped_event_bad_time"]))
    print(
        "  Skipped cross-task subtasks : {:,}".format(
            stats["skipped_subtask_wrong_task_vocabulary"]
        )
    )
    print("  Event files re-grouped      : {:,}".format(stats["event_inputs_reordered_by_pid"]))
    print("  Output folder               : {}/".format(args.output))
    if args.uncorrected_output:
        print("  Uncorrected folder          : {}/".format(args.uncorrected_output))


if __name__ == "__main__":
    main()
