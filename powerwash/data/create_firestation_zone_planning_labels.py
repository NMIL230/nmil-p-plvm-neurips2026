#!/usr/bin/env python3
"""
Derive player-level zone-planning labels from one task's subtask completion order.

Subtask locations are clustered into spatial zones, each player's completion
order becomes a zone sequence, and a ZoneHoppingScore (see
``assign_zone_labels_cv``) splits players into Zone Planner (low), Mixed and
Zone Hopper (high). Reads ``powerwash/data/player_sequences/`` and writes labels,
features, fold manifests and figures to ``powerwash/outputs/<prefix>_zone_labels/``
(prefix ``firestation`` for Fire Station, else the task name);
``--label_variant drop_<component>`` variants go to
``powerwash/outputs/firestation_zone_labels_sensitivity/<variant>/``.

Each player is labelled from one attempt: the one with the highest
``LevelProgressionAmount`` (ties: lowest attempt number). That attempt must also
meet the completion and unique-subtask coverage thresholds.
"""

import argparse
import csv
import json
import math
import pickle
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.cluster import KMeans

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plvm import artifact_paths  # noqa: E402

from powerwash.analysis.plot_subtask_task_maps import maybe_set_plot_style  # noqa: E402
from powerwash.data.derive_features_for_transformer import POSITION_PATTERN  # noqa: E402
from powerwash.data.powerwash_subtask_cleaning import (  # noqa: E402
    EXPECTED_NORMALIZED_SUBTASKS_BY_TASK,
    normalize_subtask_label,
)


DEFAULT_TASK = "fire_station"
DEFAULT_INPUT_DIR = artifact_paths.PLAYER_SEQUENCES_DIR
DEFAULT_SUBTASK_MAP = artifact_paths.SUBTASK_TASK_MAP_POINTS_CSV
DEFAULT_OUTPUT_ROOT = artifact_paths.POWERWASH_OUTPUTS_DIR
DEFAULT_SENSITIVITY_MANIFEST_ROOT = artifact_paths.FIRESTATION_LABELS_DIR
SENSITIVITY_OUTPUT_ROOT = artifact_paths.FIRESTATION_SENSITIVITY_LABELS_DIR
DEFAULT_LABEL_VARIANT = "original"
SENSITIVITY_LABEL_VARIANTS = (
    "original",
    "drop_zone_switch_rate",
    "drop_return_rate",
    "drop_mean_zone_run_length",
    "drop_mean_zone_jump_distance",
)
_LABEL_VARIANT_TO_DROPPED_FEATURE = {
    "original": "",
    "drop_zone_switch_rate": "zone_switch_rate",
    "drop_return_rate": "return_rate",
    "drop_mean_zone_run_length": "mean_zone_run_length",
    "drop_mean_zone_jump_distance": "mean_zone_jump_distance",
}

ZONE_PALETTE = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#999999",
]

LABEL_COLORS = {
    "Zone Planner": "#377eb8",
    "Mixed": "#4daf4a",
    "Zone Hopper": "#e41a1c",
    "": "#cccccc",
}

_DATETIME_FMTS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")
MIN_UNIQUE_SUBTASK_FRACTION = 0.90
SCORE_HISTOGRAM_BIN_WIDTH = 0.5
PUBLICATION_PANEL_WIDTH_IN = 2.13
PUBLICATION_HISTOGRAM_HEIGHT_IN = 2.10
PUBLICATION_PANEL_HEIGHT_IN = PUBLICATION_HISTOGRAM_HEIGHT_IN
PUBLICATION_TITLE_FONTSIZE = 8.2
PUBLICATION_LABEL_FONTSIZE = 7.0
PUBLICATION_TICK_FONTSIZE = 6.0
PUBLICATION_LEGEND_FONTSIZE = 5.8
PUBLICATION_COMPACT_ANNOTATION_FONTSIZE = 4.6
PUBLICATION_COMPACT_CAPTION_FONTSIZE = 6.6
PUBLICATION_COMPACT_ROUTE_ALPHA = 0.56
PUBLICATION_SAVEFIG_DPI = 600


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_dt(s: str):
    """Parse a UTC datetime string tolerantly."""
    s = (s or "").strip()
    for fmt in _DATETIME_FMTS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def _parse_xz(raw: str):
    """Return ``(x, z)`` from a ``CurrentPosition`` string, or ``None``."""
    cleaned = (raw or "").strip()
    if not cleaned:
        return None
    m = POSITION_PATTERN.match(cleaned)
    if m is None:
        return None
    return float(m.group(1)), float(m.group(3))


def _normalize(name: str) -> str:
    """Normalise a raw subtask label to its canonical leaf name."""
    return normalize_subtask_label(name)


def _task_label(task_name: str) -> str:
    """Convert an internal task name into a presentation label."""
    return task_name.replace("_", " ").title()


def _artifact_prefix(task_name: str) -> str:
    """Return the filename prefix used for outputs for one task."""
    return "firestation" if task_name == DEFAULT_TASK else str(task_name)


def _expected_subtasks(task_name: str):
    """Return the canonical normalized subtask set for one task."""
    expected = EXPECTED_NORMALIZED_SUBTASKS_BY_TASK.get(str(task_name))
    if expected is None:
        available = ", ".join(sorted(EXPECTED_NORMALIZED_SUBTASKS_BY_TASK))
        raise ValueError("Unknown task '{}'. Available tasks: {}".format(task_name, available))
    return expected


def _min_unique_completed_subtasks(task_name: str) -> int:
    """Return the minimum number of unique subtasks required for one task."""
    return int(math.ceil(MIN_UNIQUE_SUBTASK_FRACTION * len(_expected_subtasks(task_name))))


def _resolve_task_input_dir(input_dir: Path, task_name: str) -> Path:
    """Resolve a task-specific input directory from either a root or task dir."""
    input_dir = Path(input_dir).expanduser().resolve()
    task_dir = input_dir / task_name
    if task_dir.exists() and task_dir.is_dir():
        return task_dir
    return input_dir


def _default_output_dir(task_name: str) -> Path:
    """Return the default output directory for one task."""
    return artifact_paths.zone_labels_dir(_artifact_prefix(task_name))


def _resolve_output_dir(output_dir, task_name: str) -> Path:
    """Resolve the output directory, defaulting to a task-specific location."""
    if output_dir is None:
        return _default_output_dir(task_name)
    return Path(output_dir).expanduser().resolve()


def _robust_z_params(series: pd.Series) -> tuple:
    """Return (median, iqr) for a robust z-score fit. Zero IQR is replaced by 1."""
    med = float(series.median())
    iqr = float(series.quantile(0.75) - series.quantile(0.25))
    if iqr == 0:
        iqr = 1.0
    return med, iqr


def _robust_z_apply(series: pd.Series, med: float, iqr: float) -> pd.Series:
    """Apply a pre-fitted robust z-score transform."""
    return (series - med) / iqr


def _score_histogram_bin_edges(
    scores: pd.Series,
    bin_width: float = SCORE_HISTOGRAM_BIN_WIDTH,
) -> np.ndarray:
    """Return fixed-width histogram bin edges spanning all finite score values."""
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")

    numeric_scores = pd.to_numeric(scores, errors="coerce")
    finite_scores = numeric_scores[np.isfinite(numeric_scores.to_numpy(dtype="float64"))]
    if finite_scores.empty:
        half_width = bin_width / 2.0
        return np.asarray([-half_width, half_width], dtype="float64")

    lower = math.floor(float(finite_scores.min()) / bin_width) * bin_width
    upper = math.ceil(float(finite_scores.max()) / bin_width) * bin_width
    if math.isclose(lower, upper):
        upper = lower + bin_width

    n_bins = max(1, int(round((upper - lower) / bin_width)))
    return lower + (np.arange(n_bins + 1, dtype="float64") * bin_width)


def _save_publication_figure(fig, path: Path):
    """Save a single-column figure panel at a fixed page size."""
    fig.savefig(path, dpi=PUBLICATION_SAVEFIG_DPI)


def _apply_publication_axis_style(
    ax,
    xlabel=None,
    ylabel=None,
    show_ticks=True,
):
    """Apply compact axis styling suitable for narrow figure panels."""
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=PUBLICATION_LABEL_FONTSIZE, labelpad=1.5)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=PUBLICATION_LABEL_FONTSIZE, labelpad=1.5)

    if show_ticks:
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=PUBLICATION_TICK_FONTSIZE,
            length=2.5,
            pad=1.2,
        )
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def _zone_color(zone_id: int) -> str:
    return ZONE_PALETTE[int(zone_id) % len(ZONE_PALETTE)]


def _zone_hopping_score_formula(label_variant: str = DEFAULT_LABEL_VARIANT) -> str:
    """Return a compact human-readable ZoneHoppingScore formula string."""
    terms = []
    for feature_name, sign in _active_zone_hopping_features(label_variant):
        prefix = "+" if sign > 0 and terms else "-" if sign < 0 else ""
        body = "z({})".format(feature_name)
        if prefix:
            terms.append("{} {}".format(prefix, body))
        else:
            terms.append(body if sign > 0 else "- {}".format(body))
    return "ZoneHoppingScore = {}".format(" ".join(terms))


def _count_unique_completed_subtasks(subtask_sequence) -> int:
    """Count unique normalized completed subtasks in one attempt sequence."""
    return len(
        {
            normalized
            for _time_utc, subtask_name, _x, _z in subtask_sequence
            for normalized in [_normalize(subtask_name)]
            if normalized
        }
    )


def _apply_record_eligibility(
    record: dict, min_completion: float, task_name: str = DEFAULT_TASK
) -> dict:
    """Set one player record's exclusion fields from the eligibility rules."""
    n_unique_completed_subtasks = record.get("n_unique_completed_subtasks")
    if n_unique_completed_subtasks is None:
        n_unique_completed_subtasks = _count_unique_completed_subtasks(
            record.get("subtask_sequence", [])
        )
        record["n_unique_completed_subtasks"] = n_unique_completed_subtasks
    min_unique_completed_subtasks = _min_unique_completed_subtasks(task_name)

    exclusion_reasons = []
    if float(record.get("completion", 0.0)) < min_completion:
        exclusion_reasons.append(
            "completion={:.3f} < {:.3f}".format(
                float(record.get("completion", 0.0)),
                float(min_completion),
            )
        )
    if int(n_unique_completed_subtasks) < min_unique_completed_subtasks:
        exclusion_reasons.append(
            "n_unique_completed_subtasks={} < {}".format(
                int(n_unique_completed_subtasks),
                min_unique_completed_subtasks,
            )
        )

    record["excluded"] = bool(exclusion_reasons)
    record["exclusion_reason"] = "; ".join(exclusion_reasons)
    return record


def refresh_player_records(
    player_records: dict,
    min_completion: float,
    task_name: str = DEFAULT_TASK,
) -> dict:
    """Fill in missing record metadata and re-apply eligibility filters in place."""
    for record in player_records.values():
        record.setdefault("task", task_name)
        _apply_record_eligibility(record, min_completion, task_name)
    return player_records


# ---------------------------------------------------------------------------
# 1. Load task data
# ---------------------------------------------------------------------------


def load_firestation_data(
    input_dir: Path,
    min_completion: float = 0.90,
    task_name: str = DEFAULT_TASK,
):
    """
    Load one task's attempt CSVs and select one attempt per player.

    The attempt with the highest ``LevelProgressionAmount`` is kept (ties: lowest
    attempt number), then marked ``excluded`` unless it meets the completion and
    unique-subtask coverage thresholds. Returns ``player_id -> record``; each
    record's ``subtask_sequence`` is a time-sorted list of
    ``(Time_utc, subtask_name, x, z)`` and ``duration_s`` is wall-clock seconds.
    """
    input_dir = Path(input_dir)
    task_token = "_{}_".format(task_name)
    csv_paths = [
        csv_path for csv_path in sorted(input_dir.glob("*.csv")) if task_token in csv_path.stem
    ]
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    # Group files by player
    by_player: dict[str, list[Path]] = defaultdict(list)
    for p in csv_paths:
        stem = p.stem  # p<pid>_<task>_attempt<n>
        pid_part = stem.split(task_token)[0]
        by_player[pid_part].append(p)

    print(f"[load] Found {len(csv_paths)} files for {len(by_player)} unique players.")

    records = {}
    for pid, paths in by_player.items():
        best = None  # (completion, attempt_num, record_dict)

        for path in paths:
            stem = path.stem
            # Extract attempt number for tie-breaking
            m = re.search(r"attempt(\d+)$", stem)
            attempt_num = int(m.group(1)) if m else 999

            subtask_rows = []
            unique_completed_subtasks = set()
            job_start_dt = job_end_dt = None
            last_elapsed_raw = 0.0

            with path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    # Parse datetime fields from first available row
                    if job_start_dt is None:
                        job_start_dt = _parse_dt(row.get("job_start_utc", ""))
                        job_end_dt = _parse_dt(row.get("job_end_utc", ""))

                    if (row.get("event_type") or "").strip() != "subtask_completed":
                        continue

                    subtask_name = _normalize(row.get("LastSubtaskCompleted", ""))
                    if not subtask_name:
                        continue
                    unique_completed_subtasks.add(subtask_name)

                    xz = _parse_xz(row.get("CurrentPosition", ""))
                    x = xz[0] if xz else None
                    z = xz[1] if xz else None

                    try:
                        elapsed_raw = float(row.get("time_elapsed_since_job_start_raw") or 0)
                        last_elapsed_raw = max(last_elapsed_raw, elapsed_raw)
                    except (ValueError, TypeError):
                        pass

                    subtask_rows.append(
                        (
                            row.get("Time_utc", ""),
                            subtask_name,
                            x,
                            z,
                        )
                    )

            if not subtask_rows:
                continue

            # Sort by timestamp
            subtask_rows.sort(key=lambda r: r[0])

            # Completion = max LevelProgressionAmount across subtask_completed rows
            # Re-read to get the progression values
            progression_vals = []
            with path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if (row.get("event_type") or "").strip() != "subtask_completed":
                        continue
                    try:
                        progression_vals.append(float(row["LevelProgressionAmount"]))
                    except (ValueError, TypeError, KeyError):
                        pass
            completion = max(progression_vals) if progression_vals else 0.0

            # Duration: wall-clock from job timestamps, fall back to raw elapsed
            if job_start_dt and job_end_dt:
                duration_s = (job_end_dt - job_start_dt).total_seconds()
                if duration_s <= 0:
                    duration_s = last_elapsed_raw
            else:
                duration_s = last_elapsed_raw

            attempt_id = stem.split(task_token)[-1]  # e.g. "attempt1"

            record = {
                "player_id": pid,
                "attempt_id": attempt_id,
                "task": task_name,
                "completion": completion,
                "duration_s": max(duration_s, 0.0),
                "n_unique_completed_subtasks": len(unique_completed_subtasks),
                "subtask_sequence": subtask_rows,
                "excluded": False,
                "exclusion_reason": "",
            }

            if (
                best is None
                or completion > best[0]
                or (completion == best[0] and attempt_num < best[1])
            ):
                best = (completion, attempt_num, record)

        if best is None:
            continue

        records[pid] = _apply_record_eligibility(best[2], min_completion, task_name)

    n_included = sum(1 for r in records.values() if not r["excluded"])
    n_excluded = len(records) - n_included
    min_unique_completed_subtasks = _min_unique_completed_subtasks(task_name)
    print(
        f"[load] {n_included} included, {n_excluded} excluded "
        f"(completion >= {min_completion:.3f}, "
        f"n_unique_completed_subtasks >= {min_unique_completed_subtasks} "
        f"for task '{task_name}')."
    )
    return records


# ---------------------------------------------------------------------------
# 2. Build subtask zones
# ---------------------------------------------------------------------------


def build_subtask_zones(
    subtask_map_path: Path,
    n_zones: int = 6,
    rng_seed: int = 42,
    task_name: str = DEFAULT_TASK,
):
    """
    Cluster one task's canonical subtask x/z coordinates into ``n_zones`` KMeans zones.

    Returns ``(subtask_zone_map, zone_centroids, df_map)``: subtask name -> zone id,
    an ``(n_zones, 2)`` array of x/z centroids, and the per-subtask zone table.
    """
    df = pd.read_csv(subtask_map_path)
    df_fs = df[df["task"] == task_name].copy()
    df_fs = df_fs.dropna(subset=["x", "z"]).reset_index(drop=True)

    if df_fs.empty:
        raise ValueError(
            "No '{}' rows with valid x/z in subtask_task_map_points.csv".format(task_name)
        )

    coords = df_fs[["x", "z"]].values
    km = KMeans(n_clusters=n_zones, n_init=20, random_state=rng_seed)
    df_fs["zone_id"] = km.fit_predict(coords)
    zone_centroids = km.cluster_centers_  # shape (n_zones, 2)

    subtask_zone_map = dict(zip(df_fs["subtask_name"], df_fs["zone_id"].astype(int)))

    df_map = df_fs[["subtask_name", "x", "z", "zone_id"]].copy()
    df_map.insert(
        0,
        "subtask_id",
        df_fs["point_id"].values if "point_id" in df_fs.columns else range(1, len(df_map) + 1),
    )

    print(
        "[zones] Clustered {} subtasks into {} zones for task '{}'.".format(
            len(df_fs),
            n_zones,
            task_name,
        )
    )
    for zi in range(n_zones):
        cnt = (df_fs["zone_id"] == zi).sum()
        cx, cz = zone_centroids[zi]
        print(f"  Zone {zi}: {cnt} subtasks, centroid=({cx:.2f}, {cz:.2f})")

    return subtask_zone_map, zone_centroids, df_map


# ---------------------------------------------------------------------------
# 3. Build player zone sequences
# ---------------------------------------------------------------------------


def build_player_zone_sequences(player_records: dict, subtask_zone_map: dict) -> dict:
    """
    Convert each included player's subtask completion order to zone sequences.

    Returns ``player_id -> {"raw_zones", "collapsed_zones"}``; the collapsed
    sequence drops consecutive repeats ([2, 2, 5, 1, 1, 5] -> [2, 5, 1, 5]).
    Subtasks without a zone are skipped.
    """
    result = {}
    n_missing_warn = 0
    for pid, rec in player_records.items():
        if rec["excluded"]:
            continue

        raw_zones = []
        for _t, subtask_name, _x, _z in rec["subtask_sequence"]:
            if subtask_name in subtask_zone_map:
                raw_zones.append(subtask_zone_map[subtask_name])
            else:
                n_missing_warn += 1

        # Collapse consecutive identical zones
        collapsed = []
        for z in raw_zones:
            if not collapsed or collapsed[-1] != z:
                collapsed.append(z)

        result[pid] = {"raw_zones": raw_zones, "collapsed_zones": collapsed}

    if n_missing_warn:
        print(
            f"[sequences] {n_missing_warn} subtask completions had no zone mapping "
            "(not in subtask_task_map_points.csv) — skipped."
        )
    print(f"[sequences] Built zone sequences for {len(result)} players.")
    return result


# ---------------------------------------------------------------------------
# 4. Compute zone features
# ---------------------------------------------------------------------------


def compute_zone_features(
    player_records: dict,
    zone_sequences: dict,
    zone_centroids: np.ndarray,
    task_name: str = DEFAULT_TASK,
) -> pd.DataFrame:
    """
    Compute zone-order features per included player.

    With ``n_completed_subtasks`` mapped completions (raw sequence) and
    ``n_zone_visits`` zone runs (collapsed sequence):

    - ``zone_switch_rate`` = (n_zone_visits - 1) / max(n_completed_subtasks - 1, 1)
    - ``mean_zone_run_length``: mean length of same-zone runs in the raw sequence
    - ``return_rate`` = visits to an already-visited zone / max(n_zone_visits, 1)
    - ``mean_zone_jump_distance``: mean centroid distance between consecutive
      zones of the collapsed sequence
    """
    rows = []
    for pid, rec in player_records.items():
        if rec["excluded"]:
            continue
        if pid not in zone_sequences:
            continue

        raw = zone_sequences[pid]["raw_zones"]
        collapsed = zone_sequences[pid]["collapsed_zones"]
        n_completed = len(raw)
        n_visits = len(collapsed)

        if n_completed == 0:
            continue

        # zone_switch_rate
        zone_switch_rate = (n_visits - 1) / max(n_completed - 1, 1)

        # mean_zone_run_length from raw sequence
        if raw:
            run_lengths = []
            run = 1
            for i in range(1, len(raw)):
                if raw[i] == raw[i - 1]:
                    run += 1
                else:
                    run_lengths.append(run)
                    run = 1
            run_lengths.append(run)
            mean_zone_run_length = float(np.mean(run_lengths))
        else:
            mean_zone_run_length = 0.0

        # return_rate: a return is a visit to a zone that was already visited
        # and then left at least once before this visit
        if collapsed:
            seen = set()
            n_returns = 0
            for i, z in enumerate(collapsed):
                if i > 0 and z in seen:
                    n_returns += 1
                seen.add(z)
            return_rate = n_returns / max(n_visits, 1)
        else:
            return_rate = 0.0

        # mean_zone_jump_distance
        if len(collapsed) >= 2:
            dists = []
            for i in range(1, len(collapsed)):
                c0 = zone_centroids[collapsed[i - 1]]
                c1 = zone_centroids[collapsed[i]]
                dists.append(math.sqrt((c0[0] - c1[0]) ** 2 + (c0[1] - c1[1]) ** 2))
            mean_zone_jump_distance = float(np.mean(dists))
        else:
            mean_zone_jump_distance = 0.0

        rows.append(
            {
                "player_id": pid,
                "task": task_name,
                "n_unique_completed_subtasks": rec["n_unique_completed_subtasks"],
                "n_completed_subtasks": n_completed,
                "n_zone_visits": n_visits,
                "zone_switch_rate": zone_switch_rate,
                "mean_zone_run_length": mean_zone_run_length,
                "return_rate": return_rate,
                "mean_zone_jump_distance": mean_zone_jump_distance,
                "task_completion": rec["completion"],
                "task_duration": rec["duration_s"],
            }
        )

    df = pd.DataFrame(rows)
    print(f"[features] Computed features for {len(df)} players.")
    return df


# ---------------------------------------------------------------------------
# 5. Assign zone labels
# ---------------------------------------------------------------------------

_ZONE_HOPPING_FEATURES = [
    ("zone_switch_rate", +1),
    ("return_rate", +1),
    ("mean_zone_run_length", -1),
    ("mean_zone_jump_distance", +1),
]


def _active_zone_hopping_features(label_variant: str):
    """Return the active score terms for one label-variant setting."""
    variant = str(label_variant or DEFAULT_LABEL_VARIANT).strip()
    dropped = _LABEL_VARIANT_TO_DROPPED_FEATURE.get(variant)
    if dropped is None:
        raise ValueError(
            "Unknown label_variant '{}'. Expected one of: {}".format(
                variant,
                ", ".join(SENSITIVITY_LABEL_VARIANTS),
            )
        )
    return [spec for spec in _ZONE_HOPPING_FEATURES if spec[0] != dropped]


def _label_variant_suffix(label_variant: str) -> str:
    """Return the filesystem-safe suffix used for sensitivity output roots."""
    variant = str(label_variant or DEFAULT_LABEL_VARIANT).strip()
    return "" if variant == DEFAULT_LABEL_VARIANT else variant


def _sensitivity_output_dir(label_variant: str) -> Path:
    """Return the default root output directory for one sensitivity variant.

    Variant outputs always go under the Fire Station sensitivity root
    (``firestation_zone_labels_sensitivity/<variant>/``), whatever ``--task`` is;
    the full score (``original``) uses the Fire Station label folder.
    """
    suffix = _label_variant_suffix(label_variant)
    if not suffix:
        return _default_output_dir(DEFAULT_TASK)
    return SENSITIVITY_OUTPUT_ROOT / suffix


def _load_existing_split_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load one existing player split manifest and validate its schema."""
    manifest = pd.read_csv(manifest_path).copy()
    required = {"player_id", "split", "fold"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(
            "Manifest {} missing required columns: {}".format(
                manifest_path,
                ", ".join(missing),
            )
        )
    manifest["player_id"] = manifest["player_id"].astype(str)
    manifest["split"] = manifest["split"].astype(str).str.strip().str.lower()
    manifest["fold"] = manifest["fold"].astype(str).str.strip()
    return manifest


def _load_sensitivity_fold_manifests(manifest_root: Path) -> list:
    """Return ordered ``(fold_name, manifest_path, manifest_df)`` triples."""
    root = Path(manifest_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError("Manifest root not found: {}".format(root))

    fold_dirs = sorted(
        path
        for path in root.iterdir()
        if (path.is_dir() and re.fullmatch(r"fold_\d+", path.name) is not None)
    )
    if not fold_dirs:
        raise ValueError("No fold_* directories found under {}".format(root))

    manifests = []
    for fold_dir in fold_dirs:
        manifest_path = fold_dir / "player_split_manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError("Missing manifest: {}".format(manifest_path))
        manifests.append(
            (fold_dir.name, manifest_path, _load_existing_split_manifest(manifest_path))
        )
    return manifests


def build_cv_folds(player_ids: list, n_folds: int = 10, seed: int = 42) -> dict:
    """
    Assign each player to one of ``n_folds`` disjoint folds.

    Returns
    -------
    dict mapping player_id -> fold_index (0 .. n_folds - 1)
    """
    rng = np.random.default_rng(seed)
    shuffled = list(player_ids)
    rng.shuffle(shuffled)
    return {pid: i % n_folds for i, pid in enumerate(shuffled)}


def assign_zone_labels_cv(
    features_df: pd.DataFrame,
    train_ids: set,
    lower_q: float = 0.35,
    upper_q: float = 0.65,
    label_variant: str = DEFAULT_LABEL_VARIANT,
) -> tuple:
    """
    Compute ZoneHoppingScore and assign labels with train-only normalization.

    ZoneHoppingScore = z(zone_switch_rate) + z(return_rate) - z(mean_zone_run_length)
    + z(mean_zone_jump_distance), with robust z (median / IQR) fitted on
    ``train_ids``; a ``drop_<component>`` variant omits that term. Scores at or
    below the train ``lower_q`` quantile are Zone Planner, at or above ``upper_q``
    Zone Hopper, else Mixed; ``label_2_class`` splits at the train median.
    Returns ``(labeled_df, thresholds)``, the latter written to
    ``label_thresholds.json``.
    """
    df = features_df.copy()
    train_mask = df["player_id"].isin(train_ids)

    score = pd.Series(0.0, index=df.index, dtype="float64")
    robust_z_params: dict = {}
    active_features = _active_zone_hopping_features(label_variant)
    for col, sign in active_features:
        med, iqr = _robust_z_params(df.loc[train_mask, col])
        robust_z_params[col] = {"median": med, "iqr": iqr}
        score = score + sign * _robust_z_apply(df[col], med, iqr)

    df["zone_hopping_score"] = score.values

    train_scores = df.loc[train_mask, "zone_hopping_score"]
    lo = float(train_scores.quantile(lower_q))
    hi = float(train_scores.quantile(upper_q))
    median_score = float(train_scores.median())

    def _label_3(s):
        if s <= lo:
            return "Zone Planner"
        if s >= hi:
            return "Zone Hopper"
        return "Mixed"

    df["label"] = df["zone_hopping_score"].apply(_label_3)
    df["label_2_class"] = df["zone_hopping_score"].apply(
        lambda s: "Zone Hopper" if s >= median_score else "Zone Planner"
    )
    df["label_2_class_clean"] = df["label"].apply(
        lambda lbl: lbl if lbl in ("Zone Planner", "Zone Hopper") else ""
    )

    thresholds = {
        "label_variant": str(label_variant),
        "dropped_feature": _LABEL_VARIANT_TO_DROPPED_FEATURE[str(label_variant)],
        "zone_hopping_features": [
            {"feature": feature_name, "sign": int(sign)} for feature_name, sign in active_features
        ],
        "zone_hopping_score_formula": _zone_hopping_score_formula(label_variant=label_variant),
        "lower_quantile": lower_q,
        "upper_quantile": upper_q,
        "lower_threshold": lo,
        "upper_threshold": hi,
        "median_threshold": median_score,
        "n_train_players": int(train_mask.sum()),
        "robust_z_params": robust_z_params,
    }

    counts_3 = df["label"].value_counts()
    counts_2 = df["label_2_class"].value_counts()
    counts_2c = df["label_2_class_clean"].value_counts()
    print(
        f"[labels 3-class] Zone Planner: {counts_3.get('Zone Planner', 0)}, "
        f"Mixed: {counts_3.get('Mixed', 0)}, "
        f"Zone Hopper: {counts_3.get('Zone Hopper', 0)}"
    )
    print(
        f"[labels 2-class] Zone Planner: {counts_2.get('Zone Planner', 0)}, "
        f"Zone Hopper: {counts_2.get('Zone Hopper', 0)}"
    )
    print(
        f"[labels 2-class-clean] Zone Planner: {counts_2c.get('Zone Planner', 0)}, "
        f"Zone Hopper: {counts_2c.get('Zone Hopper', 0)}, "
        f"Excluded (Mixed): {(df['label_2_class_clean'] == '').sum()}"
    )
    return df, thresholds


def assign_zone_labels(
    features_df: pd.DataFrame,
    lower_q: float = 0.35,
    upper_q: float = 0.65,
    label_variant: str = DEFAULT_LABEL_VARIANT,
) -> pd.DataFrame:
    """
    Compute ZoneHoppingScore and assign both 3-class and 2-class labels.

    ``assign_zone_labels_cv`` with all players as the training set (used when
    ``--n_folds`` <= 1).
    """
    labeled_df, _thresholds = assign_zone_labels_cv(
        features_df,
        train_ids=set(features_df["player_id"]),
        lower_q=lower_q,
        upper_q=upper_q,
        label_variant=label_variant,
    )
    return labeled_df


# ---------------------------------------------------------------------------
# 6. Save outputs
# ---------------------------------------------------------------------------


def save_outputs(
    labeled_df: pd.DataFrame,
    excluded_records: dict,
    df_map: pd.DataFrame,
    zone_centroids: np.ndarray,
    output_dir: Path,
    summary_meta: dict,
    zone_sequences: dict,
    task_name: str = DEFAULT_TASK,
    label_variant: str = DEFAULT_LABEL_VARIANT,
    player_split_manifest_df=None,
    label_thresholds=None,
    player_records=None,
):
    """
    Write all output files (CSVs, JSON, figures) to ``output_dir``.

    Optional inputs are written only when given: ``player_split_manifest_df`` as
    ``player_split_manifest.csv``, ``label_thresholds`` as ``label_thresholds.json``,
    and ``player_records`` to render exemplar zone-switching figures.
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    artifact_prefix = _artifact_prefix(task_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --- <task>_zone_labels.csv ---
    # Include excluded players as rows with empty label
    excluded_rows = []
    for pid, rec in excluded_records.items():
        excluded_rows.append(
            {
                "player_id": pid,
                "task": task_name,
                "label": "",
                "zone_hopping_score": float("nan"),
                "zone_switch_rate": float("nan"),
                "return_rate": float("nan"),
                "mean_zone_run_length": float("nan"),
                "mean_zone_jump_distance": float("nan"),
                "n_unique_completed_subtasks": rec["n_unique_completed_subtasks"],
                "n_completed_subtasks": float("nan"),
                "n_zone_visits": float("nan"),
                "task_completion": rec["completion"],
                "task_duration": rec["duration_s"],
                "excluded": True,
                "exclusion_reason": rec["exclusion_reason"],
            }
        )

    base_cols = [
        "player_id",
        "task",
        "zone_hopping_score",
        "zone_switch_rate",
        "return_rate",
        "mean_zone_run_length",
        "mean_zone_jump_distance",
        "n_unique_completed_subtasks",
        "n_completed_subtasks",
        "n_zone_visits",
        "task_completion",
        "task_duration",
        "excluded",
        "exclusion_reason",
    ]
    included_export = labeled_df.copy()
    included_export["excluded"] = False
    included_export["exclusion_reason"] = ""

    excluded_df = pd.DataFrame(excluded_rows)

    def _build_all_df(label_col: str) -> pd.DataFrame:
        cols = ["player_id", "task", "label"] + [
            c for c in base_cols if c not in ("player_id", "task")
        ]
        inc = included_export.copy()
        inc["label"] = inc[label_col]
        exc = excluded_df.copy() if not excluded_df.empty else pd.DataFrame(columns=cols)
        return pd.concat(
            [inc[cols], exc[cols] if not exc.empty else pd.DataFrame(columns=cols)],
            ignore_index=True,
        )

    # 3-class labels CSV
    all_df_3 = _build_all_df("label")
    labels_3_path = output_dir / "{}_zone_labels_3_class.csv".format(artifact_prefix)
    all_df_3.to_csv(labels_3_path, index=False)
    print(f"[save] {labels_3_path}  ({len(all_df_3)} rows)")

    # Also written as <prefix>_zone_labels.csv, the name the figure scripts read
    labels_path = output_dir / "{}_zone_labels.csv".format(artifact_prefix)
    all_df_3.to_csv(labels_path, index=False)
    print(f"[save] {labels_path}  ({len(all_df_3)} rows, alias for 3-class)")

    # 2-class labels CSV
    all_df_2 = _build_all_df("label_2_class")
    labels_2_path = output_dir / "{}_zone_labels_2_class.csv".format(artifact_prefix)
    all_df_2.to_csv(labels_2_path, index=False)
    print(f"[save] {labels_2_path}  ({len(all_df_2)} rows)")

    # 2-class clean labels CSV (extreme groups only; Mixed rows have empty label)
    all_df_2c = _build_all_df("label_2_class_clean")
    labels_2c_path = output_dir / "{}_zone_labels_2_class_clean.csv".format(artifact_prefix)
    all_df_2c.to_csv(labels_2c_path, index=False)
    n_labeled = (all_df_2c["label"] != "").sum()
    print(f"[save] {labels_2c_path}  ({n_labeled} labeled, {len(all_df_2c) - n_labeled} excluded)")

    # --- <task>_subtask_zone_map.csv ---
    map_path = output_dir / "{}_subtask_zone_map.csv".format(artifact_prefix)
    df_map.to_csv(map_path, index=False)
    print(f"[save] {map_path}  ({len(df_map)} rows)")

    # --- <task>_zone_centroids.csv ---
    centroids_df = pd.DataFrame(
        {
            "zone_id": range(len(zone_centroids)),
            "centroid_x": zone_centroids[:, 0],
            "centroid_z": zone_centroids[:, 1],
        }
    )
    centroids_path = output_dir / "{}_zone_centroids.csv".format(artifact_prefix)
    centroids_df.to_csv(centroids_path, index=False)
    print(f"[save] {centroids_path}")

    # --- label_summary.json ---
    lv3 = labeled_df["label"].value_counts()
    lv2 = labeled_df["label_2_class"].value_counts()
    lv2c = labeled_df["label_2_class_clean"].value_counts()
    summary = {
        **summary_meta,
        "n_zone_planner": int(lv3.get("Zone Planner", 0)),
        "n_zone_hopper": int(lv3.get("Zone Hopper", 0)),
        "n_mixed": int(lv3.get("Mixed", 0)),
        "n_zone_planner_2_class": int(lv2.get("Zone Planner", 0)),
        "n_zone_hopper_2_class": int(lv2.get("Zone Hopper", 0)),
        "n_zone_planner_2_class_clean": int(lv2c.get("Zone Planner", 0)),
        "n_zone_hopper_2_class_clean": int(lv2c.get("Zone Hopper", 0)),
        "n_excluded_2_class_clean": int((labeled_df["label_2_class_clean"] == "").sum()),
    }
    summary_path = output_dir / "label_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[save] {summary_path}")

    # --- player_split_manifest.csv (K-fold CV only) ---
    if player_split_manifest_df is not None:
        manifest_path = output_dir / "player_split_manifest.csv"
        player_split_manifest_df.to_csv(manifest_path, index=False)
        print(f"[save] {manifest_path}  ({len(player_split_manifest_df)} rows)")

    # --- label_thresholds.json (K-fold CV only) ---
    if label_thresholds is not None:
        thresholds_path = output_dir / "label_thresholds.json"
        with open(thresholds_path, "w") as fh:
            json.dump(label_thresholds, fh, indent=2)
        print(f"[save] {thresholds_path}")

    # --- Figures ---
    maybe_set_plot_style()
    _plot_zone_clusters(df_map, zone_centroids, figures_dir, task_name=task_name)
    _plot_score_histograms_by_labeling_method(
        labeled_df,
        figures_dir,
        task_name=task_name,
        label_variant=label_variant,
        lower_q=summary_meta.get("lower_quantile", 0.35),
        upper_q=summary_meta.get("upper_quantile", 0.65),
    )
    _plot_three_class_score_histogram(
        labeled_df,
        figures_dir,
        task_name=task_name,
        label_variant=label_variant,
    )
    _plot_top_bottom_trajectories(
        labeled_df,
        zone_sequences,
        df_map,
        zone_centroids,
        figures_dir,
        task_name=task_name,
        label_variant=label_variant,
    )
    _plot_planner_vs_hopper_zone_switching(
        labeled_df,
        player_records,
        df_map,
        figures_dir,
        task_name=task_name,
    )

    print(f"[save] Figures written to {figures_dir}/")


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------


def _plot_zone_clusters(
    df_map: pd.DataFrame,
    zone_centroids: np.ndarray,
    figures_dir: Path,
    task_name: str = DEFAULT_TASK,
):
    """Plot the zone clusters as a fixed-size panel for side-by-side layouts."""
    fig, ax = plt.subplots(figsize=(PUBLICATION_PANEL_WIDTH_IN, PUBLICATION_PANEL_HEIGHT_IN))
    n_zones = len(zone_centroids)
    x_min = float(df_map["x"].min())
    x_max = float(df_map["x"].max())
    z_min = float(df_map["z"].min())
    z_max = float(df_map["z"].max())
    x_pad = max((x_max - x_min) * 0.003, 0.05)
    z_pad = max((z_max - z_min) * 0.003, 0.05)

    for zi in range(n_zones):
        sub = df_map[df_map["zone_id"] == zi]
        ax.scatter(
            sub["x"],
            sub["z"],
            s=9,
            alpha=0.88,
            linewidths=0,
            color=_zone_color(zi),
            zorder=2,
        )
        ax.scatter(
            zone_centroids[zi, 0],
            zone_centroids[zi, 1],
            s=72,
            marker="o",
            color=_zone_color(zi),
            edgecolors="black",
            linewidths=0.5,
            zorder=4,
        )
        ax.text(
            zone_centroids[zi, 0],
            zone_centroids[zi, 1],
            str(zi),
            ha="center",
            va="center",
            fontsize=PUBLICATION_COMPACT_ANNOTATION_FONTSIZE,
            fontweight="bold",
            color="white",
            zorder=5,
        )

    ax.set_aspect("auto")
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(z_min - z_pad, z_max + z_pad)
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_edgecolor("#222222")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    path = figures_dir / "{}_zone_clusters.pdf".format(_artifact_prefix(task_name))
    _save_publication_figure(fig, path)
    plt.close(fig)


def _score_histogram_plot_context(labeled_df: pd.DataFrame) -> dict:
    """Collect shared binning, colours, and thresholds for score histograms."""
    planner_scores = labeled_df.loc[
        labeled_df["label"] == "Zone Planner",
        "zone_hopping_score",
    ]
    hopper_scores = labeled_df.loc[
        labeled_df["label"] == "Zone Hopper",
        "zone_hopping_score",
    ]
    return {
        "bins": _score_histogram_bin_edges(
            labeled_df["zone_hopping_score"],
            bin_width=SCORE_HISTOGRAM_BIN_WIDTH,
        ),
        "planner_color": LABEL_COLORS.get("Zone Planner", "steelblue"),
        "mixed_color": LABEL_COLORS.get("Mixed", "gold"),
        "hopper_color": LABEL_COLORS.get("Zone Hopper", "tomato"),
        "lower_threshold": (
            float(planner_scores.max()) if not planner_scores.empty else float("nan")
        ),
        "upper_threshold": (
            float(hopper_scores.min()) if not hopper_scores.empty else float("nan")
        ),
        "median_score": float(labeled_df["zone_hopping_score"].median()),
    }


def _three_class_publication_labels(labeled_df: pd.DataFrame) -> list:
    """Return compact legend labels with counts for 3-class plots."""
    labels = []
    for long_label, short_label in [
        ("Zone Planner", "Planner"),
        ("Mixed", "Mixed"),
        ("Zone Hopper", "Hopper"),
    ]:
        count = int((labeled_df["label"] == long_label).sum())
        labels.append("{} (n = {})".format(short_label, count))
    return labels


def _plot_three_class_score_histogram_panel(
    ax,
    labeled_df: pd.DataFrame,
    plot_context: dict,
    *,
    title: str,
    title_fontsize: int = 10,
    legend_fontsize: int = 8,
    show_ylabel: bool = True,
    compact_legend: bool = False,
    threshold_label_mode: str = "full",
):
    """Draw the 3-class score distribution on one axis."""
    planner_color = plot_context["planner_color"]
    mixed_color = plot_context["mixed_color"]
    hopper_color = plot_context["hopper_color"]

    for label, color in [
        ("Zone Planner", planner_color),
        ("Mixed", mixed_color),
        ("Zone Hopper", hopper_color),
    ]:
        sub = labeled_df.loc[labeled_df["label"] == label, "zone_hopping_score"]
        if not sub.empty:
            ax.hist(sub, bins=plot_context["bins"], alpha=0.7, color=color, label=label)

    lower_threshold = plot_context["lower_threshold"]
    upper_threshold = plot_context["upper_threshold"]
    if np.isfinite(lower_threshold):
        ax.axvline(
            lower_threshold,
            color=planner_color,
            lw=1.6,
            ls="--",
            label=f"Lower ({lower_threshold:.2f})" if threshold_label_mode == "full" else None,
        )
    if np.isfinite(upper_threshold):
        ax.axvline(
            upper_threshold,
            color=hopper_color,
            lw=1.6,
            ls="--",
            label=f"Upper ({upper_threshold:.2f})" if threshold_label_mode == "full" else None,
        )

    if title:
        ax.set_title(title, fontsize=title_fontsize)
    ax.grid(axis="y", linewidth=0.35, alpha=0.25)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    if show_ylabel:
        ax.set_ylabel("Count")
    ax.set_xlim(float(plot_context["bins"][0]), float(plot_context["bins"][-1]))
    if compact_legend:
        planner_label, mixed_label, hopper_label = _three_class_publication_labels(labeled_df)
        handles = [
            mpatches.Patch(color=planner_color, label=planner_label),
            mpatches.Patch(color=mixed_color, label=mixed_label),
            mpatches.Patch(color=hopper_color, label=hopper_label),
        ]
        ax.legend(
            handles=handles,
            fontsize=legend_fontsize,
            frameon=False,
            loc="best",
            handlelength=0.9,
            labelspacing=0.25,
            handletextpad=0.4,
        )
    else:
        ax.legend(fontsize=legend_fontsize)


def _plot_score_histograms_by_labeling_method(
    labeled_df: pd.DataFrame,
    figures_dir: Path,
    task_name: str = DEFAULT_TASK,
    label_variant: str = DEFAULT_LABEL_VARIANT,
    lower_q: float = 0.35,
    upper_q: float = 0.65,
):
    """Three-panel figure comparing all labeling methods on the same score axis.

    Panel 1 — 3-class (Zone Planner / Mixed / Zone Hopper)
    Panel 2 — 2-class median split
    Panel 3 — 2-class clean (extreme groups, Mixed excluded / greyed)
    """
    plot_context = _score_histogram_plot_context(labeled_df)
    bins = plot_context["bins"]
    planner_color = plot_context["planner_color"]
    hopper_color = plot_context["hopper_color"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

    # --- Panel 1: 3-class ---
    ax = axes[0]
    _plot_three_class_score_histogram_panel(
        ax,
        labeled_df,
        plot_context,
        title="3-Class\n({:.0f}% / {:.0f}% / {:.0f}%)".format(
            100 * lower_q, 100 * (upper_q - lower_q), 100 * (1 - upper_q)
        ),
    )

    # --- Panel 2: 2-class median split ---
    ax = axes[1]
    median_score = plot_context["median_score"]
    for label, color in [("Zone Planner", planner_color), ("Zone Hopper", hopper_color)]:
        sub = labeled_df[labeled_df["label_2_class"] == label]["zone_hopping_score"]
        if not sub.empty:
            ax.hist(sub, bins=bins, alpha=0.65, color=color, label=label)
    ax.axvline(median_score, color="black", lw=1.5, ls="--", label=f"Median ({median_score:.2f})")
    ax.set_title("2-Class Median Split\n(50% / 50%)", fontsize=10)
    ax.set_xlabel("ZoneHoppingScore")
    ax.legend(fontsize=8)

    # --- Panel 3: 2-class clean (extreme groups) ---
    ax = axes[2]
    mixed = labeled_df[labeled_df["label_2_class_clean"] == ""]["zone_hopping_score"]
    if not mixed.empty:
        ax.hist(mixed, bins=bins, alpha=0.25, color="grey", label="Excluded (Mixed)")
    for label, color in [("Zone Planner", planner_color), ("Zone Hopper", hopper_color)]:
        sub = labeled_df[labeled_df["label_2_class_clean"] == label]["zone_hopping_score"]
        if not sub.empty:
            ax.hist(sub, bins=bins, alpha=0.65, color=color, label=label)
    lower_threshold = plot_context["lower_threshold"]
    upper_threshold = plot_context["upper_threshold"]
    if np.isfinite(lower_threshold):
        ax.axvline(
            lower_threshold,
            color=planner_color,
            lw=1.5,
            ls="--",
            label=f"Lower ({lower_threshold:.2f})",
        )
    if np.isfinite(upper_threshold):
        ax.axvline(
            upper_threshold,
            color=hopper_color,
            lw=1.5,
            ls="--",
            label=f"Upper ({upper_threshold:.2f})",
        )
    n_clean = (labeled_df["label_2_class_clean"] != "").sum()
    n_total = len(labeled_df)
    ax.set_title(
        f"2-Class Clean (Extreme Groups)\n"
        f"({n_clean}/{n_total} labeled, {100 * (n_total - n_clean) / n_total:.0f}% excluded)",
        fontsize=10,
    )
    ax.set_xlabel("ZoneHoppingScore")
    ax.legend(fontsize=8)

    for ax in axes:
        ax.set_xlim(float(bins[0]), float(bins[-1]))

    fig.suptitle(
        "{} — ZoneHoppingScore by Labeling Method\n{}".format(
            _task_label(task_name),
            _zone_hopping_score_formula(label_variant=label_variant),
        ),
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = figures_dir / "{}_score_histograms_by_labeling_method.pdf".format(
        _artifact_prefix(task_name)
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_three_class_score_histogram(
    labeled_df: pd.DataFrame,
    figures_dir: Path,
    task_name: str = DEFAULT_TASK,
    label_variant: str = DEFAULT_LABEL_VARIANT,
):
    """Write a narrow single-column 3-class score distribution PDF."""
    plot_context = _score_histogram_plot_context(labeled_df)
    fig, ax = plt.subplots(figsize=(PUBLICATION_PANEL_WIDTH_IN, PUBLICATION_HISTOGRAM_HEIGHT_IN))
    _plot_three_class_score_histogram_panel(
        ax,
        labeled_df,
        plot_context,
        title="",
        title_fontsize=PUBLICATION_TITLE_FONTSIZE,
        legend_fontsize=PUBLICATION_LEGEND_FONTSIZE,
        compact_legend=True,
        threshold_label_mode="none",
    )
    _apply_publication_axis_style(ax, xlabel="Score", ylabel="Players")
    fig.tight_layout(pad=0.25)
    path = figures_dir / "{}_score_histogram_3_class.pdf".format(_artifact_prefix(task_name))
    _save_publication_figure(fig, path)
    plt.close(fig)


def _select_trajectory_plot_groups(
    labeled_df: pd.DataFrame,
    n_each: int = 5,
):
    """Select representative player IDs for hopper, mixed, and planner rows."""
    hoppers = (
        labeled_df[labeled_df["label"] == "Zone Hopper"]
        .nlargest(n_each, "zone_hopping_score")["player_id"]
        .tolist()
    )
    planners = (
        labeled_df[labeled_df["label"] == "Zone Planner"]
        .nsmallest(n_each, "zone_hopping_score")["player_id"]
        .tolist()
    )

    mixed_df = (
        labeled_df[labeled_df["label"] == "Mixed"]
        .sort_values(["zone_hopping_score", "player_id"])
        .reset_index(drop=True)
    )
    if len(mixed_df) <= n_each:
        mixed = mixed_df["player_id"].tolist()
    else:
        start = max((len(mixed_df) - n_each) // 2, 0)
        mixed = mixed_df.iloc[start : start + n_each]["player_id"].tolist()

    return [
        ("Zone Hopper", hoppers),
        ("Mixed", mixed),
        ("Zone Planner", planners),
    ]


def _select_extreme_trajectory_players(labeled_df: pd.DataFrame):
    """Return the most extreme planner and hopper player IDs for standalone plots."""
    planners = (
        labeled_df[labeled_df["label"] == "Zone Planner"]
        .nsmallest(1, "zone_hopping_score")["player_id"]
        .tolist()
    )
    hoppers = (
        labeled_df[labeled_df["label"] == "Zone Hopper"]
        .nlargest(1, "zone_hopping_score")["player_id"]
        .tolist()
    )

    selected = []
    if planners:
        selected.append(("Zone Planner", planners[0]))
    if hoppers:
        selected.append(("Zone Hopper", hoppers[0]))
    return selected


def _zone_entry_path_points(player_record, df_map: pd.DataFrame):
    """Return one point per contiguous zone run using observed coords when available."""
    if not player_record:
        return []

    zone_by_subtask = dict(zip(df_map["subtask_name"], df_map["zone_id"]))
    coord_by_subtask = dict(zip(df_map["subtask_name"], zip(df_map["x"], df_map["z"])))

    path_points = []
    last_zone_id = None
    for _time_utc, subtask_name, observed_x, observed_z in player_record.get(
        "subtask_sequence", []
    ):
        zone_id = zone_by_subtask.get(subtask_name)
        if zone_id is None:
            continue

        if observed_x is None or observed_z is None:
            mapped_coord = coord_by_subtask.get(subtask_name)
            if mapped_coord is None:
                continue
            point_x, point_z = mapped_coord
        else:
            point_x, point_z = observed_x, observed_z

        zone_id = int(zone_id)
        if last_zone_id == zone_id:
            continue

        path_points.append(
            {
                "x": float(point_x),
                "z": float(point_z),
                "zone_id": zone_id,
            }
        )
        last_zone_id = zone_id

    return path_points


def _planner_hopper_caption_text(label_name: str, score_val: float) -> str:
    """Return the compact caption shown on each planner/hopper comparison panel."""
    panel_label = "Planner" if label_name == "Zone Planner" else "Hopper"
    if np.isfinite(score_val):
        return "{}\nStyle-score = {:.1f}".format(panel_label, score_val)
    return "{}\nStyle-score = N/A".format(panel_label)


def _plot_planner_vs_hopper_zone_switching(
    labeled_df: pd.DataFrame,
    player_records,
    df_map: pd.DataFrame,
    figures_dir: Path,
    task_name: str = DEFAULT_TASK,
    max_annotations: int = 5,
    selected_players=None,
):
    """Write a fixed-size side-by-side planner-vs-hopper panel.

    ``selected_players`` is a list of ``(label, player_id)`` pairs to draw; by default the
    most extreme planner and hopper by zone-hopping score are used.
    """
    if selected_players is None:
        selected_players = _select_extreme_trajectory_players(labeled_df)
    if player_records is None or len(selected_players) < 2:
        print("[plot] Skipping planner-vs-hopper zone-switching PDF; missing examples.")
        return

    zone_ids = sorted(int(zone_id) for zone_id in df_map["zone_id"].unique())
    x_min = float(df_map["x"].min())
    x_max = float(df_map["x"].max())
    z_min = float(df_map["z"].min())
    z_max = float(df_map["z"].max())
    x_pad = max((x_max - x_min) * 0.01, 0.15)
    z_pad = max((z_max - z_min) * 0.01, 0.15)
    fig, axes = plt.subplots(
        1,
        len(selected_players),
        figsize=(PUBLICATION_PANEL_WIDTH_IN, PUBLICATION_PANEL_HEIGHT_IN),
        sharex=True,
        sharey=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for ax, (label_name, player_id) in zip(axes, selected_players):
        for zone_id in zone_ids:
            sub = df_map[df_map["zone_id"] == zone_id]
            ax.scatter(
                sub["x"],
                sub["z"],
                s=4,
                alpha=0.14,
                color=_zone_color(zone_id),
                linewidths=0,
                zorder=1,
            )

        path_points = _zone_entry_path_points(player_records.get(player_id), df_map)
        if path_points:
            xs = [point["x"] for point in path_points]
            zs = [point["z"] for point in path_points]
            colors = [_zone_color(point["zone_id"]) for point in path_points]

            ax.plot(
                xs,
                zs,
                color="#333333",
                linewidth=0.9,
                alpha=PUBLICATION_COMPACT_ROUTE_ALPHA,
                zorder=2,
            )
            ax.scatter(xs, zs, s=18, c=colors, edgecolors="white", linewidths=0.35, zorder=3)
            ax.scatter(
                xs[0],
                zs[0],
                s=28,
                facecolors="none",
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
            )
            ax.scatter(
                xs[-1],
                zs[-1],
                s=22,
                marker="s",
                c=[colors[-1]],
                edgecolors="black",
                linewidths=0.45,
                zorder=4,
            )
            ax.text(
                xs[0],
                zs[0],
                "S",
                ha="center",
                va="bottom",
                fontsize=PUBLICATION_COMPACT_ANNOTATION_FONTSIZE,
                color="black",
                zorder=5,
            )
            ax.text(
                xs[-1],
                zs[-1],
                "E",
                ha="center",
                va="bottom",
                fontsize=PUBLICATION_COMPACT_ANNOTATION_FONTSIZE,
                color="black",
                zorder=5,
            )

        score_series = labeled_df.loc[
            labeled_df["player_id"] == player_id,
            "zone_hopping_score",
        ]
        score_val = float(score_series.iloc[0]) if not score_series.empty else float("nan")
        ax.text(
            0.97,
            0.04,
            _planner_hopper_caption_text(label_name, score_val),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=PUBLICATION_COMPACT_CAPTION_FONTSIZE,
            fontweight="bold",
            linespacing=1.2,
            bbox={
                "facecolor": "white",
                "alpha": 0.9,
                "edgecolor": "#d1d5db",
                "boxstyle": "round,pad=0.35",
            },
            zorder=6,
        )
        ax.set_aspect("auto")
        ax.set_facecolor("white")
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(z_min - z_pad, z_max + z_pad)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.75)
            spine.set_edgecolor("#222222")

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.02)
    path = figures_dir / "{}_planner_vs_hopper_zone_switching.pdf".format(
        _artifact_prefix(task_name)
    )
    _save_publication_figure(fig, path)
    plt.close(fig)


def _plot_top_bottom_trajectories(
    labeled_df: pd.DataFrame,
    zone_sequences: dict,
    df_map: pd.DataFrame,
    zone_centroids: np.ndarray,
    figures_dir: Path,
    n_each: int = 5,
    task_name: str = DEFAULT_TASK,
    label_variant: str = DEFAULT_LABEL_VARIANT,
):
    """
    Completion-order trajectory plots for top-N Zone Hoppers, middle-N Mixed
    players, and bottom-N Zone Planners, coloured by zone.
    """
    subjects = _select_trajectory_plot_groups(labeled_df, n_each=n_each)
    n_cols = n_each
    n_rows = len(subjects)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 4 * n_rows), sharey=False)
    if n_rows == 1:
        axes = [axes]

    for row_idx, (label_name, pids) in enumerate(subjects):
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("#f5f5f5")

            if col_idx >= len(pids):
                ax.axis("off")
                continue

            pid = pids[col_idx]
            if pid not in zone_sequences:
                ax.axis("off")
                continue

            raw = zone_sequences[pid]["raw_zones"]
            seq_score = labeled_df.loc[labeled_df["player_id"] == pid, "zone_hopping_score"]
            score_val = float(seq_score.iloc[0]) if len(seq_score) else float("nan")

            # Draw the completion-order path through the zone centroids.
            xs = [zone_centroids[z][0] for z in raw]
            zs = [zone_centroids[z][1] for z in raw]
            colors = [_zone_color(z) for z in raw]

            for i in range(1, len(xs)):
                ax.plot(
                    [xs[i - 1], xs[i]],
                    [zs[i - 1], zs[i]],
                    color=colors[i],
                    alpha=0.6,
                    linewidth=1.2,
                )

            ax.set_title(f"{pid}\nscore={score_val:.2f}", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(label_name, fontsize=9, fontweight="bold")

    # Zone legend
    handles = [
        mpatches.Patch(color=_zone_color(zi), label=f"Zone {zi}")
        for zi in range(len(zone_centroids))
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=len(zone_centroids), fontsize=8, frameon=True
    )
    fig.suptitle(
        "{} | Top-{} Zone Hoppers vs Middle-{} Mixed vs Bottom-{} Zone Planners".format(
            _task_label(task_name),
            n_each,
            n_each,
            n_each,
        ),
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.955,
        _zone_hopping_score_formula(label_variant=label_variant),
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])

    path = figures_dir / "{}_top_bottom_trajectories.pdf".format(_artifact_prefix(task_name))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    """Parse the command-line arguments of the labeller."""
    p = argparse.ArgumentParser(
        description="Derive task-specific zone-planning labels per player.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--task",
        default=DEFAULT_TASK,
        choices=sorted(EXPECTED_NORMALIZED_SUBTASKS_BY_TASK),
        help="Task to label.",
    )
    p.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing task subfolders or a single task's per-attempt CSVs.",
    )
    p.add_argument(
        "--subtask_map",
        type=Path,
        default=DEFAULT_SUBTASK_MAP,
        help="subtask_task_map_points.csv path.",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for all artefacts. Defaults to a task-specific folder under outputs/.",
    )
    p.add_argument("--n_zones", type=int, default=6, help="Number of KMeans clusters (zones).")
    p.add_argument(
        "--min_completion", type=float, default=0.90, help="Minimum task completion to be included."
    )
    p.add_argument(
        "--lower_quantile",
        type=float,
        default=0.35,
        help="Score quantile below which players are Zone Planners.",
    )
    p.add_argument(
        "--upper_quantile",
        type=float,
        default=0.65,
        help="Score quantile above which players are Zone Hoppers.",
    )
    p.add_argument(
        "--no_cache", action="store_true", help="Ignore and overwrite any existing data cache."
    )
    p.add_argument(
        "--n_folds",
        type=int,
        default=10,
        help="Number of CV folds. When > 1, runs K-fold CV with disjoint player "
        "folds and writes one output subdirectory per fold (fold_00, fold_01, …). "
        "When 0 or 1, runs a single global labeling pass.",
    )
    p.add_argument(
        "--fold_seed",
        type=int,
        default=42,
        help="RNG seed used to shuffle players into folds.",
    )
    p.add_argument(
        "--label_variant",
        type=str,
        default=DEFAULT_LABEL_VARIANT,
        choices=list(SENSITIVITY_LABEL_VARIANTS),
        help=(
            "Zone-hopping score formula variant. "
            "'original' is the full score; each drop_* variant removes one term."
        ),
    )
    p.add_argument(
        "--sensitivity_manifest_root",
        type=Path,
        default=None,
        help=(
            "Optional root containing existing fold_*/player_split_manifest.csv files "
            "to reuse exactly for label sensitivity analysis."
        ),
    )
    return p.parse_args(argv)


def _cache_path(output_dir: Path, task_name: str = DEFAULT_TASK) -> Path:
    return output_dir / ".cache" / "{}_player_records.pkl".format(_artifact_prefix(task_name))


def _cache_is_fresh(cache_path: Path, input_dir: Path) -> bool:
    """Return True iff the cache is newer than every CSV in input_dir."""
    if not cache_path.exists():
        return False
    cache_mtime = cache_path.stat().st_mtime
    for csv_path in input_dir.glob("*.csv"):
        if csv_path.stat().st_mtime > cache_mtime:
            return False
    return True


def main(argv=None):
    """Load the task's attempts, build zones and scores, assign labels and write outputs."""
    args = parse_args(argv)
    input_dir = _resolve_task_input_dir(args.input_dir, args.task)
    sensitivity_mode = args.sensitivity_manifest_root is not None
    if sensitivity_mode:
        if args.label_variant == DEFAULT_LABEL_VARIANT:
            raise ValueError("--sensitivity_manifest_root requires a non-original --label_variant.")
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir is not None
            else _sensitivity_output_dir(args.label_variant)
        )
    else:
        output_dir = _resolve_output_dir(args.output_dir, args.task)
    min_unique_completed_subtasks = _min_unique_completed_subtasks(args.task)
    expected_subtasks = _expected_subtasks(args.task)

    # ---- 1. Load data (with optional cache) ----
    cache = _cache_path(output_dir, args.task)
    if not args.no_cache and _cache_is_fresh(cache, input_dir):
        print(f"[cache] Loading player records from {cache}")
        with open(cache, "rb") as fh:
            player_records = pickle.load(fh)
        player_records = refresh_player_records(
            player_records,
            args.min_completion,
            task_name=args.task,
        )
    else:
        player_records = load_firestation_data(
            input_dir,
            args.min_completion,
            task_name=args.task,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as fh:
            pickle.dump(player_records, fh)
        print(f"[cache] Saved player records to {cache}")

    # ---- 2. Build subtask zones ----
    subtask_zone_map, zone_centroids, df_map = build_subtask_zones(
        args.subtask_map,
        n_zones=args.n_zones,
        task_name=args.task,
    )

    # ---- 3. Build player zone sequences ----
    zone_sequences = build_player_zone_sequences(player_records, subtask_zone_map)

    # ---- 4. Compute features ----
    features_df = compute_zone_features(
        player_records,
        zone_sequences,
        zone_centroids,
        task_name=args.task,
    )
    if features_df.empty:
        print("ERROR: No features computed — check that included players have valid data.")
        sys.exit(1)

    # ---- 5 & 6. Assign labels and save outputs ----
    excluded_records = {pid: rec for pid, rec in player_records.items() if rec["excluded"]}
    n_total = len(player_records)
    n_included = n_total - len(excluded_records)

    if sensitivity_mode:
        manifest_triplets = _load_sensitivity_fold_manifests(args.sensitivity_manifest_root)
        print(
            "\n[sensitivity] Reusing {} existing fold manifests from {}".format(
                len(manifest_triplets),
                Path(args.sensitivity_manifest_root).expanduser().resolve(),
            )
        )
        print("[sensitivity] label_variant={}".format(args.label_variant))

        for fold_name, manifest_path, split_manifest_df in manifest_triplets:
            fold_output_dir = output_dir / fold_name
            print("\n=== {} ===".format(fold_name))

            train_ids = set(
                split_manifest_df.loc[split_manifest_df["split"] == "train", "player_id"].astype(
                    str
                )
            )
            val_ids = set(
                split_manifest_df.loc[split_manifest_df["split"] == "val", "player_id"].astype(str)
            )
            included_ids = {str(pid) for pid, rec in player_records.items() if not rec["excluded"]}
            train_ids &= included_ids
            val_ids &= included_ids
            if not train_ids or not val_ids:
                raise ValueError(
                    "Fold {} from {} has no train/val players after eligibility filtering.".format(
                        fold_name,
                        manifest_path,
                    )
                )
            print(
                "[sensitivity] train={} players, val={} players".format(
                    len(train_ids),
                    len(val_ids),
                )
            )

            labeled_df, label_thresholds = assign_zone_labels_cv(
                features_df,
                train_ids=train_ids,
                lower_q=args.lower_quantile,
                upper_q=args.upper_quantile,
                label_variant=args.label_variant,
            )

            summary_meta = {
                "task": args.task,
                "fold": fold_name,
                "label_variant": args.label_variant,
                "sensitivity_manifest_root": str(
                    Path(args.sensitivity_manifest_root).expanduser().resolve()
                ),
                "source_manifest_path": str(manifest_path),
                "n_total_players": n_total,
                "n_included": n_included,
                "n_excluded": len(excluded_records),
                "n_train_players": len(train_ids),
                "n_val_players": len(val_ids),
                "n_zones": args.n_zones,
                "min_completion": args.min_completion,
                "expected_task_subtasks": len(expected_subtasks),
                "min_unique_subtask_fraction": MIN_UNIQUE_SUBTASK_FRACTION,
                "min_unique_completed_subtasks": min_unique_completed_subtasks,
                "lower_quantile": args.lower_quantile,
                "upper_quantile": args.upper_quantile,
            }

            save_outputs(
                labeled_df=labeled_df,
                excluded_records=excluded_records,
                df_map=df_map,
                zone_centroids=zone_centroids,
                output_dir=fold_output_dir,
                summary_meta=summary_meta,
                zone_sequences=zone_sequences,
                task_name=args.task,
                label_variant=args.label_variant,
                player_split_manifest_df=split_manifest_df.copy(),
                label_thresholds=label_thresholds,
                player_records=player_records,
            )

        print("\n=== Done (sensitivity manifests reused) ===")
        print("  Label variant:  {}".format(args.label_variant))
        print("  Total players:  {}".format(n_total))
        print("  Included:       {}".format(n_included))
        print("  Excluded:       {}".format(len(excluded_records)))
        print("  Outputs:        {}/fold_*/".format(output_dir))

    elif args.n_folds > 1:
        # --- K-fold CV path ---
        included_ids = [pid for pid, rec in player_records.items() if not rec["excluded"]]
        fold_assignment = build_cv_folds(included_ids, n_folds=args.n_folds, seed=args.fold_seed)
        print(
            f"\n[cv] Running {args.n_folds}-fold CV "
            f"(fold_seed={args.fold_seed}, {len(included_ids)} eligible players)"
        )

        for fold_idx in range(args.n_folds):
            fold_name = "fold_{:02d}".format(fold_idx)
            fold_output_dir = output_dir / fold_name
            print(f"\n=== {fold_name} ===")

            val_ids = {pid for pid, f in fold_assignment.items() if f == fold_idx}
            train_ids = {pid for pid, f in fold_assignment.items() if f != fold_idx}
            print(f"[cv] train={len(train_ids)} players, val={len(val_ids)} players")

            labeled_df, label_thresholds = assign_zone_labels_cv(
                features_df,
                train_ids=train_ids,
                lower_q=args.lower_quantile,
                upper_q=args.upper_quantile,
                label_variant=args.label_variant,
            )

            manifest_rows = [
                {
                    "player_id": pid,
                    "split": "val" if fold_assignment[pid] == fold_idx else "train",
                    "fold": fold_name,
                }
                for pid in included_ids
            ]
            split_manifest_df = pd.DataFrame(manifest_rows)

            summary_meta = {
                "task": args.task,
                "fold": fold_name,
                "fold_seed": args.fold_seed,
                "n_folds": args.n_folds,
                "label_variant": args.label_variant,
                "n_total_players": n_total,
                "n_included": n_included,
                "n_excluded": len(excluded_records),
                "n_train_players": len(train_ids),
                "n_val_players": len(val_ids),
                "n_zones": args.n_zones,
                "min_completion": args.min_completion,
                "expected_task_subtasks": len(expected_subtasks),
                "min_unique_subtask_fraction": MIN_UNIQUE_SUBTASK_FRACTION,
                "min_unique_completed_subtasks": min_unique_completed_subtasks,
                "lower_quantile": args.lower_quantile,
                "upper_quantile": args.upper_quantile,
            }

            save_outputs(
                labeled_df=labeled_df,
                excluded_records=excluded_records,
                df_map=df_map,
                zone_centroids=zone_centroids,
                output_dir=fold_output_dir,
                summary_meta=summary_meta,
                zone_sequences=zone_sequences,
                task_name=args.task,
                label_variant=args.label_variant,
                player_split_manifest_df=split_manifest_df,
                label_thresholds=label_thresholds,
                player_records=player_records,
            )

        print(f"\n=== Done (K-fold CV, {args.n_folds} folds) ===")
        print(f"  Total players:  {n_total}")
        print(f"  Included:       {n_included}")
        print(f"  Excluded:       {len(excluded_records)}")
        print(f"  Outputs:        {output_dir}/fold_*/")

    else:
        # --- Single global labeling pass ---
        labeled_df = assign_zone_labels(
            features_df,
            lower_q=args.lower_quantile,
            upper_q=args.upper_quantile,
            label_variant=args.label_variant,
        )
        label_counts = labeled_df["label"].value_counts()

        summary_meta = {
            "task": args.task,
            "label_variant": args.label_variant,
            "n_total_players": n_total,
            "n_included": n_included,
            "n_excluded": len(excluded_records),
            "n_zones": args.n_zones,
            "min_completion": args.min_completion,
            "expected_task_subtasks": len(expected_subtasks),
            "min_unique_subtask_fraction": MIN_UNIQUE_SUBTASK_FRACTION,
            "min_unique_completed_subtasks": min_unique_completed_subtasks,
            "lower_quantile": args.lower_quantile,
            "upper_quantile": args.upper_quantile,
        }

        save_outputs(
            labeled_df=labeled_df,
            excluded_records=excluded_records,
            df_map=df_map,
            zone_centroids=zone_centroids,
            output_dir=output_dir,
            summary_meta=summary_meta,
            zone_sequences=zone_sequences,
            task_name=args.task,
            label_variant=args.label_variant,
            player_records=player_records,
        )

        label_counts_2 = labeled_df["label_2_class"].value_counts()
        label_counts_2c = labeled_df["label_2_class_clean"].value_counts()
        n_excluded_clean = (labeled_df["label_2_class_clean"] == "").sum()
        print("\n=== Done ===")
        print(f"  Total players:  {n_total}")
        print(f"  Included:       {n_included}")
        print(f"  Excluded:       {len(excluded_records)}")
        print(
            f"  3-class        → Zone Planner: {label_counts.get('Zone Planner', 0)}, "
            f"Mixed: {label_counts.get('Mixed', 0)}, "
            f"Zone Hopper: {label_counts.get('Zone Hopper', 0)}"
        )
        print(
            f"  2-class        → Zone Planner: {label_counts_2.get('Zone Planner', 0)}, "
            f"Zone Hopper: {label_counts_2.get('Zone Hopper', 0)}"
        )
        print(
            f"  2-class clean  → Zone Planner: {label_counts_2c.get('Zone Planner', 0)}, "
            f"Zone Hopper: {label_counts_2c.get('Zone Hopper', 0)}, "
            f"Excluded (Mixed): {n_excluded_clean}"
        )
        print(f"  Outputs:        {output_dir}/")


if __name__ == "__main__":
    main()
