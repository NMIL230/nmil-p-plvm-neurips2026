#!/usr/bin/env python3
"""
navigation_style_observed_sequences.py
======================================
Helper for ``table14_feature_ladder.py``: reconstructs, per player and observed task,
the event sequence the frozen PowerWash transformers see, with the subtask names the
processed feature CSVs drop.

The hand-crafted baselines must see the PLVM's observation window, which is not the
raw CSV truncated at a cutoff: feature extraction (``derive_features``) keeps only
first completions of each subtask, the processed elapsed axis is a pause-adjusted
clock that can differ from the raw one, and the PLVM keeps the prefix with
``time_elapsed_since_job_start <= cutoff``. So the rows and clock come from the
processed CSV and the subtask names from the raw attempt CSV, joined by position.
The join is verified by matching raw (x, z) to processed ``pos_x``/``pos_z`` row for
row; players that fail are dropped. Zones are not inferred from positions, since the
position at completion does not reliably identify the completed subtask.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import powerwash.data.derive_features_for_transformer as D  # noqa: E402
from plvm import artifact_paths  # noqa: E402

RAW_DIR = artifact_paths.PLAYER_SEQUENCES_DIR
PROC_DIR = artifact_paths.PLAYER_SEQUENCE_FEATURES_DIR
ELAPSED_COL = "time_elapsed_since_job_start"
POSITION_TOLERANCE = 1e-3


@dataclass
class ObservedSequence:
    """One player's PLVM-visible event sequence for one task."""

    player_id: str
    task: str
    elapsed: np.ndarray  # (n,) float, ascending (the PLVM's stable sort)
    subtask: np.ndarray  # (n,) normalized subtask names
    completion: np.ndarray  # (n,) float, forward-filled LevelProgressionAmount

    def prefix_len(self, cutoff_seconds: float) -> int:
        """Events visible at ``cutoff_seconds`` -- the PLVM's searchsorted rule."""
        return int(np.searchsorted(self.elapsed, float(cutoff_seconds), side="right"))


def _parse_xz(raw: object) -> Tuple[float, float]:
    """Parse a ``CurrentPosition`` cell into (x, z). NaNs when unparseable."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return (np.nan, np.nan)
    text = str(raw).strip().strip("()[]")
    if not text:
        return (np.nan, np.nan)
    parts = [p for p in text.replace(";", ",").split(",") if p.strip()]
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return (np.nan, np.nan)
    if len(values) >= 3:
        return (values[0], values[2])  # x, z; y is vertical
    if len(values) == 2:
        return (values[0], values[1])
    return (np.nan, np.nan)


def _raw_subtask_frame(raw_csv: Path) -> Optional[pd.DataFrame]:
    """Apply the feature-extraction row filtering to one raw attempt CSV."""
    try:
        df = pd.read_csv(raw_csv, low_memory=False)
    except Exception:
        return None
    if df.empty:
        return None
    prepared = D.sort_attempt_rows(D.ensure_required_columns(df.copy()))
    prepared = D.keep_only_subtask_rows(prepared)
    prepared = D.deduplicate_attempt_rows(prepared)
    if prepared.empty:
        return None
    return prepared.reset_index(drop=True)


def _processed_path(player_id: str, task: str) -> Optional[Path]:
    direct = PROC_DIR / task / f"{player_id}_{task}_attempt1.csv"
    if direct.exists():
        return direct
    matches = sorted((PROC_DIR / task).glob(f"{player_id}_{task}_attempt*.csv"))
    return matches[0] if matches else None


def load_observed_sequence(player_id: str, task: str) -> Optional[ObservedSequence]:
    """Rebuild one player's PLVM-visible sequence, or None if unverifiable."""
    proc_csv = _processed_path(player_id, task)
    if proc_csv is None:
        return None
    raw_csv = RAW_DIR / task / proc_csv.name
    if not raw_csv.exists():
        return None

    try:
        ref = pd.read_csv(proc_csv, usecols=["pos_x", "pos_z", ELAPSED_COL])
    except Exception:
        return None
    prepared = _raw_subtask_frame(raw_csv)
    if prepared is None or len(prepared) != len(ref):
        return None

    ref_pos = ref[["pos_x", "pos_z"]].to_numpy(dtype=float)
    raw_pos = np.array([_parse_xz(v) for v in prepared.get("CurrentPosition", [])], dtype=float)
    if raw_pos.shape != ref_pos.shape:
        return None
    if not np.allclose(
        np.nan_to_num(raw_pos, nan=-9e9), np.nan_to_num(ref_pos, nan=-9e9), atol=POSITION_TOLERANCE
    ):
        return None

    # Authoritative clock from the processed file; PLVM applies a stable sort.
    elapsed = pd.to_numeric(ref[ELAPSED_COL], errors="coerce").to_numpy(dtype=float)
    order = np.argsort(elapsed, kind="stable")

    subtask = (
        prepared["LastSubtaskCompleted"]
        .map(D.normalize_subtask_label)
        .fillna("")
        .astype(str)
        .str.strip()
        .to_numpy()
    )
    completion = (
        pd.to_numeric(
            prepared.get("LevelProgressionAmount", pd.Series(index=prepared.index, dtype=float)),
            errors="coerce",
        )
        .ffill()
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    return ObservedSequence(
        player_id=player_id,
        task=task,
        elapsed=elapsed[order],
        subtask=subtask[order],
        completion=completion[order],
    )


def load_task_sequences(
    task: str,
    player_ids: Optional[Iterable[str]] = None,
    progress_every: int = 1000,
) -> Tuple[Dict[str, ObservedSequence], dict]:
    """Load every verifiable observed sequence for one task."""
    # one entry per player: a player with several attempt files is still one candidate
    available = sorted(
        {p.stem.split(f"_{task}_")[0] for p in (PROC_DIR / task).glob(f"*_{task}_attempt*.csv")}
    )
    wanted = set(str(p) for p in player_ids) if player_ids is not None else None
    targets = [p for p in available if wanted is None or p in wanted]

    sequences: Dict[str, ObservedSequence] = {}
    dropped = []
    for i, pid in enumerate(targets, 1):
        seq = load_observed_sequence(pid, task)
        if seq is None:
            dropped.append(pid)
        else:
            sequences[pid] = seq
        if progress_every and i % progress_every == 0:
            print(f"  [{task}] {i}/{len(targets)} scanned, {len(dropped)} unverified", flush=True)

    stats = {
        "task": task,
        "n_candidates": len(targets),
        "n_verified": len(sequences),
        "n_unverified": len(dropped),
        "pct_verified": 100.0 * len(sequences) / max(len(targets), 1),
        "dropped_player_ids": dropped,
    }
    return sequences, stats
