#!/usr/bin/env python3
"""Shared path helpers for PowerWash hierarchical training logs."""

import csv
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from plvm import artifact_paths


MODEL_PERFORMANCE_CSV_NAME = "model_performance.csv"
# Shared log roots; a log dir with one of these names is routed into a model-family subfolder.
SHARED_LOG_ROOT_NAMES = {"training_logs", "debug_logs"}
_MODEL_TYPE_TO_FAMILY_DIR = {
    "concat_logistic_probe": "concat_logistic_probe",
    "concat_mlp_probe": "concat_mlp_probe",
    "single_task_logistic_probe": "single_task_logistic_probe",
    "plvm": "plvm_models",
    "plvm_full_partial": "plvm_models",
    "plvm_both_partial": "plvm_models",
    "single_task": "single_task",
    "rf_baseline": "rf_baselines",
    "outcome_summary_baseline": "outcome_summary_baseline",
}
FIRESTATION_SENSITIVITY_FAMILY_DIRS = {
    "plvm_models": "plvm_models_firestation_sensitivity",
    "single_task": "single_task_firestation_sensitivity",
}
STORYBOOK_FAMILY_DIRS = {
    "plvm_models": "plvm_models_storybook",
    "single_task": "single_task_storybook",
    "rf_baselines": "rf_baselines_storybook",
    "concat_logistic_probe": "concat_logistic_probe_storybook",
}
KNOWN_FAMILY_DIRS = tuple(
    dict.fromkeys(
        list(_MODEL_TYPE_TO_FAMILY_DIR.values())
        + list(FIRESTATION_SENSITIVITY_FAMILY_DIRS.values())
        + list(STORYBOOK_FAMILY_DIRS.values())
    )
)


def normalize_powerwash_hierarchical_model_type(model_type: str) -> str:
    """Return a normalized PowerWash hierarchical model type token."""
    return str(model_type or "").strip().lower()


def powerwash_hierarchical_family_name(model_type: str) -> str:
    """Return the subfolder name used for one PowerWash hierarchical model type."""
    return _MODEL_TYPE_TO_FAMILY_DIR.get(
        normalize_powerwash_hierarchical_model_type(model_type),
        "",
    )


def default_powerwash_hierarchical_logs_root(
    project_root: Optional[Path] = None, run_mode: str = "run"
) -> Path:
    """Return the shared PowerWash hierarchical log root for the selected run mode.

    ``project_root`` lets a caller point the same relative layout at its own copy of
    the tree; without it the location comes from :mod:`plvm.artifact_paths`.
    """
    mode = str(run_mode or "run").strip().lower()
    default_root = (
        artifact_paths.POWERWASH_DEBUG_LOGS_DIR
        if mode == "debug"
        else artifact_paths.POWERWASH_TRAINING_LOGS_DIR
    )
    if project_root is None:
        return default_root
    return Path(project_root) / default_root.relative_to(artifact_paths.ARTIFACT_ROOT)


def resolve_powerwash_hierarchical_log_base(
    log_dir: Optional[Path],
    model_type: str,
    *,
    project_root: Optional[Path] = None,
    run_mode: str = "run",
) -> Path:
    """Route a shared log root to the appropriate model-family subdirectory."""
    if log_dir:
        base_dir = Path(log_dir)
    else:
        base_dir = default_powerwash_hierarchical_logs_root(project_root, run_mode=run_mode)

    family_dir = powerwash_hierarchical_family_name(model_type)
    if not family_dir:
        return base_dir
    if base_dir.name == family_dir:
        return base_dir
    if base_dir.name in SHARED_LOG_ROOT_NAMES:
        return base_dir / family_dir
    return base_dir


def resolve_powerwash_hierarchical_performance_csv_path(
    csv_path: Optional[Path],
    model_type: str,
    *,
    project_root: Optional[Path] = None,
    run_mode: str = "run",
) -> Path:
    """Route a shared model-performance CSV path to the family-specific CSV."""
    if csv_path:
        path = Path(csv_path)
        if path.name != MODEL_PERFORMANCE_CSV_NAME:
            return path
        if path.parent.name == powerwash_hierarchical_family_name(model_type):
            return path
        if path.parent.name in SHARED_LOG_ROOT_NAMES:
            return (
                resolve_powerwash_hierarchical_log_base(
                    path.parent,
                    model_type,
                    project_root=project_root,
                    run_mode=run_mode,
                )
                / MODEL_PERFORMANCE_CSV_NAME
            )
        return path

    return (
        resolve_powerwash_hierarchical_log_base(
            None,
            model_type,
            project_root=project_root,
            run_mode=run_mode,
        )
        / MODEL_PERFORMANCE_CSV_NAME
    )


def iter_powerwash_hierarchical_performance_csv_paths(
    log_source: Path,
    model_types: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Return all relevant family CSVs under one training-log source root."""
    source = Path(log_source)
    if source.is_file():
        return [source]

    family_names: Iterable[str]
    if model_types:
        family_names = tuple(
            dict.fromkeys(
                family_name
                for family_name in (
                    powerwash_hierarchical_family_name(model_type) for model_type in model_types
                )
                if family_name
            )
        )
    else:
        family_names = KNOWN_FAMILY_DIRS

    csv_paths: List[Path] = []
    if source.name in KNOWN_FAMILY_DIRS:
        candidate = source / MODEL_PERFORMANCE_CSV_NAME
        if candidate.exists():
            csv_paths.append(candidate)
        return csv_paths

    for family_name in family_names:
        candidate = source / family_name / MODEL_PERFORMANCE_CSV_NAME
        if candidate.exists():
            csv_paths.append(candidate)

    # A registry directly under the source root (runs not split into family folders).
    root_csv = source / MODEL_PERFORMANCE_CSV_NAME
    if root_csv.exists():
        csv_paths.append(root_csv)

    deduped_paths: List[Path] = []
    seen = set()
    for csv_path in csv_paths:
        resolved = csv_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped_paths.append(csv_path)
    return deduped_paths


def load_powerwash_hierarchical_performance_rows(
    log_source: Path,
    model_types: Optional[Sequence[str]] = None,
) -> List[dict]:
    """Load and deduplicate performance rows from family-specific CSVs."""
    rows: List[dict] = []
    seen_model_names = set()
    for csv_path in iter_powerwash_hierarchical_performance_csv_paths(
        log_source,
        model_types=model_types,
    ):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                model_name = str(row.get("model_name", "") or "").strip()
                if model_name and model_name in seen_model_names:
                    continue
                if model_name:
                    seen_model_names.add(model_name)
                row_copy = dict(row)
                row_copy["_source_csv"] = str(csv_path)
                rows.append(row_copy)
    return rows


def resolve_powerwash_hierarchical_run_dir(
    log_root: Path,
    run_name: str,
    model_type: str = "",
) -> Path:
    """Return the run directory for one saved PowerWash hierarchical model."""
    root = Path(log_root)
    if root.is_file():
        root = root.parent

    run_name = str(run_name or "").strip()
    if not run_name:
        raise ValueError("run_name must be non-empty.")

    candidate_paths: List[Path] = []
    if root.name in KNOWN_FAMILY_DIRS:
        candidate_paths.append(root / run_name)
    else:
        preferred_family = powerwash_hierarchical_family_name(model_type)
        if preferred_family:
            candidate_paths.append(root / preferred_family / run_name)
        for family_name in KNOWN_FAMILY_DIRS:
            candidate = root / family_name / run_name
            if candidate not in candidate_paths:
                candidate_paths.append(candidate)
        candidate_paths.append(root / run_name)

    existing_paths = [path for path in candidate_paths if path.is_dir()]
    if len(existing_paths) == 1:
        return existing_paths[0]
    if len(existing_paths) > 1:
        preferred_family = powerwash_hierarchical_family_name(model_type)
        preferred_path = root / preferred_family / run_name if preferred_family else None
        if preferred_path is not None and preferred_path in existing_paths:
            return preferred_path
        raise ValueError(
            "Multiple run directories matched '{}': {}".format(
                run_name,
                ", ".join(str(path) for path in existing_paths),
            )
        )

    raise FileNotFoundError(
        "Run directory not found for '{}'. Checked: {}".format(
            run_name,
            ", ".join(str(path) for path in candidate_paths),
        )
    )
