#!/usr/bin/env python3
"""Shared target metadata for PowerWash model-training entry points."""

import sys
from pathlib import Path

try:
    from powerwash.data.derive_features_for_transformer import (
        TARGET_COLUMNS as DERIVED_TARGET_COLUMNS,
    )
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from powerwash.data.derive_features_for_transformer import (
        TARGET_COLUMNS as DERIVED_TARGET_COLUMNS,
    )

TARGET_AVAILABILITY_SUFFIX = "_available"


def _is_trainable_target(target_col: str) -> bool:
    """Return True when a derived target column is a trainable prediction label."""
    return not str(target_col).endswith(TARGET_AVAILABILITY_SUFFIX)


TARGET_CHOICES = [
    str(target_col) for target_col in DERIVED_TARGET_COLUMNS if _is_trainable_target(target_col)
]

BINARY_TARGETS = {"subtask_in_next_10s"}


def is_binary_target(target_col: str) -> bool:
    """Return True when the requested PowerWash target is binary."""
    return str(target_col) in BINARY_TARGETS
