"""Single source of truth for every artifact location used by this code base.

Artifacts (data, simulated episodes, trained models, training logs) are not tracked
in git; every module asks this module for their paths.  ``REPO_ROOT`` is where the
code lives (paper outputs: see :mod:`paper.output_paths`).  ``ARTIFACT_ROOT``
defaults to ``<repo>/artifacts`` and can be relocated with ``PLVM_ARTIFACT_ROOT``::

    PLVM_ARTIFACT_ROOT=/data/plvm-artifacts python paper/make_paper.py

The module is stdlib-only so that the standalone data-pipeline scripts can import it.
"""

import os
from pathlib import Path
from typing import Union


#: Repository root (the parent of the ``plvm`` package).
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Environment variable that relocates the artifact tree.
ARTIFACT_ROOT_ENV_VAR = "PLVM_ARTIFACT_ROOT"


#: Directory holding the artifact tree inside the repository checkout.
DEFAULT_ARTIFACT_DIRNAME = "artifacts"


def _resolve_artifact_root() -> Path:
    """Return the artifact root: ``$PLVM_ARTIFACT_ROOT`` if set, else ``<repo>/artifacts``."""
    override = os.environ.get(ARTIFACT_ROOT_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return REPO_ROOT / DEFAULT_ARTIFACT_DIRNAME


#: Root of the artifact tree: ``$PLVM_ARTIFACT_ROOT`` if set, else ``REPO_ROOT / "artifacts"``.
ARTIFACT_ROOT = _resolve_artifact_root()


# ---------------------------------------------------------------------------
# PowerWash study: raw and derived data
# ---------------------------------------------------------------------------

POWERWASH_ROOT = ARTIFACT_ROOT / "powerwash"
POWERWASH_DATA_DIR = POWERWASH_ROOT / "data"

#: Per-player raw event windows produced by ``generate_simplified_data.py``.
SIMPLIFIED_DATA_DIR = POWERWASH_DATA_DIR / "simplified_data"

#: Per-attempt sequences with the 300 s implicit-gap correction applied.
PLAYER_SEQUENCES_DIR = POWERWASH_DATA_DIR / "player_sequences"

#: Per-attempt sequences extracted with the implicit-gap correction disabled.
PLAYER_SEQUENCES_NO_GAP_CORRECTION_DIR = POWERWASH_DATA_DIR / "player_sequences_no_gap_correction"

#: Transformer input features derived from :data:`PLAYER_SEQUENCES_DIR`.
PLAYER_SEQUENCE_FEATURES_DIR = POWERWASH_DATA_DIR / "player_sequence_features"

#: Second copy of the uncorrected sequences, stored inside the feature root.
UNCORRECTED_SEQUENCES_DIR = PLAYER_SEQUENCE_FEATURES_DIR / "player_sequencesy"

#: Fallback inputs for feature derivation, most preferred first.
DEFAULT_FALLBACK_SEQUENCE_DIRS = (
    PLAYER_SEQUENCES_NO_GAP_CORRECTION_DIR,
    UNCORRECTED_SEQUENCES_DIR,
)

#: Encoder feature schema stored alongside the derived features.
TRANSFORMER_FEATURE_SCHEMA_JSON = PLAYER_SEQUENCE_FEATURES_DIR / "transformer_feature_schema.json"

#: Encoder embedding cache for the task-specific PowerWash models.
POWERWASH_TASK_SPECIFIC_CACHE_DIR = (
    POWERWASH_DATA_DIR / "model_training" / "task_specific" / "cache"
)


# ---------------------------------------------------------------------------
# PowerWash study: derived outputs (labels, folds, navigation-style scores)
# ---------------------------------------------------------------------------

POWERWASH_OUTPUTS_DIR = POWERWASH_ROOT / "outputs"

#: Fire Station zone-planning labels and cross-validation folds.
FIRESTATION_LABELS_DIR = POWERWASH_OUTPUTS_DIR / "firestation_zone_labels"

#: Fire Station label variants, each recomputed with one label-formula component left out.
FIRESTATION_SENSITIVITY_LABELS_DIR = POWERWASH_OUTPUTS_DIR / "firestation_zone_labels_sensitivity"

#: Back Garden zone-planning labels.
BACK_GARDEN_LABELS_DIR = POWERWASH_OUTPUTS_DIR / "back_garden_zone_labels"

#: Bungalow 3-class zone-planning labels (input of the navigation-style scores).
BUNGALOW_LABELS_DIR = POWERWASH_OUTPUTS_DIR / "bungalow_zone_labels"

#: Back Garden labels of the full cohort (one row per player).
BACK_GARDEN_LABELS_CSV = BACK_GARDEN_LABELS_DIR / "back_garden_zone_labels.csv"

#: Per-player navigation-style scores and their companion outputs.
NAVIGATION_STYLE_DIR = POWERWASH_OUTPUTS_DIR / "navigation_style"

#: Per-player navigation-style scores over the three labelled tasks.
NAVSTYLE_PER_PLAYER_SCORES_CSV = NAVIGATION_STYLE_DIR / "navstyle_per_player_scores_3tasks.csv"

POWERWASH_ANALYSIS_DIR = POWERWASH_ROOT / "analysis"
POWERWASH_ANALYSIS_FIGURES_DIR = POWERWASH_ANALYSIS_DIR / "figures"
POWERWASH_ANALYSIS_TABLES_DIR = POWERWASH_ANALYSIS_DIR / "tables"

#: Canonical subtask → task-map point mapping used by the trace figures.
SUBTASK_TASK_MAP_POINTS_CSV = POWERWASH_ANALYSIS_TABLES_DIR / "subtask_task_map_points.csv"

#: Output root of the PowerWash model-comparison figures.
POWERWASH_MODEL_COMPARISON_FIGURES_DIR = POWERWASH_ANALYSIS_FIGURES_DIR / "model_comparisons"


def zone_labels_dir(artifact_prefix: str) -> Path:
    """Return the zone-label directory of one task, e.g. ``"firestation"``.

    The prefix is the artifact prefix used by
    ``create_firestation_zone_planning_labels.py`` (``firestation``,
    ``back_garden``, ``bungalow``, ``storybook_house``, ...).
    """
    return POWERWASH_OUTPUTS_DIR / "{}_zone_labels".format(artifact_prefix)


# ---------------------------------------------------------------------------
# Simulation (MiniGrid foraging/cleaning)
# ---------------------------------------------------------------------------

SIMULATION_ROOT = ARTIFACT_ROOT / "simulation"

#: Simulated episodes written by ``simulate.py``.
SIMULATION_EPISODES_DIR = SIMULATION_ROOT / "outputs" / "simulations" / "foraging_cleaning"


# ---------------------------------------------------------------------------
# Trained models and training logs
# ---------------------------------------------------------------------------

MODELS_DIR = ARTIFACT_ROOT / "models"

POWERWASH_MODELS_DIR = MODELS_DIR / "powerwash"

#: Task-specific PowerWash encoders (one subtree per task).
POWERWASH_TASK_SPECIFIC_MODELS_DIR = POWERWASH_MODELS_DIR / "task_specific"

#: Spec listing the task-specific encoder of each task; the default for the PowerWash PLVM runs.
POWERWASH_TASK_SPECIFIC_SPEC_JSON = POWERWASH_MODELS_DIR / "pw_task_specific_next_subtask.json"

POWERWASH_HIERARCHICAL_DIR = POWERWASH_MODELS_DIR / "hierarchical"

#: Root holding one subdirectory per PowerWash training family.
POWERWASH_TRAINING_LOGS_DIR = POWERWASH_HIERARCHICAL_DIR / "training_logs"

#: Same, for runs started in debug mode.
POWERWASH_DEBUG_LOGS_DIR = POWERWASH_HIERARCHICAL_DIR / "debug_logs"

#: Frozen-encoder embedding cache shared by the PowerWash PLVM runs.
POWERWASH_EMBEDDING_CACHE_DIR = POWERWASH_HIERARCHICAL_DIR / "embedding_cache"

SIMULATION_MODELS_DIR = MODELS_DIR / "simulations" / "minigrid_foraging_cleaning"

#: Task-specific MiniGrid encoders (one subtree per task).
SIMULATION_TASK_SPECIFIC_MODELS_DIR = SIMULATION_MODELS_DIR / "task_specific"

#: Spec listing the task-specific encoder of each MiniGrid task; the default for the MiniGrid PLVM runs.
SIMULATION_TASK_SPECIFIC_SPEC_JSON = (
    SIMULATION_MODELS_DIR / "pw_task_specific_models_best_256_new.json"
)


def powerwash_family_logs_dir(family: str) -> Path:
    """Return the PowerWash training-log directory of one run family."""
    return POWERWASH_TRAINING_LOGS_DIR / family


def simulation_family_logs_dir(family: str) -> Path:
    """Return the MiniGrid training-log directory of one run family.

    MiniGrid families keep their runs one level deeper than the PowerWash ones:
    ``models/simulations/minigrid_foraging_cleaning/<family>/training_logs/``.
    """
    return SIMULATION_MODELS_DIR / family / "training_logs"


# ---------------------------------------------------------------------------
# Aggregated training analyses (per-run metrics collected across run families)
# ---------------------------------------------------------------------------

MODEL_TRAINING_ANALYSIS_DIR = ARTIFACT_ROOT / "analysis" / "model_training_analysis"

#: Aggregated PowerWash PLVM performance summaries.
POWERWASH_PERFORMANCE_ANALYSIS_DIR = (
    MODEL_TRAINING_ANALYSIS_DIR / "powerwash_hierarchical_performance"
)

#: Aggregated MiniGrid PLVM performance summaries.
SIMULATION_PERFORMANCE_ANALYSIS_DIR = (
    MODEL_TRAINING_ANALYSIS_DIR / "minigrid_foraging_cleaning_performance"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: Top-level directory names at which a recorded path is cut and reattached to
#: :data:`ARTIFACT_ROOT`.  ``powerwash_study`` and ``simulations`` occur in paths
#: stored in the hyperparameter files of the runs in the artifact bundle.
_RECORDED_PATH_ANCHORS = (
    "models",
    "powerwash",
    "simulation",
    "analysis",
    "powerwash_study",
    "simulations",
)


def display_path(path: Union[str, Path]) -> str:
    """Return ``path`` relative to :data:`REPO_ROOT` or :data:`ARTIFACT_ROOT` when it
    lies under one of them (tried in that order), else unchanged."""
    resolved = Path(path)
    for root in (REPO_ROOT, ARTIFACT_ROOT):
        try:
            return str(resolved.relative_to(root))
        except ValueError:
            continue
    return str(resolved)


def rebase_recorded_artifact_path(path: Union[str, Path]) -> Path:
    """Rebase a path recorded at training time onto the current artifact root.

    When the recorded path does not exist, the part starting at a known top-level
    artifact directory is reattached to :data:`ARTIFACT_ROOT`.  The input is
    returned unchanged if it exists, has no anchor, or the rebased path is missing
    too, so callers keep their usual error messages.
    """
    candidate = Path(path)
    if candidate.exists():
        return candidate

    parts = candidate.parts
    # Walk from the right so the deepest anchor wins; a bundle path such as
    # /data/bundle/models/... must not be rebased on its own "models" prefix.
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] in _RECORDED_PATH_ANCHORS:
            rebased = ARTIFACT_ROOT.joinpath(*parts[index:])
            if rebased.exists():
                return rebased
    return candidate
