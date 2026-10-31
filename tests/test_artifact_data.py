"""Checks that the artifact tree is present, complete and loadable.

These tests run against the real artifact tree (``artifacts/`` or ``$PLVM_ARTIFACT_ROOT``) and are
skipped when it is absent, so ``pytest tests`` still passes on a fresh clone.

Two kinds of data are checked:

* Data in the artifact bundle, including everything ``paper/make_paper.py`` reads (labels and
  folds, task encoders, training-run logs, navigation-style scores, the subtask map, MiniGrid
  runs). These must be present, and they are loaded with the same project loaders the pipeline
  uses.
* Data that is not in the artifact bundle and is only needed to rerun the pipeline
  (simplified raw data, the full sequence and feature sets, the label variants used only for
  training, the embedding cache, MiniGrid episodes). A missing item only raises a warning; a
  present one is sanity-checked.
"""

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from plvm import artifact_paths as ap
from powerwash.train.powerwash_plvm_data_utils import (
    load_player_label_lookup,
    load_task_model_spec,
    resolve_powerwash_player_label_class_names,
)

pytestmark = pytest.mark.skipif(
    not ap.ARTIFACT_ROOT.is_dir(),
    reason="artifact tree not found at {} (set PLVM_ARTIFACT_ROOT)".format(ap.ARTIFACT_ROOT),
)

FOLDS = ["fold_{:02d}".format(i) for i in range(10)]
LABEL_FILES = {
    "player_label_2_class_clean_zonehopping": "firestation_zone_labels_2_class_clean.csv",
    "player_label_3_class_zonehopping": "firestation_zone_labels_3_class.csv",
}
# Training-run families that make_paper reads (one folder each under training_logs/).
RELEASED_FAMILIES = [
    "plvm_models",
    "single_task",
    "concat_logistic_probe",
    "concat_mlp_probe",
    "single_task_logistic_probe",
    "rf_baselines",
    "plvm_models_storybook",
    "single_task_storybook",
    "concat_logistic_probe_storybook",
    "rf_baselines_storybook",
    "plvm_models_firestation_sensitivity",
    "single_task_firestation_sensitivity",
]
# Families trained by iteration, which use early stopping.
EARLY_STOPPING_FAMILIES = ("plvm_models", "single_task", "concat_mlp_probe")


def _optional(path: Path, purpose: str) -> bool:
    """Warn (do not fail) when data outside the artifact bundle is missing."""
    if Path(path).exists():
        return True
    warnings.warn("{} not found ({}); it is not in the artifact bundle.".format(path, purpose))
    return False


def _runs(family_dir: Path):
    """Run folders of one training family (hidden folders such as .ipynb_checkpoints excluded)."""
    return sorted(p for p in family_dir.iterdir() if p.is_dir() and not p.name.startswith("."))


# ---------------------------------------------------------------------------------------------
# Data in the artifact bundle
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fold", FOLDS)
def test_fire_station_labels_and_split_load(fold):
    """Each fold's labels load with the pipeline loader and every labelled player has a split."""
    fold_dir = ap.FIRESTATION_LABELS_DIR / fold
    manifest = pd.read_csv(fold_dir / "player_split_manifest.csv", dtype={"player_id": str})
    assert {"player_id", "split", "fold"} <= set(manifest.columns)
    assert set(manifest["split"]) <= {"train", "val"}
    assert manifest["player_id"].is_unique
    val_fraction = (manifest["split"] == "val").mean()
    assert 0.05 < val_fraction < 0.15, val_fraction

    split_players = set(manifest["player_id"])
    for target_mode, filename in LABEL_FILES.items():
        labels = load_player_label_lookup(
            fold_dir / filename,
            allowed_labels=resolve_powerwash_player_label_class_names(target_mode),
        )
        assert labels, "no labelled players in {}".format(filename)
        assert set(labels) <= split_players, "labelled players missing from the split manifest"

    thresholds = json.loads((fold_dir / "label_thresholds.json").read_text())
    assert thresholds["lower_threshold"] < thresholds["upper_threshold"]


def test_fire_station_three_class_proportions():
    """The 3-class label splits players about 35 / 30 / 35 (Planner / Mixed / Hopper)."""
    labels = pd.read_csv(
        ap.FIRESTATION_LABELS_DIR / "fold_00" / "firestation_zone_labels_3_class.csv"
    )
    shares = labels["label"].value_counts(normalize=True)
    assert abs(shares["Zone Planner"] - 0.35) < 0.03
    assert abs(shares["Mixed"] - 0.30) < 0.03
    assert abs(shares["Zone Hopper"] - 0.35) < 0.03


def test_fire_station_figure_inputs():
    """Zone map, centroids and the player-record cache used by Figures 3a-3c are present."""
    fold_dir = ap.FIRESTATION_LABELS_DIR / "fold_00"
    zone_map = pd.read_csv(fold_dir / "firestation_subtask_zone_map.csv")
    centroids = pd.read_csv(fold_dir / "firestation_zone_centroids.csv")
    assert {"subtask_name", "x", "z", "zone_id"} <= set(zone_map.columns)
    assert zone_map["subtask_name"].is_unique
    assert len(centroids) == zone_map["zone_id"].nunique()
    assert (ap.FIRESTATION_LABELS_DIR / ".cache" / "firestation_player_records.pkl").is_file()


def test_task_encoders_resolve():
    """The encoder spec names one run per observed task, each with weights and a feature manifest."""
    spec = load_task_model_spec(ap.POWERWASH_TASK_SPECIFIC_SPEC_JSON, ap.ARTIFACT_ROOT)
    assert set(spec) == {"back_garden", "bungalow"}
    for task_name, entry in spec.items():
        run_dir = Path(entry["model_log_dir"])
        assert run_dir.is_dir(), "encoder run for {} not found: {}".format(task_name, run_dir)
        manifest = json.loads((run_dir / "feature_manifest.json").read_text())
        assert manifest["feature_columns"], "empty feature manifest for {}".format(task_name)
        assert list(run_dir.glob("*_model.pth")), "no weights for {}".format(task_name)


@pytest.mark.parametrize("family", RELEASED_FAMILIES)
def test_training_family_logs(family):
    """Each run has a registry row and validation metrics, and used attempt_policy=first.

    Neural runs trained by iteration also used early stopping.
    """
    family_dir = ap.POWERWASH_TRAINING_LOGS_DIR / family
    runs = _runs(family_dir)
    registry = pd.read_csv(family_dir / "model_performance.csv")
    assert {r.name for r in runs} == set(registry["model_name"].astype(str))
    folds = set()
    for run in runs:
        metrics = run / "plots/val/cutoff_metrics/metrics_by_cutoff.csv"
        assert metrics.is_file(), "missing validation metrics for {}".format(run.name)
        for hp_path in run.glob("*_hyperparameters.json"):
            hp = json.loads(hp_path.read_text())
            assert hp.get("attempt_policy", "first") == "first", run.name
            if family.startswith(EARLY_STOPPING_FAMILIES) and "logistic" not in family:
                assert hp.get("use_early_stopping"), "{} trained without early stopping".format(
                    run.name
                )
            folds.add(Path(str(hp.get("player_split_manifest", ""))).parent.name)
    if folds:
        assert set(FOLDS) <= folds, "{} lacks folds {}".format(family, sorted(set(FOLDS) - folds))


def test_navigation_style_inputs():
    """App D.6 inputs: per-player scores and the Back Garden / Bungalow labels they come from."""
    scores = pd.read_csv(ap.NAVSTYLE_PER_PLAYER_SCORES_CSV, dtype={"player_id": str})
    for task in ("back_garden", "bungalow", "fire_station"):
        assert "zone_hopping_score_{}".format(task) in scores.columns
    assert scores["player_id"].is_unique
    assert ap.BACK_GARDEN_LABELS_CSV.is_file()
    assert (ap.BUNGALOW_LABELS_DIR / "bungalow_zone_labels_3_class.csv").is_file()


def test_subtask_map_points():
    """Canonical subtask map: every study task, one row per subtask, finite coordinates."""
    points = pd.read_csv(ap.SUBTASK_TASK_MAP_POINTS_CSV)
    assert {"back_garden", "bungalow", "fire_station"} <= set(points["task"])
    assert not points.duplicated(["task", "subtask_name"]).any()
    assert points[["x", "z"]].notna().all().all()


def test_minigrid_runs_present():
    """MiniGrid training runs read by Figure 7 and Table 9."""
    assert ap.SIMULATION_MODELS_DIR.is_dir()
    assert any(ap.SIMULATION_MODELS_DIR.iterdir())


# ---------------------------------------------------------------------------------------------
# Data outside the artifact bundle: warn when missing
# ---------------------------------------------------------------------------------------------


def test_simplified_data_if_present():
    if not _optional(
        ap.SIMPLIFIED_DATA_DIR, "rebuilt from the OSF download by generate_simplified_data.py"
    ):
        return
    assert len(list(ap.SIMPLIFIED_DATA_DIR.glob("raw_*.csv"))) == 16


@pytest.mark.parametrize("task", ["back_garden", "bungalow", "fire_station", "storybook_house"])
def test_full_sequences_and_features_if_present(task):
    if _optional(
        ap.PLAYER_SEQUENCES_DIR / task,
        "per-attempt sequences, rebuilt by extract_player_sequences.py",
    ):
        assert list((ap.PLAYER_SEQUENCES_DIR / task).glob("*_attempt*.csv"))
    if _optional(
        ap.PLAYER_SEQUENCE_FEATURES_DIR / task,
        "encoder inputs, rebuilt by derive_features_for_transformer.py",
    ):
        assert list((ap.PLAYER_SEQUENCE_FEATURES_DIR / task).glob("*_attempt*.csv"))


def test_training_only_labels_if_present():
    """Label variants used only to train the Table 11 and Table 13 runs."""
    for label_dir, purpose in (
        (ap.FIRESTATION_SENSITIVITY_LABELS_DIR, "label-sensitivity labels for training"),
        (ap.zone_labels_dir("storybook_house"), "Storybook House labels for training"),
    ):
        if _optional(label_dir, purpose):
            assert list(label_dir.rglob("player_split_manifest.csv"))


def test_feature_schema_if_present():
    if _optional(ap.TRANSFORMER_FEATURE_SCHEMA_JSON, "needed only to retrain the task encoders"):
        assert json.loads(ap.TRANSFORMER_FEATURE_SCHEMA_JSON.read_text())


def test_embedding_cache_and_episodes_if_present():
    _optional(ap.POWERWASH_EMBEDDING_CACHE_DIR, "derived cache, recomputed from the encoders")
    _optional(
        ap.SIMULATION_EPISODES_DIR, "MiniGrid episodes, regenerated by simulation/simulate.py"
    )
