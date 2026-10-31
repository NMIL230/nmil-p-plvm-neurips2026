"""Configuration helpers for the paired-task foraging and cleaning simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Dict, Tuple

from plvm.artifact_paths import SIMULATION_EPISODES_DIR


TASK_NAME_FORAGING = "foraging"
TASK_NAME_CLEANING = "cleaning"
LABEL_PLANNER = "Planner"
LABEL_HOPPER = "Hopper"
CLASS_ID_BY_LABEL = {
    LABEL_PLANNER: 0,
    LABEL_HOPPER: 1,
}

RELATIONSHIP_CONDITION_COMPLEMENTARY = "complementary"
RELATIONSHIP_CONDITION_REDUNDANT = "redundant"
RELATIONSHIP_CONDITION_WEAK_SHARED_STRUCTURE = "weak_shared_structure"
RELATIONSHIP_CONDITION_WEAK_STRUCTURE_ALIAS = "weak_structure"
SUPPORTED_RELATIONSHIP_CONDITIONS: Tuple[str, ...] = (
    RELATIONSHIP_CONDITION_COMPLEMENTARY,
    RELATIONSHIP_CONDITION_REDUNDANT,
    RELATIONSHIP_CONDITION_WEAK_SHARED_STRUCTURE,
)
RELATIONSHIP_CONDITION_ALIASES = {
    RELATIONSHIP_CONDITION_WEAK_STRUCTURE_ALIAS: RELATIONSHIP_CONDITION_WEAK_SHARED_STRUCTURE,
}
# Default simulator output directories, next to the complementary dataset that the
# trainers in simulation/train/ read by default.
_SIMULATION_OUTPUT_ROOT = SIMULATION_EPISODES_DIR.parent
RELATIONSHIP_CONDITION_DEFAULT_OUTPUT_DIRS = {
    RELATIONSHIP_CONDITION_COMPLEMENTARY: str(SIMULATION_EPISODES_DIR),
    RELATIONSHIP_CONDITION_REDUNDANT: str(_SIMULATION_OUTPUT_ROOT / "foraging_cleaning_redundant"),
    RELATIONSHIP_CONDITION_WEAK_SHARED_STRUCTURE: str(
        _SIMULATION_OUTPUT_ROOT / "foraging_cleaning_weak_structure"
    ),
}
RELATIONSHIP_CONDITION_PRESET_OVERRIDES = {
    RELATIONSHIP_CONDITION_COMPLEMENTARY: {},
    RELATIONSHIP_CONDITION_REDUNDANT: {
        "foraging_beta_P": 1.4,
        "foraging_beta_O": 1.6,
        "cleaning_beta_P": 1.4,
        "cleaning_beta_O": 1.6,
    },
    RELATIONSHIP_CONDITION_WEAK_SHARED_STRUCTURE: {
        "foraging_beta_P": 0.2,
        "foraging_beta_O": 1.5,
        "cleaning_beta_P": 1.0,
        "cleaning_beta_O": 0.35,
        "sigma_noise": 0.8,
    },
}


def normalize_relationship_condition(condition: str) -> str:
    """Return the canonical relationship-condition name for CLI and config use."""
    normalized = str(condition).strip()
    if normalized in RELATIONSHIP_CONDITION_ALIASES:
        return RELATIONSHIP_CONDITION_ALIASES[normalized]
    return normalized


def default_output_dir_for_condition(condition: str) -> str:
    """Return the default simulation output directory for one condition."""
    canonical_condition = normalize_relationship_condition(condition)
    if canonical_condition not in RELATIONSHIP_CONDITION_DEFAULT_OUTPUT_DIRS:
        raise ValueError(
            "relationship_condition must be one of {}.".format(
                list(SUPPORTED_RELATIONSHIP_CONDITIONS)
            )
        )
    return RELATIONSHIP_CONDITION_DEFAULT_OUTPUT_DIRS[canonical_condition]


@dataclass(frozen=True)
class SimulationConfig:
    """Container for all synthetic simulation parameters."""

    n_players: int = 10000
    seed: int = 123
    out_dir: str = str(SIMULATION_EPISODES_DIR)
    relationship_condition: str = RELATIONSHIP_CONDITION_COMPLEMENTARY
    experiment_tag: str = ""

    grid_size: int = 20
    foraging_targets: int = 32
    cleaning_targets: int = 48
    n_high_value_food: int = 6

    step_cost: float = 0.01
    beta_distance: float = 0.05
    sigma_noise: float = 0.5
    temperature: float = 1.0

    # High/low latent levels shared by the four subtypes (see agents.sample_players):
    # planner_P_mean / hopper_P_mean are the high / low P means, and
    # hopper_O_mean / planner_O_mean are the high / low O means.
    planner_P_mean: float = 0.8
    planner_O_mean: float = -0.8
    hopper_P_mean: float = -0.8
    hopper_O_mean: float = 0.8
    latent_sd: float = 0.35

    foraging_beta_P: float = 0.35
    foraging_beta_O: float = 2.8
    cleaning_beta_P: float = 1.8
    cleaning_beta_O: float = 0.6

    foraging_patch_count_min: int = 4
    foraging_patch_count_max: int = 6
    n_example_plots: int = 2

    foraging_low_value_reward: float = 1.0
    foraging_high_value_reward: float = 3.0
    cleaning_reward: float = 1.0
    cleaning_salience_low: float = 0.5
    cleaning_salience_medium: float = 1.0
    cleaning_salience_high: float = 2.0
    cleaning_salience_jitter: float = 0.15
    early_fraction: float = 0.25

    def validate(self) -> None:
        """Validate basic parameter consistency before simulation starts."""
        if self.n_players <= 0:
            raise ValueError("n_players must be positive.")
        canonical_condition = normalize_relationship_condition(self.relationship_condition)
        if canonical_condition not in SUPPORTED_RELATIONSHIP_CONDITIONS:
            raise ValueError(
                "relationship_condition must be one of {}.".format(
                    list(SUPPORTED_RELATIONSHIP_CONDITIONS)
                )
            )
        if self.grid_size < 14:
            raise ValueError("grid_size must be at least 14 for the default layouts.")
        if self.foraging_targets <= 0 or self.cleaning_targets <= 0:
            raise ValueError("Target counts must be positive.")
        if self.n_high_value_food <= 0 or self.n_high_value_food >= self.foraging_targets:
            raise ValueError("n_high_value_food must be between 1 and foraging_targets - 1.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if self.sigma_noise < 0:
            raise ValueError("sigma_noise must be non-negative.")
        if self.step_cost < 0:
            raise ValueError("step_cost must be non-negative.")
        if self.foraging_low_value_reward <= 0 or self.foraging_high_value_reward <= 0:
            raise ValueError("Foraging rewards must be positive.")
        if self.foraging_high_value_reward <= self.foraging_low_value_reward:
            raise ValueError("foraging_high_value_reward must exceed foraging_low_value_reward.")
        if self.beta_distance < 0:
            raise ValueError("beta_distance must be non-negative.")
        if self.foraging_patch_count_min <= 0:
            raise ValueError("foraging_patch_count_min must be positive.")
        if self.foraging_patch_count_max < self.foraging_patch_count_min:
            raise ValueError("foraging_patch_count_max must be >= foraging_patch_count_min.")
        if self.n_example_plots < 0:
            raise ValueError("n_example_plots must be non-negative.")
        if (
            min(
                self.cleaning_salience_low,
                self.cleaning_salience_medium,
                self.cleaning_salience_high,
            )
            <= 0
        ):
            raise ValueError("Cleaning salience levels must be positive.")
        if self.cleaning_salience_jitter < 0:
            raise ValueError("cleaning_salience_jitter must be non-negative.")
        if not 0 < self.early_fraction <= 1:
            raise ValueError("early_fraction must be in (0, 1].")

    def condition_overrides(self) -> Dict[str, float]:
        """Return simulator parameter overrides implied by the relationship condition."""
        canonical_condition = normalize_relationship_condition(self.relationship_condition)
        return dict(RELATIONSHIP_CONDITION_PRESET_OVERRIDES[canonical_condition])

    def resolve_condition_preset(self) -> "SimulationConfig":
        """Return a config copy with the relationship-condition preset applied."""
        canonical_condition = normalize_relationship_condition(self.relationship_condition)
        overrides = self.condition_overrides()
        update_kwargs = {"relationship_condition": canonical_condition}
        update_kwargs.update(overrides)
        return replace(self, **update_kwargs)

    def task_choice_params(self, task_name: str) -> Dict[str, float]:
        """Return task-specific target-choice loadings."""
        if task_name == TASK_NAME_FORAGING:
            return {
                "beta_P": float(self.foraging_beta_P),
                "beta_O": float(self.foraging_beta_O),
            }
        if task_name == TASK_NAME_CLEANING:
            return {
                "beta_P": float(self.cleaning_beta_P),
                "beta_O": float(self.cleaning_beta_O),
            }
        raise ValueError(f"Unsupported task_name: {task_name}")

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable configuration dictionary."""
        payload = asdict(self)
        payload["applied_condition_overrides"] = self.condition_overrides()
        payload["class_id_by_label"] = dict(CLASS_ID_BY_LABEL)
        payload["task_names"] = {
            "foraging": TASK_NAME_FORAGING,
            "cleaning": TASK_NAME_CLEANING,
        }
        return payload
