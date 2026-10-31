"""Player sampling and target-choice logic for the paired-task simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from simulation.config import (
    CLASS_ID_BY_LABEL,
    LABEL_HOPPER,
    LABEL_PLANNER,
    SimulationConfig,
    TASK_NAME_FORAGING,
)
from simulation.environments import Position, Target, TaskEnvironment


@dataclass(frozen=True)
class PlayerProfile:
    """Latent profile and ground-truth label for one synthetic player."""

    player_id: int
    latent_P: float
    latent_O: float
    label: str
    class_id: int


@dataclass(frozen=True)
class TargetChoice:
    """Result of one target-selection decision."""

    selected_target: Target
    shortest_distance: int
    score: float
    probability: float


def sample_players(config: SimulationConfig, rng: np.random.Generator) -> List[PlayerProfile]:
    """Sample balanced players from diagonal latent subtypes collapsed into two labels."""
    planner_count = config.n_players // 2
    hopper_count = config.n_players - planner_count

    players: List[PlayerProfile] = []
    next_player_id = 0

    planner_high_high_count = planner_count // 2
    planner_low_low_count = planner_count - planner_high_high_count
    hopper_high_low_count = hopper_count // 2
    hopper_low_high_count = hopper_count - hopper_high_low_count

    next_player_id = _append_player_group(
        players=players,
        start_player_id=next_player_id,
        count=planner_high_high_count,
        latent_P_mean=config.planner_P_mean,
        latent_O_mean=config.hopper_O_mean,
        latent_sd=config.latent_sd,
        label=LABEL_PLANNER,
        rng=rng,
    )
    next_player_id = _append_player_group(
        players=players,
        start_player_id=next_player_id,
        count=planner_low_low_count,
        latent_P_mean=config.hopper_P_mean,
        latent_O_mean=config.planner_O_mean,
        latent_sd=config.latent_sd,
        label=LABEL_PLANNER,
        rng=rng,
    )
    next_player_id = _append_player_group(
        players=players,
        start_player_id=next_player_id,
        count=hopper_high_low_count,
        latent_P_mean=config.planner_P_mean,
        latent_O_mean=config.planner_O_mean,
        latent_sd=config.latent_sd,
        label=LABEL_HOPPER,
        rng=rng,
    )
    _append_player_group(
        players=players,
        start_player_id=next_player_id,
        count=hopper_low_high_count,
        latent_P_mean=config.hopper_P_mean,
        latent_O_mean=config.hopper_O_mean,
        latent_sd=config.latent_sd,
        label=LABEL_HOPPER,
        rng=rng,
    )

    permutation = rng.permutation(len(players))
    return [players[int(index)] for index in permutation]


def choose_next_target(
    player: PlayerProfile,
    environment: TaskEnvironment,
    unfinished_targets: Sequence[Target],
    current_position: Position,
    current_patch: Optional[int],
    config: SimulationConfig,
    rng: np.random.Generator,
) -> TargetChoice:
    """Sample the next unfinished target from a softmax over the noisy choice scores."""
    task_params = config.task_choice_params(environment.task_name)
    distances = np.array(
        [
            environment.shortest_distances[(current_position, target.position)]
            for target in unfinished_targets
        ],
        dtype=float,
    )
    opportunistic_values = np.array(
        [
            target.reward if environment.task_name == TASK_NAME_FORAGING else target.salience
            for target in unfinished_targets
        ],
        dtype=float,
    )
    same_patch = np.array(
        [
            1.0 if current_patch is not None and target.patch_id == current_patch else 0.0
            for target in unfinished_targets
        ],
        dtype=float,
    )
    epsilon = rng.normal(0.0, config.sigma_noise, size=len(unfinished_targets))

    scores = (
        task_params["beta_P"] * player.latent_P * same_patch
        + task_params["beta_O"] * player.latent_O * opportunistic_values
        - config.beta_distance * distances
        + epsilon
    )
    logits = scores / config.temperature
    probabilities = _softmax(logits)
    selected_index = int(rng.choice(len(unfinished_targets), p=probabilities))

    return TargetChoice(
        selected_target=unfinished_targets[selected_index],
        shortest_distance=int(distances[selected_index]),
        score=float(scores[selected_index]),
        probability=float(probabilities[selected_index]),
    )


def select_example_players(
    players: Sequence[PlayerProfile],
    n_per_label: int,
) -> Tuple[List[PlayerProfile], List[PlayerProfile]]:
    """Choose illustrative Planner and Hopper players for trajectory plots."""
    planners = [player for player in players if player.label == LABEL_PLANNER]
    hoppers = [player for player in players if player.label == LABEL_HOPPER]

    planner_examples = _select_examples_from_quadrants(
        planners,
        quadrant_predicates=(
            lambda player: player.latent_P >= 0 and player.latent_O >= 0,
            lambda player: player.latent_P < 0 and player.latent_O < 0,
        ),
        n_examples=n_per_label,
    )
    hopper_examples = _select_examples_from_quadrants(
        hoppers,
        quadrant_predicates=(
            lambda player: player.latent_P >= 0 and player.latent_O < 0,
            lambda player: player.latent_P < 0 and player.latent_O >= 0,
        ),
        n_examples=n_per_label,
    )
    return planner_examples, hopper_examples


def _append_player_group(
    players: List[PlayerProfile],
    start_player_id: int,
    count: int,
    latent_P_mean: float,
    latent_O_mean: float,
    latent_sd: float,
    label: str,
    rng: np.random.Generator,
) -> int:
    """Append one latent subtype and return the next available player id."""
    class_id = int(CLASS_ID_BY_LABEL[label])
    for offset in range(count):
        players.append(
            PlayerProfile(
                player_id=start_player_id + offset,
                latent_P=float(rng.normal(latent_P_mean, latent_sd)),
                latent_O=float(rng.normal(latent_O_mean, latent_sd)),
                label=label,
                class_id=class_id,
            )
        )
    return start_player_id + count


def _select_examples_from_quadrants(
    players: Sequence[PlayerProfile],
    quadrant_predicates: Sequence[Callable[[PlayerProfile], bool]],
    n_examples: int,
) -> List[PlayerProfile]:
    """Select diverse examples by drawing from each label-specific latent quadrant."""
    if n_examples <= 0:
        return []

    quadrant_groups = [
        sorted(
            [player for player in players if predicate(player)],
            key=lambda player: (
                abs(player.latent_P * player.latent_O),
                abs(player.latent_P) + abs(player.latent_O),
            ),
            reverse=True,
        )
        for predicate in quadrant_predicates
    ]

    selected: List[PlayerProfile] = []
    selected_ids = set()
    group_index = 0
    while len(selected) < n_examples and any(quadrant_groups):
        group = quadrant_groups[group_index % len(quadrant_groups)]
        if group:
            candidate = group.pop(0)
            if candidate.player_id not in selected_ids:
                selected.append(candidate)
                selected_ids.add(candidate.player_id)
        group_index += 1

    if len(selected) >= n_examples:
        return selected[:n_examples]

    remaining_players = sorted(
        [player for player in players if player.player_id not in selected_ids],
        key=lambda player: (
            abs(player.latent_P * player.latent_O),
            abs(player.latent_P) + abs(player.latent_O),
        ),
        reverse=True,
    )
    selected.extend(remaining_players[: max(0, n_examples - len(selected))])
    return selected[:n_examples]


def _softmax(logits: np.ndarray) -> np.ndarray:
    centered = logits - np.max(logits)
    exp_values = np.exp(centered)
    return exp_values / np.sum(exp_values)
