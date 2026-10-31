"""Run the paired-task synthetic grid-world simulation.

Each simulated player has two latent traits, P (patch persistence) and O (opportunism),
drawn around the means of their class (Planner or Hopper). In each task the player
repeatedly picks the next unfinished target with a softmax over

    beta_P * P * same_patch + beta_O * O * value - beta_distance * distance + noise,

where ``value`` is the food reward (foraging) or the dirt salience (cleaning). The
``--relationship_condition`` preset sets the task loadings (and the noise for
``weak_shared_structure``), overriding the corresponding CLI values; the complementary
preset keeps the CLI defaults.

Outputs go to ``--out_dir``, by default
``<artifact root>/simulation/outputs/simulations/foraging_cleaning[_<condition>]``, which
the trainers in ``simulation/train/`` read by default.  Run from the repository root
with ``python -m simulation.simulate``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from simulation.agents import (
    PlayerProfile,
    choose_next_target,
    sample_players,
    select_example_players,
)
from simulation.config import (
    RELATIONSHIP_CONDITION_ALIASES,
    RELATIONSHIP_CONDITION_COMPLEMENTARY,
    SUPPORTED_RELATIONSHIP_CONDITIONS,
    SimulationConfig,
    TASK_NAME_CLEANING,
    TASK_NAME_FORAGING,
    default_output_dir_for_condition,
    normalize_relationship_condition,
)
from simulation.environments import (
    Position,
    Target,
    TaskEnvironment,
    build_cleaning_environment,
    build_foraging_environment,
)
from simulation.logging_utils import (
    EVENT_COLUMNS,
    SUMMARY_COLUMNS,
    build_compact_sequence_record,
    build_sanity_summary,
    compute_task_summary,
    environments_to_metadata,
    players_to_frame,
    save_json,
    targets_to_frame,
    write_jsonl_record,
)
from simulation.visualization import (
    plot_behavior_summaries,
    plot_environment,
    plot_latent_distribution,
    plot_planner_vs_hopper,
    plot_trajectory,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the paired-task simulator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_players", type=int, default=10000, help="Number of simulated players.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help=(
            "Output directory (default: the condition's directory under "
            "<artifact root>/simulation/outputs/simulations/)."
        ),
    )
    parser.add_argument(
        "--relationship_condition",
        type=str,
        default=RELATIONSHIP_CONDITION_COMPLEMENTARY,
        help="Controls how strongly the two tasks share informative latent structure.",
    )
    parser.add_argument(
        "--experiment_tag",
        type=str,
        default="",
        help="Optional tag saved into manifests for one condition-specific simulation run.",
    )
    parser.add_argument("--grid_size", type=int, default=20, help="Grid side length (>= 14).")
    parser.add_argument(
        "--foraging_targets", type=int, default=32, help="Number of food items (foraging)."
    )
    parser.add_argument(
        "--cleaning_targets", type=int, default=48, help="Number of dirt patches (cleaning)."
    )
    parser.add_argument(
        "--n_high_value_food", type=int, default=6, help="Number of high-value food items."
    )
    parser.add_argument("--step_cost", type=float, default=0.01, help="Cost per grid step.")
    parser.add_argument(
        "--beta_distance", type=float, default=0.05, help="Distance penalty in the choice score."
    )
    parser.add_argument(
        "--sigma_noise", type=float, default=0.5, help="SD of the Gaussian choice-score noise."
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Softmax temperature of target choice."
    )
    parser.add_argument(
        "--planner_P_mean", type=float, default=0.8, help="Mean of latent P in the high-P subtypes."
    )
    parser.add_argument(
        "--planner_O_mean", type=float, default=-0.8, help="Mean of latent O in the low-O subtypes."
    )
    parser.add_argument(
        "--hopper_P_mean", type=float, default=-0.8, help="Mean of latent P in the low-P subtypes."
    )
    parser.add_argument(
        "--hopper_O_mean", type=float, default=0.8, help="Mean of latent O in the high-O subtypes."
    )
    parser.add_argument(
        "--latent_sd", type=float, default=0.35, help="Within-class SD of the latent traits."
    )
    parser.add_argument(
        "--foraging_beta_P", type=float, default=0.35, help="Foraging loading of latent P."
    )
    parser.add_argument(
        "--foraging_beta_O", type=float, default=2.8, help="Foraging loading of latent O."
    )
    parser.add_argument(
        "--foraging_low_value_reward", type=float, default=1.0, help="Reward of ordinary food."
    )
    parser.add_argument(
        "--foraging_high_value_reward", type=float, default=3.0, help="Reward of high-value food."
    )
    parser.add_argument(
        "--cleaning_beta_P", type=float, default=1.8, help="Cleaning loading of latent P."
    )
    parser.add_argument(
        "--cleaning_beta_O", type=float, default=0.6, help="Cleaning loading of latent O."
    )
    parser.add_argument(
        "--cleaning_salience_low", type=float, default=0.5, help="Low dirt salience level."
    )
    parser.add_argument(
        "--cleaning_salience_medium", type=float, default=1.0, help="Medium dirt salience level."
    )
    parser.add_argument(
        "--cleaning_salience_high", type=float, default=2.0, help="High dirt salience level."
    )
    parser.add_argument(
        "--cleaning_salience_jitter",
        type=float,
        default=0.15,
        help="Random jitter added to dirt salience.",
    )
    parser.add_argument(
        "--n_example_plots", type=int, default=2, help="Number of example trajectory plots."
    )
    return parser.parse_args(argv)


def build_config_from_args(args: argparse.Namespace) -> SimulationConfig:
    """Build a validated config object from parsed CLI args."""
    canonical_condition = normalize_relationship_condition(args.relationship_condition)
    supported_condition_names = set(SUPPORTED_RELATIONSHIP_CONDITIONS) | set(
        RELATIONSHIP_CONDITION_ALIASES.keys()
    )
    if canonical_condition not in SUPPORTED_RELATIONSHIP_CONDITIONS:
        raise ValueError(
            "relationship_condition must be one of {}.".format(sorted(supported_condition_names))
        )
    out_dir = str(args.out_dir or "").strip()
    if not out_dir:
        out_dir = default_output_dir_for_condition(canonical_condition)

    config = SimulationConfig(
        n_players=args.n_players,
        seed=args.seed,
        out_dir=out_dir,
        relationship_condition=canonical_condition,
        experiment_tag=args.experiment_tag,
        grid_size=args.grid_size,
        foraging_targets=args.foraging_targets,
        cleaning_targets=args.cleaning_targets,
        n_high_value_food=args.n_high_value_food,
        step_cost=args.step_cost,
        beta_distance=args.beta_distance,
        sigma_noise=args.sigma_noise,
        temperature=args.temperature,
        planner_P_mean=args.planner_P_mean,
        planner_O_mean=args.planner_O_mean,
        hopper_P_mean=args.hopper_P_mean,
        hopper_O_mean=args.hopper_O_mean,
        latent_sd=args.latent_sd,
        foraging_beta_P=args.foraging_beta_P,
        foraging_beta_O=args.foraging_beta_O,
        foraging_low_value_reward=args.foraging_low_value_reward,
        foraging_high_value_reward=args.foraging_high_value_reward,
        cleaning_beta_P=args.cleaning_beta_P,
        cleaning_beta_O=args.cleaning_beta_O,
        cleaning_salience_low=args.cleaning_salience_low,
        cleaning_salience_medium=args.cleaning_salience_medium,
        cleaning_salience_high=args.cleaning_salience_high,
        cleaning_salience_jitter=args.cleaning_salience_jitter,
        n_example_plots=args.n_example_plots,
    )
    config.validate()
    return config


def simulate_dataset(config: SimulationConfig) -> Dict[str, object]:
    """Simulate the full paired-task dataset and save all outputs."""
    config.validate()
    requested_config = config
    config = config.resolve_condition_preset()
    config.validate()

    out_dir = Path(config.out_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    root_rng = np.random.default_rng(config.seed)
    players = sample_players(config, np.random.default_rng(int(root_rng.integers(0, 2**32 - 1))))
    foraging_env = build_foraging_environment(
        config,
        np.random.default_rng(int(root_rng.integers(0, 2**32 - 1))),
    )
    cleaning_env = build_cleaning_environment(
        config,
        np.random.default_rng(int(root_rng.integers(0, 2**32 - 1))),
    )
    environments = [foraging_env, cleaning_env]

    players_df = players_to_frame(players)
    targets_df = targets_to_frame(environments)
    players_df.to_csv(out_dir / "players.csv", index=False)
    targets_df.to_csv(out_dir / "targets.csv", index=False)
    save_json(out_dir / "config.json", config.to_dict())
    save_json(
        out_dir / "condition_manifest.json",
        _build_condition_manifest(requested_config, config),
    )
    save_json(out_dir / "environment_metadata.json", environments_to_metadata(environments))

    planner_examples, hopper_examples = select_example_players(players, config.n_example_plots)
    example_player_ids = {player.player_id for player in planner_examples + hopper_examples}
    retained_examples: Dict[Tuple[int, str], Dict[str, object]] = {}
    task_summaries: List[Dict[str, object]] = []

    with (
        (out_dir / "events.csv").open("w", encoding="utf-8", newline="") as events_handle,
        (out_dir / "sequences.jsonl").open("w", encoding="utf-8") as sequences_handle,
    ):
        events_writer = csv.DictWriter(events_handle, fieldnames=EVENT_COLUMNS)
        events_writer.writeheader()

        for player in players:
            for task_offset, environment in enumerate(environments):
                rng_seed = int(
                    (config.seed * 100_003 + player.player_id * 101 + task_offset * 17)
                    % (2**32 - 1)
                )
                task_rng = np.random.default_rng(rng_seed)
                simulation_payload = _simulate_player_task(
                    player=player,
                    environment=environment,
                    config=config,
                    rng=task_rng,
                )

                for row in simulation_payload["event_rows"]:
                    events_writer.writerow(row)

                sequence_record = build_compact_sequence_record(
                    player=player,
                    task_name=environment.task_name,
                    event_payload=simulation_payload["compact_events"],
                )
                write_jsonl_record(sequences_handle, sequence_record)
                task_summaries.append(simulation_payload["summary"])

                if player.player_id in example_player_ids:
                    retained_examples[(player.player_id, environment.task_name)] = {
                        "player": {
                            "player_id": player.player_id,
                            "latent_P": player.latent_P,
                            "latent_O": player.latent_O,
                            "label": player.label,
                        },
                        "summary": simulation_payload["summary"],
                        "events": simulation_payload["compact_events"],
                    }

    task_summaries_df = pd.DataFrame(task_summaries, columns=SUMMARY_COLUMNS)
    task_summaries_df.to_csv(out_dir / "task_summaries.csv", index=False)

    # An invalid cached path raises inside _simulate_player_task, so a finished run
    # always records zero failed paths.
    sanity_summary = build_sanity_summary(players_df, task_summaries_df, failed_paths=0)
    save_json(out_dir / "sanity_summary.json", sanity_summary)
    _print_sanity_summary(sanity_summary)

    _assert_task_completions(task_summaries_df, config)

    plot_environment(foraging_env, plots_dir / "foraging_layout.pdf")
    plot_environment(cleaning_env, plots_dir / "cleaning_layout.pdf")
    plot_latent_distribution(players_df, plots_dir / "latent_distribution.pdf")
    plot_behavior_summaries(task_summaries_df, plots_dir)

    for player in planner_examples + hopper_examples:
        for environment in environments:
            payload = retained_examples[(player.player_id, environment.task_name)]
            plot_trajectory(
                environment=environment,
                trajectory_payload=payload,
                output_path=plots_dir
                / f"player_{player.player_id}_{environment.task_name}_trajectory.pdf",
            )

    if planner_examples and hopper_examples:
        comparison_players = _select_task_relevant_comparison_players(
            planner_examples=planner_examples,
            hopper_examples=hopper_examples,
        )
        plot_planner_vs_hopper(
            environment=foraging_env,
            planner_payload=retained_examples[
                (comparison_players[TASK_NAME_FORAGING][0].player_id, TASK_NAME_FORAGING)
            ],
            hopper_payload=retained_examples[
                (comparison_players[TASK_NAME_FORAGING][1].player_id, TASK_NAME_FORAGING)
            ],
            output_path=plots_dir / "example_planner_vs_hopper_foraging.pdf",
        )
        plot_planner_vs_hopper(
            environment=cleaning_env,
            planner_payload=retained_examples[
                (comparison_players[TASK_NAME_CLEANING][0].player_id, TASK_NAME_CLEANING)
            ],
            hopper_payload=retained_examples[
                (comparison_players[TASK_NAME_CLEANING][1].player_id, TASK_NAME_CLEANING)
            ],
            output_path=plots_dir / "example_planner_vs_hopper_cleaning.pdf",
        )

    return {
        "out_dir": out_dir,
        "players_df": players_df,
        "targets_df": targets_df,
        "task_summaries_df": task_summaries_df,
        "sanity_summary": sanity_summary,
        "condition_manifest": _build_condition_manifest(requested_config, config),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    config = build_config_from_args(args)
    simulate_dataset(config)
    return 0


def _select_task_relevant_comparison_players(
    *,
    planner_examples: Sequence[PlayerProfile],
    hopper_examples: Sequence[PlayerProfile],
) -> Dict[str, Tuple[PlayerProfile, PlayerProfile]]:
    """Choose one Planner/Hopper comparison pair per task from the quadrant examples.

    Foraging pairs match on the sign of P and differ on O (low P/low O Planner vs low P/high
    O Hopper); cleaning pairs match on O and differ on P (high P/high O Planner vs low
    P/high O Hopper).  Without an example in the exact quadrant, the example with the
    largest absolute task-relevant latent is used.
    """
    if not planner_examples or not hopper_examples:
        raise ValueError("Need at least one Planner and one Hopper example.")

    def _pick_example(
        candidates: Sequence[PlayerProfile],
        *,
        predicate,
        ranking_key,
    ) -> PlayerProfile:
        matching = [candidate for candidate in candidates if predicate(candidate)]
        pool = matching if matching else list(candidates)
        return sorted(pool, key=ranking_key, reverse=True)[0]

    foraging_planner = _pick_example(
        planner_examples,
        predicate=lambda player: player.latent_P < 0 and player.latent_O < 0,
        ranking_key=lambda player: abs(player.latent_O),
    )
    foraging_hopper = _pick_example(
        hopper_examples,
        predicate=lambda player: player.latent_P < 0 and player.latent_O > 0,
        ranking_key=lambda player: abs(player.latent_O),
    )
    cleaning_planner = _pick_example(
        planner_examples,
        predicate=lambda player: player.latent_P > 0 and player.latent_O > 0,
        ranking_key=lambda player: abs(player.latent_P),
    )
    cleaning_hopper = _pick_example(
        hopper_examples,
        predicate=lambda player: player.latent_P < 0 and player.latent_O > 0,
        ranking_key=lambda player: abs(player.latent_P),
    )

    return {
        TASK_NAME_FORAGING: (foraging_planner, foraging_hopper),
        TASK_NAME_CLEANING: (cleaning_planner, cleaning_hopper),
    }


def _build_condition_manifest(
    requested_config: SimulationConfig,
    resolved_config: SimulationConfig,
) -> Dict[str, object]:
    """Describe the requested condition and the resolved simulator parameters."""
    return {
        "experiment_tag": str(resolved_config.experiment_tag or ""),
        "relationship_condition": str(requested_config.relationship_condition),
        "applied_parameter_overrides": resolved_config.condition_overrides(),
        "requested_config": requested_config.to_dict(),
        "resolved_config": resolved_config.to_dict(),
    }


def _simulate_player_task(
    player: PlayerProfile,
    environment: TaskEnvironment,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> Dict[str, object]:
    unfinished_targets: Dict[str, Target] = {
        target.target_id: target for target in environment.targets
    }
    completed_target_ids = set()
    current_position = environment.start_pos
    previous_patch: Optional[int] = None
    visited_patches = set()

    event_rows: List[Dict[str, object]] = []
    compact_events: List[Dict[str, object]] = []
    completion_records: List[Dict[str, object]] = []

    event_index = 0
    decision_index = 0
    cumulative_reward = 0.0
    cumulative_cost = 0.0
    total_steps = 0
    patch_switch_count = 0
    patch_revisit_count = 0

    while unfinished_targets:
        decision_index += 1
        current_patch_context = previous_patch
        choice = choose_next_target(
            player=player,
            environment=environment,
            unfinished_targets=list(unfinished_targets.values()),
            current_position=current_position,
            current_patch=current_patch_context,
            config=config,
            rng=rng,
        )
        selected_target = choice.selected_target
        if selected_target.target_id not in unfinished_targets:
            raise AssertionError(f"Target {selected_target.target_id} selected after completion.")

        path = environment.shortest_paths[(current_position, selected_target.position)]
        distance = environment.shortest_distances[(current_position, selected_target.position)]
        if not path or path[0] != current_position or path[-1] != selected_target.position:
            raise AssertionError(f"Invalid cached path for {environment.task_name}.")

        _assert_path_is_valid(environment, path)
        selected_patch = selected_target.patch_id
        num_completed_targets = len(completed_target_ids)
        total_targets = len(environment.targets)

        for next_position in path[1:]:
            action = _movement_action(current_position, next_position)
            previous_position = current_position
            current_position = next_position
            total_steps += 1
            cumulative_cost += config.step_cost

            event_rows.append(
                _build_event_row(
                    player=player,
                    environment=environment,
                    event_index=event_index,
                    decision_index=decision_index,
                    event_type="move",
                    action=action,
                    current_position=current_position,
                    previous_position=previous_position,
                    current_patch=selected_patch,
                    previous_patch=previous_patch,
                    selected_target=selected_target,
                    completed_target=None,
                    completion_rank=None,
                    shortest_distance=distance,
                    step_cost=config.step_cost,
                    reward_received=0.0,
                    cumulative_reward=cumulative_reward,
                    cumulative_cost=cumulative_cost,
                    remaining_targets=len(unfinished_targets),
                    num_completed_targets=num_completed_targets,
                    total_targets=total_targets,
                    patch_switch_event=0,
                    patch_revisit_event=0,
                )
            )
            compact_events.append(
                _build_compact_event(
                    event_type="move",
                    action=action,
                    current_position=current_position,
                    selected_target=selected_target,
                    remaining_targets=len(unfinished_targets),
                    completion_rank=None,
                )
            )
            event_index += 1

        patch_switch_event = int(previous_patch is not None and selected_patch != previous_patch)
        patch_revisit_event = int(selected_patch in visited_patches and patch_switch_event == 1)
        patch_switch_count += patch_switch_event
        patch_revisit_count += patch_revisit_event

        previous_position = current_position
        cumulative_reward += float(selected_target.reward)
        completed_target_ids.add(selected_target.target_id)
        completion_rank = len(completed_target_ids)
        del unfinished_targets[selected_target.target_id]
        visited_patches.add(selected_patch)
        previous_patch = selected_patch

        event_rows.append(
            _build_event_row(
                player=player,
                environment=environment,
                event_index=event_index,
                decision_index=decision_index,
                event_type="collect" if environment.task_name == TASK_NAME_FORAGING else "clean",
                action="COLLECT" if environment.task_name == TASK_NAME_FORAGING else "CLEAN",
                current_position=current_position,
                previous_position=previous_position,
                current_patch=selected_patch,
                previous_patch=current_patch_context,
                selected_target=selected_target,
                completed_target=selected_target,
                completion_rank=completion_rank,
                shortest_distance=distance,
                step_cost=config.step_cost,
                reward_received=float(selected_target.reward),
                cumulative_reward=cumulative_reward,
                cumulative_cost=cumulative_cost,
                remaining_targets=len(unfinished_targets),
                num_completed_targets=completion_rank,
                total_targets=total_targets,
                patch_switch_event=patch_switch_event,
                patch_revisit_event=patch_revisit_event,
            )
        )
        compact_events.append(
            _build_compact_event(
                event_type="collect" if environment.task_name == TASK_NAME_FORAGING else "clean",
                action="COLLECT" if environment.task_name == TASK_NAME_FORAGING else "CLEAN",
                current_position=current_position,
                selected_target=selected_target,
                remaining_targets=len(unfinished_targets),
                completion_rank=completion_rank,
            )
        )
        completion_records.append(
            {
                "target_id": selected_target.target_id,
                "patch_id": selected_patch,
                "reward": float(selected_target.reward),
                "completion_rank": completion_rank,
                "is_high_value": bool(selected_target.is_high_value),
            }
        )
        event_index += 1

    if len(completed_target_ids) != len(environment.targets):
        raise AssertionError(f"Incomplete task trajectory for player {player.player_id}.")

    summary = compute_task_summary(
        player=player,
        environment=environment,
        completion_records=completion_records,
        total_steps=total_steps,
        total_reward=cumulative_reward,
        total_cost=cumulative_cost,
        patch_switch_count=patch_switch_count,
        patch_revisit_count=patch_revisit_count,
        early_fraction=config.early_fraction,
    )

    return {
        "event_rows": event_rows,
        "compact_events": compact_events,
        "summary": summary,
    }


def _build_event_row(
    player: PlayerProfile,
    environment: TaskEnvironment,
    event_index: int,
    decision_index: int,
    event_type: str,
    action: str,
    current_position: Position,
    previous_position: Position,
    current_patch: Optional[int],
    previous_patch: Optional[int],
    selected_target: Target,
    completed_target: Optional[Target],
    completion_rank: Optional[int],
    shortest_distance: int,
    step_cost: float,
    reward_received: float,
    cumulative_reward: float,
    cumulative_cost: float,
    remaining_targets: int,
    num_completed_targets: int,
    total_targets: int,
    patch_switch_event: int,
    patch_revisit_event: int,
) -> Dict[str, object]:
    return {
        "player_id": player.player_id,
        "task_name": environment.task_name,
        "label": player.label,
        "class_id": player.class_id,
        "latent_P": player.latent_P,
        "latent_O": player.latent_O,
        "event_index": event_index,
        "decision_index": decision_index,
        "time_step": event_index,
        "event_type": event_type,
        "action": action,
        "agent_x": current_position[0],
        "agent_y": current_position[1],
        "previous_x": previous_position[0],
        "previous_y": previous_position[1],
        "current_patch": current_patch,
        "previous_patch": previous_patch,
        "selected_target_id": selected_target.target_id,
        "selected_target_x": selected_target.x,
        "selected_target_y": selected_target.y,
        "selected_target_patch": selected_target.patch_id,
        "selected_target_type": selected_target.target_type,
        "selected_target_reward": selected_target.reward,
        "selected_target_salience": selected_target.salience,
        "completed_target_id": None if completed_target is None else completed_target.target_id,
        "completed_target_x": None if completed_target is None else completed_target.x,
        "completed_target_y": None if completed_target is None else completed_target.y,
        "completed_target_patch": None if completed_target is None else completed_target.patch_id,
        "completed_target_type": None if completed_target is None else completed_target.target_type,
        "completed_target_reward": None if completed_target is None else completed_target.reward,
        "completed_target_salience": None
        if completed_target is None
        else completed_target.salience,
        "completion_rank": completion_rank,
        "shortest_path_distance_to_selected": shortest_distance,
        "step_cost": step_cost,
        "reward_received": reward_received,
        "cumulative_reward": cumulative_reward,
        "cumulative_cost": cumulative_cost,
        "cumulative_net_reward": cumulative_reward - cumulative_cost,
        "remaining_targets": remaining_targets,
        "num_completed_targets": num_completed_targets,
        "total_targets": total_targets,
        "patch_switch_event": patch_switch_event,
        "patch_revisit_event": patch_revisit_event,
    }


def _build_compact_event(
    event_type: str,
    action: str,
    current_position: Position,
    selected_target: Target,
    remaining_targets: int,
    completion_rank: Optional[int],
) -> Dict[str, object]:
    return {
        "event_type": event_type,
        "action": action,
        "x": current_position[0],
        "y": current_position[1],
        "target_id": selected_target.target_id,
        "target_type": selected_target.target_type,
        "target_reward": selected_target.reward,
        "target_salience": selected_target.salience,
        "target_patch": selected_target.patch_id,
        "remaining_targets": remaining_targets,
        "completion_rank": completion_rank,
    }


def _movement_action(previous_position: Position, current_position: Position) -> str:
    delta = (current_position[0] - previous_position[0], current_position[1] - previous_position[1])
    if delta == (0, -1):
        return "UP"
    if delta == (0, 1):
        return "DOWN"
    if delta == (-1, 0):
        return "LEFT"
    if delta == (1, 0):
        return "RIGHT"
    raise ValueError(f"Non-adjacent move detected: {previous_position} -> {current_position}")


def _assert_path_is_valid(environment: TaskEnvironment, path: Sequence[Position]) -> None:
    for previous_position, current_position in zip(path[:-1], path[1:]):
        if not environment.is_passable(current_position):
            raise AssertionError(f"Path enters blocked cell {current_position}.")
        if (
            abs(current_position[0] - previous_position[0])
            + abs(current_position[1] - previous_position[1])
            != 1
        ):
            raise AssertionError("Path contains a non-adjacent step.")


def _assert_task_completions(task_summaries_df: pd.DataFrame, config: SimulationConfig) -> None:
    foraging_counts = task_summaries_df.loc[
        task_summaries_df["task_name"] == TASK_NAME_FORAGING,
        "total_targets",
    ]
    cleaning_counts = task_summaries_df.loc[
        task_summaries_df["task_name"] == TASK_NAME_CLEANING,
        "total_targets",
    ]
    if not (foraging_counts == config.foraging_targets).all():
        raise AssertionError("At least one foraging trajectory did not complete all targets.")
    if not (cleaning_counts == config.cleaning_targets).all():
        raise AssertionError("At least one cleaning trajectory did not complete all targets.")


def _print_sanity_summary(sanity_summary: Dict[str, object]) -> None:
    print("Simulation summary")
    print(f"  players: {sanity_summary['n_players']}")
    print(f"  label_counts: {sanity_summary['label_counts']}")
    print(
        f"  mean_total_steps_by_label_and_task: {sanity_summary['mean_total_steps_by_label_and_task']}"
    )
    print(
        f"  mean_patch_switches_by_label_and_task: {sanity_summary['mean_patch_switches_by_label_and_task']}"
    )
    print(
        f"  mean_patch_run_length_by_label_and_task: {sanity_summary['mean_patch_run_length_by_label_and_task']}"
    )
    print(
        f"  foraging_early_high_value_rate_by_label: {sanity_summary['foraging_early_high_value_rate_by_label']}"
    )
    print(f"  average_trajectory_length: {sanity_summary['average_trajectory_length']:.3f}")
    print(f"  failed_paths: {sanity_summary['failed_paths']}")


if __name__ == "__main__":
    raise SystemExit(main())
