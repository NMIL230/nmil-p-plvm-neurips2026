"""Environment builders and pathfinding utilities for the paired-task simulator."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from simulation.config import (
    SimulationConfig,
    TASK_NAME_CLEANING,
    TASK_NAME_FORAGING,
)


Position = Tuple[int, int]


@dataclass(frozen=True)
class Target:
    """One task target with explicit patch metadata."""

    task_name: str
    target_id: str
    x: int
    y: int
    patch_id: int
    target_type: str
    reward: float
    salience: float = 1.0
    is_high_value: bool = False

    @property
    def position(self) -> Position:
        """Return the `(x, y)` position of the target."""
        return (int(self.x), int(self.y))


@dataclass
class TaskEnvironment:
    """Grid-world environment shared by all players for one task."""

    task_name: str
    grid_size: int
    terrain: np.ndarray
    start_pos: Position
    targets: List[Target]
    obstacle_labels: Dict[int, str]
    shortest_paths: Dict[Tuple[Position, Position], List[Position]] = field(default_factory=dict)
    shortest_distances: Dict[Tuple[Position, Position], int] = field(default_factory=dict)

    def is_passable(self, position: Position) -> bool:
        """Return whether the supplied cell can be traversed."""
        x, y = position
        return bool(
            0 <= x < self.grid_size and 0 <= y < self.grid_size and int(self.terrain[y, x]) == 0
        )


def build_foraging_environment(
    config: SimulationConfig, rng: np.random.Generator
) -> TaskEnvironment:
    """Build the outdoor foraging environment with messy patches and obstacles."""
    food_types = ["apple", "berries", "mushroom", "nut"]
    size = config.grid_size

    for _ in range(200):
        terrain = np.zeros((size, size), dtype=np.int8)
        _paint_foraging_obstacles(terrain, rng)
        start_pos = _nearest_passable(terrain, (1, size - 2))
        _clear_square(terrain, start_pos, radius=1)

        patch_count = int(
            rng.integers(config.foraging_patch_count_min, config.foraging_patch_count_max + 1)
        )
        target_counts = _allocate_counts(
            total=config.foraging_targets,
            n_groups=patch_count,
            min_count=_feasible_min_count(
                total=config.foraging_targets,
                n_groups=patch_count,
                requested_min=max(3, config.foraging_targets // max(1, patch_count * 3)),
            ),
            rng=rng,
        )

        candidate_centers = _foraging_candidate_centers(size)
        rng.shuffle(candidate_centers)
        desired_centers = candidate_centers[:patch_count]

        used_positions: Set[Position] = {start_pos}
        patch_positions: List[List[Position]] = []
        valid_layout = True

        for patch_id, (desired_center, count) in enumerate(zip(desired_centers, target_counts)):
            jittered_center = (
                int(np.clip(desired_center[0] + rng.integers(-1, 2), 1, size - 2)),
                int(np.clip(desired_center[1] + rng.integers(-1, 2), 1, size - 2)),
            )
            center = _nearest_passable(terrain, jittered_center, forbidden=used_positions)
            if center is None:
                valid_layout = False
                break
            positions = _cluster_positions(
                terrain=terrain,
                allowed_cells=None,
                desired_center=center,
                count=count,
                used_positions=used_positions,
                rng=rng,
                max_radius=max(3, size // 5),
            )
            if len(positions) != count:
                valid_layout = False
                break
            patch_positions.append(positions)
            used_positions.update(positions)

        if not valid_layout:
            continue

        targets: List[Target] = []
        flat_positions = [position for patch in patch_positions for position in patch]
        high_value_indices = _choose_high_value_indices(
            patch_positions=patch_positions,
            n_high_value=config.n_high_value_food,
            rng=rng,
        )

        for patch_id, positions in enumerate(patch_positions):
            for position in positions:
                flat_index = flat_positions.index(position)
                is_high_value = flat_index in high_value_indices
                target_type = "golden_fruit" if is_high_value else str(rng.choice(food_types))
                reward = (
                    config.foraging_high_value_reward
                    if is_high_value
                    else config.foraging_low_value_reward
                )
                targets.append(
                    Target(
                        task_name=TASK_NAME_FORAGING,
                        target_id=f"foraging_target_{len(targets):03d}",
                        x=position[0],
                        y=position[1],
                        patch_id=patch_id,
                        target_type=target_type,
                        reward=float(reward),
                        salience=float(reward),
                        is_high_value=is_high_value,
                    )
                )

        environment = TaskEnvironment(
            task_name=TASK_NAME_FORAGING,
            grid_size=size,
            terrain=terrain,
            start_pos=start_pos,
            targets=targets,
            obstacle_labels={
                0: "ground",
                1: "water",
                2: "trees",
                3: "rocks",
                4: "logs",
            },
        )
        _precompute_pairwise_paths(environment)
        _validate_environment(environment)
        return environment

    raise RuntimeError("Failed to generate a valid foraging environment.")


def build_cleaning_environment(
    config: SimulationConfig, rng: np.random.Generator
) -> TaskEnvironment:
    """Build the indoor cleaning environment with rooms, doors, and dirty patches."""
    dirt_types = ["dirt_tile", "spill", "dust", "stain", "footprint"]
    size = config.grid_size

    for _ in range(100):
        terrain = np.zeros((size, size), dtype=np.int8)
        terrain[0, :] = 1
        terrain[-1, :] = 1
        terrain[:, 0] = 1
        terrain[:, -1] = 1

        x_split_1 = max(4, size // 3)
        x_split_2 = min(size - 5, (2 * size) // 3)
        y_split = size // 2

        _paint_vertical_wall(
            terrain,
            x_split_1,
            gap_centers=[max(2, size // 4), min(size - 3, (3 * size) // 4)],
            gap_half_width=1,
        )
        _paint_vertical_wall(
            terrain,
            x_split_2,
            gap_centers=[max(2, size // 3), min(size - 3, (2 * size) // 3)],
            gap_half_width=1,
        )
        _paint_horizontal_wall(
            terrain,
            y_split,
            gap_centers=[max(2, size // 4), size // 2, min(size - 3, (3 * size) // 4)],
            gap_half_width=1,
        )

        start_pos = _nearest_passable(terrain, (size // 2, y_split))
        _clear_square(terrain, start_pos, radius=1)

        rooms = {
            "room_a": (1, x_split_1 - 1, 1, y_split - 1),
            "room_b": (x_split_1 + 1, x_split_2 - 1, 1, y_split - 1),
            "room_c": (x_split_2 + 1, size - 2, 1, y_split - 1),
            "room_d": (1, x_split_1 - 1, y_split + 1, size - 2),
            "room_e": (x_split_1 + 1, x_split_2 - 1, y_split + 1, size - 2),
            "room_f": (x_split_2 + 1, size - 2, y_split + 1, size - 2),
        }

        n_isolated = max(2, min(6, config.cleaning_targets // 12))
        main_total = config.cleaning_targets - n_isolated
        main_counts = _allocate_counts(
            total=main_total,
            n_groups=6,
            min_count=_feasible_min_count(total=main_total, n_groups=6, requested_min=4),
            rng=rng,
        )

        used_positions: Set[Position] = {start_pos}
        targets: List[Target] = []
        patch_id = 0

        shape_specs = [
            ("cluster", "room_a", {"max_radius": max(2, size // 7)}),
            ("strip_horizontal", "room_b", {}),
            ("cluster", "room_c", {"max_radius": max(2, size // 8)}),
            ("corner_sw", "room_d", {}),
            ("strip_vertical", "room_e", {}),
            ("corner_ne", "room_f", {}),
        ]

        valid_layout = True
        for count, (shape_name, room_name, extras) in zip(main_counts, shape_specs):
            rect = rooms[room_name]
            allowed_cells = _rect_passable_cells(terrain, rect)

            if shape_name == "cluster":
                desired_center = _rect_center(rect)
                positions = _cluster_positions(
                    terrain=terrain,
                    allowed_cells=allowed_cells,
                    desired_center=desired_center,
                    count=count,
                    used_positions=used_positions,
                    rng=rng,
                    max_radius=int(extras["max_radius"]),
                )
            elif shape_name == "strip_horizontal":
                positions = _strip_positions(
                    allowed_cells=allowed_cells,
                    rect=rect,
                    count=count,
                    orientation="horizontal",
                    used_positions=used_positions,
                    rng=rng,
                )
            elif shape_name == "strip_vertical":
                positions = _strip_positions(
                    allowed_cells=allowed_cells,
                    rect=rect,
                    count=count,
                    orientation="vertical",
                    used_positions=used_positions,
                    rng=rng,
                )
            elif shape_name == "corner_sw":
                positions = _corner_positions(
                    allowed_cells=allowed_cells,
                    rect=rect,
                    count=count,
                    corner="sw",
                    used_positions=used_positions,
                    rng=rng,
                )
            elif shape_name == "corner_ne":
                positions = _corner_positions(
                    allowed_cells=allowed_cells,
                    rect=rect,
                    count=count,
                    corner="ne",
                    used_positions=used_positions,
                    rng=rng,
                )
            else:
                raise ValueError(f"Unsupported shape spec: {shape_name}")

            if len(positions) != count:
                valid_layout = False
                break

            for position in positions:
                target_type = str(rng.choice(dirt_types))
                targets.append(
                    Target(
                        task_name=TASK_NAME_CLEANING,
                        target_id=f"cleaning_target_{len(targets):03d}",
                        x=position[0],
                        y=position[1],
                        patch_id=patch_id,
                        target_type=target_type,
                        reward=float(config.cleaning_reward),
                        salience=_sample_cleaning_salience(target_type, config, rng),
                        is_high_value=False,
                    )
                )
            used_positions.update(positions)
            patch_id += 1

        if not valid_layout:
            continue

        isolated_rooms = ["room_b", "room_c", "room_d", "room_e", "room_a", "room_f"]
        for isolated_index in range(n_isolated):
            room_name = isolated_rooms[isolated_index % len(isolated_rooms)]
            rect = rooms[room_name]
            allowed_cells = _rect_passable_cells(terrain, rect)
            position = _choose_isolated_position(allowed_cells, used_positions, rng)
            if position is None:
                valid_layout = False
                break
            target_type = str(rng.choice(dirt_types))
            targets.append(
                Target(
                    task_name=TASK_NAME_CLEANING,
                    target_id=f"cleaning_target_{len(targets):03d}",
                    x=position[0],
                    y=position[1],
                    patch_id=patch_id,
                    target_type=target_type,
                    reward=float(config.cleaning_reward),
                    salience=_sample_cleaning_salience(target_type, config, rng),
                    is_high_value=False,
                )
            )
            used_positions.add(position)
            patch_id += 1

        if not valid_layout or len(targets) != config.cleaning_targets:
            continue

        environment = TaskEnvironment(
            task_name=TASK_NAME_CLEANING,
            grid_size=size,
            terrain=terrain,
            start_pos=start_pos,
            targets=targets,
            obstacle_labels={0: "floor", 1: "wall"},
        )
        _precompute_pairwise_paths(environment)
        _validate_environment(environment)
        return environment

    raise RuntimeError("Failed to generate a valid cleaning environment.")


def _validate_environment(environment: TaskEnvironment) -> None:
    """Assert that every target is reachable from the shared start position."""
    for target in environment.targets:
        key = (environment.start_pos, target.position)
        if key not in environment.shortest_paths:
            raise ValueError(f"Missing path from start to target {target.target_id}.")
        path = environment.shortest_paths[key]
        if not path or path[0] != environment.start_pos or path[-1] != target.position:
            raise ValueError(f"Invalid path cached for target {target.target_id}.")


def _neighbors(terrain: np.ndarray, position: Position) -> Iterable[Position]:
    x, y = position
    height, width = terrain.shape
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        nx = x + dx
        ny = y + dy
        if 0 <= nx < width and 0 <= ny < height and int(terrain[ny, nx]) == 0:
            yield (nx, ny)


def _bfs_tree(
    terrain: np.ndarray, start: Position
) -> Tuple[Dict[Position, Optional[Position]], Dict[Position, int]]:
    parents: Dict[Position, Optional[Position]] = {start: None}
    distances: Dict[Position, int] = {start: 0}
    frontier: deque[Position] = deque([start])

    while frontier:
        current = frontier.popleft()
        for neighbor in _neighbors(terrain, current):
            if neighbor in parents:
                continue
            parents[neighbor] = current
            distances[neighbor] = distances[current] + 1
            frontier.append(neighbor)

    return parents, distances


def _reconstruct_path(
    parents: Dict[Position, Optional[Position]],
    start: Position,
    goal: Position,
) -> List[Position]:
    path = [goal]
    current = goal
    while current != start:
        parent = parents[current]
        if parent is None:
            break
        path.append(parent)
        current = parent
    path.reverse()
    return path


def _precompute_pairwise_paths(environment: TaskEnvironment) -> None:
    nodes = [environment.start_pos] + [target.position for target in environment.targets]
    unique_nodes = list(dict.fromkeys(nodes))

    for source in unique_nodes:
        parents, distances = _bfs_tree(environment.terrain, source)
        for destination in unique_nodes:
            if destination not in parents:
                raise ValueError(
                    f"Unreachable node pair in {environment.task_name}: {source} -> {destination}"
                )
            path = _reconstruct_path(parents, source, destination)
            environment.shortest_paths[(source, destination)] = path
            environment.shortest_distances[(source, destination)] = int(distances[destination])


def _allocate_counts(
    total: int,
    n_groups: int,
    min_count: int,
    rng: np.random.Generator,
) -> List[int]:
    if n_groups <= 0:
        raise ValueError("n_groups must be positive.")
    if min_count * n_groups > total:
        raise ValueError("min_count is too large for the requested total.")

    counts = np.full(n_groups, min_count, dtype=int)
    remaining = int(total - counts.sum())
    if remaining > 0:
        counts += rng.multinomial(remaining, np.full(n_groups, 1.0 / n_groups))
    rng.shuffle(counts)
    return counts.tolist()


def _feasible_min_count(total: int, n_groups: int, requested_min: int) -> int:
    if n_groups <= 0:
        raise ValueError("n_groups must be positive.")
    return max(1, min(int(requested_min), int(total // n_groups)))


def _foraging_candidate_centers(size: int) -> List[Position]:
    return [
        (max(2, size // 6), max(2, size // 5)),
        (max(2, size // 4), min(size - 3, (4 * size) // 5)),
        (max(2, size // 2 - 2), max(2, size // 4)),
        (min(size - 3, (4 * size) // 5), max(2, size // 3)),
        (min(size - 3, (5 * size) // 6), min(size - 3, (3 * size) // 4)),
        (min(size - 3, size - 4), min(size - 3, size // 2)),
    ]


def _paint_foraging_obstacles(terrain: np.ndarray, rng: np.random.Generator) -> None:
    size = terrain.shape[0]
    water_x = size // 2 + int(rng.integers(-1, 2))
    water_gaps = {max(2, size // 4), size // 2, min(size - 3, (3 * size) // 4)}
    for y in range(1, size - 1):
        if any(abs(y - gap) <= 1 for gap in water_gaps):
            continue
        drift = int(rng.integers(-1, 2))
        width = 2 if rng.random() < 0.35 else 1
        for dx in range(width):
            x = int(np.clip(water_x + drift + dx, 2, size - 3))
            terrain[y, x] = 1
            if rng.random() < 0.2 and x + 1 < size - 1:
                terrain[y, x + 1] = 1

    log_y = int(np.clip(round(size * 0.58), 2, size - 3))
    log_gaps = {max(2, size // 5), size // 2, min(size - 3, (4 * size) // 5)}
    for x in range(2, size - 2):
        if any(abs(x - gap) <= 1 for gap in log_gaps):
            continue
        terrain[log_y, x] = 4
        if rng.random() < 0.18 and log_y + 1 < size - 1:
            terrain[log_y + 1, x] = 4

    for _ in range(3):
        seed = (int(rng.integers(2, size - 2)), int(rng.integers(2, size - 2)))
        _paint_blob(terrain, seed=seed, code=2, blob_size=max(5, size // 3), rng=rng)
    for _ in range(2):
        seed = (int(rng.integers(2, size - 2)), int(rng.integers(2, size - 2)))
        _paint_blob(terrain, seed=seed, code=3, blob_size=max(4, size // 4), rng=rng)


def _paint_blob(
    terrain: np.ndarray,
    seed: Position,
    code: int,
    blob_size: int,
    rng: np.random.Generator,
) -> None:
    height, width = terrain.shape
    cells = {seed}
    frontier = [seed]
    while frontier and len(cells) < blob_size:
        current = frontier[int(rng.integers(0, len(frontier)))]
        dx, dy = ((0, -1), (0, 1), (-1, 0), (1, 0))[int(rng.integers(0, 4))]
        candidate = (current[0] + dx, current[1] + dy)
        if (
            1 <= candidate[0] < width - 1
            and 1 <= candidate[1] < height - 1
            and candidate not in cells
        ):
            cells.add(candidate)
            frontier.append(candidate)
        if rng.random() < 0.08 and len(frontier) > 1:
            frontier.pop(0)

    for x, y in cells:
        terrain[y, x] = code


def _paint_vertical_wall(
    terrain: np.ndarray,
    x: int,
    gap_centers: Sequence[int],
    gap_half_width: int,
) -> None:
    height = terrain.shape[0]
    for y in range(1, height - 1):
        if any(abs(y - gap_center) <= gap_half_width for gap_center in gap_centers):
            continue
        terrain[y, x] = 1


def _paint_horizontal_wall(
    terrain: np.ndarray,
    y: int,
    gap_centers: Sequence[int],
    gap_half_width: int,
) -> None:
    width = terrain.shape[1]
    for x in range(1, width - 1):
        if any(abs(x - gap_center) <= gap_half_width for gap_center in gap_centers):
            continue
        terrain[y, x] = 1


def _sample_cleaning_salience(
    target_type: str,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> float:
    """Assign weakly varying cleaning salience while keeping reward constant."""
    if target_type in {"spill", "stain"}:
        base_salience = float(config.cleaning_salience_high)
    elif target_type in {"footprint", "dirt_tile"}:
        base_salience = float(config.cleaning_salience_medium)
    else:
        base_salience = float(config.cleaning_salience_low)

    if config.cleaning_salience_jitter <= 0:
        return base_salience

    jitter = float(rng.uniform(-config.cleaning_salience_jitter, config.cleaning_salience_jitter))
    return float(max(0.1, base_salience + jitter))


def _cluster_positions(
    terrain: np.ndarray,
    allowed_cells: Optional[Sequence[Position]],
    desired_center: Position,
    count: int,
    used_positions: Set[Position],
    rng: np.random.Generator,
    max_radius: int,
) -> List[Position]:
    allowed = (
        set(allowed_cells)
        if allowed_cells is not None
        else {
            (x, y)
            for y in range(terrain.shape[0])
            for x in range(terrain.shape[1])
            if int(terrain[y, x]) == 0
        }
    )
    eligible = [position for position in allowed if position not in used_positions]
    if not eligible:
        return []

    center = _nearest_position(eligible, desired_center)
    if center is None:
        return []

    cluster = [center]
    cluster_set = {center}
    frontier = [center]
    attempts = 0

    while frontier and len(cluster) < count and attempts < count * 120:
        attempts += 1
        current = frontier[int(rng.integers(0, len(frontier)))]
        dx, dy = ((0, -1), (0, 1), (-1, 0), (1, 0))[int(rng.integers(0, 4))]
        candidate = (current[0] + dx, current[1] + dy)
        if candidate not in allowed or candidate in used_positions or candidate in cluster_set:
            continue
        if _manhattan(candidate, center) > max_radius + int(rng.integers(0, 2)):
            continue
        cluster.append(candidate)
        cluster_set.add(candidate)
        frontier.append(candidate)

    if len(cluster) < count:
        fallbacks = sorted(
            (position for position in eligible if position not in cluster_set),
            key=lambda position: (
                _manhattan(position, center),
                float(rng.random()),
            ),
        )
        for position in fallbacks:
            cluster.append(position)
            cluster_set.add(position)
            if len(cluster) == count:
                break

    return cluster[:count]


def _strip_positions(
    allowed_cells: Sequence[Position],
    rect: Tuple[int, int, int, int],
    count: int,
    orientation: str,
    used_positions: Set[Position],
    rng: np.random.Generator,
) -> List[Position]:
    x_min, x_max, y_min, y_max = rect
    allowed = set(allowed_cells)
    positions: List[Position] = []

    if orientation == "horizontal":
        base_y = int(rng.integers(y_min + 1, max(y_min + 2, y_max)))
        ordered_x = list(range(x_min, x_max + 1))
        rng.shuffle(ordered_x)
        ordered_x.sort()
        y_offsets = [0, -1, 1]
        for y_offset in y_offsets:
            for x in ordered_x:
                candidate = (x, int(np.clip(base_y + y_offset, y_min, y_max)))
                if (
                    candidate in allowed
                    and candidate not in used_positions
                    and candidate not in positions
                ):
                    positions.append(candidate)
                    if len(positions) == count:
                        return positions
    elif orientation == "vertical":
        base_x = int(rng.integers(x_min + 1, max(x_min + 2, x_max)))
        ordered_y = list(range(y_min, y_max + 1))
        ordered_y.sort()
        x_offsets = [0, -1, 1]
        for x_offset in x_offsets:
            for y in ordered_y:
                candidate = (int(np.clip(base_x + x_offset, x_min, x_max)), y)
                if (
                    candidate in allowed
                    and candidate not in used_positions
                    and candidate not in positions
                ):
                    positions.append(candidate)
                    if len(positions) == count:
                        return positions
    else:
        raise ValueError(f"Unsupported strip orientation: {orientation}")

    fallbacks = sorted(
        (
            position
            for position in allowed
            if position not in used_positions and position not in positions
        ),
        key=lambda position: (
            _manhattan(position, _rect_center(rect)),
            float(rng.random()),
        ),
    )
    for position in fallbacks:
        positions.append(position)
        if len(positions) == count:
            break
    return positions


def _corner_positions(
    allowed_cells: Sequence[Position],
    rect: Tuple[int, int, int, int],
    count: int,
    corner: str,
    used_positions: Set[Position],
    rng: np.random.Generator,
) -> List[Position]:
    x_min, x_max, y_min, y_max = rect
    if corner == "nw":
        anchor = (x_min, y_min)
    elif corner == "ne":
        anchor = (x_max, y_min)
    elif corner == "sw":
        anchor = (x_min, y_max)
    elif corner == "se":
        anchor = (x_max, y_max)
    else:
        raise ValueError(f"Unsupported corner token: {corner}")

    ranked = sorted(
        (position for position in allowed_cells if position not in used_positions),
        key=lambda position: (
            _manhattan(position, anchor),
            float(rng.random()),
        ),
    )
    return ranked[:count]


def _choose_isolated_position(
    allowed_cells: Sequence[Position],
    used_positions: Set[Position],
    rng: np.random.Generator,
) -> Optional[Position]:
    candidates = [position for position in allowed_cells if position not in used_positions]
    if not candidates:
        return None
    if not used_positions:
        return candidates[int(rng.integers(0, len(candidates)))]

    return max(
        candidates,
        key=lambda position: (
            min(_manhattan(position, used) for used in used_positions),
            float(rng.random()),
        ),
    )


def _rect_passable_cells(terrain: np.ndarray, rect: Tuple[int, int, int, int]) -> List[Position]:
    x_min, x_max, y_min, y_max = rect
    cells = []
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            if int(terrain[y, x]) == 0:
                cells.append((x, y))
    return cells


def _rect_center(rect: Tuple[int, int, int, int]) -> Position:
    x_min, x_max, y_min, y_max = rect
    return ((x_min + x_max) // 2, (y_min + y_max) // 2)


def _nearest_passable(
    terrain: np.ndarray,
    desired_position: Position,
    forbidden: Optional[Set[Position]] = None,
) -> Optional[Position]:
    forbidden = forbidden or set()
    passable_positions = [
        (x, y)
        for y in range(terrain.shape[0])
        for x in range(terrain.shape[1])
        if int(terrain[y, x]) == 0 and (x, y) not in forbidden
    ]
    return _nearest_position(passable_positions, desired_position)


def _nearest_position(
    candidates: Sequence[Position], desired_position: Position
) -> Optional[Position]:
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda position: (_manhattan(position, desired_position), position[1], position[0]),
    )


def _clear_square(terrain: np.ndarray, center: Position, radius: int) -> None:
    width = terrain.shape[1]
    height = terrain.shape[0]
    for y in range(max(0, center[1] - radius), min(height, center[1] + radius + 1)):
        for x in range(max(0, center[0] - radius), min(width, center[0] + radius + 1)):
            terrain[y, x] = 0


def _manhattan(position_a: Position, position_b: Position) -> int:
    return abs(position_a[0] - position_b[0]) + abs(position_a[1] - position_b[1])


def _choose_high_value_indices(
    patch_positions: Sequence[Sequence[Position]],
    n_high_value: int,
    rng: np.random.Generator,
) -> Set[int]:
    flat_index = 0
    patch_groups: List[List[int]] = []
    for patch in patch_positions:
        indices = list(range(flat_index, flat_index + len(patch)))
        patch_groups.append(indices)
        flat_index += len(patch)

    chosen: Set[int] = set()
    shuffled_groups = list(range(len(patch_groups)))
    rng.shuffle(shuffled_groups)
    for patch_id in shuffled_groups:
        if len(chosen) >= n_high_value:
            break
        indices = patch_groups[patch_id]
        chosen.add(int(rng.choice(indices)))

    all_indices = list(range(flat_index))
    rng.shuffle(all_indices)
    for index in all_indices:
        if len(chosen) >= n_high_value:
            break
        chosen.add(int(index))
    return chosen
