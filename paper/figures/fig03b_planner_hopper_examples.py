#!/usr/bin/env python3
"""Generate Figure 3b: one Zone Planner and one Zone Hopper Fire Station route.

Reads the fold-00 zone-label artifacts. The planner is the most extreme Zone
Planner; the hopper is ``p3738`` (``DEFAULT_HOPPER_ID``) when it is a labelled
Zone Hopper, otherwise the most extreme hopper. ``p3738`` is preferred because
its completions are spread over distinct times; the most extreme hopper may
finish many subtasks in the same second, whose order is not recorded, so its
plotted route is unreliable.
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper.output_paths import resolve_figures_dir
from plvm import artifact_paths
from powerwash.data.create_firestation_zone_planning_labels import (
    DEFAULT_TASK,
    _artifact_prefix,
    _plot_planner_vs_hopper_zone_switching,
    _select_extreme_trajectory_players,
    maybe_set_plot_style,
)


DEFAULT_ARTIFACT_DIR = artifact_paths.FIRESTATION_LABELS_DIR / "fold_00"
DEFAULT_PLAYER_RECORDS_CACHE = (
    artifact_paths.FIRESTATION_LABELS_DIR / ".cache" / "firestation_player_records.pkl"
)
DEFAULT_OUTPUT_PATH = resolve_figures_dir() / "Figure_3b.pdf"
DEFAULT_HOPPER_ID = "p3738"
PLANNER_LABEL = "Zone Planner"
HOPPER_LABEL = "Zone Hopper"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for Figure 3b."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact_dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory containing Fire Station zone-label CSV artifacts.",
    )
    parser.add_argument(
        "--player_records_cache",
        type=Path,
        default=DEFAULT_PLAYER_RECORDS_CACHE,
        help="Pickle cache containing Fire Station player records.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional directory override for the Figure 3b output.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Optional explicit PDF path. Defaults to manuscript_figures/figures/Figure_3b.pdf.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=DEFAULT_TASK,
        help="Task name; it only sets the name of the intermediate PDF, which is then moved to "
        "the output path.",
    )
    parser.add_argument(
        "--planner_id",
        type=str,
        default=None,
        help="Zone Planner to draw (default: the most extreme planner).",
    )
    parser.add_argument(
        "--hopper_id",
        type=str,
        default=None,
        help="Zone Hopper to draw (default: {} if labelled a hopper, else the most "
        "extreme hopper).".format(DEFAULT_HOPPER_ID),
    )
    parser.add_argument(
        "--max_annotations",
        type=int,
        default=5,
        help="Maximum number of route-step numbers to annotate per panel.",
    )
    return parser.parse_args(argv)


def resolve_output_path(
    output_dir: Optional[Path],
    output_path: Optional[Path],
) -> Path:
    """Resolve the output path for Figure 3b."""
    if output_path is not None:
        return Path(output_path)
    return resolve_figures_dir(output_dir) / DEFAULT_OUTPUT_PATH.name


def load_firestation_plot_inputs(
    artifact_dir: Path,
    player_records_cache: Path,
) -> Tuple[pd.DataFrame, Dict[str, dict], pd.DataFrame]:
    """Load the labeled rows, cached player records, and zone map for plotting."""
    resolved_artifact_dir = Path(artifact_dir).expanduser().resolve()
    labels_path = resolved_artifact_dir / "firestation_zone_labels.csv"
    zone_map_path = resolved_artifact_dir / "firestation_subtask_zone_map.csv"
    cache_path = Path(player_records_cache).expanduser().resolve()

    missing_paths = [path for path in (labels_path, zone_map_path, cache_path) if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing Fire Station planner/hopper inputs: {}.".format(
                ", ".join(str(path) for path in missing_paths)
            )
        )

    labeled_df = pd.read_csv(labels_path)
    df_map = pd.read_csv(zone_map_path)
    with cache_path.open("rb") as handle:
        player_records = pickle.load(handle)

    required_label_columns = {"player_id", "label", "zone_hopping_score"}
    missing_label_columns = sorted(required_label_columns.difference(labeled_df.columns))
    if missing_label_columns:
        raise ValueError(
            "Missing required Fire Station label columns: {}.".format(
                ", ".join(missing_label_columns)
            )
        )

    required_map_columns = {"subtask_name", "x", "z", "zone_id"}
    missing_map_columns = sorted(required_map_columns.difference(df_map.columns))
    if missing_map_columns:
        raise ValueError(
            "Missing required Fire Station zone-map columns: {}.".format(
                ", ".join(missing_map_columns)
            )
        )

    if not isinstance(player_records, dict):
        raise TypeError("Expected cached player records to be a dict.")

    return labeled_df, player_records, df_map


def select_example_players(
    labeled_df: pd.DataFrame,
    planner_id: Optional[str] = None,
    hopper_id: Optional[str] = None,
) -> list:
    """Return ``[(label, player_id), ...]`` for the planner and hopper panels.

    An explicitly requested id must be a labelled player of that class. Without one, the
    planner is the most extreme planner and the hopper is ``DEFAULT_HOPPER_ID`` when it is a
    labelled hopper, otherwise the most extreme hopper.
    """
    ids_by_label = {
        label: set(labeled_df.loc[labeled_df["label"] == label, "player_id"].astype(str))
        for label in (PLANNER_LABEL, HOPPER_LABEL)
    }
    extreme = dict(_select_extreme_trajectory_players(labeled_df))
    selected = []
    for label, requested in ((PLANNER_LABEL, planner_id), (HOPPER_LABEL, hopper_id)):
        if requested is not None:
            if requested not in ids_by_label[label]:
                raise ValueError("{} is not a labelled {}.".format(requested, label))
            selected.append((label, requested))
        elif label == HOPPER_LABEL and DEFAULT_HOPPER_ID in ids_by_label[label]:
            selected.append((label, DEFAULT_HOPPER_ID))
        elif label in extreme:
            if label == HOPPER_LABEL:
                print(
                    "{} is not a labelled hopper; using the most extreme hopper {}.".format(
                        DEFAULT_HOPPER_ID, extreme[label]
                    )
                )
            selected.append((label, extreme[label]))
    return selected


def render_firestation_planner_hopper_figure(
    labeled_df: pd.DataFrame,
    player_records: Dict[str, dict],
    df_map: pd.DataFrame,
    output_path: Path,
    *,
    task_name: str,
    max_annotations: int,
    selected_players=None,
) -> Path:
    """Render the Fire Station planner-vs-hopper figure (Figure 3b) to one PDF."""
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    maybe_set_plot_style()
    _plot_planner_vs_hopper_zone_switching(
        labeled_df,
        player_records,
        df_map,
        resolved_output_path.parent,
        task_name=task_name,
        max_annotations=max_annotations,
        selected_players=selected_players,
    )
    generated_path = resolved_output_path.parent / (
        "{}_planner_vs_hopper_zone_switching.pdf".format(_artifact_prefix(task_name))
    )
    if generated_path != resolved_output_path:
        generated_path.replace(resolved_output_path)
    return resolved_output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Generate Figure 3b, the Fire Station Planner/Hopper figure."""
    args = parse_args(argv)
    resolved_output_path = resolve_output_path(args.output_dir, args.output_path)
    labeled_df, player_records, df_map = load_firestation_plot_inputs(
        args.artifact_dir,
        args.player_records_cache,
    )
    render_firestation_planner_hopper_figure(
        labeled_df,
        player_records,
        df_map,
        resolved_output_path,
        task_name=args.task_name,
        max_annotations=args.max_annotations,
        selected_players=select_example_players(labeled_df, args.planner_id, args.hopper_id),
    )
    print("Wrote {}".format(resolved_output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
