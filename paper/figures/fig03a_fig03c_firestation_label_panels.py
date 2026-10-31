#!/usr/bin/env python3
"""Generate Figures 3a and 3c from saved Fire Station label artifacts."""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper.navigation_style_panels_common import (
    DEFAULT_FIRE_STATION_ARTIFACT_DIR,
    load_fire_station_artifacts,
    load_label_thresholds,
    render_fire_station_zones,
    render_navigation_style_score_distribution,
    resolve_fire_station_output_paths,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for Figures 3a and 3c."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fire_station_artifact_dir",
        type=Path,
        default=DEFAULT_FIRE_STATION_ARTIFACT_DIR,
        help="Directory containing Fire Station zone-label CSV artifacts.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional directory override for both figure outputs.",
    )
    parser.add_argument(
        "--figure_3a_pdf",
        type=Path,
        default=None,
        help="Optional explicit output path for Figure 3a.",
    )
    parser.add_argument(
        "--figure_3c_pdf",
        type=Path,
        default=None,
        help="Optional explicit output path for Figure 3c.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Generate Figures 3a and 3c."""
    args = parse_args(argv)
    figure_3a_path, figure_3c_path = resolve_fire_station_output_paths(
        output_dir=args.output_dir,
        figure_3a_pdf=args.figure_3a_pdf,
        figure_3c_pdf=args.figure_3c_pdf,
    )
    labeled_df, df_map, zone_centroids = load_fire_station_artifacts(args.fire_station_artifact_dir)
    render_fire_station_zones(df_map, zone_centroids, figure_3a_path)
    render_navigation_style_score_distribution(
        labeled_df,
        figure_3c_path,
        thresholds=load_label_thresholds(args.fire_station_artifact_dir),
    )
    print("Wrote {}".format(figure_3a_path))
    print("Wrote {}".format(figure_3c_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
