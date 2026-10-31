"""Default output directories for the generated figures and tables, and a figure-copy helper."""

from pathlib import Path
from shutil import copy2
from typing import Optional


# Outputs are written under <repo>/manuscript_figures/{figures,tables}.
MODULE_DIR = Path(__file__).resolve().parent
_OUTPUT_ROOT = MODULE_DIR.parent / "manuscript_figures"
DEFAULT_FIGURES_DIR = _OUTPUT_ROOT / "figures"
DEFAULT_TABLES_DIR = _OUTPUT_ROOT / "tables"


def resolve_figures_dir(override: Optional[Path] = None) -> Path:
    """Return the figures output directory, honoring an optional override."""
    return Path(override) if override is not None else DEFAULT_FIGURES_DIR


def resolve_tables_dir(override: Optional[Path] = None) -> Path:
    """Return the tables output directory, honoring an optional override."""
    return Path(override) if override is not None else DEFAULT_TABLES_DIR


def copy_figure_to_path(path: Path, destination_path: Path) -> Path:
    """Copy one existing figure to an explicit destination path."""
    source_path = Path(path)
    resolved_destination = Path(destination_path)
    resolved_destination.parent.mkdir(parents=True, exist_ok=True)
    copy2(source_path, resolved_destination)
    return resolved_destination
