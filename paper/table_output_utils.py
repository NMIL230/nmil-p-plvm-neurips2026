"""Helpers for writing the numbered table outputs (Table_N.csv and Table_N.tex)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


_PLUS_MINUS_TOKEN = "PLUSSIGNMINUSPLACEHOLDER"
_LATEX_ESCAPES = (
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
)


def resolve_table_output_paths(table_stem: str, tables_dir: Path) -> Tuple[Path, Path]:
    """Return the CSV and TeX output paths for one numbered table."""
    resolved_tables_dir = Path(tables_dir)
    return (
        resolved_tables_dir / "{}.csv".format(str(table_stem)),
        resolved_tables_dir / "{}.tex".format(str(table_stem)),
    )


def default_column_alignment(
    table_frame: pd.DataFrame,
    *,
    leading_left_columns: int = 1,
) -> str:
    """Return a simple LaTeX column-alignment string for one table frame."""
    num_columns = len(list(table_frame.columns))
    if num_columns <= 0:
        return ""
    left_count = max(0, min(int(leading_left_columns), num_columns))
    return ("l" * left_count) + ("c" * (num_columns - left_count))


def format_latex_cell(value: object) -> str:
    """Format one scalar cell for a compact LaTeX tabular."""
    if pd.isna(value):
        return ""
    text = str(value).replace("±", _PLUS_MINUS_TOKEN)
    for raw_token, escaped_token in _LATEX_ESCAPES:
        text = text.replace(raw_token, escaped_token)
    return text.replace(_PLUS_MINUS_TOKEN, r"$\pm$")


def write_latex_table(
    table_frame: pd.DataFrame,
    output_path: Path,
    *,
    column_alignment: Optional[str] = None,
    leading_left_columns: int = 1,
) -> None:
    """Write one compact LaTeX tabular using booktabs-style rules."""
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_column_alignment = column_alignment or default_column_alignment(
        table_frame,
        leading_left_columns=leading_left_columns,
    )
    columns = [str(column) for column in table_frame.columns]
    with resolved_output_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\\begin{{tabular}}{{{}}}\n".format(resolved_column_alignment))
        handle.write("\\toprule\n")
        handle.write(
            " {} \\\\\n".format(" & ".join(format_latex_cell(column) for column in columns))
        )
        handle.write("\\midrule\n")
        for _, row in table_frame.iterrows():
            handle.write(
                " {} \\\\\n".format(
                    " & ".join(format_latex_cell(row[column]) for column in columns)
                )
            )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")
