#!/usr/bin/env python3
"""
table14_feature_ladder_latex.py
===============================
Generates the Table 14 LaTeX from the ``full`` and ``shuffled`` runs of
``table14_feature_ladder.py`` (run both first; see its docstring).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper.output_paths import resolve_tables_dir  # noqa: E402

DEFAULT_DIR = resolve_tables_dir() / "table14_feature_ladder"

ROWS = [
    ("acc_majority", "Majority class", "--", "main"),
    ("acc_O", "Outcome", "completion, \\#subtasks", "main"),
    ("acc_P-static", "\\quad + zone proportions", "order-free spatial", "main"),
    ("acc_P-order", "\\quad + navigation statistics", "order-dependent", "main"),
    ("acc_P-order", "\\quad\\quad \\emph{same, prefix shuffled}", "ordering removed", "shuf"),
    ("acc_P-learned", "PLVM (frozen encoders + fuser)", "learned", "main"),
]


def fmt(frame: pd.DataFrame, col: str, cutoff: int) -> str:
    """Format one ladder cell as LaTeX ``mean \\pm SD`` (percent, over folds)."""
    sub = frame[frame["cutoff_seconds"] == cutoff][col].dropna()
    if sub.empty:
        return "$--$"
    return f"${100 * sub.mean():.1f} \\pm {100 * sub.std(ddof=1):.1f}$"


def main(argv=None) -> int:
    """Render Table 14 from the ``full`` and ``shuffled`` ladder runs."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_DIR,
        help="Directory holding navstyle_ladder_by_fold_{full,shuffled}.csv.",
    )
    ap.add_argument(
        "--cutoffs_minutes",
        default="20,60,120",
        help="Comma-separated table columns in minutes.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .tex path (default: <dir>/navstyle_ladder_table.tex).",
    )
    args = ap.parse_args(argv)

    main_df = pd.read_csv(args.dir / "navstyle_ladder_by_fold_full.csv")
    shuf_df = pd.read_csv(args.dir / "navstyle_ladder_by_fold_shuffled.csv")
    frames = {"main": main_df, "shuf": shuf_df}
    cutoffs = [int(float(c) * 60) for c in args.cutoffs_minutes.split(",")]

    lines = [
        r"\begin{tabular}{ll" + "c" * len(cutoffs) + "}",
        r"\toprule",
        r"Predictor & Information used & " + " & ".join(f"{c // 60} min" for c in cutoffs) + r" \\",
        r"\midrule",
    ]
    for col, label, info, which in ROWS:
        cells = " & ".join(fmt(frames[which], col, c) for c in cutoffs)
        lines.append(f"{label} & {info} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(lines)

    out = args.out or (args.dir / "navstyle_ladder_table.tex")
    out.write_text(tex + "\n")
    print(tex)

    # ---- console only: ordering ablation and hand-crafted vs PLVM tests ----
    print("\n% ---- paired per-fold tests ----")
    for cutoff in cutoffs:
        m = main_df[main_df["cutoff_seconds"] == cutoff]
        s = shuf_df[shuf_df["cutoff_seconds"] == cutoff]
        merged = m.merge(s, on="fold", suffixes=("", "_shuf"))
        pairs = [
            (
                "ordering ablation: P-order vs shuffled",
                merged["acc_P-order"],
                merged["acc_P-order_shuf"],
            ),
            ("hand-crafted vs PLVM: P-order vs P-learned", m["acc_P-order"], m["acc_P-learned"]),
        ]
        for name, a, b in pairs:
            a, b = np.asarray(a, float), np.asarray(b, float)
            ok = np.isfinite(a) & np.isfinite(b)
            try:
                p = float(stats.wilcoxon(a[ok], b[ok]).pvalue)
            except ValueError:
                p = float("nan")
            print(
                f"%  {cutoff // 60:3d} min  {name:45s} "
                f"{100 * a[ok].mean():5.1f} vs {100 * b[ok].mean():5.1f}  "
                f"diff {100 * (a[ok] - b[ok]).mean():+5.1f} pp  p={p:.4f}"
            )
    print(f"\n% wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
