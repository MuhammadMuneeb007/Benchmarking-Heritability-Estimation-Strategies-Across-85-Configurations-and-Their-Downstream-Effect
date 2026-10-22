 
# -*- coding: utf-8 -*-

"""
FinalAnalysis17-CorrelationHeatmapWithCI.py
===========================================

Creates a journal-style inter-method FAMILY correlation heatmap with:

    * Pearson r in each cell
    * Fisher-z 95% confidence interval in each off-diagonal cell
    * BH-FDR significance marker (*) for q < 0.05
    * No footer text or caption embedded in the image

Primary reporting scale
-----------------------
Binary phenotypes -> liability scale
BMI               -> native quantitative scale

How family profiles are constructed
-----------------------------------
1. Select empirical results on the primary reporting scale.
2. Average configuration-level mean h2 within each:
       phenotype x method family
3. Correlate the resulting phenotype profiles pairwise across families.
4. Use pairwise-complete phenotypes for each family pair.
5. Calculate Fisher-z 95% CI for Pearson r:
       z = arctanh(r)
       SE_z = 1 / sqrt(N - 3)
       95% CI_z = z +/- 1.96 * SE_z
       transform back with tanh()
6. Apply Benjamini-Hochberg FDR correction across the 15 unique family pairs.

IMPORTANT
---------
The 95% CI shown in this figure is uncertainty in the CORRELATION COEFFICIENT.
It is NOT the estimator-specific CI of the individual h2 estimates.

Input
-----
FinalAnalysis06-UncertaintyReliability/
    FinalAnalysis06_unit_stability.csv

Run
---
cd /data/ascher01/uqmmune1/finalheritability
python FinalAnalysis17-CorrelationHeatmapWithCI.py

Output
------
FinalFigures-Heritability-Uncertainty/
    Figure_InterFamily_Correlation_With_CI.png
    Figure_InterFamily_Correlation_With_CI.pdf
    Figure_InterFamily_Correlation_With_CI_Statistics.csv
    Figure_InterFamily_Correlation_Matrix.csv
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# =============================================================================
# 1. PATHS
# =============================================================================

def locate_root() -> Path:
    env = os.environ.get("HERITABILITY_ROOT")

    if env:
        root = Path(env).expanduser().resolve()
        if (root / "FinalAnalysis06-UncertaintyReliability").exists():
            return root

    cwd = Path.cwd().resolve()

    candidates = [
        cwd,
        cwd.parent,
        Path("/data/ascher01/uqmmune1/finalheritability"),
    ]

    for root in candidates:
        if (root / "FinalAnalysis06-UncertaintyReliability").exists():
            return root

    raise FileNotFoundError(
        "Could not locate FinalAnalysis06-UncertaintyReliability. "
        "Run this script from finalheritability or set HERITABILITY_ROOT."
    )


ROOT = locate_root()
INPUT_DIR = ROOT / "FinalAnalysis06-UncertaintyReliability"
OUTPUT_DIR = ROOT / "FinalFigures-Heritability-Uncertainty"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNITS_FILE = INPUT_DIR / "FinalAnalysis06_unit_stability.csv"


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return path


# =============================================================================
# 2. FAMILY DEFINITIONS
# =============================================================================

FAMILY_ORDER = [
    "LDpred2",
    "GCTA",
    "GEMMA",
    "DPR + GEMMA",
    "LDSC",
    "LDAK",
]


def method_family(method: str) -> str:
    method = str(method)

    if method == "LDpred-2-Heritability":
        return "LDpred2"

    if method == "GCTA-Heritability":
        return "GCTA"

    if method in {
        "Gemma1-Heritability",
        "Gemma2-Heritability",
        "Gemma-LDSC-Heritability",
    }:
        return "GEMMA"

    if method == "DPR-Gemma-Heritability":
        return "DPR + GEMMA"

    if method in {
        "LDSC-Heritability",
        "LDSC-Genotype-Heritability",
    }:
        return "LDSC"

    if method in {
        "LDAK-Calculated-Heritability",
        "LDAK-Precomputed-Heritability",
    }:
        return "LDAK"

    return method


# =============================================================================
# 3. STATISTICAL HELPERS
# =============================================================================

def fisher_ci(r: float, n: int, alpha: float = 0.05):
    """
    Fisher-z confidence interval for Pearson r.

    Returns (lower, upper).
    For n <= 3, returns NaN.
    """
    if not np.isfinite(r) or n <= 3:
        return np.nan, np.nan

    # Avoid atanh(+/-1) overflow.
    r_clip = float(np.clip(r, -0.999999999, 0.999999999))

    z = np.arctanh(r_clip)
    se = 1.0 / np.sqrt(n - 3.0)

    zcrit = stats.norm.ppf(1.0 - alpha / 2.0)

    lo_z = z - zcrit * se
    hi_z = z + zcrit * se

    return float(np.tanh(lo_z)), float(np.tanh(hi_z))


def bh_adjust(pvalues):
    """
    Benjamini-Hochberg FDR adjustment.
    Returns q-values in original order.
    """
    p = np.asarray(pvalues, dtype=float)

    q = np.full_like(p, np.nan, dtype=float)

    ok = np.isfinite(p)
    pv = p[ok]

    if pv.size == 0:
        return q

    order = np.argsort(pv)
    ranked = pv[order]

    m = len(ranked)

    adjusted = ranked * m / np.arange(1, m + 1)

    # enforce monotonicity from largest to smallest rank
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    original_indices = np.where(ok)[0]
    q_indices = original_indices[order]
    q[q_indices] = adjusted

    return q


def fmt_num(x, digits=2):
    if not np.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


# =============================================================================
# 4. LOAD + PRIMARY SCALE
# =============================================================================

print("=" * 105)
print("INTER-METHOD FAMILY CORRELATION HEATMAP WITH 95% CI")
print("=" * 105)
print("ROOT   :", ROOT)
print("INPUT  :", INPUT_DIR)
print("OUTPUT :", OUTPUT_DIR)
print()

units = pd.read_csv(require(UNITS_FILE))

emp = units[
    units["Domain"].astype(str).str.lower() == "empirical"
].copy()

primary_mask = (
    (
        (emp["TraitType"].astype(str).str.lower() == "binary")
        & (emp["Scale"].astype(str) == "liability_binary")
    )
    |
    (
        (emp["TraitType"].astype(str).str.lower() == "continuous")
        & (emp["Scale"].astype(str) == "native_quantitative")
    )
)

emp = emp[primary_mask].copy()

emp["Phenotype"] = emp["DatasetLabel"].astype(str)
emp["MeanH2"] = pd.to_numeric(emp["MeanH2"], errors="coerce")
emp["Family"] = emp["Method"].map(method_family)


# =============================================================================
# 5. BUILD FAMILY-LEVEL PHENOTYPE PROFILES
# =============================================================================
#
# One value per phenotype x method family:
# mean of configuration-level MeanH2 estimates within that family.
# =============================================================================

profile = (
    emp.groupby(
        ["Phenotype", "Family"],
        dropna=False,
    )["MeanH2"]
    .mean()
    .reset_index()
)

families_present = list(profile["Family"].dropna().unique())

family_order = [
    f for f in FAMILY_ORDER
    if f in families_present
]

family_order += [
    f for f in families_present
    if f not in family_order
]

wide = (
    profile.pivot(
        index="Phenotype",
        columns="Family",
        values="MeanH2",
    )
    .reindex(columns=family_order)
)

print("Phenotypes represented:", len(wide))
print("Families:", ", ".join(family_order))
print()


# =============================================================================
# 6. PAIRWISE PEARSON CORRELATION + FISHER-Z CI
# =============================================================================

pair_rows = []

for family_a, family_b in combinations(family_order, 2):

    x = pd.to_numeric(
        wide[family_a],
        errors="coerce",
    )

    y = pd.to_numeric(
        wide[family_b],
        errors="coerce",
    )

    valid = x.notna() & y.notna()

    xv = x[valid].to_numpy(dtype=float)
    yv = y[valid].to_numpy(dtype=float)

    n = len(xv)

    if n >= 2 and np.std(xv, ddof=1) > 0 and np.std(yv, ddof=1) > 0:
        r, p = stats.pearsonr(xv, yv)
    else:
        r, p = np.nan, np.nan

    ci_lo, ci_hi = fisher_ci(r, n)

    pair_rows.append(
        {
            "Family1": family_a,
            "Family2": family_b,
            "N": n,
            "Pearson_r": r,
            "CI95_Lower": ci_lo,
            "CI95_Upper": ci_hi,
            "P": p,
        }
    )


stats_df = pd.DataFrame(pair_rows)

stats_df["Q_BH"] = bh_adjust(
    stats_df["P"].to_numpy(dtype=float)
)

stats_df["FDR_Significant"] = (
    stats_df["Q_BH"] < 0.05
).astype(int)


# =============================================================================
# 7. BUILD MATRICES
# =============================================================================

n_fam = len(family_order)

r_matrix = pd.DataFrame(
    np.eye(n_fam),
    index=family_order,
    columns=family_order,
    dtype=float,
)

lo_matrix = pd.DataFrame(
    np.nan,
    index=family_order,
    columns=family_order,
    dtype=float,
)

hi_matrix = pd.DataFrame(
    np.nan,
    index=family_order,
    columns=family_order,
    dtype=float,
)

q_matrix = pd.DataFrame(
    np.nan,
    index=family_order,
    columns=family_order,
    dtype=float,
)

n_matrix = pd.DataFrame(
    np.nan,
    index=family_order,
    columns=family_order,
    dtype=float,
)

for _, row in stats_df.iterrows():

    a = row["Family1"]
    b = row["Family2"]

    r_matrix.loc[a, b] = row["Pearson_r"]
    r_matrix.loc[b, a] = row["Pearson_r"]

    lo_matrix.loc[a, b] = row["CI95_Lower"]
    lo_matrix.loc[b, a] = row["CI95_Lower"]

    hi_matrix.loc[a, b] = row["CI95_Upper"]
    hi_matrix.loc[b, a] = row["CI95_Upper"]

    q_matrix.loc[a, b] = row["Q_BH"]
    q_matrix.loc[b, a] = row["Q_BH"]

    n_matrix.loc[a, b] = row["N"]
    n_matrix.loc[b, a] = row["N"]


# =============================================================================
# 8. SAVE STATISTICS
# =============================================================================

stats_df.to_csv(
    OUTPUT_DIR / "Figure_InterFamily_Correlation_With_CI_Statistics.csv",
    index=False,
)

r_matrix.to_csv(
    OUTPUT_DIR / "Figure_InterFamily_Correlation_Matrix.csv",
)


# =============================================================================
# 9. DRAW HEATMAP
# =============================================================================

fig, ax = plt.subplots(
    figsize=(10.5, 9.0)
)

values = r_matrix.to_numpy(dtype=float)

cmap = plt.get_cmap("coolwarm")

im = ax.imshow(
    values,
    cmap=cmap,
    vmin=-1,
    vmax=1,
    aspect="equal",
    interpolation="nearest",
)

# Major ticks
ax.set_xticks(np.arange(n_fam))
ax.set_yticks(np.arange(n_fam))

ax.set_xticklabels(
    family_order,
    fontsize=10,
)

ax.set_yticklabels(
    family_order,
    fontsize=10,
)

plt.setp(
    ax.get_xticklabels(),
    rotation=40,
    ha="right",
    rotation_mode="anchor",
)

# Thin white cell boundaries
ax.set_xticks(
    np.arange(-0.5, n_fam, 1),
    minor=True,
)

ax.set_yticks(
    np.arange(-0.5, n_fam, 1),
    minor=True,
)

ax.grid(
    which="minor",
    color="white",
    linewidth=1.5,
)

ax.tick_params(
    which="minor",
    bottom=False,
    left=False,
)

# Clean outer frame
for spine in ax.spines.values():
    spine.set_visible(False)


# =============================================================================
# 10. CELL ANNOTATION
# =============================================================================
#
# Diagonal:
#    1.00
#
# Off diagonal:
#    0.84*
#    [0.29, 0.97]
#
# * = BH-FDR q < 0.05
# =============================================================================

for i, row_family in enumerate(family_order):

    for j, col_family in enumerate(family_order):

        r = r_matrix.iloc[i, j]

        if i == j:
            label = "1.00"
        else:
            lo = lo_matrix.iloc[i, j]
            hi = hi_matrix.iloc[i, j]
            q = q_matrix.iloc[i, j]

            star = "*" if np.isfinite(q) and q < 0.05 else ""

            if np.isfinite(r):
                first = f"{r:.2f}{star}"
            else:
                first = "NA"

            if np.isfinite(lo) and np.isfinite(hi):
                second = f"[{lo:.2f}, {hi:.2f}]"
            else:
                second = "[NA]"

            label = first + "\n" + second

        # Text contrast based on cell value.
        if np.isfinite(r):
            text_color = "white" if abs(r) >= 0.55 else "black"
        else:
            text_color = "black"

        ax.text(
            j,
            i,
            label,
            ha="center",
            va="center",
            fontsize=9.0 if i != j else 9.5,
            color=text_color,
            fontweight="bold" if i == j else "normal",
            linespacing=1.15,
        )


# =============================================================================
# 11. TITLE + COLORBAR
# =============================================================================

ax.set_title(
    "Inter-method family correlation and 95% confidence intervals",
    fontsize=14,
    fontweight="bold",
    pad=16,
)

cbar = fig.colorbar(
    im,
    ax=ax,
    fraction=0.046,
    pad=0.04,
)

cbar.set_label(
    "Pearson correlation (r)",
    rotation=270,
    labelpad=18,
    fontsize=10,
)

cbar.ax.tick_params(
    labelsize=9,
)

fig.tight_layout()


# =============================================================================
# 12. SAVE
# =============================================================================

fig.savefig(
    OUTPUT_DIR / "Figure_InterFamily_Correlation_With_CI.png",
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    OUTPUT_DIR / "Figure_InterFamily_Correlation_With_CI.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# 13. PRINT AUDIT
# =============================================================================

print("=" * 105)
print("PAIRWISE FAMILY CORRELATIONS")
print("=" * 105)

display_cols = [
    "Family1",
    "Family2",
    "N",
    "Pearson_r",
    "CI95_Lower",
    "CI95_Upper",
    "P",
    "Q_BH",
    "FDR_Significant",
]

with pd.option_context(
    "display.max_rows", 100,
    "display.max_columns", 20,
    "display.width", 180,
):
    print(
        stats_df[display_cols]
        .sort_values("Pearson_r", ascending=False)
        .to_string(index=False)
    )

print()
print("=" * 105)
print("DONE")
print("=" * 105)
print()
print("Generated:")
print(" ", OUTPUT_DIR / "Figure_InterFamily_Correlation_With_CI.png")
print(" ", OUTPUT_DIR / "Figure_InterFamily_Correlation_With_CI.pdf")
print(" ", OUTPUT_DIR / "Figure_InterFamily_Correlation_With_CI_Statistics.csv")
print(" ", OUTPUT_DIR / "Figure_InterFamily_Correlation_Matrix.csv")
print()
print("Figure convention:")
print(" * Cell colour = Pearson r")
print(" * Cell first line = Pearson r")
print(" * Cell second line = Fisher-z 95% CI")
print(" * * = BH-FDR q < 0.05")
print(" * Binary phenotypes = liability scale")
print(" * BMI = native quantitative scale")
print()
 
