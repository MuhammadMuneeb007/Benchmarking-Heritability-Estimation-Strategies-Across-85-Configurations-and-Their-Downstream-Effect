#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinalAnalysis12-FigureHeatmap.py
================================

Creates two reviewer-facing heatmaps from the existing uncertainty-analysis outputs:

1) Figure_H2_Heatmap.png / .pdf
   - Main manuscript point-estimate heatmap
   - Binary phenotypes: liability scale
   - BMI: native quantitative scale
   - EACH CELL IS ANNOTATED WITH THE NUMERIC h2 VALUE
   - Negative and >1 h2 values retained

2) Figure_Estimator_CIWidth_Heatmap.png / .pdf
   - Supplementary uncertainty heatmap
   - Same row/column layout as the h2 heatmap
   - Cell colour = estimator-specific 95% CI width
   - Missing estimator-specific CI values remain missing/grey
   - CI width is shown on a logarithmic colour scale
   - CI-width cells are also annotated where values are available

Also writes the exact matrices used to CSV.

IMPORTANT SCIENTIFIC INTERPRETATION
-----------------------------------
* Binary phenotypes are plotted on the LIABILITY SCALE.
* BMI is plotted on its NATIVE QUANTITATIVE SCALE.
* Estimator-specific SE / CI comes from the heritability-estimation output where available.
* Cross-fold SD is a descriptive stability measure across overlapping CV folds.
* Cross-fold SD is NOT used as an inferential standard error or CI.
* The configuration-level CI summary uses the median fold-level lower bound and
  median fold-level upper bound, matching the existing uncertainty workflow.
* The CI-width figure therefore visualises estimator-specific uncertainty, not
  uncertainty of the across-fold mean h2.
* Missing CI values are never imputed.
* Negative and >1 h2 values are retained without clipping.

EXPECTED INPUT DIRECTORY
------------------------
FinalAnalysis06-UncertaintyReliability/

Required files:
    FinalAnalysis06_unit_stability.csv
    FinalAnalysis06_raw_scale_rows.csv

RUN
---
From:
    /data/ascher01/uqmmune1/finalheritability

run:
    python FinalAnalysis12-FigureHeatmap.py

OUTPUT
------
FinalFigures-Heritability-Uncertainty/
    Figure_H2_Heatmap.png
    Figure_H2_Heatmap.pdf
    Figure_Estimator_CIWidth_Heatmap.png
    Figure_Estimator_CIWidth_Heatmap.pdf
    Figure_H2_Heatmap_Data.csv
    Figure_Estimator_CIWidth_Heatmap_Data.csv
    Figure_Configuration_Order.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm


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
        "Run this script from the finalheritability directory or set "
        "HERITABILITY_ROOT to the project root."
    )


ROOT = locate_root()
INPUT_DIR = ROOT / "FinalAnalysis06-UncertaintyReliability"
OUTPUT_DIR = ROOT / "FinalFigures-Heritability-Uncertainty"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNITS_FILE = INPUT_DIR / "FinalAnalysis06_unit_stability.csv"
RAW_FILE = INPUT_DIR / "FinalAnalysis06_raw_scale_rows.csv"


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return path


# =============================================================================
# 2. PLOT OPTIONS
# =============================================================================

# h2 point estimates are the main figure, so show the number in every finite cell.
SHOW_H2_VALUES = True

# Keep CI-width values visible too. Set to False if the supplementary figure feels too busy.
SHOW_CI_VALUES = True

# Three decimals is a good compromise for h2 while remaining readable.
H2_DECIMALS = 3

# Because there are ~85 configuration rows, use a large canvas.
H2_FIGSIZE = (15, 24)
CI_FIGSIZE = (15, 24)

H2_TEXT_SIZE = 4.8
CI_TEXT_SIZE = 4.3


# =============================================================================
# 3. PHENOTYPE / METHOD ORDER
# =============================================================================

PHENOTYPE_ORDER = [
    "Asthma",
    "Body Mass Index",
    "Depression",
    "Gastro-Oesophageal Reflux",
    "High Cholesterol",
    "Hypertension",
    "Hypothyroidism",
    "Irritable Bowel Syndrome",
    "Migraine",
    "Osteoarthritis",
]

PHENOTYPE_LABELS = {
    "Asthma": "Asthma",
    "Body Mass Index": "BMI",
    "Depression": "Depression",
    "Gastro-Oesophageal Reflux": "GORD",
    "High Cholesterol": "High\ncholesterol",
    "Hypertension": "Hypertension",
    "Hypothyroidism": "Hypothyroidism",
    "Irritable Bowel Syndrome": "IBS",
    "Migraine": "Migraine",
    "Osteoarthritis": "Osteoarthritis",
}

METHOD_ORDER = [
    "LDpred-2-Heritability",
    "GCTA-Heritability",
    "Gemma1-Heritability",
    "Gemma2-Heritability",
    "Gemma-LDSC-Heritability",
    "DPR-Gemma-Heritability",
    "LDSC-Heritability",
    "LDAK-Calculated-Heritability",
    "LDAK-Precomputed-Heritability",
    "LDSC-Genotype-Heritability",
]

METHOD_SHORT = {
    "LDpred-2-Heritability": "LDpred2",
    "GCTA-Heritability": "GCTA",
    "Gemma1-Heritability": "GEMMA1",
    "Gemma2-Heritability": "GEMMA2",
    "Gemma-LDSC-Heritability": "GEMMA-LDSC",
    "DPR-Gemma-Heritability": "DPR-GEMMA",
    "LDSC-Heritability": "LDSC",
    "LDAK-Calculated-Heritability": "LDAK-Calculated",
    "LDAK-Precomputed-Heritability": "LDAK-Precomputed",
    "LDSC-Genotype-Heritability": "LDSC-Genotype",
}


# =============================================================================
# 4. LOAD INPUTS
# =============================================================================

print("=" * 100)
print("HERITABILITY POINT-ESTIMATE + UNCERTAINTY HEATMAPS")
print("=" * 100)
print("ROOT:", ROOT)
print("INPUT:", INPUT_DIR)
print("OUTPUT:", OUTPUT_DIR)
print()

units = pd.read_csv(require(UNITS_FILE))
raw = pd.read_csv(require(RAW_FILE))

print("Unit-stability rows:", len(units))
print("Raw-scale rows:", len(raw))


# =============================================================================
# 5. PRIMARY EMPIRICAL SCALE
# =============================================================================
# Binary phenotypes -> liability scale
# BMI               -> native quantitative scale
# =============================================================================

u = units[units["Domain"].astype(str).str.lower() == "empirical"].copy()

primary_mask = (
    (
        (u["TraitType"].astype(str).str.lower() == "binary")
        & (u["Scale"].astype(str) == "liability_binary")
    )
    |
    (
        (u["TraitType"].astype(str).str.lower() == "continuous")
        & (u["Scale"].astype(str) == "native_quantitative")
    )
)

u = u[primary_mask].copy()
u["Phenotype"] = u["DatasetLabel"].astype(str)
u["MeanH2"] = pd.to_numeric(u["MeanH2"], errors="coerce")


# =============================================================================
# 6. ESTIMATOR-SPECIFIC UNCERTAINTY
# =============================================================================

r = raw[raw["Domain"].astype(str).str.lower() == "empirical"].copy()

primary_raw_mask = (
    (
        (r["TraitType"].astype(str).str.lower() == "binary")
        & (r["Scale"].astype(str) == "liability_binary")
    )
    |
    (
        (r["TraitType"].astype(str).str.lower() == "continuous")
        & (r["Scale"].astype(str) == "native_quantitative")
    )
)

r = r[primary_raw_mask].copy()
r["Phenotype"] = r["DatasetLabel"].astype(str)

for col in ["ToolSE", "ToolCILower", "ToolCIUpper"]:
    if col not in r.columns:
        r[col] = np.nan
    r[col] = pd.to_numeric(r[col], errors="coerce")

uncertainty = (
    r.groupby(
        ["Phenotype", "Method", "ConfigID", "Scale"],
        dropna=False,
    )
    .agg(
        EstimatorSE=("ToolSE", "median"),
        EstimatorCILower=("ToolCILower", "median"),
        EstimatorCIUpper=("ToolCIUpper", "median"),
        EstimatorSEAvailableFolds=("ToolSE", lambda z: int(z.notna().sum())),
        EstimatorCIAvailableFolds=("ToolCILower", lambda z: int(z.notna().sum())),
    )
    .reset_index()
)

uncertainty["EstimatorCIWidth"] = (
    pd.to_numeric(uncertainty["EstimatorCIUpper"], errors="coerce")
    - pd.to_numeric(uncertainty["EstimatorCILower"], errors="coerce")
)

u = u.merge(
    uncertainty,
    on=["Phenotype", "Method", "ConfigID", "Scale"],
    how="left",
)


# =============================================================================
# 7. CONFIGURATION ORDER
# =============================================================================

configurations = (
    u[["Method", "ConfigID"]]
    .drop_duplicates()
    .copy()
)

configurations["MethodOrder"] = pd.Categorical(
    configurations["Method"],
    categories=METHOD_ORDER,
    ordered=True,
)

unexpected = configurations["MethodOrder"].isna()
if unexpected.any():
    unexpected_methods = sorted(
        configurations.loc[unexpected, "Method"].dropna().unique()
    )
    print()
    print("WARNING: methods not present in METHOD_ORDER:")
    for method in unexpected_methods:
        print("  ", method)

configurations = (
    configurations
    .sort_values(
        ["MethodOrder", "Method", "ConfigID"],
        kind="stable",
        na_position="last",
    )
    .drop(columns="MethodOrder")
    .reset_index(drop=True)
)

configurations["RowID"] = np.arange(len(configurations), dtype=int)

u = u.merge(
    configurations,
    on=["Method", "ConfigID"],
    how="left",
)

print()
print("Unique configurations:", len(configurations))

configurations.to_csv(
    OUTPUT_DIR / "Figure_Configuration_Order.csv",
    index=False,
)


# =============================================================================
# 8. METHOD BLOCKS FOR Y-AXIS
# =============================================================================

method_blocks = []

methods_in_plot = list(configurations["Method"].dropna().unique())

ordered_methods = [m for m in METHOD_ORDER if m in methods_in_plot]
ordered_methods += [m for m in methods_in_plot if m not in ordered_methods]

for method in ordered_methods:
    rows = configurations.index[
        configurations["Method"] == method
    ].tolist()

    if not rows:
        continue

    start = min(rows)
    end = max(rows)

    method_blocks.append(
        {
            "method": method,
            "start": start,
            "end": end,
            "midpoint": (start + end) / 2.0,
        }
    )


# =============================================================================
# 9. BUILD MATRICES
# =============================================================================

h2_matrix = (
    u.pivot_table(
        index="RowID",
        columns="Phenotype",
        values="MeanH2",
        aggfunc="first",
    )
    .reindex(
        index=range(len(configurations)),
        columns=PHENOTYPE_ORDER,
    )
)

ci_matrix = (
    u.pivot_table(
        index="RowID",
        columns="Phenotype",
        values="EstimatorCIWidth",
        aggfunc="first",
    )
    .reindex(
        index=range(len(configurations)),
        columns=PHENOTYPE_ORDER,
    )
)

h2_matrix.to_csv(
    OUTPUT_DIR / "Figure_H2_Heatmap_Data.csv",
    index=True,
)

ci_matrix.to_csv(
    OUTPUT_DIR / "Figure_Estimator_CIWidth_Heatmap_Data.csv",
    index=True,
)


# =============================================================================
# 10. AXIS FORMATTING
# =============================================================================

def format_configuration_axis(ax):
    ax.set_xticks(np.arange(len(PHENOTYPE_ORDER)))
    ax.set_xticklabels(
        [PHENOTYPE_LABELS[x] for x in PHENOTYPE_ORDER],
        rotation=40,
        ha="right",
        fontsize=10,
    )

    ax.set_yticks([b["midpoint"] for b in method_blocks])
    ax.set_yticklabels(
        [METHOD_SHORT.get(b["method"], b["method"]) for b in method_blocks],
        fontsize=9,
    )

    for block in method_blocks[:-1]:
        ax.axhline(
            block["end"] + 0.5,
            linewidth=0.9,
            color="black",
            alpha=0.75,
        )

    # Thin white grid to make individual cells easier to read.
    ax.set_xticks(
        np.arange(-0.5, len(PHENOTYPE_ORDER), 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, len(configurations), 1),
        minor=True,
    )

    ax.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=0.25,
        alpha=0.35,
    )

    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_ylabel(
        "Heritability-estimation configurations",
        fontsize=11,
    )

    ax.set_xlabel(
        "Phenotype",
        fontsize=11,
    )


def luminance(rgb):
    """Simple perceived luminance for choosing black/white text."""
    r, g, b = rgb[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def annotate_matrix(
    ax,
    values,
    image,
    formatter,
    fontsize,
):
    """
    Draw a value in every finite cell using black/white text chosen
    automatically for contrast with the cell background.
    """
    cmap = image.cmap
    norm = image.norm

    nrows, ncols = values.shape

    for i in range(nrows):
        for j in range(ncols):
            value = values[i, j]

            if not np.isfinite(value):
                continue

            try:
                rgba = cmap(norm(value))
                text_color = "black" if luminance(rgba) > 0.58 else "white"
            except Exception:
                text_color = "black"

            ax.text(
                j,
                i,
                formatter(value),
                ha="center",
                va="center",
                fontsize=fontsize,
                color=text_color,
                fontweight="normal",
                clip_on=True,
            )


# =============================================================================
# 11. MAIN FIGURE — H2 HEATMAP WITH NUMERIC VALUES
# =============================================================================

h2_values = np.array(
    h2_matrix,
    dtype=float,
    copy=True,
)

h2_values = np.where(
    np.isfinite(h2_values),
    h2_values,
    np.nan,
)

finite_h2 = h2_values[
    np.isfinite(h2_values)
]

if finite_h2.size == 0:
    raise RuntimeError(
        "No finite h2 values available for the heatmap."
    )

h2_min = float(
    np.nanmin(finite_h2)
)

h2_max = float(
    np.nanmax(finite_h2)
)

print()
print("h2 heatmap:")
print("  finite cells :", finite_h2.size)
print("  missing cells:", int(np.isnan(h2_values).sum()))
print("  minimum      :", h2_min)
print("  maximum      :", h2_max)
print("  binary scale : liability")
print("  BMI scale    : native quantitative")

if h2_min < 0 < h2_max:
    h2_norm = TwoSlopeNorm(
        vmin=h2_min,
        vcenter=0.0,
        vmax=h2_max,
    )
else:
    h2_norm = None

fig, ax = plt.subplots(
    figsize=H2_FIGSIZE
)

cmap_h2 = plt.get_cmap(
    "coolwarm"
).copy()

cmap_h2.set_bad(
    "lightgray"
)

image = ax.imshow(
    h2_values,
    aspect="auto",
    interpolation="nearest",
    cmap=cmap_h2,
    norm=h2_norm,
)

format_configuration_axis(ax)

if SHOW_H2_VALUES:
    annotate_matrix(
        ax=ax,
        values=h2_values,
        image=image,
        formatter=lambda v: f"{v:.{H2_DECIMALS}f}",
        fontsize=H2_TEXT_SIZE,
    )

ax.set_title(
    "Configuration-specific SNP-heritability estimates",
    fontsize=14,
    pad=14,
)

# Scale reminder directly on figure.
ax.text(
    0.0,
    1.012,
    "Binary phenotypes: liability scale   |   BMI: native quantitative scale",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=9,
)

cbar = fig.colorbar(
    image,
    ax=ax,
    fraction=0.028,
    pad=0.020,
)

cbar.set_label(
    r"Mean $h^2$",
    rotation=270,
    labelpad=18,
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "Figure_H2_Heatmap.png",
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    OUTPUT_DIR / "Figure_H2_Heatmap.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# 12. SUPPLEMENTARY FIGURE — ESTIMATOR 95% CI WIDTH HEATMAP
# =============================================================================

ci_values = np.array(
    ci_matrix,
    dtype=float,
    copy=True,
)

# No in-place assignment: avoids read-only NumPy view errors.
ci_values = np.where(
    np.isfinite(ci_values) & (ci_values > 0),
    ci_values,
    np.nan,
)

finite_ci = ci_values[
    np.isfinite(ci_values)
]

print()
print("Estimator-specific CI-width heatmap:")
print("  finite CI cells :", finite_ci.size)
print("  missing CI cells:", int(np.isnan(ci_values).sum()))

if finite_ci.size == 0:

    print()
    print(
        "WARNING: No valid positive estimator-specific "
        "95% CI widths were available."
    )
    print(
        "CI-width heatmap was not generated."
    )

else:

    ci_min = float(
        np.nanmin(finite_ci)
    )

    ci_max = float(
        np.nanmax(finite_ci)
    )

    print("  minimum CI width:", ci_min)
    print("  maximum CI width:", ci_max)

    if np.isclose(ci_min, ci_max):

        ci_vmin = max(
            ci_min * 0.5,
            1e-12,
        )

        ci_vmax = ci_max * 1.5

        if np.isclose(
            ci_vmin,
            ci_vmax,
        ):
            ci_vmax = ci_vmin * 10.0

    else:

        ci_vmin = ci_min
        ci_vmax = ci_max

    fig, ax = plt.subplots(
        figsize=CI_FIGSIZE
    )

    cmap_ci = plt.get_cmap(
        "viridis"
    ).copy()

    cmap_ci.set_bad(
        "lightgray"
    )

    image = ax.imshow(
        ci_values,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap_ci,
        norm=LogNorm(
            vmin=ci_vmin,
            vmax=ci_vmax,
        ),
    )

    format_configuration_axis(ax)

    if SHOW_CI_VALUES:
        annotate_matrix(
            ax=ax,
            values=ci_values,
            image=image,
            formatter=lambda v: f"{v:.2g}",
            fontsize=CI_TEXT_SIZE,
        )

    ax.set_title(
        "Estimator-specific uncertainty across SNP-heritability configurations",
        fontsize=14,
        pad=14,
    )

    ax.text(
        0.0,
        1.012,
        "Cell value = estimator-specific 95% CI width   |   Binary: liability scale   |   BMI: native quantitative scale",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.028,
        pad=0.020,
    )

    cbar.set_label(
        "Estimator-specific 95% CI width (log scale)",
        rotation=270,
        labelpad=22,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "Figure_Estimator_CIWidth_Heatmap.png",
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_DIR / "Figure_Estimator_CIWidth_Heatmap.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# 13. FINAL SUMMARY
# =============================================================================

print()
print("=" * 100)
print("FIGURE GENERATION COMPLETE")
print("=" * 100)

print()
print("Configuration count:", len(configurations))

print()
print("Main h2 heatmap:")
print(
    " ",
    OUTPUT_DIR / "Figure_H2_Heatmap.png",
)
print(
    " ",
    OUTPUT_DIR / "Figure_H2_Heatmap.pdf",
)

if finite_ci.size > 0:
    print()
    print("Supplementary estimator-CI heatmap:")
    print(
        " ",
        OUTPUT_DIR / "Figure_Estimator_CIWidth_Heatmap.png",
    )
    print(
        " ",
        OUTPUT_DIR / "Figure_Estimator_CIWidth_Heatmap.pdf",
    )

print()
print("Underlying plotted data:")
print(
    " ",
    OUTPUT_DIR / "Figure_H2_Heatmap_Data.csv",
)
print(
    " ",
    OUTPUT_DIR / "Figure_Estimator_CIWidth_Heatmap_Data.csv",
)
print(
    " ",
    OUTPUT_DIR / "Figure_Configuration_Order.csv",
)

print()
print("IMPORTANT INTERPRETATION:")
print(
    " * Main heatmap contains the numeric h2 value inside every finite cell."
)
print(
    " * Binary phenotypes are on the LIABILITY SCALE."
)
print(
    " * BMI is on its NATIVE QUANTITATIVE SCALE."
)
print(
    " * CI heatmap = estimator-specific 95% CI width."
)
print(
    " * Grey CI cells = estimator-specific CI unavailable."
)
print(
    " * Cross-fold SD is NOT used as an inferential confidence interval."
)
print(
    " * Negative and >1 h2 estimates remain untruncated."
)
print()
