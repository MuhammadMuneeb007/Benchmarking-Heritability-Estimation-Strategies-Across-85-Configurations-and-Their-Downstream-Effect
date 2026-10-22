 
# -*- coding: utf-8 -*-

"""
FinalAnalysis07-HorizontalUncertainty.py
=======================================

FINAL reviewer-facing replacement for the old horizontal Supplementary Table S1.

WHAT THIS SCRIPT DOES
---------------------
1. Creates ONE horizontal XLSX table in the same structure as the existing workbook:
      Configuration | Asthma | BMI | Depression | ... | Osteoarthritis

2. Keeps FOUR visible configuration columns:
      Method | h2model | clumpprune | numberofpca

3. For EVERY binary phenotype, reports horizontally:
      Original mean h2
      Original estimator SE (median across folds)
      Original estimator 95% CI (median fold-level bounds)
      Original cross-fold SD
      Liability mean h2
      Liability estimator SE (median across folds)
      Liability estimator 95% CI (median fold-level bounds)
      Liability cross-fold SD

4. For BMI, the "Original" block is the native quantitative scale.
   Liability columns are shown as N/A (quantitative).

5. Phenotype blocks are COLOURED individually, matching the style of the
   user's existing workbook.

6. DOES NOT create another heatmap. The old point-estimate heatmap should be
   removed from the main manuscript.

7. Creates reviewer-friendly uncertainty figures instead:
      Figure_PointEstimate_Distribution.png
      Figure_Estimator_CIWidth.png
      Figure_CrossFold_SD.png
      Figure_Simulation_Binary_Stability_N750_vs_N5000.png
      Figure_Simulation_Continuous_Stability_N750_vs_N5000.png

SCIENTIFIC INTERPRETATION
-------------------------
* Estimator SE / CI:
  These are estimator-specific uncertainty quantities from the software output
  where available. We take the MEDIAN across the available fold-specific
  estimator SEs / CI bounds for the configuration-level summary.

* Cross-fold SD:
  This is descriptive stability across the five OVERLAPPING CV training folds.
  It is NOT an inferential standard error.

* The "Estimator 95% CI" displayed here is the median of the fold-specific
  estimator CI lower bounds and the median of the fold-specific upper bounds.
  It is NOT presented as a formal CI for the across-fold mean h2.

* LDpred2:
  If the underlying parsed output does not provide estimator-specific SE/CI,
  the cells are explicitly reported as NA. Cross-fold SD is still reported.

* Negative h2 and h2 > 1:
  Retained without clipping.

INPUTS
------
Uses the outputs already created by:
    FinalAnalysis06-UncertaintyReliability.py

Required directory:
    FinalAnalysis06-UncertaintyReliability/

Required files:
    FinalAnalysis06_unit_stability.csv
    FinalAnalysis06_raw_scale_rows.csv
    FinalAnalysis06_boundary_summary.csv
    FinalAnalysis06_uncertainty_availability.csv
    FinalAnalysis06_sample_size_stability.csv
    FinalAnalysis06_file_audit.csv

RUN
---
From:
    /data/ascher01/uqmmune1/finalheritability

run:
    python FinalAnalysis07-HorizontalUncertainty.py

OUTPUT
------
Creates:
    FinalAnalysis07-HorizontalUncertainty/

Main workbook:
    FinalSupplementaryTableS1_Horizontal_Uncertainty.xlsx

Figures:
    Figure_PointEstimate_Distribution.png
    Figure_Estimator_CIWidth.png
    Figure_CrossFold_SD.png
    Figure_Simulation_Binary_Stability_N750_vs_N5000.png
    Figure_Simulation_Continuous_Stability_N750_vs_N5000.png

Text:
    Manuscript_Figure_Captions.tex
    Supplementary_Table_S1_Description.tex
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
OUTPUT_DIR = ROOT / "FinalAnalysis07-HorizontalUncertainty"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_XLSX = OUTPUT_DIR / "FinalSupplementaryTableS1_Horizontal_Uncertainty.xlsx"


# =============================================================================
# 2. PHENOTYPES / COLOURS / METHODS
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

# Explicitly chosen to resemble the user's existing coloured phenotype blocks.
PHENOTYPE_COLORS = {
    "Asthma": "#C89999",                    # rose
    "Body Mass Index": "#9EAFCC",           # blue
    "Depression": "#AEC296",                # green
    "Gastro-Oesophageal Reflux": "#D5AB78", # orange
    "High Cholesterol": "#BAA4C8",          # lavender
    "Hypertension": "#D59A8C",              # salmon
    "Hypothyroidism": "#91BBB5",            # teal
    "Irritable Bowel Syndrome": "#D6C776",  # mustard
    "Migraine": "#AAA0CA",                  # purple
    "Osteoarthritis": "#8FB7C9",            # cyan-blue
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
# 3. BASIC HELPERS
# =============================================================================

def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return path


def num(x):
    try:
        y = float(x)
        return y if np.isfinite(y) else np.nan
    except Exception:
        return np.nan


def safe_series(x):
    return pd.to_numeric(pd.Series(x), errors="coerce")


def lighten(hex_color: str, amount: float = 0.18) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = int(r + (255-r)*amount)
    g = int(g + (255-g)*amount)
    b = int(b + (255-b)*amount)
    return f"#{r:02X}{g:02X}{b:02X}"


def fmt_ci(lo, hi) -> str:
    lo = num(lo)
    hi = num(hi)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return "NA"
    return f"[{lo:.4f}, {hi:.4f}]"


def scale_label(scale):
    if scale == "observed_binary":
        return "Observed"
    if scale == "liability_binary":
        return "Liability"
    if scale == "native_quantitative":
        return "Native quantitative"
    return str(scale)


# =============================================================================
# 4. CONFIGURATION PARSING
# =============================================================================

def parse_config_id(config_id: str):
    """
    FinalAnalysis06 uses ConfigID strings like:
        Method | h2model_standardized=... | clumpprune=yes | numberofpca=6 | ...

    We preserve the same four visible configuration columns as the user's
    current workbook.
    """
    text = str(config_id)
    fields = {}

    for piece in text.split("|"):
        piece = piece.strip()
        if "=" not in piece:
            continue
        k, v = piece.split("=", 1)
        fields[k.strip().lower()] = v.strip()

    h2model = (
        fields.get("h2model_standardized")
        or fields.get("h2model")
        or fields.get("model")
        or fields.get("configuration")
        or fields.get("config")
        or fields.get("tagging_file")
        or ""
    )

    clump = (
        fields.get("clumpprune")
        or fields.get("clump")
        or ""
    )

    pca = (
        fields.get("numberofpca")
        or fields.get("number_of_pca")
        or fields.get("pca")
        or ""
    )

    # If h2model is blank, create a compact readable model description from
    # the remaining configuration fields, excluding the two visible columns.
    if not h2model:
        skip = {
            "clumpprune", "clump",
            "numberofpca", "number_of_pca", "pca",
        }
        pieces = []
        for k, v in fields.items():
            if k in skip:
                continue
            pieces.append(f"{k}={v}")
        h2model = "; ".join(pieces)

    return h2model, clump, pca


# =============================================================================
# 5. LOAD ANALYSIS 06 OUTPUTS
# =============================================================================

def load_inputs():
    paths = {
        "units": require(INPUT_DIR / "FinalAnalysis06_unit_stability.csv"),
        "raw": require(INPUT_DIR / "FinalAnalysis06_raw_scale_rows.csv"),
        "boundary": require(INPUT_DIR / "FinalAnalysis06_boundary_summary.csv"),
        "uncertainty": require(INPUT_DIR / "FinalAnalysis06_uncertainty_availability.csv"),
        "sample_size": require(INPUT_DIR / "FinalAnalysis06_sample_size_stability.csv"),
        "file_audit": require(INPUT_DIR / "FinalAnalysis06_file_audit.csv"),
    }

    return {
        k: pd.read_csv(v)
        for k, v in paths.items()
    }


# =============================================================================
# 6. BUILD CONFIGURATION-LEVEL LONG TABLE
# =============================================================================

def build_empirical_long(units: pd.DataFrame, raw: pd.DataFrame):
    """
    One row per:
        phenotype x method x configuration x scale

    Carries:
        mean h2
        estimator SE (median fold-specific tool SE)
        estimator CI lower/upper (median fold-level bounds)
        cross-fold SD
    """
    u = units[units["Domain"] == "empirical"].copy()

    valid_scale = (
        ((u["TraitType"] == "binary") &
         (u["Scale"].isin(["observed_binary", "liability_binary"])))
        |
        ((u["TraitType"] == "continuous") &
         (u["Scale"] == "native_quantitative"))
    )

    u = u[valid_scale].copy()
    u["Phenotype"] = u["DatasetLabel"].astype(str)

    # Fold-level estimator uncertainty from raw-scale rows.
    r = raw[raw["Domain"] == "empirical"].copy()

    valid_scale_raw = (
        ((r["TraitType"] == "binary") &
         (r["Scale"].isin(["observed_binary", "liability_binary"])))
        |
        ((r["TraitType"] == "continuous") &
         (r["Scale"] == "native_quantitative"))
    )

    r = r[valid_scale_raw].copy()
    r["Phenotype"] = r["DatasetLabel"].astype(str)

    for c in ["ToolSE", "ToolCILower", "ToolCIUpper"]:
        r[c] = pd.to_numeric(r[c], errors="coerce")

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

    keep = [
        "Phenotype",
        "TraitType",
        "Scale",
        "Method",
        "MethodFamily",
        "ConfigID",
        "ValidH2Folds",
        "MeanH2",
        "MedianH2",
        "MinH2",
        "MaxH2",
        "FoldSD",
        "FoldMAD",
        "FoldIQR",
        "FoldRange",
        "MeanBoundaryClass",
        "AnyNegativeFold",
        "AnyGreaterThan1Fold",
        "SignFlipAcrossFolds",
    ]

    keep = [c for c in keep if c in u.columns]
    out = u[keep].copy()

    out = out.merge(
        uncertainty,
        on=["Phenotype", "Method", "ConfigID", "Scale"],
        how="left",
    )

    parsed = out["ConfigID"].apply(parse_config_id)
    out["h2model"] = [x[0] for x in parsed]
    out["clumpprune"] = [x[1] for x in parsed]
    out["numberofpca"] = [x[2] for x in parsed]

    out["Estimator95CI"] = [
        fmt_ci(lo, hi)
        for lo, hi in zip(
            out["EstimatorCILower"],
            out["EstimatorCIUpper"],
        )
    ]

    out["EstimatorCIWidth"] = (
        pd.to_numeric(out["EstimatorCIUpper"], errors="coerce")
        -
        pd.to_numeric(out["EstimatorCILower"], errors="coerce")
    )

    return out


# =============================================================================
# 7. BUILD 86 HORIZONTAL CONFIGURATION ROWS
# =============================================================================

def build_configurations(long: pd.DataFrame):
    configs = (
        long[
            ["Method", "ConfigID", "h2model", "clumpprune", "numberofpca"]
        ]
        .drop_duplicates()
        .copy()
    )

    configs["MethodOrder"] = pd.Categorical(
        configs["Method"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    configs = (
        configs.sort_values(
            ["MethodOrder", "h2model", "clumpprune", "numberofpca", "ConfigID"],
            kind="stable",
        )
        .drop(columns="MethodOrder")
        .reset_index(drop=True)
    )

    print(f"Unique empirical configurations found: {len(configs)}")

    if len(configs) != 86:
        print(
            "WARNING: expected 86 unique configurations but found "
            f"{len(configs)}. The workbook will preserve every unique ConfigID "
            "found in FinalAnalysis06."
        )

    return configs


def get_one(long, method, config_id, phenotype, scale):
    z = long[
        (long["Method"] == method)
        & (long["ConfigID"] == config_id)
        & (long["Phenotype"] == phenotype)
        & (long["Scale"] == scale)
    ]

    if z.empty:
        return None

    return z.iloc[0]


# =============================================================================
# 8. SUMMARY TABLES
# =============================================================================

def build_method_summary(long):
    # Primary reporting scale: liability for binary; native quantitative for BMI.
    p = long[
        (
            (long["TraitType"] == "binary")
            & (long["Scale"] == "liability_binary")
        )
        |
        (
            (long["TraitType"] == "continuous")
            & (long["Scale"] == "native_quantitative")
        )
    ].copy()

    rows = []

    for (method, scale), g in p.groupby(["Method", "Scale"], dropna=False):
        h = pd.to_numeric(g["MeanH2"], errors="coerce")
        se = pd.to_numeric(g["EstimatorSE"], errors="coerce")
        ciw = pd.to_numeric(g["EstimatorCIWidth"], errors="coerce")
        sd = pd.to_numeric(g["FoldSD"], errors="coerce")

        n = int(h.notna().sum())

        rows.append({
            "Scale": scale_label(scale),
            "Method": method,
            "N": n,
            "Median h2": h.median(),
            "Median estimator SE": se.median(),
            "Median estimator CI width": ciw.median(),
            "Estimator SE available, n": int(se.notna().sum()),
            "Estimator SE coverage (%)": 100*se.notna().sum()/n if n else np.nan,
            "Median cross-fold SD": sd.median(),
            "h2 < 0, n": int((h < 0).sum()),
            "h2 < 0 (%)": 100*(h < 0).sum()/n if n else np.nan,
            "0 <= h2 <= 1, n": int(((h >= 0) & (h <= 1)).sum()),
            "0 <= h2 <= 1 (%)": 100*((h >= 0) & (h <= 1)).sum()/n if n else np.nan,
            "h2 > 1, n": int((h > 1).sum()),
            "h2 > 1 (%)": 100*(h > 1).sum()/n if n else np.nan,
        })

    out = pd.DataFrame(rows)

    out["MethodOrder"] = pd.Categorical(
        out["Method"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    return (
        out.sort_values(["Scale", "MethodOrder"])
        .drop(columns="MethodOrder")
        .reset_index(drop=True)
    )


def build_phenotype_summary(long):
    p = long[
        (
            (long["TraitType"] == "binary")
            & (long["Scale"] == "liability_binary")
        )
        |
        (
            (long["TraitType"] == "continuous")
            & (long["Scale"] == "native_quantitative")
        )
    ].copy()

    rows = []

    for phenotype, g in p.groupby("Phenotype", dropna=False):
        h = pd.to_numeric(g["MeanH2"], errors="coerce")
        se = pd.to_numeric(g["EstimatorSE"], errors="coerce")
        ciw = pd.to_numeric(g["EstimatorCIWidth"], errors="coerce")
        sd = pd.to_numeric(g["FoldSD"], errors="coerce")

        n = int(h.notna().sum())

        rows.append({
            "Phenotype": phenotype,
            "Scale": "Native quantitative" if phenotype == "Body Mass Index" else "Liability",
            "N": n,
            "Median h2": h.median(),
            "Minimum h2": h.min(),
            "Maximum h2": h.max(),
            "Median estimator SE": se.median(),
            "Median estimator CI width": ciw.median(),
            "Estimator SE coverage (%)": 100*se.notna().sum()/n if n else np.nan,
            "Median cross-fold SD": sd.median(),
            "h2 < 0, n": int((h < 0).sum()),
            "h2 < 0 (%)": 100*(h < 0).sum()/n if n else np.nan,
            "h2 > 1, n": int((h > 1).sum()),
            "h2 > 1 (%)": 100*(h > 1).sum()/n if n else np.nan,
        })

    out = pd.DataFrame(rows)

    out["PhenotypeOrder"] = pd.Categorical(
        out["Phenotype"],
        categories=PHENOTYPE_ORDER,
        ordered=True,
    )

    return (
        out.sort_values("PhenotypeOrder")
        .drop(columns="PhenotypeOrder")
        .reset_index(drop=True)
    )


# =============================================================================
# 9. WRITE HORIZONTAL XLSX
# =============================================================================

def write_horizontal_workbook(
    long,
    configs,
    method_summary,
    phenotype_summary,
    sample_size,
    file_audit,
):
    try:
        import xlsxwriter
    except ImportError as e:
        raise RuntimeError(
            "xlsxwriter is required. Install with: pip install xlsxwriter"
        ) from e

    writer = pd.ExcelWriter(
        OUT_XLSX,
        engine="xlsxwriter",
    )

    wb = writer.book
    ws = wb.add_worksheet("Horizontal_S1")
    writer.sheets["Horizontal_S1"] = ws

    # -------------------------------------------------------------------------
    # Formats
    # -------------------------------------------------------------------------
    fmt_config_group = wb.add_format({
        "bold": True,
        "font_color": "white",
        "bg_color": "#1F4E78",
        "align": "center",
        "valign": "vcenter",
        "border": 1,
        "font_size": 11,
    })

    fmt_config_header = wb.add_format({
        "bold": True,
        "font_color": "white",
        "bg_color": "#2D6593",
        "align": "center",
        "valign": "vcenter",
        "border": 1,
        "text_wrap": True,
    })

    phenotype_top = {}
    phenotype_sub = {}

    for ph, color in PHENOTYPE_COLORS.items():
        phenotype_top[ph] = wb.add_format({
            "bold": True,
            "bg_color": color,
            "font_color": "#000000",
            "align": "center",
            "valign": "vcenter",
            "border": 1,
            "font_size": 11,
        })

        phenotype_sub[ph] = wb.add_format({
            "bold": True,
            "bg_color": lighten(color, 0.18),
            "font_color": "#000000",
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
            "border": 1,
            "font_size": 9,
        })

    fmt_text = wb.add_format({
        "border": 1,
        "valign": "vcenter",
    })

    fmt_text_alt = wb.add_format({
        "border": 1,
        "valign": "vcenter",
        "bg_color": "#F5F5F5",
    })

    fmt_num = wb.add_format({
        "border": 1,
        "num_format": "0.0000",
        "valign": "vcenter",
        "align": "center",
    })

    fmt_num_alt = wb.add_format({
        "border": 1,
        "num_format": "0.0000",
        "valign": "vcenter",
        "align": "center",
        "bg_color": "#F5F5F5",
    })

    fmt_ci = wb.add_format({
        "border": 1,
        "valign": "vcenter",
        "align": "center",
        "font_size": 9,
    })

    fmt_ci_alt = wb.add_format({
        "border": 1,
        "valign": "vcenter",
        "align": "center",
        "font_size": 9,
        "bg_color": "#F5F5F5",
    })

    fmt_na = wb.add_format({
        "border": 1,
        "font_color": "#7F7F7F",
        "italic": True,
        "align": "center",
        "valign": "vcenter",
    })

    fmt_na_alt = wb.add_format({
        "border": 1,
        "font_color": "#7F7F7F",
        "italic": True,
        "align": "center",
        "valign": "vcenter",
        "bg_color": "#F5F5F5",
    })

    # -------------------------------------------------------------------------
    # Header rows
    # -------------------------------------------------------------------------
    config_headers = [
        "Method",
        "h2model",
        "clumpprune",
        "numberofpca",
    ]

    ws.merge_range(
        0, 0, 0, 3,
        "Configuration",
        fmt_config_group,
    )

    for col, header in enumerate(config_headers):
        ws.write(1, col, header, fmt_config_header)

    phenotype_columns = {}
    current_col = 4

    subheaders = [
        "Original mean h2",
        "Original estimator SE\n(median fold)",
        "Original estimator 95% CI\n(median fold bounds)",
        "Original cross-fold SD",
        "Liability mean h2",
        "Liability estimator SE\n(median fold)",
        "Liability estimator 95% CI\n(median fold bounds)",
        "Liability cross-fold SD",
    ]

    for ph in PHENOTYPE_ORDER:
        start = current_col
        end = current_col + len(subheaders) - 1

        phenotype_columns[ph] = {
            "start": start,
            "end": end,
            "subheaders": subheaders,
        }

        ws.merge_range(
            0, start, 0, end,
            ph,
            phenotype_top[ph],
        )

        for j, header in enumerate(subheaders):
            ws.write(
                1,
                start+j,
                header,
                phenotype_sub[ph],
            )

        current_col = end + 1

    # -------------------------------------------------------------------------
    # Data rows
    # -------------------------------------------------------------------------
    h2_columns = []

    for row_number, cfg in configs.iterrows():
        excel_row = row_number + 2
        alt = (row_number % 2 == 1)

        tformat = fmt_text_alt if alt else fmt_text
        nformat = fmt_num_alt if alt else fmt_num
        ciformat = fmt_ci_alt if alt else fmt_ci
        naformat = fmt_na_alt if alt else fmt_na

        ws.write(excel_row, 0, cfg["Method"], tformat)
        ws.write(excel_row, 1, cfg["h2model"], tformat)
        ws.write(excel_row, 2, cfg["clumpprune"], tformat)
        ws.write(excel_row, 3, cfg["numberofpca"], tformat)

        for ph in PHENOTYPE_ORDER:
            start = phenotype_columns[ph]["start"]

            if ph == "Body Mass Index":
                original_scale = "native_quantitative"
            else:
                original_scale = "observed_binary"

            ro = get_one(
                long,
                cfg["Method"],
                cfg["ConfigID"],
                ph,
                original_scale,
            )

            # Original/native 4 columns.
            if ro is None:
                original_values = [np.nan, np.nan, "NA", np.nan]
            else:
                original_values = [
                    ro["MeanH2"],
                    ro["EstimatorSE"],
                    ro["Estimator95CI"],
                    ro["FoldSD"],
                ]

            # Liability 4 columns.
            if ph == "Body Mass Index":
                liability_values = [
                    "N/A (quantitative)",
                    "N/A",
                    "N/A",
                    "N/A",
                ]
            else:
                rl = get_one(
                    long,
                    cfg["Method"],
                    cfg["ConfigID"],
                    ph,
                    "liability_binary",
                )

                if rl is None:
                    liability_values = [np.nan, np.nan, "NA", np.nan]
                else:
                    liability_values = [
                        rl["MeanH2"],
                        rl["EstimatorSE"],
                        rl["Estimator95CI"],
                        rl["FoldSD"],
                    ]

            values = original_values + liability_values

            for j, value in enumerate(values):
                col = start+j
                header = subheaders[j]

                if "mean h2" in header:
                    h2_columns.append(col)

                if "95% CI" in header:
                    if value in [None, "", "NA"]:
                        ws.write(excel_row, col, "NA", naformat)
                    else:
                        ws.write(excel_row, col, str(value), ciformat)

                elif isinstance(value, str):
                    ws.write(excel_row, col, value, naformat)

                else:
                    v = num(value)
                    if np.isfinite(v):
                        ws.write_number(excel_row, col, v, nformat)
                    else:
                        ws.write(excel_row, col, "NA", naformat)

    last_data_row = len(configs) + 1
    last_col = current_col - 1

    # -------------------------------------------------------------------------
    # Widths / freezing / zoom
    # -------------------------------------------------------------------------
    ws.set_column(0, 0, 27)
    ws.set_column(1, 1, 33)
    ws.set_column(2, 2, 13)
    ws.set_column(3, 3, 13)

    for ph in PHENOTYPE_ORDER:
        start = phenotype_columns[ph]["start"]

        ws.set_column(start+0, start+0, 14)
        ws.set_column(start+1, start+1, 15)
        ws.set_column(start+2, start+2, 23)
        ws.set_column(start+3, start+3, 15)
        ws.set_column(start+4, start+4, 14)
        ws.set_column(start+5, start+5, 15)
        ws.set_column(start+6, start+6, 23)
        ws.set_column(start+7, start+7, 15)

    ws.set_row(0, 24)
    ws.set_row(1, 54)

    ws.freeze_panes(2, 4)
    ws.set_zoom(50)

    # -------------------------------------------------------------------------
    # Filter
    # -------------------------------------------------------------------------
    ws.autofilter(
        1, 0,
        last_data_row, last_col,
    )

    # -------------------------------------------------------------------------
    # Conditional formatting on h2 point estimates
    # -------------------------------------------------------------------------
    negative_fmt = wb.add_format({
        "font_color": "#9C0006",
        "bg_color": "#FFC7CE",
    })

    above1_fmt = wb.add_format({
        "font_color": "#9C6500",
        "bg_color": "#FFEB9C",
    })

    # unique h2 columns only
    for col in sorted(set(h2_columns)):
        ws.conditional_format(
            2, col,
            last_data_row, col,
            {
                "type": "cell",
                "criteria": "<",
                "value": 0,
                "format": negative_fmt,
            }
        )

        ws.conditional_format(
            2, col,
            last_data_row, col,
            {
                "type": "cell",
                "criteria": ">",
                "value": 1,
                "format": above1_fmt,
            }
        )

    # -------------------------------------------------------------------------
    # Comments explaining uncertainty
    # -------------------------------------------------------------------------
    ws.write_comment(
        1, 4,
        "Mean h2 is averaged across available cross-validation folds."
    )

    ws.write_comment(
        1, 5,
        "Estimator SE is the median estimator-specific standard error across "
        "available fold outputs. This is distinct from cross-fold SD."
    )

    ws.write_comment(
        1, 6,
        "This is the median of the fold-level estimator-specific lower and "
        "upper CI bounds. It is NOT a formal confidence interval for the "
        "cross-fold mean h2."
    )

    ws.write_comment(
        1, 7,
        "Cross-fold SD is descriptive stability across overlapping CV "
        "training folds and must NOT be interpreted as an inferential SE."
    )

    # -------------------------------------------------------------------------
    # Add note at far right
    # -------------------------------------------------------------------------
    note_col = last_col + 2

    note_fmt = wb.add_format({
        "text_wrap": True,
        "valign": "top",
        "bg_color": "#FFF2CC",
        "border": 1,
    })

    ws.set_column(note_col, note_col+1, 28)

    ws.write(0, note_col, "Interpretation", fmt_config_group)
    ws.merge_range(
        1, note_col,
        8, note_col+1,
        "Estimator-specific SE/CI and cross-fold SD represent different "
        "uncertainty concepts. Estimator SE/CI are taken from the underlying "
        "statistical estimator where available. Cross-fold SD describes "
        "sensitivity to the five overlapping training partitions and is not "
        "an inferential standard error. NA means estimator-specific uncertainty "
        "was unavailable or liability conversion could not be performed. "
        "Negative and >1 h2 estimates are retained without clipping.",
        note_fmt,
    )

    # -------------------------------------------------------------------------
    # Summary sheets
    # -------------------------------------------------------------------------
    def write_summary_sheet(df, name, color):
        df.to_excel(
            writer,
            sheet_name=name,
            index=False,
        )

        sws = writer.sheets[name]

        header_fmt = wb.add_format({
            "bold": True,
            "font_color": "white",
            "bg_color": color,
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
        })

        number_fmt = wb.add_format({
            "num_format": "0.0000",
            "border": 1,
        })

        text_fmt2 = wb.add_format({
            "border": 1,
        })

        for c, header in enumerate(df.columns):
            sws.write(0, c, header, header_fmt)

            if pd.api.types.is_numeric_dtype(df[header]):
                sws.set_column(c, c, 16, number_fmt)
            else:
                sws.set_column(c, c, min(max(14, len(str(header))+2), 32), text_fmt2)

        sws.freeze_panes(1, 0)
        sws.autofilter(0, 0, len(df), len(df.columns)-1)

    write_summary_sheet(
        method_summary,
        "Method_Summary",
        "#5B9BD5",
    )

    write_summary_sheet(
        phenotype_summary,
        "Phenotype_Summary",
        "#70AD47",
    )

    write_summary_sheet(
        sample_size,
        "SampleSize_Stability",
        "#8064A2",
    )

    write_summary_sheet(
        file_audit,
        "File_Completeness",
        "#7F7F7F",
    )

    # -------------------------------------------------------------------------
    # README
    # -------------------------------------------------------------------------
    readme = wb.add_worksheet("README")
    writer.sheets["README"] = readme

    title_fmt = wb.add_format({
        "bold": True,
        "font_size": 15,
        "font_color": "white",
        "bg_color": "#1F4E78",
    })

    key_fmt = wb.add_format({
        "bold": True,
        "bg_color": "#D9EAF7",
        "border": 1,
    })

    val_fmt = wb.add_format({
        "text_wrap": True,
        "valign": "top",
        "border": 1,
    })

    readme.write(
        "A1",
        "Supplementary Table S1 — Heritability estimates with uncertainty",
        title_fmt,
    )

    notes = [
        (
            "Purpose",
            "Horizontal reviewer-facing table replacing the old point-estimate / "
            "fold-SE table."
        ),
        (
            "Original mean h2",
            "Mean of the native/observed-scale h2 estimates across available folds. "
            "For BMI this is the native quantitative scale."
        ),
        (
            "Liability mean h2",
            "Primary liability-scale mean for binary phenotypes only."
        ),
        (
            "Estimator SE",
            "Median estimator-specific SE across fold outputs, where supplied by "
            "the underlying method."
        ),
        (
            "Estimator 95% CI",
            "Median lower and upper fold-level estimator CI bounds. These are "
            "not claimed to be a formal CI for the across-fold mean."
        ),
        (
            "Cross-fold SD",
            "Descriptive variability across the five overlapping training folds. "
            "This is NOT an inferential SE."
        ),
        (
            "LDpred2",
            "If estimator-specific uncertainty was unavailable in the parsed output, "
            "SE/CI cells are explicitly NA; cross-fold SD remains available."
        ),
        (
            "Negative / >1 estimates",
            "Retained without clipping and highlighted in the horizontal table."
        ),
        (
            "Missing liability values",
            "Reported as NA rather than replacing them with observed-scale values."
        ),
    ]

    rr = 3

    for key, value in notes:
        readme.write(rr, 0, key, key_fmt)
        readme.write(rr, 1, value, val_fmt)
        rr += 1

    readme.set_column("A:A", 28)
    readme.set_column("B:B", 110)

    writer.close()

    print("Wrote:", OUT_XLSX)


# =============================================================================
# 10. FIGURE HELPERS
# =============================================================================

def primary_plot_data(long):
    """
    Primary manuscript scale:
      binary -> liability
      BMI    -> native quantitative
    """
    return long[
        (
            (long["TraitType"] == "binary")
            & (long["Scale"] == "liability_binary")
        )
        |
        (
            (long["TraitType"] == "continuous")
            & (long["Scale"] == "native_quantitative")
        )
    ].copy()


def method_positions(data):
    methods = [
        m for m in METHOD_ORDER
        if m in set(data["Method"])
    ]
    return methods, {m: i for i, m in enumerate(methods)}


def add_phenotype_legend(ax):
    handles = []

    for ph in PHENOTYPE_ORDER:
        handles.append(
            plt.Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markerfacecolor=PHENOTYPE_COLORS[ph],
                markeredgecolor=PHENOTYPE_COLORS[ph],
                label=ph,
                markersize=6,
            )
        )

    ax.legend(
        handles=handles,
        title="Phenotype",
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        fontsize=8,
        title_fontsize=9,
        frameon=False,
    )


# =============================================================================
# 11. FIGURE: POINT-ESTIMATE DISTRIBUTION
# =============================================================================

def figure_point_estimates(long):
    d = primary_plot_data(long)
    methods, xpos = method_positions(d)

    rng = np.random.default_rng(20260916)

    fig, ax = plt.subplots(figsize=(16, 8))

    # Light boxplots as distribution summaries.
    boxdata = [
        pd.to_numeric(
            d.loc[d["Method"] == m, "MeanH2"],
            errors="coerce",
        ).dropna().to_numpy()
        for m in methods
    ]

    ax.boxplot(
        boxdata,
        positions=np.arange(len(methods)),
        widths=0.50,
        showfliers=False,
        patch_artist=False,
    )

    for ph in PHENOTYPE_ORDER:
        z = d[d["Phenotype"] == ph]

        xs = []
        ys = []

        for _, r in z.iterrows():
            y = num(r["MeanH2"])

            if r["Method"] not in xpos or not np.isfinite(y):
                continue

            xs.append(
                xpos[r["Method"]] + rng.uniform(-0.20, 0.20)
            )
            ys.append(y)

        ax.scatter(
            xs,
            ys,
            s=18,
            alpha=0.65,
            color=PHENOTYPE_COLORS[ph],
            edgecolors="none",
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
        color="black",
    )

    ax.axhline(
        1,
        linestyle=":",
        linewidth=1,
        color="black",
    )

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(
        [METHOD_SHORT.get(m, m) for m in methods],
        rotation=35,
        ha="right",
    )

    ax.set_ylabel(r"Mean SNP heritability ($h^2$)")
    ax.set_xlabel("Heritability-estimation method")
    ax.set_title(
        "Configuration-level SNP-heritability estimates on the primary reporting scale"
    )

    ax.grid(
        axis="y",
        alpha=0.18,
    )

    add_phenotype_legend(ax)

    fig.savefig(
        OUTPUT_DIR / "Figure_PointEstimate_Distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_DIR / "Figure_PointEstimate_Distribution.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# 12. FIGURE: ESTIMATOR CI WIDTH
# =============================================================================

def figure_ci_width(long):
    d = primary_plot_data(long)
    d["EstimatorCIWidth"] = pd.to_numeric(
        d["EstimatorCIWidth"],
        errors="coerce",
    )

    # CI width must be finite and positive.
    d = d[
        d["EstimatorCIWidth"].notna()
        & (d["EstimatorCIWidth"] > 0)
    ].copy()

    methods, xpos = method_positions(d)
    rng = np.random.default_rng(20260916)

    fig, ax = plt.subplots(figsize=(16, 8))

    for ph in PHENOTYPE_ORDER:
        z = d[d["Phenotype"] == ph]

        xs = []
        ys = []

        for _, r in z.iterrows():
            if r["Method"] not in xpos:
                continue

            y = num(r["EstimatorCIWidth"])

            if not np.isfinite(y) or y <= 0:
                continue

            xs.append(
                xpos[r["Method"]] + rng.uniform(-0.20, 0.20)
            )
            ys.append(y)

        ax.scatter(
            xs,
            ys,
            s=18,
            alpha=0.65,
            color=PHENOTYPE_COLORS[ph],
            edgecolors="none",
        )

    ax.set_yscale("log")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(
        [METHOD_SHORT.get(m, m) for m in methods],
        rotation=35,
        ha="right",
    )

    ax.set_ylabel("Estimator-specific 95% CI width (log scale)")
    ax.set_xlabel("Heritability-estimation method")
    ax.set_title(
        "Estimator-specific uncertainty across heritability configurations"
    )

    ax.grid(
        axis="y",
        alpha=0.18,
    )

    add_phenotype_legend(ax)

    fig.savefig(
        OUTPUT_DIR / "Figure_Estimator_CIWidth.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_DIR / "Figure_Estimator_CIWidth.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# 13. FIGURE: CROSS-FOLD SD
# =============================================================================

def figure_crossfold_sd(long):
    d = primary_plot_data(long)
    d["FoldSD"] = pd.to_numeric(
        d["FoldSD"],
        errors="coerce",
    )

    d = d[
        d["FoldSD"].notna()
        & (d["FoldSD"] >= 0)
    ].copy()

    methods, xpos = method_positions(d)
    rng = np.random.default_rng(20260916)

    fig, ax = plt.subplots(figsize=(16, 8))

    for ph in PHENOTYPE_ORDER:
        z = d[d["Phenotype"] == ph]

        xs = []
        ys = []

        for _, r in z.iterrows():
            if r["Method"] not in xpos:
                continue

            y = num(r["FoldSD"])

            if not np.isfinite(y):
                continue

            xs.append(
                xpos[r["Method"]] + rng.uniform(-0.20, 0.20)
            )
            ys.append(y)

        ax.scatter(
            xs,
            ys,
            s=18,
            alpha=0.65,
            color=PHENOTYPE_COLORS[ph],
            edgecolors="none",
        )

    ax.set_yscale(
        "symlog",
        linthresh=1e-4,
    )

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(
        [METHOD_SHORT.get(m, m) for m in methods],
        rotation=35,
        ha="right",
    )

    ax.set_ylabel("Cross-fold SD (symlog scale)")
    ax.set_xlabel("Heritability-estimation method")
    ax.set_title(
        "Cross-fold stability across overlapping training partitions"
    )

    ax.grid(
        axis="y",
        alpha=0.18,
    )

    add_phenotype_legend(ax)

    fig.savefig(
        OUTPUT_DIR / "Figure_CrossFold_SD.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_DIR / "Figure_CrossFold_SD.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# 14. SIMULATION SAMPLE-SIZE FIGURES
# =============================================================================

def figure_simulation_stability(sample_size, trait_type, scale, stem, title):
    d = sample_size[
        (sample_size["TraitType"] == trait_type)
        & (sample_size["Scale"] == scale)
    ].copy()

    if d.empty:
        print(f"No rows for {trait_type} / {scale}; skipped {stem}.")
        return

    d["MethodOrder"] = pd.Categorical(
        d["Method"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    d = d.sort_values("MethodOrder")

    y = np.arange(len(d))

    n750 = pd.to_numeric(
        d["MedianFoldSD_N750"],
        errors="coerce",
    ).to_numpy()

    n5000 = pd.to_numeric(
        d["MedianFoldSD_N5000"],
        errors="coerce",
    ).to_numpy()

    fig, ax = plt.subplots(figsize=(11, 7))

    for i in range(len(d)):
        if np.isfinite(n750[i]) and np.isfinite(n5000[i]):
            ax.plot(
                [n750[i], n5000[i]],
                [i, i],
                linewidth=1,
                color="#9A9A9A",
                alpha=0.7,
            )

    ax.scatter(
        n750,
        y,
        s=45,
        marker="o",
        label="N=750",
    )

    ax.scatter(
        n5000,
        y,
        s=45,
        marker="s",
        label="N=5,000",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            METHOD_SHORT.get(m, m)
            for m in d["Method"]
        ]
    )

    ax.set_xscale(
        "symlog",
        linthresh=1e-4,
    )

    ax.set_xlabel("Median cross-fold SD (symlog scale)")
    ax.set_title(title)

    ax.grid(
        axis="x",
        alpha=0.18,
    )

    ax.legend(
        frameon=False,
    )

    fig.savefig(
        OUTPUT_DIR / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_DIR / f"{stem}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# 15. CAPTIONS / SUPPLEMENT DESCRIPTION
# =============================================================================

def write_text_outputs():
    captions = r"""
% ======================================================================
% MAIN / SUPPLEMENTARY FIGURE CAPTIONS
% ======================================================================

% POINT-ESTIMATE DISTRIBUTION
\caption{\textbf{Distribution of SNP-heritability estimates across analytical configurations.} Each point represents one configuration--phenotype combination and is coloured by phenotype. Binary phenotypes are shown on the primary liability scale and body mass index on its native quantitative scale. Boxplots summarise the corresponding method-level distributions. Estimates below zero and above one were retained without truncation. Complete configuration-level estimator-specific uncertainty and cross-fold stability measures are reported in Supplementary Table~S1.}

% ESTIMATOR UNCERTAINTY
\caption{\textbf{Estimator-specific uncertainty across SNP-heritability configurations.} Points show the width of the estimator-specific 95\% confidence interval for each evaluable configuration--phenotype combination and are coloured by phenotype. Binary phenotypes are shown on the primary liability scale and body mass index on its native quantitative scale. The vertical axis is logarithmic because estimator uncertainty spans several orders of magnitude. Configurations for which estimator-specific standard errors or confidence intervals were not available are not assigned an artificial interval and are reported as NA in Supplementary Table~S1.}

% CROSS-FOLD STABILITY
\caption{\textbf{Cross-fold stability of SNP-heritability estimates.} Points show the cross-fold standard deviation for each configuration--phenotype combination and are coloured by phenotype. Cross-fold SD quantifies sensitivity to changes in the five overlapping cross-validation training partitions and is interpreted as a descriptive stability measure rather than an inferential standard error.}

% SIMULATION SAMPLE-SIZE STABILITY
\caption{\textbf{Effect of target sample size on cross-fold stability in controlled simulations.} Method-level median cross-fold SD is shown for matched analyses at $N=750$ and $N=5{,}000$. Connecting lines indicate the direction of change for the same method. The larger target sample substantially reduced instability for several individual-level variance-component workflows.}
""".strip()

    supp = r"""
Supplementary Table~S1 reports the complete configuration-level heritability benchmark in horizontal phenotype blocks. For each binary phenotype and analytical configuration, the table provides the observed-scale mean $h^2$, the median estimator-specific standard error across available fold outputs, the median fold-level estimator-specific 95\% confidence-interval bounds, and cross-fold SD, together with the corresponding liability-scale quantities. Body mass index is quantitative and is reported on its native scale. Estimator-specific uncertainty is reported separately from cross-fold variability: cross-fold SD describes stability across the five overlapping training partitions and is not interpreted as an inferential standard error. Where estimator-specific uncertainty was unavailable, the corresponding SE/CI cells are reported as NA rather than replaced by fold-derived uncertainty. Negative and greater-than-one estimates were retained without truncation.
""".strip()

    (OUTPUT_DIR / "Manuscript_Figure_Captions.tex").write_text(
        captions,
        encoding="utf-8",
    )

    (OUTPUT_DIR / "Supplementary_Table_S1_Description.tex").write_text(
        supp,
        encoding="utf-8",
    )


# =============================================================================
# 16. MAIN
# =============================================================================

def main():
    print("="*115)
    print("FINAL ANALYSIS 07 — HORIZONTAL UNCERTAINTY TABLE + REVIEWER FIGURES")
    print("="*115)
    print("ROOT      :", ROOT)
    print("INPUT DIR :", INPUT_DIR)
    print("OUTPUT DIR:", OUTPUT_DIR)
    print()

    print("[1/7] Loading FinalAnalysis06 outputs ...")
    data = load_inputs()

    print("[2/7] Building empirical observed/liability configuration table ...")
    long = build_empirical_long(
        data["units"],
        data["raw"],
    )

    print("  long rows:", len(long))

    print("[3/7] Recovering horizontal configuration rows ...")
    configs = build_configurations(long)

    print("[4/7] Building method / phenotype summaries ...")
    method_summary = build_method_summary(long)
    phenotype_summary = build_phenotype_summary(long)

    print("[5/7] Writing horizontal coloured XLSX ...")
    write_horizontal_workbook(
        long=long,
        configs=configs,
        method_summary=method_summary,
        phenotype_summary=phenotype_summary,
        sample_size=data["sample_size"],
        file_audit=data["file_audit"],
    )

    print("[6/7] Creating figures — NO HEATMAP ...")
    figure_point_estimates(long)
    figure_ci_width(long)
    figure_crossfold_sd(long)

    figure_simulation_stability(
        data["sample_size"],
        trait_type="binary",
        scale="liability_binary",
        stem="Figure_Simulation_Binary_Stability_N750_vs_N5000",
        title="Binary simulations: liability-scale cross-fold stability",
    )

    figure_simulation_stability(
        data["sample_size"],
        trait_type="continuous",
        scale="native_quantitative",
        stem="Figure_Simulation_Continuous_Stability_N750_vs_N5000",
        title="Continuous simulations: cross-fold stability",
    )

    print("[7/7] Writing manuscript captions / Supplementary Table S1 description ...")
    write_text_outputs()

    print()
    print("="*115)
    print("DONE")
    print("="*115)

    print("\nMAIN WORKBOOK:")
    print(" ", OUT_XLSX)

    print("\nMAIN FIGURE CANDIDATES:")
    print(" ", OUTPUT_DIR / "Figure_PointEstimate_Distribution.png")
    print(" ", OUTPUT_DIR / "Figure_Estimator_CIWidth.png")

    print("\nSUPPLEMENTARY FIGURES:")
    print(" ", OUTPUT_DIR / "Figure_CrossFold_SD.png")
    print(" ", OUTPUT_DIR / "Figure_Simulation_Binary_Stability_N750_vs_N5000.png")
    print(" ", OUTPUT_DIR / "Figure_Simulation_Continuous_Stability_N750_vs_N5000.png")

    print("\nTEXT:")
    print(" ", OUTPUT_DIR / "Manuscript_Figure_Captions.tex")
    print(" ", OUTPUT_DIR / "Supplementary_Table_S1_Description.tex")

    print("\nIMPORTANT:")
    print(" * REMOVE the old point-estimate heatmap from the main manuscript.")
    print(" * Use Horizontal_S1 as the replacement full Supplementary Table S1.")
    print(" * Do NOT call cross-fold SD an estimator SE.")
    print(" * Do NOT call the median fold-level estimator CI a CI for the fold-averaged mean.")
    print(" * Keep unavailable estimator uncertainty as NA.")
    print(" * Keep negative and >1 estimates untruncated.")


if __name__ == "__main__":
    main()
 
