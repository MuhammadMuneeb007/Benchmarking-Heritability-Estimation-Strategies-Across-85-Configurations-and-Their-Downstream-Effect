#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinalAnalysis09-CorrelationAudit-PrintOnly.py
=============================================

PURPOSE
-------
This is a PRINT-ONLY final heritability audit.

It DOES NOT save CSV/XLSX/PNG/PDF files.
It only reads the final analysis outputs, recomputes the remaining
heritability-related correlation/audit values, and PRINTS them to the terminal.

It is designed to give the exact values needed to update the manuscript.

MAIN OUTPUTS
------------
A. EMPIRICAL / REAL PHENOTYPES
   1. Inter-method Pearson correlation matrix on the PRIMARY reporting scale:
        - binary phenotypes = liability scale
        - BMI = native quantitative scale
   2. Inter-method Spearman matrix
   3. Ranked method-pair correlations with N and p-values
   4. Method-family correlation matrix
   5. Same-family vs cross-family correlation comparison
   6. Friedman test for method-profile differences where complete phenotypes exist

B. SIMULATED PHENOTYPES
   1. Inter-method correlations for:
        - continuous, N=750
        - continuous, N=5000
        - continuous, overall paired
        - binary liability, N=750
        - binary liability, N=5000
        - binary liability, overall paired
   2. Correlation of each method with the TRUE generating h2
      for every simulation stratum above.

C. SNP COUNT vs h2
   Uses FinalAnalysis_Liability/FinalResults_Liability_Long.csv if present.
   Recalculates the SNP-count association on the PRIMARY empirical scale.
   Prints:
        - overall Spearman
        - per method family
        - per method group
        - per phenotype
        - BH-FDR q-values for family/phenotype testing families

D. FINAL HERITABILITY VERIFICATION
   Prints and verifies:
        - empirical negative / 0--1 / >1 counts
        - simulation boundary counts at N=750 and N=5000
        - estimator SE availability
        - output completeness
        - sample-size cross-fold stability
        - observed-vs-liability scale correlations
        - known-truth best configuration / method summaries
   It prints PASS/FAIL against the headline values currently used in the revision.

NO FILE IS WRITTEN BY THIS SCRIPT.

RUN
---
cd /data/ascher01/uqmmune1/finalheritability
python FinalAnalysis09-CorrelationAudit-PrintOnly.py
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# PATHS
# =============================================================================

def locate_root() -> Path:
    env = os.environ.get("HERITABILITY_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "FinalAnalysis06-UncertaintyReliability").exists():
            return p

    candidates = [
        Path.cwd().resolve(),
        Path.cwd().resolve().parent,
        Path("/data/ascher01/uqmmune1/finalheritability"),
    ]
    for p in candidates:
        if (p / "FinalAnalysis06-UncertaintyReliability").exists():
            return p

    raise FileNotFoundError(
        "Cannot locate FinalAnalysis06-UncertaintyReliability. "
        "Run this from /data/ascher01/uqmmune1/finalheritability "
        "or set HERITABILITY_ROOT."
    )


ROOT = locate_root()
A06 = ROOT / "FinalAnalysis06-UncertaintyReliability"

UNITS_FILE = A06 / "FinalAnalysis06_unit_stability.csv"
RAW_FILE = A06 / "FinalAnalysis06_raw_scale_rows.csv"
BOUNDARY_FILE = A06 / "FinalAnalysis06_boundary_summary.csv"
UNCERTAINTY_FILE = A06 / "FinalAnalysis06_uncertainty_availability.csv"
STABILITY_FILE = A06 / "FinalAnalysis06_sample_size_stability.csv"
FILE_AUDIT_FILE = A06 / "FinalAnalysis06_file_audit.csv"

LIABILITY_LONG_CANDIDATES = [
    ROOT / "FinalAnalysis_Liability" / "FinalResults_Liability_Long.csv",
    ROOT / "FinalAnalysis_Liability" / "FinalResults_Liability_Long_Compact.csv",
]


# =============================================================================
# ORDERS / MAPPINGS
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

FAMILY_NORMALISE = {
    "LDpred2": "LDpred2",
    "LDpred-2": "LDpred2",
    "GCTA": "GCTA",
    "GEMMA": "GEMMA",
    "Gemma": "GEMMA",
    "DPR+GEMMA": "DPR+GEMMA",
    "DPR-GEMMA": "DPR+GEMMA",
    "LDSC": "LDSC",
    "LDAK": "LDAK",
    "SumHer": "LDAK",
}


# =============================================================================
# FORMATTING / STATS HELPERS
# =============================================================================

LINE = "=" * 118
SUB = "-" * 118

def section(title: str):
    print()
    print(LINE)
    print(title)
    print(LINE)

def subsection(title: str):
    print()
    print(SUB)
    print(title)
    print(SUB)

def fnum(x):
    try:
        y = float(x)
        return y if np.isfinite(y) else np.nan
    except Exception:
        return np.nan

def fmt(x, digits=6):
    x = fnum(x)
    if not np.isfinite(x):
        return "NA"
    if x != 0 and abs(x) < 1e-4:
        return f"{x:.3e}"
    return f"{x:.{digits}f}"

def bh_fdr(pvalues):
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out

    vals = p[ok]
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)

    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)

    tmp = np.empty(m)
    tmp[order] = q
    out[np.where(ok)[0]] = tmp
    return out

def pearson_pair(x, y):
    z = pd.DataFrame({
        "x": pd.to_numeric(x, errors="coerce"),
        "y": pd.to_numeric(y, errors="coerce"),
    }).dropna()
    if len(z) < 3 or z["x"].nunique() < 2 or z["y"].nunique() < 2:
        return np.nan, np.nan, len(z)
    r, p = stats.pearsonr(z["x"], z["y"])
    return float(r), float(p), len(z)

def spearman_pair(x, y):
    z = pd.DataFrame({
        "x": pd.to_numeric(x, errors="coerce"),
        "y": pd.to_numeric(y, errors="coerce"),
    }).dropna()
    if len(z) < 3 or z["x"].nunique() < 2 or z["y"].nunique() < 2:
        return np.nan, np.nan, len(z)
    r, p = stats.spearmanr(z["x"], z["y"])
    return float(r), float(p), len(z)

def wilcoxon_p(x, y):
    z = pd.DataFrame({
        "x": pd.to_numeric(x, errors="coerce"),
        "y": pd.to_numeric(y, errors="coerce"),
    }).dropna()
    if len(z) == 0:
        return np.nan
    d = z["x"] - z["y"]
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(z["x"], z["y"], alternative="two-sided").pvalue)
    except Exception:
        return np.nan

def normalise_family(x):
    s = str(x)
    return FAMILY_NORMALISE.get(s, s)

def print_matrix(df: pd.DataFrame, digits=3):
    if df.empty:
        print("EMPTY")
        return
    with pd.option_context(
        "display.max_rows", 200,
        "display.max_columns", 200,
        "display.width", 240,
        "display.precision", digits,
    ):
        print(df.round(digits).to_string())

def print_table(df: pd.DataFrame, digits=6, max_rows=200):
    if df is None or df.empty:
        print("EMPTY")
        return
    with pd.option_context(
        "display.max_rows", max_rows,
        "display.max_columns", 100,
        "display.width", 260,
        "display.precision", digits,
    ):
        print(df.to_string(index=False))

def require(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return path


# =============================================================================
# LOAD FINAL ANALYSIS 06
# =============================================================================

def load_analysis06():
    return {
        "units": pd.read_csv(require(UNITS_FILE)),
        "raw": pd.read_csv(require(RAW_FILE)),
        "boundary": pd.read_csv(require(BOUNDARY_FILE)),
        "uncertainty": pd.read_csv(require(UNCERTAINTY_FILE)),
        "stability": pd.read_csv(require(STABILITY_FILE)),
        "file_audit": pd.read_csv(require(FILE_AUDIT_FILE)),
    }


# =============================================================================
# PRIMARY EMPIRICAL SCALE
# =============================================================================

def empirical_primary(units):
    e = units[units["Domain"].astype(str).str.lower() == "empirical"].copy()

    keep = (
        ((e["TraitType"] == "binary") & (e["Scale"] == "liability_binary"))
        |
        ((e["TraitType"] == "continuous") & (e["Scale"] == "native_quantitative"))
    )

    e = e[keep].copy()
    e["Phenotype"] = e["DatasetLabel"].astype(str)
    e["MeanH2"] = pd.to_numeric(e["MeanH2"], errors="coerce")
    return e


# =============================================================================
# PAIRWISE CORRELATION ENGINE
# =============================================================================

def pairwise_correlations(pivot: pd.DataFrame, family_by_method=None):
    rows = []

    for a, b in combinations(pivot.columns, 2):
        pr, pp, n = pearson_pair(pivot[a], pivot[b])
        sr, sp, ns = spearman_pair(pivot[a], pivot[b])

        same_family = np.nan
        fam_a = ""
        fam_b = ""

        if family_by_method is not None:
            fam_a = normalise_family(family_by_method.get(a, ""))
            fam_b = normalise_family(family_by_method.get(b, ""))
            same_family = fam_a == fam_b and fam_a != ""

        rows.append({
            "MethodA": a,
            "MethodB": b,
            "FamilyA": fam_a,
            "FamilyB": fam_b,
            "SameFamily": same_family,
            "N": n,
            "Pearson_r": pr,
            "Pearson_p": pp,
            "Spearman_rho": sr,
            "Spearman_p": sp,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["Pearson_BH_q"] = bh_fdr(out["Pearson_p"].to_numpy())
        out["Spearman_BH_q"] = bh_fdr(out["Spearman_p"].to_numpy())
    return out


# =============================================================================
# A. EMPIRICAL CORRELATION
# =============================================================================

def empirical_correlation_audit(units):
    section("A. EMPIRICAL / REAL-PHENOTYPE CORRELATION — PRIMARY SCALE")

    e = empirical_primary(units)

    method_family = (
        e[["Method", "MethodFamily"]]
        .dropna()
        .drop_duplicates("Method")
        .set_index("Method")["MethodFamily"]
        .to_dict()
    )

    # Average configurations within method x phenotype, which reproduces the
    # "method h2 profile across phenotypes" concept without mixing scales.
    prof = (
        e.groupby(["Phenotype", "Method"], dropna=False)
        .agg(MethodMeanH2=("MeanH2", "mean"))
        .reset_index()
    )

    pivot = (
        prof.pivot(index="Phenotype", columns="Method", values="MethodMeanH2")
        .reindex(index=PHENOTYPE_ORDER)
    )

    methods = [m for m in METHOD_ORDER if m in pivot.columns]
    other = [m for m in pivot.columns if m not in methods]
    pivot = pivot.reindex(columns=methods + other)

    short_cols = {m: METHOD_SHORT.get(m, m) for m in pivot.columns}

    subsection("A1. EMPIRICAL METHOD x PHENOTYPE PROFILE USED FOR CORRELATION")
    temp = pivot.rename(columns=short_cols)
    print_matrix(temp, digits=4)

    pearson = pivot.corr(method="pearson").rename(index=short_cols, columns=short_cols)
    spearman = pivot.corr(method="spearman").rename(index=short_cols, columns=short_cols)

    subsection("A2. EMPIRICAL PEARSON INTER-METHOD CORRELATION MATRIX")
    print_matrix(pearson, digits=3)

    subsection("A3. EMPIRICAL SPEARMAN INTER-METHOD CORRELATION MATRIX")
    print_matrix(spearman, digits=3)

    pairs = pairwise_correlations(pivot, family_by_method=method_family)
    pairs["MethodA_short"] = pairs["MethodA"].map(lambda x: METHOD_SHORT.get(x, x))
    pairs["MethodB_short"] = pairs["MethodB"].map(lambda x: METHOD_SHORT.get(x, x))

    show = pairs[
        [
            "MethodA_short", "MethodB_short", "N",
            "Pearson_r", "Pearson_p", "Pearson_BH_q",
            "Spearman_rho", "Spearman_p", "Spearman_BH_q",
            "FamilyA", "FamilyB", "SameFamily"
        ]
    ].sort_values("Pearson_r", ascending=False)

    subsection("A4. ALL EMPIRICAL METHOD PAIRS — RANKED BY PEARSON r")
    print_table(show, digits=6)

    valid = pairs.dropna(subset=["Pearson_r"]).copy()

    subsection("A5. EMPIRICAL HEADLINE VALUES TO REPLACE THE OLD CORRELATION TEXT")
    print(f"Total method pairs tested: {len(valid)}")
    print(f"Raw Pearson p<0.05 positive pairs: {int(((valid.Pearson_p < 0.05) & (valid.Pearson_r > 0)).sum())}")
    print(f"BH-FDR q<0.05 positive pairs:     {int(((valid.Pearson_BH_q < 0.05) & (valid.Pearson_r > 0)).sum())}")

    if len(valid):
        strongest = valid.sort_values("Pearson_r", ascending=False).iloc[0]
        weakest = valid.sort_values("Pearson_r", ascending=True).iloc[0]
        print(
            "Strongest pair:",
            METHOD_SHORT.get(strongest.MethodA, strongest.MethodA),
            "vs",
            METHOD_SHORT.get(strongest.MethodB, strongest.MethodB),
            f"r={fmt(strongest.Pearson_r,3)}, p={fmt(strongest.Pearson_p,6)}, "
            f"q={fmt(strongest.Pearson_BH_q,6)}, N={int(strongest.N)}"
        )
        print(
            "Weakest pair:",
            METHOD_SHORT.get(weakest.MethodA, weakest.MethodA),
            "vs",
            METHOD_SHORT.get(weakest.MethodB, weakest.MethodB),
            f"r={fmt(weakest.Pearson_r,3)}, p={fmt(weakest.Pearson_p,6)}, "
            f"q={fmt(weakest.Pearson_BH_q,6)}, N={int(weakest.N)}"
        )

    # Same-family vs cross-family
    sf = valid[valid["SameFamily"] == True]["Pearson_r"].dropna()
    cf = valid[valid["SameFamily"] == False]["Pearson_r"].dropna()

    if len(sf) and len(cf):
        u = stats.mannwhitneyu(sf, cf, alternative="two-sided")
        print()
        print("Same-family vs cross-family Pearson correlations:")
        print(f"  same-family N={len(sf)}, mean r={fmt(sf.mean(),3)}, median r={fmt(sf.median(),3)}")
        print(f"  cross-family N={len(cf)}, mean r={fmt(cf.mean(),3)}, median r={fmt(cf.median(),3)}")
        print(f"  Mann-Whitney U={fmt(u.statistic,3)}, p={fmt(u.pvalue,6)}")
    else:
        print("Same-family vs cross-family test unavailable: insufficient pair groups.")

    # Family-level profile
    family_prof = (
        e.assign(MethodFamilyNorm=e["MethodFamily"].map(normalise_family))
        .groupby(["Phenotype", "MethodFamilyNorm"], dropna=False)
        .agg(FamilyMeanH2=("MeanH2", "mean"))
        .reset_index()
    )

    family_pivot = (
        family_prof.pivot(
            index="Phenotype",
            columns="MethodFamilyNorm",
            values="FamilyMeanH2"
        )
        .reindex(index=PHENOTYPE_ORDER)
    )

    subsection("A6. EMPIRICAL METHOD-FAMILY PEARSON CORRELATION MATRIX")
    family_corr = family_pivot.corr(method="pearson")
    print_matrix(family_corr, digits=3)

    fam_pairs = pairwise_correlations(family_pivot)
    fam_show = fam_pairs[
        ["MethodA", "MethodB", "N", "Pearson_r", "Pearson_p", "Pearson_BH_q"]
    ].sort_values("Pearson_r", ascending=False)

    subsection("A7. EMPIRICAL FAMILY PAIRS — RANKED")
    print_table(fam_show, digits=6)

    # Friedman using phenotypes with all method values available.
    complete = pivot.dropna(axis=0, how="any")
    subsection("A8. EMPIRICAL FRIEDMAN METHOD-PROFILE TEST")
    print(f"Phenotypes complete across all methods: {len(complete)} / {len(pivot)}")
    if len(complete) >= 3 and complete.shape[1] >= 3:
        arrays = [complete[c].to_numpy(dtype=float) for c in complete.columns]
        fr = stats.friedmanchisquare(*arrays)
        print(f"Friedman chi2={fmt(fr.statistic,6)}, p={fmt(fr.pvalue,8)}")
    else:
        print("Not run: fewer than 3 complete phenotype rows across all methods.")

    return pivot, pairs


# =============================================================================
# B. SIMULATION CORRELATIONS
# =============================================================================

def simulation_profile(units, trait_type, scale, sample_size=None, overall=False):
    s = units[
        (units["Domain"].astype(str).str.lower() == "simulated")
        & (units["TraitType"] == trait_type)
        & (units["Scale"] == scale)
    ].copy()

    s["MeanH2"] = pd.to_numeric(s["MeanH2"], errors="coerce")
    s["TrueH2LiabilityOrQuantitative"] = pd.to_numeric(
        s["TrueH2LiabilityOrQuantitative"], errors="coerce"
    )

    # First average configurations within method for each underlying biological trait.
    by_n = (
        s.groupby(
            ["UnderlyingTrait", "SampleSize", "Method"],
            dropna=False
        )
        .agg(
            MethodMeanH2=("MeanH2", "mean"),
            TrueH2=("TrueH2LiabilityOrQuantitative", "first"),
        )
        .reset_index()
    )

    if overall:
        # Equal weight to nested N=750 and N=5000 for same trait.
        p = (
            by_n.groupby(["UnderlyingTrait", "Method"], dropna=False)
            .agg(
                MethodMeanH2=("MethodMeanH2", "mean"),
                TrueH2=("TrueH2", "first"),
                Nconditions=("SampleSize", "nunique"),
            )
            .reset_index()
        )
    else:
        p = by_n[pd.to_numeric(by_n["SampleSize"], errors="coerce") == float(sample_size)].copy()

    pivot = p.pivot(
        index="UnderlyingTrait",
        columns="Method",
        values="MethodMeanH2"
    )

    true = (
        p[["UnderlyingTrait", "TrueH2"]]
        .drop_duplicates("UnderlyingTrait")
        .set_index("UnderlyingTrait")["TrueH2"]
        .reindex(pivot.index)
    )

    methods = [m for m in METHOD_ORDER if m in pivot.columns]
    other = [m for m in pivot.columns if m not in methods]
    pivot = pivot.reindex(columns=methods + other)

    return p, pivot, true


def print_simulation_correlation_block(units, trait_type, scale, label):
    section(f"B. SIMULATION CORRELATIONS — {label}")

    strata = [
        ("N=750", 750, False),
        ("N=5000", 5000, False),
        ("OVERALL PAIRED", None, True),
    ]

    for title, n, overall in strata:
        subsection(f"{label} — {title}")

        p, pivot, true = simulation_profile(
            units,
            trait_type=trait_type,
            scale=scale,
            sample_size=n,
            overall=overall,
        )

        if pivot.empty:
            print("No data.")
            continue

        short = {m: METHOD_SHORT.get(m, m) for m in pivot.columns}

        print(f"Underlying traits represented: {pivot.shape[0]}")
        print(f"Methods represented: {pivot.shape[1]}")

        corr = pivot.corr(method="pearson").rename(index=short, columns=short)
        print()
        print("Pearson inter-method matrix:")
        print_matrix(corr, digits=3)

        pairs = pairwise_correlations(pivot)
        valid = pairs.dropna(subset=["Pearson_r"]).copy()

        if len(valid):
            strongest = valid.sort_values("Pearson_r", ascending=False).iloc[0]
            weakest = valid.sort_values("Pearson_r", ascending=True).iloc[0]

            print()
            print(
                "Strongest inter-method pair:",
                METHOD_SHORT.get(strongest.MethodA, strongest.MethodA),
                "vs",
                METHOD_SHORT.get(strongest.MethodB, strongest.MethodB),
                f"r={fmt(strongest.Pearson_r,3)}, p={fmt(strongest.Pearson_p,6)}, N={int(strongest.N)}"
            )
            print(
                "Weakest inter-method pair:",
                METHOD_SHORT.get(weakest.MethodA, weakest.MethodA),
                "vs",
                METHOD_SHORT.get(weakest.MethodB, weakest.MethodB),
                f"r={fmt(weakest.Pearson_r,3)}, p={fmt(weakest.Pearson_p,6)}, N={int(weakest.N)}"
            )

        # Method estimate profile vs TRUE h2.
        truth_rows = []
        for m in pivot.columns:
            pr, pp, nn = pearson_pair(pivot[m], true)
            sr, sp, ns = spearman_pair(pivot[m], true)
            truth_rows.append({
                "Method": METHOD_SHORT.get(m, m),
                "N": nn,
                "Pearson_r_vs_true_h2": pr,
                "Pearson_p": pp,
                "Spearman_rho_vs_true_h2": sr,
                "Spearman_p": sp,
            })

        truth_df = pd.DataFrame(truth_rows)
        truth_df["Pearson_BH_q"] = bh_fdr(truth_df["Pearson_p"].to_numpy())
        truth_df["Spearman_BH_q"] = bh_fdr(truth_df["Spearman_p"].to_numpy())
        truth_df = truth_df.sort_values("Pearson_r_vs_true_h2", ascending=False)

        print()
        print("Correlation of method-level estimates with TRUE generating h2:")
        print_table(truth_df, digits=6)


# =============================================================================
# C. EMPIRICAL SNP COUNT vs PRIMARY-SCALE h2
# =============================================================================

def locate_liability_long():
    for p in LIABILITY_LONG_CANDIDATES:
        if p.exists():
            return p
    return None

def detect_col(df, candidates):
    low = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in low:
            return low[candidate.lower()]
    return None

def detect_numeric_count_col(df, preferred_names, keywords=("variant", "snp")):
    """
    Select a genuinely numeric count column rather than a categorical column
    such as 'variants_standardized'.

    Preference order:
      1) exact preferred names with usable positive numeric values;
      2) other variant/SNP-named columns, ranked by number of positive numeric rows.
    """
    low = {str(c).strip().lower(): c for c in df.columns}

    scored = []

    # Exact preferred names first.
    for rank, name in enumerate(preferred_names):
        if name.lower() in low:
            col = low[name.lower()]
            vals = pd.to_numeric(df[col], errors="coerce")
            positive = int((vals > 0).sum())
            numeric = int(vals.notna().sum())
            scored.append((2, -rank, positive, numeric, col))

    # Broad fallback search.
    for col in df.columns:
        lname = str(col).strip().lower()
        if not any(k in lname for k in keywords):
            continue
        if "standardized" in lname or "standardised" in lname:
            # Usually a categorical descriptor, not a count.
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        positive = int((vals > 0).sum())
        numeric = int(vals.notna().sum())
        scored.append((1, 0, positive, numeric, col))

    if not scored:
        return None, pd.DataFrame()

    # Prefer exact-name candidates, then most positive numeric values.
    scored = sorted(scored, key=lambda x: (x[0], x[2], x[3], x[1]), reverse=True)

    audit = pd.DataFrame(
        scored,
        columns=["PreferenceClass", "NameRank", "PositiveNumericRows", "NumericRows", "Column"]
    ).drop_duplicates("Column")

    best = audit.iloc[0]
    if int(best["PositiveNumericRows"]) == 0:
        return None, audit

    return best["Column"], audit

def snp_count_audit():
    section("C. EMPIRICAL SNP COUNT vs PRIMARY-SCALE h2")

    path = locate_liability_long()
    if path is None:
        print("SKIPPED.")
        print("Could not find FinalAnalysis_Liability/FinalResults_Liability_Long.csv")
        print("This does NOT affect the inter-method correlation analysis above.")
        return None

    print("Reading:", path)
    d = pd.read_csv(path, low_memory=False)

    c_mode = detect_col(d, ["analysis_mode", "domain"])
    c_ph = detect_col(d, ["phenotype_standardized", "phenotype", "datasetlabel", "dataset"])
    c_tt = detect_col(d, ["trait_type", "traittype"])
    c_method = detect_col(d, ["method_standardized", "method"])
    c_family = detect_col(d, ["method_family", "methodfamily"])

    # IMPORTANT: do not use variants_standardized as a count. It is often categorical.
    preferred_variant_names = [
        "numberofvariants",
        "number_of_variants",
        "number_variants",
        "variant_count",
        "variants_count",
        "n_variants",
        "num_variants",
        "variants_used",
        "n_variants_used",
        "final_variant_count",
        "final_snp_count",
        "snp_count",
        "snps_used",
        "n_snps",
        "nsnps",
        "num_snps",
        "numberofsnps",
        "number_of_snps",
        "common_snps",
        "common_snps_count",
    ]
    c_var, var_audit = detect_numeric_count_col(
        d,
        preferred_variant_names,
        keywords=("variant", "snp"),
    )

    c_native = detect_col(d, ["h2_native", "heritabilityvalue", "heritability", "h2"])
    c_liab = detect_col(d, ["h2_liability_primary", "h2_liability", "liability_h2"])

    subsection("C0. COLUMN DETECTION AUDIT")
    print("Phenotype column :", c_ph)
    print("Trait-type column:", c_tt)
    print("Method column    :", c_method)
    print("Family column    :", c_family)
    print("Native h2 column :", c_native)
    print("Liability column :", c_liab)
    print("Variant-count col:", c_var)

    if not var_audit.empty:
        print()
        print("Variant/SNP numeric-column candidates:")
        print_table(var_audit, digits=0, max_rows=50)

    required = {
        "phenotype": c_ph,
        "trait_type": c_tt,
        "method": c_method,
        "variants": c_var,
        "h2_native": c_native,
        "h2_liability": c_liab,
    }

    missing = [k for k, v in required.items() if v is None]
    if missing:
        print()
        print("SNP-COUNT AUDIT SKIPPED because required columns were not detected:", missing)
        print("Relevant variant/SNP columns present in this file:")
        rel = [c for c in d.columns if ("variant" in str(c).lower() or "snp" in str(c).lower())]
        for c in rel:
            vals = pd.to_numeric(d[c], errors="coerce")
            print(
                f"  {c}: numeric={int(vals.notna().sum())}, "
                f"positive={int((vals > 0).sum())}, "
                f"examples={d[c].dropna().astype(str).head(5).tolist()}"
            )
        print()
        print("The rest of the final audit WILL CONTINUE.")
        return None

    if c_mode is not None:
        mode = d[c_mode].astype(str).str.lower()
        mask = mode.str.contains("emp|real", regex=True)
        if mask.any():
            d = d[mask].copy()

    d["Phenotype"] = d[c_ph].astype(str)
    d["TraitType"] = d[c_tt].astype(str).str.lower()
    d["Method"] = d[c_method].astype(str)
    d["Variants"] = pd.to_numeric(d[c_var], errors="coerce")
    d["H2Native"] = pd.to_numeric(d[c_native], errors="coerce")
    d["H2Liability"] = pd.to_numeric(d[c_liab], errors="coerce")

    # Normalise obvious trait labels.
    d["TraitType"] = np.where(
        d["TraitType"].str.contains("binary|case|disease", regex=True),
        "binary",
        np.where(
            d["TraitType"].str.contains("continuous|quantitative|bmi", regex=True),
            "continuous",
            d["TraitType"],
        )
    )

    d["H2Primary"] = np.where(
        d["TraitType"].eq("binary"),
        d["H2Liability"],
        d["H2Native"],
    )

    if c_family is not None:
        d["MethodFamily"] = d[c_family].astype(str).map(normalise_family)
    else:
        method_to_family = {
            "LDpred-2-Heritability": "LDpred2",
            "GCTA-Heritability": "GCTA",
            "Gemma1-Heritability": "GEMMA",
            "Gemma2-Heritability": "GEMMA",
            "Gemma-LDSC-Heritability": "GEMMA",
            "DPR-Gemma-Heritability": "DPR+GEMMA",
            "LDSC-Heritability": "LDSC",
            "LDAK-Calculated-Heritability": "LDAK",
            "LDAK-Precomputed-Heritability": "LDAK",
            "LDSC-Genotype-Heritability": "LDSC",
        }
        d["MethodFamily"] = d["Method"].map(method_to_family).fillna(d["Method"])

    config_candidates = [
        detect_col(d, ["h2model_standardized", "h2model"]),
        detect_col(d, ["clumpprune"]),
        detect_col(d, ["numberofpca", "number_of_pca", "pca"]),
        detect_col(d, ["configid", "configuration_id"]),
    ]
    config_cols = []
    for c in config_candidates:
        if c is not None and c not in config_cols:
            config_cols.append(c)

    grp = ["Phenotype", "TraitType", "Method", "MethodFamily"] + config_cols

    # Fold rows -> one configuration-phenotype row.
    config = (
        d.groupby(grp, dropna=False)
        .agg(
            MeanH2Primary=("H2Primary", "mean"),
            MeanVariants=("Variants", "mean"),
            ValidRows=("H2Primary", lambda z: int(pd.Series(z).notna().sum())),
        )
        .reset_index()
    )

    before = len(config)
    config = config[
        config["MeanH2Primary"].notna()
        & config["MeanVariants"].notna()
        & (config["MeanVariants"] > 0)
    ].copy()

    print()
    print(f"Configuration-phenotype groups before numeric filtering: {before}")
    print(f"Usable configuration-phenotype rows: {len(config)}")

    if config.empty:
        print()
        print("SNP-COUNT AUDIT SKIPPED: selected columns produced zero usable rows.")
        print("Diagnostics:")
        print("  TraitType value counts:")
        print(d["TraitType"].value_counts(dropna=False).head(20).to_string())
        print("  Variants numeric summary:")
        print(d["Variants"].describe().to_string())
        print("  H2Native non-missing:", int(d["H2Native"].notna().sum()))
        print("  H2Liability non-missing:", int(d["H2Liability"].notna().sum()))
        print("  H2Primary non-missing:", int(pd.Series(d["H2Primary"]).notna().sum()))
        print()
        print("The rest of the final audit WILL CONTINUE.")
        return None

    r, p, n = spearman_pair(config["MeanVariants"], config["MeanH2Primary"])
    print()
    print(f"OVERALL PRIMARY-SCALE Spearman rho={fmt(r,4)}, p={fmt(p,8)}, N={n}")

    rows = []
    for fam, g in config.groupby("MethodFamily", dropna=False):
        r, p, n = spearman_pair(g["MeanVariants"], g["MeanH2Primary"])
        rows.append({
            "Family": fam,
            "N": n,
            "Spearman_rho": r,
            "p": p,
        })

    fam_df = pd.DataFrame(rows, columns=["Family", "N", "Spearman_rho", "p"])
    if not fam_df.empty:
        fam_df["BH_q"] = bh_fdr(fam_df["p"].to_numpy())
        fam_df = fam_df.sort_values("Spearman_rho", ascending=False)

    subsection("C1. SNP COUNT ASSOCIATION — PER METHOD FAMILY")
    print_table(fam_df, digits=6)

    rows = []
    for method, g in config.groupby("Method", dropna=False):
        r, p, n = spearman_pair(g["MeanVariants"], g["MeanH2Primary"])
        rows.append({
            "Method": METHOD_SHORT.get(method, method),
            "N": n,
            "Spearman_rho": r,
            "p": p,
        })

    method_df = pd.DataFrame(rows, columns=["Method", "N", "Spearman_rho", "p"])
    if not method_df.empty:
        method_df["BH_q"] = bh_fdr(method_df["p"].to_numpy())
        method_df = method_df.sort_values("Spearman_rho", ascending=False)

    subsection("C2. SNP COUNT ASSOCIATION — PER METHOD GROUP")
    print_table(method_df, digits=6)

    rows = []
    for ph, g in config.groupby("Phenotype", dropna=False):
        r, p, n = spearman_pair(g["MeanVariants"], g["MeanH2Primary"])
        rows.append({
            "Phenotype": ph,
            "N": n,
            "Spearman_rho": r,
            "p": p,
        })

    ph_df = pd.DataFrame(rows, columns=["Phenotype", "N", "Spearman_rho", "p"])
    if not ph_df.empty:
        ph_df["BH_q"] = bh_fdr(ph_df["p"].to_numpy())
        ph_df = ph_df.sort_values("Spearman_rho", ascending=False)

    subsection("C3. SNP COUNT ASSOCIATION — PER PHENOTYPE")
    print_table(ph_df, digits=6)

    print()
    print("IMPORTANT INTERPRETATION:")
    print(
        "The variant count here is the method-reported/effective variant count. "
        "Counts are not necessarily directly comparable across tools because software-specific "
        "internal filtering can change the final number of contributing variants."
    )
    print(
        "Use this as a DESCRIPTIVE association only. The exact one-factor SNP-set "
        "analysis is stronger evidence for the effect of changing the SNP set."
    )

    return config, fam_df, method_df, ph_df


# =============================================================================
# D. BOUNDARY / UNCERTAINTY / COMPLETENESS VERIFICATION
# =============================================================================

EXPECTED_BOUNDARY = {
    ("empirical", "binary", "observed_binary", None): (759, 121, 638, 0),
    ("empirical", "binary", "liability_binary", None): (729, 118, 467, 144),
    ("empirical", "continuous", "native_quantitative", None): (85, 11, 73, 1),

    ("simulated", "continuous", "native_quantitative", 750): (3900, 227, 3465, 208),
    ("simulated", "continuous", "native_quantitative", 5000): (3900, 108, 3590, 202),

    ("simulated", "binary", "observed_binary", 750): (3900, 384, 3391, 125),
    ("simulated", "binary", "observed_binary", 5000): (3900, 256, 3519, 125),

    ("simulated", "binary", "liability_binary", 750): (3900, 384, 3231, 285),
    ("simulated", "binary", "liability_binary", 5000): (3900, 256, 3459, 185),
}

def classify_boundary(series):
    h = pd.to_numeric(series, errors="coerce").dropna()
    return (
        len(h),
        int((h < 0).sum()),
        int(((h >= 0) & (h <= 1)).sum()),
        int((h > 1).sum()),
    )

def verify_boundaries(units):
    section("D. FINAL BOUNDARY / RANGE VERIFICATION")

    rows = []
    all_pass = True

    for key, expected in EXPECTED_BOUNDARY.items():
        domain, tt, scale, n = key

        z = units[
            (units["Domain"].astype(str).str.lower() == domain)
            & (units["TraitType"] == tt)
            & (units["Scale"] == scale)
        ].copy()

        if n is not None:
            z = z[pd.to_numeric(z["SampleSize"], errors="coerce") == float(n)]

        observed = classify_boundary(z["MeanH2"])
        ok = observed == expected
        all_pass = all_pass and ok

        rows.append({
            "Domain": domain,
            "TraitType": tt,
            "Scale": scale,
            "N_target": n if n is not None else "",
            "N": observed[0],
            "h2<0": observed[1],
            "0<=h2<=1": observed[2],
            "h2>1": observed[3],
            "Expected": str(expected),
            "PASS": ok,
        })

    out = pd.DataFrame(rows)
    print_table(out, digits=4)
    print()
    print("BOUNDARY AUDIT OVERALL:", "PASS" if all_pass else "FAIL — inspect before updating manuscript")
    return all_pass


def print_uncertainty_and_completeness(data):
    section("E. UNCERTAINTY / COMPLETENESS / SAMPLE-SIZE STABILITY — FINAL VALUES")

    uncertainty = data["uncertainty"].copy()
    subsection("E1. ESTIMATOR-SPECIFIC SE / CI AVAILABILITY")
    print_table(uncertainty, digits=6)

    file_audit = data["file_audit"].copy()
    subsection("E2. FILE / OUTPUT COMPLETENESS")
    print_table(file_audit, digits=6)

    stability = data["stability"].copy()
    subsection("E3. N=750 vs N=5000 CROSS-FOLD STABILITY — ALL METHODS")
    print_table(stability, digits=6)

    # Headline methods for easy manuscript checking.
    if "Method" in stability.columns:
        names = {
            "GCTA-Heritability",
            "Gemma1-Heritability",
            "DPR-Gemma-Heritability",
        }
        headline = stability[stability["Method"].isin(names)].copy()
        subsection("E4. HEADLINE STABILITY VALUES USED IN THE MANUSCRIPT")
        print_table(headline, digits=9)


# =============================================================================
# F. OBSERVED-vs-LIABILITY SCALE AUDIT
# =============================================================================

def scale_audit(units):
    section("F. OBSERVED vs LIABILITY SCALE CORRELATION AUDIT")

    b = units[units["TraitType"] == "binary"].copy()

    keys = [
        "Domain", "Dataset", "SampleSize", "UnderlyingTrait",
        "Method", "ConfigID",
    ]

    obs = b[b["Scale"] == "observed_binary"][
        keys + ["MeanH2", "FoldSD"]
    ].copy()

    liab = b[b["Scale"] == "liability_binary"][
        keys + ["MeanH2", "FoldSD"]
    ].copy()

    z = obs.merge(
        liab,
        on=keys,
        how="inner",
        suffixes=("_Observed", "_Liability"),
    )

    rows = []
    for (domain, n), g in z.groupby(["Domain", "SampleSize"], dropna=False):
        hr, hp, hn = spearman_pair(g["MeanH2_Observed"], g["MeanH2_Liability"])
        sr, sp, sn = spearman_pair(g["FoldSD_Observed"], g["FoldSD_Liability"])
        rows.append({
            "Domain": domain,
            "SampleSize": n,
            "MatchedUnits": len(g),
            "Spearman_MeanH2_ObservedVsLiability": hr,
            "p_h2": hp,
            "Spearman_FoldSD_ObservedVsLiability": sr,
            "p_foldsd": sp,
        })

    print_table(pd.DataFrame(rows), digits=9)


# =============================================================================
# G. KNOWN-TRUTH ACCURACY RECHECK
# =============================================================================

EXPECTED_BEST = {
    ("continuous", "native_quantitative", "750"): {
        "best_method": "LDSC-Heritability",
        "best_config_mae_approx": 0.0378,
    },
    ("continuous", "native_quantitative", "5000"): {
        "best_method": "GCTA-Heritability",
        "best_config_mae_approx": 0.0200,
    },
    ("continuous", "native_quantitative", "overall"): {
        "best_method": "GCTA-Heritability",
        "best_config_mae_approx": 0.0332,
    },
    ("binary", "liability_binary", "750"): {
        "best_method": "LDSC-Heritability",
        "best_config_mae_approx": 0.04408,
    },
    ("binary", "liability_binary", "5000"): {
        "best_method": "LDSC-Heritability",
        "best_config_mae_approx": 0.04377,
    },
    ("binary", "liability_binary", "overall"): {
        "best_method": "LDSC-Heritability",
        "best_config_mae_approx": 0.04392,
    },
}

def known_truth_table(units, trait_type, scale, sample_size=None, overall=False):
    s = units[
        (units["Domain"].astype(str).str.lower() == "simulated")
        & (units["TraitType"] == trait_type)
        & (units["Scale"] == scale)
    ].copy()

    s["MeanH2"] = pd.to_numeric(s["MeanH2"], errors="coerce")
    s["TrueH2"] = pd.to_numeric(
        s["TrueH2LiabilityOrQuantitative"], errors="coerce"
    )
    s["AE"] = (s["MeanH2"] - s["TrueH2"]).abs()
    s["SignedError"] = s["MeanH2"] - s["TrueH2"]

    if overall:
        # Average errors across nested N within biological trait/configuration.
        a = (
            s.groupby(
                ["UnderlyingTrait", "Method", "ConfigID"],
                dropna=False
            )
            .agg(
                AE=("AE", "mean"),
                SignedError=("SignedError", "mean"),
                TrueH2=("TrueH2", "first"),
                Nconditions=("SampleSize", "nunique"),
            )
            .reset_index()
        )
    else:
        a = s[pd.to_numeric(s["SampleSize"], errors="coerce") == float(sample_size)].copy()

    config = (
        a.groupby(["Method", "ConfigID"], dropna=False)
        .agg(
            N=("AE", "count"),
            MAE=("AE", "mean"),
            Bias=("SignedError", "mean"),
            MedianAE=("AE", "median"),
        )
        .reset_index()
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    method = (
        a.groupby("Method", dropna=False)
        .agg(
            N=("AE", "count"),
            MAE=("AE", "mean"),
            Bias=("SignedError", "mean"),
        )
        .reset_index()
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    return a, config, method


def verify_known_truth(units):
    section("G. KNOWN-TRUTH HEADLINE VALUE RECHECK")

    rows = []

    for tt, scale in [
        ("continuous", "native_quantitative"),
        ("binary", "liability_binary"),
    ]:
        for label, n, overall in [
            ("750", 750, False),
            ("5000", 5000, False),
            ("overall", None, True),
        ]:
            a, config, method = known_truth_table(
                units, tt, scale, sample_size=n, overall=overall
            )

            if config.empty:
                continue

            best = config.iloc[0]
            exp = EXPECTED_BEST[(tt, scale, label)]

            method_ok = best["Method"] == exp["best_method"]
            mae_ok = abs(float(best["MAE"]) - exp["best_config_mae_approx"]) <= 0.0015

            rows.append({
                "TraitType": tt,
                "Scale": scale,
                "Analysis": label,
                "BestMethod": METHOD_SHORT.get(best["Method"], best["Method"]),
                "BestConfigMAE": float(best["MAE"]),
                "ExpectedBestMethod": METHOD_SHORT.get(exp["best_method"], exp["best_method"]),
                "ExpectedMAE_approx": exp["best_config_mae_approx"],
                "MethodPASS": method_ok,
                "MAEPASS": mae_ok,
                "BestMethodLevel": METHOD_SHORT.get(method.iloc[0]["Method"], method.iloc[0]["Method"]),
                "BestMethodLevelMAE": float(method.iloc[0]["MAE"]),
            })

    out = pd.DataFrame(rows)
    print_table(out, digits=6)

    print()
    print(
        "NOTE: The configuration-level MAE check uses a tolerance because manuscript values "
        "are rounded. If MethodPASS=True and MAEPASS=True, the headline known-truth result "
        "is consistent with the final unit-level data."
    )


# =============================================================================
# H. PRINT WHAT IS STILL PENDING IN THE HERITABILITY TEXT
# =============================================================================

def pending_text_checklist():
    section("H. WHAT IS STILL PENDING TO UPDATE IN THE HERITABILITY TEXT")

    print("1. INTER-METHOD CORRELATION SECTION")
    print("   Replace the OLD empirical/native-scale values with section A above.")
    print("   The primary empirical scale must be liability for binary phenotypes + native BMI.")
    print("   Replace the old Figure inter-correlation numbers/caption with the new primary-scale values.")
    print()
    print("2. SNP COUNT SECTION")
    print("   If section C ran successfully, replace the old r=0.330 / family / phenotype values.")
    print("   Describe this as a DESCRIPTIVE association with reported/effective variant count.")
    print("   Do not treat it as stronger causal evidence than the exact one-factor SNP-set analysis.")
    print()
    print("3. SIMULATION CORRELATION")
    print("   Section B gives the new continuous and binary-liability inter-method matrices.")
    print("   It also prints each method's correlation with TRUE h2.")
    print("   This can be used in Supplementary Material or to support the revised correlation discussion.")
    print()
    print("4. COMMENT 3 / UNCERTAINTY")
    print("   Keep estimator SE/CI separate from cross-fold SD/MAD/IQR/range.")
    print("   Remove any remaining fold-SE = SD/sqrt(5) or fold-derived 95% CI wording.")
    print("   Do not use significant_fold.")
    print()
    print("5. HIGH CHOLESTEROL")
    print("   Do not fill summary-statistic liability values when exact P_GWAS is unavailable.")
    print("   Missing liability values must remain missing; never fall back to observed scale in primary analysis.")
    print()
    print("6. FIGURES / FINAL S1")
    print("   Figures can be updated later, but the final S1 should report:")
    print("      mean h2 | estimator SE | estimator 95% CI | cross-fold SD")
    print("   for observed/native and liability scales where applicable.")
    print()
    print("If A, C, and all PASS audits above are clean, there is no additional heritability ANALYSIS to run.")
    print("At that point only manuscript replacement text / figures / final response-letter synchronization remain.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(LINE)
    print("FINAL ANALYSIS 09 — HERITABILITY CORRELATION + FINAL VALUE AUDIT (PRINT ONLY)")
    print(LINE)
    print("ROOT:", ROOT)
    print("NO FILES WILL BE WRITTEN.")
    print()

    data = load_analysis06()
    units = data["units"]

    print(f"Loaded unit-level rows: {len(units):,}")
    print(f"Empirical units: {(units['Domain'].astype(str).str.lower() == 'empirical').sum():,}")
    print(f"Simulated units: {(units['Domain'].astype(str).str.lower() == 'simulated').sum():,}")

    # A. Real phenotype correlation
    empirical_correlation_audit(units)

    # B. Simulated correlations
    print_simulation_correlation_block(
        units,
        trait_type="continuous",
        scale="native_quantitative",
        label="CONTINUOUS / NATIVE QUANTITATIVE",
    )

    print_simulation_correlation_block(
        units,
        trait_type="binary",
        scale="liability_binary",
        label="BINARY / LIABILITY SCALE",
    )

    # C. SNP count association
    try:
        snp_count_audit()
    except Exception as exc:
        section("C. EMPIRICAL SNP COUNT vs PRIMARY-SCALE h2")
        print("SNP-count audit raised an exception but the final audit will continue.")
        print("Exception:", repr(exc))

    # D-G verification
    verify_boundaries(units)
    print_uncertainty_and_completeness(data)
    scale_audit(units)
    verify_known_truth(units)

    # H pending items
    pending_text_checklist()

    print()
    print(LINE)
    print("END OF PRINT-ONLY AUDIT")
    print(LINE)


if __name__ == "__main__":
    main()
