#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinalAnalysis04-XX.py
=====================

Unified empirical-reference and simulated-known-truth heritability accuracy
analysis.

PRIMARY SCIENTIFIC DISTINCTION
------------------------------
SIMULATED DATA have genuine known generating heritability.

EMPIRICAL DATA do NOT have ground truth. Published SNP-heritability values are
therefore used only as EXTERNAL LITERATURE REFERENCE ANCHORS. Two empirical
reference sets are analysed separately:

    PRIMARY:
        publication-distinct reference chosen, where feasible, to avoid using
        the exact GWAS-source publication used by the benchmark.

    SECONDARY:
        alternative literature anchor used only for reference-sensitivity
        analysis. Some secondary anchors intentionally include publications
        that overlap with, or are identical to, a benchmark GWAS-source paper.

The primary and secondary anchors are NEVER averaged together.

TRAIT / SCALE ANALYSIS
----------------------
SIMULATED:
    continuous:
        native quantitative h2 vs known generating h2

    binary liability:
        h2_liability_primary vs known generating liability h2

    binary observed:
        native/observed h2 vs the SAME generating h2 projected to the
        row-specific observed scale

EMPIRICAL:
    quantitative BMI:
        native h2 vs external quantitative literature anchor

    binary liability:
        liability h2 vs scale-matched external literature anchor

    binary observed:
        native h2 vs scale-matched observed literature anchor

For binary literature anchors originally reported on the LIABILITY scale:
    reference_observed = reference_liability / liability_factor_primary

For binary literature anchors originally reported on the OBSERVED scale:
    reference_liability = reference_observed * liability_factor_primary

CRITICAL SCALE-INVARIANCE POINT
-------------------------------
If estimate and reference are transformed consistently by the same positive
factor C,

    h_L = C * h_O
    r_L = C * r_O

then relative absolute error is algebraically invariant:

    |h_L-r_L|/|r_L| = |h_O-r_O|/|r_O|.

Therefore this script DOES NOT perform a Wilcoxon test asking whether observed
or liability scale is "more accurate". Such a test would be scientifically
misleading. Instead it performs a SCALE-INVARIANCE AUDIT:
    * maximum difference in matched relative error
    * Spearman concordance
    * method-rank preservation
    * configuration-rank preservation

The meaningful inferential question is which METHOD / CONFIGURATION has lower
error from a scale-matched reference, not which mathematical representation is
"closer".

INFERENCE
---------
Accuracy ranking:
    lower absolute error = closer to the appropriate reference.

Configuration-level and method-level:
    * global Friedman test where enough complete paired units exist
    * all pairwise two-sided Wilcoxon signed-rank tests
    * Benjamini-Hochberg FDR correction within each analysis family

Bias:
    signed error = estimate - reference
    * one-sample two-sided Wilcoxon signed-rank test against zero
    * BH-FDR across groups within each analysis family

IMPORTANT:
    Failure to reject zero bias is NOT evidence of statistical equivalence.

INDEPENDENT UNITS
-----------------
EMPIRICAL BINARY:
    phenotype is the inferential unit.

EMPIRICAL CONTINUOUS:
    BMI is the only phenotype; rankings are descriptive only.

SIMULATED, BY N:
    underlying simulated trait at that N is the unit.

SIMULATED, OVERALL:
    N=750 and N=5000 are nested versions of the same underlying trait.
    Errors are averaged within underlying trait before overall inference.

MISSINGNESS / BOUNDARIES
------------------------
* Reads Results_Liability.csv.
* Never substitutes native h2 when liability h2 is unavailable.
* Never compares native binary h2 directly with a liability-scale reference.
* Negative estimates and estimates >1 are retained without clipping.

USAGE
-----
    python FinalAnalysis04-XX.py

Optional:
    HERITABILITY_ROOT=/path/to/finalheritability python FinalAnalysis04-XX.py

DEFAULT ROOT
------------
    /data/ascher01/uqmmune1/finalheritability

OUTPUT DIRECTORY
----------------
    FinalAnalysis04-XX/

Main outputs:
    FinalAnalysis04-XX.xlsx
    FinalAnalysis04-XX_MAIN_RESULTS.txt

CSV outputs include:
    empirical_references
    fold_rows
    units
    analysis_units
    configuration_summary
    method_summary
    configuration_pairwise
    method_pairwise
    global_tests
    top_accuracy
    scale_invariance
    reference_set_comparison
    excluded
"""


from __future__ import annotations

import itertools
import math
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PATHS / GENERAL SETTINGS
# =============================================================================

ROOT = Path(
    os.environ.get(
        "HERITABILITY_ROOT",
        "/data/ascher01/uqmmune1/finalheritability",
    )
).resolve()

OUTDIR = ROOT / "FinalAnalysis04-XX"

MIN_COVERAGE = 0.80
MIN_WILCOXON_N = 5
FDR_ALPHA = 0.05


# =============================================================================
# 2. DATASETS
# =============================================================================

REAL_PHENOTYPES = [
    "asthma",
    "body_mass_index_bmi",
    "depression",
    "gastro_oesophageal_reflux_gord_gastric_reflux",
    "high_cholesterol",
    "hypertension",
    "hypothyroidism_myxoedema",
    "irritable_bowel_syndrome",
    "migraine",
    "osteoarthritis",
]

PHENOTYPE_LABELS = {
    "asthma": "Asthma",
    "body_mass_index_bmi": "Body Mass Index",
    "depression": "Depression",
    "gastro_oesophageal_reflux_gord_gastric_reflux":
        "Gastro-Oesophageal Reflux",
    "high_cholesterol": "High Cholesterol",
    "hypertension": "Hypertension",
    "hypothyroidism_myxoedema": "Hypothyroidism",
    "irritable_bowel_syndrome": "Irritable Bowel Syndrome",
    "migraine": "Migraine",
    "osteoarthritis": "Osteoarthritis",
}

METHODS = [
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


# =============================================================================
# 3. EMPIRICAL EXTERNAL LITERATURE REFERENCE SETS
# =============================================================================
#
# These are EXTERNAL LITERATURE ANCHORS, NOT empirical ground truth.
#
# PRIMARY set:
#   Chosen to use a publication distinct from the benchmark GWAS-source paper
#   where feasible. A distinct publication does NOT guarantee participant-level
#   independence; several large European studies include UK Biobank.
#
# SECONDARY set:
#   Alternative literature estimate used for reference-sensitivity analysis.
#   It is analysed separately and NEVER averaged with the primary anchor.
#
# Scale:
#   liability    = reported liability-scale SNP h2
#   observed     = reported observed/raw binary SNP h2
#   quantitative = quantitative-trait SNP h2
#
# NOTE: High-cholesterol secondary Hou_2019 = 0.082 is treated as OBSERVED.
#       The primary high-cholesterol 0.189 anchor is the liability-scale
#       previously reported MMHE value shown in Tang_2022 / Ge_2017.
# =============================================================================

EMPIRICAL_REFERENCE_SETS = {
    "asthma": {
        "primary": {
            "reference_h2": 0.2100,
            "reference_scale": "liability",
            "citation_key": "Johansson_2019",
            "doi": "10.1093/hmg/ddz175",
            "source_short": "Johansson et al. (2019), asthma liability SNP h2=0.210",
            "source": "Genome-wide association analysis of 350,000 Caucasians from the UK Biobank identifies novel loci for asthma, hay fever and eczema",
            "url": "https://doi.org/10.1093/hmg/ddz175",
            "note": "Publication-distinct asthma anchor; UK Biobank participant overlap remains possible.",
        },
        "secondary": {
            "reference_h2": 0.1130,
            "reference_scale": "liability",
            "citation_key": "Valette_2021",
            "doi": "10.1038/s42003-021-02227-6",
            "source_short": "Valette et al. (2021), asthma liability SNP h2=0.113",
            "source": "Prioritization of candidate causal genes for asthma in susceptibility loci derived from UK Biobank",
            "url": "https://doi.org/10.1038/s42003-021-02227-6",
            "note": "Alternative UK Biobank asthma liability-scale estimate.",
        },
    },

    "body_mass_index_bmi": {
        "primary": {
            "reference_h2": 0.2240,
            "reference_scale": "quantitative",
            "citation_key": "Yengo_2018",
            "doi": "10.1093/hmg/ddy271",
            "source_short": "Yengo et al. (2018), BMI common-SNP h2=0.224",
            "source": "Meta-analysis of genome-wide association studies for height and body mass index in approximately 700,000 individuals of European ancestry",
            "url": "https://doi.org/10.1093/hmg/ddy271",
            "note": "Large European-ancestry quantitative BMI reference; includes UK Biobank in the meta-analysis.",
        },
        "secondary": {
            "reference_h2": 0.2850,
            "reference_scale": "quantitative",
            "citation_key": "Hou_2019",
            "doi": "10.1038/s41588-019-0465-0",
            "source_short": "Hou et al. (2019), UK Biobank BMI GRE h2=0.285",
            "source": "Accurate estimation of SNP-heritability from biobank-scale data irrespective of genetic architecture",
            "url": "https://doi.org/10.1038/s41588-019-0465-0",
            "note": "Alternative UK Biobank quantitative estimate obtained with GRE.",
        },
    },

    "depression": {
        "primary": {
            "reference_h2": 0.1200,
            "reference_scale": "liability",
            "citation_key": "Huider_2026",
            "doi": "10.1038/s41380-026-03666-5",
            "source_short": "Huider et al. (2026), DSM lifetime MDD LDSC liability h2=0.120",
            "source": "Genetics of major depressive disorder in a homogeneous population with uniform phenotyping",
            "url": "https://doi.org/10.1038/s41380-026-03666-5",
            "note": "Independent Dutch BIONIC cohort; clinically defined lifetime MDD differs from the benchmark broad-depression phenotype.",
        },
        "secondary": {
            "reference_h2": 0.1020,
            "reference_scale": "liability",
            "citation_key": "Howard_2018",
            "doi": "10.1038/s41467-018-03819-3",
            "source_short": "Howard et al. (2018), UKB broad-depression liability h2=0.102",
            "source": "Genome-wide association study of depression phenotypes in UK Biobank identifies variants in excitatory synaptic pathways",
            "url": "https://doi.org/10.1038/s41467-018-03819-3",
            "note": "Phenotype-matched sensitivity anchor; this publication also supplies the benchmark depression GWAS source, so it is secondary only.",
        },
    },

    "gastro_oesophageal_reflux_gord_gastric_reflux": {
        "primary": {
            "reference_h2": 0.0700,
            "reference_scale": "liability",
            "citation_key": "Gharahkhani_2015",
            "doi": "10.1093/hmg/ddv512",
            "source_short": "Gharahkhani et al. (2015), GERD liability h2=0.070",
            "source": "Chronic gastroesophageal reflux disease shares genetic background with esophageal adenocarcinoma and Barrett's esophagus",
            "url": "https://doi.org/10.1093/hmg/ddv512",
            "note": "23andMe GERD reference; published conversion used an 18% population prevalence.",
        },
        "secondary": {
            "reference_h2": 0.1130,
            "reference_scale": "liability",
            "citation_key": "An_2019",
            "doi": "10.1038/s41467-019-11968-2",
            "source_short": "An et al. (2019), GERD liability h2=0.113",
            "source": "Gastroesophageal reflux GWAS identifies risk loci that also associate with subsequent severe esophageal diseases",
            "url": "https://doi.org/10.1038/s41467-019-11968-2",
            "note": "Alternative GERD liability reference; published analysis assumed approximately 12% prevalence.",
        },
    },

    "high_cholesterol": {
        "primary": {
            "reference_h2": 0.1890,
            "reference_scale": "liability",
            "citation_key": "Tang_2022,Ge_2017",
            "doi": "10.1093/bib/bbac115; 10.1371/journal.pgen.1006711",
            "source_short": "Tang et al. (2022) / Ge et al. (2017), high-cholesterol liability reference h2=0.189",
            "source": "Fast heritability estimation based on MINQUE and batch training; reference value traced to Phenome-wide heritability analysis of the UK Biobank",
            "url": "https://doi.org/10.1093/bib/bbac115",
            "note": "Primary scale-matched high-cholesterol anchor; UK Biobank-based and therefore not participant-independent.",
        },
        "secondary": {
            "reference_h2": 0.0820,
            "reference_scale": "observed",
            "citation_key": "Hou_2019",
            "doi": "10.1038/s41588-019-0465-0",
            "source_short": "Hou et al. (2019), high-cholesterol GRE h2=0.082",
            "source": "Accurate estimation of SNP-heritability from biobank-scale data irrespective of genetic architecture",
            "url": "https://doi.org/10.1038/s41588-019-0465-0",
            "note": "Alternative UK Biobank GRE estimate treated as observed/raw binary-scale; liability representation is generated only by the benchmark row-specific conversion factor.",
        },
    },

    "hypertension": {
        "primary": {
            "reference_h2": 0.3200,
            "reference_scale": "liability",
            "citation_key": "Mu_oz_2016",
            "doi": "10.1038/ng.3618",
            "source_short": "Munoz et al. (2016), hypertension liability SNP h2=0.320",
            "source": "Evaluating the contribution of genetics and familial shared environment to common disease using the UK Biobank",
            "url": "https://doi.org/10.1038/ng.3618",
            "note": "UK Biobank disease reference; publication-distinct but participant overlap is possible.",
        },
        "secondary": {
            "reference_h2": 0.3110,
            "reference_scale": "liability",
            "citation_key": "Hujoel_2022",
            "doi": "10.1016/j.xgen.2022.100152",
            "source_short": "Hujoel et al. (2022), hypertension liability SNP h2=0.311",
            "source": "Incorporating family history of disease improves polygenic risk scores in diverse populations",
            "url": "https://doi.org/10.1016/j.xgen.2022.100152",
            "note": "Alternative large UK Biobank British liability-scale estimate.",
        },
    },

    "hypothyroidism_myxoedema": {
        "primary": {
            "reference_h2": 0.1450,
            "reference_scale": "liability",
            "citation_key": "Rand_2025",
            "doi": "10.1038/s41588-025-02410-z",
            "source_short": "Rand et al. (2025), hypothyroidism liability h2=0.145",
            "source": "Genome-wide association study and polygenic risk prediction of hypothyroidism",
            "url": "https://doi.org/10.1038/s41588-025-02410-z",
            "note": "Large multi-cohort European hypothyroidism estimate using SumHer BLD-LDAK.",
        },
        "secondary": {
            "reference_h2": 0.1080,
            "reference_scale": "liability",
            "citation_key": "Mathieu_2022",
            "doi": "10.1016/j.isci.2022.104992",
            "source_short": "Mathieu et al. (2022), hypothyroidism liability h2=0.108",
            "source": "Genetic association and Mendelian randomization for hypothyroidism highlight immune molecular mechanisms",
            "url": "https://doi.org/10.1016/j.isci.2022.104992",
            "note": "Alternative European hypothyroidism liability-scale estimate.",
        },
    },

    "irritable_bowel_syndrome": {
        "primary": {
            "reference_h2": 0.0600,
            "reference_scale": "liability",
            "citation_key": "Wu_2021",
            "doi": "10.1038/s41467-021-21280-7",
            "source_short": "Wu et al. (2021), IBS liability h2=0.060",
            "source": "GWAS of peptic ulcer disease implicates Helicobacter pylori infection, other gastrointestinal disorders and depression",
            "url": "https://doi.org/10.1038/s41467-021-21280-7",
            "note": "Publication-distinct IBS liability estimate.",
        },
        "secondary": {
            "reference_h2": 0.0577,
            "reference_scale": "liability",
            "citation_key": "Eijsbouts_2021",
            "doi": "10.1038/s41588-021-00950-8",
            "source_short": "Eijsbouts et al. (2021), IBS SNP h2=0.0577",
            "source": "Genome-wide analysis of 53,400 people with irritable bowel syndrome highlights shared genetic pathways with mood and anxiety disorders",
            "url": "https://doi.org/10.1038/s41588-021-00950-8",
            "note": "Phenotype-matched sensitivity anchor; this publication also supplies the benchmark IBS GWAS source, so it is secondary only.",
        },
    },

    "migraine": {
        "primary": {
            "reference_h2": 0.1120,
            "reference_scale": "liability",
            "citation_key": "Hautakangas_2022",
            "doi": "10.1038/s41588-021-00990-0",
            "source_short": "Hautakangas et al. (2022), migraine liability h2=0.112",
            "source": "Genome-wide analysis of 102,084 migraine cases identifies 123 risk loci and subtype-specific risk alleles",
            "url": "https://doi.org/10.1038/s41588-021-00990-0",
            "note": "Large European migraine meta-analysis; liability conversion used 16% population prevalence.",
        },
        "secondary": {
            "reference_h2": 0.1460,
            "reference_scale": "liability",
            "citation_key": "Gormley_2016",
            "doi": "10.1038/ng.3598",
            "source_short": "Gormley et al. (2016), migraine liability h2=0.146",
            "source": "Meta-analysis of 375,000 individuals identifies 38 susceptibility loci for migraine",
            "url": "https://doi.org/10.1038/ng.3598",
            "note": "Alternative migraine estimate; later work reported that differences in liability-transformation convention materially affected this value.",
        },
    },

    "osteoarthritis": {
        "primary": {
            "reference_h2": 0.0940,
            "reference_scale": "liability",
            "citation_key": "Barowsky_2021",
            "doi": "10.3389/fgene.2021.687687",
            "source_short": "Barowsky et al. (2021), osteoarthritis liability h2=0.094",
            "source": "Cross-Disorder Genomics Data Analysis Elucidates a Shared Genetic Basis Between Major Depression and Osteoarthritis Pain",
            "url": "https://doi.org/10.3389/fgene.2021.687687",
            "note": "Publication-distinct OA liability estimate.",
        },
        "secondary": {
            "reference_h2": 0.0892,
            "reference_scale": "liability",
            "citation_key": "Huang_2023",
            "doi": "10.3389/fimmu.2023.1184958",
            "source_short": "Huang et al. (2023), osteoarthritis liability h2=0.0892",
            "source": "A shared genetic contribution to osteoarthritis and COVID-19 outcomes: a large-scale genome-wide cross-trait analysis",
            "url": "https://doi.org/10.3389/fimmu.2023.1184958",
            "note": "Alternative liability-scale osteoarthritis estimate.",
        },
    },
}


# =============================================================================
# 4. HELPERS
# =============================================================================

def safe_float(value):
    try:
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def safe_str(value):
    if pd.isna(value):
        return "NA"
    s = str(value).strip()
    return s if s else "NA"


def pick_column(df_or_row, candidates):
    if isinstance(df_or_row, pd.Series):
        cols = list(df_or_row.index)
    else:
        cols = list(df_or_row.columns)

    lower = {str(c).lower(): c for c in cols}

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    return None


def bh_fdr(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    qvalues = np.full(len(pvalues), np.nan)

    valid = np.where(np.isfinite(pvalues))[0]

    if len(valid) == 0:
        return qvalues

    p = pvalues[valid]
    order = np.argsort(p)
    ranked = p[order]

    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    temp = np.empty(m)
    temp[order] = adjusted
    qvalues[valid] = temp

    return qvalues


def one_sample_wilcoxon(values):
    x = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).dropna().values.astype(float)

    if len(x) < MIN_WILCOXON_N:
        return np.nan, len(x)

    if np.allclose(x, 0.0):
        return 1.0, len(x)

    try:
        result = stats.wilcoxon(
            x,
            alternative="two-sided",
            zero_method="wilcox",
            correction=False,
            method="auto",
        )
        return float(result.pvalue), len(x)
    except Exception:
        return np.nan, len(x)


def paired_wilcoxon(a, b):
    z = pd.DataFrame({
        "a": pd.to_numeric(pd.Series(a), errors="coerce"),
        "b": pd.to_numeric(pd.Series(b), errors="coerce"),
    }).dropna()

    if len(z) < MIN_WILCOXON_N:
        return np.nan, len(z)

    diff = z["a"].values - z["b"].values

    if np.allclose(diff, 0.0):
        return 1.0, len(z)

    try:
        result = stats.wilcoxon(
            z["a"].values,
            z["b"].values,
            alternative="two-sided",
            zero_method="wilcox",
            correction=False,
            method="auto",
        )
        return float(result.pvalue), len(z)
    except Exception:
        return np.nan, len(z)


def friedman_from_matrix(matrix, label, group_type):
    complete = matrix.dropna(axis=0, how="any")

    result = {
        "Analysis": label,
        "GroupType": group_type,
        "IndependentUnitsComplete": int(len(complete)),
        "Groups": int(matrix.shape[1]),
        "FriedmanChi2": np.nan,
        "FriedmanP": np.nan,
    }

    if len(complete) < MIN_WILCOXON_N or matrix.shape[1] < 3:
        return result

    arrays = [complete[c].values for c in complete.columns]

    try:
        stat, p = stats.friedmanchisquare(*arrays)
        result["FriedmanChi2"] = float(stat)
        result["FriedmanP"] = float(p)
    except Exception:
        pass

    return result


def fmt(value, digits=4):
    try:
        x = float(value)
    except Exception:
        return "NA"

    if not np.isfinite(x):
        return "NA"

    return f"{x:.{digits}g}"


# =============================================================================
# 5. SIMULATION METADATA
# =============================================================================

def simulation_metadata(sim_name):
    number = int(sim_name.replace("Sim", ""))

    if 1 <= number <= 100:
        phenotype_type = "continuous"

        if number <= 50:
            target_n = 750
            local = number
        else:
            target_n = 5000
            local = number - 50

        underlying_trait = f"Continuous_T{local:03d}"

    elif 101 <= number <= 200:
        phenotype_type = "binary"

        if number <= 150:
            target_n = 750
            local = number - 100
        else:
            target_n = 5000
            local = number - 150

        underlying_trait = f"Binary_T{local:03d}"

    else:
        raise ValueError(f"Unexpected simulation id: {sim_name}")

    true_h2_values = [0.05, 0.10, 0.30, 0.50, 0.70]
    true_h2 = true_h2_values[(local - 1) // 10]

    inside = (local - 1) % 10

    if inside < 5:
        causal_snps = 100
        replicate = inside + 1
    else:
        causal_snps = 5000
        replicate = inside - 4

    return {
        "TraitType": phenotype_type,
        "TargetN": target_n,
        "TrueH2": true_h2,
        "CausalSNPs": causal_snps,
        "Replicate": replicate,
        "UnderlyingTrait": underlying_trait,
    }


# =============================================================================
# 6. CONFIGURATION ID
# =============================================================================

CONFIG_CANDIDATES = [
    "h2model_standardized",
    "h2model",
    "model",

    "clumpprune",
    "clump_p1",
    "clump_r2",
    "clump_kb",

    "p_window_size",
    "p_slide_size",
    "p_LD_threshold",

    "numberofpca",
    "covariates",
    "pca",

    "tagging_model",
    "grm_type",
    "grm_source",
    "reference_source",
    "ld_reference",
    "ld_weight",
]


def make_configuration_id(method, row):
    bits = []

    for candidate in CONFIG_CANDIDATES:
        col = pick_column(row, [candidate])

        if col is None:
            continue

        value = safe_str(row[col])

        if value == "NA":
            continue

        bits.append(f"{candidate}={value}")

    if not bits:
        bits = [f"row_signature={row.name}"]

    return method + " | " + " | ".join(bits)


# =============================================================================
# 7. LOAD FOLD-LEVEL ROWS
# =============================================================================

def results_path(dataset, fold, method):
    return (
        ROOT
        / dataset
        / f"Fold_{fold}"
        / method
        / "Results_Liability.csv"
    )


def extract_h2_columns(row):
    native_col = pick_column(
        row,
        [
            "h2_native",
            "Original h2",
            "original_h2",
            "h2",
        ],
    )

    liability_col = pick_column(
        row,
        [
            "h2_liability_primary",
            "liability_h2",
            "h2_liability",
            "h2_liability_scale",
        ],
    )

    factor_col = pick_column(
        row,
        [
            "liability_factor_primary",
            "liability_factor",
        ],
    )

    native = (
        safe_float(row[native_col])
        if native_col is not None
        else np.nan
    )

    liability = (
        safe_float(row[liability_col])
        if liability_col is not None
        else np.nan
    )

    factor = (
        safe_float(row[factor_col])
        if factor_col is not None
        else np.nan
    )

    return native, liability, factor


def add_accuracy_row(
    output_rows,
    excluded_rows,
    *,
    domain,
    dataset,
    dataset_label,
    trait_type,
    method,
    config_id,
    fold,
    scale,
    estimate,
    reference,
    reference_kind,
    reference_source,
    reference_url,
    reference_note,
    reference_set="known_truth",
    reference_citation_key="simulation_generating_parameter",
    reference_doi="NA",
    reference_reported_scale="generating",
    reference_reported_h2=np.nan,
    target_n=np.nan,
    true_h2=np.nan,
    causal_snps=np.nan,
    replicate=np.nan,
    underlying_trait="NA",
    factor=np.nan,
):
    common_excluded = {
        "Domain": domain,
        "Dataset": dataset,
        "TraitType": trait_type,
        "Method": method,
        "ConfigID": config_id,
        "Fold": fold,
        "Scale": scale,
        "ReferenceSet": reference_set,
        "ReferenceCitationKey": reference_citation_key,
        "ReferenceDOI": reference_doi,
        "ReferenceReportedScale": reference_reported_scale,
        "ReferenceReportedH2": reference_reported_h2,
    }

    if not np.isfinite(estimate):
        rec = dict(common_excluded)
        rec["Reason"] = "missing_estimate"
        excluded_rows.append(rec)
        return

    if not np.isfinite(reference):
        rec = dict(common_excluded)
        rec["Reason"] = "missing_reference"
        excluded_rows.append(rec)
        return

    signed_error = estimate - reference
    abs_error = abs(signed_error)

    relative_abs_error = (
        abs_error / abs(reference)
        if reference != 0
        else np.nan
    )

    output_rows.append({
        "Domain": domain,
        "Dataset": dataset,
        "DatasetLabel": dataset_label,
        "TraitType": trait_type,
        "Method": method,
        "ConfigID": config_id,
        "Fold": fold,
        "Scale": scale,

        "Estimate": estimate,
        "Reference": reference,
        "SignedError": signed_error,
        "AbsError": abs_error,
        "RelativeAbsError": relative_abs_error,

        "ReferenceSet": reference_set,
        "ReferenceKind": reference_kind,
        "ReferenceCitationKey": reference_citation_key,
        "ReferenceDOI": reference_doi,
        "ReferenceReportedScale": reference_reported_scale,
        "ReferenceReportedH2": reference_reported_h2,
        "ReferenceSource": reference_source,
        "ReferenceURL": reference_url,
        "ReferenceNote": reference_note,

        "TargetN": target_n,
        "TrueH2": true_h2,
        "CausalSNPs": causal_snps,
        "Replicate": replicate,
        "UnderlyingTrait": underlying_trait,

        "LiabilityFactorPrimary": factor,
    })


def load_all_fold_rows():
    rows = []
    excluded = []

    print("=" * 120)
    print("FINAL ANALYSIS 04-XX — REFERENCE / KNOWN-TRUTH ACCURACY")
    print("=" * 120)
    print("ROOT:", ROOT)
    print()

    # -------------------------------------------------------------------------
    # EMPIRICAL
    # -------------------------------------------------------------------------

    empirical_files_found = 0

    for phenotype in REAL_PHENOTYPES:
        phenotype_dir = ROOT / phenotype

        if not phenotype_dir.exists():
            continue

        reference_sets = EMPIRICAL_REFERENCE_SETS.get(phenotype)

        if not reference_sets:
            continue

        label = PHENOTYPE_LABELS.get(phenotype, phenotype)

        trait_type = (
            "continuous"
            if phenotype == "body_mass_index_bmi"
            else "binary"
        )

        for fold in range(5):
            for method in METHODS:
                path = results_path(
                    dataset=phenotype,
                    fold=fold,
                    method=method,
                )

                if not path.exists():
                    continue

                empirical_files_found += 1

                try:
                    df = pd.read_csv(path)
                except Exception:
                    continue

                for _, row in df.iterrows():
                    native, liability, factor = extract_h2_columns(row)

                    config_id = make_configuration_id(
                        method=method,
                        row=row,
                    )

                    # Analyse PRIMARY and SECONDARY literature sets separately.
                    for reference_set, reference_info in reference_sets.items():
                        ref_h2 = float(reference_info["reference_h2"])
                        ref_scale = str(reference_info["reference_scale"]).lower()

                        common = dict(
                            reference_set=reference_set,
                            reference_citation_key=reference_info["citation_key"],
                            reference_doi=reference_info["doi"],
                            reference_reported_scale=ref_scale,
                            reference_reported_h2=ref_h2,
                            reference_source=reference_info["source"],
                            reference_url=reference_info["url"],
                            reference_note=reference_info["note"],
                        )

                        # -----------------------------------------------------
                        # Quantitative empirical phenotype: direct comparison.
                        # -----------------------------------------------------
                        if trait_type == "continuous":
                            if ref_scale != "quantitative":
                                excluded.append({
                                    "Domain": "empirical",
                                    "Dataset": phenotype,
                                    "TraitType": trait_type,
                                    "Method": method,
                                    "ConfigID": config_id,
                                    "Fold": fold,
                                    "Scale": "native_quantitative",
                                    "ReferenceSet": reference_set,
                                    "ReferenceCitationKey": reference_info["citation_key"],
                                    "ReferenceDOI": reference_info["doi"],
                                    "ReferenceReportedScale": ref_scale,
                                    "ReferenceReportedH2": ref_h2,
                                    "Reason": "continuous_reference_not_quantitative",
                                })
                                continue

                            add_accuracy_row(
                                rows,
                                excluded,
                                domain="empirical",
                                dataset=phenotype,
                                dataset_label=label,
                                trait_type=trait_type,
                                method=method,
                                config_id=config_id,
                                fold=fold,
                                scale="native_quantitative",
                                estimate=native,
                                reference=ref_h2,
                                reference_kind="external_literature_anchor_not_ground_truth",
                                factor=factor,
                                **common,
                            )
                            continue

                        # -----------------------------------------------------
                        # Binary empirical phenotype.
                        # Convert the REFERENCE, not the estimator, when a
                        # scale-matched representation is needed.
                        # -----------------------------------------------------

                        liability_reference = np.nan
                        observed_reference = np.nan

                        if ref_scale == "liability":
                            # Directly reported liability anchor.
                            liability_reference = ref_h2

                            if np.isfinite(factor) and factor > 0:
                                observed_reference = ref_h2 / factor

                        elif ref_scale == "observed":
                            # Directly reported observed/raw anchor.
                            observed_reference = ref_h2

                            if np.isfinite(factor) and factor > 0:
                                liability_reference = ref_h2 * factor

                        else:
                            excluded.append({
                                "Domain": "empirical",
                                "Dataset": phenotype,
                                "TraitType": trait_type,
                                "Method": method,
                                "ConfigID": config_id,
                                "Fold": fold,
                                "Scale": "binary_unknown_reference_scale",
                                "ReferenceSet": reference_set,
                                "ReferenceCitationKey": reference_info["citation_key"],
                                "ReferenceDOI": reference_info["doi"],
                                "ReferenceReportedScale": ref_scale,
                                "ReferenceReportedH2": ref_h2,
                                "Reason": "unsupported_binary_reference_scale",
                            })
                            continue

                        # Liability representation.
                        if np.isfinite(liability_reference):
                            liability_note = reference_info["note"]

                            if ref_scale == "observed":
                                liability_note += (
                                    " The reported observed-scale anchor was "
                                    "projected to liability scale using the "
                                    "row-specific primary liability factor."
                                )

                            add_accuracy_row(
                                rows,
                                excluded,
                                domain="empirical",
                                dataset=phenotype,
                                dataset_label=label,
                                trait_type=trait_type,
                                method=method,
                                config_id=config_id,
                                fold=fold,
                                scale="liability_binary",
                                estimate=liability,
                                reference=liability_reference,
                                reference_kind="external_literature_anchor_not_ground_truth",
                                factor=factor,
                                **{**common, "reference_note": liability_note},
                            )
                        else:
                            excluded.append({
                                "Domain": "empirical",
                                "Dataset": phenotype,
                                "TraitType": trait_type,
                                "Method": method,
                                "ConfigID": config_id,
                                "Fold": fold,
                                "Scale": "liability_binary",
                                "ReferenceSet": reference_set,
                                "ReferenceCitationKey": reference_info["citation_key"],
                                "ReferenceDOI": reference_info["doi"],
                                "ReferenceReportedScale": ref_scale,
                                "ReferenceReportedH2": ref_h2,
                                "Reason": "cannot_project_reference_to_liability_missing_factor",
                            })

                        # Observed representation.
                        if np.isfinite(observed_reference):
                            observed_note = reference_info["note"]

                            if ref_scale == "liability":
                                observed_note += (
                                    " The reported liability-scale anchor was "
                                    "projected to observed scale using the "
                                    "row-specific primary liability factor."
                                )

                            add_accuracy_row(
                                rows,
                                excluded,
                                domain="empirical",
                                dataset=phenotype,
                                dataset_label=label,
                                trait_type=trait_type,
                                method=method,
                                config_id=config_id,
                                fold=fold,
                                scale="observed_binary_scale_matched",
                                estimate=native,
                                reference=observed_reference,
                                reference_kind="external_literature_anchor_not_ground_truth",
                                factor=factor,
                                **{**common, "reference_note": observed_note},
                            )
                        else:
                            excluded.append({
                                "Domain": "empirical",
                                "Dataset": phenotype,
                                "TraitType": trait_type,
                                "Method": method,
                                "ConfigID": config_id,
                                "Fold": fold,
                                "Scale": "observed_binary_scale_matched",
                                "ReferenceSet": reference_set,
                                "ReferenceCitationKey": reference_info["citation_key"],
                                "ReferenceDOI": reference_info["doi"],
                                "ReferenceReportedScale": ref_scale,
                                "ReferenceReportedH2": ref_h2,
                                "Reason": "cannot_project_reference_to_observed_missing_factor",
                            })

    # -------------------------------------------------------------------------
    # SIMULATED
    # -------------------------------------------------------------------------

    simulated_files_found = 0

    simulation_dirs = sorted(
        p
        for p in ROOT.glob("Sim[0-9][0-9][0-9]")
        if p.is_dir()
    )

    for sim_dir in simulation_dirs:
        sim_name = sim_dir.name

        try:
            meta = simulation_metadata(sim_name)
        except Exception:
            continue

        trait_type = meta["TraitType"]
        true_h2 = meta["TrueH2"]

        for fold in range(5):
            for method in METHODS:
                path = results_path(
                    dataset=sim_name,
                    fold=fold,
                    method=method,
                )

                if not path.exists():
                    continue

                simulated_files_found += 1

                try:
                    df = pd.read_csv(path)
                except Exception:
                    continue

                for _, row in df.iterrows():
                    native, liability, factor = extract_h2_columns(row)

                    config_id = make_configuration_id(
                        method=method,
                        row=row,
                    )

                    if trait_type == "continuous":
                        add_accuracy_row(
                            rows,
                            excluded,
                            domain="simulated",
                            dataset=sim_name,
                            dataset_label=sim_name,
                            trait_type=trait_type,
                            method=method,
                            config_id=config_id,
                            fold=fold,
                            scale="native_quantitative",
                            estimate=native,
                            reference=true_h2,
                            reference_kind="known_simulated_generating_h2",
                            reference_source="simulation generating parameter",
                            reference_url="NA",
                            reference_note=
                                "Known generating quantitative SNP heritability.",
                            target_n=meta["TargetN"],
                            true_h2=true_h2,
                            causal_snps=meta["CausalSNPs"],
                            replicate=meta["Replicate"],
                            underlying_trait=meta["UnderlyingTrait"],
                            factor=factor,
                        )

                    else:
                        # Binary liability-scale truth comparison.
                        add_accuracy_row(
                            rows,
                            excluded,
                            domain="simulated",
                            dataset=sim_name,
                            dataset_label=sim_name,
                            trait_type=trait_type,
                            method=method,
                            config_id=config_id,
                            fold=fold,
                            scale="liability_binary",
                            estimate=liability,
                            reference=true_h2,
                            reference_kind="known_simulated_generating_liability_h2",
                            reference_source="simulation generating parameter",
                            reference_url="NA",
                            reference_note=
                                "Known liability-scale generating SNP heritability; K=0.10.",
                            target_n=meta["TargetN"],
                            true_h2=true_h2,
                            causal_snps=meta["CausalSNPs"],
                            replicate=meta["Replicate"],
                            underlying_trait=meta["UnderlyingTrait"],
                            factor=factor,
                        )

                        # Binary observed-scale scale-matched truth comparison.
                        if np.isfinite(factor) and factor > 0:
                            observed_truth = true_h2 / factor

                            add_accuracy_row(
                                rows,
                                excluded,
                                domain="simulated",
                                dataset=sim_name,
                                dataset_label=sim_name,
                                trait_type=trait_type,
                                method=method,
                                config_id=config_id,
                                fold=fold,
                                scale="observed_binary_scale_matched",
                                estimate=native,
                                reference=observed_truth,
                                reference_kind=
                                    "known_generating_h2_inverse_transformed_to_observed",
                                reference_source="simulation generating parameter",
                                reference_url="NA",
                                reference_note=(
                                    "Known liability generating h2 inverse-transformed "
                                    "using the row-specific primary liability factor."
                                ),
                                target_n=meta["TargetN"],
                                true_h2=true_h2,
                                causal_snps=meta["CausalSNPs"],
                                replicate=meta["Replicate"],
                                underlying_trait=meta["UnderlyingTrait"],
                                factor=factor,
                            )
                        else:
                            excluded.append({
                                "Domain": "simulated",
                                "Dataset": sim_name,
                                "TraitType": trait_type,
                                "Method": method,
                                "ConfigID": config_id,
                                "Fold": fold,
                                "Scale": "observed_binary_scale_matched",
                                "Reason": "missing_or_invalid_liability_factor",
                            })

    data = pd.DataFrame(rows)
    excluded_df = pd.DataFrame(excluded)

    print("Empirical Results_Liability.csv files found:", empirical_files_found)
    print("Simulated Results_Liability.csv files found:", simulated_files_found)
    print("Accuracy fold rows created:", len(data))
    print("Excluded row-scale combinations:", len(excluded_df))
    print()

    return data, excluded_df


# =============================================================================
# 8. FOLD AVERAGING
# =============================================================================

def build_units(fold_rows):
    group_cols = [
        "Domain",
        "Dataset",
        "DatasetLabel",
        "TraitType",
        "Method",
        "ConfigID",
        "Scale",

        "ReferenceSet",
        "ReferenceKind",
        "ReferenceCitationKey",
        "ReferenceDOI",
        "ReferenceReportedScale",
        "ReferenceReportedH2",
        "ReferenceSource",
        "ReferenceURL",
        "ReferenceNote",

        "TargetN",
        "TrueH2",
        "CausalSNPs",
        "Replicate",
        "UnderlyingTrait",
    ]

    units = (
        fold_rows
        .groupby(
            group_cols,
            as_index=False,
            dropna=False,
        )
        .agg(
            Estimate=("Estimate", "mean"),
            Reference=("Reference", "mean"),
            NFolds=("Fold", "nunique"),
            FoldEstimateSD=("Estimate", "std"),
            MeanLiabilityFactor=(
                "LiabilityFactorPrimary",
                "mean",
            ),
        )
    )

    units["SignedError"] = (
        units["Estimate"]
        - units["Reference"]
    )

    units["AbsError"] = (
        units["SignedError"]
        .abs()
    )

    units["SqError"] = (
        units["SignedError"]
        ** 2
    )

    units["RelativeAbsError"] = np.where(
        units["Reference"].abs() > 0,
        units["AbsError"]
        / units["Reference"].abs(),
        np.nan,
    )

    units["BelowZero"] = (
        units["Estimate"] < 0
    )

    units["AboveOne"] = (
        units["Estimate"] > 1
    )

    return units


# =============================================================================
# 9. ANALYSIS FAMILIES
# =============================================================================

def make_analysis_families(units):
    """
    Returns long-form independent-unit analysis families.

    Empirical analyses are split by:
        reference set (primary / secondary)
        trait type
        scale

    Simulated analyses use the known generating parameter and remain split by
    trait type, scale and sample-size condition.
    """
    frames = []

    # -------------------------------------------------------------------------
    # EMPIRICAL
    # -------------------------------------------------------------------------

    empirical = units[
        units["Domain"] == "empirical"
    ].copy()

    for reference_set in ["primary", "secondary"]:
        emp_ref = empirical[
            empirical["ReferenceSet"]
            == reference_set
        ].copy()

        if emp_ref.empty:
            continue

        for trait_type, scale, suffix in [
            (
                "binary",
                "liability_binary",
                "binary_liability",
            ),
            (
                "binary",
                "observed_binary_scale_matched",
                "binary_observed",
            ),
            (
                "continuous",
                "native_quantitative",
                "continuous_native",
            ),
        ]:
            z = emp_ref[
                (emp_ref["TraitType"] == trait_type)
                & (emp_ref["Scale"] == scale)
            ].copy()

            if z.empty:
                continue

            z["Analysis"] = (
                f"empirical_{reference_set}_{suffix}"
            )

            z["IndependentUnit"] = z["Dataset"]
            frames.append(z)

    # -------------------------------------------------------------------------
    # SIMULATED
    # -------------------------------------------------------------------------

    simulated = units[
        units["Domain"] == "simulated"
    ].copy()

    sim_specs = [
        (
            "continuous",
            "native_quantitative",
            "continuous_native",
        ),
        (
            "binary",
            "liability_binary",
            "binary_liability",
        ),
        (
            "binary",
            "observed_binary_scale_matched",
            "binary_observed",
        ),
    ]

    for trait_type, scale, base_label in sim_specs:
        base = simulated[
            (simulated["TraitType"] == trait_type)
            & (simulated["Scale"] == scale)
        ].copy()

        if base.empty:
            continue

        # Separate N analyses.
        for n in [750, 5000]:
            z = base[
                pd.to_numeric(
                    base["TargetN"],
                    errors="coerce",
                ) == n
            ].copy()

            if z.empty:
                continue

            z["Analysis"] = (
                f"sim_{base_label}_N{n}"
            )

            z["IndependentUnit"] = (
                z["UnderlyingTrait"]
            )

            frames.append(z)

        # Nested-N-aware overall analysis.
        overall_group_cols = [
            "Domain",
            "TraitType",
            "Method",
            "ConfigID",
            "Scale",
            "UnderlyingTrait",

            "ReferenceSet",
            "ReferenceKind",
            "ReferenceCitationKey",
            "ReferenceDOI",
            "ReferenceReportedScale",
            "ReferenceReportedH2",
            "ReferenceSource",
            "ReferenceURL",
            "ReferenceNote",

            "TrueH2",
            "CausalSNPs",
            "Replicate",
        ]

        overall = (
            base
            .groupby(
                overall_group_cols,
                as_index=False,
                dropna=False,
            )
            .agg(
                Estimate=("Estimate", "mean"),
                Reference=("Reference", "mean"),
                SignedError=("SignedError", "mean"),
                AbsError=("AbsError", "mean"),
                SqError=("SqError", "mean"),
                RelativeAbsError=(
                    "RelativeAbsError",
                    "mean",
                ),
                NFolds=("NFolds", "sum"),
                SampleSizes=("TargetN", "nunique"),
            )
        )

        # Require both nested N values.
        overall = overall[
            overall["SampleSizes"] == 2
        ].copy()

        if not overall.empty:
            overall["Dataset"] = (
                overall["UnderlyingTrait"]
            )
            overall["DatasetLabel"] = (
                overall["UnderlyingTrait"]
            )
            overall["TargetN"] = "overall_equalN"
            overall["FoldEstimateSD"] = np.nan
            overall["MeanLiabilityFactor"] = np.nan
            overall["BelowZero"] = np.nan
            overall["AboveOne"] = np.nan
            overall["Analysis"] = (
                f"sim_{base_label}_overall_equalN"
            )
            overall["IndependentUnit"] = (
                overall["UnderlyingTrait"]
            )
            frames.append(overall)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


# =============================================================================
# 10. GROUP SUMMARIES
# =============================================================================

def group_accuracy_summary(analysis_units, group_col, group_type):
    rows = []

    for analysis, family in analysis_units.groupby(
        "Analysis",
        sort=False,
    ):
        expected_units = family[
            "IndependentUnit"
        ].nunique()

        family_rows = []

        for group, g in family.groupby(
            group_col,
            dropna=False,
        ):
            unit_table = (
                g.groupby(
                    "IndependentUnit",
                    as_index=False,
                )
                .agg(
                    SignedError=("SignedError", "mean"),
                    AbsError=("AbsError", "mean"),
                    RelativeAbsError=(
                        "RelativeAbsError",
                        "mean",
                    ),
                )
            )

            n_units = unit_table[
                "IndependentUnit"
            ].nunique()

            coverage = (
                n_units / expected_units
                if expected_units > 0
                else np.nan
            )

            errors = unit_table["SignedError"]
            abs_errors = unit_table["AbsError"]
            rel_errors = unit_table["RelativeAbsError"]

            bias_p, bias_n = one_sample_wilcoxon(
                errors.values
            )

            family_rows.append({
                "Analysis": analysis,
                "GroupType": group_type,
                "Group": group,
                "IndependentUnits": n_units,
                "ExpectedUnits": expected_units,
                "Coverage": coverage,
                "EligibleForRank":
                    bool(
                        np.isfinite(coverage)
                        and coverage >= MIN_COVERAGE
                    ),

                "MAE": abs_errors.mean(),
                "MedianAbsError": abs_errors.median(),
                "Bias": errors.mean(),
                "MedianSignedError": errors.median(),
                "AbsBias": abs(errors.mean()),
                "RMSE": np.sqrt(
                    np.nanmean(
                        np.square(errors.values)
                    )
                ),
                "MeanRelativeAbsError":
                    rel_errors.mean(),
                "MedianRelativeAbsError":
                    rel_errors.median(),

                "PctWithin0.025":
                    100.0 * np.mean(
                        abs_errors.values <= 0.025
                    ),
                "PctWithin0.05":
                    100.0 * np.mean(
                        abs_errors.values <= 0.05
                    ),
                "PctWithin0.10":
                    100.0 * np.mean(
                        abs_errors.values <= 0.10
                    ),

                "BiasWilcoxonP": bias_p,
                "BiasWilcoxonN": bias_n,
            })

        family_df = pd.DataFrame(
            family_rows
        )

        if family_df.empty:
            continue

        family_df["BiasFDR_Q"] = bh_fdr(
            family_df["BiasWilcoxonP"].values
        )

        family_df["BiasSignificantFDR"] = (
            family_df["BiasFDR_Q"]
            < FDR_ALPHA
        )

        family_df["AccuracyRank"] = np.nan

        eligible = (
            family_df["EligibleForRank"]
            == True
        )

        if eligible.any():
            family_df.loc[
                eligible,
                "AccuracyRank",
            ] = (
                family_df.loc[
                    eligible,
                    "MAE",
                ]
                .rank(
                    method="min",
                    ascending=True,
                )
            )

        rows.append(family_df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(
        rows,
        ignore_index=True,
    )


# =============================================================================
# 11. METHOD-LEVEL UNITS
# =============================================================================

def method_unit_table(analysis_units):
    """
    Average configuration errors within method within independent unit.
    This ensures one value per method per independent unit for inference.
    """
    return (
        analysis_units
        .groupby(
            [
                "Analysis",
                "IndependentUnit",
                "Method",
            ],
            as_index=False,
        )
        .agg(
            SignedError=("SignedError", "mean"),
            AbsError=("AbsError", "mean"),
            RelativeAbsError=(
                "RelativeAbsError",
                "mean",
            ),
        )
    )


# =============================================================================
# 12. PAIRWISE TESTS + GLOBAL FRIEDMAN
# =============================================================================

def pairwise_accuracy_tests(
    unit_table,
    *,
    analysis_col,
    unit_col,
    group_col,
    value_col,
    group_type,
):
    all_rows = []
    global_rows = []

    for analysis, family in unit_table.groupby(
        analysis_col,
        sort=False,
    ):
        expected_units = family[
            unit_col
        ].nunique()

        coverage = (
            family.groupby(group_col)[unit_col]
            .nunique()
            / expected_units
        )

        keep_groups = coverage[
            coverage >= MIN_COVERAGE
        ].index.tolist()

        z = family[
            family[group_col]
            .isin(keep_groups)
        ].copy()

        matrix = z.pivot_table(
            index=unit_col,
            columns=group_col,
            values=value_col,
            aggfunc="mean",
        )

        global_rows.append(
            friedman_from_matrix(
                matrix=matrix,
                label=analysis,
                group_type=group_type,
            )
        )

        pair_rows = []

        columns = list(matrix.columns)

        for a, b in itertools.combinations(
            columns,
            2,
        ):
            paired = matrix[[a, b]].dropna()

            if len(paired) < MIN_WILCOXON_N:
                continue

            p, n = paired_wilcoxon(
                paired[a].values,
                paired[b].values,
            )

            delta = (
                paired[a]
                - paired[b]
            )

            pair_rows.append({
                "Analysis": analysis,
                "GroupType": group_type,
                "GroupA": a,
                "GroupB": b,
                "NPaired": n,

                "MeanAbsErrorA":
                    paired[a].mean(),
                "MeanAbsErrorB":
                    paired[b].mean(),

                "MedianAbsErrorA":
                    paired[a].median(),
                "MedianAbsErrorB":
                    paired[b].median(),

                "MedianDeltaError_A_minus_B":
                    delta.median(),
                "MeanDeltaError_A_minus_B":
                    delta.mean(),

                "PctA_LowerError":
                    100.0
                    * np.mean(delta.values < 0),

                "P": p,
            })

        pair_df = pd.DataFrame(
            pair_rows
        )

        if not pair_df.empty:
            pair_df["Q_BH_FDR"] = bh_fdr(
                pair_df["P"].values
            )

            pair_df["SignificantFDR"] = (
                pair_df["Q_BH_FDR"]
                < FDR_ALPHA
            )

            all_rows.append(pair_df)

    pairwise = (
        pd.concat(
            all_rows,
            ignore_index=True,
        )
        if all_rows
        else pd.DataFrame()
    )

    global_df = pd.DataFrame(
        global_rows
    )

    if not global_df.empty:
        global_df["GlobalQ_BH_FDR"] = bh_fdr(
            global_df["FriedmanP"].values
        )

    return pairwise, global_df


# =============================================================================
# 13. TOP-ACCURACY SUPPORT
# =============================================================================

def summarize_top_accuracy(
    summary,
    pairwise,
    group_type,
):
    rows = []

    for analysis, family in summary.groupby(
        "Analysis",
        sort=False,
    ):
        eligible = family[
            family["EligibleForRank"] == True
        ].sort_values(
            [
                "MAE",
                "AbsBias",
            ]
        )

        if eligible.empty:
            continue

        best = eligible.iloc[0]
        best_group = best["Group"]

        if len(eligible) > 1:
            runner_up = eligible.iloc[1]
            runner_group = runner_up["Group"]
        else:
            runner_up = None
            runner_group = "NA"

        relevant = pd.DataFrame()

        if not pairwise.empty:
            relevant = pairwise[
                (pairwise["Analysis"] == analysis)
                & (
                    (pairwise["GroupA"] == best_group)
                    | (pairwise["GroupB"] == best_group)
                )
            ].copy()

        significantly_better = 0
        significantly_worse = 0
        not_significant = 0

        runner_p = np.nan
        runner_q = np.nan

        for _, r in relevant.iterrows():
            other = (
                r["GroupB"]
                if r["GroupA"] == best_group
                else r["GroupA"]
            )

            if other == runner_group:
                runner_p = r["P"]
                runner_q = r["Q_BH_FDR"]

            sig = bool(
                r.get(
                    "SignificantFDR",
                    False,
                )
            )

            if not sig:
                not_significant += 1
                continue

            delta = r[
                "MedianDeltaError_A_minus_B"
            ]

            if r["GroupA"] == best_group:
                # A-B < 0 means best A has lower error.
                if delta < 0:
                    significantly_better += 1
                elif delta > 0:
                    significantly_worse += 1
                else:
                    not_significant += 1

            else:
                # best is B. A-B > 0 means B has lower error.
                if delta > 0:
                    significantly_better += 1
                elif delta < 0:
                    significantly_worse += 1
                else:
                    not_significant += 1

        rows.append({
            "Analysis": analysis,
            "GroupType": group_type,

            "BestGroup": best_group,
            "BestMAE": best["MAE"],
            "BestMedianAbsError":
                best["MedianAbsError"],
            "BestBias": best["Bias"],
            "BestRMSE": best["RMSE"],
            "BestCoverage": best["Coverage"],
            "BestIndependentUnits":
                best["IndependentUnits"],

            "RunnerUp": runner_group,
            "RunnerUpMAE": (
                runner_up["MAE"]
                if runner_up is not None
                else np.nan
            ),

            "BestVsRunnerUpP": runner_p,
            "BestVsRunnerUpQ": runner_q,

            "ComparatorsTested": len(relevant),
            "SignificantlyMoreAccurateThanN":
                significantly_better,
            "SignificantlyLessAccurateThanN":
                significantly_worse,
            "NotSignificantlyDifferentN":
                not_significant,

            "BiasP": best["BiasWilcoxonP"],
            "BiasQ": best["BiasFDR_Q"],
        })

    return pd.DataFrame(rows)


# =============================================================================
# 14. SCALE-INVARIANCE AND REFERENCE-SET SENSITIVITY
# =============================================================================

def _spearman_or_nan(a, b):
    z = pd.DataFrame({
        "a": pd.to_numeric(pd.Series(a), errors="coerce"),
        "b": pd.to_numeric(pd.Series(b), errors="coerce"),
    }).dropna()

    if len(z) < 2:
        return np.nan

    if z["a"].nunique() < 2 or z["b"].nunique() < 2:
        return np.nan

    try:
        return float(
            stats.spearmanr(
                z["a"],
                z["b"],
            ).statistic
        )
    except Exception:
        return np.nan


def scale_invariance_audit(fold_rows):
    """
    Audit observed-vs-liability representation consistency.

    This intentionally performs NO p-value / FDR test.

    With a consistent positive transformation applied to both estimate and
    reference, relative absolute error is algebraically invariant. The useful
    audit is therefore numerical/rank preservation, not a hypothesis test about
    which scale is "more accurate".
    """
    binary = fold_rows[
        fold_rows["TraitType"] == "binary"
    ].copy()

    liab = binary[
        binary["Scale"] == "liability_binary"
    ].copy()

    obs = binary[
        binary["Scale"]
        == "observed_binary_scale_matched"
    ].copy()

    key = [
        "Domain",
        "Dataset",
        "Method",
        "ConfigID",
        "Fold",
        "ReferenceSet",
    ]

    paired = liab[
        key + ["RelativeAbsError"]
    ].merge(
        obs[
            key + ["RelativeAbsError"]
        ],
        on=key,
        how="inner",
        suffixes=("_liability", "_observed"),
    )

    if paired.empty:
        return pd.DataFrame()

    paired["AbsRelativeErrorDifference"] = (
        paired["RelativeAbsError_liability"]
        - paired["RelativeAbsError_observed"]
    ).abs()

    rows = []

    for (domain, refset), g in paired.groupby(
        ["Domain", "ReferenceSet"],
        dropna=False,
    ):
        method_rank = (
            g.groupby("Method", as_index=False)
            .agg(
                liability=(
                    "RelativeAbsError_liability",
                    "mean",
                ),
                observed=(
                    "RelativeAbsError_observed",
                    "mean",
                ),
            )
        )

        method_rank["rank_liability"] = (
            method_rank["liability"]
            .rank(method="min")
        )
        method_rank["rank_observed"] = (
            method_rank["observed"]
            .rank(method="min")
        )

        config_rank = (
            g.groupby("ConfigID", as_index=False)
            .agg(
                liability=(
                    "RelativeAbsError_liability",
                    "mean",
                ),
                observed=(
                    "RelativeAbsError_observed",
                    "mean",
                ),
            )
        )

        config_rank["rank_liability"] = (
            config_rank["liability"]
            .rank(method="min")
        )
        config_rank["rank_observed"] = (
            config_rank["observed"]
            .rank(method="min")
        )

        rows.append({
            "Domain": domain,
            "ReferenceSet": refset,
            "MatchedFoldRows": len(g),

            "MaxAbsRelativeErrorDifference":
                g["AbsRelativeErrorDifference"].max(),
            "MeanAbsRelativeErrorDifference":
                g["AbsRelativeErrorDifference"].mean(),
            "RelativeErrorSpearman":
                _spearman_or_nan(
                    g["RelativeAbsError_liability"],
                    g["RelativeAbsError_observed"],
                ),

            "MethodRankSpearman":
                _spearman_or_nan(
                    method_rank["rank_liability"],
                    method_rank["rank_observed"],
                ),
            "MaxMethodRankShift":
                (
                    method_rank["rank_liability"]
                    - method_rank["rank_observed"]
                ).abs().max(),

            "ConfigurationRankSpearman":
                _spearman_or_nan(
                    config_rank["rank_liability"],
                    config_rank["rank_observed"],
                ),
            "MaxConfigurationRankShift":
                (
                    config_rank["rank_liability"]
                    - config_rank["rank_observed"]
                ).abs().max(),

            "AllRelativeErrorsEqualWithin1e10":
                bool(
                    (
                        g["AbsRelativeErrorDifference"]
                        <= 1e-10
                    ).all()
                ),
        })

    return pd.DataFrame(rows)


def reference_set_comparison(
    config_summary,
    method_summary,
):
    """
    Descriptive PRIMARY-vs-SECONDARY empirical reference sensitivity.

    No formal p-value is used because the two literature anchors are alternative
    assumptions/references, not independent experimental samples.
    """
    rows = []

    comparisons = [
        (
            "binary_liability",
            "empirical_primary_binary_liability",
            "empirical_secondary_binary_liability",
        ),
        (
            "binary_observed",
            "empirical_primary_binary_observed",
            "empirical_secondary_binary_observed",
        ),
        (
            "continuous_native",
            "empirical_primary_continuous_native",
            "empirical_secondary_continuous_native",
        ),
    ]

    for group_type, summary in [
        ("configuration", config_summary),
        ("method", method_summary),
    ]:
        if summary.empty:
            continue

        for comparison_name, primary_label, secondary_label in comparisons:
            p = summary[
                (summary["Analysis"] == primary_label)
                & (summary["EligibleForRank"] == True)
            ][
                [
                    "Group",
                    "MAE",
                    "AccuracyRank",
                ]
            ].copy()

            s = summary[
                (summary["Analysis"] == secondary_label)
                & (summary["EligibleForRank"] == True)
            ][
                [
                    "Group",
                    "MAE",
                    "AccuracyRank",
                ]
            ].copy()

            if p.empty or s.empty:
                continue

            z = p.merge(
                s,
                on="Group",
                how="inner",
                suffixes=("_Primary", "_Secondary"),
            )

            if z.empty:
                continue

            rank_shift = (
                z["AccuracyRank_Primary"]
                - z["AccuracyRank_Secondary"]
            ).abs()

            top_primary = (
                z.sort_values(
                    ["AccuracyRank_Primary", "MAE_Primary"]
                ).iloc[0]["Group"]
            )

            top_secondary = (
                z.sort_values(
                    ["AccuracyRank_Secondary", "MAE_Secondary"]
                ).iloc[0]["Group"]
            )

            rows.append({
                "Comparison": comparison_name,
                "GroupType": group_type,
                "CommonGroups": len(z),

                "RankSpearman_PrimaryVsSecondary":
                    _spearman_or_nan(
                        z["AccuracyRank_Primary"],
                        z["AccuracyRank_Secondary"],
                    ),

                "MaxAbsoluteRankShift":
                    rank_shift.max(),
                "MeanAbsoluteRankShift":
                    rank_shift.mean(),

                "TopPrimary": top_primary,
                "TopSecondary": top_secondary,
                "TopGroupPreserved":
                    bool(top_primary == top_secondary),

                "PrimaryAnalysis": primary_label,
                "SecondaryAnalysis": secondary_label,
            })

    return pd.DataFrame(rows)


# =============================================================================
# 15. EMPIRICAL REFERENCE TABLE
# =============================================================================

def empirical_reference_table():
    rows = []

    for phenotype in REAL_PHENOTYPES:
        for reference_set, info in (
            EMPIRICAL_REFERENCE_SETS[
                phenotype
            ].items()
        ):
            rows.append({
                "PhenotypeID": phenotype,
                "Phenotype":
                    PHENOTYPE_LABELS[
                        phenotype
                    ],
                "ReferenceSet": reference_set,
                "ReferenceH2":
                    info["reference_h2"],
                "ReferenceScale":
                    info["reference_scale"],
                "CitationKey":
                    info["citation_key"],
                "DOI":
                    info["doi"],
                "ReferenceKind":
                    "external literature anchor; NOT empirical ground truth",
                "SourceShort":
                    info["source_short"],
                "Source":
                    info["source"],
                "URL":
                    info["url"],
                "Caveat":
                    info["note"],
            })

    return pd.DataFrame(rows)


# =============================================================================
# 16. MAIN ANALYSIS
# =============================================================================

def main():
    if not ROOT.exists():
        raise SystemExit(
            f"ERROR: ROOT does not exist: {ROOT}"
        )

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_rows, excluded = (
        load_all_fold_rows()
    )

    if fold_rows.empty:
        raise SystemExit(
            "ERROR: no usable accuracy rows were created."
        )

    units = build_units(
        fold_rows
    )

    analysis_units = make_analysis_families(
        units
    )

    if analysis_units.empty:
        raise SystemExit(
            "ERROR: no analysis families were created."
        )

    # -------------------------------------------------------------------------
    # Configuration summary
    # -------------------------------------------------------------------------

    config_summary = group_accuracy_summary(
        analysis_units=analysis_units,
        group_col="ConfigID",
        group_type="configuration",
    )

    # -------------------------------------------------------------------------
    # Method-level units + summary
    # -------------------------------------------------------------------------

    method_units = method_unit_table(
        analysis_units
    )

    method_summary_rows = []

    for analysis, family in method_units.groupby(
        "Analysis",
        sort=False,
    ):
        expected = family[
            "IndependentUnit"
        ].nunique()

        family_rows = []

        for method, g in family.groupby(
            "Method",
            dropna=False,
        ):
            n_units = g[
                "IndependentUnit"
            ].nunique()

            coverage = (
                n_units / expected
                if expected > 0
                else np.nan
            )

            p_bias, n_bias = one_sample_wilcoxon(
                g["SignedError"]
            )

            family_rows.append({
                "Analysis": analysis,
                "GroupType": "method",
                "Group": method,
                "IndependentUnits": n_units,
                "ExpectedUnits": expected,
                "Coverage": coverage,
                "EligibleForRank":
                    bool(
                        np.isfinite(coverage)
                        and coverage >= MIN_COVERAGE
                    ),

                "MAE": g["AbsError"].mean(),
                "MedianAbsError":
                    g["AbsError"].median(),
                "Bias": g["SignedError"].mean(),
                "MedianSignedError":
                    g["SignedError"].median(),
                "AbsBias":
                    abs(
                        g["SignedError"].mean()
                    ),
                "RMSE":
                    np.sqrt(
                        np.nanmean(
                            np.square(
                                g["SignedError"]
                                .values
                            )
                        )
                    ),

                "MeanRelativeAbsError":
                    g["RelativeAbsError"].mean(),
                "MedianRelativeAbsError":
                    g["RelativeAbsError"].median(),

                "PctWithin0.025":
                    100.0 * np.mean(
                        g["AbsError"].values
                        <= 0.025
                    ),
                "PctWithin0.05":
                    100.0 * np.mean(
                        g["AbsError"].values
                        <= 0.05
                    ),
                "PctWithin0.10":
                    100.0 * np.mean(
                        g["AbsError"].values
                        <= 0.10
                    ),

                "BiasWilcoxonP": p_bias,
                "BiasWilcoxonN": n_bias,
            })

        family_df = pd.DataFrame(
            family_rows
        )

        if family_df.empty:
            continue

        family_df["BiasFDR_Q"] = bh_fdr(
            family_df["BiasWilcoxonP"].values
        )

        family_df["BiasSignificantFDR"] = (
            family_df["BiasFDR_Q"]
            < FDR_ALPHA
        )

        family_df["AccuracyRank"] = np.nan

        eligible = (
            family_df["EligibleForRank"]
            == True
        )

        if eligible.any():
            family_df.loc[
                eligible,
                "AccuracyRank",
            ] = (
                family_df.loc[
                    eligible,
                    "MAE",
                ]
                .rank(
                    method="min",
                    ascending=True,
                )
            )

        method_summary_rows.append(
            family_df
        )

    method_summary = (
        pd.concat(
            method_summary_rows,
            ignore_index=True,
        )
        if method_summary_rows
        else pd.DataFrame()
    )

    # -------------------------------------------------------------------------
    # Pairwise configuration tests
    # -------------------------------------------------------------------------

    config_pairwise, config_global = (
        pairwise_accuracy_tests(
            unit_table=analysis_units,
            analysis_col="Analysis",
            unit_col="IndependentUnit",
            group_col="ConfigID",
            value_col="AbsError",
            group_type="configuration",
        )
    )

    # -------------------------------------------------------------------------
    # Pairwise method tests
    # -------------------------------------------------------------------------

    method_pairwise, method_global = (
        pairwise_accuracy_tests(
            unit_table=method_units,
            analysis_col="Analysis",
            unit_col="IndependentUnit",
            group_col="Method",
            value_col="AbsError",
            group_type="method",
        )
    )

    global_tests = pd.concat(
        [
            config_global,
            method_global,
        ],
        ignore_index=True,
    )

    # Recalculate one BH-FDR across the set of GLOBAL tests.
    # Pairwise tests already have FDR within their own intended families.
    if not global_tests.empty:
        global_tests[
            "GlobalQAcrossAllGlobalTests"
        ] = bh_fdr(
            global_tests[
                "FriedmanP"
            ].values
        )

    # -------------------------------------------------------------------------
    # Top support
    # -------------------------------------------------------------------------

    top_config = summarize_top_accuracy(
        summary=config_summary,
        pairwise=config_pairwise,
        group_type="configuration",
    )

    top_method = summarize_top_accuracy(
        summary=method_summary,
        pairwise=method_pairwise,
        group_type="method",
    )

    top_accuracy = pd.concat(
        [
            top_config,
            top_method,
        ],
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # Scale-invariance audit
    # -------------------------------------------------------------------------

    scale_df = scale_invariance_audit(
        fold_rows
    )

    # -------------------------------------------------------------------------
    # Primary-vs-secondary reference sensitivity
    # -------------------------------------------------------------------------

    reference_set_df = reference_set_comparison(
        config_summary=config_summary,
        method_summary=method_summary,
    )

    # -------------------------------------------------------------------------
    # Reference table
    # -------------------------------------------------------------------------

    reference_df = (
        empirical_reference_table()
    )

    # -------------------------------------------------------------------------
    # Save CSVs
    # -------------------------------------------------------------------------

    outputs = {
        "empirical_references":
            reference_df,
        "fold_rows":
            fold_rows,
        "units":
            units,
        "analysis_units":
            analysis_units,
        "configuration_summary":
            config_summary,
        "method_summary":
            method_summary,
        "configuration_pairwise":
            config_pairwise,
        "method_pairwise":
            method_pairwise,
        "global_tests":
            global_tests,
        "top_accuracy":
            top_accuracy,
        "scale_invariance":
            scale_df,
        "reference_set_comparison":
            reference_set_df,
        "excluded":
            excluded,
    }

    for name, df in outputs.items():
        df.to_csv(
            OUTDIR
            / f"FinalAnalysis04-XX_{name}.csv",
            index=False,
        )

    # -------------------------------------------------------------------------
    # Excel workbook
    # -------------------------------------------------------------------------

    workbook = (
        OUTDIR
        / "FinalAnalysis04-XX.xlsx"
    )

    try:
        with pd.ExcelWriter(
            workbook,
            engine="openpyxl",
        ) as writer:
            for name, df in outputs.items():
                sheet = name[:31]

                # Excel cannot hold >1,048,576 rows. Guard just in case.
                if len(df) > 1_000_000:
                    df.head(
                        1_000_000
                    ).to_excel(
                        writer,
                        sheet_name=sheet,
                        index=False,
                    )
                else:
                    df.to_excel(
                        writer,
                        sheet_name=sheet,
                        index=False,
                    )

    except Exception as exc:
        print(
            "WARNING: Excel workbook not written:",
            repr(exc),
        )

    # -------------------------------------------------------------------------
    # Console / text report
    # -------------------------------------------------------------------------

    lines = []

    lines.append(
        "FINAL ANALYSIS 04-XX — EXTERNAL-REFERENCE / KNOWN-TRUTH ACCURACY"
    )
    lines.append("=" * 110)
    lines.append("")

    lines.append(
        "SIMULATED DATA: reference = known generating h2 (true ground truth)."
    )
    lines.append(
        "EMPIRICAL DATA: reference = external literature anchor; NOT ground truth."
    )
    lines.append(
        "Empirical PRIMARY and SECONDARY literature anchors are analysed separately and never averaged."
    )
    lines.append(
        "A publication-distinct empirical reference does not necessarily imply participant-level independence."
    )
    lines.append(
        "Binary estimates are compared only with scale-matched references."
    )
    lines.append("")

    lines.append(
        "EMPIRICAL EXTERNAL LITERATURE REFERENCES"
    )
    lines.append("-" * 110)

    for _, r in reference_df.sort_values(
        ["Phenotype", "ReferenceSet"]
    ).iterrows():
        lines.append(
            f"{r['Phenotype']} | {r['ReferenceSet'].upper()} | "
            f"h2={r['ReferenceH2']:.4f} ({r['ReferenceScale']}) | "
            f"{r['CitationKey']} | DOI={r['DOI']}"
        )

    lines.append("")
    lines.append(
        "TOP METHOD AND TOP CONFIGURATION BY ANALYSIS"
    )
    lines.append("-" * 110)

    for analysis in top_accuracy[
        "Analysis"
    ].drop_duplicates().tolist():

        lines.append("")
        lines.append(analysis)

        z = top_accuracy[
            top_accuracy["Analysis"]
            == analysis
        ]

        for _, r in z.iterrows():
            lines.append(
                f"  {r['GroupType'].upper()}: "
                f"{r['BestGroup']} | "
                f"MAE={fmt(r['BestMAE'])}; "
                f"bias={fmt(r['BestBias'])}; "
                f"runner-up={r['RunnerUp']}; "
                f"best-vs-runner p={fmt(r['BestVsRunnerUpP'])}; "
                f"q={fmt(r['BestVsRunnerUpQ'])}; "
                f"FDR-significantly closer than "
                f"{int(r['SignificantlyMoreAccurateThanN'])}/"
                f"{int(r['ComparatorsTested'])} tested comparators"
            )

    lines.append("")
    lines.append(
        "GLOBAL FRIEDMAN TESTS"
    )
    lines.append("-" * 110)

    for _, r in global_tests.iterrows():
        lines.append(
            f"{r['Analysis']} | {r['GroupType']} | "
            f"paired units={int(r['IndependentUnitsComplete'])}; "
            f"groups={int(r['Groups'])}; "
            f"chi2={fmt(r['FriedmanChi2'])}; "
            f"p={fmt(r['FriedmanP'])}; "
            f"q(global family)={fmt(r['GlobalQAcrossAllGlobalTests'])}"
        )

    lines.append("")
    lines.append(
        "BINARY OBSERVED-vs-LIABILITY SCALE-INVARIANCE AUDIT"
    )
    lines.append("-" * 110)
    lines.append(
        "No Wilcoxon/FDR comparison between scales is performed because a consistent positive scale transformation preserves relative error algebraically."
    )

    if scale_df.empty:
        lines.append(
            "No matched scale-invariance rows were available."
        )
    else:
        for _, r in scale_df.iterrows():
            lines.append(
                f"{r['Domain']} | reference={r['ReferenceSet']} | "
                f"matched fold rows={int(r['MatchedFoldRows'])}; "
                f"max relative-error difference="
                f"{fmt(r['MaxAbsRelativeErrorDifference'])}; "
                f"relative-error rho={fmt(r['RelativeErrorSpearman'])}; "
                f"method-rank rho={fmt(r['MethodRankSpearman'])}; "
                f"max method-rank shift={fmt(r['MaxMethodRankShift'])}; "
                f"config-rank rho={fmt(r['ConfigurationRankSpearman'])}; "
                f"max config-rank shift={fmt(r['MaxConfigurationRankShift'])}"
            )

    lines.append("")
    lines.append(
        "PRIMARY-vs-SECONDARY EMPIRICAL REFERENCE SENSITIVITY"
    )
    lines.append("-" * 110)
    lines.append(
        "This is descriptive sensitivity analysis; no formal test treats the two literature anchors as independent data."
    )

    if reference_set_df.empty:
        lines.append(
            "No primary-vs-secondary reference comparisons were available."
        )
    else:
        for _, r in reference_set_df.iterrows():
            lines.append(
                f"{r['Comparison']} | {r['GroupType']} | "
                f"common groups={int(r['CommonGroups'])}; "
                f"rank rho={fmt(r['RankSpearman_PrimaryVsSecondary'])}; "
                f"max rank shift={fmt(r['MaxAbsoluteRankShift'])}; "
                f"top primary={r['TopPrimary']}; "
                f"top secondary={r['TopSecondary']}; "
                f"top preserved={r['TopGroupPreserved']}"
            )

    lines.append("")
    lines.append(
        "INTERPRETATION RULES"
    )
    lines.append("-" * 110)
    lines.append(
        "1. Simulated analyses measure accuracy against genuine generating truth."
    )
    lines.append(
        "2. Empirical analyses measure concordance with external literature anchors, not empirical accuracy against truth."
    )
    lines.append(
        "3. Primary and secondary empirical anchors are analysed separately; they are never averaged."
    )
    lines.append(
        "4. Lower MAE means closer to the selected scale-matched reference."
    )
    lines.append(
        "5. Pairwise Wilcoxon + BH-FDR evaluates whether methods/configurations differ in paired reference error within one analysis family."
    )
    lines.append(
        "6. Failure to reject zero bias does not prove equivalence."
    )
    lines.append(
        "7. Empirical BMI contains one phenotype and therefore supports descriptive ranking only."
    )
    lines.append(
        "8. Observed-vs-liability is a representation audit, not an accuracy contest: consistent transformation preserves relative error."
    )
    lines.append(
        "9. A different publication may still share UK Biobank or other participants; publication independence is not cohort independence."
    )

    report = "\n".join(
        lines
    )

    report_path = (
        OUTDIR
        / "FinalAnalysis04-XX_MAIN_RESULTS.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print()
    print(report)
    print()
    print("=" * 120)
    print("OUTPUT DIRECTORY:")
    print(OUTDIR)
    print()
    print("MAIN WORKBOOK:")
    print(workbook)
    print()
    print("MAIN TEXT SUMMARY:")
    print(report_path)
    print("=" * 120)


if __name__ == "__main__":
    main()
