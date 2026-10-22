#!/usr/bin/env python3
"""
FinalAnalysis04-OneFactorMatched.py

Purpose
-------
Repeat the exact one-factor matched heritability sensitivity analysis using the
HARMONISED analysis scale, while retaining a native-scale comparison on the
same eligible matched rows.

Primary analysis scale
----------------------
REAL:
    binary phenotype -> h2_liability_primary
    BMI              -> h2_native

SIMULATED:
    continuous       -> h2_native
    binary           -> h2_liability_primary

IMPORTANT:
    * This script reads Results_Liability.csv ONLY for heritability values.
    * It NEVER falls back to native h2 when a binary liability value is missing.
    * A missing selected h2 is excluded and audited.
    * Original Results.csv is never modified.
    * Overlapping CV folds are averaged before inference.
    * Real-data inference uses phenotype as the independent unit.
    * Simulated-data inference averages nested N=750/N=5000 conditions within
      the same underlying simulated trait before inferential testing.
    * All valid one-factor contrasts form one BH-FDR family per analysis scale.

Usage
-----
    python FinalAnalysis04-OneFactorMatched.py real
    python FinalAnalysis04-OneFactorMatched.py simulated
    python FinalAnalysis04-OneFactorMatched.py both

Optional environment variables
------------------------------
    HERITABILITY_REAL_ROOT
    HERITABILITY_SIM_ROOT

Default root for both modes:
    /data/ascher01/uqmmune1/finalheritability

Outputs
-------
Under:
    FinalAnalysis04/

For each mode:
    FinalAnalysis04_<mode>_RowAudit.csv
    FinalAnalysis04_<mode>_ExcludedRows.csv
    FinalAnalysis04_<mode>_FactorAudit.csv
    FinalAnalysis04_<mode>_FoldPairs.csv
    FinalAnalysis04_<mode>_PairUnits.csv
    FinalAnalysis04_<mode>_IndependentUnitEffects.csv
    FinalAnalysis04_<mode>_InferentialSummary.csv
    FinalAnalysis04_<mode>_FactorRanking.csv
    FinalAnalysis04_<mode>_CategorySummary.csv
    FinalAnalysis04_<mode>_ScaleComparison.csv
    FinalAnalysis04_<mode>.xlsx

Scientific interpretation
-------------------------
The "harmonised" result is the primary revised-manuscript result:
    liability-scale h2 for binary traits + native quantitative h2 for continuous.

The "native" result is retained only to answer whether harmonisation changed
the one-factor conclusions.

Do not interpret the broad three-category summary as formal variance partitioning.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PATHS
# =============================================================================

DEFAULT_ROOT = "/data/ascher01/uqmmune1/finalheritability"

REAL_ROOT = Path(
    os.environ.get("HERITABILITY_REAL_ROOT", DEFAULT_ROOT)
).resolve()

SIM_ROOT = Path(
    os.environ.get("HERITABILITY_SIM_ROOT", DEFAULT_ROOT)
).resolve()


# =============================================================================
# 2. REAL PHENOTYPES
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


# =============================================================================
# 3. METHOD METADATA
# =============================================================================

METHODS = {
    "LDpred-2-Heritability": {
        "family": "LDpred2",
        "engine": "LDpred2-LDSC",
        "input_type": "GWAS+genotype",
        "uses_genotype": True,
        "uses_gwas": True,
        "grm_source": "none",
        "reference_source": "internal_LDpred2",
    },
    "GCTA-Heritability": {
        "family": "GCTA",
        "engine": "GCTA",
        "input_type": "genotype",
        "uses_genotype": True,
        "uses_gwas": False,
        "grm_source": "GCTA",
        "reference_source": "none",
    },
    "Gemma1-Heritability": {
        "family": "GEMMA",
        "engine": "GEMMA",
        "input_type": "genotype",
        "uses_genotype": True,
        "uses_gwas": False,
        "grm_source": "GEMMA",
        "reference_source": "none",
    },
    "Gemma2-Heritability": {
        "family": "GEMMA",
        "engine": "GEMMA",
        "input_type": "GWAS+genotype",
        "uses_genotype": True,
        "uses_gwas": True,
        "grm_source": "GEMMA",
        "reference_source": "none",
    },
    "Gemma-LDSC-Heritability": {
        "family": "GEMMA",
        "engine": "GEMMA",
        "input_type": "GWAS+genotype",
        "uses_genotype": True,
        "uses_gwas": True,
        "grm_source": "LDSC-derived",
        "reference_source": "genotype_LD_scores",
    },
    "DPR-Gemma-Heritability": {
        "family": "DPR+GEMMA",
        "engine": "GEMMA",
        "input_type": "genotype",
        "uses_genotype": True,
        "uses_gwas": False,
        "grm_source": "DPR",
        "reference_source": "none",
    },
    "LDSC-Heritability": {
        "family": "LDSC",
        "engine": "LDSC",
        "input_type": "GWAS_summary",
        "uses_genotype": False,
        "uses_gwas": True,
        "grm_source": "none",
        "reference_source": "external_EUR",
    },
    "LDAK-Calculated-Heritability": {
        "family": "LDAK",
        "engine": "SumHer",
        "input_type": "GWAS+genotype",
        "uses_genotype": True,
        "uses_gwas": True,
        "grm_source": "none",
        "reference_source": "calculated_genotype_tagging",
    },
    "LDAK-Precomputed-Heritability": {
        "family": "LDAK",
        "engine": "SumHer",
        "input_type": "GWAS_summary",
        "uses_genotype": False,
        "uses_gwas": True,
        "grm_source": "none",
        "reference_source": "precomputed_tagging",
    },
    "LDSC-Genotype-Heritability": {
        "family": "LDSC",
        "engine": "LDSC",
        "input_type": "GWAS+genotype_reference",
        "uses_genotype": True,
        "uses_gwas": True,
        "grm_source": "none",
        "reference_source": "genotype_derived_LD",
    },
}


# =============================================================================
# 4. FACTOR DEFINITIONS
# =============================================================================

FACTOR_DEFINITIONS = [
    # -------------------------------------------------------------------------
    # 1. DATA / PREPROCESSING
    # -------------------------------------------------------------------------
    {
        "category": "1. DATA / PREPROCESSING",
        "factor": "clumpprune",
        "label": "Clumping / pruning",
        "scope": "within_method",
        "require_same_genotype": False,
        "require_same_gwas": True,
        "ignore_controls": {"genotype_fingerprint"},
    },
    {
        "category": "1. DATA / PREPROCESSING",
        "factor": "covariates",
        "label": "Covariate inclusion",
        "scope": "within_method",
        "require_same_genotype": True,
        "require_same_gwas": True,
        "ignore_controls": set(),
    },
    {
        "category": "1. DATA / PREPROCESSING",
        "factor": "pca",
        "label": "PCA inclusion",
        "scope": "within_method",
        "require_same_genotype": True,
        "require_same_gwas": True,
        "ignore_controls": set(),
    },
    {
        "category": "1. DATA / PREPROCESSING",
        "factor": "snp_set",
        "label": "SNP / variant-set choice",
        "scope": "within_method",
        "require_same_genotype": False,
        "require_same_gwas": True,
        "ignore_controls": {"genotype_fingerprint"},
    },

    # -------------------------------------------------------------------------
    # 2. ESTIMATOR / MODEL CONFIGURATION
    # -------------------------------------------------------------------------
    {
        "category": "2. ESTIMATOR / MODEL CONFIGURATION",
        "factor": "algorithm",
        "label": "Estimation algorithm",
        "scope": "within_method",
        "require_same_genotype": True,
        "require_same_gwas": True,
        "ignore_controls": set(),
    },
    {
        "category": "2. ESTIMATOR / MODEL CONFIGURATION",
        "factor": "grm_type",
        "label": "GRM construction",
        "scope": "within_method",
        "require_same_genotype": True,
        "require_same_gwas": True,
        "ignore_controls": set(),
    },
    {
        "category": "2. ESTIMATOR / MODEL CONFIGURATION",
        "factor": "tagging_model",
        "label": "LDAK tagging / model assumption",
        "scope": "within_method",
        "require_same_genotype": True,
        "require_same_gwas": True,
        "ignore_controls": set(),
    },

    # -------------------------------------------------------------------------
    # 3. INPUT / REFERENCE RESOURCE
    # -------------------------------------------------------------------------
    {
        "category": "3. INPUT / REFERENCE RESOURCE",
        "factor": "grm_source",
        "label": "GRM source (e.g. GEMMA vs DPR)",
        "scope": "cross_method",
        "require_same_genotype": True,
        "require_same_gwas": False,
        "ignore_controls": {"family"},
    },
    {
        "category": "3. INPUT / REFERENCE RESOURCE",
        "factor": "reference_source",
        "label": "Reference strategy",
        "scope": "cross_method",
        "require_same_genotype": False,
        "require_same_gwas": True,
        "ignore_controls": {"input_type", "family"},
    },
    {
        "category": "3. INPUT / REFERENCE RESOURCE",
        "factor": "ld_reference",
        "label": "LDSC LD reference",
        "scope": "within_method",
        "require_same_genotype": False,
        "require_same_gwas": True,
        "ignore_controls": {"reference_source"},
    },
    {
        "category": "3. INPUT / REFERENCE RESOURCE",
        "factor": "ld_weight",
        "label": "LDSC regression weights",
        "scope": "within_method",
        "require_same_genotype": False,
        "require_same_gwas": True,
        "ignore_controls": {"reference_source"},
    },
]

STRUCTURALLY_CONFOUNDED = [
    "Input data type (individual genotype vs GWAS summary statistics)"
]

CONTROL_FIELDS = [
    "family",
    "engine",
    "input_type",

    "clumpprune",
    "covariates",
    "pca",
    "snp_set",

    "algorithm",
    "grm_type",
    "tagging_model",

    "grm_source",
    "reference_source",

    "ld_reference",
    "ld_weight",

    "clump_p1_raw",
    "clump_r2_raw",
    "clump_kb_raw",

    "p_window_size_raw",
    "p_slide_size_raw",
    "p_LD_threshold_raw",

    "numberofpca_raw",

    "genotype_fingerprint",
    "gwas_fingerprint",
]

LEVEL_ORDER = {
    "clumpprune": ["no", "yes"],
    "covariates": ["no", "yes"],
    "pca": ["no", "yes"],
    "snp_set": ["full", "all", "genotyped", "hapmap"],
    "algorithm": [
        "HE",
        "REML",
        "GCTA_REML",
        "LDSC",
        "LDpred2_LDSC",
        "SumHer",
    ],
    "grm_type": ["centered", "standardized", "GCTA", "none"],
    "grm_source": ["GEMMA", "DPR", "GCTA", "LDSC-derived", "none"],
}


# =============================================================================
# 5. BASIC HELPERS
# =============================================================================

def safe_str(x):
    if pd.isna(x):
        return "NA"
    return str(x).strip()


def safe_float(x):
    try:
        value = float(x)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return np.nan


def first_existing(row_or_df, candidates):
    cols = row_or_df.index if isinstance(row_or_df, pd.Series) else row_or_df.columns
    lower = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def normalise_yes_no(x):
    x = safe_str(x).lower()
    if x in {"yes", "y", "true", "1"}:
        return "yes"
    if x in {"no", "n", "false", "0"}:
        return "no"
    return x


def ordered_levels(factor, levels):
    levels = list(dict.fromkeys(levels))
    preferred = LEVEL_ORDER.get(factor)
    if preferred is None:
        return sorted(levels)

    out = [x for x in preferred if x in levels]
    out.extend(sorted(x for x in levels if x not in out))
    return out


def bh_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    output = np.full(len(pvalues), np.nan)

    valid = np.where(np.isfinite(pvalues))[0]
    if len(valid) == 0:
        return output

    p = pvalues[valid]
    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)

    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)

    temp = np.empty(m)
    temp[order] = q
    output[valid] = temp
    return output


def wilcoxon_onesample(values):
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().values

    if len(x) < 2:
        return np.nan

    if not np.any(np.abs(x) > 0):
        return np.nan

    try:
        return float(
            stats.wilcoxon(
                x,
                alternative="two-sided",
                zero_method="wilcox",
            ).pvalue
        )
    except Exception:
        return np.nan


def spearman_safe(x, y):
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 2:
        return np.nan

    if df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return np.nan

    try:
        return float(stats.spearmanr(df["x"], df["y"]).statistic)
    except Exception:
        return np.nan


# =============================================================================
# 6. FILE / INPUT FINGERPRINTING
# =============================================================================

HASH_CACHE = {}
BIM_COUNT_CACHE = {}


def sha256_file(path):
    path = str(path)

    if path in HASH_CACHE:
        return HASH_CACHE[path]

    h = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)

    value = h.hexdigest()
    HASH_CACHE[path] = value
    return value


def count_lines(path):
    path = str(path)

    if path in BIM_COUNT_CACHE:
        return BIM_COUNT_CACHE[path]

    count = 0

    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            count += chunk.count(b"\n")

    BIM_COUNT_CACHE[path] = count
    return count


def choose_best_candidate(candidates):
    existing = [
        Path(x).resolve()
        for x in candidates
        if Path(x).is_file()
    ]

    if not existing:
        return None

    existing = sorted(
        existing,
        key=lambda x: (
            len(x.parts),
            len(str(x)),
            str(x),
        ),
    )

    return existing[0]


def find_bim(dataset_dir, fold, clumpprune):
    fold_dir = dataset_dir / f"Fold_{fold}"

    if clumpprune == "yes":
        exact = [
            fold_dir / "train_data.QC.clumped.pruned.bim",
            fold_dir / "train_data.clumped.pruned.bim",
        ]

        found = choose_best_candidate(exact)
        if found:
            return found

        recursive = list(
            fold_dir.glob("**/*clumped*pruned*.bim")
        )

        found = choose_best_candidate(recursive)
        if found:
            return found

    else:
        exact = [
            fold_dir / "train_data.QC.bim",
            fold_dir / "train_data.bim",
        ]

        found = choose_best_candidate(exact)
        if found:
            return found

        recursive = [
            x
            for x in fold_dir.glob("**/train_data*.bim")
            if "clumped" not in x.name.lower()
            and "pruned" not in x.name.lower()
        ]

        found = choose_best_candidate(recursive)
        if found:
            return found

    return None


def find_gwas_source(dataset_dir):
    """
    Find a dataset-level GWAS source used only for equality/fingerprint control.

    Prefer obvious top-level GWAS/summary files. If none exist, use the shortest
    top-level candidate among common GWAS-like formats.
    """
    dataset_dir = Path(dataset_dir)
    base = dataset_dir.name

    preferred = [
        dataset_dir / f"{base}.gz",
        dataset_dir / f"{base}.txt.gz",
        dataset_dir / f"{base}.tsv.gz",
        dataset_dir / f"{base}.ldsc.txt",
        dataset_dir / f"{base}.ldak",
        dataset_dir / "gwas.gz",
        dataset_dir / "gwas.txt.gz",
        dataset_dir / "gwas.tsv.gz",
        dataset_dir / "summary_stats.gz",
        dataset_dir / "summary_statistics.gz",
    ]

    found = choose_best_candidate(preferred)
    if found:
        return found

    obvious = []
    for pattern in [
        "*gwas*.gz",
        "*summary*.gz",
        "*.ldsc.txt",
        "*.ldak",
    ]:
        obvious.extend(dataset_dir.glob(pattern))

    found = choose_best_candidate(obvious)
    if found:
        return found

    generic = []
    for pattern in [
        "*.gz",
        "*.ldsc.txt",
        "*.ldak",
    ]:
        generic.extend(dataset_dir.glob(pattern))

    return choose_best_candidate(generic)


def sniff_inputs(dataset_dir, fold, method, clumpprune):
    meta = METHODS[method]

    genotype_path = None
    genotype_fingerprint = "NOT_USED"
    genotype_input_snps = np.nan
    genotype_verified = True

    if meta["uses_genotype"]:
        genotype_path = find_bim(
            dataset_dir=dataset_dir,
            fold=fold,
            clumpprune=clumpprune,
        )

        if genotype_path is None:
            genotype_fingerprint = "MISSING"
            genotype_verified = False
        else:
            genotype_fingerprint = sha256_file(genotype_path)
            genotype_input_snps = count_lines(genotype_path)

    gwas_path = None
    gwas_fingerprint = "NOT_USED"
    gwas_verified = True

    if meta["uses_gwas"]:
        gwas_path = find_gwas_source(dataset_dir)

        if gwas_path is None:
            gwas_fingerprint = "MISSING"
            gwas_verified = False
        else:
            gwas_fingerprint = sha256_file(gwas_path)

    return {
        "genotype_path": (
            str(genotype_path) if genotype_path else "NA"
        ),
        "genotype_fingerprint": genotype_fingerprint,
        "genotype_input_snps": genotype_input_snps,
        "genotype_verified": genotype_verified,
        "gwas_path": (
            str(gwas_path) if gwas_path else "NA"
        ),
        "gwas_fingerprint": gwas_fingerprint,
        "gwas_verified": gwas_verified,
    }


# =============================================================================
# 7. SIMULATION METADATA
# =============================================================================

def simulation_metadata(sim_name):
    try:
        number = int(sim_name.replace("Sim", ""))
    except Exception:
        return {
            "PhenotypeType": "unknown",
            "TargetN": np.nan,
            "TrueH2": np.nan,
            "CausalSNPs": np.nan,
            "Replicate": np.nan,
            "Scenario": "unknown",
            "UnderlyingTrait": "unknown",
        }

    if 1 <= number <= 100:
        phenotype_type = "Continuous"

        if number <= 50:
            target_n = 750
            local = number
        else:
            target_n = 5000
            local = number - 50

        underlying_trait = f"Continuous_T{local:03d}"

    elif 101 <= number <= 200:
        phenotype_type = "Binary"

        if number <= 150:
            target_n = 750
            local = number - 100
        else:
            target_n = 5000
            local = number - 150

        underlying_trait = f"Binary_T{local:03d}"

    else:
        return {
            "PhenotypeType": "unknown",
            "TargetN": np.nan,
            "TrueH2": np.nan,
            "CausalSNPs": np.nan,
            "Replicate": np.nan,
            "Scenario": "unknown",
            "UnderlyingTrait": "unknown",
        }

    h2_values = [0.05, 0.10, 0.30, 0.50, 0.70]
    true_h2 = h2_values[(local - 1) // 10]

    inside = (local - 1) % 10

    if inside < 5:
        causal = 100
        replicate = inside + 1
    else:
        causal = 5000
        replicate = inside - 4

    scenario = (
        f"{phenotype_type}"
        f" | N={target_n}"
        f" | h2={true_h2:.2f}"
        f" | causal={causal}"
    )

    return {
        "PhenotypeType": phenotype_type,
        "TargetN": target_n,
        "TrueH2": true_h2,
        "CausalSNPs": causal,
        "Replicate": replicate,
        "Scenario": scenario,
        "UnderlyingTrait": underlying_trait,
    }


# =============================================================================
# 8. CONFIGURATION PARSER
# =============================================================================

def parse_configuration(method, row):
    info = METHODS[method]

    model_col = first_existing(
        row,
        [
            "h2model_standardized",
            "h2model",
            "model",
        ],
    )

    model = (
        safe_str(row[model_col])
        if model_col is not None
        else "NA"
    )

    model_lower = model.lower()

    # -------------------------------------------------------------------------
    # CLUMPING
    # -------------------------------------------------------------------------
    cp_col = first_existing(row, ["clumpprune"])
    clumpprune = normalise_yes_no(
        row[cp_col] if cp_col is not None else "NA"
    )

    # -------------------------------------------------------------------------
    # COVARIATES / PCA
    # -------------------------------------------------------------------------
    if "genotype_covariate_pca" in model_lower:
        covariates = "yes"
        pca = "yes"
    elif "genotype_covariate" in model_lower:
        covariates = "yes"
        pca = "no"
    else:
        covariates = "no"
        pca = "no"

    # -------------------------------------------------------------------------
    # ALGORITHM
    # -------------------------------------------------------------------------
    if (
        "he regression" in model_lower
        or "haseman" in model_lower
    ):
        algorithm = "HE"

    elif (
        "reml" in model_lower
        or "ai algorithm" in model_lower
    ):
        algorithm = "REML"

    elif method == "GCTA-Heritability":
        algorithm = "GCTA_REML"

    elif method in {
        "LDSC-Heritability",
        "LDSC-Genotype-Heritability",
    }:
        algorithm = "LDSC"

    elif method == "LDpred-2-Heritability":
        algorithm = "LDpred2_LDSC"

    elif method in {
        "LDAK-Calculated-Heritability",
        "LDAK-Precomputed-Heritability",
    }:
        algorithm = "SumHer"

    else:
        algorithm = "NA"

    # -------------------------------------------------------------------------
    # GRM TYPE
    # -------------------------------------------------------------------------
    if "centered" in model_lower:
        grm_type = "centered"

    elif (
        "standardized" in model_lower
        or "standardised" in model_lower
    ):
        grm_type = "standardized"

    elif method == "GCTA-Heritability":
        grm_type = "GCTA"

    elif method in {
        "Gemma2-Heritability",
        "Gemma-LDSC-Heritability",
    }:
        grm_type = "centered"

    else:
        grm_type = "none"

    # -------------------------------------------------------------------------
    # SNP SET
    # -------------------------------------------------------------------------
    if "hapmap" in model_lower:
        snp_set = "hapmap"

    elif "genotyped" in model_lower:
        snp_set = "genotyped"

    elif (
        method == "LDpred-2-Heritability"
        and "full" in model_lower
    ):
        snp_set = "full"

    else:
        snp_set = "all"

    # -------------------------------------------------------------------------
    # LDAK TAGGING MODEL
    # -------------------------------------------------------------------------
    tagging_model = "NA"

    if method == "LDAK-Calculated-Heritability":

        if "ldak_human" in model_lower:
            tagging_model = "Human"

        elif "ldak_gcta" in model_lower:
            tagging_model = "GCTA"

        elif "bld-ldak" in model_lower:
            tagging_model = "BLD-LDAK"

        elif "alpha_" in model_lower:
            parts = model.split("_")
            lower_parts = [x.lower() for x in parts]

            try:
                index = lower_parts.index("alpha")
                tagging_model = (
                    "Alpha_"
                    + parts[index + 1]
                )
            except Exception:
                tagging_model = model

    elif method == "LDAK-Precomputed-Heritability":
        temp = model_lower
        temp = temp.replace("ldak_sim_precomputed_", "")
        temp = temp.replace("ldak_", "")
        temp = temp.replace(".tagging", "")

        for token in [
            ".hapmap.gbr",
            ".genotyped.gbr",
            "hapmap.gbr",
            "genotyped.gbr",
        ]:
            temp = temp.replace(token, "")

        tagging_model = (
            temp.strip("._")
            or "precomputed"
        )

    # -------------------------------------------------------------------------
    # REFERENCE SOURCE
    # -------------------------------------------------------------------------
    reference_source = info["reference_source"]

    if "sim_reference5000" in model_lower:
        reference_source = "simulation_fixed_N5000"

    if "simulationreference" in model_lower:
        reference_source = "simulation_genotype_derived"

    if "sim_precomputed" in model_lower:
        reference_source = "simulation_precomputed_N5000"

    # -------------------------------------------------------------------------
    # LDSC REFERENCE / WEIGHTS
    # -------------------------------------------------------------------------
    ld_reference = "NA"
    ld_weight = "NA"

    if method == "LDSC-Heritability":

        if "sim_reference5000" in model_lower:
            ld_reference = "SIM_N5000"
            ld_weight = "SIM_N5000"

        else:
            body = (
                model[5:]
                if model.startswith("LDSC_")
                else model
            )

            possible = [
                ("eur_ref_ld_chr", "eur_ref_ld_chr"),
                ("eur_ref_ld_chr", "eur_w_ld_chr"),
                ("eur_w_ld_chr", "eur_ref_ld_chr"),
                ("eur_w_ld_chr", "eur_w_ld_chr"),
            ]

            for reference, weight in possible:
                if body == reference + "_" + weight:
                    ld_reference = reference
                    ld_weight = weight
                    break

    elif method == "LDSC-Genotype-Heritability":
        ld_reference = "genotype_derived"
        ld_weight = "genotype_derived"

    def raw(name, default="NA"):
        c = first_existing(row, [name])
        return safe_str(row[c]) if c is not None else default

    return {
        "family": info["family"],
        "engine": info["engine"],
        "input_type": info["input_type"],
        "uses_genotype": info["uses_genotype"],
        "uses_gwas": info["uses_gwas"],

        "clumpprune": clumpprune,
        "covariates": covariates,
        "pca": pca,
        "snp_set": snp_set,

        "algorithm": algorithm,
        "grm_type": grm_type,
        "tagging_model": tagging_model,

        "grm_source": info["grm_source"],
        "reference_source": reference_source,

        "ld_reference": ld_reference,
        "ld_weight": ld_weight,

        "clump_p1_raw": raw("clump_p1"),
        "clump_r2_raw": raw("clump_r2"),
        "clump_kb_raw": raw("clump_kb"),

        "p_window_size_raw": raw("p_window_size"),
        "p_slide_size_raw": raw("p_slide_size"),
        "p_LD_threshold_raw": raw("p_LD_threshold"),

        "numberofpca_raw": raw("numberofpca"),

        "raw_h2model": model,
    }


# =============================================================================
# 9. HARMONISED h2 SELECTION
# =============================================================================

def choose_analysis_h2(mode, dataset, row):
    """
    Return:
        active_h2, native_h2, analysis_scale, trait_type, exclusion_reason

    STRICT RULE:
        binary + missing h2_liability_primary -> EXCLUDE.
        No native fallback is allowed.
    """
    native_col = first_existing(
        row,
        [
            "h2_native",
            "original_h2",
            "h2",
        ],
    )

    liab_col = first_existing(
        row,
        [
            "h2_liability_primary",
            "liability_h2",
        ],
    )

    trait_col = first_existing(
        row,
        [
            "trait_type",
            "PhenotypeType",
        ],
    )

    native_h2 = (
        safe_float(row[native_col])
        if native_col is not None
        else np.nan
    )

    liability_h2 = (
        safe_float(row[liab_col])
        if liab_col is not None
        else np.nan
    )

    if mode == "real":
        if dataset == "body_mass_index_bmi":
            trait_type = "continuous"
        else:
            trait_type = "binary"

    else:
        meta = simulation_metadata(dataset)
        trait_type = meta["PhenotypeType"].lower()

    if trait_col is not None:
        stated = safe_str(row[trait_col]).lower()

        if stated in {"binary", "continuous"}:
            trait_type = stated

    if trait_type == "continuous":
        if not np.isfinite(native_h2):
            return (
                np.nan,
                native_h2,
                "native_quantitative",
                trait_type,
                "missing_native_h2_continuous",
            )

        return (
            native_h2,
            native_h2,
            "native_quantitative",
            trait_type,
            "",
        )

    if trait_type == "binary":
        if not np.isfinite(liability_h2):
            return (
                np.nan,
                native_h2,
                "liability_primary",
                trait_type,
                "missing_liability_h2_binary",
            )

        return (
            liability_h2,
            native_h2,
            "liability_primary",
            trait_type,
            "",
        )

    return (
        np.nan,
        native_h2,
        "unknown",
        trait_type,
        "unknown_trait_type",
    )


# =============================================================================
# 10. DATASET / FILE DISCOVERY
# =============================================================================

def discover_datasets(mode, root):
    if mode == "real":
        return [
            x
            for x in REAL_PHENOTYPES
            if (root / x).is_dir()
        ]

    sims = []

    for p in sorted(root.glob("Sim[0-9][0-9][0-9]")):
        if not p.is_dir():
            continue

        try:
            n = int(p.name[3:])
        except Exception:
            continue

        if 1 <= n <= 200:
            sims.append(p.name)

    return sims


def results_liability_path(root, dataset, fold, method):
    return (
        root
        / dataset
        / f"Fold_{fold}"
        / method
        / "Results_Liability.csv"
    )


# =============================================================================
# 11. LOAD HARMONISED RESULTS
# =============================================================================

def load_mode_data(mode, root):
    datasets = discover_datasets(mode, root)

    print()
    print("=" * 140)
    print("FINAL ANALYSIS 04 — HARMONISED EXACT ONE-FACTOR MATCHED ANALYSIS")
    print("=" * 140)
    print("MODE:", mode.upper())
    print("ROOT:", root)
    print("Datasets found:", len(datasets))
    print("HERITABILITY SOURCE: Results_Liability.csv ONLY")
    print()

    rows = []
    excluded = []
    row_audit = []

    files_found = 0
    files_missing = 0
    rows_seen = 0
    rows_used = 0

    input_sniff_cache = {}

    for dataset in datasets:
        dataset_dir = root / dataset

        for fold in range(5):
            for method in METHODS:

                path = results_liability_path(
                    root=root,
                    dataset=dataset,
                    fold=fold,
                    method=method,
                )

                if not path.exists():
                    files_missing += 1
                    continue

                files_found += 1

                try:
                    result_df = pd.read_csv(path)
                except Exception as exc:
                    excluded.append({
                        "Dataset": dataset,
                        "Fold": fold,
                        "Method": method,
                        "Reason": f"read_error:{repr(exc)}",
                        "SourceFile": str(path),
                    })
                    continue

                if result_df.empty:
                    continue

                for source_row, row in result_df.iterrows():
                    rows_seen += 1

                    active_h2, native_h2, analysis_scale, trait_type, reason = (
                        choose_analysis_h2(
                            mode=mode,
                            dataset=dataset,
                            row=row,
                        )
                    )

                    conversion_col = first_existing(
                        row,
                        ["conversion_status"],
                    )

                    conversion_status = (
                        safe_str(row[conversion_col])
                        if conversion_col is not None
                        else "NA"
                    )

                    psource_col = first_existing(
                        row,
                        ["P_source"],
                    )

                    p_source = (
                        safe_str(row[psource_col])
                        if psource_col is not None
                        else "NA"
                    )

                    row_audit.append({
                        "Dataset": dataset,
                        "Fold": fold,
                        "Method": method,
                        "SourceRow": source_row,
                        "TraitType": trait_type,
                        "AnalysisScale": analysis_scale,
                        "NativeH2": native_h2,
                        "AnalysisH2": active_h2,
                        "ConversionStatus": conversion_status,
                        "PSource": p_source,
                        "Included": np.isfinite(active_h2),
                        "ExclusionReason": reason,
                        "SourceFile": str(path),
                    })

                    if not np.isfinite(active_h2):
                        excluded.append({
                            "Dataset": dataset,
                            "Fold": fold,
                            "Method": method,
                            "SourceRow": source_row,
                            "TraitType": trait_type,
                            "ConversionStatus": conversion_status,
                            "PSource": p_source,
                            "NativeH2": native_h2,
                            "Reason": reason,
                            "SourceFile": str(path),
                        })
                        continue

                    metadata = parse_configuration(
                        method=method,
                        row=row,
                    )

                    sniff_key = (
                        dataset,
                        fold,
                        method,
                        metadata["clumpprune"],
                    )

                    if sniff_key not in input_sniff_cache:
                        input_sniff_cache[sniff_key] = sniff_inputs(
                            dataset_dir=dataset_dir,
                            fold=fold,
                            method=method,
                            clumpprune=metadata["clumpprune"],
                        )

                    sniff = input_sniff_cache[sniff_key]

                    variant_col = first_existing(
                        row,
                        [
                            "variants_standardized",
                            "numberofvariants",
                            "number_of_variants",
                            "variants",
                        ],
                    )

                    effective_variants = (
                        safe_float(row[variant_col])
                        if variant_col is not None
                        else np.nan
                    )

                    record = {
                        "Dataset": dataset,
                        "Fold": fold,
                        "Method": method,

                        # PRIMARY revised-manuscript h2
                        "h2": active_h2,

                        # Same-row native h2 retained only for scale comparison
                        "h2_native_compare": native_h2,

                        "TraitType": trait_type,
                        "AnalysisScale": analysis_scale,
                        "ConversionStatus": conversion_status,
                        "PSource": p_source,

                        "effective_variants": effective_variants,

                        "SourceRow": source_row,
                        "SourceFile": str(path),
                    }

                    record.update(metadata)
                    record.update(sniff)

                    if mode == "real":
                        record["DatasetLabel"] = PHENOTYPE_LABELS.get(
                            dataset,
                            dataset,
                        )
                        record["Scenario"] = record["DatasetLabel"]
                        record["UnderlyingTrait"] = dataset
                        record["TargetN"] = np.nan
                        record["TrueH2"] = np.nan
                        record["CausalSNPs"] = np.nan
                        record["Replicate"] = np.nan

                    else:
                        smeta = simulation_metadata(dataset)
                        record.update(smeta)
                        record["DatasetLabel"] = dataset

                    rows.append(record)
                    rows_used += 1

    data = pd.DataFrame(rows)
    excluded_df = pd.DataFrame(excluded)
    row_audit_df = pd.DataFrame(row_audit)

    print("Results_Liability.csv files found:  ", files_found)
    print("Results_Liability.csv files missing:", files_missing)
    print("Rows seen:                           ", rows_seen)
    print("Rows used on harmonised scale:       ", rows_used)
    print("Rows excluded:                       ", len(excluded_df))

    if data.empty:
        raise SystemExit(
            "ERROR: no usable harmonised heritability rows found."
        )

    print()
    print("ANALYSIS-SCALE COUNTS")
    print(
        data["AnalysisScale"]
        .value_counts(dropna=False)
        .to_string()
    )

    print()
    print("TRAIT-TYPE COUNTS")
    print(
        data["TraitType"]
        .value_counts(dropna=False)
        .to_string()
    )

    if not excluded_df.empty:
        print()
        print("EXCLUDED ROWS BY REASON")
        print(
            excluded_df["Reason"]
            .value_counts(dropna=False)
            .to_string()
        )

    # Critical safety checks
    binary_used = data[data["TraitType"] == "binary"]

    if len(binary_used):
        wrong_binary_scale = binary_used[
            binary_used["AnalysisScale"] != "liability_primary"
        ]

        if len(wrong_binary_scale):
            raise RuntimeError(
                "SAFETY STOP: at least one included binary row is not "
                "using h2_liability_primary."
            )

    continuous_used = data[data["TraitType"] == "continuous"]

    if len(continuous_used):
        wrong_continuous_scale = continuous_used[
            continuous_used["AnalysisScale"] != "native_quantitative"
        ]

        if len(wrong_continuous_scale):
            raise RuntimeError(
                "SAFETY STOP: at least one included continuous row is not "
                "using h2_native."
            )

    return data, excluded_df, row_audit_df


# =============================================================================
# 12. INPUT-SNIFF AUDIT
# =============================================================================

def print_input_audit(data):
    print()
    print("=" * 140)
    print("INPUT-SNIFF AUDIT")
    print("=" * 140)

    genotype_rows = data[data["uses_genotype"] == True].copy()
    gwas_rows = data[data["uses_gwas"] == True].copy()

    if len(genotype_rows):
        print(
            "Genotype-using rows:",
            len(genotype_rows),
            "| verified:",
            int(genotype_rows["genotype_verified"].sum()),
            "| unverified:",
            int((~genotype_rows["genotype_verified"]).sum()),
        )

    if len(gwas_rows):
        print(
            "GWAS-using rows:    ",
            len(gwas_rows),
            "| verified:",
            int(gwas_rows["gwas_verified"].sum()),
            "| unverified:",
            int((~gwas_rows["gwas_verified"]).sum()),
        )


# =============================================================================
# 13. MATCH CONSTRUCTION
# =============================================================================

def factor_match_columns(definition):
    factor = definition["factor"]
    ignore = set(
        definition.get(
            "ignore_controls",
            set(),
        )
    )

    columns = [
        "Dataset",
        "Fold",
    ]

    if definition["scope"] == "within_method":
        columns.append("Method")
    else:
        # Cross-method comparisons at minimum share the estimator engine.
        columns.append("engine")

    for column in CONTROL_FIELDS:
        if column == factor:
            continue

        if column in ignore:
            continue

        if column in {
            "genotype_fingerprint",
            "gwas_fingerprint",
        }:
            continue

        columns.append(column)

    if definition["require_same_genotype"]:
        columns.append("genotype_fingerprint")

    if definition["require_same_gwas"]:
        columns.append("gwas_fingerprint")

    return list(dict.fromkeys(columns))


def build_all_pairs(data):
    pair_rows = []
    factor_audit = []

    for definition in FACTOR_DEFINITIONS:
        factor = definition["factor"]
        label = definition["label"]
        category = definition["category"]

        temp = data.copy()

        temp = temp[
            temp[factor].notna()
        ].copy()

        temp = temp[
            ~temp[factor]
            .astype(str)
            .str.lower()
            .isin({"na", "nan", ""})
        ].copy()

        if definition["require_same_genotype"]:
            temp = temp[
                (~temp["uses_genotype"])
                | (temp["genotype_verified"])
            ].copy()

        if definition["require_same_gwas"]:
            temp = temp[
                (~temp["uses_gwas"])
                | (temp["gwas_verified"])
            ].copy()

        match_columns = factor_match_columns(definition)

        for column in match_columns + [factor]:
            temp[column] = (
                temp[column]
                .fillna("NA")
                .astype(str)
            )

        candidate_groups = 0
        matched_groups = 0

        grouped = temp.groupby(
            match_columns,
            dropna=False,
            sort=False,
        )

        for _, group in grouped:
            levels = ordered_levels(
                factor,
                group[factor].unique().tolist(),
            )

            if len(levels) < 2:
                continue

            candidate_groups += 1

            for level_a, level_b in itertools.combinations(levels, 2):
                group_a = group[
                    group[factor] == level_a
                ]

                group_b = group[
                    group[factor] == level_b
                ]

                if group_a.empty or group_b.empty:
                    continue

                # Primary harmonised analysis
                h2_a = pd.to_numeric(
                    group_a["h2"],
                    errors="coerce",
                ).mean()

                h2_b = pd.to_numeric(
                    group_b["h2"],
                    errors="coerce",
                ).mean()

                if not (
                    np.isfinite(h2_a)
                    and np.isfinite(h2_b)
                ):
                    continue

                # Native comparison on the exact same matched rows
                native_a = pd.to_numeric(
                    group_a["h2_native_compare"],
                    errors="coerce",
                ).mean()

                native_b = pd.to_numeric(
                    group_b["h2_native_compare"],
                    errors="coerce",
                ).mean()

                effective_a = pd.to_numeric(
                    group_a["effective_variants"],
                    errors="coerce",
                ).mean()

                effective_b = pd.to_numeric(
                    group_b["effective_variants"],
                    errors="coerce",
                ).mean()

                if (
                    np.isfinite(effective_a)
                    and np.isfinite(effective_b)
                ):
                    effective_delta = effective_b - effective_a
                else:
                    effective_delta = np.nan

                input_snps_a = pd.to_numeric(
                    group_a["genotype_input_snps"],
                    errors="coerce",
                ).mean()

                input_snps_b = pd.to_numeric(
                    group_b["genotype_input_snps"],
                    errors="coerce",
                ).mean()

                method_a = ",".join(
                    sorted(group_a["Method"].unique())
                )

                method_b = ",".join(
                    sorted(group_b["Method"].unique())
                )

                stratum_bits = []

                for column in match_columns:
                    if column in {
                        "Dataset",
                        "Fold",
                    }:
                        continue

                    stratum_bits.append(
                        f"{column}="
                        f"{safe_str(group.iloc[0][column])}"
                    )

                stratum = " | ".join(stratum_bits)

                first = group.iloc[0]

                pair_rows.append({
                    "Category": category,
                    "Factor": factor,
                    "FactorLabel": label,

                    "Dataset": first["Dataset"],
                    "DatasetLabel": first["DatasetLabel"],
                    "Scenario": first["Scenario"],
                    "UnderlyingTrait": first["UnderlyingTrait"],
                    "TargetN": first["TargetN"],
                    "TrueH2": first["TrueH2"],
                    "CausalSNPs": first["CausalSNPs"],
                    "Replicate": first["Replicate"],

                    "Fold": int(first["Fold"]),

                    "Method_A": method_a,
                    "Method_B": method_b,

                    "Level_A": level_a,
                    "Level_B": level_b,
                    "Contrast": f"{level_b} - {level_a}",

                    "h2_A": h2_a,
                    "h2_B": h2_b,
                    "Delta": h2_b - h2_a,
                    "AbsDelta": abs(h2_b - h2_a),

                    "Native_h2_A": native_a,
                    "Native_h2_B": native_b,
                    "NativeDelta": (
                        native_b - native_a
                        if np.isfinite(native_a)
                        and np.isfinite(native_b)
                        else np.nan
                    ),
                    "NativeAbsDelta": (
                        abs(native_b - native_a)
                        if np.isfinite(native_a)
                        and np.isfinite(native_b)
                        else np.nan
                    ),

                    "EffectiveVariants_A": effective_a,
                    "EffectiveVariants_B": effective_b,
                    "EffectiveVariantDelta": effective_delta,

                    "InputGenotypeSNPs_A": input_snps_a,
                    "InputGenotypeSNPs_B": input_snps_b,

                    "Stratum": stratum,
                })

                matched_groups += 1

        factor_audit.append({
            "Category": category,
            "Factor": factor,
            "FactorLabel": label,
            "CandidateGroups": candidate_groups,
            "MatchedComparisons": matched_groups,
        })

    return (
        pd.DataFrame(pair_rows),
        pd.DataFrame(factor_audit),
    )


# =============================================================================
# 14. AVERAGE OVER OVERLAPPING FOLDS
# =============================================================================

def average_folds(pairs_fold):
    if pairs_fold.empty:
        return pd.DataFrame()

    group_cols = [
        "Category",
        "Factor",
        "FactorLabel",

        "Dataset",
        "DatasetLabel",
        "Scenario",
        "UnderlyingTrait",
        "TargetN",
        "TrueH2",
        "CausalSNPs",
        "Replicate",

        "Method_A",
        "Method_B",

        "Contrast",
        "Level_A",
        "Level_B",

        "Stratum",
    ]

    pair_units = (
        pairs_fold
        .groupby(
            group_cols,
            as_index=False,
            dropna=False,
        )
        .agg(
            Delta=("Delta", "mean"),
            AbsDelta=("AbsDelta", "mean"),
            NativeDelta=("NativeDelta", "mean"),
            NativeAbsDelta=("NativeAbsDelta", "mean"),

            CommonFolds=("Fold", "nunique"),

            h2_A=("h2_A", "mean"),
            h2_B=("h2_B", "mean"),

            Native_h2_A=("Native_h2_A", "mean"),
            Native_h2_B=("Native_h2_B", "mean"),

            EffectiveVariants_A=("EffectiveVariants_A", "mean"),
            EffectiveVariants_B=("EffectiveVariants_B", "mean"),
            EffectiveVariantDelta=("EffectiveVariantDelta", "mean"),

            InputGenotypeSNPs_A=("InputGenotypeSNPs_A", "mean"),
            InputGenotypeSNPs_B=("InputGenotypeSNPs_B", "mean"),
        )
    )

    return pair_units


# =============================================================================
# 15. INDEPENDENT INFERENTIAL UNITS
# =============================================================================

def build_independent_unit_effects(mode, pair_units):
    """
    REAL:
        one mean effect per phenotype.

    SIMULATED:
        first one mean effect per simulation dataset;
        then average N=750 and N=5000 versions sharing one UnderlyingTrait.
    """
    contrast_cols = [
        "Category",
        "Factor",
        "FactorLabel",
        "Contrast",
        "Level_A",
        "Level_B",
    ]

    if mode == "real":
        unit = (
            pair_units
            .groupby(
                contrast_cols
                + [
                    "Dataset",
                    "DatasetLabel",
                ],
                as_index=False,
                dropna=False,
            )
            .agg(
                Effect=("Delta", "mean"),
                AbsEffect=("AbsDelta", "mean"),
                NativeEffect=("NativeDelta", "mean"),
                NativeAbsEffect=("NativeAbsDelta", "mean"),
                MatchedPairs=("Delta", "size"),
            )
        )

        unit["IndependentUnit"] = unit["Dataset"]
        unit["IndependentUnitLabel"] = unit["DatasetLabel"]
        unit["NestedDatasets"] = 1

        return unit

    # -------------------------------------------------------------------------
    # SIMULATION: first one effect per SimXXX dataset
    # -------------------------------------------------------------------------
    per_dataset = (
        pair_units
        .groupby(
            contrast_cols
            + [
                "Dataset",
                "DatasetLabel",
                "UnderlyingTrait",
                "TargetN",
                "TrueH2",
                "CausalSNPs",
                "Replicate",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            Effect=("Delta", "mean"),
            AbsEffect=("AbsDelta", "mean"),
            NativeEffect=("NativeDelta", "mean"),
            NativeAbsEffect=("NativeAbsDelta", "mean"),
            MatchedPairs=("Delta", "size"),
        )
    )

    # -------------------------------------------------------------------------
    # Then average the nested N=750 / N=5000 versions.
    # -------------------------------------------------------------------------
    unit = (
        per_dataset
        .groupby(
            contrast_cols
            + [
                "UnderlyingTrait",
                "TrueH2",
                "CausalSNPs",
                "Replicate",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            Effect=("Effect", "mean"),
            AbsEffect=("AbsEffect", "mean"),
            NativeEffect=("NativeEffect", "mean"),
            NativeAbsEffect=("NativeAbsEffect", "mean"),
            MatchedPairs=("MatchedPairs", "sum"),
            NestedDatasets=("Dataset", "nunique"),
        )
    )

    unit["IndependentUnit"] = unit["UnderlyingTrait"]
    unit["IndependentUnitLabel"] = unit["UnderlyingTrait"]

    return unit


# =============================================================================
# 16. CONTRAST-LEVEL INFERENCE
# =============================================================================

def build_inferential_summary(independent_effects):
    summary_rows = []

    group_cols = [
        "Category",
        "Factor",
        "FactorLabel",
        "Contrast",
        "Level_A",
        "Level_B",
    ]

    for keys, subset in independent_effects.groupby(
        group_cols,
        dropna=False,
    ):
        (
            category,
            factor,
            label,
            contrast,
            level_a,
            level_b,
        ) = keys

        active = pd.to_numeric(
            subset["Effect"],
            errors="coerce",
        ).dropna()

        active_abs = pd.to_numeric(
            subset["AbsEffect"],
            errors="coerce",
        ).dropna()

        native = pd.to_numeric(
            subset["NativeEffect"],
            errors="coerce",
        ).dropna()

        native_abs = pd.to_numeric(
            subset["NativeAbsEffect"],
            errors="coerce",
        ).dropna()

        p_active = wilcoxon_onesample(active.values)
        p_native = wilcoxon_onesample(native.values)

        all_zero = bool(
            len(active) > 0
            and np.all(
                np.isclose(
                    active.values,
                    0.0,
                    rtol=0,
                    atol=1e-15,
                )
            )
        )

        pct_positive = (
            100.0 * np.mean(active.values > 0)
            if len(active)
            else np.nan
        )

        pct_negative = (
            100.0 * np.mean(active.values < 0)
            if len(active)
            else np.nan
        )

        rho_native_active = spearman_safe(
            subset["NativeEffect"],
            subset["Effect"],
        )

        active_median = (
            float(active.median())
            if len(active)
            else np.nan
        )

        native_median = (
            float(native.median())
            if len(native)
            else np.nan
        )

        if np.isfinite(active_median) and np.isfinite(native_median):
            direction_preserved = (
                np.sign(active_median)
                == np.sign(native_median)
            )
        else:
            direction_preserved = np.nan

        summary_rows.append({
            "Category": category,
            "Factor": factor,
            "FactorLabel": label,
            "Contrast": contrast,
            "Level_A": level_a,
            "Level_B": level_b,

            "IndependentUnits": subset[
                "IndependentUnit"
            ].nunique(),

            "MatchedPairContributions": int(
                pd.to_numeric(
                    subset["MatchedPairs"],
                    errors="coerce",
                )
                .fillna(0)
                .sum()
            ),

            # Harmonised / primary
            "MeanDelta_Harmonised": active.mean(),
            "MedianDelta_Harmonised": active.median(),
            "MeanAbsDelta_Harmonised": active_abs.mean(),
            "MedianAbsDelta_Harmonised": active_abs.median(),
            "Q25Delta_Harmonised": active.quantile(0.25),
            "Q75Delta_Harmonised": active.quantile(0.75),
            "MinDelta_Harmonised": active.min(),
            "MaxDelta_Harmonised": active.max(),
            "PctPositive_Harmonised": pct_positive,
            "PctNegative_Harmonised": pct_negative,
            "WilcoxonP_Harmonised": p_active,

            # Native comparison on same eligible units
            "MeanDelta_Native": native.mean(),
            "MedianDelta_Native": native.median(),
            "MeanAbsDelta_Native": native_abs.mean(),
            "MedianAbsDelta_Native": native_abs.median(),
            "WilcoxonP_Native": p_native,

            "Spearman_Native_vs_Harmonised":
                rho_native_active,

            "MedianDirectionPreserved":
                direction_preserved,

            "AllZero_Harmonised": all_zero,
        })

    summary = pd.DataFrame(summary_rows)

    if summary.empty:
        return summary

    summary["FDR_Q_Harmonised"] = bh_adjust(
        summary["WilcoxonP_Harmonised"].values
    )

    summary["FDR_Q_Native"] = bh_adjust(
        summary["WilcoxonP_Native"].values
    )

    summary["FDR_Significant_Harmonised"] = np.where(
        pd.to_numeric(
            summary["FDR_Q_Harmonised"],
            errors="coerce",
        ) < 0.05,
        "YES",
        "NO",
    )

    summary.loc[
        summary["FDR_Q_Harmonised"].isna(),
        "FDR_Significant_Harmonised",
    ] = "NA"

    summary["FDR_Significant_Native"] = np.where(
        pd.to_numeric(
            summary["FDR_Q_Native"],
            errors="coerce",
        ) < 0.05,
        "YES",
        "NO",
    )

    summary.loc[
        summary["FDR_Q_Native"].isna(),
        "FDR_Significant_Native",
    ] = "NA"

    summary["SignificancePreserved"] = (
        summary["FDR_Significant_Harmonised"]
        == summary["FDR_Significant_Native"]
    )

    return summary


# =============================================================================
# 17. FACTOR RANKING
# =============================================================================

def build_factor_ranking(mode, pair_units):
    """
    Dataset-balanced factor ranking.

    Real:
        balance by phenotype.

    Simulated:
        balance first by simulation dataset here for descriptive magnitude.
        Inferential independence is handled separately in the Wilcoxon table.
    """
    factor_dataset = (
        pair_units
        .groupby(
            [
                "Category",
                "Factor",
                "FactorLabel",
                "Dataset",
                "DatasetLabel",
            ],
            as_index=False,
        )
        .agg(
            MeanAbsDelta_Harmonised=(
                "AbsDelta",
                "mean",
            ),
            MeanAbsDelta_Native=(
                "NativeAbsDelta",
                "mean",
            ),
            MeanSignedDelta_Harmonised=(
                "Delta",
                "mean",
            ),
            MeanSignedDelta_Native=(
                "NativeDelta",
                "mean",
            ),
            MatchedPairs=(
                "Delta",
                "size",
            ),
        )
    )

    pair_counts = (
        pair_units
        .groupby(
            [
                "Category",
                "Factor",
                "FactorLabel",
            ],
            as_index=False,
        )
        .agg(
            TotalMatchedPairs=(
                "Delta",
                "size",
            )
        )
    )

    ranking = (
        factor_dataset
        .groupby(
            [
                "Category",
                "Factor",
                "FactorLabel",
            ],
            as_index=False,
        )
        .agg(
            Datasets=("Dataset", "nunique"),

            MeanAbsDelta_Harmonised=(
                "MeanAbsDelta_Harmonised",
                "mean",
            ),
            MedianAbsDelta_Harmonised=(
                "MeanAbsDelta_Harmonised",
                "median",
            ),
            MaxAbsDelta_Harmonised=(
                "MeanAbsDelta_Harmonised",
                "max",
            ),

            MeanAbsDelta_Native=(
                "MeanAbsDelta_Native",
                "mean",
            ),
            MedianAbsDelta_Native=(
                "MeanAbsDelta_Native",
                "median",
            ),
            MaxAbsDelta_Native=(
                "MeanAbsDelta_Native",
                "max",
            ),
        )
        .merge(
            pair_counts,
            on=[
                "Category",
                "Factor",
                "FactorLabel",
            ],
            how="left",
        )
    )

    ranking["Rank_Harmonised"] = (
        ranking["MedianAbsDelta_Harmonised"]
        .rank(
            method="min",
            ascending=False,
        )
    )

    ranking["Rank_Native"] = (
        ranking["MedianAbsDelta_Native"]
        .rank(
            method="min",
            ascending=False,
        )
    )

    ranking["RankShift_HarmonisedMinusNative"] = (
        ranking["Rank_Harmonised"]
        - ranking["Rank_Native"]
    )

    ranking = ranking.sort_values(
        [
            "Rank_Harmonised",
            "MedianAbsDelta_Harmonised",
        ],
        ascending=[
            True,
            False,
        ],
    )

    return ranking


# =============================================================================
# 18. CATEGORY SUMMARY
# =============================================================================

def build_category_summary(pair_units):
    factor_dataset = (
        pair_units
        .groupby(
            [
                "Category",
                "Factor",
                "FactorLabel",
                "Dataset",
                "DatasetLabel",
            ],
            as_index=False,
        )
        .agg(
            MeanAbsDelta_Harmonised=(
                "AbsDelta",
                "mean",
            ),
            MeanAbsDelta_Native=(
                "NativeAbsDelta",
                "mean",
            ),
        )
    )

    category_dataset = (
        factor_dataset
        .groupby(
            [
                "Category",
                "Dataset",
                "DatasetLabel",
            ],
            as_index=False,
        )
        .agg(
            MeanAbsDelta_Harmonised=(
                "MeanAbsDelta_Harmonised",
                "mean",
            ),
            MeanAbsDelta_Native=(
                "MeanAbsDelta_Native",
                "mean",
            ),
        )
    )

    category_pairs = (
        pair_units
        .groupby(
            "Category",
            as_index=False,
        )
        .agg(
            TotalMatchedPairs=(
                "Delta",
                "size",
            )
        )
    )

    category_summary = (
        category_dataset
        .groupby(
            "Category",
            as_index=False,
        )
        .agg(
            Datasets=("Dataset", "nunique"),

            MeanAbsDelta_Harmonised=(
                "MeanAbsDelta_Harmonised",
                "mean",
            ),
            MedianAbsDelta_Harmonised=(
                "MeanAbsDelta_Harmonised",
                "median",
            ),
            MaxAbsDelta_Harmonised=(
                "MeanAbsDelta_Harmonised",
                "max",
            ),

            MeanAbsDelta_Native=(
                "MeanAbsDelta_Native",
                "mean",
            ),
            MedianAbsDelta_Native=(
                "MeanAbsDelta_Native",
                "median",
            ),
            MaxAbsDelta_Native=(
                "MeanAbsDelta_Native",
                "max",
            ),
        )
        .merge(
            category_pairs,
            on="Category",
            how="left",
        )
        .sort_values(
            "MedianAbsDelta_Harmonised",
            ascending=False,
        )
    )

    return category_summary


# =============================================================================
# 19. SCALE COMPARISON TABLE
# =============================================================================

def build_scale_comparison(summary, ranking):
    if summary.empty:
        return pd.DataFrame()

    keep = [
        "Category",
        "Factor",
        "FactorLabel",
        "Contrast",
        "IndependentUnits",

        "MedianDelta_Native",
        "MedianDelta_Harmonised",

        "MedianAbsDelta_Native",
        "MedianAbsDelta_Harmonised",

        "WilcoxonP_Native",
        "FDR_Q_Native",
        "FDR_Significant_Native",

        "WilcoxonP_Harmonised",
        "FDR_Q_Harmonised",
        "FDR_Significant_Harmonised",

        "Spearman_Native_vs_Harmonised",
        "MedianDirectionPreserved",
        "SignificancePreserved",
    ]

    out = summary[keep].copy()

    rank_keep = ranking[
        [
            "Factor",
            "Rank_Native",
            "Rank_Harmonised",
            "RankShift_HarmonisedMinusNative",
        ]
    ].drop_duplicates("Factor")

    out = out.merge(
        rank_keep,
        on="Factor",
        how="left",
    )

    return out


# =============================================================================
# 20. PRINT REPORT
# =============================================================================

def print_report(
    mode,
    data,
    excluded,
    factor_audit,
    pairs_fold,
    pair_units,
    independent_effects,
    summary,
    ranking,
    category_summary,
):
    print()
    print("=" * 140)
    print("MATCH-CONSTRUCTION AUDIT")
    print("=" * 140)

    if factor_audit.empty:
        print("No factor audit available.")
    else:
        print(factor_audit.to_string(index=False))

    print()
    print("STRUCTURALLY CONFOUNDED / NOT CLAIMED AS ONE-FACTOR EFFECT:")
    for item in STRUCTURALLY_CONFOUNDED:
        print(" -", item)

    print()
    print("=" * 140)
    print("MATCHING SUMMARY")
    print("=" * 140)
    print("Fold-level exact matches:                        ", len(pairs_fold))
    print("Matched configuration pairs after fold averaging:", len(pair_units))

    if mode == "real":
        independent_label = "phenotypes"
    else:
        independent_label = "underlying simulated traits after nested-N averaging"

    print(
        f"Independent inferential units ({independent_label}):",
        independent_effects["IndependentUnit"].nunique()
        if not independent_effects.empty
        else 0,
    )

    print()
    print("=" * 140)
    print("MULTIPLE-TESTING POLICY")
    print("=" * 140)
    print("Primary revised scale : harmonised")
    print("Raw test              : two-sided Wilcoxon signed-rank")
    print(
        "Independent unit      :",
        (
            "phenotype"
            if mode == "real"
            else "underlying simulated trait; nested N averaged first"
        ),
    )
    print("Correction            : Benjamini-Hochberg FDR")
    print("Primary significance  : q < 0.05")
    print("Native-scale results  : comparison only, same eligible matched rows")

    if summary.empty:
        print()
        print("NO VALID INFERENTIAL CONTRASTS.")
        return

    print()
    print("=" * 180)
    print("FINAL INFERENTIAL TABLE — HARMONISED SCALE IS PRIMARY")
    print("=" * 180)

    inferential = summary.sort_values(
        [
            "FDR_Q_Harmonised",
            "MedianAbsDelta_Harmonised",
        ],
        ascending=[True, False],
        na_position="last",
    )

    header = (
        f"{'Category':<36}"
        f"{'Factor':<31}"
        f"{'Contrast':<34}"
        f"{'N':>5}"
        f"{'Med D H':>12}"
        f"{'Med |D| H':>12}"
        f"{'P H':>11}"
        f"{'Q H':>11}"
        f"{'Sig H':>7}"
        f"{'Sig N':>7}"
        f"{'rho N/H':>10}"
    )

    print(header)
    print("-" * len(header))

    for _, row in inferential.iterrows():
        def fmt(x, nd=4):
            return (
                f"{x:.{nd}g}"
                if pd.notna(x)
                else "NA"
            )

        rho = (
            f"{row['Spearman_Native_vs_Harmonised']:.3f}"
            if pd.notna(row["Spearman_Native_vs_Harmonised"])
            else "NA"
        )

        print(
            f"{row['Category']:<36}"
            f"{row['FactorLabel']:<31}"
            f"{row['Contrast']:<34}"
            f"{int(row['IndependentUnits']):>5}"
            f"{row['MedianDelta_Harmonised']:>12.5f}"
            f"{row['MedianAbsDelta_Harmonised']:>12.5f}"
            f"{fmt(row['WilcoxonP_Harmonised']):>11}"
            f"{fmt(row['FDR_Q_Harmonised']):>11}"
            f"{row['FDR_Significant_Harmonised']:>7}"
            f"{row['FDR_Significant_Native']:>7}"
            f"{rho:>10}"
        )

    print()
    print("=" * 140)
    print("FINAL FACTOR RANKING — HARMONISED SCALE")
    print("=" * 140)

    print(
        f"{'Rank':<6}"
        f"{'Category':<40}"
        f"{'Factor':<40}"
        f"{'Pairs':>9}"
        f"{'Data':>8}"
        f"{'Med |D| H':>14}"
        f"{'Med |D| N':>14}"
        f"{'Native rank':>13}"
        f"{'Shift':>8}"
    )

    print("-" * 160)

    for _, row in ranking.iterrows():
        print(
            f"{int(row['Rank_Harmonised']):<6}"
            f"{row['Category']:<40}"
            f"{row['FactorLabel']:<40}"
            f"{int(row['TotalMatchedPairs']):>9}"
            f"{int(row['Datasets']):>8}"
            f"{row['MedianAbsDelta_Harmonised']:>14.5f}"
            f"{row['MedianAbsDelta_Native']:>14.5f}"
            f"{int(row['Rank_Native']):>13}"
            f"{int(row['RankShift_HarmonisedMinusNative']):>8}"
        )

    print()
    print("=" * 140)
    print("BROAD SOURCE SUMMARY — MATCHED SENSITIVITY, NOT VARIANCE PARTITIONING")
    print("=" * 140)

    print(
        f"{'Source':<48}"
        f"{'Pairs':>10}"
        f"{'Data':>8}"
        f"{'Mean |D| H':>16}"
        f"{'Median |D| H':>18}"
        f"{'Mean |D| N':>16}"
    )

    print("-" * 125)

    for _, row in category_summary.iterrows():
        print(
            f"{row['Category']:<48}"
            f"{int(row['TotalMatchedPairs']):>10}"
            f"{int(row['Datasets']):>8}"
            f"{row['MeanAbsDelta_Harmonised']:>16.5f}"
            f"{row['MedianAbsDelta_Harmonised']:>18.5f}"
            f"{row['MeanAbsDelta_Native']:>16.5f}"
        )

    zero_flags = summary[
        summary["AllZero_Harmonised"] == True
    ]

    print()
    print("=" * 140)
    print("UPSTREAM ZERO / IDENTICAL-OUTPUT AUDIT FLAGS")
    print("=" * 140)

    if zero_flags.empty:
        print(
            "No harmonised contrast was exactly zero "
            "across every represented inferential unit."
        )
    else:
        for _, row in zero_flags.iterrows():
            print(
                " -",
                row["FactorLabel"],
                "|",
                row["Contrast"],
                "| N=",
                int(row["IndependentUnits"]),
            )

    print()
    print("=" * 140)
    print("MISSING-H2 POLICY")
    print("=" * 140)

    if excluded.empty:
        print("No h2 rows were excluded.")
    else:
        reason_counts = (
            excluded["Reason"]
            .value_counts(dropna=False)
        )

        print(reason_counts.to_string())

        print()
        print(
            "These rows were excluded. "
            "No binary row was allowed to fall back to observed/native h2."
        )


# =============================================================================
# 21. SAVE OUTPUTS
# =============================================================================

def save_outputs(
    root,
    mode,
    row_audit,
    excluded,
    factor_audit,
    pairs_fold,
    pair_units,
    independent_effects,
    summary,
    ranking,
    category_summary,
    scale_comparison,
):
    outdir = root / "FinalAnalysis04"
    outdir.mkdir(parents=True, exist_ok=True)

    prefix = outdir / f"FinalAnalysis04_{mode}"

    outputs = {
        "RowAudit": row_audit,
        "ExcludedRows": excluded,
        "FactorAudit": factor_audit,
        "FoldPairs": pairs_fold,
        "PairUnits": pair_units,
        "IndependentUnitEffects": independent_effects,
        "InferentialSummary": summary,
        "FactorRanking": ranking,
        "CategorySummary": category_summary,
        "ScaleComparison": scale_comparison,
    }

    for name, df in outputs.items():
        path = Path(f"{prefix}_{name}.csv")
        df.to_csv(path, index=False)

    xlsx_path = Path(f"{prefix}.xlsx")

    try:
        with pd.ExcelWriter(
            xlsx_path,
            engine="openpyxl",
        ) as writer:

            for name, df in outputs.items():
                # Excel sheet names must be <=31 chars.
                sheet = name[:31]
                df.to_excel(
                    writer,
                    sheet_name=sheet,
                    index=False,
                )

    except Exception as exc:
        print(
            "WARNING: Excel workbook was not written:",
            repr(exc),
        )

    print()
    print("=" * 140)
    print("OUTPUTS SAVED")
    print("=" * 140)
    print(outdir)

    for name in outputs:
        print(
            " -",
            f"FinalAnalysis04_{mode}_{name}.csv",
        )

    if xlsx_path.exists():
        print(
            " -",
            xlsx_path.name,
        )


# =============================================================================
# 22. POST-MATCH INPUT CONTROL CHECK
# =============================================================================

def post_match_input_control_check(pair_units):
    rows = []

    for definition in FACTOR_DEFINITIONS:
        subset = pair_units[
            pair_units["Factor"]
            == definition["factor"]
        ]

        if subset.empty:
            rows.append({
                "Factor": definition["factor"],
                "FactorLabel": definition["label"],
                "MatchedPairs": 0,
                "InputControlStatus": "no_valid_comparisons",
            })
            continue

        status_parts = []

        if definition["require_same_genotype"]:
            known = (
                subset["InputGenotypeSNPs_A"].notna()
                & subset["InputGenotypeSNPs_B"].notna()
            )

            if known.any():
                same = np.isclose(
                    subset.loc[
                        known,
                        "InputGenotypeSNPs_A",
                    ],
                    subset.loc[
                        known,
                        "InputGenotypeSNPs_B",
                    ],
                    rtol=0,
                    atol=0,
                )

                status_parts.append(
                    "same_actual_input_SNP_count="
                    f"{int(same.sum())}/{int(known.sum())}"
                )
            else:
                status_parts.append(
                    "genotype_fingerprint_controlled_count_NA"
                )
        else:
            status_parts.append(
                "genotype_input_allowed_to_differ_by_design"
            )

        if definition["require_same_gwas"]:
            status_parts.append(
                "GWAS_fingerprint_controlled"
            )

        rows.append({
            "Factor": definition["factor"],
            "FactorLabel": definition["label"],
            "MatchedPairs": len(subset),
            "InputControlStatus": "; ".join(status_parts),
        })

    return pd.DataFrame(rows)


# =============================================================================
# 23. RUN ONE MODE
# =============================================================================

def run_mode(mode):
    root = REAL_ROOT if mode == "real" else SIM_ROOT

    if not root.exists():
        raise SystemExit(
            f"ERROR: root does not exist: {root}"
        )

    data, excluded, row_audit = load_mode_data(
        mode=mode,
        root=root,
    )

    print_input_audit(data)

    pairs_fold, factor_audit = build_all_pairs(
        data=data,
    )

    if pairs_fold.empty:
        print()
        print(
            "NO valid exact one-factor comparisons were found."
        )

        outdir = root / "FinalAnalysis04"
        outdir.mkdir(parents=True, exist_ok=True)

        row_audit.to_csv(
            outdir / f"FinalAnalysis04_{mode}_RowAudit.csv",
            index=False,
        )

        excluded.to_csv(
            outdir / f"FinalAnalysis04_{mode}_ExcludedRows.csv",
            index=False,
        )

        factor_audit.to_csv(
            outdir / f"FinalAnalysis04_{mode}_FactorAudit.csv",
            index=False,
        )

        return

    pair_units = average_folds(
        pairs_fold=pairs_fold,
    )

    input_control = post_match_input_control_check(
        pair_units=pair_units,
    )

    # Append post-match control status to factor audit.
    factor_audit = factor_audit.merge(
        input_control,
        on=[
            "Factor",
            "FactorLabel",
        ],
        how="left",
    )

    independent_effects = build_independent_unit_effects(
        mode=mode,
        pair_units=pair_units,
    )

    summary = build_inferential_summary(
        independent_effects=independent_effects,
    )

    ranking = build_factor_ranking(
        mode=mode,
        pair_units=pair_units,
    )

    category_summary = build_category_summary(
        pair_units=pair_units,
    )

    scale_comparison = build_scale_comparison(
        summary=summary,
        ranking=ranking,
    )

    print_report(
        mode=mode,
        data=data,
        excluded=excluded,
        factor_audit=factor_audit,
        pairs_fold=pairs_fold,
        pair_units=pair_units,
        independent_effects=independent_effects,
        summary=summary,
        ranking=ranking,
        category_summary=category_summary,
    )

    save_outputs(
        root=root,
        mode=mode,
        row_audit=row_audit,
        excluded=excluded,
        factor_audit=factor_audit,
        pairs_fold=pairs_fold,
        pair_units=pair_units,
        independent_effects=independent_effects,
        summary=summary,
        ranking=ranking,
        category_summary=category_summary,
        scale_comparison=scale_comparison,
    )

    print()
    print("=" * 140)
    print("INTERPRETATION NOTES")
    print("=" * 140)
    print(
        """
1. Primary revised-manuscript h2:
      binary    -> h2_liability_primary
      continuous -> h2_native

2. The script reads Results_Liability.csv for h2 values.
   It never reads Results.csv as the h2 analysis source.

3. Missing binary liability values are EXCLUDED.
   They are never replaced by native/observed h2.

4. The native comparison is calculated on the same harmonised-eligible
   matched rows, so differences reflect scale harmonisation rather than
   different row sets.

5. HE vs REML, GRM, covariate, PCA, LDAK-model and GEMMA-vs-DPR
   comparisons retain actual-input .bim fingerprint controls whenever
   genotype identity must be held fixed.

6. Clumping/pruning and SNP-set comparisons intentionally allow the
   genotype-input fingerprint to differ because the SNP set is the
   intervention being tested.

7. Overlapping CV folds are averaged before inference.

8. REAL mode:
      phenotype is the independent inferential unit.

9. SIMULATED mode:
      N=750 and N=5000 are nested versions of the same underlying trait.
      Their effects are averaged within UnderlyingTrait before Wilcoxon
      inference. Scenario-level outputs remain descriptive.

10. All valid one-factor contrasts form one BH-FDR family separately
    for the harmonised and native comparison analyses.

11. The broad three-category summary is a matched sensitivity summary,
    not formal variance partitioning.

12. Exactly-zero contrasts are audit flags, not automatic biological
    evidence of no effect.
"""
    )

    print("=" * 140)
    print("FINAL ANALYSIS 04 COMPLETE:", mode.upper())
    print("=" * 140)


# =============================================================================
# 24. COMMAND LINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Harmonised exact one-factor matched heritability analysis."
        )
    )

    parser.add_argument(
        "mode",
        choices=[
            "real",
            "simulated",
            "both",
        ],
        help=(
            "Run real data, simulated data, or both."
        ),
    )

    args = parser.parse_args()

    if args.mode == "both":
        run_mode("real")
        run_mode("simulated")
    else:
        run_mode(args.mode)


if __name__ == "__main__":
    main()
