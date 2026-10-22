#!/usr/bin/env python3
"""
FinalAnalysis01-LiabilityScale.py

Usage
-----
python FinalAnalysis01-LiabilityScale.py real
python FinalAnalysis01-LiabilityScale.py simulated

What it does
------------
1. Uses the exact phenotype/Fold_x/Method/Results.csv layout used by the benchmark.
2. Preserves the original/native h2 and SE.
3. For binary traits, converts fold-level h2 to liability scale.
4. Writes Results_Liability.csv beside each Results.csv.
5. Builds:
      FinalResults_Liability_Long.csv
      FinalResults_Liability_Summary.csv
      Liability_Audit.csv
6. Never overwrites the original Results.csv.

IMPORTANT
---------
For individual-level methods, the scientifically preferred P is the EXACT
case fraction in the training fold used for that estimate.

The script therefore looks, in this order, for:
    a) case/control columns in Results.csv
    b) an optional metadata file
    c) overall real-data target prevalence as a FALLBACK only

For summary-statistic methods, P is the discovery GWAS case fraction unless
the summary-statistic N is explicitly marked as effective-N, in which case
P=0.5.

For High Cholesterol summary-statistic methods, conversion is left as NA
until exact GWAS case/control counts (or a justified effective-N convention)
are supplied.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm


# =============================================================================
# 1. PATHS
# =============================================================================

# Change these defaults if needed, or set environment variables:
#   export HERITABILITY_REAL_ROOT=/your/path
#   export HERITABILITY_SIM_ROOT=/your/path
#
# Example:
#   HERITABILITY_REAL_ROOT=/data/ascher01/uqmmune1/finalheritability \
#       python FinalAnalysis01-LiabilityScale.py real

REAL_ROOT = Path(
    os.environ.get(
        "HERITABILITY_REAL_ROOT",
        "/data/ascher01/uqmmune1/finalheritability",
    )
)

SIM_ROOT = Path(
    os.environ.get(
        "HERITABILITY_SIM_ROOT",
        "/data/ascher01/uqmmune1/finalheritability",
    )
)


# Fast parallel processing. Override with:
#   LIABILITY_WORKERS=16 python FinalAnalysis01-LiabilityScale.py real
MAX_WORKERS = max(
    1,
    int(
        os.environ.get(
            "LIABILITY_WORKERS",
            str(min(8, os.cpu_count() or 1)),
        )
    ),
)


# =============================================================================
# 2. REAL-PHENOTYPE METADATA
# =============================================================================
#
# K = population/source-appropriate prevalence used for liability conversion.
# P_target_overall = only a fallback when exact fold counts cannot be found.
# P_GWAS = discovery GWAS case fraction.
#
# High cholesterol GWAS case/control counts remain unknown -> summary-method
# liability conversion is deliberately not fabricated.

REAL_META = {
    "asthma": {
        "aliases": ["asthma"],
        "trait_type": "binary",
        "K_primary": 0.156,
        "K_low": 0.143,
        "K_high": 0.169,
        "target_cases": 115,
        "target_controls": 618,
        "gwas_cases": 38369,
        "gwas_controls": 411131,
    },
    "body_mass_index_bmi": {
        "aliases": [
            "body_mass_index_bmi",
            "body_mass_index",
            "bmi",
        ],
        "trait_type": "continuous",
        "K_primary": np.nan,
        "K_low": np.nan,
        "K_high": np.nan,
        "target_cases": np.nan,
        "target_controls": np.nan,
        "gwas_cases": np.nan,
        "gwas_controls": np.nan,
    },
    "depression": {
        "aliases": ["depression"],
        "trait_type": "binary",
        "K_primary": 0.3527,
        "K_low": 0.3527,
        "K_high": 0.3527,
        "target_cases": 114,
        "target_controls": 619,
        "gwas_cases": 113769,
        "gwas_controls": 208811,
    },
    "gastro_oesophageal_reflux_gord_gastric_reflux": {
        "aliases": [
            "gastro_oesophageal_reflux_gord_gastric_reflux",
            "gastro_oesophageal_reflux",
            "gastric_reflux",
            "gord",
            "gerd",
        ],
        "trait_type": "binary",
        "K_primary": 0.1453,
        "K_low": 0.0916,
        "K_high": 0.2085,
        "target_cases": 45,
        "target_controls": 688,
        "gwas_cases": 32957,
        "gwas_controls": 434296,
    },
    "high_cholesterol": {
        "aliases": [
            "high_cholesterol",
            "highcholesterol",
            "cholesterol",
        ],
        "trait_type": "binary",
        "K_primary": 0.135,
        "K_low": 0.135,
        "K_high": 0.235,
        "target_cases": 82,
        "target_controls": 651,
        "gwas_cases": np.nan,
        "gwas_controls": np.nan,
    },
    "hypertension": {
        "aliases": ["hypertension"],
        "trait_type": "binary",
        "K_primary": 0.244,
        "K_low": 0.244,
        "K_high": 0.362,
        "target_cases": 188,
        "target_controls": 545,
        "gwas_cases": 28391,
        "gwas_controls": 28246,
    },
    "hypothyroidism_myxoedema": {
        "aliases": [
            "hypothyroidism_myxoedema",
            "hypothyroidism",
            "myxoedema",
        ],
        "trait_type": "binary",
        "K_primary": 0.036,
        "K_low": 0.032,
        "K_high": 0.045,
        "target_cases": 70,
        "target_controls": 663,
        "gwas_cases": 30155,
        "gwas_controls": 379986,
    },
    "irritable_bowel_syndrome": {
        "aliases": [
            "irritable_bowel_syndrome",
            "ibs",
        ],
        "trait_type": "binary",
        "K_primary": 0.088,
        "K_low": 0.043,
        "K_high": 0.145,
        "target_cases": 78,
        "target_controls": 655,
        "gwas_cases": 53400,
        "gwas_controls": 433201,
    },
    "migraine": {
        "aliases": ["migraine"],
        "trait_type": "binary",
        "K_primary": 0.104,
        "K_low": 0.076,
        "K_high": 0.183,
        "target_cases": 53,
        "target_controls": 680,
        "gwas_cases": 1488,
        "gwas_controls": 454860,
    },
    "osteoarthritis": {
        "aliases": ["osteoarthritis"],
        "trait_type": "binary",
        "K_primary": 0.107,
        "K_low": 0.107,
        "K_high": 0.108,
        "target_cases": 94,
        "target_controls": 639,
        "gwas_cases": 54039,
        "gwas_controls": 239780,
    },
}



# =============================================================================
# 3. EXACT BENCHMARK DIRECTORY LAYOUT
# =============================================================================
#
# Matches the existing analysis scripts:
#   <phenotype>/Fold_<0..4>/<method>/Results.csv
#
# REAL DATA:
# 10 phenotypes × 5 folds × 10 methods = exactly 500 expected Results.csv files.
# The 86 configurations are rows inside these files; they are not 860 files.

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

METHOD_DIRS = [
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

FOLDS = [0, 1, 2, 3, 4]


def expected_real_result_paths(root: Path) -> list[Path]:
    """Return the exact 500 real-data benchmark result locations."""
    return [
        root / phenotype / f"Fold_{fold}" / method / "Results.csv"
        for phenotype in REAL_PHENOTYPES
        for fold in FOLDS
        for method in METHOD_DIRS
    ]


def is_sim_id(name: str) -> bool:
    return re.fullmatch(r"Sim\d{3}", str(name)) is not None


def simulation_directories(root: Path) -> list[Path]:
    """
    Return ONLY the 200 production simulation directories.

    Supported layouts:
      root/Sim001 ... root/Sim200
      root/simulation/phenotypes/Sim001 ... Sim200
      root/phenotypes/Sim001 ... Sim200
    """
    candidates = [
        root,
        root / "simulation" / "phenotypes",
        root / "phenotypes",
    ]

    found = {}
    for base in candidates:
        if not base.is_dir():
            continue
        for p in base.iterdir():
            if p.is_dir() and is_sim_id(p.name):
                found[p.name] = p

    return [found[k] for k in sorted(found)]


def discover_simulated_result_paths(root: Path) -> list[Path]:
    """
    Deterministic discovery for the production benchmark only.

    Expected maximum:
        200 simulations x 5 folds x 10 methods = 10,000 Results.csv files.

    SampleData1 is deliberately excluded.
    """
    paths = []

    for sim_dir in simulation_directories(root):
        for fold in FOLDS:
            for method in METHOD_DIRS:
                p = sim_dir / f"Fold_{fold}" / method / "Results.csv"
                if p.exists():
                    paths.append(p)

    return paths


# =============================================================================
# 4. METHOD CLASSIFICATION
# =============================================================================
#
# "target"  -> individual-level phenotype generated the h2 signal
# "gwas"    -> external/discovery summary statistics generated the h2 signal
#
# ORDER MATTERS: more specific patterns should occur before generic LDSC/GEMMA.

METHOD_RULES = [
    (r"dpr.*gemma|dpr[-_ ]?gemma", "target"),
    (r"gcta", "target"),

    # GEMMA individual-level / Method 3
    (r"gemma1|he\s*regression|gemma.*genotype", "target"),

    # Summary-statistic GEMMA / LDSC combinations
    (r"gemma[-_ ]?ldsc|ldsc[-_ ]?gemma", "gwas"),
    (r"gemma2|gemma.*gwas|gemma.*summary", "gwas"),

    # LDpred2 heritability / internal LDSC
    (r"ldpred", "gwas"),

    # LDAK / SumHer
    (r"ldak|sumher", "gwas"),

    # Other LDSC
    (r"ldsc", "gwas"),
]


# =============================================================================
# 4. COLUMN ALIASES
# =============================================================================

H2_ALIASES = [
    "h2",
    "h2_est",
    "h2_estimate",
    "heritability",
    "heritability_value",
    "heritability_estimate",
    "h2_obs",
    "h2_observed",
]

SE_ALIASES = [
    "h2_se",
    "se",
    "heritability_se",
    "h2_standard_error",
    "standard_error",
]

METHOD_ALIASES = [
    "Method",
    "method",
    "method_name",
    "tool",
]

MODEL_ALIASES = [
    "h2model",
    "model",
    "configuration",
    "config",
]

FOLD_ALIASES = [
    "fold",
    "Fold",
    "fold_id",
    "cv_fold",
]

VARIANT_ALIASES = [
    "variants",
    "n_variants",
    "number_of_variants",
    "num_variants",
    "snps",
    "n_snps",
]

TARGET_CASE_ALIASES = [
    "target_cases",
    "train_cases",
    "training_cases",
    "n_cases_train",
    "n_case_train",
    "n_cases",
    "cases",
]

TARGET_CONTROL_ALIASES = [
    "target_controls",
    "train_controls",
    "training_controls",
    "n_controls_train",
    "n_control_train",
    "n_controls",
    "controls",
]

GWAS_CASE_ALIASES = [
    "gwas_cases",
    "discovery_cases",
    "summary_cases",
    "n_gwas_cases",
]

GWAS_CONTROL_ALIASES = [
    "gwas_controls",
    "discovery_controls",
    "summary_controls",
    "n_gwas_controls",
]

N_TYPE_ALIASES = [
    "N_type",
    "n_type",
    "sample_size_type",
    "gwas_n_type",
]


# =============================================================================
# 5. HELPERS
# =============================================================================

def normalize_text(x: object) -> str:
    s = str(x).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def first_existing_column(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    exact = {str(c): c for c in df.columns}
    lower = {str(c).lower(): c for c in df.columns}

    for a in aliases:
        if a in exact:
            return exact[a]
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def safe_float(x) -> float:
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def fraction(cases, controls) -> float:
    cases = safe_float(cases)
    controls = safe_float(controls)
    if not np.isfinite(cases) or not np.isfinite(controls):
        return np.nan
    n = cases + controls
    if n <= 0:
        return np.nan
    return cases / n


def infer_fold(path: Path, row: pd.Series, fold_col: Optional[str]) -> object:
    if fold_col is not None:
        value = row.get(fold_col, np.nan)
        if not pd.isna(value):
            return value

    for part in reversed(path.parts):
        m = re.search(r"fold[_\- ]?(\d+)", part, flags=re.I)
        if m:
            return int(m.group(1))
    return np.nan


def infer_method(path: Path, row: pd.Series, method_col: Optional[str]) -> str:
    if method_col is not None:
        val = row.get(method_col, "")
        if str(val).strip() and str(val).lower() != "nan":
            return str(val).strip()

    # Usually the method name is present somewhere in the parent path.
    for part in reversed(path.parts):
        low = part.lower()
        if any(re.search(pattern, low, flags=re.I) for pattern, _ in METHOD_RULES):
            return part

    return path.parent.name


def classify_method(method: str, path: Path) -> str:
    text = f"{method} {' '.join(path.parts[-6:])}".lower()
    for pattern, source in METHOD_RULES:
        if re.search(pattern, text, flags=re.I):
            return source
    return "unknown"


def identify_real_phenotype(path: Path, row: pd.Series) -> Optional[str]:
    """
    Identify real phenotype safely.

    The benchmark path is:
        <root>/<phenotype>/Fold_x/<method>/Results.csv

    Therefore an exact path component matching REAL_META is authoritative.
    This is safer than guessing from arbitrary row text.
    """
    # 1) Exact directory-name match: authoritative for real benchmark data.
    for part in path.parts:
        if part in REAL_META:
            return part

    # 2) Fall back to likely phenotype columns.
    for col in ["phenotype", "Phenotype", "trait", "Trait", "disease"]:
        if col in row.index:
            value = normalize_text(row[col])
            if value and value != "nan":
                for canonical, meta in REAL_META.items():
                    for alias in meta["aliases"]:
                        if normalize_text(alias) in value or value in normalize_text(alias):
                            return canonical

    # Then inspect path.
    text = normalize_text("_".join(path.parts))

    # Longer aliases first prevents "bmi" etc. from matching badly.
    candidates = []
    for canonical, meta in REAL_META.items():
        for alias in meta["aliases"]:
            candidates.append((len(normalize_text(alias)), canonical, normalize_text(alias)))

    for _, canonical, alias in sorted(candidates, reverse=True):
        if alias and alias in text:
            return canonical

    return None


def extract_simulation_id(path: Path, row: Optional[pd.Series] = None) -> Optional[str]:
    """Get SimXXX from the path first, then from a row if necessary."""
    for part in path.parts:
        if is_sim_id(part):
            return part

    if row is not None:
        for col in ["simulation_id", "phenotype", "Phenotype", "scenario", "trait"]:
            if col in row.index:
                value = str(row[col]).strip()
                if is_sim_id(value):
                    return value

    return None


def load_simulation_manifest(root: Path) -> Optional[pd.DataFrame]:
    """Load the authoritative 200-row simulation manifest."""
    candidates = [
        root / "simulation_manifest.csv",
        root / "simulation" / "simulation_manifest.csv",
    ]

    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            if "simulation_id" in df.columns:
                df["simulation_id"] = df["simulation_id"].astype(str).str.strip()
                return df

    return None


def simulation_manifest_row(
    metadata: Optional[pd.DataFrame],
    sim_id: str,
) -> Optional[pd.Series]:
    if metadata is None or metadata.empty or "simulation_id" not in metadata.columns:
        return None

    sub = metadata.loc[
        metadata["simulation_id"].astype(str).str.strip().eq(sim_id)
    ]
    return sub.iloc[0] if not sub.empty else None


def infer_simulation_trait(
    path: Path,
    row: pd.Series,
    metadata: Optional[pd.DataFrame] = None,
) -> tuple[str, str]:
    """
    Use simulation_manifest.csv as the authority.

    Fallback ordering from the production design:
      Sim001-Sim100 = continuous
      Sim101-Sim200 = binary
    """
    sim_id = extract_simulation_id(path, row)
    if sim_id is None:
        return "unknown", "unknown"

    meta = simulation_manifest_row(metadata, sim_id)
    if meta is not None and "phenotype_type" in meta.index:
        trait_type = normalize_text(meta["phenotype_type"])
        if trait_type in {"binary", "continuous"}:
            return sim_id, trait_type

    # Safe fallback to the fixed design ordering.
    n = int(sim_id[3:])
    if 1 <= n <= 100:
        return sim_id, "continuous"
    if 101 <= n <= 200:
        return sim_id, "binary"

    return sim_id, "unknown"


def find_sim_dir(results_path: Path) -> Optional[Path]:
    for parent in [results_path.parent, *results_path.parents]:
        if is_sim_id(parent.name):
            return parent
    return None


@lru_cache(maxsize=4096)
def _target_fold_prevalence(sim_dir_str: str, fold: int) -> float:
    """
    Exact binary case fraction in the training fold.

    PLINK phenotype coding is expected to be 1=control, 2=case.
    Also supports 0/1 if encountered.
    """
    p = Path(sim_dir_str) / f"Fold_{int(fold)}" / "train_data.fam"
    if not p.exists():
        return np.nan

    try:
        fam = pd.read_csv(p, sep=r"\s+", header=None)
        if fam.shape[1] < 6:
            return np.nan

        y = pd.to_numeric(fam.iloc[:, 5], errors="coerce").dropna()
        y = y[~y.isin([-9])]
        if y.empty:
            return np.nan

        values = set(y.astype(float).unique().tolist())
        if values.issubset({1.0, 2.0}):
            return float((y == 2).mean())
        if values.issubset({0.0, 1.0}):
            return float(y.mean())
    except Exception:
        return np.nan

    return np.nan


@lru_cache(maxsize=512)
def _discovery_prevalence(sim_dir_str: str) -> float:
    """
    Exact observed case fraction in the independent discovery GWAS cohort.
    """
    p = Path(sim_dir_str) / "discovery_phenotype.tsv"
    if not p.exists():
        return np.nan

    try:
        df = pd.read_csv(p, sep="\t")
        for col in ["phenotype_internal", "phenotype_pipeline", "phenotype"]:
            if col not in df.columns:
                continue

            y = pd.to_numeric(df[col], errors="coerce").dropna()
            if y.empty:
                continue

            values = set(y.astype(float).unique().tolist())
            if values.issubset({0.0, 1.0}):
                return float(y.mean())
            if values.issubset({1.0, 2.0}):
                return float((y == 2).mean())
    except Exception:
        return np.nan

    return np.nan



@lru_cache(maxsize=4096)
def _prevalence_from_fam_file(fam_path_str: str) -> float:
    """Return exact binary case fraction from a PLINK .fam phenotype column."""
    p = Path(fam_path_str)
    if not p.exists():
        return np.nan

    try:
        fam = pd.read_csv(p, sep=r"\s+", header=None)
        if fam.shape[1] < 6:
            return np.nan

        y = pd.to_numeric(fam.iloc[:, 5], errors="coerce").dropna()
        y = y[~y.isin([-9])]
        if y.empty:
            return np.nan

        vals = set(y.astype(float).unique().tolist())

        if vals.issubset({1.0, 2.0}):
            return float((y == 2).mean())

        if vals.issubset({0.0, 1.0}):
            return float(y.mean())

    except Exception:
        return np.nan

    return np.nan


@lru_cache(maxsize=1024)
def _real_target_fold_prevalence(
    root_str: str,
    phenotype: str,
    fold: int,
) -> float:
    """
    Exact training-fold prevalence for empirical target-based methods.

    The benchmark stores training files under:
        <root>/<phenotype>/Fold_<fold>/

    We prefer train_data.fam, then train_data.QC.fam, then any train*.fam.
    """
    fold_dir = Path(root_str) / phenotype / f"Fold_{int(fold)}"

    preferred = [
        fold_dir / "train_data.fam",
        fold_dir / "train_data.QC.fam",
    ]

    for p in preferred:
        value = _prevalence_from_fam_file(str(p))
        if np.isfinite(value):
            return value

    try:
        candidates = sorted(
            fold_dir.glob("train*.fam"),
            key=lambda x: (len(x.name), x.name),
        )
    except Exception:
        candidates = []

    for p in candidates:
        value = _prevalence_from_fam_file(str(p))
        if np.isfinite(value):
            return value

    return np.nan

def liability_factor(K: float, P: float) -> float:
    """
    C(K,P) = K^2(1-K)^2 / [P(1-P) z^2]
    where threshold t = Phi^-1(1-K), z = phi(t).
    """
    K = safe_float(K)
    P = safe_float(P)

    if not (np.isfinite(K) and np.isfinite(P)):
        return np.nan

    if not (0 < K < 1 and 0 < P < 1):
        return np.nan

    t = norm.ppf(1.0 - K)
    z = norm.pdf(t)

    if z <= 0:
        return np.nan

    return (K**2 * (1.0 - K)**2) / (P * (1.0 - P) * z**2)


def load_optional_metadata(root: Path, mode: str) -> Optional[pd.DataFrame]:
    if mode == "simulated":
        manifest = load_simulation_manifest(root)
        if manifest is not None:
            return manifest

    candidate = root / f"liability_metadata_{mode}.csv"
    if candidate.exists():
        return pd.read_csv(candidate)

    return None


def metadata_lookup(
    metadata: Optional[pd.DataFrame],
    phenotype: str,
    fold: object,
) -> Optional[pd.Series]:
    if metadata is None or metadata.empty:
        return None

    df = metadata.copy()

    pheno_col = first_existing_column(df, ["simulation_id", "phenotype", "Phenotype", "trait"])
    fold_col = first_existing_column(df, FOLD_ALIASES)

    if pheno_col is not None:
        target = normalize_text(phenotype)
        mask = df[pheno_col].astype(str).map(normalize_text) == target
        sub = df.loc[mask].copy()
    else:
        sub = df.copy()

    if sub.empty:
        return None

    if fold_col is not None and not pd.isna(fold):
        # String comparison is more tolerant of 0 vs 0.0.
        fold_norm = str(fold).replace(".0", "")
        fmask = (
            sub[fold_col]
            .astype(str)
            .str.replace(".0", "", regex=False)
            == fold_norm
        )
        folded = sub.loc[fmask]
        if not folded.empty:
            return folded.iloc[0]

    # Fall back to phenotype-level metadata row.
    return sub.iloc[0]


def row_value_from_aliases(row: pd.Series, aliases: list[str]) -> float:
    lower_map = {str(c).lower(): c for c in row.index}
    for alias in aliases:
        c = lower_map.get(alias.lower())
        if c is not None:
            value = safe_float(row[c])
            if np.isfinite(value):
                return value
    return np.nan


def text_value_from_aliases(row: pd.Series, aliases: list[str]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in row.index}
    for alias in aliases:
        c = lower_map.get(alias.lower())
        if c is not None:
            value = str(row[c]).strip()
            if value and value.lower() != "nan":
                return value
    return None


def get_P(
    *,
    mode: str,
    method_source: str,
    phenotype: str,
    trait_type: str,
    row: pd.Series,
    meta_row: Optional[pd.Series],
    real_meta: Optional[dict],
) -> tuple[float, str, str]:
    """
    Returns:
        P, P_source, status_note
    """

    if trait_type != "binary":
        return np.nan, "not_applicable", "continuous_trait"

    # ---------------------------------------------------------------------
    # TARGET / INDIVIDUAL-LEVEL METHOD
    # ---------------------------------------------------------------------
    if method_source == "target":
        # 1) Prefer exact counts in Results.csv
        cases = row_value_from_aliases(row, TARGET_CASE_ALIASES)
        controls = row_value_from_aliases(row, TARGET_CONTROL_ALIASES)
        P = fraction(cases, controls)
        if np.isfinite(P):
            return P, "target_training_fold_from_results", "exact_or_row_level"

        # 2) Prefer exact fold metadata if supplied
        if meta_row is not None:
            cases = row_value_from_aliases(meta_row, TARGET_CASE_ALIASES)
            controls = row_value_from_aliases(meta_row, TARGET_CONTROL_ALIASES)
            P = fraction(cases, controls)
            if np.isfinite(P):
                return P, "target_training_fold_from_metadata", "exact_or_metadata"

        # 3) Real-data fallback only: overall cohort case fraction.
        #    This keeps the pipeline running but is explicitly flagged.
        if mode == "real" and real_meta is not None:
            P = fraction(
                real_meta.get("target_cases"),
                real_meta.get("target_controls"),
            )
            if np.isfinite(P):
                return (
                    P,
                    "target_overall_FALLBACK",
                    "WARNING_exact_fold_prevalence_not_found",
                )

        return np.nan, "missing_target_prevalence", "cannot_convert"

    # ---------------------------------------------------------------------
    # GWAS / SUMMARY-STATISTIC METHOD
    # ---------------------------------------------------------------------
    if method_source == "gwas":
        # If N is explicitly labelled effective-N, use P = 0.5.
        n_type = text_value_from_aliases(row, N_TYPE_ALIASES)
        if n_type is None and meta_row is not None:
            n_type = text_value_from_aliases(meta_row, N_TYPE_ALIASES)

        if n_type:
            n_norm = normalize_text(n_type)
            if (
                "effective" in n_norm
                or n_norm in {"neff", "n_eff", "effective_n"}
            ):
                return 0.5, "GWAS_effective_N_convention", "P_set_to_0.5"

        # 1) Prefer GWAS counts in Results.csv
        cases = row_value_from_aliases(row, GWAS_CASE_ALIASES)
        controls = row_value_from_aliases(row, GWAS_CONTROL_ALIASES)
        P = fraction(cases, controls)
        if np.isfinite(P):
            return P, "discovery_GWAS_from_results", "row_level"

        # 2) Optional metadata
        if meta_row is not None:
            cases = row_value_from_aliases(meta_row, GWAS_CASE_ALIASES)
            controls = row_value_from_aliases(meta_row, GWAS_CONTROL_ALIASES)
            P = fraction(cases, controls)
            if np.isfinite(P):
                return P, "discovery_GWAS_from_metadata", "metadata"

        # 3) Real-data hard-coded accession metadata
        if mode == "real" and real_meta is not None:
            P = fraction(
                real_meta.get("gwas_cases"),
                real_meta.get("gwas_controls"),
            )
            if np.isfinite(P):
                return P, "discovery_GWAS", "real_metadata"

        return np.nan, "missing_GWAS_prevalence", "cannot_convert"

    return np.nan, "unknown_method_source", "cannot_convert"


def get_K(
    *,
    mode: str,
    trait_type: str,
    row: pd.Series,
    meta_row: Optional[pd.Series],
    real_meta: Optional[dict],
) -> tuple[float, float, float]:
    if trait_type != "binary":
        return np.nan, np.nan, np.nan

    # Optional row-level K values
    def from_any(source: pd.Series, aliases: list[str]) -> float:
        return row_value_from_aliases(source, aliases)

    # Results.csv first
    kp = from_any(row, ["K_primary", "K", "population_prevalence"])
    kl = from_any(row, ["K_low"])
    kh = from_any(row, ["K_high"])

    # Optional metadata second
    if meta_row is not None:
        if not np.isfinite(kp):
            kp = from_any(meta_row, ["K_primary", "K", "population_prevalence"])
        if not np.isfinite(kl):
            kl = from_any(meta_row, ["K_low"])
        if not np.isfinite(kh):
            kh = from_any(meta_row, ["K_high"])

    if mode == "real" and real_meta is not None:
        if not np.isfinite(kp):
            kp = safe_float(real_meta.get("K_primary"))
        if not np.isfinite(kl):
            kl = safe_float(real_meta.get("K_low"))
        if not np.isfinite(kh):
            kh = safe_float(real_meta.get("K_high"))

    if mode == "simulated":
        # The current binary simulation generating prevalence is K=0.10.
        if not np.isfinite(kp):
            kp = 0.10
        if not np.isfinite(kl):
            kl = kp
        if not np.isfinite(kh):
            kh = kp

    return kp, kl, kh


def detect_h2_column(df: pd.DataFrame) -> Optional[str]:
    return first_existing_column(df, H2_ALIASES)


def detect_se_column(df: pd.DataFrame) -> Optional[str]:
    return first_existing_column(df, SE_ALIASES)


def process_results_file(
    results_path: Path,
    root: Path,
    mode: str,
    metadata: Optional[pd.DataFrame],
) -> pd.DataFrame:
    df = pd.read_csv(results_path)

    if df.empty:
        return pd.DataFrame()

    h2_col = detect_h2_column(df)
    if h2_col is None:
        return pd.DataFrame()

    se_col = detect_se_column(df)
    method_col = first_existing_column(df, METHOD_ALIASES)
    model_col = first_existing_column(df, MODEL_ALIASES)
    fold_col = first_existing_column(df, FOLD_ALIASES)
    variant_col = first_existing_column(df, VARIANT_ALIASES)

    output_rows = []

    for idx, row in df.iterrows():
        fold = infer_fold(results_path, row, fold_col)
        method = infer_method(results_path, row, method_col)
        method_source = classify_method(method, results_path)

        # Extra guard: simulated mode must never overwrite real phenotype outputs.
        if mode == "simulated":
            try:
                rel_parts = results_path.relative_to(root).parts
            except ValueError:
                rel_parts = results_path.parts

            if any(part in REAL_PHENOTYPES for part in rel_parts):
                raise RuntimeError(
                    f"SAFETY STOP: simulated mode attempted to process "
                    f"a real phenotype file: {results_path}"
                )

        if mode == "real":
            phenotype = identify_real_phenotype(results_path, row)
            if phenotype is None:
                phenotype = "unknown"
                trait_type = "unknown"
                real_meta = None
            else:
                real_meta = REAL_META[phenotype]
                trait_type = real_meta["trait_type"]

        else:
            phenotype, trait_type = infer_simulation_trait(
                results_path,
                row,
                metadata=metadata,
            )
            real_meta = None

        meta_row = metadata_lookup(metadata, phenotype, fold)

        K_primary, K_low, K_high = get_K(
            mode=mode,
            trait_type=trait_type,
            row=row,
            meta_row=meta_row,
            real_meta=real_meta,
        )

        # For simulations, the manifest is authoritative for generating K.
        if mode == "simulated" and trait_type == "binary" and meta_row is not None:
            if "population_prevalence" in meta_row.index:
                manifest_k = safe_float(meta_row["population_prevalence"])
                if np.isfinite(manifest_k):
                    K_primary = manifest_k
                    K_low = manifest_k
                    K_high = manifest_k

        P, P_source, P_note = get_P(
            mode=mode,
            method_source=method_source,
            phenotype=phenotype,
            trait_type=trait_type,
            row=row,
            meta_row=meta_row,
            real_meta=real_meta,
        )

        # EMPIRICAL TARGET METHODS:
        # Always prefer the actual training-fold phenotype composition from
        # the PLINK FAM file. This makes the implementation match the manuscript.
        if (
            mode == "real"
            and trait_type == "binary"
            and method_source == "target"
        ):
            exact_fold_p = _real_target_fold_prevalence(
                str(root),
                phenotype,
                int(fold),
            )
            if np.isfinite(exact_fold_p):
                P = exact_fold_p
                P_source = "target_training_fold_from_FAM"
                P_note = "exact_from_training_fold_FAM"

        # Exact P for binary simulations comes from the actual generated samples.
        if mode == "simulated" and trait_type == "binary" and not np.isfinite(P):
            sim_dir = find_sim_dir(results_path)

            if sim_dir is not None:
                if method_source == "target":
                    sim_p = _target_fold_prevalence(str(sim_dir), int(fold))
                    if np.isfinite(sim_p):
                        P = sim_p
                        P_source = "simulation_target_training_fold"
                        P_note = "exact_from_train_data_fam"

                elif method_source == "gwas":
                    sim_p = _discovery_prevalence(str(sim_dir))
                    if np.isfinite(sim_p):
                        P = sim_p
                        P_source = "simulation_discovery_GWAS"
                        P_note = "exact_from_discovery_phenotype"

            # Last-resort design fallback: binary cohorts were random samples from K=0.10.
            # This is explicitly flagged and should be rare because generated phenotype
            # files normally exist.
            if not np.isfinite(P) and np.isfinite(K_primary):
                P = K_primary
                P_source = "simulation_design_prevalence_FALLBACK"
                P_note = "WARNING_exact_sample_prevalence_not_found"

        h2_native = safe_float(row[h2_col])
        se_native = safe_float(row[se_col]) if se_col else np.nan

        factor_primary = liability_factor(K_primary, P)
        factor_low = liability_factor(K_low, P)
        factor_high = liability_factor(K_high, P)

        if trait_type == "continuous":
            h2_liab = np.nan
            se_liab = np.nan
            h2_liab_low = np.nan
            h2_liab_high = np.nan
            conversion_status = "not_applicable_continuous"

        elif trait_type != "binary":
            h2_liab = np.nan
            se_liab = np.nan
            h2_liab_low = np.nan
            h2_liab_high = np.nan
            conversion_status = "unknown_trait_type"

        elif not np.isfinite(P):
            h2_liab = np.nan
            se_liab = np.nan
            h2_liab_low = np.nan
            h2_liab_high = np.nan
            conversion_status = "not_converted_missing_P"

        elif not np.isfinite(K_primary):
            h2_liab = np.nan
            se_liab = np.nan
            h2_liab_low = np.nan
            h2_liab_high = np.nan
            conversion_status = "not_converted_missing_K"

        elif not np.isfinite(h2_native):
            h2_liab = np.nan
            se_liab = np.nan
            h2_liab_low = np.nan
            h2_liab_high = np.nan
            conversion_status = "not_converted_missing_h2"

        else:
            h2_liab = h2_native * factor_primary
            h2_liab_low = (
                h2_native * factor_low if np.isfinite(factor_low) else np.nan
            )
            h2_liab_high = (
                h2_native * factor_high if np.isfinite(factor_high) else np.nan
            )

            se_liab = (
                se_native * factor_primary
                if np.isfinite(se_native)
                else np.nan
            )
            conversion_status = "converted"

        out = row.to_dict()

        # Provenance
        out["source_file"] = str(results_path.relative_to(root))
        out["source_row"] = idx
        out["analysis_mode"] = mode
        out["phenotype_standardized"] = phenotype
        out["trait_type"] = trait_type
        out["fold_standardized"] = fold
        out["method_standardized"] = method
        out["method_signal_source"] = method_source

        # Preserve native scale explicitly
        out["h2_native"] = h2_native
        out["h2_se_native"] = se_native
        out["native_scale"] = (
            "quantitative"
            if trait_type == "continuous"
            else "observed"
            if trait_type == "binary"
            else "unknown"
        )

        # Conversion metadata
        out["K_primary"] = K_primary
        out["K_low"] = K_low
        out["K_high"] = K_high
        out["P_used"] = P
        out["P_source"] = P_source
        out["P_note"] = P_note

        out["liability_factor_primary"] = factor_primary
        out["liability_factor_low"] = factor_low
        out["liability_factor_high"] = factor_high

        # Liability results
        out["h2_liability_primary"] = h2_liab
        out["h2_liability_lowK"] = h2_liab_low
        out["h2_liability_highK"] = h2_liab_high
        out["h2_se_liability_primary"] = se_liab

        out["CI_lower_liability_primary"] = (
            h2_liab - 1.96 * se_liab
            if np.isfinite(h2_liab) and np.isfinite(se_liab)
            else np.nan
        )
        out["CI_upper_liability_primary"] = (
            h2_liab + 1.96 * se_liab
            if np.isfinite(h2_liab) and np.isfinite(se_liab)
            else np.nan
        )

        out["conversion_status"] = conversion_status

        if variant_col is not None:
            out["variants_standardized"] = safe_float(row[variant_col])
        else:
            out["variants_standardized"] = np.nan

        if model_col is not None:
            out["h2model_standardized"] = str(row[model_col])
        else:
            out["h2model_standardized"] = ""

        output_rows.append(out)

    out_df = pd.DataFrame(output_rows)

    # Save beside original without overwriting.
    out_path = results_path.with_name("Results_Liability.csv")
    out_df.to_csv(out_path, index=False)

    return out_df


def sem(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) <= 1:
        return np.nan
    return float(x.std(ddof=1) / math.sqrt(len(x)))


def mean_or_nan(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() == 0:
        return np.nan
    return float(x.mean())


def build_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    # Retain common configuration columns if present.
    group_cols = [
        "analysis_mode",
        "phenotype_standardized",
        "trait_type",
        "method_standardized",
        "h2model_standardized",
        "method_signal_source",
    ]

    for optional in [
        "clumpprune",
        "numberofpca",
        "number_of_pca",
        "pca",
    ]:
        if optional in long_df.columns and optional not in group_cols:
            group_cols.append(optional)

    rows = []

    for keys, sub in long_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        rec = dict(zip(group_cols, keys))

        native = pd.to_numeric(sub["h2_native"], errors="coerce")
        liab = pd.to_numeric(
            sub["h2_liability_primary"],
            errors="coerce",
        )

        native_mean = mean_or_nan(native)
        native_sem = sem(native)

        liab_mean = mean_or_nan(liab)
        liab_sem = sem(liab)

        rec["n_folds_native"] = int(native.notna().sum())
        rec["mean_h2_native"] = native_mean
        rec["fold_SE_native"] = native_sem
        rec["CI_lower_fold_native"] = (
            native_mean - 1.96 * native_sem
            if np.isfinite(native_mean) and np.isfinite(native_sem)
            else np.nan
        )
        rec["CI_upper_fold_native"] = (
            native_mean + 1.96 * native_sem
            if np.isfinite(native_mean) and np.isfinite(native_sem)
            else np.nan
        )
        rec["significant_fold_native"] = (
            (
                rec["CI_lower_fold_native"] > 0
                or rec["CI_upper_fold_native"] < 0
            )
            if np.isfinite(rec["CI_lower_fold_native"])
            and np.isfinite(rec["CI_upper_fold_native"])
            else np.nan
        )

        rec["n_folds_liability"] = int(liab.notna().sum())
        rec["mean_h2_liability"] = liab_mean
        rec["fold_SE_liability"] = liab_sem
        rec["CI_lower_fold_liability"] = (
            liab_mean - 1.96 * liab_sem
            if np.isfinite(liab_mean) and np.isfinite(liab_sem)
            else np.nan
        )
        rec["CI_upper_fold_liability"] = (
            liab_mean + 1.96 * liab_sem
            if np.isfinite(liab_mean) and np.isfinite(liab_sem)
            else np.nan
        )
        rec["significant_fold_liability"] = (
            (
                rec["CI_lower_fold_liability"] > 0
                or rec["CI_upper_fold_liability"] < 0
            )
            if np.isfinite(rec["CI_lower_fold_liability"])
            and np.isfinite(rec["CI_upper_fold_liability"])
            else np.nan
        )

        # Mean tool-reported SE if available
        native_tool_se = pd.to_numeric(
            sub["h2_se_native"],
            errors="coerce",
        )
        liab_tool_se = pd.to_numeric(
            sub["h2_se_liability_primary"],
            errors="coerce",
        )

        rec["tool_SE_mean_native"] = mean_or_nan(native_tool_se)
        rec["tool_SE_mean_liability"] = mean_or_nan(liab_tool_se)

        if np.isfinite(native_mean) and np.isfinite(rec["tool_SE_mean_native"]):
            rec["CI_lower_tool_native"] = (
                native_mean - 1.96 * rec["tool_SE_mean_native"]
            )
            rec["CI_upper_tool_native"] = (
                native_mean + 1.96 * rec["tool_SE_mean_native"]
            )
        else:
            rec["CI_lower_tool_native"] = np.nan
            rec["CI_upper_tool_native"] = np.nan

        if np.isfinite(liab_mean) and np.isfinite(rec["tool_SE_mean_liability"]):
            rec["CI_lower_tool_liability"] = (
                liab_mean - 1.96 * rec["tool_SE_mean_liability"]
            )
            rec["CI_upper_tool_liability"] = (
                liab_mean + 1.96 * rec["tool_SE_mean_liability"]
            )
        else:
            rec["CI_lower_tool_liability"] = np.nan
            rec["CI_upper_tool_liability"] = np.nan

        rec["mean_variants"] = mean_or_nan(sub["variants_standardized"])
        rec["mean_K_primary"] = mean_or_nan(sub["K_primary"])
        rec["mean_P_used"] = mean_or_nan(sub["P_used"])

        p_sources = sorted(
            set(
                sub["P_source"]
                .dropna()
                .astype(str)
                .loc[lambda s: s.str.len() > 0]
            )
        )
        rec["P_sources"] = ";".join(p_sources)

        statuses = sorted(
            set(
                sub["conversion_status"]
                .dropna()
                .astype(str)
                .loc[lambda s: s.str.len() > 0]
            )
        )
        rec["conversion_statuses"] = ";".join(statuses)

        rows.append(rec)

    return pd.DataFrame(rows)


def build_audit(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    cols = [
        "phenotype_standardized",
        "trait_type",
        "method_standardized",
        "method_signal_source",
        "fold_standardized",
        "K_primary",
        "K_low",
        "K_high",
        "P_used",
        "P_source",
        "P_note",
        "native_scale",
        "conversion_status",
        "source_file",
    ]

    cols = [c for c in cols if c in long_df.columns]

    audit = (
        long_df[cols]
        .drop_duplicates()
        .sort_values(
            [
                c
                for c in [
                    "phenotype_standardized",
                    "method_standardized",
                    "fold_standardized",
                ]
                if c in cols
            ]
        )
    )

    return audit


def print_real_prevalence_table():
    rows = []
    for phenotype, meta in REAL_META.items():
        rows.append(
            {
                "phenotype": phenotype,
                "type": meta["trait_type"],
                "K_primary": meta["K_primary"],
                "P_target_overall": fraction(
                    meta["target_cases"],
                    meta["target_controls"],
                ),
                "P_GWAS": fraction(
                    meta["gwas_cases"],
                    meta["gwas_controls"],
                ),
            }
        )

    table = pd.DataFrame(rows)
    print("\n=== REAL PHENOTYPE PREVALENCE AUDIT ===")
    print(table.to_string(index=False))


def print_method_audit(long_df: pd.DataFrame):
    if long_df.empty:
        return

    audit = (
        long_df[
            [
                "method_standardized",
                "method_signal_source",
                "P_source",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["method_standardized", "P_source"]
        )
    )

    print("\n=== METHOD / P-SOURCE AUDIT ===")
    print(audit.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Convert native binary h2 estimates to liability scale."
    )
    parser.add_argument(
        "mode",
        choices=["real", "simulated"],
        help="Choose real or simulated data.",
    )
    args = parser.parse_args()

    mode = args.mode
    root = REAL_ROOT if mode == "real" else SIM_ROOT

    if not root.exists():
        sys.exit(
            f"ERROR: Root directory does not exist: {root}\n"
            "Set HERITABILITY_REAL_ROOT / HERITABILITY_SIM_ROOT "
            "or edit the paths in the script."
        )

    metadata = load_optional_metadata(root, mode)

    # Deterministic benchmark discovery.
    if mode == "real":
        expected_files = expected_real_result_paths(root)
        result_files = [p for p in expected_files if p.exists()]
        missing_files = [p for p in expected_files if not p.exists()]
    else:
        result_files = discover_simulated_result_paths(root)
        missing_files = []

    if not result_files:
        sys.exit(f"ERROR: No benchmark Results.csv files found under: {root}")

    def _worker(results_path: Path):
        try:
            frame = process_results_file(
                results_path=results_path,
                root=root,
                mode=mode,
                metadata=metadata,
            )
            return frame, None
        except Exception as exc:
            # Keep errors for a final machine-readable file instead of
            # flooding stdout/stderr.
            return pd.DataFrame(), {
                "source_file": str(results_path),
                "error": repr(exc),
            }

    all_frames = []
    errors = []

    # Independent files can be converted in parallel.
    # executor.map preserves a very low-overhead streaming workflow.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        iterator = executor.map(_worker, result_files)
        for frame, error in tqdm(
            iterator,
            total=len(result_files),
            desc=f"Liability ({mode})",
            unit="file",
            dynamic_ncols=True,
            mininterval=0.5,
            leave=True,
        ):
            if error is not None:
                errors.append(error)
            if frame is not None and not frame.empty:
                all_frames.append(frame)

    if not all_frames:
        sys.exit("ERROR: No usable Results.csv files were processed.")

    long_df = pd.concat(all_frames, ignore_index=True)

    # HARD VALIDATION FOR REAL DATA:
    # No real benchmark row may silently have unknown trait type.
    if mode == "simulated":
        unknown = long_df[
            long_df["trait_type"].astype(str).eq("unknown")
        ]
        if not unknown.empty:
            examples = unknown["source_file"].drop_duplicates().head(10).tolist()
            raise RuntimeError(
                "Simulation trait identification failed. "
                f"{len(unknown)} rows have trait_type='unknown'. "
                f"Examples: {examples}"
            )

    if mode == "real":
        unknown = long_df[
            long_df["trait_type"].astype(str).eq("unknown")
        ]
        if not unknown.empty:
            examples = unknown["source_file"].drop_duplicates().head(10).tolist()
            raise RuntimeError(
                "Real-data phenotype identification failed. "
                f"{len(unknown)} rows have trait_type='unknown'. "
                f"Examples: {examples}"
            )

        # All non-BMI real phenotypes must have K populated.
        binary_rows = long_df[
            long_df["trait_type"].astype(str).eq("binary")
        ]
        missing_k = binary_rows["K_primary"].isna()
        if missing_k.any():
            examples = (
                binary_rows.loc[missing_k, "source_file"]
                .drop_duplicates()
                .head(10)
                .tolist()
            )
            raise RuntimeError(
                "Real binary rows are missing K_primary. "
                f"Examples: {examples}"
            )

    output_dir = root / "FinalAnalysis_Liability"
    output_dir.mkdir(parents=True, exist_ok=True)

    long_path = output_dir / f"FinalResults_Liability_Long_{mode}.csv"
    summary_path = output_dir / f"FinalResults_Liability_Summary_{mode}.csv"
    audit_path = output_dir / f"Liability_Audit_{mode}.csv"
    error_path = output_dir / "Liability_Errors.csv"
    missing_path = output_dir / "Liability_MissingExpectedFiles.csv"

    # Quietly record any of the 500 expected real-data files that are absent.
    if missing_files:
        pd.DataFrame(
            {"missing_expected_file": [str(p.relative_to(root)) for p in missing_files]}
        ).to_csv(missing_path, index=False)
    elif missing_path.exists():
        missing_path.unlink()

    # Write each master table once.
    long_df.to_csv(long_path, index=False)

    summary_df = build_summary(long_df)
    summary_df.to_csv(summary_path, index=False)

    audit_df = build_audit(long_df)
    audit_df.to_csv(audit_path, index=False)

    # Compact conversion-status summary for quick verification.
    status_summary = (
        long_df.groupby(
            ["analysis_mode", "phenotype_standardized", "trait_type", "conversion_status"],
            dropna=False
        )
        .size()
        .reset_index(name="n_rows")
    )
    status_summary.to_csv(
        output_dir / "Liability_ConversionStatusSummary.csv",
        index=False
    )

    # Errors are saved rather than printed.
    if errors:
        pd.DataFrame(errors).to_csv(error_path, index=False)
    elif error_path.exists():
        error_path.unlink()

    # No verbose terminal output. tqdm is the only normal screen output.


if __name__ == "__main__":
    main()
