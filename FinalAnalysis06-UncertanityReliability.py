# -*- coding: utf-8 -*-

"""
FinalAnalysis06-UncertaintyReliability.py
=========================================

Reviewer-facing uncertainty / stability / reliability audit for the heritability
benchmark.

PURPOSE
-------
This script directly addresses the reviewer concern that large differences in
point-estimate SNP heritability may reflect finite-sample uncertainty rather
than meaningful methodological differences.

It analyses BOTH representations of every eligible binary-trait estimate:

    1. observed/native scale
    2. primary liability scale

for BOTH:

    1. empirical phenotypes
    2. simulated phenotypes

Continuous phenotypes are analysed once on their native quantitative scale.

CRITICAL INTERPRETATION
-----------------------
The five cross-validation training folds overlap. Therefore:

    * cross-fold SD, MAD, IQR and range are DESCRIPTIVE STABILITY measures;
    * cross-fold SEM and the corresponding 95% interval are reported only as
      DESCRIPTIVE FOLD-STABILITY summaries;
    * they MUST NOT be interpreted as estimator-specific inferential standard
      errors or confidence intervals.

Software-native / estimator-native SE or CI values are retained separately
where they exist in Results_Liability.csv. They are never replaced by the
cross-fold quantities.

Negative h2 values and h2 > 1 are retained without clipping. Configuration-level
mean estimates are classified into three mutually exclusive descriptive groups:

    h2 < 0
    0 <= h2 <= 1
    h2 > 1

The conventional 0--1 interval is used as a descriptive biological range only,
NOT as an exclusion rule.

CONVERGENCE
-----------
Convergence is reported ONLY when explicit convergence/status/warning/error
information is present in the underlying output row. Missing files are NOT
automatically called convergence failures.

INPUT
-----
Reads ONLY:
    Results_Liability.csv

It never falls back to Results.csv for heritability values.

Expected layout:
    <root>/<dataset>/Fold_<0..4>/<method>/Results_Liability.csv

Default root:
    /data/ascher01/uqmmune1/finalheritability

Override:
    HERITABILITY_ROOT=/your/path python FinalAnalysis06-UncertaintyReliability.py

RUN
---
    python FinalAnalysis06-UncertaintyReliability.py \
        | tee FinalAnalysis06-UncertaintyReliability_terminal.txt

OUTPUT
------
Created under:
    FinalAnalysis06-UncertaintyReliability/

Main files:
    FinalAnalysis06_MAIN_RESULTS.txt
    FinalAnalysis06-UncertaintyReliability.xlsx

CSV files:
    FinalAnalysis06_raw_scale_rows.csv
    FinalAnalysis06_file_audit.csv
    FinalAnalysis06_unit_stability.csv
    FinalAnalysis06_boundary_summary.csv
    FinalAnalysis06_stability_summary.csv
    FinalAnalysis06_uncertainty_availability.csv
    FinalAnalysis06_convergence_summary.csv
    FinalAnalysis06_sample_size_stability.csv
    FinalAnalysis06_scale_pair_audit.csv
    FinalAnalysis06_excluded_or_missing_scale.csv

SCIENTIFIC QUESTIONS ANSWERED
-----------------------------
1. How much cross-fold instability is present for each method/configuration?
2. What proportion of configuration-level mean estimates are:
       <0, within 0--1, or >1?
3. Do those proportions differ on observed vs liability scale?
4. Are software-native uncertainty measures available, and for which methods?
5. What explicit convergence information is available?
6. How many files/outputs are missing, separately from convergence?
7. In simulations, does increasing N from 750 to 5000 reduce cross-fold
   instability?
8. Does the small empirical sample likely amplify instability while workflow
   effects still persist in the larger-N controlled benchmark?

This script is deliberately conservative: it does not invent uncertainty,
does not treat overlapping folds as independent replicates, does not clip
out-of-range values, and does not infer non-convergence merely from missing
output.
"""

from __future__ import annotations

import math
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from scipy.stats import spearmanr, wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# =============================================================================
# 1. PATHS / CONSTANTS
# =============================================================================

ROOT = Path(
    os.environ.get(
        "HERITABILITY_ROOT",
        "/data/ascher01/uqmmune1/finalheritability",
    )
).resolve()

OUTDIR = ROOT / "FinalAnalysis06-UncertaintyReliability"
OUTDIR.mkdir(parents=True, exist_ok=True)

EXPECTED_FOLDS = 5
MIN_FOLDS_FOR_STABILITY = 4

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

METHOD_FAMILY = {
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

REAL_LABELS = {
    "asthma": "Asthma",
    "body_mass_index_bmi": "Body Mass Index",
    "depression": "Depression",
    "gastro_oesophageal_reflux_gord_gastric_reflux": "Gastro-Oesophageal Reflux",
    "high_cholesterol": "High Cholesterol",
    "hypertension": "Hypertension",
    "hypothyroidism_myxoedema": "Hypothyroidism",
    "irritable_bowel_syndrome": "Irritable Bowel Syndrome",
    "migraine": "Migraine",
    "osteoarthritis": "Osteoarthritis",
}

# Configuration columns used elsewhere in this benchmark.  Only columns that are
# actually present are included in the ID.
CONFIG_COLUMN_NAMES = [
    "h2model_standardized",
    "h2model",
    "model",
    "configuration",
    "config",
    "clumpprune",
    "clump_p1",
    "clump_r2",
    "clump_kb",
    "p_window_size",
    "p_slide_size",
    "p_ld_threshold",
    "numberofpca",
    "number_of_pca",
    "pca",
    "grm_source",
    "grm",
    "snp_set",
    "variant_set",
    "ld_reference",
    "ld_ref",
    "ldsc_weights",
    "weights",
    "tagging_file",
    "alpha",
]

# Original/native CI aliases. Liability CI has explicit standardized columns.
NATIVE_CI_LO_ALIASES = [
    "ci_lower_native",
    "native_ci_lower",
    "h2_ci_lower",
    "h2_lower",
    "heritability_lower",
    "ci_lower",
    "lower_ci",
    "lci",
    "ci95_lower",
    "lower_95",
]

NATIVE_CI_HI_ALIASES = [
    "ci_upper_native",
    "native_ci_upper",
    "h2_ci_upper",
    "h2_upper",
    "heritability_upper",
    "ci_upper",
    "upper_ci",
    "uci",
    "ci95_upper",
    "upper_95",
]


# =============================================================================
# 2. HELPERS
# =============================================================================

def norm(x) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).lower()).strip("_")


def clean(x) -> str:
    if pd.isna(x):
        return ""
    s = re.sub(r"\s+", " ", str(x)).strip()
    if s.lower() in {"", "nan", "none", "na", "n/a", "-", "--"}:
        return ""
    return s


def fnum(x) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else np.nan
    except Exception:
        return np.nan


def fmt(x, d=4) -> str:
    try:
        x = float(x)
        if not np.isfinite(x):
            return "NA"
        if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e4):
            return f"{x:.3e}"
        return f"{x:.{d}f}"
    except Exception:
        return "NA"


def pct(n, d) -> float:
    return 100.0 * n / d if d else np.nan


def mad(values) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    if len(x) == 0:
        return np.nan
    m = np.median(x)
    return float(np.median(np.abs(x - m)))


def iqr(values) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    if len(x) == 0:
        return np.nan
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    q = np.full(len(p), np.nan)
    ok = np.where(np.isfinite(p))[0]
    if len(ok) == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    q[ok[order]] = adj
    return q


def safe_wilcoxon(diff) -> float:
    x = pd.to_numeric(pd.Series(diff), errors="coerce").dropna().to_numpy()
    if len(x) < 5 or not HAVE_SCIPY:
        return np.nan
    if np.allclose(x, 0):
        return 1.0
    try:
        return float(
            wilcoxon(
                x,
                zero_method="wilcox",
                alternative="two-sided",
            ).pvalue
        )
    except Exception:
        return np.nan


def safe_spearman(x, y):
    a = pd.to_numeric(pd.Series(x), errors="coerce")
    b = pd.to_numeric(pd.Series(y), errors="coerce")
    keep = a.notna() & b.notna()
    a = a[keep].to_numpy()
    b = b[keep].to_numpy()
    if len(a) < 3 or not HAVE_SCIPY:
        return np.nan
    if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return np.nan
    try:
        return float(spearmanr(a, b).statistic)
    except Exception:
        return np.nan


def norm_map(df: pd.DataFrame) -> dict[str, str]:
    """Map normalized column name -> original column name."""
    out = {}
    for c in df.columns:
        out.setdefault(norm(c), c)
    return out


def first_col(df: pd.DataFrame, aliases) -> str | None:
    nm = norm_map(df)
    for a in aliases:
        if norm(a) in nm:
            return nm[norm(a)]
    return None


def get_row_value(row: pd.Series, df: pd.DataFrame, aliases, default=np.nan):
    c = first_col(df, aliases)
    if c is None:
        return default
    return row.get(c, default)


def get_method(row: pd.Series, df: pd.DataFrame, method_dir: str) -> str:
    for name in ["method_standardized", "Method", "method", "method_name", "tool"]:
        c = first_col(df, [name])
        if c is not None:
            v = clean(row.get(c, ""))
            if v:
                return v
    return method_dir


def make_config_id(row: pd.Series, df: pd.DataFrame, method: str, source_row: int) -> str:
    """
    Stable semantic configuration identifier.

    Uses the actual configuration fields present in Results_Liability.csv.
    Falls back to source-row index only if no configuration field exists.
    """
    nm = norm_map(df)
    parts = [method]

    seen = set()
    for wanted in CONFIG_COLUMN_NAMES:
        nw = norm(wanted)
        if nw not in nm:
            continue
        c = nm[nw]
        v = clean(row.get(c, ""))
        if not v:
            continue
        key = f"{nw}={v}"
        if key not in seen:
            parts.append(key)
            seen.add(key)

    if len(parts) == 1:
        parts.append(f"source_row={source_row}")

    return " | ".join(parts)


def infer_simulation_metadata(sim_name: str):
    """
    Deterministic fallback based on the benchmark's 200-dataset design.

    Sim001--050   : continuous, N=750
    Sim051--100   : continuous, N=5000
    Sim101--150   : binary, N=750
    Sim151--200   : binary, N=5000

    Within each 50-trait block:
      traits 01--10 -> h2=0.05
      traits 11--20 -> h2=0.10
      traits 21--30 -> h2=0.30
      traits 31--40 -> h2=0.50
      traits 41--50 -> h2=0.70

    Within each 10-trait h2 block:
      first 5 -> 100 causal variants
      next 5  -> 5000 causal variants
    """
    m = re.fullmatch(r"Sim0*(\d+)", str(sim_name), re.I)
    if not m:
        return {
            "trait_type": "",
            "sample_size": np.nan,
            "underlying_trait": "",
            "true_h2": np.nan,
            "causal_snps": np.nan,
            "replicate": np.nan,
        }

    sid = int(m.group(1))

    if 1 <= sid <= 50:
        trait_type = "continuous"
        n = 750
        idx = sid
    elif 51 <= sid <= 100:
        trait_type = "continuous"
        n = 5000
        idx = sid - 50
    elif 101 <= sid <= 150:
        trait_type = "binary"
        n = 750
        idx = sid - 100
    elif 151 <= sid <= 200:
        trait_type = "binary"
        n = 5000
        idx = sid - 150
    else:
        return {
            "trait_type": "",
            "sample_size": np.nan,
            "underlying_trait": "",
            "true_h2": np.nan,
            "causal_snps": np.nan,
            "replicate": np.nan,
        }

    hvals = [0.05, 0.10, 0.30, 0.50, 0.70]
    hgroup = min(4, (idx - 1) // 10)
    within10 = (idx - 1) % 10 + 1

    true_h2 = hvals[hgroup]
    causal = 100 if within10 <= 5 else 5000
    rep = ((within10 - 1) % 5) + 1
    prefix = "continuous" if trait_type == "continuous" else "binary"

    return {
        "trait_type": trait_type,
        "sample_size": n,
        "underlying_trait": f"{prefix}_trait_{idx:02d}",
        "true_h2": true_h2,
        "causal_snps": causal,
        "replicate": rep,
    }


def parse_explicit_convergence(row: pd.Series, df: pd.DataFrame):
    """
    Conservative convergence parser.

    Returns:
        convergence_status
        convergence_evidence

    It does NOT classify a missing file as non-convergence.
    """
    evidence = []
    explicit_convergence_values = []
    warning_values = []
    error_values = []

    for c in df.columns:
        nc = norm(c)

        # Ignore the liability-conversion status added by our own harmonization.
        if nc in {
            "conversion_status",
            "analysis_mode",
            "p_source",
            "p_note",
            "native_scale",
        }:
            continue

        v = clean(row.get(c, ""))
        if not v:
            continue

        lv = v.lower()

        if "converg" in nc:
            explicit_convergence_values.append((c, v))
            evidence.append(f"{c}={v}")

        elif "warning" in nc:
            warning_values.append((c, v))
            evidence.append(f"{c}={v}")

        elif "error" in nc or "failure" in nc or nc in {"status", "run_status"}:
            # Generic status columns are used only when the VALUE itself is
            # informative (converg/fail/error/warn).
            if any(tok in lv for tok in ["converg", "fail", "error", "warn", "success"]):
                if "warn" in lv:
                    warning_values.append((c, v))
                elif "fail" in lv or "error" in lv:
                    error_values.append((c, v))
                else:
                    explicit_convergence_values.append((c, v))
                evidence.append(f"{c}={v}")

    # Strong explicit convergence field takes priority.
    if explicit_convergence_values:
        vals = " ".join(v.lower() for _, v in explicit_convergence_values)

        bad_tokens = [
            "not converg",
            "nonconverg",
            "non-converg",
            "failed",
            "failure",
            "false",
            "no",
            "0",
        ]
        good_tokens = [
            "converged",
            "success",
            "true",
            "yes",
            "1",
        ]

        if any(t in vals for t in bad_tokens):
            return "explicit_nonconverged", "; ".join(evidence)

        if any(t in vals for t in good_tokens):
            if warning_values:
                return "explicit_converged_with_warning", "; ".join(evidence)
            return "explicit_converged", "; ".join(evidence)

    if error_values:
        return "explicit_failure_or_error", "; ".join(evidence)

    if warning_values:
        return "explicit_warning", "; ".join(evidence)

    return "not_reported", ""


def native_ci_columns(df: pd.DataFrame):
    lo = first_col(df, NATIVE_CI_LO_ALIASES)
    hi = first_col(df, NATIVE_CI_HI_ALIASES)

    # Never accidentally use the liability CI as a native CI.
    if lo is not None and "liability" in norm(lo):
        lo = None
    if hi is not None and "liability" in norm(hi):
        hi = None

    return lo, hi


def boundary_class(x):
    if not np.isfinite(x):
        return "missing"
    if x < 0:
        return "<0"
    if x > 1:
        return ">1"
    return "0_to_1"


# =============================================================================
# 3. FILE AUDIT
# =============================================================================

def expected_file_audit():
    rows = []

    # Empirical: exactly 10 phenotypes x 5 folds x 10 methods = 500 expected.
    for dataset in REAL_PHENOTYPES:
        ddir = ROOT / dataset
        trait_type = "continuous" if dataset == "body_mass_index_bmi" else "binary"

        for fold in range(EXPECTED_FOLDS):
            for method in METHODS:
                p_liab = ddir / f"Fold_{fold}" / method / "Results_Liability.csv"
                p_orig = ddir / f"Fold_{fold}" / method / "Results.csv"

                if p_liab.exists():
                    status = "liability_file_present"
                elif p_orig.exists():
                    status = "original_present_liability_missing"
                else:
                    status = "result_missing"

                rows.append({
                    "Domain": "empirical",
                    "Dataset": dataset,
                    "DatasetLabel": REAL_LABELS.get(dataset, dataset),
                    "TraitType": trait_type,
                    "SampleSize": np.nan,
                    "Fold": fold,
                    "Method": method,
                    "ExpectedLiabilityFile": str(p_liab),
                    "OriginalResultsExists": p_orig.exists(),
                    "LiabilityResultsExists": p_liab.exists(),
                    "FileStatus": status,
                })

    # Simulated: 200 x 5 x 10 = 10,000 expected if all Sim001--Sim200 exist.
    for sid in range(1, 201):
        sim = f"Sim{sid:03d}"
        meta = infer_simulation_metadata(sim)
        ddir = ROOT / sim

        for fold in range(EXPECTED_FOLDS):
            for method in METHODS:
                p_liab = ddir / f"Fold_{fold}" / method / "Results_Liability.csv"
                p_orig = ddir / f"Fold_{fold}" / method / "Results.csv"

                if p_liab.exists():
                    status = "liability_file_present"
                elif p_orig.exists():
                    status = "original_present_liability_missing"
                else:
                    status = "result_missing"

                rows.append({
                    "Domain": "simulated",
                    "Dataset": sim,
                    "DatasetLabel": sim,
                    "TraitType": meta["trait_type"],
                    "SampleSize": meta["sample_size"],
                    "Fold": fold,
                    "Method": method,
                    "ExpectedLiabilityFile": str(p_liab),
                    "OriginalResultsExists": p_orig.exists(),
                    "LiabilityResultsExists": p_liab.exists(),
                    "FileStatus": status,
                })

    return pd.DataFrame(rows)


# =============================================================================
# 4. READ Results_Liability.csv AND EXPAND TO SCALES
# =============================================================================

def read_one_liability_file(
    path: Path,
    *,
    domain: str,
    dataset: str,
    dataset_label: str,
    trait_type: str,
    sample_size=np.nan,
    underlying_trait="",
    true_h2=np.nan,
    causal_snps=np.nan,
    replicate=np.nan,
):
    rows = []
    missing_scale = []

    try:
        df = pd.read_csv(path)
    except Exception as e:
        missing_scale.append({
            "Domain": domain,
            "Dataset": dataset,
            "Fold": int(re.search(r"Fold_(\d+)", str(path)).group(1))
                if re.search(r"Fold_(\d+)", str(path)) else np.nan,
            "Method": path.parent.name,
            "Scale": "file",
            "Reason": f"read_error:{e}",
            "ResultFile": str(path),
        })
        return rows, missing_scale

    fold_match = re.search(r"Fold_(\d+)", str(path))
    fold = int(fold_match.group(1)) if fold_match else np.nan
    method_dir = path.parent.name

    # Required standardized columns created by FinalAnalysis01.
    c_h2_native = first_col(df, ["h2_native", "h2_observed"])
    c_se_native = first_col(df, ["h2_se_native"])
    c_h2_liab = first_col(df, ["h2_liability_primary", "h2_liability"])
    c_se_liab = first_col(df, ["h2_se_liability_primary"])
    c_liab_lo = first_col(df, ["CI_lower_liability_primary"])
    c_liab_hi = first_col(df, ["CI_upper_liability_primary"])
    c_factor = first_col(df, ["liability_factor_primary"])
    c_conv = first_col(df, ["conversion_status"])
    c_k = first_col(df, ["K_primary"])
    c_p = first_col(df, ["P_used"])
    c_p_source = first_col(df, ["P_source"])

    native_lo_col, native_hi_col = native_ci_columns(df)

    if c_h2_native is None:
        missing_scale.append({
            "Domain": domain,
            "Dataset": dataset,
            "Fold": fold,
            "Method": method_dir,
            "Scale": "native",
            "Reason": "missing_h2_native_column",
            "ResultFile": str(path),
        })
        return rows, missing_scale

    for source_row, row in df.iterrows():
        method = get_method(row, df, method_dir)
        family = METHOD_FAMILY.get(method, METHOD_FAMILY.get(method_dir, "Other"))
        config_id = make_config_id(row, df, method, source_row)

        h_native = fnum(row.get(c_h2_native, np.nan))
        se_native = fnum(row.get(c_se_native, np.nan)) if c_se_native else np.nan

        h_liab = fnum(row.get(c_h2_liab, np.nan)) if c_h2_liab else np.nan
        se_liab = fnum(row.get(c_se_liab, np.nan)) if c_se_liab else np.nan

        factor = fnum(row.get(c_factor, np.nan)) if c_factor else np.nan
        K_primary = fnum(row.get(c_k, np.nan)) if c_k else np.nan
        P_used = fnum(row.get(c_p, np.nan)) if c_p else np.nan
        P_source = clean(row.get(c_p_source, "")) if c_p_source else ""
        conversion_status = clean(row.get(c_conv, "")) if c_conv else ""

        # Native CI: use explicit software/native CI if it exists.
        native_lo = fnum(row.get(native_lo_col, np.nan)) if native_lo_col else np.nan
        native_hi = fnum(row.get(native_hi_col, np.nan)) if native_hi_col else np.nan
        native_ci_source = "software_explicit_ci" if (
            np.isfinite(native_lo) and np.isfinite(native_hi)
        ) else ""

        # If explicit native CI is absent but software-native SE exists, derive a
        # normal-approximation CI and LABEL it clearly as derived from tool SE.
        if (
            not np.isfinite(native_lo)
            or not np.isfinite(native_hi)
        ) and np.isfinite(h_native) and np.isfinite(se_native):
            native_lo = h_native - 1.96 * se_native
            native_hi = h_native + 1.96 * se_native
            native_ci_source = "derived_from_tool_native_se"

        # Liability CI is created in the harmonization script from transformed
        # tool-native SE. It remains estimator uncertainty, not fold uncertainty.
        liab_lo = fnum(row.get(c_liab_lo, np.nan)) if c_liab_lo else np.nan
        liab_hi = fnum(row.get(c_liab_hi, np.nan)) if c_liab_hi else np.nan
        liab_ci_source = ""

        if np.isfinite(liab_lo) and np.isfinite(liab_hi):
            liab_ci_source = "derived_from_transformed_tool_se"
        elif np.isfinite(h_liab) and np.isfinite(se_liab):
            liab_lo = h_liab - 1.96 * se_liab
            liab_hi = h_liab + 1.96 * se_liab
            liab_ci_source = "derived_from_transformed_tool_se"

        convergence_status, convergence_evidence = parse_explicit_convergence(row, df)

        common = {
            "Domain": domain,
            "Dataset": dataset,
            "DatasetLabel": dataset_label,
            "TraitType": trait_type,
            "SampleSize": sample_size,
            "UnderlyingTrait": underlying_trait,
            "TrueH2LiabilityOrQuantitative": true_h2,
            "CausalSNPs": causal_snps,
            "Replicate": replicate,
            "Fold": fold,
            "Method": method,
            "MethodFamily": family,
            "ConfigID": config_id,
            "SourceRow": source_row,
            "ResultFile": str(path),
            "LiabilityFactorPrimary": factor,
            "KPrimary": K_primary,
            "PUsed": P_used,
            "PSource": P_source,
            "ConversionStatus": conversion_status,
            "ConvergenceStatus": convergence_status,
            "ConvergenceEvidence": convergence_evidence,
        }

        if trait_type == "continuous":
            rec = dict(common)
            rec.update({
                "Scale": "native_quantitative",
                "H2": h_native,
                "ToolSE": se_native,
                "ToolCILower": native_lo,
                "ToolCIUpper": native_hi,
                "ToolCISource": native_ci_source,
            })
            rows.append(rec)

            if not np.isfinite(h_native):
                missing_scale.append({
                    "Domain": domain,
                    "Dataset": dataset,
                    "Fold": fold,
                    "Method": method,
                    "ConfigID": config_id,
                    "Scale": "native_quantitative",
                    "Reason": "missing_native_h2",
                    "ResultFile": str(path),
                })

        elif trait_type == "binary":
            # OBSERVED / NATIVE representation.
            rec_obs = dict(common)
            rec_obs.update({
                "Scale": "observed_binary",
                "H2": h_native,
                "ToolSE": se_native,
                "ToolCILower": native_lo,
                "ToolCIUpper": native_hi,
                "ToolCISource": native_ci_source,
            })
            rows.append(rec_obs)

            if not np.isfinite(h_native):
                missing_scale.append({
                    "Domain": domain,
                    "Dataset": dataset,
                    "Fold": fold,
                    "Method": method,
                    "ConfigID": config_id,
                    "Scale": "observed_binary",
                    "Reason": "missing_native_h2",
                    "ResultFile": str(path),
                })

            # LIABILITY representation. NEVER fall back to native h2.
            rec_liab = dict(common)
            rec_liab.update({
                "Scale": "liability_binary",
                "H2": h_liab,
                "ToolSE": se_liab,
                "ToolCILower": liab_lo,
                "ToolCIUpper": liab_hi,
                "ToolCISource": liab_ci_source,
            })
            rows.append(rec_liab)

            if not np.isfinite(h_liab):
                missing_scale.append({
                    "Domain": domain,
                    "Dataset": dataset,
                    "Fold": fold,
                    "Method": method,
                    "ConfigID": config_id,
                    "Scale": "liability_binary",
                    "Reason": (
                        f"missing_liability_h2:{conversion_status}"
                        if conversion_status
                        else "missing_liability_h2"
                    ),
                    "ResultFile": str(path),
                })

        else:
            missing_scale.append({
                "Domain": domain,
                "Dataset": dataset,
                "Fold": fold,
                "Method": method,
                "ConfigID": config_id,
                "Scale": "unknown",
                "Reason": "unknown_trait_type",
                "ResultFile": str(path),
            })

    return rows, missing_scale


def load_all_scale_rows():
    raw = []
    missing = []

    # Empirical
    for dataset in REAL_PHENOTYPES:
        trait_type = "continuous" if dataset == "body_mass_index_bmi" else "binary"
        label = REAL_LABELS.get(dataset, dataset)

        for fold in range(EXPECTED_FOLDS):
            for method in METHODS:
                path = ROOT / dataset / f"Fold_{fold}" / method / "Results_Liability.csv"
                if not path.exists():
                    continue

                r, m = read_one_liability_file(
                    path,
                    domain="empirical",
                    dataset=dataset,
                    dataset_label=label,
                    trait_type=trait_type,
                )
                raw.extend(r)
                missing.extend(m)

    # Simulated
    for sid in range(1, 201):
        sim = f"Sim{sid:03d}"
        meta = infer_simulation_metadata(sim)

        for fold in range(EXPECTED_FOLDS):
            for method in METHODS:
                path = ROOT / sim / f"Fold_{fold}" / method / "Results_Liability.csv"
                if not path.exists():
                    continue

                r, m = read_one_liability_file(
                    path,
                    domain="simulated",
                    dataset=sim,
                    dataset_label=sim,
                    trait_type=meta["trait_type"],
                    sample_size=meta["sample_size"],
                    underlying_trait=meta["underlying_trait"],
                    true_h2=meta["true_h2"],
                    causal_snps=meta["causal_snps"],
                    replicate=meta["replicate"],
                )
                raw.extend(r)
                missing.extend(m)

    return pd.DataFrame(raw), pd.DataFrame(missing)


# =============================================================================
# 5. FOLD AGGREGATION / STABILITY
# =============================================================================

UNIT_GROUP_COLS = [
    "Domain",
    "Dataset",
    "DatasetLabel",
    "TraitType",
    "SampleSize",
    "UnderlyingTrait",
    "TrueH2LiabilityOrQuantitative",
    "CausalSNPs",
    "Replicate",
    "Method",
    "MethodFamily",
    "ConfigID",
    "Scale",
]


def aggregate_units(raw: pd.DataFrame):
    if raw.empty:
        return pd.DataFrame()

    rows = []

    for keys, g in raw.groupby(
        UNIT_GROUP_COLS,
        dropna=False,
        sort=False,
    ):
        rec = dict(zip(UNIT_GROUP_COLS, keys))

        # Collapse duplicate rows inside a fold before cross-fold stability.
        fold_rows = []

        for fold, gf in g.groupby("Fold", dropna=True):
            h = pd.to_numeric(gf["H2"], errors="coerce").dropna()
            se = pd.to_numeric(gf["ToolSE"], errors="coerce").dropna()

            lo = pd.to_numeric(gf["ToolCILower"], errors="coerce")
            hi = pd.to_numeric(gf["ToolCIUpper"], errors="coerce")
            ok_ci = lo.notna() & hi.notna()

            fold_rows.append({
                "Fold": fold,
                "H2": float(h.mean()) if len(h) else np.nan,
                "ToolSE": float(se.median()) if len(se) else np.nan,
                "ToolCILower": float(lo[ok_ci].median()) if ok_ci.any() else np.nan,
                "ToolCIUpper": float(hi[ok_ci].median()) if ok_ci.any() else np.nan,
                "RawRows": len(gf),
            })

        fg = pd.DataFrame(fold_rows)

        h = pd.to_numeric(
            fg.get("H2", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()

        rec["RawRows"] = len(g)
        rec["FoldsPresent"] = int(fg["Fold"].nunique()) if not fg.empty else 0
        rec["ValidH2Folds"] = len(h)
        rec["MissingH2Folds"] = max(0, EXPECTED_FOLDS - len(h))
        rec["DuplicateRowsCollapsed"] = max(
            0,
            len(g) - rec["FoldsPresent"],
        )

        if len(h):
            rec["MeanH2"] = float(h.mean())
            rec["MedianH2"] = float(h.median())
            rec["MinH2"] = float(h.min())
            rec["MaxH2"] = float(h.max())
            rec["FoldSD"] = float(h.std(ddof=1)) if len(h) >= 2 else np.nan
            rec["FoldMAD"] = mad(h)
            rec["FoldIQR"] = iqr(h)
            rec["FoldRange"] = float(h.max() - h.min())

            # Descriptive only. DO NOT call this inferential SE.
            rec["FoldSEM_Descriptive"] = (
                rec["FoldSD"] / math.sqrt(len(h))
                if len(h) >= 2 and np.isfinite(rec["FoldSD"])
                else np.nan
            )

            rec["FoldStability95Lower_Descriptive"] = (
                rec["MeanH2"] - 1.96 * rec["FoldSEM_Descriptive"]
                if np.isfinite(rec["FoldSEM_Descriptive"])
                else np.nan
            )
            rec["FoldStability95Upper_Descriptive"] = (
                rec["MeanH2"] + 1.96 * rec["FoldSEM_Descriptive"]
                if np.isfinite(rec["FoldSEM_Descriptive"])
                else np.nan
            )

            rec["MeanBoundaryClass"] = boundary_class(rec["MeanH2"])
            rec["NegativeFoldFraction"] = float((h < 0).mean())
            rec["Within01FoldFraction"] = float(((h >= 0) & (h <= 1)).mean())
            rec["GreaterThan1FoldFraction"] = float((h > 1).mean())
            rec["AnyNegativeFold"] = int((h < 0).any())
            rec["AnyGreaterThan1Fold"] = int((h > 1).any())
            rec["SignFlipAcrossFolds"] = int(h.min() < 0 < h.max())
        else:
            for c in [
                "MeanH2",
                "MedianH2",
                "MinH2",
                "MaxH2",
                "FoldSD",
                "FoldMAD",
                "FoldIQR",
                "FoldRange",
                "FoldSEM_Descriptive",
                "FoldStability95Lower_Descriptive",
                "FoldStability95Upper_Descriptive",
                "NegativeFoldFraction",
                "Within01FoldFraction",
                "GreaterThan1FoldFraction",
            ]:
                rec[c] = np.nan

            rec["MeanBoundaryClass"] = "missing"
            rec["AnyNegativeFold"] = 0
            rec["AnyGreaterThan1Fold"] = 0
            rec["SignFlipAcrossFolds"] = 0

        # Software-native uncertainty is kept separate.
        se = pd.to_numeric(
            fg.get("ToolSE", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()

        rec["ToolSEAvailableFolds"] = len(se)
        rec["ToolSEMedian"] = float(se.median()) if len(se) else np.nan
        rec["ToolSEMean"] = float(se.mean()) if len(se) else np.nan

        lo = pd.to_numeric(
            fg.get("ToolCILower", pd.Series(dtype=float)),
            errors="coerce",
        )
        hi = pd.to_numeric(
            fg.get("ToolCIUpper", pd.Series(dtype=float)),
            errors="coerce",
        )

        ok_ci = lo.notna() & hi.notna()
        rec["ToolCIAvailableFolds"] = int(ok_ci.sum())

        if ok_ci.any():
            widths = hi[ok_ci] - lo[ok_ci]
            rec["ToolCIMedianWidth"] = float(widths.median())
            rec["ToolCIContainsZeroFraction"] = float(
                ((lo[ok_ci] <= 0) & (hi[ok_ci] >= 0)).mean()
            )
        else:
            rec["ToolCIMedianWidth"] = np.nan
            rec["ToolCIContainsZeroFraction"] = np.nan

        rec["StabilityEvaluable"] = int(
            rec["ValidH2Folds"] >= MIN_FOLDS_FOR_STABILITY
        )

        rows.append(rec)

    return pd.DataFrame(rows)


# =============================================================================
# 6. BOUNDARY / PLAUSIBLE-RANGE SUMMARIES
# =============================================================================

def summarize_boundaries_for_group(g, extra):
    valid = g[pd.to_numeric(g["MeanH2"], errors="coerce").notna()].copy()
    n = len(valid)

    n_neg = int((valid["MeanH2"] < 0).sum()) if n else 0
    n_01 = int(((valid["MeanH2"] >= 0) & (valid["MeanH2"] <= 1)).sum()) if n else 0
    n_gt1 = int((valid["MeanH2"] > 1).sum()) if n else 0

    out = dict(extra)
    out.update({
        "ValidConfigurationPhenotypeUnits": n,
        "N_Below0": n_neg,
        "Pct_Below0": pct(n_neg, n),
        "N_Within0To1": n_01,
        "Pct_Within0To1": pct(n_01, n),
        "N_Above1": n_gt1,
        "Pct_Above1": pct(n_gt1, n),
        "MeanH2": float(valid["MeanH2"].mean()) if n else np.nan,
        "MedianH2": float(valid["MeanH2"].median()) if n else np.nan,
        "MinH2": float(valid["MeanH2"].min()) if n else np.nan,
        "MaxH2": float(valid["MeanH2"].max()) if n else np.nan,
        "PctUnitsWithAnyNegativeFold": (
            100.0 * valid["AnyNegativeFold"].mean() if n else np.nan
        ),
        "PctUnitsWithAnyAbove1Fold": (
            100.0 * valid["AnyGreaterThan1Fold"].mean() if n else np.nan
        ),
    })
    return out


def boundary_summary(units: pd.DataFrame):
    rows = []

    if units.empty:
        return pd.DataFrame()

    # Overall by domain / trait / scale / N.
    group_cols = ["Domain", "TraitType", "Scale", "SampleSize"]
    for keys, g in units.groupby(group_cols, dropna=False):
        extra = dict(zip(group_cols, keys))
        extra.update({
            "SummaryLevel": "overall",
            "Group": "ALL",
        })
        rows.append(summarize_boundaries_for_group(g, extra))

    # By exact method.
    group_cols = ["Domain", "TraitType", "Scale", "SampleSize", "Method"]
    for keys, g in units.groupby(group_cols, dropna=False):
        d = dict(zip(group_cols, keys))
        method = d.pop("Method")
        d.update({
            "SummaryLevel": "method",
            "Group": method,
        })
        rows.append(summarize_boundaries_for_group(g, d))

    # By method family.
    group_cols = ["Domain", "TraitType", "Scale", "SampleSize", "MethodFamily"]
    for keys, g in units.groupby(group_cols, dropna=False):
        d = dict(zip(group_cols, keys))
        family = d.pop("MethodFamily")
        d.update({
            "SummaryLevel": "method_family",
            "Group": family,
        })
        rows.append(summarize_boundaries_for_group(g, d))

    # By phenotype/simulation dataset.
    group_cols = ["Domain", "TraitType", "Scale", "SampleSize", "DatasetLabel"]
    for keys, g in units.groupby(group_cols, dropna=False):
        d = dict(zip(group_cols, keys))
        label = d.pop("DatasetLabel")
        d.update({
            "SummaryLevel": "dataset",
            "Group": label,
        })
        rows.append(summarize_boundaries_for_group(g, d))

    return pd.DataFrame(rows)


# =============================================================================
# 7. STABILITY SUMMARY
# =============================================================================

def stability_summary(units: pd.DataFrame):
    rows = []

    if units.empty:
        return pd.DataFrame()

    group_cols = [
        "Domain",
        "TraitType",
        "Scale",
        "SampleSize",
        "Method",
        "MethodFamily",
    ]

    for keys, g in units.groupby(group_cols, dropna=False):
        r = dict(zip(group_cols, keys))

        evalg = g[
            (g["StabilityEvaluable"] == 1)
            & pd.to_numeric(g["FoldSD"], errors="coerce").notna()
        ].copy()

        r["UnitsTotal"] = len(g)
        r["UnitsStabilityEvaluable"] = len(evalg)
        r["PctStabilityEvaluable"] = pct(len(evalg), len(g))

        for col in ["FoldSD", "FoldMAD", "FoldIQR", "FoldRange"]:
            x = pd.to_numeric(evalg[col], errors="coerce").dropna()
            r[f"{col}_Mean"] = float(x.mean()) if len(x) else np.nan
            r[f"{col}_Median"] = float(x.median()) if len(x) else np.nan
            r[f"{col}_Q25"] = float(x.quantile(0.25)) if len(x) else np.nan
            r[f"{col}_Q75"] = float(x.quantile(0.75)) if len(x) else np.nan

        r["PctSignFlipAcrossFolds"] = (
            100.0 * evalg["SignFlipAcrossFolds"].mean()
            if len(evalg)
            else np.nan
        )

        r["PctAnyNegativeFold"] = (
            100.0 * evalg["AnyNegativeFold"].mean()
            if len(evalg)
            else np.nan
        )

        r["PctAnyAbove1Fold"] = (
            100.0 * evalg["AnyGreaterThan1Fold"].mean()
            if len(evalg)
            else np.nan
        )

        rows.append(r)

    return pd.DataFrame(rows)


# =============================================================================
# 8. SOFTWARE-NATIVE UNCERTAINTY AVAILABILITY
# =============================================================================

def uncertainty_availability(units: pd.DataFrame):
    rows = []

    if units.empty:
        return pd.DataFrame()

    group_cols = [
        "Domain",
        "TraitType",
        "Scale",
        "SampleSize",
        "Method",
    ]

    for keys, g in units.groupby(group_cols, dropna=False):
        r = dict(zip(group_cols, keys))

        valid = g[g["ValidH2Folds"] >= 1].copy()
        n = len(valid)

        n_se = int((valid["ToolSEAvailableFolds"] > 0).sum())
        n_ci = int((valid["ToolCIAvailableFolds"] > 0).sum())

        r.update({
            "UnitsWithNumericalH2": n,
            "UnitsWithToolSE": n_se,
            "PctUnitsWithToolSE": pct(n_se, n),
            "UnitsWithToolCI": n_ci,
            "PctUnitsWithToolCI": pct(n_ci, n),
            "MedianToolSE": (
                float(pd.to_numeric(valid["ToolSEMedian"], errors="coerce").median())
                if n_se else np.nan
            ),
            "MedianToolCIWidth": (
                float(pd.to_numeric(valid["ToolCIMedianWidth"], errors="coerce").median())
                if n_ci else np.nan
            ),
        })

        rows.append(r)

    return pd.DataFrame(rows)


# =============================================================================
# 9. EXPLICIT CONVERGENCE SUMMARY
# =============================================================================

def convergence_summary(raw: pd.DataFrame):
    if raw.empty:
        return pd.DataFrame()

    # Convergence is identical across scale representations for the same source
    # output row. De-duplicate so binary rows are not counted twice.
    base_cols = [
        "Domain",
        "Dataset",
        "TraitType",
        "SampleSize",
        "Fold",
        "Method",
        "ConfigID",
        "SourceRow",
        "ResultFile",
        "ConvergenceStatus",
        "ConvergenceEvidence",
    ]

    x = raw[base_cols].drop_duplicates().copy()

    rows = []

    group_cols = [
        "Domain",
        "TraitType",
        "SampleSize",
        "Method",
    ]

    for keys, g in x.groupby(group_cols, dropna=False):
        r = dict(zip(group_cols, keys))

        n = len(g)
        reported = g["ConvergenceStatus"] != "not_reported"

        r.update({
            "SourceRows": n,
            "RowsWithExplicitConvergenceOrWarningInfo": int(reported.sum()),
            "PctRowsWithExplicitConvergenceOrWarningInfo": pct(int(reported.sum()), n),
            "ExplicitConverged": int(
                g["ConvergenceStatus"]
                .isin(["explicit_converged", "explicit_converged_with_warning"])
                .sum()
            ),
            "ExplicitNonConverged": int(
                (g["ConvergenceStatus"] == "explicit_nonconverged").sum()
            ),
            "ExplicitWarnings": int(
                g["ConvergenceStatus"]
                .isin(["explicit_warning", "explicit_converged_with_warning"])
                .sum()
            ),
            "ExplicitFailureOrError": int(
                (g["ConvergenceStatus"] == "explicit_failure_or_error").sum()
            ),
            "NotReported": int(
                (g["ConvergenceStatus"] == "not_reported").sum()
            ),
        })

        rows.append(r)

    return pd.DataFrame(rows)


# =============================================================================
# 10. SIMULATED N=750 vs N=5000 STABILITY EFFECT
# =============================================================================

def sample_size_stability(units: pd.DataFrame):
    """
    Pair the same underlying simulated trait + method + configuration across
    N=750 and N=5000.

    Primary statistic:
        delta FoldSD = FoldSD_5000 - FoldSD_750

    Negative delta means greater cross-fold stability at N=5000.

    BH-FDR is applied across method-level tests separately within each
    trait-type/scale analysis family.
    """
    sim = units[
        (units["Domain"] == "simulated")
        & (units["StabilityEvaluable"] == 1)
    ].copy()

    if sim.empty:
        return pd.DataFrame()

    sim["SampleSize"] = pd.to_numeric(sim["SampleSize"], errors="coerce")
    sim = sim[sim["SampleSize"].isin([750, 5000])].copy()

    pair_keys = [
        "TraitType",
        "Scale",
        "UnderlyingTrait",
        "Method",
        "MethodFamily",
        "ConfigID",
    ]

    p = sim.pivot_table(
        index=pair_keys,
        columns="SampleSize",
        values=["FoldSD", "FoldRange", "MeanH2"],
        aggfunc="mean",
    )

    if p.empty:
        return pd.DataFrame()

    p.columns = [
        f"{metric}_{int(n)}"
        for metric, n in p.columns
    ]
    p = p.reset_index()

    needed = ["FoldSD_750", "FoldSD_5000"]
    if not all(c in p.columns for c in needed):
        return pd.DataFrame()

    p = p[
        p["FoldSD_750"].notna()
        & p["FoldSD_5000"].notna()
    ].copy()

    p["DeltaFoldSD_5000_minus_750"] = (
        p["FoldSD_5000"] - p["FoldSD_750"]
    )

    if "FoldRange_750" in p.columns and "FoldRange_5000" in p.columns:
        p["DeltaFoldRange_5000_minus_750"] = (
            p["FoldRange_5000"] - p["FoldRange_750"]
        )

    rows = []

    for (tt, scale, method, family), g in p.groupby(
        ["TraitType", "Scale", "Method", "MethodFamily"],
        dropna=False,
    ):
        d = pd.to_numeric(
            g["DeltaFoldSD_5000_minus_750"],
            errors="coerce",
        ).dropna()

        rows.append({
            "TraitType": tt,
            "Scale": scale,
            "Method": method,
            "MethodFamily": family,
            "PairedConfigurationTraitUnits": len(d),
            "MedianFoldSD_N750": float(g["FoldSD_750"].median()) if len(g) else np.nan,
            "MedianFoldSD_N5000": float(g["FoldSD_5000"].median()) if len(g) else np.nan,
            "MedianDeltaFoldSD_5000_minus_750": float(d.median()) if len(d) else np.nan,
            "MeanDeltaFoldSD_5000_minus_750": float(d.mean()) if len(d) else np.nan,
            "PctPairsMoreStableAtN5000": (
                100.0 * (d < 0).mean() if len(d) else np.nan
            ),
            "WilcoxonP": safe_wilcoxon(d),
        })

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out["BH_FDR_Q"] = np.nan

    # Correction within phenotype-type / scale family.
    for (tt, scale), idx in out.groupby(
        ["TraitType", "Scale"],
        dropna=False,
    ).groups.items():
        idx = list(idx)
        out.loc[idx, "BH_FDR_Q"] = bh_fdr(
            out.loc[idx, "WilcoxonP"].to_numpy()
        )

    return out


# =============================================================================
# 11. OBSERVED-vs-LIABILITY SCALE PAIR AUDIT
# =============================================================================

def scale_pair_audit(units: pd.DataFrame):
    binary = units[units["TraitType"] == "binary"].copy()

    obs = binary[binary["Scale"] == "observed_binary"].copy()
    liab = binary[binary["Scale"] == "liability_binary"].copy()

    keys = [
        "Domain",
        "Dataset",
        "SampleSize",
        "UnderlyingTrait",
        "Method",
        "ConfigID",
    ]

    z = obs[
        keys + ["MeanH2", "FoldSD", "ValidH2Folds"]
    ].merge(
        liab[
            keys + ["MeanH2", "FoldSD", "ValidH2Folds"]
        ],
        on=keys,
        how="inner",
        suffixes=("_Observed", "_Liability"),
    )

    if z.empty:
        return pd.DataFrame()

    rows = []

    for (domain, n), g in z.groupby(
        ["Domain", "SampleSize"],
        dropna=False,
    ):
        rows.append({
            "Domain": domain,
            "SampleSize": n,
            "MatchedUnits": len(g),
            "SpearmanMeanH2_ObservedVsLiability": safe_spearman(
                g["MeanH2_Observed"],
                g["MeanH2_Liability"],
            ),
            "SpearmanFoldSD_ObservedVsLiability": safe_spearman(
                g["FoldSD_Observed"],
                g["FoldSD_Liability"],
            ),
            "MedianObservedMeanH2": float(
                pd.to_numeric(g["MeanH2_Observed"], errors="coerce").median()
            ),
            "MedianLiabilityMeanH2": float(
                pd.to_numeric(g["MeanH2_Liability"], errors="coerce").median()
            ),
            "MedianObservedFoldSD": float(
                pd.to_numeric(g["FoldSD_Observed"], errors="coerce").median()
            ),
            "MedianLiabilityFoldSD": float(
                pd.to_numeric(g["FoldSD_Liability"], errors="coerce").median()
            ),
        })

    return pd.DataFrame(rows)


# =============================================================================
# 12. MAIN-RESULTS TEXT
# =============================================================================

def find_overall_boundary(boundary, domain, trait_type, scale, sample_size=None):
    z = boundary[
        (boundary["SummaryLevel"] == "overall")
        & (boundary["Domain"] == domain)
        & (boundary["TraitType"] == trait_type)
        & (boundary["Scale"] == scale)
    ].copy()

    if sample_size is None:
        z = z[z["SampleSize"].isna()]
    else:
        z = z[
            pd.to_numeric(z["SampleSize"], errors="coerce")
            == sample_size
        ]

    return z.iloc[0] if len(z) else None


def write_main_summary(
    raw,
    file_audit,
    units,
    boundary,
    stability,
    uncertainty,
    convergence,
    sample_n,
    scale_audit,
    missing_scale,
):
    L = []

    L.append("FINAL ANALYSIS 06 — UNCERTAINTY / STABILITY / RELIABILITY")
    L.append("=" * 115)
    L.append("")
    L.append(f"ROOT: {ROOT}")
    L.append("")
    L.append("CRITICAL INTERPRETATION")
    L.append("-" * 115)
    L.append(
        "Cross-fold SD/MAD/IQR/range are DESCRIPTIVE stability metrics because the five CV training folds overlap."
    )
    L.append(
        "Fold SEM and fold-stability intervals are descriptive only; they are NOT estimator-specific inferential SE/CI."
    )
    L.append(
        "Software-native/tool-native SE and CI are retained separately where available."
    )
    L.append(
        "Negative and >1 estimates are retained without clipping."
    )
    L.append(
        "Missing files are reported separately and are NOT automatically classified as convergence failures."
    )
    L.append("")

    # -------------------------------------------------------------------------
    # File completeness
    # -------------------------------------------------------------------------
    L.append("FILE-LEVEL COMPLETENESS")
    L.append("-" * 115)

    for domain, g in file_audit.groupby("Domain"):
        total = len(g)
        present = int(g["LiabilityResultsExists"].sum())
        orig_no_liab = int(
            (g["FileStatus"] == "original_present_liability_missing").sum()
        )
        absent = int((g["FileStatus"] == "result_missing").sum())

        L.append(
            f"{domain}: Results_Liability.csv present {present}/{total} ({pct(present,total):.2f}%); "
            f"original Results.csv present but liability file missing={orig_no_liab}; "
            f"both missing={absent}."
        )
    L.append("")

    # -------------------------------------------------------------------------
    # Empirical boundary counts
    # -------------------------------------------------------------------------
    L.append("EMPIRICAL CONFIGURATION-LEVEL MEAN h2: CONVENTIONAL-RANGE AUDIT")
    L.append("-" * 115)

    specs = [
        ("binary", "observed_binary", "Binary observed/native"),
        ("binary", "liability_binary", "Binary liability"),
        ("continuous", "native_quantitative", "BMI native quantitative"),
    ]

    for tt, scale, label in specs:
        r = find_overall_boundary(
            boundary,
            "empirical",
            tt,
            scale,
            None,
        )
        if r is None:
            continue

        L.append(
            f"{label}: N={int(r['ValidConfigurationPhenotypeUnits'])}; "
            f"h2<0: {int(r['N_Below0'])} ({r['Pct_Below0']:.1f}%); "
            f"0<=h2<=1: {int(r['N_Within0To1'])} ({r['Pct_Within0To1']:.1f}%); "
            f"h2>1: {int(r['N_Above1'])} ({r['Pct_Above1']:.1f}%); "
            f"range={fmt(r['MinH2'])} to {fmt(r['MaxH2'])}."
        )
    L.append("")

    # -------------------------------------------------------------------------
    # Simulated boundary counts by N
    # -------------------------------------------------------------------------
    L.append("SIMULATED CONFIGURATION-LEVEL MEAN h2: CONVENTIONAL-RANGE AUDIT")
    L.append("-" * 115)

    specs = [
        ("continuous", "native_quantitative", "Continuous native"),
        ("binary", "observed_binary", "Binary observed"),
        ("binary", "liability_binary", "Binary liability"),
    ]

    for tt, scale, label in specs:
        for n in [750, 5000]:
            r = find_overall_boundary(
                boundary,
                "simulated",
                tt,
                scale,
                n,
            )
            if r is None:
                continue

            L.append(
                f"{label}, N={n}: units={int(r['ValidConfigurationPhenotypeUnits'])}; "
                f"h2<0={int(r['N_Below0'])} ({r['Pct_Below0']:.1f}%); "
                f"0<=h2<=1={int(r['N_Within0To1'])} ({r['Pct_Within0To1']:.1f}%); "
                f"h2>1={int(r['N_Above1'])} ({r['Pct_Above1']:.1f}%)."
            )
    L.append("")

    # -------------------------------------------------------------------------
    # Empirical stability by method
    # -------------------------------------------------------------------------
    L.append("EMPIRICAL CROSS-FOLD STABILITY BY METHOD")
    L.append("-" * 115)

    emp = stability[stability["Domain"] == "empirical"].copy()
    for scale in ["observed_binary", "liability_binary", "native_quantitative"]:
        z = emp[emp["Scale"] == scale].copy()
        if z.empty:
            continue

        L.append(scale)
        z = z.sort_values("FoldSD_Median")

        for _, r in z.iterrows():
            L.append(
                f"  {r['Method']}: evaluable={int(r['UnitsStabilityEvaluable'])}/{int(r['UnitsTotal'])}; "
                f"median cross-fold SD={fmt(r['FoldSD_Median'])}; "
                f"median range={fmt(r['FoldRange_Median'])}; "
                f"sign-flip units={fmt(r['PctSignFlipAcrossFolds'],1)}%."
            )
    L.append("")

    # -------------------------------------------------------------------------
    # Tool uncertainty availability
    # -------------------------------------------------------------------------
    L.append("SOFTWARE-NATIVE / ESTIMATOR-NATIVE UNCERTAINTY AVAILABILITY")
    L.append("-" * 115)

    for domain in ["empirical", "simulated"]:
        z = uncertainty[uncertainty["Domain"] == domain].copy()
        if z.empty:
            continue

        L.append(domain)

        for _, r in z.iterrows():
            if int(r["UnitsWithNumericalH2"]) == 0:
                continue

            L.append(
                f"  {r['Scale']} | {r['Method']}: "
                f"tool-SE available for {int(r['UnitsWithToolSE'])}/{int(r['UnitsWithNumericalH2'])} units "
                f"({fmt(r['PctUnitsWithToolSE'],1)}%); "
                f"tool-CI available for {int(r['UnitsWithToolCI'])}/{int(r['UnitsWithNumericalH2'])} "
                f"({fmt(r['PctUnitsWithToolCI'],1)}%)."
            )
    L.append("")

    # -------------------------------------------------------------------------
    # Explicit convergence
    # -------------------------------------------------------------------------
    L.append("EXPLICIT CONVERGENCE / WARNING INFORMATION")
    L.append("-" * 115)

    for domain in ["empirical", "simulated"]:
        z = convergence[convergence["Domain"] == domain].copy()
        if z.empty:
            continue

        total = int(z["SourceRows"].sum())
        reported = int(z["RowsWithExplicitConvergenceOrWarningInfo"].sum())
        nonconv = int(z["ExplicitNonConverged"].sum())
        errors = int(z["ExplicitFailureOrError"].sum())
        warns = int(z["ExplicitWarnings"].sum())

        L.append(
            f"{domain}: explicit convergence/warning/error information available for "
            f"{reported}/{total} source rows ({pct(reported,total):.1f}%); "
            f"explicit non-converged={nonconv}; explicit failure/error={errors}; explicit warnings={warns}."
        )

    L.append(
        "NOTE: not-reported convergence is NOT interpreted as successful convergence or failure."
    )
    L.append("")

    # -------------------------------------------------------------------------
    # Simulated sample-size effect
    # -------------------------------------------------------------------------
    L.append("SIMULATED PAIRED SAMPLE-SIZE EFFECT ON CROSS-FOLD STABILITY")
    L.append("-" * 115)
    L.append(
        "Delta SD = cross-fold SD at N=5000 minus cross-fold SD at N=750; negative = greater stability at N=5000."
    )

    if sample_n.empty:
        L.append("No pairable sample-size stability results.")
    else:
        for _, r in sample_n.sort_values(
            ["TraitType", "Scale", "BH_FDR_Q", "Method"]
        ).iterrows():
            L.append(
                f"{r['TraitType']} | {r['Scale']} | {r['Method']}: "
                f"paired units={int(r['PairedConfigurationTraitUnits'])}; "
                f"median SD N750={fmt(r['MedianFoldSD_N750'])}; "
                f"N5000={fmt(r['MedianFoldSD_N5000'])}; "
                f"median delta={fmt(r['MedianDeltaFoldSD_5000_minus_750'])}; "
                f"{fmt(r['PctPairsMoreStableAtN5000'],1)}% more stable at N5000; "
                f"p={fmt(r['WilcoxonP'])}; q={fmt(r['BH_FDR_Q'])}."
            )
    L.append("")

    # -------------------------------------------------------------------------
    # Scale pair audit
    # -------------------------------------------------------------------------
    L.append("OBSERVED-vs-LIABILITY SCALE AUDIT")
    L.append("-" * 115)

    if scale_audit.empty:
        L.append("No matched observed/liability units.")
    else:
        for _, r in scale_audit.iterrows():
            nlab = (
                "empirical"
                if pd.isna(r["SampleSize"])
                else f"N={int(r['SampleSize'])}"
            )
            L.append(
                f"{r['Domain']} {nlab}: matched units={int(r['MatchedUnits'])}; "
                f"rho(mean h2)={fmt(r['SpearmanMeanH2_ObservedVsLiability'])}; "
                f"rho(fold SD)={fmt(r['SpearmanFoldSD_ObservedVsLiability'])}; "
                f"median SD observed={fmt(r['MedianObservedFoldSD'])}; "
                f"liability={fmt(r['MedianLiabilityFoldSD'])}."
            )
    L.append("")

    # -------------------------------------------------------------------------
    # Missing-scale audit
    # -------------------------------------------------------------------------
    L.append("MISSING-SCALE / EXCLUSION AUDIT")
    L.append("-" * 115)

    if missing_scale.empty:
        L.append("No missing scale-specific estimates were recorded.")
    else:
        ms = (
            missing_scale
            .groupby(
                ["Domain", "Scale", "Reason"],
                dropna=False,
            )
            .size()
            .reset_index(name="N")
            .sort_values(["Domain", "Scale", "N"], ascending=[True, True, False])
        )

        for _, r in ms.iterrows():
            L.append(
                f"{r['Domain']} | {r['Scale']} | {r['Reason']}: N={int(r['N'])}"
            )
    L.append("")

    # -------------------------------------------------------------------------
    # Reviewer-facing interpretation
    # -------------------------------------------------------------------------
    L.append("REVIEWER-FACING INTERPRETATION")
    L.append("-" * 115)
    L.append(
        "1. Point-estimate variation is NOT interpreted by itself as estimator failure or meaningful biological variation."
    )
    L.append(
        "2. Negative estimates are expected from unconstrained estimators when signal is weak relative to sampling variability; >1 estimates are retained as indicators of substantial uncertainty/model instability rather than biological h2 > 1."
    )
    L.append(
        "3. Every configuration-level estimate is accompanied by cross-fold stability metrics; these are descriptive because folds overlap."
    )
    L.append(
        "4. Estimator-native SE/CI are reported separately wherever the software provides them; no missing inferential SE/CI is fabricated."
    )
    L.append(
        "5. Explicit convergence information is separated from file/output missingness."
    )
    L.append(
        "6. The proportion of estimates below 0, within 0--1, and above 1 is reported on both observed and liability scales."
    )
    L.append(
        "7. Paired N=750 vs N=5000 simulation tests quantify how finite target sample size changes cross-fold stability while holding the underlying simulated trait/configuration pairing fixed."
    )
    L.append(
        "8. Formal configuration effects should be interpreted together with the exact one-factor Wilcoxon/BH-FDR analysis and the known-truth simulation analysis already reported elsewhere in the revision."
    )

    text = "\n".join(L)
    out = OUTDIR / "FinalAnalysis06_MAIN_RESULTS.txt"
    out.write_text(text, encoding="utf-8")
    return text


# =============================================================================
# 13. SAVE WORKBOOK
# =============================================================================

def save_workbook(outputs: dict[str, pd.DataFrame]):
    path = OUTDIR / "FinalAnalysis06-UncertaintyReliability.xlsx"

    try:
        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            for name, df in outputs.items():
                sheet = re.sub(r"[^A-Za-z0-9_]+", "_", name)[:31]
                df.to_excel(writer, sheet_name=sheet, index=False)
        return str(path)
    except Exception:
        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                for name, df in outputs.items():
                    sheet = re.sub(r"[^A-Za-z0-9_]+", "_", name)[:31]
                    df.to_excel(writer, sheet_name=sheet, index=False)
            return str(path)
        except Exception as e:
            return f"XLSX not written: {e}"


# =============================================================================
# 14. RUN
# =============================================================================

def main():
    print("=" * 115)
    print("FINAL ANALYSIS 06 — UNCERTAINTY / STABILITY / RELIABILITY")
    print("=" * 115)
    print("ROOT:", ROOT)
    print("OUTDIR:", OUTDIR)
    print("SciPy available:", HAVE_SCIPY)
    print()

    # -------------------------------------------------------------------------
    # File audit
    # -------------------------------------------------------------------------
    print("[1/9] Building expected-file audit ...")
    file_audit = expected_file_audit()
    print(
        "  empirical present:",
        int(
            file_audit[
                file_audit["Domain"] == "empirical"
            ]["LiabilityResultsExists"].sum()
        ),
        "/",
        int((file_audit["Domain"] == "empirical").sum()),
    )
    print(
        "  simulated present:",
        int(
            file_audit[
                file_audit["Domain"] == "simulated"
            ]["LiabilityResultsExists"].sum()
        ),
        "/",
        int((file_audit["Domain"] == "simulated").sum()),
    )
    print()

    # -------------------------------------------------------------------------
    # Raw rows
    # -------------------------------------------------------------------------
    print("[2/9] Reading Results_Liability.csv and expanding scales ...")
    raw, missing_scale = load_all_scale_rows()
    print("  scale rows:", len(raw))
    print("  missing/excluded scale records:", len(missing_scale))
    print()

    # -------------------------------------------------------------------------
    # Aggregate
    # -------------------------------------------------------------------------
    print("[3/9] Aggregating overlapping CV folds into configuration-dataset units ...")
    units = aggregate_units(raw)
    print("  units:", len(units))
    print(
        "  stability evaluable:",
        int((units["StabilityEvaluable"] == 1).sum()) if not units.empty else 0,
    )
    print()

    # -------------------------------------------------------------------------
    # Boundary
    # -------------------------------------------------------------------------
    print("[4/9] Computing <0 / 0--1 / >1 summaries ...")
    boundaries = boundary_summary(units)
    print("  boundary summary rows:", len(boundaries))
    print()

    # -------------------------------------------------------------------------
    # Stability
    # -------------------------------------------------------------------------
    print("[5/9] Computing method-level stability summaries ...")
    stability = stability_summary(units)
    print("  stability summary rows:", len(stability))
    print()

    # -------------------------------------------------------------------------
    # Tool uncertainty
    # -------------------------------------------------------------------------
    print("[6/9] Auditing estimator-native SE / CI availability ...")
    uncertainty = uncertainty_availability(units)
    print("  uncertainty summary rows:", len(uncertainty))
    print()

    # -------------------------------------------------------------------------
    # Convergence
    # -------------------------------------------------------------------------
    print("[7/9] Auditing explicit convergence / warning information ...")
    convergence = convergence_summary(raw)
    print("  convergence summary rows:", len(convergence))
    print()

    # -------------------------------------------------------------------------
    # N effect + scale audit
    # -------------------------------------------------------------------------
    print("[8/9] Testing simulated N=750 vs N=5000 stability and scale pairing ...")
    sample_n = sample_size_stability(units)
    scale_audit = scale_pair_audit(units)
    print("  sample-size stability rows:", len(sample_n))
    print("  scale-audit rows:", len(scale_audit))
    print()

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    print("[9/9] Saving CSV / XLSX / main-results summary ...")

    outputs = {
        "raw_scale_rows": raw,
        "file_audit": file_audit,
        "unit_stability": units,
        "boundary_summary": boundaries,
        "stability_summary": stability,
        "uncertainty_availability": uncertainty,
        "convergence_summary": convergence,
        "sample_size_stability": sample_n,
        "scale_pair_audit": scale_audit,
        "excluded_missing_scale": missing_scale,
    }

    csv_names = {
        "raw_scale_rows": "FinalAnalysis06_raw_scale_rows.csv",
        "file_audit": "FinalAnalysis06_file_audit.csv",
        "unit_stability": "FinalAnalysis06_unit_stability.csv",
        "boundary_summary": "FinalAnalysis06_boundary_summary.csv",
        "stability_summary": "FinalAnalysis06_stability_summary.csv",
        "uncertainty_availability": "FinalAnalysis06_uncertainty_availability.csv",
        "convergence_summary": "FinalAnalysis06_convergence_summary.csv",
        "sample_size_stability": "FinalAnalysis06_sample_size_stability.csv",
        "scale_pair_audit": "FinalAnalysis06_scale_pair_audit.csv",
        "excluded_missing_scale": "FinalAnalysis06_excluded_or_missing_scale.csv",
    }

    for key, df in outputs.items():
        path = OUTDIR / csv_names[key]
        df.to_csv(path, index=False)

    xlsx_status = save_workbook(outputs)

    text = write_main_summary(
        raw=raw,
        file_audit=file_audit,
        units=units,
        boundary=boundaries,
        stability=stability,
        uncertainty=uncertainty,
        convergence=convergence,
        sample_n=sample_n,
        scale_audit=scale_audit,
        missing_scale=missing_scale,
    )

    print()
    print(text)
    print()
    print("=" * 115)
    print("OUTPUTS")
    print("=" * 115)
    print("Directory:", OUTDIR)
    for fn in csv_names.values():
        print(" -", fn)
    print(" - FinalAnalysis06_MAIN_RESULTS.txt")
    print(" -", xlsx_status)
    print()
    print("IMPORTANT:")
    print(" * Cross-fold stability != inferential uncertainty.")
    print(" * Tool-native SE/CI are separate.")
    print(" * Negative and >1 values are retained.")
    print(" * Missing file != convergence failure.")
    print(" * Liability-missing binary rows never fall back to observed/native h2.")
    print("DONE.")


if __name__ == "__main__":
    main()
