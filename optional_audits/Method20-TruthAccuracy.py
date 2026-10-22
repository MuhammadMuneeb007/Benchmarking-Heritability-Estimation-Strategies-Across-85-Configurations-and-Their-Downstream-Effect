 
"""
Method20-TruthAccuracy.py

Purpose
-------
Known-truth accuracy benchmark for the simulated heritability experiment.

IMPORTANT:
- Published empirical h2 values are NOT treated as ground truth.
- Continuous simulations are directly comparable to predefined TrueH2.
- Binary simulations are included in truth-accuracy calculations ONLY if a
  liability-scale h2 column is already present in Results.csv
  (supported names: h2_liability, liability_h2, h2_liability_scale).
- Five overlapping CV folds are averaged before inferential testing.
- Simulation dataset is the inferential unit.
- BH-FDR is applied within each inferential family.

Usage
-----
python Method20-TruthAccuracy.py simulated

Optional:
python Method20-TruthAccuracy.py simulated | tee Method20_SIMULATED_FINAL.txt

Reads existing Results.csv files only.
Saves NOTHING.
"""

import os
import sys
import glob
import itertools
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# =============================================================================
# COMMAND LINE
# =============================================================================

if len(sys.argv) != 2 or sys.argv[1].strip().lower() not in {
    "sim", "simulation", "simulated"
}:
    raise SystemExit(
        "\nUSAGE:\n"
        "  python Method20-TruthAccuracy.py simulated\n"
    )

# =============================================================================
# METHODS — kept aligned with Method19
# =============================================================================

METHODS = {
    "LDpred-2-Heritability": {
        "family": "LDpred2",
        "engine": "LDpred2-LDSC",
        "grm_source": "none",
        "reference_source": "internal_LDpred2",
    },
    "GCTA-Heritability": {
        "family": "GCTA",
        "engine": "GCTA",
        "grm_source": "GCTA",
        "reference_source": "none",
    },
    "Gemma1-Heritability": {
        "family": "GEMMA",
        "engine": "GEMMA",
        "grm_source": "GEMMA",
        "reference_source": "none",
    },
    "Gemma2-Heritability": {
        "family": "GEMMA",
        "engine": "GEMMA",
        "grm_source": "GEMMA",
        "reference_source": "none",
    },
    "Gemma-LDSC-Heritability": {
        "family": "GEMMA",
        "engine": "GEMMA",
        "grm_source": "LDSC-derived",
        "reference_source": "genotype_LD_scores",
    },
    "DPR-Gemma-Heritability": {
        "family": "DPR+GEMMA",
        "engine": "GEMMA",
        "grm_source": "DPR",
        "reference_source": "none",
    },
    "LDSC-Heritability": {
        "family": "LDSC",
        "engine": "LDSC",
        "grm_source": "none",
        "reference_source": "external_EUR",
    },
    "LDAK-Calculated-Heritability": {
        "family": "LDAK",
        "engine": "SumHer",
        "grm_source": "none",
        "reference_source": "calculated_genotype_tagging",
    },
    "LDAK-Precomputed-Heritability": {
        "family": "LDAK",
        "engine": "SumHer",
        "grm_source": "none",
        "reference_source": "precomputed_tagging",
    },
    "LDSC-Genotype-Heritability": {
        "family": "LDSC",
        "engine": "LDSC",
        "grm_source": "none",
        "reference_source": "genotype_derived_LD",
    },
}

# =============================================================================
# HELPERS
# =============================================================================

def safe_str(x):
    if pd.isna(x):
        return "NA"
    return str(x).strip()

def safe_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan

def yes_no(x):
    s = safe_str(x).lower()
    if s in {"yes", "y", "true", "1"}:
        return "yes"
    if s in {"no", "n", "false", "0"}:
        return "no"
    return s

def bh_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    out = np.full(len(pvalues), np.nan)
    valid = np.where(np.isfinite(pvalues))[0]
    if len(valid) == 0:
        return out
    p = pvalues[valid]
    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    temp = np.empty(m)
    temp[order] = q
    out[valid] = temp
    return out

def wilcoxon_safe(x):
    x = pd.Series(x).dropna().astype(float).values
    if len(x) < 3:
        return np.nan
    if np.allclose(x, 0):
        return np.nan
    try:
        return float(stats.wilcoxon(x, alternative="two-sided").pvalue)
    except Exception:
        return np.nan

def bootstrap_ci(values, statistic=np.mean, n_boot=2000, seed=20260914):
    """Simple dataset-level bootstrap CI."""
    x = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sample = rng.choice(x, size=len(x), replace=True)
        vals.append(statistic(sample))
    return tuple(np.quantile(vals, [0.025, 0.975]))

# =============================================================================
# SIMULATION METADATA
# Same simulation map used in Method19.
# =============================================================================

def simulation_metadata(sim_name):
    number = int(sim_name.replace("Sim", ""))

    if 1 <= number <= 100:
        phenotype_type = "Continuous"
        if number <= 50:
            target_n = 750
            local = number
        else:
            target_n = 5000
            local = number - 50
        trait_index = local  # paired 750/5000 underlying trait: 1..50

    elif 101 <= number <= 200:
        phenotype_type = "Binary"
        if number <= 150:
            target_n = 750
            local = number - 100
        else:
            target_n = 5000
            local = number - 150
        trait_index = local  # paired 750/5000 underlying trait: 1..50

    else:
        raise ValueError(f"Unexpected simulation id: {sim_name}")

    h2_values = [0.05, 0.10, 0.30, 0.50, 0.70]
    true_h2 = h2_values[(local - 1) // 10]

    inside = (local - 1) % 10
    if inside < 5:
        causal = 100
        replicate = inside + 1
    else:
        causal = 5000
        replicate = inside - 4

    trait_key = f"{phenotype_type}_Trait{trait_index:02d}"

    return {
        "PhenotypeType": phenotype_type,
        "TargetN": target_n,
        "TrueH2": true_h2,
        "CausalSNPs": causal,
        "Replicate": replicate,
        "TraitKey": trait_key,
        "Scenario": (
            f"{phenotype_type} | N={target_n} | "
            f"h2={true_h2:.2f} | causal={causal}"
        ),
    }

# =============================================================================
# CONFIGURATION PARSER
# =============================================================================

def parse_configuration(method, row):
    model = safe_str(row.get("h2model", "NA"))
    ml = model.lower()

    clumpprune = yes_no(row.get("clumpprune", "NA"))

    if "genotype_covariate_pca" in ml:
        covariates, pca = "yes", "yes"
    elif "genotype_covariate" in ml:
        covariates, pca = "yes", "no"
    else:
        covariates, pca = "no", "no"

    if "he regression" in ml or "haseman" in ml:
        algorithm = "HE"
    elif "reml" in ml or "ai algorithm" in ml:
        algorithm = "REML"
    elif method == "GCTA-Heritability":
        algorithm = "GCTA_REML"
    elif method in {"LDSC-Heritability", "LDSC-Genotype-Heritability"}:
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

    if "centered" in ml:
        grm_type = "centered"
    elif "standardized" in ml or "standardised" in ml:
        grm_type = "standardized"
    elif method == "GCTA-Heritability":
        grm_type = "GCTA"
    elif method in {"Gemma2-Heritability", "Gemma-LDSC-Heritability"}:
        grm_type = "centered"
    else:
        grm_type = "none"

    if "hapmap" in ml:
        snp_set = "hapmap"
    elif "genotyped" in ml:
        snp_set = "genotyped"
    elif method == "LDpred-2-Heritability" and "full" in ml:
        snp_set = "full"
    else:
        snp_set = "all"

    tagging_model = "NA"
    if method == "LDAK-Calculated-Heritability":
        if "ldak_human" in ml:
            tagging_model = "Human"
        elif "ldak_gcta" in ml:
            tagging_model = "GCTA"
        elif "bld-ldak" in ml:
            tagging_model = "BLD-LDAK"
        elif "alpha_" in ml:
            parts = model.split("_")
            low = [x.lower() for x in parts]
            try:
                i = low.index("alpha")
                tagging_model = "Alpha_" + parts[i + 1]
            except Exception:
                tagging_model = model

    elif method == "LDAK-Precomputed-Heritability":
        temp = ml.replace("ldak_sim_precomputed_", "").replace("ldak_", "")
        temp = temp.replace(".tagging", "")
        for token in [
            ".hapmap.gbr", ".genotyped.gbr", "hapmap.gbr", "genotyped.gbr"
        ]:
            temp = temp.replace(token, "")
        tagging_model = temp.strip("._") or "precomputed"

    reference_source = METHODS[method]["reference_source"]
    if "sim_reference5000" in ml:
        reference_source = "simulation_fixed_N5000"
    if "simulationreference" in ml:
        reference_source = "simulation_genotype_derived"
    if "sim_precomputed" in ml:
        reference_source = "simulation_precomputed_N5000"

    ld_reference = "NA"
    ld_weight = "NA"
    if method == "LDSC-Heritability":
        if "sim_reference5000" in ml:
            ld_reference = "SIM_N5000"
            ld_weight = "SIM_N5000"
        else:
            body = model[5:] if model.startswith("LDSC_") else model
            possibilities = [
                ("eur_ref_ld_chr", "eur_ref_ld_chr"),
                ("eur_ref_ld_chr", "eur_w_ld_chr"),
                ("eur_w_ld_chr", "eur_ref_ld_chr"),
                ("eur_w_ld_chr", "eur_w_ld_chr"),
            ]
            for r, w in possibilities:
                if body == f"{r}_{w}":
                    ld_reference, ld_weight = r, w
                    break

    elif method == "LDSC-Genotype-Heritability":
        ld_reference = "genotype_derived"
        ld_weight = "genotype_derived"

    return {
        "family": METHODS[method]["family"],
        "engine": METHODS[method]["engine"],
        "clumpprune": clumpprune,
        "covariates": covariates,
        "pca": pca,
        "snp_set": snp_set,
        "algorithm": algorithm,
        "grm_type": grm_type,
        "tagging_model": tagging_model,
        "grm_source": METHODS[method]["grm_source"],
        "reference_source": reference_source,
        "ld_reference": ld_reference,
        "ld_weight": ld_weight,
        "clump_p1": safe_str(row.get("clump_p1", "NA")),
        "clump_r2": safe_str(row.get("clump_r2", "NA")),
        "clump_kb": safe_str(row.get("clump_kb", "NA")),
        "p_window_size": safe_str(row.get("p_window_size", "NA")),
        "p_slide_size": safe_str(row.get("p_slide_size", "NA")),
        "p_LD_threshold": safe_str(row.get("p_LD_threshold", "NA")),
        "numberofpca": safe_str(row.get("numberofpca", "NA")),
        "raw_h2model": model,
    }

CONFIG_FIELDS = [
    "clumpprune", "covariates", "pca", "snp_set",
    "algorithm", "grm_type", "tagging_model",
    "grm_source", "reference_source", "ld_reference", "ld_weight",
    "clump_p1", "clump_r2", "clump_kb",
    "p_window_size", "p_slide_size", "p_LD_threshold",
    "numberofpca", "raw_h2model",
]

FACTOR_DEFINITIONS = [
    ("1. DATA / PREPROCESSING", "clumpprune", "Clumping / pruning", "within"),
    ("1. DATA / PREPROCESSING", "covariates", "Covariate inclusion", "within"),
    ("1. DATA / PREPROCESSING", "pca", "PCA inclusion", "within"),
    ("1. DATA / PREPROCESSING", "snp_set", "SNP / variant-set choice", "within"),
    ("2. ESTIMATOR / MODEL CONFIGURATION", "algorithm", "Estimation algorithm", "within"),
    ("2. ESTIMATOR / MODEL CONFIGURATION", "grm_type", "GRM construction", "within"),
    ("2. ESTIMATOR / MODEL CONFIGURATION", "tagging_model", "LDAK tagging / model assumption", "within"),
    ("3. INPUT / REFERENCE RESOURCE", "grm_source", "GRM source", "cross"),
    ("3. INPUT / REFERENCE RESOURCE", "ld_reference", "LDSC LD reference", "within"),
    ("3. INPUT / REFERENCE RESOURCE", "ld_weight", "LDSC regression weights", "within"),
]

LEVEL_ORDER = {
    "clumpprune": ["no", "yes"],
    "covariates": ["no", "yes"],
    "pca": ["no", "yes"],
    "snp_set": ["full", "all", "genotyped", "hapmap"],
    "algorithm": ["HE", "REML", "GCTA_REML", "LDSC", "LDpred2_LDSC", "SumHer"],
    "grm_type": ["centered", "standardized", "GCTA", "none"],
    "grm_source": ["GEMMA", "DPR", "GCTA", "LDSC-derived", "none"],
}

def ordered_levels(factor, levels):
    levels = list(dict.fromkeys(levels))
    pref = LEVEL_ORDER.get(factor)
    if pref is None:
        return sorted(levels)
    out = [x for x in pref if x in levels]
    out.extend(sorted(x for x in levels if x not in out))
    return out

# =============================================================================
# LOAD RESULTS
# =============================================================================

datasets = sorted(
    os.path.basename(x)
    for x in glob.glob("Sim[0-9][0-9][0-9]")
    if os.path.isdir(x)
)

print("=" * 120)
print("METHOD 20 — KNOWN-TRUTH HERITABILITY ACCURACY")
print("=" * 120)
print("Datasets found:", len(datasets))
print("Primary truth benchmark: simulated predefined h2")
print("Published empirical estimates are NOT treated as ground truth.")
print()

rows = []
file_audit = []

liability_columns_seen = set()

for dataset in datasets:
    meta = simulation_metadata(dataset)

    for fold in range(5):
        for method in METHODS:
            path = os.path.join(
                dataset, f"Fold_{fold}", method, "Results.csv"
            )

            if not os.path.exists(path):
                file_audit.append((dataset, fold, method, False))
                continue

            file_audit.append((dataset, fold, method, True))

            try:
                df = pd.read_csv(path)
            except Exception:
                continue

            if df.empty or "h2" not in df.columns:
                continue

            liability_col = None
            for candidate in [
                "h2_liability",
                "liability_h2",
                "h2_liability_scale",
            ]:
                if candidate in df.columns:
                    liability_col = candidate
                    liability_columns_seen.add(candidate)
                    break

            for _, row in df.iterrows():
                h2_native = safe_float(row.get("h2", np.nan))
                if not np.isfinite(h2_native):
                    continue

                cfg = parse_configuration(method, row)

                if meta["PhenotypeType"] == "Continuous":
                    h2_eval = h2_native
                    truth_comparable = True
                    scale_used = "native_continuous"
                else:
                    if liability_col is not None:
                        h2_eval = safe_float(row.get(liability_col, np.nan))
                        truth_comparable = np.isfinite(h2_eval)
                        scale_used = "liability"
                    else:
                        h2_eval = np.nan
                        truth_comparable = False
                        scale_used = "native_binary_not_comparable"

                rec = {
                    "Dataset": dataset,
                    "Fold": fold,
                    "Method": method,
                    "h2_native": h2_native,
                    "h2_eval": h2_eval,
                    "TruthComparable": truth_comparable,
                    "ScaleUsed": scale_used,
                }
                rec.update(meta)
                rec.update(cfg)
                rows.append(rec)

data = pd.DataFrame(rows)

if data.empty:
    raise SystemExit("ERROR: no valid Results.csv rows were loaded.")

file_audit = pd.DataFrame(
    file_audit,
    columns=["Dataset", "Fold", "Method", "Found"]
)

print("Results rows loaded:", len(data))
print("Methods represented:", data["Method"].nunique())
print("Files expected:", len(datasets) * 5 * len(METHODS))
print("Files found:", int(file_audit["Found"].sum()))
print("Files missing:", int((~file_audit["Found"]).sum()))
print()

n_binary = int((data["PhenotypeType"] == "Binary").sum())
n_binary_comparable = int(
    ((data["PhenotypeType"] == "Binary") & data["TruthComparable"]).sum()
)
print("Binary rows loaded:", n_binary)
print("Binary rows comparable to liability truth:", n_binary_comparable)
if n_binary and n_binary_comparable == 0:
    print(
        "NOTE: Binary native-scale estimates are NOT compared against "
        "liability-scale TrueH2. They will enter Method20 only after "
        "liability-scale harmonisation."
    )
print()

# =============================================================================
# DEFINE EXACT CONFIGURATION ID
# =============================================================================

def make_config_label(row):
    bits = [f"{field}={safe_str(row[field])}" for field in CONFIG_FIELDS]
    return " | ".join(bits)

data["ConfigLabel"] = data.apply(make_config_label, axis=1)

# =============================================================================
# AVERAGE OVER OVERLAPPING FOLDS BEFORE ACCURACY INFERENCE
# =============================================================================

group_cols = [
    "Dataset", "TraitKey", "PhenotypeType", "TargetN", "TrueH2",
    "CausalSNPs", "Replicate", "Scenario",
    "Method", "family", "engine", "ConfigLabel",
] + CONFIG_FIELDS

units = (
    data.groupby(group_cols, as_index=False, dropna=False)
    .agg(
        MeanH2=("h2_eval", "mean"),
        MeanNativeH2=("h2_native", "mean"),
        FoldSD=("h2_native", "std"),
        FoldMAD=("h2_native", lambda x: np.median(np.abs(x - np.median(x)))),
        MinH2=("h2_native", "min"),
        MaxH2=("h2_native", "max"),
        CommonFolds=("Fold", "nunique"),
        ComparableFolds=("TruthComparable", "sum"),
    )
)

units["TruthComparable"] = np.isfinite(units["MeanH2"])
acc = units[units["TruthComparable"]].copy()

if acc.empty:
    raise SystemExit(
        "ERROR: no truth-comparable units found. "
        "Continuous simulations should always be comparable."
    )

acc["Error"] = acc["MeanH2"] - acc["TrueH2"]
acc["AbsError"] = acc["Error"].abs()
acc["SqError"] = acc["Error"] ** 2
acc["BelowZero"] = acc["MeanH2"] < 0
acc["AboveOne"] = acc["MeanH2"] > 1
acc["Plausible01"] = (acc["MeanH2"] >= 0) & (acc["MeanH2"] <= 1)

print("=" * 120)
print("TRUTH-COMPARABLE CONFIGURATION × DATASET UNITS")
print("=" * 120)
print("Units:", len(acc))
print("Datasets:", acc["Dataset"].nunique())
print("Continuous datasets:", acc.loc[
    acc["PhenotypeType"] == "Continuous", "Dataset"
].nunique())
print("Binary datasets:", acc.loc[
    acc["PhenotypeType"] == "Binary", "Dataset"
].nunique())
print()

# =============================================================================
# METHOD-LEVEL ACCURACY
# Dataset is first collapsed so huge configuration families do not get
# artificial inferential weight.
# =============================================================================

method_dataset = (
    acc.groupby(
        ["Method", "family", "Dataset", "PhenotypeType", "TargetN"],
        as_index=False
    )
    .agg(
        DatasetBias=("Error", "mean"),
        DatasetMAE=("AbsError", "mean"),
        DatasetMedianAE=("AbsError", "median"),
        DatasetRMSE=("SqError", lambda x: float(np.sqrt(np.mean(x)))),
        NegativeRate=("BelowZero", "mean"),
        AboveOneRate=("AboveOne", "mean"),
    )
)

method_summary_rows = []
for (method, family, target_n), g in method_dataset.groupby(
    ["Method", "family", "TargetN"], dropna=False
):
    lo, hi = bootstrap_ci(g["DatasetMAE"], statistic=np.mean)
    method_summary_rows.append({
        "Method": method,
        "Family": family,
        "N": int(target_n),
        "Datasets": g["Dataset"].nunique(),
        "MeanBias": g["DatasetBias"].mean(),
        "MedianBias": g["DatasetBias"].median(),
        "MeanMAE": g["DatasetMAE"].mean(),
        "MedianMAE": g["DatasetMAE"].median(),
        "MAE_CI95_L": lo,
        "MAE_CI95_U": hi,
        "MeanRMSE": g["DatasetRMSE"].mean(),
        "NegativePct": 100 * g["NegativeRate"].mean(),
        "AboveOnePct": 100 * g["AboveOneRate"].mean(),
    })

method_summary = pd.DataFrame(method_summary_rows).sort_values(
    ["N", "MeanMAE", "MeanRMSE"]
)

print("=" * 120)
print("METHOD-LEVEL ACCURACY — LOWER MAE/RMSE IS BETTER")
print("=" * 120)
print(method_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print()

# =============================================================================
# SCENARIO SUMMARY
# =============================================================================

scenario_summary = (
    acc.groupby(
        ["PhenotypeType", "TargetN", "TrueH2", "CausalSNPs"],
        as_index=False
    )
    .agg(
        Units=("AbsError", "size"),
        MeanBias=("Error", "mean"),
        MAE=("AbsError", "mean"),
        MedianAE=("AbsError", "median"),
        RMSE=("SqError", lambda x: float(np.sqrt(np.mean(x)))),
        NegativePct=("BelowZero", lambda x: 100 * np.mean(x)),
        AboveOnePct=("AboveOne", lambda x: 100 * np.mean(x)),
    )
)

print("=" * 120)
print("ACCURACY BY PHENOTYPE TYPE × N × TRUE h2 × ARCHITECTURE")
print("=" * 120)
print(scenario_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print()

# =============================================================================
# TOP CONFIGURATIONS / PIPELINES BY KNOWN-TRUTH ACCURACY
# Descriptive only. Require broad representation.
# =============================================================================

config_summary = (
    acc.groupby(["Method", "family", "ConfigLabel"], as_index=False)
    .agg(
        Datasets=("Dataset", "nunique"),
        MeanBias=("Error", "mean"),
        MAE=("AbsError", "mean"),
        MedianAE=("AbsError", "median"),
        RMSE=("SqError", lambda x: float(np.sqrt(np.mean(x)))),
        FoldSD=("FoldSD", "mean"),
    )
)

min_datasets = max(10, int(0.25 * acc["Dataset"].nunique()))
eligible = config_summary[config_summary["Datasets"] >= min_datasets].copy()

print("=" * 120)
print(f"TOP 25 CONFIGURATIONS BY MAE (requires >= {min_datasets} datasets)")
print("=" * 120)
if eligible.empty:
    print("No configurations passed the representation threshold.")
else:
    print(
        eligible.sort_values(["MAE", "RMSE"])
        .head(25)
        .to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )
print()

# =============================================================================
# PAIRED N=750 vs N=5000 ACCURACY
# Same underlying trait + same method + same exact config.
# DeltaAE = AE_5000 - AE_750; negative => larger N is more accurate.
# =============================================================================

paired_rows = []
key_cols = ["TraitKey", "Method", "ConfigLabel"]

for key, g in acc.groupby(key_cols, dropna=False):
    g750 = g[g["TargetN"] == 750]
    g5000 = g[g["TargetN"] == 5000]
    if g750.empty or g5000.empty:
        continue

    ae750 = g750["AbsError"].mean()
    ae5000 = g5000["AbsError"].mean()

    paired_rows.append({
        "TraitKey": key[0],
        "Method": key[1],
        "ConfigLabel": key[2],
        "AE_750": ae750,
        "AE_5000": ae5000,
        "DeltaAE_5000_minus_750": ae5000 - ae750,
    })

paired_n = pd.DataFrame(paired_rows)

sample_tests = []
if not paired_n.empty:
    # First average repeated configuration comparisons within underlying trait
    for method, gm in paired_n.groupby("Method"):
        trait_effect = (
            gm.groupby("TraitKey", as_index=False)
            ["DeltaAE_5000_minus_750"].mean()
        )
        p = wilcoxon_safe(trait_effect["DeltaAE_5000_minus_750"])
        sample_tests.append({
            "Method": method,
            "Traits": trait_effect["TraitKey"].nunique(),
            "MedianDeltaAE": trait_effect["DeltaAE_5000_minus_750"].median(),
            "MeanDeltaAE": trait_effect["DeltaAE_5000_minus_750"].mean(),
            "RawP": p,
        })

sample_tests = pd.DataFrame(sample_tests)
if not sample_tests.empty:
    sample_tests["FDR_Q"] = bh_adjust(sample_tests["RawP"].values)
    sample_tests["FDR"] = np.where(
        sample_tests["FDR_Q"] < 0.05, "YES", "NO"
    )
    sample_tests = sample_tests.sort_values(
        ["FDR_Q", "MedianDeltaAE"], na_position="last"
    )

print("=" * 120)
print("PAIRED SAMPLE-SIZE EFFECT ON ACCURACY")
print("DeltaAE = absolute error at N=5000 minus absolute error at N=750")
print("NEGATIVE DeltaAE => N=5000 is closer to truth")
print("=" * 120)
if sample_tests.empty:
    print("No paired sample-size tests available.")
else:
    print(sample_tests.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
print()

# =============================================================================
# EXACT ONE-FACTOR MATCHED ACCURACY ANALYSIS
# Outcome is absolute error to truth, not h2 itself.
#
# DeltaError = AbsError_B - AbsError_A
# Negative => level B is MORE accurate.
# =============================================================================

factor_results = []

for category, factor, label, scope in FACTOR_DEFINITIONS:
    if factor not in acc.columns:
        continue

    temp = acc.copy()
    temp = temp[
        temp[factor].notna()
        & ~temp[factor].astype(str).str.lower().isin({"na", "nan", ""})
    ].copy()

    control = [
        "Dataset",
        "PhenotypeType",
        "TargetN",
        "TrueH2",
        "CausalSNPs",
    ]

    if scope == "within":
        control.append("Method")
    else:
        control.append("engine")

    # Match on all other parsed analytical factors.
    for field in CONFIG_FIELDS:
        if field in {factor, "raw_h2model"}:
            continue

        # If SNP set or clumping itself is the intervention, related raw
        # tuning fields should not prevent the intended contrast.
        if factor == "clumpprune" and field in {
            "clump_p1", "clump_r2", "clump_kb"
        }:
            continue

        control.append(field)

    control = list(dict.fromkeys(control))

    for col in control + [factor]:
        temp[col] = temp[col].fillna("NA").astype(str)

    pair_effects = defaultdict(list)

    for _, g in temp.groupby(control, dropna=False, sort=False):
        levels = ordered_levels(factor, g[factor].unique().tolist())
        if len(levels) < 2:
            continue

        for a, b in itertools.combinations(levels, 2):
            ga = g[g[factor] == a]
            gb = g[g[factor] == b]
            if ga.empty or gb.empty:
                continue

            ae_a = ga["AbsError"].mean()
            ae_b = gb["AbsError"].mean()
            dataset = g.iloc[0]["Dataset"]

            pair_effects[(a, b, dataset)].append(ae_b - ae_a)

    if not pair_effects:
        continue

    # Average all exact matched strata within each dataset first.
    contrast_dataset = defaultdict(list)
    for (a, b, dataset), vals in pair_effects.items():
        contrast_dataset[(a, b)].append((dataset, np.mean(vals)))

    for (a, b), values in contrast_dataset.items():
        ds = pd.DataFrame(values, columns=["Dataset", "DeltaError"])
        p = wilcoxon_safe(ds["DeltaError"])
        factor_results.append({
            "Category": category,
            "Factor": label,
            "Contrast": f"{b} - {a}",
            "Datasets": ds["Dataset"].nunique(),
            "MeanDeltaError": ds["DeltaError"].mean(),
            "MedianDeltaError": ds["DeltaError"].median(),
            "MeanAbsDeltaError": ds["DeltaError"].abs().mean(),
            "RawP": p,
        })

factor_results = pd.DataFrame(factor_results)

if not factor_results.empty:
    factor_results["FDR_Q"] = bh_adjust(factor_results["RawP"].values)
    factor_results["FDR"] = np.where(
        factor_results["FDR_Q"] < 0.05, "YES", "NO"
    )
    factor_results = factor_results.sort_values(
        ["FDR_Q", "MedianDeltaError"], na_position="last"
    )

print("=" * 120)
print("EXACT ONE-FACTOR MATCHED ACCURACY")
print("DeltaError = |h2_B - truth| - |h2_A - truth|")
print("NEGATIVE DeltaError => Level B is MORE accurate")
print("=" * 120)
if factor_results.empty:
    print("No exact one-factor accuracy contrasts found.")
else:
    print(
        factor_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}"
        )
    )
print()

# =============================================================================
# STABILITY vs ACCURACY CORRELATION — bridge to Reviewer Comment 3
# Uses configuration × dataset fold SD versus absolute truth error.
# =============================================================================

corr_rows = []

for (target_n, method), g in acc.groupby(["TargetN", "Method"]):
    gg = g[np.isfinite(g["FoldSD"]) & np.isfinite(g["AbsError"])].copy()
    if len(gg) < 10:
        continue
    rho, p = stats.spearmanr(gg["FoldSD"], gg["AbsError"])
    corr_rows.append({
        "N": int(target_n),
        "Method": method,
        "Observations": len(gg),
        "SpearmanRho": rho,
        "RawP": p,
    })

corr = pd.DataFrame(corr_rows)
if not corr.empty:
    corr["FDR_Q"] = bh_adjust(corr["RawP"].values)
    corr["FDR"] = np.where(corr["FDR_Q"] < 0.05, "YES", "NO")
    corr = corr.sort_values(["FDR_Q", "N", "Method"])

print("=" * 120)
print("STABILITY vs ACCURACY")
print("Spearman correlation: cross-fold SD vs absolute error to truth")
print("Positive rho => less stable configurations tend to be less accurate")
print("=" * 120)
if corr.empty:
    print("No correlations available.")
else:
    print(corr.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
print()

# =============================================================================
# FILE MISSINGNESS / FAILURE PROXY
# =============================================================================

file_summary = (
    file_audit.groupby("Method", as_index=False)
    .agg(
        Expected=("Found", "size"),
        Found=("Found", "sum"),
    )
)
file_summary["Missing"] = file_summary["Expected"] - file_summary["Found"]
file_summary["MissingPct"] = 100 * file_summary["Missing"] / file_summary["Expected"]

print("=" * 120)
print("METHOD FILE-MISSINGNESS AUDIT")
print("=" * 120)
print(file_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print()

print("=" * 120)
print("INTERPRETATION RULES")
print("=" * 120)
print("1. Published empirical h2 values are NOT ground truth and are not used to rank accuracy.")
print("2. Lower MAE/RMSE means closer recovery of predefined simulated h2.")
print("3. For one-factor accuracy contrasts, negative DeltaError means Level B is more accurate.")
print("4. For the paired sample-size analysis, negative DeltaAE means N=5000 is more accurate.")
print("5. Binary simulations enter truth-accuracy analysis only after liability-scale harmonisation.")
print("6. Cross-fold SD is a stability metric, NOT an inferential standard error.")
print("7. Missing Results.csv is an audit/failure proxy; explicit convergence status should be handled in Method21.")
print()
print("DONE — nothing was saved.")
 