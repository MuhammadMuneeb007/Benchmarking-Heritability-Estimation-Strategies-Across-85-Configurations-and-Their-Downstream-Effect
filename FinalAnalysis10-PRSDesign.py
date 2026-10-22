 
#!/usr/bin/env python3

import os
import re
import csv
import math
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr, spearmanr, wilcoxon
except ImportError:
    raise SystemExit("ERROR: scipy is required. Run: pip install scipy")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(".")
THREADS = int(os.environ.get("PRS_THREADS", "64"))

# Current final runners use fold-local configuration indices.
# There are only 86 configurations, therefore >=100 is stale/test.
MAX_CONFIG_ID = 99

REAL = [
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

REAL_PRETTY = {
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

# Existing manuscript metadata
REAL_METADATA = [
    ["Asthma",                   733, 115, 618, 619653,   2867, "Binary",     "Sakaue2021"],
    ["Body Mass Index",          728, None,None,619653,   2866, "Continuous", "Sakaue2021"],
    ["Depression",               733, 114, 619, 619653, 545218, "Binary",     "Howard2018"],
    ["Gastro-Oesophageal Reflux",733,  45, 688, 619653,   2906, "Binary",     "Sakaue2021"],
    ["High Cholesterol",         733,  82, 651, 619653, 562302, "Binary",     "Loh2018"],
    ["Hypertension",             733, 188, 545, 619653, 619653, "Binary",     "GuindoMartnez2021"],
    ["Hypothyroidism",           733,  70, 663, 619653,   2908, "Binary",     "Sakaue2021"],
    ["Irritable Bowel Syndrome", 733,  78, 655, 619653, 568535, "Binary",     "Eijsbouts2021"],
    ["Migraine",                 733,  53, 680, 619653, 548955, "Binary",     "Jiang2021"],
    ["Osteoarthritis",           733,  94, 639, 619653, 619653, "Binary",     "McDonald2022"],
]

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.float_format", lambda x: f"{x:.5f}")


# ============================================================
# HELPERS
# ============================================================

def banner(text):
    print("\n" + "=" * 110)
    print(text)
    print("=" * 110)


def normalise(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def safe_float(x):
    try:
        if x is None or str(x).strip() == "":
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def get_col(fields, exact=None, contains=None, exclude=None):
    exact = exact or []
    contains = contains or []
    exclude = exclude or []

    norm = {normalise(c): c for c in fields}

    for x in exact:
        if normalise(x) in norm:
            return norm[normalise(x)]

    for c in fields:
        n = normalise(c)
        if all(normalise(x) in n for x in contains) and not any(
            normalise(x) in n for x in exclude
        ):
            return c

    return None


def numeric_values(rows, col):
    if col is None:
        return np.array([], dtype=float)
    return np.array([safe_float(r.get(col)) for r in rows], dtype=float)


def detect_train_test_pair(rows, fields):
    """
    Find matching train/test performance columns.
    Preference:
      full model > PRS-only > generic
      never null if an alternative exists.
    """

    candidates = []

    for test_col in fields:
        tn = normalise(test_col)
        tt = tn.split("_")

        if "test" not in tt and not tn.startswith("test"):
            continue

        tv = numeric_values(rows, test_col)
        if np.isfinite(tv).sum() == 0:
            continue

        test_base = [x for x in tt if x != "test"]

        for train_col in fields:
            rn = normalise(train_col)
            rt = rn.split("_")

            if "train" not in rt and not rn.startswith("train"):
                continue

            rv = numeric_values(rows, train_col)
            if np.isfinite(rv).sum() == 0:
                continue

            train_base = [x for x in rt if x != "train"]

            if test_base != train_base:
                continue

            joined = "_".join(test_base)

            score = 0

            if "full" in joined:
                score += 100
            if "prs" in joined:
                score += 50
            if "auc" in joined:
                score += 20
            if "r2" in joined or "r_2" in joined:
                score += 20
            if "explained" in joined:
                score += 20
            if "null" in joined:
                score -= 500

            score += min(np.isfinite(tv).sum(), np.isfinite(rv).sum())

            candidates.append((score, train_col, test_col))

    if not candidates:
        return None, None

    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1], candidates[0][2]


def metric_name_from_column(col):
    n = normalise(col)
    if "auc" in n:
        return "AUC"
    if "r2" in n or "r_2" in n or "explained" in n:
        return "R2"
    return col


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)

    valid = np.isfinite(p)
    if valid.sum() == 0:
        return out

    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]

    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)

    tmp = np.empty(m)
    tmp[order] = q
    out[np.where(valid)[0]] = tmp

    return out


def fmt_df(df):
    if df is None or len(df) == 0:
        print("(no rows)")
    else:
        print(df.to_string(index=False))


# ============================================================
# STATIC BENCHMARK DESCRIPTION
# ============================================================

banner("REVIEWER 6 — FINAL PRS BENCHMARK SPECIFICATION")

implementation = pd.DataFrame([
    {
        "PRS method": "GCTA-SBLUP",
        "External h2": "configuration/fold-specific native h2",
        "h2 role": "lambda = m * (1/h2 - 1)",
        "Parameter grid": "10 GWAS P-value thresholds, log10 1e-4 to 1",
        "Downstream SNP/LD handling": "fixed PLINK pruning/clumping",
        "Tuning used here": "best TRAIN performance only",
        "Test used for tuning": "NO",
    },
    {
        "PRS method": "LDpred2",
        "External h2": "configuration/fold-specific native h2",
        "h2 role": "direct grid h2 input",
        "Parameter grid": "10 LDpred2 polygenicity p values, log10 1e-4 to 1",
        "Downstream SNP/LD handling": "fixed matched SNP set + fold-specific LD matrix",
        "Tuning used here": "best TRAIN performance only",
        "Test used for tuning": "NO",
    },
])

fmt_df(implementation)

banner("REAL PHENOTYPE TARGET DATA CHARACTERISTICS")

realmeta = pd.DataFrame(
    REAL_METADATA,
    columns=[
        "Phenotype", "Target_N", "Cases", "Controls",
        "Post_QC_SNPs", "Common_SNPs", "Trait", "GWAS_source"
    ]
)
fmt_df(realmeta)

print(
    "\nNOTE: Discovery-GWAS total N, original GWAS SNP count, ancestry and "
    "explicit UKB discovery/target overlap must be taken from the source GWAS "
    "metadata/publications; they cannot be reconstructed reliably from PRS Results.csv files."
)

banner("SIMULATION DESIGN USED FOR PRS VALIDATION")

simdesign = pd.DataFrame([
    ["Simulated datasets", 200],
    ["Target sample sizes", "750; 5000"],
    ["Generating h2", "0.05; 0.10; 0.30; 0.50; 0.70"],
    ["Phenotype types", "continuous; binary"],
    ["Causal architectures", "100 causal SNPs; 5000 causal SNPs"],
    ["Replicates", 5],
    ["Independent discovery GWAS N", 10000],
    ["Discovery-target overlap", "none"],
])
simdesign.columns = ["Component", "Value"]
fmt_df(simdesign)


# ============================================================
# BUILD EXACT FOLD PATHS — NO RECURSIVE FIND/GLOB
# ============================================================

fold_specs = []

for phenotype in REAL:
    for fold in range(5):
        fold_specs.append({
            "dataset_class": "REAL",
            "dataset": phenotype,
            "fold": fold,
            "prs_method": "GCTA-SBLUP",
            "directory": ROOT / phenotype / f"Fold_{fold}" / "GCTA-SBLUP-PRS",
        })

        fold_specs.append({
            "dataset_class": "REAL",
            "dataset": phenotype,
            "fold": fold,
            "prs_method": "LDpred2",
            "directory": ROOT / phenotype / f"Fold_{fold}" / "LDpred2-PRS-Updated",
        })

for i in range(1, 201):
    sim = f"Sim{i:03d}"

    for fold in range(5):
        fold_specs.append({
            "dataset_class": "SIM",
            "dataset": sim,
            "fold": fold,
            "prs_method": "GCTA-SBLUP",
            "directory": ROOT / sim / f"Fold_{fold}" / "GCTA-SBLUP-PRS",
        })

        fold_specs.append({
            "dataset_class": "SIM",
            "dataset": sim,
            "fold": fold,
            "prs_method": "LDpred2",
            "directory": ROOT / sim / f"Fold_{fold}" / "LDpred2-PRS-v2",
        })


def scan_fold(spec):
    d = spec["directory"]

    found = []
    ignored_high_id = 0

    if not d.is_dir():
        return found, ignored_high_id

    try:
        with os.scandir(d) as it:
            for entry in it:

                if not entry.is_dir(follow_symlinks=False):
                    continue

                m = re.match(r"^(\d+)_(.+)$", entry.name)
                if not m:
                    continue

                config_id = int(m.group(1))
                config_label = m.group(2)

                # Remove stale global IDs and artificial test IDs.
                if config_id > MAX_CONFIG_ID:
                    ignored_high_id += 1
                    continue

                result = os.path.join(entry.path, "Results.csv")

                if os.path.isfile(result):
                    found.append({
                        **spec,
                        "config_id": config_id,
                        "config_label": config_label,
                        "result_file": result,
                    })

    except Exception:
        pass

    return found, ignored_high_id


banner("SCANNING EXACT PRS OUTPUT DIRECTORIES")

all_files = []
ignored_high = 0

with ThreadPoolExecutor(max_workers=THREADS) as ex:
    results = list(
        tqdm(
            ex.map(scan_fold, fold_specs),
            total=len(fold_specs),
            desc="Directory scan",
            unit="fold-dir",
        )
    )

for files, ignored in results:
    all_files.extend(files)
    ignored_high += ignored

print(f"\nResults.csv files located : {len(all_files):,}")
print(f"Stale/test IDs > {MAX_CONFIG_ID} ignored : {ignored_high:,}")


# ============================================================
# READ + VALIDATE EACH RESULTS.csv
# ============================================================

def read_result(item):
    rec = {
        "dataset_class": item["dataset_class"],
        "dataset": item["dataset"],
        "fold": item["fold"],
        "prs_method": item["prs_method"],
        "config_id": item["config_id"],
        "config_label": item["config_label"],
        "result_file": item["result_file"],
        "valid": False,
        "reason": "",
    }

    try:
        with open(item["result_file"], "r", newline="", errors="replace") as fh:
            reader = csv.DictReader(fh)
            fields = reader.fieldnames or []
            rows = list(reader)
    except Exception as e:
        rec["reason"] = f"READ_ERROR:{type(e).__name__}"
        return rec

    rec["rows"] = len(rows)

    # Final true pipelines must contain 10 grid rows.
    if len(rows) != 10:
        rec["reason"] = "NOT_10_ROWS"
        return rec

    # --------------------------------------------------------
    # h2
    # --------------------------------------------------------

    h2_col = get_col(
        fields,
        exact=["external_h2", "h2", "heritability"],
        contains=["h2"],
        exclude=["grid"],
    )

    grid_h2_col = get_col(
        fields,
        exact=["grid_h2"],
        contains=["grid", "h2"],
    )

    if h2_col:
        h2vals = numeric_values(rows, h2_col)
        h2vals = h2vals[np.isfinite(h2vals)]

        if len(h2vals) == 0:
            rec["reason"] = "H2_NONNUMERIC"
            return rec

        h2 = float(np.median(h2vals))

    elif grid_h2_col:
        vals = numeric_values(rows, grid_h2_col)
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            rec["reason"] = "GRID_H2_NONNUMERIC"
            return rec

        h2 = float(np.median(vals))

    else:
        rec["reason"] = "NO_H2_COLUMN"
        return rec

    rec["h2"] = h2

    # Downstream PRS eligibility.
    if not (0 < h2 < 1):
        rec["reason"] = "H2_OUTSIDE_0_1"
        return rec

    # --------------------------------------------------------
    # Structural validation + external-h2 propagation
    # --------------------------------------------------------

    propagation_ok = np.nan
    n_snps = np.nan

    if item["prs_method"] == "GCTA-SBLUP":

        param_col = get_col(
            fields,
            exact=[
                "pvalue_threshold",
                "p_value_threshold",
                "p_threshold",
                "threshold",
            ],
            contains=["threshold"],
        )

        if param_col is None:
            rec["reason"] = "NO_GCTA_THRESHOLD"
            return rec

        param_vals = numeric_values(rows, param_col)
        unique_param = np.unique(param_vals[np.isfinite(param_vals)])

        if len(unique_param) != 10:
            rec["reason"] = "GCTA_NOT_10_THRESHOLDS"
            return rec

        m_col = get_col(
            fields,
            exact=["m", "n_snps", "nsnps", "matched_snps"],
        )
        lambda_col = get_col(
            fields,
            exact=["lambda", "sblup_lambda"],
        )

        if m_col:
            mv = numeric_values(rows, m_col)
            mv = mv[np.isfinite(mv)]
            if len(mv):
                n_snps = float(np.median(mv))

        if m_col and lambda_col:
            mv = numeric_values(rows, m_col)
            lv = numeric_values(rows, lambda_col)

            ok = np.isfinite(mv) & np.isfinite(lv)

            if ok.sum():
                expected_lambda = mv[ok] * ((1.0 / h2) - 1.0)
                propagation_ok = bool(
                    np.allclose(
                        lv[ok],
                        expected_lambda,
                        rtol=1e-4,
                        atol=1e-4,
                    )
                )

    else:

        param_col = get_col(
            fields,
            exact=["grid_p"],
            contains=["grid", "p"],
            exclude=["h2"],
        )

        if param_col is None:
            rec["reason"] = "NO_LDPRED_GRID_P"
            return rec

        param_vals = numeric_values(rows, param_col)
        unique_param = np.unique(param_vals[np.isfinite(param_vals)])

        if len(unique_param) != 10:
            rec["reason"] = "LDPRED_NOT_10_GRID_P"
            return rec

        if grid_h2_col is None:
            rec["reason"] = "NO_GRID_H2"
            return rec

        gh = numeric_values(rows, grid_h2_col)
        gh = gh[np.isfinite(gh)]

        if len(gh) != 10:
            rec["reason"] = "GRID_H2_NONNUMERIC"
            return rec

        # Every LDpred2 grid model must receive the same external h2.
        if len(np.unique(np.round(gh, 10))) != 1:
            rec["reason"] = "GRID_H2_NOT_CONSTANT"
            return rec

        propagation_ok = bool(
            np.allclose(
                gh,
                h2,
                rtol=1e-6,
                atol=1e-8,
            )
        )

        for candidate in [
            "n_snps", "nsnps", "matched_snps",
            "matched_snps_count", "m",
        ]:
            c = get_col(fields, exact=[candidate])
            if c:
                vv = numeric_values(rows, c)
                vv = vv[np.isfinite(vv)]
                if len(vv):
                    n_snps = float(np.median(vv))
                    break

    # --------------------------------------------------------
    # Detect train/test metric pair
    # --------------------------------------------------------

    train_col, test_col = detect_train_test_pair(rows, fields)

    if train_col is None or test_col is None:
        rec["reason"] = "NO_MATCHED_TRAIN_TEST_METRIC"
        rec["available_columns"] = "|".join(fields)
        return rec

    train_vals = numeric_values(rows, train_col)
    test_vals = numeric_values(rows, test_col)

    ok_train = np.where(np.isfinite(train_vals))[0]

    if len(ok_train) == 0:
        rec["reason"] = "TRAIN_METRIC_NONNUMERIC"
        return rec

    # IMPORTANT:
    # Tune exclusively on TRAINING performance.
    selected_idx = ok_train[np.argmax(train_vals[ok_train])]

    selected_test = test_vals[selected_idx]

    if not np.isfinite(selected_test):
        rec["reason"] = "SELECTED_TEST_NONNUMERIC"
        return rec

    finite_test = test_vals[np.isfinite(test_vals)]

    if len(finite_test) == 0:
        rec["reason"] = "TEST_METRIC_NONNUMERIC"
        return rec

    parameter = safe_float(rows[selected_idx].get(param_col))

    rec.update({
        "train_column": train_col,
        "test_column": test_col,
        "metric": metric_name_from_column(test_col),
        "train_performance": float(train_vals[selected_idx]),
        "test_performance": float(selected_test),
        "selected_parameter": parameter,
        "grid_test_min": float(np.min(finite_test)),
        "grid_test_max": float(np.max(finite_test)),
        "grid_test_range": float(np.max(finite_test) - np.min(finite_test)),
        "grid_test_sd": float(np.std(finite_test, ddof=1))
            if len(finite_test) > 1 else 0.0,
        # Descriptive only: how much higher the best test-grid point
        # would have been than TRAIN-selected performance.
        "oracle_gap": float(np.max(finite_test) - selected_test),
        "propagation_ok": propagation_ok,
        "n_snps": n_snps,
        "valid": True,
        "reason": "VALID",
    })

    return rec


banner("READING AND VALIDATING PRS RESULTS")

with ThreadPoolExecutor(max_workers=THREADS) as ex:
    records = list(
        tqdm(
            ex.map(read_result, all_files),
            total=len(all_files),
            desc="Reading Results.csv",
            unit="file",
        )
    )

raw = pd.DataFrame(records)

if raw.empty:
    raise SystemExit("No PRS Results.csv files were found.")

banner("VALIDATION STATUS BEFORE DEDUPLICATION")

status = (
    raw.groupby(
        ["dataset_class", "prs_method", "reason"],
        dropna=False
    )
    .size()
    .reset_index(name="N")
    .sort_values(
        ["dataset_class", "prs_method", "N"],
        ascending=[True, True, False]
    )
)

fmt_df(status)


# ============================================================
# KEEP VALID CURRENT RESULTS + REMOVE DUPLICATE STALE LABELS
# ============================================================

valid = raw[raw["valid"] == True].copy()

# Same configuration may exist under both old and new IDs.
# Stable descriptor is everything after "<index>_".
valid = valid.sort_values("config_id")

valid = valid.drop_duplicates(
    subset=[
        "dataset_class",
        "dataset",
        "fold",
        "prs_method",
        "config_label",
    ],
    keep="first",
)

banner("ANALYSIS-READY RESULTS AFTER STABLE-LABEL DEDUPLICATION")

coverage = (
    valid.groupby(["dataset_class", "prs_method"])
    .agg(
        Valid_fold_configurations=("config_label", "size"),
        Datasets=("dataset", "nunique"),
        Unique_configuration_labels=("config_label", "nunique"),
        Phenotype_folds=("fold", "count"),
        H2_min=("h2", "min"),
        H2_max=("h2", "max"),
    )
    .reset_index()
)

fmt_df(coverage)


# ============================================================
# SHOW METRIC COLUMNS ACTUALLY USED
# ============================================================

banner("PERFORMANCE COLUMNS AUTOMATICALLY DETECTED")

metrics_used = (
    valid.groupby(
        ["dataset_class", "prs_method", "metric",
         "train_column", "test_column"]
    )
    .size()
    .reset_index(name="N_files")
    .sort_values(["dataset_class", "prs_method", "N_files"],
                 ascending=[True, True, False])
)

fmt_df(metrics_used)


# ============================================================
# EXTERNAL h2 PROPAGATION VALIDATION
# ============================================================

banner("EXTERNAL h2 PROPAGATION / PARAMETER VALIDATION")

prop_rows = []

for (dc, method), g in valid.groupby(["dataset_class", "prs_method"]):

    x = g["propagation_ok"]

    known = x.dropna()

    prop_rows.append({
        "Data": dc,
        "PRS_method": method,
        "Validated_files": len(g),
        "Propagation_check_available": len(known),
        "Propagation_pass": int((known == True).sum()) if len(known) else 0,
        "Propagation_fail": int((known == False).sum()) if len(known) else 0,
        "Pass_percent":
            100 * (known == True).mean()
            if len(known) else np.nan,
    })

prop = pd.DataFrame(prop_rows)
fmt_df(prop)


# ============================================================
# AGGREGATE FOLDS WITHIN CONFIGURATION
# ============================================================

def mode_value(x):
    x = list(x.dropna())
    if not x:
        return None
    return Counter(x).most_common(1)[0][0]


config = (
    valid.groupby(
        [
            "dataset_class",
            "dataset",
            "prs_method",
            "config_label",
        ]
    )
    .agg(
        metric=("metric", mode_value),
        h2=("h2", "mean"),
        train_performance=("train_performance", "mean"),
        test_performance=("test_performance", "mean"),
        n_folds=("fold", "nunique"),
        median_grid_test_range=("grid_test_range", "median"),
        median_grid_test_sd=("grid_test_sd", "median"),
        median_oracle_gap=("oracle_gap", "median"),
        median_n_snps=("n_snps", "median"),
    )
    .reset_index()
)


# ============================================================
# DATASET PERFORMANCE SUMMARY
# ============================================================

def q25(x):
    return np.nanquantile(x, 0.25)


def q75(x):
    return np.nanquantile(x, 0.75)


summary = (
    config.groupby(
        ["dataset_class", "dataset", "prs_method", "metric"]
    )
    .agg(
        Configurations=("config_label", "size"),
        Folds_min=("n_folds", "min"),
        Folds_median=("n_folds", "median"),
        Folds_max=("n_folds", "max"),
        Mean_test=("test_performance", "mean"),
        Median_test=("test_performance", "median"),
        SD_test=("test_performance", "std"),
        Q1_test=("test_performance", q25),
        Q3_test=("test_performance", q75),
        Min_test=("test_performance", "min"),
        Max_test=("test_performance", "max"),
        H2_min=("h2", "min"),
        H2_max=("h2", "max"),
    )
    .reset_index()
)

real_summary = summary[summary.dataset_class == "REAL"].copy()
real_summary["dataset"] = real_summary["dataset"].map(REAL_PRETTY)

banner("SUPPLEMENTARY TABLE S3 CANDIDATE — REAL PHENOTYPE PRS PERFORMANCE")

fmt_df(
    real_summary[
        [
            "dataset",
            "prs_method",
            "metric",
            "Configurations",
            "Folds_min",
            "Folds_median",
            "Folds_max",
            "Mean_test",
            "Median_test",
            "SD_test",
            "Q1_test",
            "Q3_test",
            "Min_test",
            "Max_test",
            "H2_min",
            "H2_max",
        ]
    ].sort_values(["dataset", "prs_method"])
)


banner("SUPPLEMENTARY TABLE S4 CANDIDATE — SIMULATED PRS PERFORMANCE, OVERALL")

sim_config = config[config.dataset_class == "SIM"].copy()

sim_overall = (
    sim_config.groupby(["prs_method", "metric"])
    .agg(
        Simulated_datasets=("dataset", "nunique"),
        Configurations=("config_label", "size"),
        Median_test=("test_performance", "median"),
        Mean_test=("test_performance", "mean"),
        SD_test=("test_performance", "std"),
        Q1_test=("test_performance", q25),
        Q3_test=("test_performance", q75),
        Median_folds=("n_folds", "median"),
        H2_min=("h2", "min"),
        H2_max=("h2", "max"),
    )
    .reset_index()
)

fmt_df(sim_overall)


banner("SIMULATED PRS PERFORMANCE — ALL 200 SIMULATION IDs")

sim_by_dataset = summary[summary.dataset_class == "SIM"].copy()

fmt_df(
    sim_by_dataset[
        [
            "dataset",
            "prs_method",
            "metric",
            "Configurations",
            "Folds_min",
            "Folds_median",
            "Folds_max",
            "Mean_test",
            "Median_test",
            "SD_test",
            "H2_min",
            "H2_max",
        ]
    ].sort_values(["dataset", "prs_method"])
)


# ============================================================
# h2 — PRS ASSOCIATIONS
# ============================================================

def correlation_table(dat):
    rows = []

    for (dataset, method, metric), g in dat.groupby(
        ["dataset", "prs_method", "metric"]
    ):
        gg = g[
            np.isfinite(g["h2"]) &
            np.isfinite(g["test_performance"])
        ]

        n = len(gg)

        pear_r = pear_p = spear_r = spear_p = np.nan

        if (
            n >= 3
            and gg["h2"].nunique() > 1
            and gg["test_performance"].nunique() > 1
        ):
            try:
                pear_r, pear_p = pearsonr(
                    gg["h2"],
                    gg["test_performance"]
                )
            except Exception:
                pass

            try:
                spear_r, spear_p = spearmanr(
                    gg["h2"],
                    gg["test_performance"]
                )
            except Exception:
                pass

        rows.append({
            "dataset": dataset,
            "prs_method": method,
            "metric": metric,
            "N_configurations": n,
            "Pearson_r": pear_r,
            "Pearson_p": pear_p,
            "Spearman_rho": spear_r,
            "Spearman_p": spear_p,
        })

    out = pd.DataFrame(rows)

    if len(out):
        out["Pearson_q_BH"] = bh_fdr(out["Pearson_p"])
        out["Spearman_q_BH"] = bh_fdr(out["Spearman_p"])

    return out


real_corr = correlation_table(
    config[config.dataset_class == "REAL"]
)
real_corr["dataset"] = real_corr["dataset"].map(REAL_PRETTY)

banner("REAL PHENOTYPES — h2 VERSUS TRAIN-SELECTED HELD-OUT PRS PERFORMANCE")

fmt_df(
    real_corr.sort_values(
        ["dataset", "prs_method"]
    )
)


sim_corr = correlation_table(
    config[config.dataset_class == "SIM"]
)

banner("SIMULATED PHENOTYPES — h2 VERSUS TRAIN-SELECTED HELD-OUT PRS PERFORMANCE")

fmt_df(
    sim_corr.sort_values(
        ["dataset", "prs_method"]
    )
)


banner("SUMMARY OF DATASET-SPECIFIC h2–PRS CORRELATIONS")

combined_corr = pd.concat([
    real_corr.assign(Data="REAL"),
    sim_corr.assign(Data="SIM"),
], ignore_index=True)

corr_summary = (
    combined_corr.groupby(["Data", "prs_method"])
    .agg(
        Dataset_tests=("Spearman_rho", "count"),
        Median_Spearman_rho=("Spearman_rho", "median"),
        Q1_Spearman_rho=("Spearman_rho", q25),
        Q3_Spearman_rho=("Spearman_rho", q75),
        FDR_significant=(
            "Spearman_q_BH",
            lambda x: int((x < 0.05).sum())
        ),
    )
    .reset_index()
)

fmt_df(corr_summary)


# ============================================================
# PARAMETER-GRID SENSITIVITY
# ============================================================

banner("SUPPLEMENTARY TABLE S5A CANDIDATE — PARAMETER-GRID SENSITIVITY")

grid_summary = (
    valid.groupby(
        ["dataset_class", "prs_method", "metric"]
    )
    .agg(
        Fold_configuration_results=("config_label", "size"),
        Median_test_range_across_grid=("grid_test_range", "median"),
        Q1_test_range_across_grid=("grid_test_range", q25),
        Q3_test_range_across_grid=("grid_test_range", q75),
        Median_test_SD_across_grid=("grid_test_sd", "median"),
        Median_oracle_gap=("oracle_gap", "median"),
    )
    .reset_index()
)

fmt_df(grid_summary)


banner("TRAINING-SELECTED PARAMETER FREQUENCIES")

for (dc, method), g in valid.groupby(
    ["dataset_class", "prs_method"]
):

    print(f"\n--- {dc} / {method} ---")

    z = (
        g["selected_parameter"]
        .dropna()
        .round(10)
        .value_counts()
        .sort_index()
        .rename_axis("Selected_parameter")
        .reset_index(name="N")
    )

    if len(z):
        z["Percent"] = 100 * z["N"] / z["N"].sum()

    fmt_df(z)


# ============================================================
# GCTA vs LDpred2 — SAME CONFIGURATION LABEL
# ============================================================

def paired_method_table(dat):

    wide = dat.pivot_table(
        index=["dataset", "config_label"],
        columns="prs_method",
        values="test_performance",
        aggfunc="first",
    ).reset_index()

    if (
        "GCTA-SBLUP" not in wide.columns
        or "LDpred2" not in wide.columns
    ):
        return pd.DataFrame()

    wide = wide.dropna(
        subset=["GCTA-SBLUP", "LDpred2"]
    )

    output = []

    for dataset, g in wide.groupby("dataset"):

        diff = (
            g["LDpred2"].to_numpy()
            - g["GCTA-SBLUP"].to_numpy()
        )

        p = np.nan

        if len(diff) >= 2:
            try:
                _, p = wilcoxon(diff)
            except Exception:
                pass

        output.append({
            "dataset": dataset,
            "Paired_configurations": len(g),
            "GCTA_mean_test": g["GCTA-SBLUP"].mean(),
            "LDpred2_mean_test": g["LDpred2"].mean(),
            "Mean_difference_LDpred2_minus_GCTA":
                diff.mean() if len(diff) else np.nan,
            "Median_difference":
                np.median(diff) if len(diff) else np.nan,
            "Wilcoxon_p": p,
        })

    out = pd.DataFrame(output)

    if len(out):
        out["Wilcoxon_q_BH"] = bh_fdr(out["Wilcoxon_p"])

    return out


real_pair = paired_method_table(
    config[config.dataset_class == "REAL"]
)

if len(real_pair):
    real_pair["dataset"] = real_pair["dataset"].map(
        REAL_PRETTY
    )

banner("SUPPLEMENTARY TABLE S5B CANDIDATE — PAIRED REAL GCTA-SBLUP vs LDpred2")

fmt_df(real_pair)


sim_pair = paired_method_table(
    config[config.dataset_class == "SIM"]
)

banner("PAIRED SIMULATED GCTA-SBLUP vs LDpred2")

fmt_df(sim_pair)


# ============================================================
# FINAL REVIEWER-6 NUMBERS
# ============================================================

banner("REVIEWER 6 — NUMBERS TO USE IN THE REVISION")

for dc in ["REAL", "SIM"]:

    print(f"\n{dc}")

    x = valid[valid.dataset_class == dc]

    for method in ["GCTA-SBLUP", "LDpred2"]:

        g = x[x.prs_method == method]

        if len(g) == 0:
            print(f"  {method}: no valid results")
            continue

        print(
            f"  {method}: "
            f"{len(g):,} valid fold-configuration results; "
            f"{g.dataset.nunique()} datasets; "
            f"{g.config_label.nunique()} configuration labels; "
            f"h2 range {g.h2.min():.6f}–{g.h2.max():.6f}"
        )


banner("INTERPRETATION RULES USED BY FINALANALYSIS10")

print("""
1. Only native heritability values satisfying 0 < h2 < 1 enter downstream PRS analysis.

2. GCTA-SBLUP and true LDpred2 are analysed separately.

3. Current configuration IDs are restricted to 0–99 because the experiment
   contains only 86 upstream configurations. This removes stale global-index
   directories and artificial high-ID test runs.

4. Results.csv must contain exactly 10 grid rows.

5. GCTA-SBLUP must contain 10 unique GWAS P-value thresholds.

6. LDpred2 must contain 10 unique polygenicity-grid p values, and grid_h2
   must be constant and equal to the supplied external h2.

7. Hyperparameter selection is performed using TRAINING performance only.
   The corresponding held-out TEST performance is then reported.
   Test performance is never used to choose the grid point.

8. Available folds are retained. A configuration does not need all five
   folds to contribute. n_folds is reported for transparency.

9. Fold results are averaged within each stable configuration before
   configuration-level correlations and GCTA-versus-LDpred2 comparisons.

10. No figures are produced by this script.
""")

 