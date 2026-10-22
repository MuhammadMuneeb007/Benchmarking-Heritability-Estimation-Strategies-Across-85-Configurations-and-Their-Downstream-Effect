 
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
    from scipy.stats import pearsonr, spearmanr, wilcoxon, norm
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
    Find a matched TRAIN/TEST performance pair.

    PRIMARY ENDPOINT:
      PRS + covariates / PCs ("prs_cov" or "full") whenever available.

    Fallback:
      PRS-only only if no full-model pair exists.

    Null/covariate-only models are never selected as the PRS endpoint.
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

            # Explicitly prioritise the full PRS + covariate/PC model.
            if "prs" in joined and "cov" in joined:
                score += 1000
            elif "full" in joined:
                score += 900
            elif "prs" in joined:
                score += 200

            if "auc" in joined:
                score += 20
            if "r2" in joined or "r_2" in joined:
                score += 20
            if "explained" in joined:
                score += 20

            # Strongly exclude null/covariate-only endpoints.
            if "null" in joined:
                score -= 5000
            if "cov" in joined and "prs" not in joined and "full" not in joined:
                score -= 3000

            score += min(np.isfinite(tv).sum(), np.isfinite(rv).sum())

            candidates.append((score, train_col, test_col))

    if not candidates:
        return None, None

    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1], candidates[0][2]


def is_primary_full_pair(train_col, test_col):
    """True when selected endpoint is PRS+covariates/PCs or an explicitly named full model."""
    t = normalise(train_col)
    e = normalise(test_col)

    def ok(s):
        return (("prs" in s and "cov" in s) or ("full" in s))

    return ok(t) and ok(e)

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
        "Grid selection": "best TRAIN setting; corresponding TEST row",
        "Test used for tuning": "NO",
    },
    {
        "PRS method": "LDpred2",
        "External h2": "configuration/fold-specific native h2",
        "h2 role": "direct grid h2 input",
        "Parameter grid": "10 LDpred2 polygenicity p values, log10 1e-4 to 1",
        "Downstream SNP/LD handling": "fixed matched SNP set + fold-specific LD matrix",
        "Grid selection": "best TRAIN setting; corresponding TEST row",
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
# FINALANALYSIS11 — BEST-TRAIN, MULTI-LEVEL h2→PRS ANALYSIS
# ============================================================

banner("ANALYSIS MODE — BEST TRAIN, MULTIPLE MODEL LEVELS")

print("""
PRIMARY SELECTION RULE
----------------------
For each dataset / fold / h2 configuration / PRS method:

  FULL:
      choose the PRS grid row with maximum Train_prs_cov
      and carry the SAME row to Test_prs_cov.

  PRS_ONLY:
      choose the PRS grid row with maximum Train_prs_only
      and carry the SAME row to Test_prs_only.

  NULL:
      no PRS hyperparameter exists.
      Train_null and Test_null must therefore be invariant across
      the 10 PRS grid rows. The script checks this explicitly.

TEST PERFORMANCE IS NEVER USED TO SELECT A PRS GRID POINT.

After fold-level selection, available folds are averaged by stable
upstream configuration identity. Correlations are then calculated
between configuration-level h2 and:

  Train
  Test
  Train_minus_Test

for:
  FULL
  PRS_ONLY
  NULL

and for:
  REAL_ALL
  SIM_ALL
  REAL_BINARY
  REAL_CONTINUOUS
  SIM_BINARY
  SIM_CONTINUOUS

Pearson r, Spearman rho, raw p, and BH-FDR q are reported.
""")


# ============================================================
# PHENOTYPE-TYPE CLASSIFICATION
# ============================================================

REAL_TYPE = {
    "asthma": "Binary",
    "body_mass_index_bmi": "Continuous",
    "depression": "Binary",
    "gastro_oesophageal_reflux_gord_gastric_reflux": "Binary",
    "high_cholesterol": "Binary",
    "hypertension": "Binary",
    "hypothyroidism_myxoedema": "Binary",
    "irritable_bowel_syndrome": "Binary",
    "migraine": "Binary",
    "osteoarthritis": "Binary",
}


def infer_fam_phenotype_type(dataset):
    """
    Infer binary/continuous phenotype directly from FAM phenotype values.

    Binary:
        <= 2 non-missing unique phenotype values.
    Continuous:
        > 2 non-missing unique values.
    """
    candidates = []
    for fold in range(5):
        candidates.extend([
            ROOT / dataset / f"Fold_{fold}" / "train_data.QC.fam",
            ROOT / dataset / f"Fold_{fold}" / "train_data.fam",
            ROOT / dataset / f"Fold_{fold}" / "test_data.fam",
        ])

    for fam in candidates:
        if not fam.is_file():
            continue

        try:
            x = pd.read_csv(
                fam,
                sep=r"\s+",
                header=None,
                usecols=[5],
            )

            y = pd.to_numeric(x.iloc[:, 0], errors="coerce")
            y = y[np.isfinite(y)]
            y = y[y != -9]

            if len(y) == 0:
                continue

            return "Binary" if len(np.unique(y)) <= 2 else "Continuous"

        except Exception:
            continue

    return "Unknown"


PHENOTYPE_TYPE_MAP = dict(REAL_TYPE)

for sim in [f"Sim{i:03d}" for i in range(1, 201)]:
    PHENOTYPE_TYPE_MAP[sim] = infer_fam_phenotype_type(sim)


phenotype_type_audit = pd.DataFrame([
    {
        "dataset_class": "REAL" if d in REAL_TYPE else "SIM",
        "dataset": d,
        "phenotype_type": t,
    }
    for d, t in PHENOTYPE_TYPE_MAP.items()
]).sort_values(
    ["dataset_class", "phenotype_type", "dataset"]
)

banner("PHENOTYPE-TYPE CLASSIFICATION AUDIT")

fmt_df(
    phenotype_type_audit.groupby(
        ["dataset_class", "phenotype_type"]
    )
    .size()
    .reset_index(name="Datasets")
)

phenotype_type_audit.to_csv(
    "Dataset_PhenotypeType_Classification.csv",
    index=False
)


# ============================================================
# PERFORMANCE COLUMN HELPERS
# ============================================================

MODEL_LEVELS = ["FULL", "PRS_ONLY", "NULL"]

PERFORMANCE_COLUMNS = {
    "FULL": {
        "train": [
            "Train_prs_cov",
            "Train_full",
            "Train_prs_covariate",
            "Train_prs_covariates",
        ],
        "test": [
            "Test_prs_cov",
            "Test_full",
            "Test_prs_covariate",
            "Test_prs_covariates",
        ],
    },
    "PRS_ONLY": {
        "train": [
            "Train_prs_only",
            "Train_prs",
        ],
        "test": [
            "Test_prs_only",
            "Test_prs",
        ],
    },
    "NULL": {
        "train": [
            "Train_null",
            "Train_cov",
            "Train_covariates",
            "Train_baseline",
        ],
        "test": [
            "Test_null",
            "Test_cov",
            "Test_covariates",
            "Test_baseline",
        ],
    },
}


def first_exact_col(fields, candidates):
    norm = {normalise(c): c for c in fields}
    for candidate in candidates:
        x = normalise(candidate)
        if x in norm:
            return norm[x]
    return None


def select_best_train(rows, train_col, test_col, param_col):
    """
    Select only from TRAIN.

    IMPORTANT:
      1. Find the maximum finite TRAIN performance.
      2. Do NOT inspect TEST when choosing the row.
      3. Read TEST from the exact same selected row.
    """

    train = numeric_values(rows, train_col)
    test = numeric_values(rows, test_col)

    train_ok = np.where(np.isfinite(train))[0]

    if len(train_ok) == 0:
        return {
            "ok": False,
            "reason": "NO_FINITE_TRAIN",
        }

    best_idx = int(train_ok[np.argmax(train[train_ok])])

    if not np.isfinite(test[best_idx]):
        return {
            "ok": False,
            "reason": "SELECTED_TEST_NONNUMERIC",
            "selected_idx": best_idx,
        }

    parameter = (
        safe_float(rows[best_idx].get(param_col))
        if param_col is not None
        else np.nan
    )

    return {
        "ok": True,
        "reason": "OK",
        "selected_idx": best_idx,
        "selected_parameter": parameter,
        "train": float(train[best_idx]),
        "test": float(test[best_idx]),
        "gap": float(train[best_idx] - test[best_idx]),
        "finite_train_rows": int(np.isfinite(train).sum()),
        "finite_test_rows": int(np.isfinite(test).sum()),
        "train_grid_min": float(np.nanmin(train)),
        "train_grid_max": float(np.nanmax(train)),
        "test_grid_min": float(np.nanmin(test)),
        "test_grid_max": float(np.nanmax(test)),
    }


def collapse_null(rows, train_col, test_col):
    """
    Null/covariate-only model has no PRS hyperparameter.

    It should therefore be exactly invariant across the 10 PRS-grid rows.
    We verify that rather than selecting a 'best' null row.
    """

    train = numeric_values(rows, train_col)
    test = numeric_values(rows, test_col)

    tr = train[np.isfinite(train)]
    te = test[np.isfinite(test)]

    if len(tr) == 0 or len(te) == 0:
        return {
            "ok": False,
            "reason": "NULL_NONNUMERIC",
        }

    train_constant = bool(
        np.allclose(tr, tr[0], rtol=1e-10, atol=1e-12)
    )
    test_constant = bool(
        np.allclose(te, te[0], rtol=1e-10, atol=1e-12)
    )

    if not (train_constant and test_constant):
        return {
            "ok": False,
            "reason": "NULL_VARIES_ACROSS_PRS_GRID",
            "train_range": float(np.max(tr) - np.min(tr)),
            "test_range": float(np.max(te) - np.min(te)),
        }

    train_value = float(tr[0])
    test_value = float(te[0])

    return {
        "ok": True,
        "reason": "OK",
        "train": train_value,
        "test": test_value,
        "gap": train_value - test_value,
        "train_range": 0.0,
        "test_range": 0.0,
    }


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
        with open(
            item["result_file"],
            "r",
            newline="",
            errors="replace",
        ) as fh:
            reader = csv.DictReader(fh)
            fields = reader.fieldnames or []
            rows = list(reader)
    except Exception as e:
        rec["reason"] = f"READ_ERROR:{type(e).__name__}"
        return rec

    rec["rows"] = len(rows)

    # True final pipelines have exactly 10 PRS-grid rows.
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

    # Same downstream eligibility rule used previously.
    if not (0 < h2 < 1):
        rec["reason"] = "H2_OUTSIDE_0_1"
        return rec

    # --------------------------------------------------------
    # Validate PRS parameter grid and h2 propagation
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
        unique_param = np.unique(
            param_vals[np.isfinite(param_vals)]
        )

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
                expected_lambda = (
                    mv[ok] * ((1.0 / h2) - 1.0)
                )

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
        unique_param = np.unique(
            param_vals[np.isfinite(param_vals)]
        )

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
            "n_snps",
            "nsnps",
            "matched_snps",
            "matched_snps_count",
            "m",
        ]:
            c = get_col(fields, exact=[candidate])
            if c:
                vv = numeric_values(rows, c)
                vv = vv[np.isfinite(vv)]
                if len(vv):
                    n_snps = float(np.median(vv))
                    break

    rec["propagation_ok"] = propagation_ok
    rec["n_snps"] = n_snps

    phenotype_type = PHENOTYPE_TYPE_MAP.get(
        item["dataset"],
        "Unknown",
    )

    rec["phenotype_type"] = phenotype_type
    rec["metric"] = (
        "AUC" if phenotype_type == "Binary"
        else "R2" if phenotype_type == "Continuous"
        else "Unknown"
    )

    # --------------------------------------------------------
    # FULL = PRS + covariates/PCs
    # --------------------------------------------------------

    full_train_col = first_exact_col(
        fields,
        PERFORMANCE_COLUMNS["FULL"]["train"],
    )
    full_test_col = first_exact_col(
        fields,
        PERFORMANCE_COLUMNS["FULL"]["test"],
    )

    if full_train_col is None or full_test_col is None:
        rec["full_ok"] = False
        rec["full_reason"] = "FULL_COLUMNS_MISSING"
    else:
        x = select_best_train(
            rows,
            full_train_col,
            full_test_col,
            param_col,
        )

        rec["full_ok"] = bool(x["ok"])
        rec["full_reason"] = x["reason"]
        rec["full_train_column"] = full_train_col
        rec["full_test_column"] = full_test_col

        if x["ok"]:
            rec["full_train"] = x["train"]
            rec["full_test"] = x["test"]
            rec["full_gap"] = x["gap"]
            rec["full_selected_parameter"] = x[
                "selected_parameter"
            ]
            rec["full_selected_grid_index"] = x[
                "selected_idx"
            ]
            rec["full_finite_train_rows"] = x[
                "finite_train_rows"
            ]
            rec["full_finite_test_rows"] = x[
                "finite_test_rows"
            ]
            rec["full_train_grid_range"] = (
                x["train_grid_max"] -
                x["train_grid_min"]
            )
            rec["full_test_grid_range"] = (
                x["test_grid_max"] -
                x["test_grid_min"]
            )

    # --------------------------------------------------------
    # PRS-ONLY
    # --------------------------------------------------------

    prs_train_col = first_exact_col(
        fields,
        PERFORMANCE_COLUMNS["PRS_ONLY"]["train"],
    )
    prs_test_col = first_exact_col(
        fields,
        PERFORMANCE_COLUMNS["PRS_ONLY"]["test"],
    )

    if prs_train_col is None or prs_test_col is None:
        rec["prs_only_ok"] = False
        rec["prs_only_reason"] = "PRS_ONLY_COLUMNS_MISSING"
    else:
        x = select_best_train(
            rows,
            prs_train_col,
            prs_test_col,
            param_col,
        )

        rec["prs_only_ok"] = bool(x["ok"])
        rec["prs_only_reason"] = x["reason"]
        rec["prs_only_train_column"] = prs_train_col
        rec["prs_only_test_column"] = prs_test_col

        if x["ok"]:
            rec["prs_only_train"] = x["train"]
            rec["prs_only_test"] = x["test"]
            rec["prs_only_gap"] = x["gap"]
            rec["prs_only_selected_parameter"] = x[
                "selected_parameter"
            ]
            rec["prs_only_selected_grid_index"] = x[
                "selected_idx"
            ]
            rec["prs_only_finite_train_rows"] = x[
                "finite_train_rows"
            ]
            rec["prs_only_finite_test_rows"] = x[
                "finite_test_rows"
            ]
            rec["prs_only_train_grid_range"] = (
                x["train_grid_max"] -
                x["train_grid_min"]
            )
            rec["prs_only_test_grid_range"] = (
                x["test_grid_max"] -
                x["test_grid_min"]
            )

    # --------------------------------------------------------
    # NULL / COVARIATE-ONLY BASELINE
    # --------------------------------------------------------

    null_train_col = first_exact_col(
        fields,
        PERFORMANCE_COLUMNS["NULL"]["train"],
    )
    null_test_col = first_exact_col(
        fields,
        PERFORMANCE_COLUMNS["NULL"]["test"],
    )

    if null_train_col is None or null_test_col is None:
        rec["null_ok"] = False
        rec["null_reason"] = "NULL_COLUMNS_MISSING"
    else:
        x = collapse_null(
            rows,
            null_train_col,
            null_test_col,
        )

        rec["null_ok"] = bool(x["ok"])
        rec["null_reason"] = x["reason"]
        rec["null_train_column"] = null_train_col
        rec["null_test_column"] = null_test_col

        if x["ok"]:
            rec["null_train"] = x["train"]
            rec["null_test"] = x["test"]
            rec["null_gap"] = x["gap"]
            rec["null_train_grid_range"] = 0.0
            rec["null_test_grid_range"] = 0.0
        else:
            rec["null_train_grid_range"] = x.get(
                "train_range",
                np.nan,
            )
            rec["null_test_grid_range"] = x.get(
                "test_range",
                np.nan,
            )

    # A Results.csv is retained when its h2/PRS structure is valid and
    # at least one performance level is analyzable.
    any_level_ok = any([
        rec.get("full_ok", False),
        rec.get("prs_only_ok", False),
        rec.get("null_ok", False),
    ])

    if not any_level_ok:
        rec["reason"] = (
            "NO_ANALYZABLE_PERFORMANCE_LEVEL:"
            f"FULL={rec.get('full_reason')};"
            f"PRS={rec.get('prs_only_reason')};"
            f"NULL={rec.get('null_reason')}"
        )
        return rec

    rec["valid"] = True
    rec["reason"] = "VALID"

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
    raise SystemExit(
        "No PRS Results.csv files were found."
    )


# ============================================================
# VALIDATION STATUS
# ============================================================

banner("VALIDATION STATUS BEFORE DEDUPLICATION")

status = (
    raw.groupby(
        ["dataset_class", "prs_method", "reason"],
        dropna=False,
    )
    .size()
    .reset_index(name="N")
    .sort_values(
        ["dataset_class", "prs_method", "N"],
        ascending=[True, True, False],
    )
)

fmt_df(status)


# ============================================================
# KEEP VALID CURRENT RESULTS + DEDUPLICATE STABLE LABELS
# ============================================================

valid = raw[raw["valid"] == True].copy()

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
    valid.groupby(
        ["dataset_class", "prs_method"]
    )
    .agg(
        Valid_fold_configurations=("config_label", "size"),
        Datasets=("dataset", "nunique"),
        Unique_configuration_labels=(
            "config_label",
            "nunique",
        ),
        H2_min=("h2", "min"),
        H2_max=("h2", "max"),
    )
    .reset_index()
)

fmt_df(coverage)


# ============================================================
# EXTERNAL h2 PROPAGATION VALIDATION
# ============================================================

banner("EXTERNAL h2 PROPAGATION / PARAMETER VALIDATION")

prop_rows = []

for (dc, method), g in valid.groupby(
    ["dataset_class", "prs_method"]
):
    known = g["propagation_ok"].dropna()

    prop_rows.append({
        "Data": dc,
        "PRS_method": method,
        "Validated_files": len(g),
        "Propagation_check_available": len(known),
        "Propagation_pass":
            int((known == True).sum())
            if len(known) else 0,
        "Propagation_fail":
            int((known == False).sum())
            if len(known) else 0,
        "Pass_percent":
            100 * (known == True).mean()
            if len(known) else np.nan,
    })

prop = pd.DataFrame(prop_rows)
fmt_df(prop)


# ============================================================
# MODEL-LEVEL AVAILABILITY / COLUMN AUDIT
# ============================================================

level_audit_rows = []

for (dc, method), g in valid.groupby(
    ["dataset_class", "prs_method"]
):

    for level, prefix_name in [
        ("FULL", "full"),
        ("PRS_ONLY", "prs_only"),
        ("NULL", "null"),
    ]:

        ok_col = f"{prefix_name}_ok"
        reason_col = f"{prefix_name}_reason"

        # Explicit boolean masks are required here because pandas may keep
        # these columns as object dtype after DataFrame construction.
        # Applying "~" directly to an object-dtype Python bool can produce
        # integers (-2/-1) instead of a boolean mask.
        ok_mask = g[ok_col].eq(True)
        bad_mask = ~ok_mask

        failures = (
            g.loc[bad_mask, reason_col]
            .dropna()
            .astype(str)
            .value_counts()
        )

        level_audit_rows.append({
            "Data": dc,
            "PRS_method": method,
            "Model_level": level,
            "Files_total": len(g),
            "Files_valid_level":
                int(ok_mask.sum()),
            "Files_invalid_level":
                int(bad_mask.sum()),
            "Most_common_failure":
                failures.index[0]
                if len(failures)
                else "",
        })

level_audit = pd.DataFrame(level_audit_rows)

banner("MODEL-LEVEL PERFORMANCE AVAILABILITY")
fmt_df(level_audit)

level_audit.to_csv(
    "ModelLevel_Performance_Availability.csv",
    index=False,
)


# ============================================================
# NULL MODEL INVARIANCE AUDIT
# ============================================================

banner("NULL MODEL GRID-INVARIANCE AUDIT")

null_audit = (
    valid.groupby(
        ["dataset_class", "prs_method"]
    )
    .agg(
        Files=("result_file", "size"),
        Null_valid=(
            "null_ok",
            lambda x: int(
                pd.Series(x).eq(True).sum()
            ),
        ),
        Null_invalid=(
            "null_ok",
            lambda x: int(
                pd.Series(x).ne(True).sum()
            ),
        ),
        Max_train_grid_range=(
            "null_train_grid_range",
            "max",
        ),
        Max_test_grid_range=(
            "null_test_grid_range",
            "max",
        ),
    )
    .reset_index()
)

fmt_df(null_audit)


# ============================================================
# CONVERT FOLD RESULTS TO LONG MODEL-LEVEL FORM
# ============================================================

fold_long_rows = []

for _, r in valid.iterrows():

    common = {
        "dataset_class": r["dataset_class"],
        "dataset": r["dataset"],
        "phenotype_type": r["phenotype_type"],
        "fold": int(r["fold"]),
        "prs_method": r["prs_method"],
        "config_id": int(r["config_id"]),
        "config_label": r["config_label"],
        "h2": float(r["h2"]),
        "metric": r["metric"],
        "n_snps": r.get("n_snps", np.nan),
        "result_file": r["result_file"],
    }

    for level, px in [
        ("FULL", "full"),
        ("PRS_ONLY", "prs_only"),
        ("NULL", "null"),
    ]:

        if not bool(r.get(f"{px}_ok", False)):
            continue

        fold_long_rows.append({
            **common,
            "model_level": level,
            "train_performance": float(
                r[f"{px}_train"]
            ),
            "test_performance": float(
                r[f"{px}_test"]
            ),
            "generalisation_gap": float(
                r[f"{px}_gap"]
            ),
            "selected_parameter":
                safe_float(
                    r.get(
                        f"{px}_selected_parameter",
                        np.nan,
                    )
                ),
            "selected_grid_index":
                safe_float(
                    r.get(
                        f"{px}_selected_grid_index",
                        np.nan,
                    )
                ),
        })


fold_long = pd.DataFrame(fold_long_rows)

if fold_long.empty:
    raise SystemExit(
        "No analyzable model-level fold results."
    )

fold_long["phenotype_group"] = (
    fold_long["dataset_class"]
    + "_"
    + fold_long["phenotype_type"].str.upper()
)


banner("FOLD-LEVEL RESULTS AFTER BEST-TRAIN SELECTION")

fmt_df(
    fold_long.groupby(
        ["dataset_class", "phenotype_type",
         "prs_method", "model_level"]
    )
    .agg(
        Fold_configuration_results=("config_label", "size"),
        Datasets=("dataset", "nunique"),
        H2_min=("h2", "min"),
        H2_max=("h2", "max"),
    )
    .reset_index()
)

fold_long.to_csv(
    "BestTrain_FoldLevel_SelectedResults.csv",
    index=False,
)


# ============================================================
# AGGREGATE FOLDS BY STABLE h2 CONFIGURATION
# ============================================================

def mode_value(x):
    x = list(pd.Series(x).dropna())
    if not x:
        return np.nan
    return Counter(x).most_common(1)[0][0]


config = (
    fold_long.groupby(
        [
            "dataset_class",
            "dataset",
            "phenotype_type",
            "phenotype_group",
            "prs_method",
            "model_level",
            "config_label",
        ]
    )
    .agg(
        metric=("metric", mode_value),
        h2=("h2", "mean"),
        train_performance=(
            "train_performance",
            "mean",
        ),
        test_performance=(
            "test_performance",
            "mean",
        ),
        generalisation_gap=(
            "generalisation_gap",
            "mean",
        ),
        n_folds=("fold", "nunique"),
        median_selected_parameter=(
            "selected_parameter",
            "median",
        ),
        median_n_snps=("n_snps", "median"),
    )
    .reset_index()
)

config.to_csv(
    "BestTrain_ConfigLevel_Values_Used_For_Correlations.csv",
    index=False,
)


banner("CONFIGURATION-LEVEL ANALYSIS DATA")

fmt_df(
    config.groupby(
        [
            "dataset_class",
            "phenotype_type",
            "prs_method",
            "model_level",
        ]
    )
    .agg(
        Configurations=("config_label", "size"),
        Datasets=("dataset", "nunique"),
        Folds_min=("n_folds", "min"),
        Folds_median=("n_folds", "median"),
        Folds_max=("n_folds", "max"),
    )
    .reset_index()
)


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

def q25(x):
    return np.nanquantile(x, 0.25)


def q75(x):
    return np.nanquantile(x, 0.75)


perf_summary = (
    config.groupby(
        [
            "dataset_class",
            "dataset",
            "phenotype_type",
            "prs_method",
            "model_level",
            "metric",
        ]
    )
    .agg(
        Configurations=("config_label", "size"),
        Mean_train=("train_performance", "mean"),
        Median_train=("train_performance", "median"),
        Mean_test=("test_performance", "mean"),
        Median_test=("test_performance", "median"),
        Mean_gap=("generalisation_gap", "mean"),
        Median_gap=("generalisation_gap", "median"),
        SD_test=("test_performance", "std"),
        H2_min=("h2", "min"),
        H2_max=("h2", "max"),
    )
    .reset_index()
)

real_perf = perf_summary[
    perf_summary["dataset_class"] == "REAL"
].copy()

real_perf["dataset"] = (
    real_perf["dataset"]
    .map(REAL_PRETTY)
    .fillna(real_perf["dataset"])
)

banner(
    "REAL PERFORMANCE SUMMARY — FULL / PRS_ONLY / NULL — BEST TRAIN"
)

fmt_df(
    real_perf.sort_values(
        ["model_level", "dataset", "prs_method"]
    )
)

perf_summary.to_csv(
    "BestTrain_MultiLevel_Performance_Summary.csv",
    index=False,
)


# ============================================================
# DATASET-SPECIFIC h2 CORRELATIONS
# ============================================================

ENDPOINTS = {
    "Train": "train_performance",
    "Test": "test_performance",
    "Train_minus_Test": "generalisation_gap",
}


def dataset_correlation_table(dat):
    rows = []

    for (
        dataset,
        phenotype_type,
        method,
        model_level,
        metric,
    ), g in dat.groupby(
        [
            "dataset",
            "phenotype_type",
            "prs_method",
            "model_level",
            "metric",
        ]
    ):

        for endpoint, perf_col in ENDPOINTS.items():

            gg = g[
                np.isfinite(g["h2"])
                & np.isfinite(g[perf_col])
            ].copy()

            n = len(gg)

            pear_r = pear_p = np.nan
            spear_r = spear_p = np.nan

            if (
                n >= 3
                and gg["h2"].nunique() > 1
                and gg[perf_col].nunique() > 1
            ):
                try:
                    pear_r, pear_p = pearsonr(
                        gg["h2"],
                        gg[perf_col],
                    )
                except Exception:
                    pass

                try:
                    spear_r, spear_p = spearmanr(
                        gg["h2"],
                        gg[perf_col],
                    )
                except Exception:
                    pass

            rows.append({
                "dataset": dataset,
                "phenotype_type": phenotype_type,
                "prs_method": method,
                "model_level": model_level,
                "metric": metric,
                "endpoint": endpoint,
                "N_configurations": n,
                "Pearson_r": pear_r,
                "Pearson_p": pear_p,
                "Spearman_rho": spear_r,
                "Spearman_p": spear_p,
            })

    return pd.DataFrame(rows)


real_corr = dataset_correlation_table(
    config[config["dataset_class"] == "REAL"]
)

sim_corr = dataset_correlation_table(
    config[config["dataset_class"] == "SIM"]
)

real_corr["dataset_class"] = "REAL"
sim_corr["dataset_class"] = "SIM"

corr = pd.concat(
    [real_corr, sim_corr],
    ignore_index=True,
)

corr["phenotype_group"] = (
    corr["dataset_class"]
    + "_"
    + corr["phenotype_type"].str.upper()
)


# ============================================================
# BH-FDR — OVERALL REAL / SIM FAMILIES
# ============================================================

corr["Pearson_q_BH_overall"] = np.nan
corr["Spearman_q_BH_overall"] = np.nan
corr["Pearson_FDR_sig_overall"] = False
corr["Spearman_FDR_sig_overall"] = False
corr["FDR_family_size_overall_Pearson"] = 0
corr["FDR_family_size_overall_Spearman"] = 0

# Family definition:
# dataset_class x model_level x endpoint x correlation type.
#
# REAL:
#   up to 10 phenotypes x 2 PRS methods = 20 tests/family.
#
# SIM:
#   up to 200 simulations x 2 PRS methods = 400 tests/family.

for (
    dataset_class,
    model_level,
    endpoint,
), idx in corr.groupby(
    [
        "dataset_class",
        "model_level",
        "endpoint",
    ]
).groups.items():

    idx = list(idx)

    pp = corr.loc[
        idx,
        "Pearson_p",
    ].to_numpy(dtype=float)

    sp = corr.loc[
        idx,
        "Spearman_p",
    ].to_numpy(dtype=float)

    corr.loc[
        idx,
        "Pearson_q_BH_overall",
    ] = bh_fdr(pp)

    corr.loc[
        idx,
        "Spearman_q_BH_overall",
    ] = bh_fdr(sp)

    corr.loc[
        idx,
        "FDR_family_size_overall_Pearson",
    ] = int(np.isfinite(pp).sum())

    corr.loc[
        idx,
        "FDR_family_size_overall_Spearman",
    ] = int(np.isfinite(sp).sum())

corr["Pearson_FDR_sig_overall"] = (
    corr["Pearson_q_BH_overall"] < 0.05
)

corr["Spearman_FDR_sig_overall"] = (
    corr["Spearman_q_BH_overall"] < 0.05
)


# ============================================================
# BH-FDR — WITHIN BINARY / CONTINUOUS TYPE
# ============================================================

corr["Pearson_q_BH_within_type"] = np.nan
corr["Spearman_q_BH_within_type"] = np.nan
corr["Pearson_FDR_sig_within_type"] = False
corr["Spearman_FDR_sig_within_type"] = False
corr["FDR_family_size_type_Pearson"] = 0
corr["FDR_family_size_type_Spearman"] = 0

# Family:
# phenotype_group x model_level x endpoint x correlation type.
#
# Expected maximum sizes:
# REAL_BINARY:       9 x 2 = 18
# REAL_CONTINUOUS:   1 x 2 = 2
# SIM_BINARY:      100 x 2 = 200
# SIM_CONTINUOUS:  100 x 2 = 200

for (
    phenotype_group,
    model_level,
    endpoint,
), idx in corr.groupby(
    [
        "phenotype_group",
        "model_level",
        "endpoint",
    ]
).groups.items():

    idx = list(idx)

    pp = corr.loc[
        idx,
        "Pearson_p",
    ].to_numpy(dtype=float)

    sp = corr.loc[
        idx,
        "Spearman_p",
    ].to_numpy(dtype=float)

    corr.loc[
        idx,
        "Pearson_q_BH_within_type",
    ] = bh_fdr(pp)

    corr.loc[
        idx,
        "Spearman_q_BH_within_type",
    ] = bh_fdr(sp)

    corr.loc[
        idx,
        "FDR_family_size_type_Pearson",
    ] = int(np.isfinite(pp).sum())

    corr.loc[
        idx,
        "FDR_family_size_type_Spearman",
    ] = int(np.isfinite(sp).sum())

corr["Pearson_FDR_sig_within_type"] = (
    corr["Pearson_q_BH_within_type"] < 0.05
)

corr["Spearman_FDR_sig_within_type"] = (
    corr["Spearman_q_BH_within_type"] < 0.05
)


# Pretty labels for real datasets only.
corr["dataset_pretty"] = corr["dataset"]
is_real = corr["dataset_class"] == "REAL"

corr.loc[
    is_real,
    "dataset_pretty",
] = (
    corr.loc[is_real, "dataset"]
    .map(REAL_PRETTY)
    .fillna(corr.loc[is_real, "dataset"])
)


corr.to_csv(
    "BestTrain_MultiLevel_DatasetSpecific_Correlations_FDR.csv",
    index=False,
)


# ============================================================
# PRIMARY TABLE — FULL MODEL, REAL DATA
# ============================================================

primary_real_full = corr[
    (corr["dataset_class"] == "REAL")
    & (corr["model_level"] == "FULL")
].copy()

banner(
    "PRIMARY ANALYSIS — REAL FULL MODEL — "
    "BEST TRAIN → SAME TEST — TRAIN / TEST / GAP"
)

fmt_df(
    primary_real_full[
        [
            "endpoint",
            "dataset_pretty",
            "phenotype_type",
            "prs_method",
            "metric",
            "N_configurations",
            "Pearson_r",
            "Pearson_p",
            "Pearson_q_BH_overall",
            "Pearson_FDR_sig_overall",
            "Spearman_rho",
            "Spearman_p",
            "Spearman_q_BH_overall",
            "Spearman_FDR_sig_overall",
            "FDR_family_size_overall_Pearson",
        ]
    ].sort_values(
        ["endpoint", "dataset_pretty", "prs_method"]
    )
)


# ============================================================
# FULL REAL MULTI-LEVEL TABLE
# ============================================================

banner(
    "REAL DATA — FULL / PRS_ONLY / NULL — "
    "ALL TRAIN / TEST / GAP CORRELATIONS"
)

fmt_df(
    corr[
        corr["dataset_class"] == "REAL"
    ][
        [
            "model_level",
            "endpoint",
            "dataset_pretty",
            "phenotype_type",
            "prs_method",
            "N_configurations",
            "Pearson_r",
            "Pearson_p",
            "Pearson_q_BH_overall",
            "Pearson_FDR_sig_overall",
            "Spearman_rho",
            "Spearman_p",
            "Spearman_q_BH_overall",
            "Spearman_FDR_sig_overall",
        ]
    ].sort_values(
        [
            "model_level",
            "endpoint",
            "dataset_pretty",
            "prs_method",
        ]
    )
)


# ============================================================
# MASTER SUMMARY BY REQUESTED ANALYSIS SCOPE
# ============================================================

SCOPE_FILTERS = {
    "REAL_ALL":
        corr["dataset_class"].eq("REAL"),

    "SIM_ALL":
        corr["dataset_class"].eq("SIM"),

    "REAL_BINARY":
        corr["phenotype_group"].eq("REAL_BINARY"),

    "REAL_CONTINUOUS":
        corr["phenotype_group"].eq("REAL_CONTINUOUS"),

    "SIM_BINARY":
        corr["phenotype_group"].eq("SIM_BINARY"),

    "SIM_CONTINUOUS":
        corr["phenotype_group"].eq("SIM_CONTINUOUS"),
}


summary_rows = []

for scope, mask in SCOPE_FILTERS.items():

    d = corr[mask].copy()

    for (
        model_level,
        endpoint,
        method,
    ), g in d.groupby(
        [
            "model_level",
            "endpoint",
            "prs_method",
        ]
    ):

        # For binary/continuous scopes use within-type FDR.
        if scope in [
            "REAL_BINARY",
            "REAL_CONTINUOUS",
            "SIM_BINARY",
            "SIM_CONTINUOUS",
        ]:
            pq = "Pearson_q_BH_within_type"
            sq = "Spearman_q_BH_within_type"
        else:
            pq = "Pearson_q_BH_overall"
            sq = "Spearman_q_BH_overall"

        summary_rows.append({
            "scope": scope,
            "model_level": model_level,
            "endpoint": endpoint,
            "prs_method": method,
            "Datasets": g["dataset"].nunique(),
            "Median_Pearson_r":
                g["Pearson_r"].median(),
            "Q1_Pearson_r":
                g["Pearson_r"].quantile(0.25),
            "Q3_Pearson_r":
                g["Pearson_r"].quantile(0.75),
            "Positive_Pearson":
                int((g["Pearson_r"] > 0).sum()),
            "Negative_Pearson":
                int((g["Pearson_r"] < 0).sum()),
            "Pearson_FDR_significant":
                int((g[pq] < 0.05).sum()),
            "Median_Spearman_rho":
                g["Spearman_rho"].median(),
            "Q1_Spearman_rho":
                g["Spearman_rho"].quantile(0.25),
            "Q3_Spearman_rho":
                g["Spearman_rho"].quantile(0.75),
            "Positive_Spearman":
                int((g["Spearman_rho"] > 0).sum()),
            "Negative_Spearman":
                int((g["Spearman_rho"] < 0).sum()),
            "Spearman_FDR_significant":
                int((g[sq] < 0.05).sum()),
        })


master_summary = pd.DataFrame(summary_rows)

banner(
    "MASTER SUMMARY — REAL/SIM × BINARY/CONTINUOUS × "
    "FULL/PRS_ONLY/NULL × TRAIN/TEST/GAP"
)

fmt_df(
    master_summary.sort_values(
        [
            "scope",
            "model_level",
            "endpoint",
            "prs_method",
        ]
    )
)

master_summary.to_csv(
    "BestTrain_MultiLevel_Master_Summary.csv",
    index=False,
)


# ============================================================
# RANDOM-EFFECTS META-ANALYSIS OF DATASET-SPECIFIC CORRELATIONS
# ============================================================

def random_effects_meta_r(r_values, n_values):
    """
    Random-effects meta-analysis using Fisher-z transformed
    dataset-specific correlations.
    """

    r = np.asarray(r_values, dtype=float)
    n = np.asarray(n_values, dtype=float)

    ok = (
        np.isfinite(r)
        & np.isfinite(n)
        & (n > 3)
        & (np.abs(r) < 1)
    )

    r = r[ok]
    n = n[ok]

    if len(r) == 0:
        return {
            "Meta_r": np.nan,
            "CI95_low": np.nan,
            "CI95_high": np.nan,
            "Meta_p": np.nan,
            "tau2": np.nan,
            "I2_percent": np.nan,
            "Datasets": 0,
            "Total_configurations": 0,
            "Meta_z_fisher": np.nan,
            "Meta_SE_fisher": np.nan,
        }

    r = np.clip(
        r,
        -0.999999,
        0.999999,
    )

    z = np.arctanh(r)
    v = 1.0 / (n - 3.0)
    w = 1.0 / v

    fixed_z = np.sum(w * z) / np.sum(w)

    if len(z) > 1:
        Q = np.sum(
            w * (z - fixed_z) ** 2
        )
        df = len(z) - 1
        C = (
            np.sum(w)
            - np.sum(w ** 2) / np.sum(w)
        )
        tau2 = (
            max(0.0, (Q - df) / C)
            if C > 0
            else 0.0
        )
        I2 = (
            max(0.0, (Q - df) / Q) * 100.0
            if Q > 0
            else 0.0
        )
    else:
        tau2 = 0.0
        I2 = 0.0

    wr = 1.0 / (v + tau2)

    meta_z = (
        np.sum(wr * z)
        / np.sum(wr)
    )

    se_z = math.sqrt(
        1.0 / np.sum(wr)
    )

    stat = (
        meta_z / se_z
        if se_z > 0
        else np.nan
    )

    p = (
        2.0 * norm.sf(abs(stat))
        if np.isfinite(stat)
        else np.nan
    )

    lo_z = meta_z - 1.96 * se_z
    hi_z = meta_z + 1.96 * se_z

    return {
        "Meta_r":
            float(np.tanh(meta_z)),
        "CI95_low":
            float(np.tanh(lo_z)),
        "CI95_high":
            float(np.tanh(hi_z)),
        "Meta_p":
            float(p),
        "tau2":
            float(tau2),
        "I2_percent":
            float(I2),
        "Datasets":
            int(len(r)),
        "Total_configurations":
            int(np.sum(n)),
        "Meta_z_fisher":
            float(meta_z),
        "Meta_SE_fisher":
            float(se_z),
    }


meta_rows = []

for scope, mask in SCOPE_FILTERS.items():

    d = corr[mask].copy()

    for (
        model_level,
        endpoint,
        method,
    ), g in d.groupby(
        [
            "model_level",
            "endpoint",
            "prs_method",
        ]
    ):

        pear = random_effects_meta_r(
            g["Pearson_r"],
            g["N_configurations"],
        )

        spear = random_effects_meta_r(
            g["Spearman_rho"],
            g["N_configurations"],
        )

        meta_rows.append({
            "scope": scope,
            "model_level": model_level,
            "endpoint": endpoint,
            "prs_method": method,

            "Pearson_meta_r":
                pear["Meta_r"],
            "Pearson_CI95_low":
                pear["CI95_low"],
            "Pearson_CI95_high":
                pear["CI95_high"],
            "Pearson_meta_p":
                pear["Meta_p"],
            "Pearson_I2_percent":
                pear["I2_percent"],
            "Pearson_datasets":
                pear["Datasets"],
            "Pearson_meta_z_fisher":
                pear["Meta_z_fisher"],
            "Pearson_meta_SE_fisher":
                pear["Meta_SE_fisher"],

            "Spearman_meta_rho":
                spear["Meta_r"],
            "Spearman_CI95_low":
                spear["CI95_low"],
            "Spearman_CI95_high":
                spear["CI95_high"],
            "Spearman_meta_p":
                spear["Meta_p"],
            "Spearman_I2_percent":
                spear["I2_percent"],
            "Spearman_datasets":
                spear["Datasets"],
            "Spearman_meta_z_fisher":
                spear["Meta_z_fisher"],
            "Spearman_meta_SE_fisher":
                spear["Meta_SE_fisher"],
        })


meta = pd.DataFrame(meta_rows)

meta["Pearson_meta_q_BH"] = np.nan
meta["Spearman_meta_q_BH"] = np.nan
meta["Pearson_meta_FDR_sig"] = False
meta["Spearman_meta_FDR_sig"] = False

# Interpretable meta-analysis FDR family:
# within each scope x model_level x endpoint,
# compare the two PRS methods.
for (
    scope,
    model_level,
    endpoint,
), idx in meta.groupby(
    [
        "scope",
        "model_level",
        "endpoint",
    ]
).groups.items():

    idx = list(idx)

    pp = meta.loc[
        idx,
        "Pearson_meta_p",
    ].to_numpy(dtype=float)

    sp = meta.loc[
        idx,
        "Spearman_meta_p",
    ].to_numpy(dtype=float)

    meta.loc[
        idx,
        "Pearson_meta_q_BH",
    ] = bh_fdr(pp)

    meta.loc[
        idx,
        "Spearman_meta_q_BH",
    ] = bh_fdr(sp)


meta["Pearson_meta_FDR_sig"] = (
    meta["Pearson_meta_q_BH"] < 0.05
)

meta["Spearman_meta_FDR_sig"] = (
    meta["Spearman_meta_q_BH"] < 0.05
)


banner(
    "GROUP-LEVEL RANDOM-EFFECTS SUMMARY — "
    "MULTI-LEVEL BEST-TRAIN ANALYSIS"
)

fmt_df(
    meta[
        [
            "scope",
            "model_level",
            "endpoint",
            "prs_method",
            "Pearson_meta_r",
            "Pearson_CI95_low",
            "Pearson_CI95_high",
            "Pearson_meta_p",
            "Pearson_meta_q_BH",
            "Pearson_meta_FDR_sig",
            "Pearson_I2_percent",
            "Pearson_datasets",
            "Spearman_meta_rho",
            "Spearman_CI95_low",
            "Spearman_CI95_high",
            "Spearman_meta_p",
            "Spearman_meta_q_BH",
            "Spearman_meta_FDR_sig",
            "Spearman_I2_percent",
            "Spearman_datasets",
        ]
    ].sort_values(
        [
            "scope",
            "model_level",
            "endpoint",
            "prs_method",
        ]
    )
)

meta.to_csv(
    "BestTrain_MultiLevel_Group_MetaAnalysis_FDR.csv",
    index=False,
)


# ============================================================
# BINARY vs CONTINUOUS FORMAL CONTRASTS
# ============================================================

contrast_rows = []

for data_class in ["REAL", "SIM"]:

    b_scope = f"{data_class}_BINARY"
    c_scope = f"{data_class}_CONTINUOUS"

    for model_level in MODEL_LEVELS:

        for endpoint in ENDPOINTS:

            for method in [
                "GCTA-SBLUP",
                "LDpred2",
            ]:

                b = meta[
                    (meta["scope"] == b_scope)
                    & (
                        meta["model_level"]
                        == model_level
                    )
                    & (
                        meta["endpoint"]
                        == endpoint
                    )
                    & (
                        meta["prs_method"]
                        == method
                    )
                ]

                c = meta[
                    (meta["scope"] == c_scope)
                    & (
                        meta["model_level"]
                        == model_level
                    )
                    & (
                        meta["endpoint"]
                        == endpoint
                    )
                    & (
                        meta["prs_method"]
                        == method
                    )
                ]

                if len(b) != 1 or len(c) != 1:
                    continue

                b = b.iloc[0]
                c = c.iloc[0]

                for corr_name, zcol, secol in [
                    (
                        "Pearson",
                        "Pearson_meta_z_fisher",
                        "Pearson_meta_SE_fisher",
                    ),
                    (
                        "Spearman",
                        "Spearman_meta_z_fisher",
                        "Spearman_meta_SE_fisher",
                    ),
                ]:

                    zb = safe_float(b[zcol])
                    zc = safe_float(c[zcol])
                    seb = safe_float(b[secol])
                    sec = safe_float(c[secol])

                    if not all(
                        np.isfinite(
                            [zb, zc, seb, sec]
                        )
                    ):
                        diff = se = stat = p = np.nan
                    else:
                        diff = zb - zc
                        se = math.sqrt(
                            seb ** 2 + sec ** 2
                        )
                        stat = (
                            diff / se
                            if se > 0
                            else np.nan
                        )
                        p = (
                            2.0
                            * norm.sf(abs(stat))
                            if np.isfinite(stat)
                            else np.nan
                        )

                    contrast_rows.append({
                        "data_class":
                            data_class,
                        "model_level":
                            model_level,
                        "endpoint":
                            endpoint,
                        "prs_method":
                            method,
                        "correlation":
                            corr_name,
                        "Binary_minus_Continuous_FisherZ":
                            diff,
                        "SE_difference":
                            se,
                        "Z_statistic":
                            stat,
                        "p":
                            p,
                        "Binary_datasets":
                            int(
                                b[
                                    "Pearson_datasets"
                                    if corr_name == "Pearson"
                                    else "Spearman_datasets"
                                ]
                            ),
                        "Continuous_datasets":
                            int(
                                c[
                                    "Pearson_datasets"
                                    if corr_name == "Pearson"
                                    else "Spearman_datasets"
                                ]
                            ),
                    })


contrasts = pd.DataFrame(contrast_rows)

if len(contrasts):

    contrasts["q_BH"] = np.nan
    contrasts["FDR_significant"] = False
    contrasts["FDR_family_size"] = 0

    # FDR within:
    # data_class x model_level x endpoint x correlation
    # across the two PRS methods.
    for (
        data_class,
        model_level,
        endpoint,
        corr_name,
    ), idx in contrasts.groupby(
        [
            "data_class",
            "model_level",
            "endpoint",
            "correlation",
        ]
    ).groups.items():

        idx = list(idx)

        pv = contrasts.loc[
            idx,
            "p",
        ].to_numpy(dtype=float)

        contrasts.loc[
            idx,
            "q_BH",
        ] = bh_fdr(pv)

        contrasts.loc[
            idx,
            "FDR_family_size",
        ] = int(np.isfinite(pv).sum())

    contrasts["FDR_significant"] = (
        contrasts["q_BH"] < 0.05
    )


banner(
    "FORMAL BINARY vs CONTINUOUS CONTRASTS — "
    "FULL / PRS_ONLY / NULL — BH-FDR"
)

fmt_df(contrasts)

contrasts.to_csv(
    "BestTrain_MultiLevel_Binary_vs_Continuous_Contrasts_FDR.csv",
    index=False,
)


# ============================================================
# DIAGNOSTIC: FULL vs PRS_ONLY vs NULL TEST CORRELATION SUMMARY
# ============================================================

diagnostic = (
    master_summary[
        master_summary["endpoint"] == "Test"
    ][
        [
            "scope",
            "model_level",
            "prs_method",
            "Datasets",
            "Median_Pearson_r",
            "Pearson_FDR_significant",
            "Median_Spearman_rho",
            "Spearman_FDR_significant",
        ]
    ]
    .sort_values(
        [
            "scope",
            "prs_method",
            "model_level",
        ]
    )
)

banner(
    "ONE-TABLE TEST-ENDPOINT DIAGNOSTIC — "
    "HOW h2 RELATES TO FULL vs PRS_ONLY vs NULL"
)

fmt_df(diagnostic)

diagnostic.to_csv(
    "BestTrain_MultiLevel_TestEndpoint_Diagnostic.csv",
    index=False,
)


# ============================================================
# SELECTED PARAMETER FREQUENCY AUDIT
# ============================================================

banner("BEST-TRAIN SELECTED PARAMETER FREQUENCIES")

for (
    dc,
    method,
    level,
), g in fold_long[
    fold_long["model_level"].isin(
        ["FULL", "PRS_ONLY"]
    )
].groupby(
    [
        "dataset_class",
        "prs_method",
        "model_level",
    ]
):

    print(
        f"\n--- {dc} / {method} / {level} ---"
    )

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
        z["Percent"] = (
            100 * z["N"] / z["N"].sum()
        )

    fmt_df(z)


# ============================================================
# FINAL INTERPRETATION / AUDIT RULES
# ============================================================

banner("INTERPRETATION RULES — BEST-TRAIN MULTI-LEVEL ANALYSIS")

print("""
1. Only native h2 values satisfying 0 < h2 < 1 enter downstream analysis.

2. GCTA-SBLUP and true LDpred2 are analysed separately.

3. Results.csv must contain exactly 10 PRS-grid rows.

4. GCTA-SBLUP must contain 10 unique GWAS P-value thresholds.

5. LDpred2 must contain 10 unique polygenicity-grid p values and must
   propagate the same external h2 to every grid model.

6. FULL analysis:
      select the row with maximum Train_prs_cov.
      use Test_prs_cov from the exact same row.

7. PRS_ONLY analysis:
      select the row with maximum Train_prs_only.
      use Test_prs_only from the exact same row.

8. TEST is NEVER inspected when selecting a PRS hyperparameter.

9. NULL has no PRS hyperparameter. Train_null/Test_null must be constant
   across the ten PRS grid rows. The script checks this rather than
   selecting a 'best null' row.

10. Fold-level selected results are averaged only ACROSS FOLDS belonging
    to the same stable upstream h2 configuration. PRS-grid performances
    are NOT averaged.

11. h2 associations are calculated separately for:
      Train
      Test
      Train_minus_Test

12. Every endpoint is analysed separately for:
      FULL
      PRS_ONLY
      NULL

13. Dataset-specific Pearson and Spearman correlations are calculated for
    REAL and SIM data.

14. BH-FDR for overall REAL/SIM dataset-specific tests is applied separately
    within:
      dataset_class x model_level x endpoint x correlation type.

15. BH-FDR for phenotype-type analyses is applied separately within:
      REAL_BINARY / REAL_CONTINUOUS / SIM_BINARY / SIM_CONTINUOUS
      x model_level x endpoint x correlation type.

16. Group-level random-effects Fisher-z meta-analysis is reported for:
      REAL_ALL
      SIM_ALL
      REAL_BINARY
      REAL_CONTINUOUS
      SIM_BINARY
      SIM_CONTINUOUS

17. Formal Binary-vs-Continuous contrasts are reported separately for REAL
    and SIM data, FULL/PRS_ONLY/NULL, Train/Test/Gap, PRS method, and
    Pearson/Spearman.

18. A correlation between h2 and NULL performance is a DIAGNOSTIC control.
    NULL does not use h2 or PRS. Such an association should not be interpreted
    as a biological effect of h2; it can indicate fold/configuration coverage
    or baseline-model structure.

19. Both raw p values and BH-adjusted q values are printed/saved.

20. No figures are produced by this script.
""")


# ============================================================
# OUTPUT MANIFEST
# ============================================================

banner("FILES WRITTEN")

print("""
Dataset_PhenotypeType_Classification.csv
ModelLevel_Performance_Availability.csv
BestTrain_FoldLevel_SelectedResults.csv
BestTrain_ConfigLevel_Values_Used_For_Correlations.csv
BestTrain_MultiLevel_Performance_Summary.csv
BestTrain_MultiLevel_DatasetSpecific_Correlations_FDR.csv
BestTrain_MultiLevel_Master_Summary.csv
BestTrain_MultiLevel_Group_MetaAnalysis_FDR.csv
BestTrain_MultiLevel_Binary_vs_Continuous_Contrasts_FDR.csv
BestTrain_MultiLevel_TestEndpoint_Diagnostic.csv
""")
