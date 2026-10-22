"""
LDpred-2-PRS-Final.py
======================
Calculates LDpred2-lassosum2 PRS for one configuration.
Uses fold-specific h2 passed directly from the parameter line.

Usage:
  python LDpred-2-PRS-Final.py <phenotype> <fold> <config_index> <binary|continuous>
                                <h2> <method> <h2model> <clumpprune> <numberofpca>

Same argument format as calculatePRSGCTA_v2.py.

Output:
  <phenotype>/Fold_<fold>/LDpred2-PRS/<config_index>_<h2model>_<clumpprune>/Results.csv

Results.csv columns — same format as GCTA but with grid params instead of pvalue_threshold:
  phenotype, fold, config_index, method, h2model, clumpprune, numberofpca,
  h2, grid_lambda, grid_delta, grid_sparsity, metric,
  Train_null,     Test_null,
  Train_prs_only, Test_prs_only,
  Train_prs_cov,  Test_prs_cov,
  Delta_null, Delta_prs_only, Delta_prs_cov
"""

import sys
import os
import re
import pandas as pd
import numpy as np
import subprocess
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, r2_score
import warnings
warnings.filterwarnings("ignore")

# ── ARGUMENTS ─────────────────────────────────────────────────────────────────

if len(sys.argv) != 10:
    print("Usage: python LDpred-2-PRS-Final.py <phenotype> <fold> <config_index> "
          "<binary|continuous> <h2> <method> <h2model> <clumpprune> <numberofpca>")
    sys.exit(1)

filedirec      = sys.argv[1]
foldnumber     = sys.argv[2]
config_index   = int(sys.argv[3])
phenotype_type = sys.argv[4].lower()
h2             = float(sys.argv[5])
method         = sys.argv[6]
h2model        = sys.argv[7]
clumpprune     = sys.argv[8]
numberofpca    = sys.argv[9]

assert phenotype_type in ("binary", "continuous"), \
    f"phenotype_type must be binary or continuous, got {phenotype_type}"
assert 0 < h2 < 1, f"h2 must be in (0,1), got {h2}"

# ── PATHS ─────────────────────────────────────────────────────────────────────

folddirec        = os.path.join(filedirec, f"Fold_{foldnumber}")
trainfilename    = "train_data"
newtrainfilename = "train_data.QC"
testfilename     = "test_data"

safe_label = re.sub(r"[^a-zA-Z0-9_\-]", "_", f"{h2model}_{clumpprune}")
outdir = os.path.join(
    folddirec, "LDpred2-PRS-Updated",
    f"{config_index}_{safe_label}"
)
os.makedirs(outdir, exist_ok=True)

results_file = os.path.join(outdir, "Results.csv")
if os.path.exists(results_file):
    existing = pd.read_csv(results_file)
    if len(existing) > 0:
        print(f"Already complete: {results_file}. Skipping.")
        sys.exit(0)

print(f"phenotype={filedirec}  fold={foldnumber}  config={config_index}")
print(f"method={method}  h2model={h2model}  clumpprune={clumpprune}")
print(f"h2={h2:.6f} (fold-specific)  type={phenotype_type}")

# ── FIXED PARAMETERS ──────────────────────────────────────────────────────────

try:
    npca = int(numberofpca)
except (ValueError, TypeError):
    npca = 6
npca_str = str(npca)

clump_p1       = "1"
clump_r2       = "0.1"
clump_kb       = "200"
p_window_size  = "200"
p_slide_size   = "50"
p_LD_threshold = "0.25"

size_param  = "200"
alpha_param = "1"
thr_r2      = "0.1"

# Mode: 2 = hapmap SNPs, 3 = all SNPs
r_mode = "2" if "hapmap" in h2model.lower() else "3"

# ── HELPERS ───────────────────────────────────────────────────────────────────

def run(cmd):
    result = subprocess.run(
        [str(c) for c in cmd], capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  WARN: {' '.join(str(c) for c in cmd[:5])} ... returned {result.returncode}")
    return result

def calc_metric(y_true, y_pred, ptype):
    try:
        if ptype == "binary":
            return float(roc_auc_score(y_true, y_pred))
        else:
            return float(r2_score(y_true, y_pred))
    except Exception:
        return np.nan

def read_pheno(fam_path, ptype):
    fam = pd.read_table(fam_path, sep=r"\s+", header=None)
    ph  = fam[5].values.astype(float)
    if ptype == "binary":
        ph = np.where(ph == 1, 0, np.where(ph == 2, 1, np.nan))
    return ph

def read_cov_pca(cov_path, eigenvec_path, n):
    pcs = pd.read_table(
        eigenvec_path, sep=r"\s+", header=None,
        names=["FID","IID"] + [f"PC{i}" for i in range(1, n+1)]
    )
    cov = pd.read_table(cov_path, sep=r"\s+").fillna(0)
    for col in ["FID","IID"]:
        cov[col] = cov[col].astype(str)
        pcs[col] = pcs[col].astype(str)
    cov = cov[cov["FID"].isin(pcs["FID"]) & cov["IID"].isin(pcs["IID"])]
    return pd.merge(cov, pcs, on=["FID","IID"]).fillna(0)

def fit_model(X, y, ptype):
    Xc = sm.add_constant(X, has_constant="add")
    try:
        if ptype == "binary":
            return sm.Logit(y, Xc).fit(disp=0)
        else:
            return sm.OLS(y, Xc).fit()
    except Exception:
        try:
            if ptype == "binary":
                return sm.Logit(y, Xc).fit_regularized(alpha=0.1, L1_wt=0.1, disp=0)
            else:
                return sm.OLS(y, Xc).fit_regularized(L1_wt=0.5)
        except Exception as e:
            print(f"  Model failed: {e}")
            return None

# ── STEP 1: SHARED FOLD-LEVEL PREPROCESSING CACHE ─────────────────────────────

cache_dir = os.path.join(folddirec, "LDpred2-CACHE-v2")
os.makedirs(cache_dir, exist_ok=True)

global_cache = os.path.abspath("LDpred2-GLOBAL-CACHE")
os.makedirs(global_cache, exist_ok=True)

train_prefix  = os.path.join(cache_dir, trainfilename)
test_prefix   = os.path.join(cache_dir, testfilename)
pruned_bfile  = os.path.join(cache_dir, newtrainfilename + ".clumped.pruned")
valid_snp     = train_prefix + ".valid.snp"
clumped       = train_prefix + ".clumped"

cache_required = [
    pruned_bfile + ".bed",
    pruned_bfile + ".bim",
    pruned_bfile + ".fam",
    valid_snp,
    train_prefix + ".eigenvec",
    test_prefix + ".eigenvec",
]

if all(os.path.exists(x) for x in cache_required):

    print("PREPROCESS CACHE HIT -- reusing PLINK/PCA files")

else:

    print("PREPROCESS CACHE MISS -- building once")

    run([
        "./plink",
        "--bfile", os.path.join(folddirec,newtrainfilename),
        "--indep-pairwise",
        p_window_size,p_slide_size,p_LD_threshold,
        "--out",train_prefix
    ])

    run([
        "./plink",
        "--bfile",os.path.join(folddirec,newtrainfilename),
        "--clump-p1",clump_p1,
        "--extract",train_prefix+".prune.in",
        "--clump-r2",clump_r2,
        "--clump-kb",clump_kb,
        "--clump",os.path.join(filedirec,filedirec+".txt"),
        "--clump-snp-field","SNP",
        "--clump-field","P",
        "--out",train_prefix
    ])

    if not os.path.exists(clumped):
        print("Clumping produced no output.")
        sys.exit(1)

    os.system(f'awk "NR!=1{{print $3}}" {clumped} > {valid_snp}')

    run([
        "./plink",
        "--make-bed",
        "--bfile",os.path.join(folddirec,newtrainfilename),
        "--extract",valid_snp,
        "--out",pruned_bfile
    ])

    run([
        "./plink",
        "--bfile",pruned_bfile,
        "--pca",npca_str,
        "--out",train_prefix
    ])

    run([
        "./plink",
        "--bfile",os.path.join(folddirec,testfilename),
        "--extract",valid_snp,
        "--pca",npca_str,
        "--out",test_prefix
    ])

# ── STEP 2: LDPRED2 USING CACHED LD MATRIX ────────────────────────────────────

r_cmd = [
    "Rscript",
    "LDpred-2-PRS-Final-real-updated.R",
    filedirec,
    outdir,
    pruned_bfile,
    size_param,
    alpha_param,
    thr_r2,
    str(h2),
    cache_dir,
    global_cache
]

r_result = subprocess.run(r_cmd)
if r_result.returncode != 0:
    sys.exit(r_result.returncode)

# ── STEP 3: LOAD LASSOSUM2 RESULTS ────────────────────────────────────────────

grid_file  = os.path.join(outdir, "ldpred2_grid.csv")
betas_file = os.path.join(outdir, "ldpred2_betas.csv")
gwas_file  = os.path.join(outdir, "ldpred2_gwas.csv")

for f in [grid_file, betas_file, gwas_file]:
    if not os.path.exists(f):
        print(f"  Missing R output: {f}. Exiting.")
        sys.exit(0)

grid  = pd.read_csv(grid_file)
betas = pd.read_csv(betas_file).fillna(0)
gwas  = pd.read_csv(gwas_file)
print(f"  Grid points: {len(grid)}  |  SNPs: {len(gwas)}")

# ── STEP 4: LOAD PHENOTYPES AND COVARIATES ────────────────────────────────────

y_tr_all = read_pheno(os.path.join(folddirec, newtrainfilename + ".fam"), phenotype_type)
y_te_all = read_pheno(os.path.join(folddirec, testfilename + ".fam"),     phenotype_type)

cov_tr = read_cov_pca(
    os.path.join(folddirec, trainfilename + ".cov"),
    train_prefix + ".eigenvec", npca)

# Test PCA already available in fold-level cache

cov_te = read_cov_pca(
    os.path.join(folddirec, testfilename + ".cov"),
    test_prefix + ".eigenvec", npca)

n_tr  = len(cov_tr)
n_te  = len(cov_te)
y_tr  = y_tr_all[:n_tr]
y_te  = y_te_all[:n_te]
tr_ok = ~np.isnan(y_tr)
te_ok = ~np.isnan(y_te)

y_tr_c   = y_tr[tr_ok]
y_te_c   = y_te[te_ok]
X_tr_cov = cov_tr.iloc[tr_ok].reset_index(drop=True).iloc[:, 2:]
X_te_cov = cov_te.iloc[te_ok].reset_index(drop=True).iloc[:, 2:]

# ── STEP 5: NULL MODEL ────────────────────────────────────────────────────────

null_model = fit_model(X_tr_cov, y_tr_c, phenotype_type)
if null_model is None:
    print("Null model failed. Exiting.")
    sys.exit(0)

pred_tr_null = null_model.predict(sm.add_constant(X_tr_cov, has_constant="add"))
pred_te_null = null_model.predict(sm.add_constant(X_te_cov, has_constant="add"))
train_null   = calc_metric(y_tr_c, pred_tr_null, phenotype_type)
test_null    = calc_metric(y_te_c, pred_te_null, phenotype_type)
print(f"  Null — train={train_null:.4f}  test={test_null:.4f}")

# ── STEP 6: SCORE EACH GRID POINT ─────────────────────────────────────────────

rows = []

for grid_idx, grid_row in grid.iterrows():
    grid_p      = grid_row["p"]
    grid_h2     = grid_row["h2"]
    grid_sparse = grid_row["sparse"]

    # Write score file for this grid point
    score_df = gwas[["SNP","a0"]].copy()
    score_df["beta"] = betas.iloc[:, grid_idx].values
    score_file = os.path.join(outdir, f"grid_{grid_idx}.score")
    score_df.to_csv(score_file, sep="\t", index=False)

    # Score train
    tr_out = os.path.join(outdir, f"train_grid_{grid_idx}")
    run(["./plink", "--bfile", pruned_bfile,
         "--score", score_file, "1", "2", "3", "header",
         "--out", tr_out])

    # Score test
    te_out = os.path.join(outdir, f"test_grid_{grid_idx}")
    run(["./plink", "--bfile", os.path.join(folddirec, testfilename),
         "--extract", valid_snp,
         "--score", score_file, "1", "2", "3", "header",
         "--out", te_out])

    tr_prof = tr_out + ".profile"
    te_prof = te_out + ".profile"

    if not os.path.exists(tr_prof) or not os.path.exists(te_prof):
        continue

    prs_tr = pd.read_table(tr_prof, sep=r"\s+", usecols=["FID","IID","SCORE"])
    prs_te = pd.read_table(te_prof, sep=r"\s+", usecols=["FID","IID","SCORE"])
    for df in [prs_tr, prs_te]:
        df["FID"] = df["FID"].astype(str)
        df["IID"] = df["IID"].astype(str)

    merged_tr = pd.merge(cov_tr[["FID","IID"]].astype(str), prs_tr, on=["FID","IID"])
    merged_te = pd.merge(cov_te[["FID","IID"]].astype(str), prs_te, on=["FID","IID"])

    if len(merged_tr) == 0 or len(merged_te) == 0:
        continue

    sc_tr = merged_tr["SCORE"].values
    sc_te = merged_te["SCORE"].values
    yt    = y_tr_c[:len(merged_tr)]
    ye    = y_te_c[:len(merged_te)]

    # PRS only — no covariates
    prs_only_tr = calc_metric(yt, sc_tr, phenotype_type)
    prs_only_te = calc_metric(ye, sc_te, phenotype_type)

    # PRS + covariates — full model
    Xf_tr = pd.concat([
        X_tr_cov.reset_index(drop=True).iloc[:len(merged_tr)],
        pd.Series(sc_tr, name="PRS")
    ], axis=1)
    Xf_te = pd.concat([
        X_te_cov.reset_index(drop=True).iloc[:len(merged_te)],
        pd.Series(sc_te, name="PRS")
    ], axis=1)

    full = fit_model(Xf_tr, yt, phenotype_type)
    if full is None:
        prs_cov_tr = prs_cov_te = np.nan
    else:
        prs_cov_tr = calc_metric(yt, full.predict(sm.add_constant(Xf_tr, has_constant="add")), phenotype_type)
        prs_cov_te = calc_metric(ye, full.predict(sm.add_constant(Xf_te, has_constant="add")), phenotype_type)

    rows.append({
        "phenotype":        filedirec,
        "fold":             foldnumber,
        "config_index":     config_index,
        "method":           method,
        "h2model":          h2model,
        "clumpprune":       clumpprune,
        "numberofpca":      numberofpca,
        "h2":               h2,           # fold-specific h2 from heritability step
        "grid_p":           grid_p,
        "grid_h2":          grid_h2,
        "grid_sparse":      grid_sparse,
        "metric":           phenotype_type,
        # Covariates only — no PRS (baseline)
        "Train_null":       train_null,
        "Test_null":        test_null,
        # PRS score only — no covariates
        "Train_prs_only":   prs_only_tr,
        "Test_prs_only":    prs_only_te,
        # PRS + covariates (full model)
        "Train_prs_cov":    prs_cov_tr,
        "Test_prs_cov":     prs_cov_te,
        # Train minus test — overfitting indicators
        "Delta_null":       train_null    - test_null,
        "Delta_prs_only":   (prs_only_tr  - prs_only_te) if not np.isnan(prs_only_tr) else np.nan,
        "Delta_prs_cov":    (prs_cov_tr   - prs_cov_te)  if not np.isnan(prs_cov_tr)  else np.nan,
    })
    print(f"  grid={grid_idx}  p={grid_p:.6g} h2={grid_h2:.6g} "
          f"prs_only={prs_only_te:.4f} full={prs_cov_te:.4f}")

if not rows:
    print("No results produced. Nothing to save.")
    sys.exit(0)

pd.DataFrame(rows).to_csv(results_file, index=False)
print(f"\nSaved {len(rows)} rows -> {results_file}")