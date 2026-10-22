"""
calculatePRSGCTA.py
====================
Calculates GCTA-SBLUP PRS for one configuration using fold-specific h2.
h2 is passed directly from the parameter line (already fold-specific).

Usage:
  python calculatePRSGCTA.py <phenotype> <fold> <config_index> <binary|continuous>
                             <h2> <method> <h2model> <clumpprune> <numberofpca>

Output:
  <phenotype>/Fold_<fold>/GCTA-SBLUP-PRS/<config_index>/Results.csv

Results.csv columns:
  phenotype, fold, config_index, method, h2model, clumpprune, numberofpca,
  h2, lambda, m, pvalue_threshold, metric,
  Train_null,     Test_null,      <- covariates only  (no PRS)
  Train_prs_only, Test_prs_only,  <- PRS score only   (no covariates)
  Train_prs_cov,  Test_prs_cov,   <- PRS + covariates (full model)
  Delta_null, Delta_prs_only, Delta_prs_cov  <- train minus test
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
    print("Usage: python calculatePRSGCTA.py <phenotype> <fold> <config_index> "
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
outdir = os.path.join(folddirec, "GCTA-SBLUP-PRS", f"{config_index}_{safe_label}")
os.makedirs(outdir, exist_ok=True)

results_file = os.path.join(outdir, "Results.csv")
if os.path.exists(results_file):
    existing = pd.read_csv(results_file)
    if len(existing) > 0:
        print(f"Already complete: {results_file}. Skipping.")
        sys.exit(0)

print(f"phenotype={filedirec}  fold={foldnumber}  config={config_index}")
print(f"method={method}  h2model={h2model}  clumpprune={clumpprune}")
print(f"h2={h2:.6f}  type={phenotype_type}")

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

minimumpvalue     = 4
numberofintervals = 10
allpvalues        = np.logspace(-minimumpvalue, 0, numberofintervals, endpoint=True)

# ── RANGE LIST ────────────────────────────────────────────────────────────────

range_list_path = os.path.join(outdir, "range_list")
with open(range_list_path, "w") as f:
    for val in allpvalues:
        f.write(f"pv_{val} 0 {val}\n")

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

# ── STEP 1: PRUNE AND CLUMP ───────────────────────────────────────────────────

train_prefix = os.path.join(outdir, trainfilename)
pruned_bfile = os.path.join(outdir, newtrainfilename + ".clumped.pruned")

run(["./plink", "--bfile", os.path.join(folddirec, newtrainfilename),
     "--indep-pairwise", p_window_size, p_slide_size, p_LD_threshold,
     "--out", train_prefix])

run(["./plink", "--bfile", os.path.join(folddirec, newtrainfilename),
     "--clump-p1", clump_p1,
     "--extract", train_prefix + ".prune.in",
     "--clump-r2", clump_r2, "--clump-kb", clump_kb,
     "--clump", os.path.join(filedirec, filedirec + ".txt"),
     "--clump-snp-field", "SNP", "--clump-field", "P",
     "--out", train_prefix])

valid_snp = train_prefix + ".valid.snp"
clumped   = train_prefix + ".clumped"
if not os.path.exists(clumped):
    print("Clumping produced no output. Exiting.")
    sys.exit(0)
os.system(f'awk "NR!=1{{print $3}}" {clumped} > {valid_snp}')

run(["./plink", "--make-bed",
     "--bfile", os.path.join(folddirec, newtrainfilename),
     "--extract", valid_snp, "--out", pruned_bfile])

run(["./plink", "--bfile", pruned_bfile,
     "--extract", valid_snp, "--pca", npca_str,
     "--out", train_prefix])

# ── STEP 2: LAMBDA AND GCTA-SBLUP ─────────────────────────────────────────────

bim = pruned_bfile + ".bim"
if not os.path.exists(bim):
    print(f"Pruned bim not found: {bim}. Exiting.")
    sys.exit(0)

m_res = subprocess.run(["wc", "-l", bim], capture_output=True, text=True)
m     = int(m_res.stdout.strip().split()[0])
lam   = float(m) * (1.0 / h2 - 1.0)
print(f"  m={m}  lambda={lam:.2f}  h2={h2:.6f} (fold-specific)")

sblup_prefix = os.path.join(outdir, "Train_data_SBLUP")
run(["./gcta",
     "--bfile", pruned_bfile,
     "--cojo-file", os.path.join(filedirec, "SBLUPGWAS.ma"),
     "--cojo-wind", clump_kb,
     "--cojo-sblup", str(lam),
     "--out", sblup_prefix])

sblup_file = sblup_prefix + ".sblup.cojo"
if not os.path.exists(sblup_file):
    print(f"GCTA-SBLUP output not found. Exiting.")
    sys.exit(0)

# ── STEP 3: SCORE TRAIN AND TEST ──────────────────────────────────────────────

gwas  = pd.read_csv(os.path.join(filedirec, filedirec + ".txt"), sep=r"\s+")
sblup = pd.read_csv(sblup_file, sep=r"\s+",
                    names=["SNP","A1","GWAS_Effect","SBLUP_Effect"])
gwas  = gwas[gwas["SNP"].isin(sblup["SNP"])]
pval_file = os.path.join(outdir, "SNP_SBLUP.pvalue")
gwas[["SNP","P"]].to_csv(pval_file, index=False, sep="\t")

# Score train
run(["./plink", "--bfile", pruned_bfile,
     "--score", sblup_file, "1", "2", "4", "header",
     "--q-score-range", range_list_path, pval_file,
     "--out", os.path.join(outdir, trainfilename)])

# PCA and score test
test_prefix = os.path.join(outdir, testfilename)
run(["./plink", "--bfile", os.path.join(folddirec, testfilename),
     "--extract", valid_snp, "--pca", npca_str,
     "--out", test_prefix])

run(["./plink", "--bfile", os.path.join(folddirec, testfilename),
     "--score", sblup_file, "1", "2", "4", "header",
     "--q-score-range", range_list_path, pval_file,
     "--out", test_prefix])

# ── STEP 4: PHENOTYPES AND COVARIATES ─────────────────────────────────────────

y_tr_all = read_pheno(os.path.join(folddirec, newtrainfilename + ".fam"), phenotype_type)
y_te_all = read_pheno(os.path.join(folddirec, testfilename + ".fam"),     phenotype_type)

cov_tr = read_cov_pca(
    os.path.join(folddirec, trainfilename + ".cov"),
    train_prefix + ".eigenvec", npca)

cov_te = read_cov_pca(
    os.path.join(folddirec, testfilename + ".cov"),
    test_prefix + ".eigenvec", npca)

n_tr   = len(cov_tr)
n_te   = len(cov_te)
y_tr   = y_tr_all[:n_tr]
y_te   = y_te_all[:n_te]
tr_ok  = ~np.isnan(y_tr)
te_ok  = ~np.isnan(y_te)

y_tr_c   = y_tr[tr_ok]
y_te_c   = y_te[te_ok]
X_tr_cov = cov_tr.iloc[tr_ok].reset_index(drop=True).iloc[:, 2:]
X_te_cov = cov_te.iloc[te_ok].reset_index(drop=True).iloc[:, 2:]

# ── STEP 5: NULL MODEL (covariates only — no PRS) ─────────────────────────────

null_model = fit_model(X_tr_cov, y_tr_c, phenotype_type)
if null_model is None:
    print("Null model failed. Exiting.")
    sys.exit(0)

pred_tr_null = null_model.predict(sm.add_constant(X_tr_cov, has_constant="add"))
pred_te_null = null_model.predict(sm.add_constant(X_te_cov, has_constant="add"))
train_null   = calc_metric(y_tr_c, pred_tr_null, phenotype_type)
test_null    = calc_metric(y_te_c, pred_te_null, phenotype_type)
print(f"  Null — train={train_null:.4f}  test={test_null:.4f}")

# ── STEP 6: PRS MODELS PER P-VALUE THRESHOLD ─────────────────────────────────

rows = []
for pval in allpvalues:
    tr_prof = os.path.join(outdir, f"train_data.pv_{pval}.profile")
    te_prof = os.path.join(outdir, f"test_data.pv_{pval}.profile")

    if not os.path.exists(tr_prof) or not os.path.exists(te_prof):
        continue

    prs_tr = pd.read_table(tr_prof, sep=r"\s+", usecols=["FID","IID","SCORE"])
    prs_te = pd.read_table(te_prof, sep=r"\s+", usecols=["FID","IID","SCORE"])
    for df in [prs_tr, prs_te]:
        df["FID"] = df["FID"].astype(str)
        df["IID"] = df["IID"].astype(str)

    cov_tr_ids = cov_tr[["FID","IID"]].astype(str)
    cov_te_ids = cov_te[["FID","IID"]].astype(str)
    merged_tr  = pd.merge(cov_tr_ids, prs_tr, on=["FID","IID"])
    merged_te  = pd.merge(cov_te_ids, prs_te, on=["FID","IID"])

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
        "h2":               h2,               # fold-specific h2
        "lambda":           lam,
        "m":                m,
        "pvalue_threshold": pval,
        "metric":           phenotype_type,
        # Covariates only — no PRS — baseline
        "Train_null":       train_null,
        "Test_null":        test_null,
        # PRS only — no covariates
        "Train_prs_only":   prs_only_tr,
        "Test_prs_only":    prs_only_te,
        # PRS + covariates — full model
        "Train_prs_cov":    prs_cov_tr,
        "Test_prs_cov":     prs_cov_te,
        # Train minus test — overfitting indicator
        "Delta_null":       train_null    - test_null,
        "Delta_prs_only":   (prs_only_tr  - prs_only_te) if not np.isnan(prs_only_tr) else np.nan,
        "Delta_prs_cov":    (prs_cov_tr   - prs_cov_te)  if not np.isnan(prs_cov_tr)  else np.nan,
    })
    print(f"  pval={pval:.5f}  prs_only={prs_only_te:.4f}  full={prs_cov_te:.4f}")

if not rows:
    print("No PRS profile files found. Nothing to save.")
    sys.exit(0)

pd.DataFrame(rows).to_csv(results_file, index=False)
print(f"\nSaved {len(rows)} rows -> {results_file}")