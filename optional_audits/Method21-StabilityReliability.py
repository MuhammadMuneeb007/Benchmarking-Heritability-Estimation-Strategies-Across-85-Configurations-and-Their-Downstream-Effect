#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Method21-StabilityReliability.py

Reviewer-facing stability/reliability analysis for the heritability benchmark.

Run from:
    /data/ascher01/uqmmune1/finalheritability

Command:
    python Method21-StabilityReliability.py | tee Method21_StabilityReliability.txt

Outputs:
    Method21_raw_real.csv
    Method21_raw_simulated.csv
    Method21_unit_stability_real.csv
    Method21_unit_stability_simulated.csv
    Method21_method_stability_real.csv
    Method21_method_stability_simulated.csv
    Method21_simulation_accuracy.csv
    Method21_sample_size_effect.csv
    Method21_stability_accuracy_association.csv
    Method21_real_vs_sim_stability.csv
    Method21_top_stable_real.csv
    Method21_top_unstable_real.csv
    Method21_MAIN_MANUSCRIPT_SUMMARY.txt

Important:
* Overlapping CV folds -> fold SD/MAD/IQR/range are STABILITY measures only.
* Do NOT use fold_SD/sqrt(5) as inferential SE.
* Tool-reported SE/CI are kept separate.
* Negative and >1 h2 are retained, not clipped.
* Known-truth accuracy is restricted to CONTINUOUS simulations until binary
  estimates are harmonized to liability scale.
"""

import re
import math
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from scipy.stats import wilcoxon, spearmanr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

ROOT = Path(".").resolve()
EXPECTED_FOLDS = 5

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
    "body_mass_index_bmi": "BMI",
    "depression": "Depression",
    "gastro_oesophageal_reflux_gord_gastric_reflux": "GORD",
    "high_cholesterol": "High cholesterol",
    "hypertension": "Hypertension",
    "hypothyroidism_myxoedema": "Hypothyroidism",
    "irritable_bowel_syndrome": "IBS",
    "migraine": "Migraine",
    "osteoarthritis": "Osteoarthritis",
}

SIM_METADATA_FILES = [
    "simulation_manifest.csv",
    "simulation_characteristics_complete.csv",
    "simulation_phenotype_characteristics_complete.csv",
    "Step2_Simulated_Data_Characteristics_All200.csv",
    "simulation_design_summary.csv",
    "simulation_truth_summary.csv",
]

# descriptive flags only; manuscript inference should use continuous metrics
STABLE_SD_005 = 0.05
STABLE_SD_010 = 0.10
STRICT_RANGE = 0.15


def norm(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_")


def clean(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def fnum(x):
    try:
        y = float(x)
        return y if np.isfinite(y) else np.nan
    except Exception:
        return np.nan


def MAD(x):
    a = pd.to_numeric(pd.Series(x), errors="coerce").dropna().values
    if len(a) == 0:
        return np.nan
    m = np.median(a)
    return float(np.median(np.abs(a - m)))


def IQR(x):
    a = pd.to_numeric(pd.Series(x), errors="coerce").dropna().values
    if len(a) == 0:
        return np.nan
    return float(np.percentile(a, 75) - np.percentile(a, 25))


def bh(pvals):
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
    adj = np.clip(adj, 0, 1)
    q[ok[order]] = adj
    return q


def safe_wilcoxon(d):
    a = pd.to_numeric(pd.Series(d), errors="coerce").dropna().values
    if len(a) < 3 or not HAVE_SCIPY:
        return np.nan
    if np.allclose(a, 0):
        return 1.0
    try:
        return float(wilcoxon(a, zero_method="wilcox").pvalue)
    except Exception:
        return np.nan


def safe_spearman(x, y):
    a = pd.to_numeric(pd.Series(x), errors="coerce")
    b = pd.to_numeric(pd.Series(y), errors="coerce")
    k = a.notna() & b.notna()
    a, b = a[k].values, b[k].values
    if len(a) < 5 or not HAVE_SCIPY or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return np.nan, np.nan, len(a)
    try:
        r, p = spearmanr(a, b)
        return float(r), float(p), len(a)
    except Exception:
        return np.nan, np.nan, len(a)


def detect_h2(df):
    nmap = {c: norm(c) for c in df.columns}
    exact = [
        "h2", "heritability", "h2_estimate", "heritability_estimate",
        "h2_obs", "h2_observed", "h2_liability"
    ]
    for t in exact:
        for c in df.columns:
            if nmap[c] == t:
                return c
    for c in df.columns:
        n = nmap[c]
        if ("heritability" in n or re.search(r"(^|_)h2($|_)", n)) and not any(
            z in n for z in ["se", "ci", "lower", "upper", "true", "truth", "prs", "pvalue", "p_value"]
        ):
            return c
    return None


def detect_se(df):
    for t in ["h2_se", "heritability_se", "se_h2", "se", "stderr", "standard_error"]:
        for c in df.columns:
            if norm(c) == t:
                return c
    for c in df.columns:
        n = norm(c)
        if ("h2" in n or "heritability" in n) and any(z in n for z in ["se", "stderr", "standard_error"]):
            return c
    return None


def detect_ci(df):
    lo = hi = None
    for t in ["ci_lower", "lower_ci", "lci", "ci95_lower", "lower_95", "h2_lower", "heritability_lower"]:
        for c in df.columns:
            if norm(c) == t:
                lo = c
                break
        if lo:
            break
    for t in ["ci_upper", "upper_ci", "uci", "ci95_upper", "upper_95", "h2_upper", "heritability_upper"]:
        for c in df.columns:
            if norm(c) == t:
                hi = c
                break
        if hi:
            break
    return lo, hi


def classify_method(text):
    x = str(text).lower()
    if "ldsc-genotype" in x or "ldsc_genotype" in x:
        return "LDSC-Genotype"
    if "gemma-ldsc" in x or "gemma_ldsc" in x:
        return "Gemma-LDSC"
    if "dpr" in x:
        return "DPR-Gemma"
    if "gemma1" in x or "gemma_1" in x:
        return "Gemma1"
    if "gemma2" in x or "gemma_2" in x:
        return "Gemma2"
    if "ldpred" in x:
        return "LDpred2"
    if "ldak" in x:
        pre = [
            "precomputed", "tagging_no", "bld_ldak_hapmap", "ldak_thin_hapmap",
            "bld_ldak_genotyped", "ldak_thin_genotyped", "lite_alpha_genotyped"
        ]
        return "LDAK-Precomputed" if any(z in x for z in pre) else "LDAK-Calculated"
    if "ldsc" in x:
        return "LDSC"
    if any(z in x for z in ["gcta", "reml_ai", "he_regression", "haseman", "elston"]):
        return "GCTA"
    if "gemma" in x:
        # Generic GEMMA genotype-based configurations correspond to Gemma1
        # in this benchmark. More specific Gemma2/Gemma-LDSC/DPR labels are
        # already caught above.
        return "Gemma1"
    return "Unknown"


def fold_from_path(path):
    for p in path.parts:
        m = re.fullmatch(r"Fold_(\d+)", p)
        if m:
            return int(m.group(1))
    return np.nan


def result_files(dataset_dir):
    shallow = []
    for f in range(EXPECTED_FOLDS):
        shallow += sorted((dataset_dir / f"Fold_{f}").glob("*/Results.csv"))
    if shallow:
        return shallow
    out = []
    for f in range(EXPECTED_FOLDS):
        d = dataset_dir / f"Fold_{f}"
        if d.is_dir():
            out += sorted(d.rglob("Results.csv"))
    return out


OUTCOME_TOKENS = [
    "heritability", "h2", "stderr", "standard_error", "ci", "lower", "upper",
    "pvalue", "p_value", "auc", "r2", "rmse", "mae", "correlation", "prs",
    "runtime", "time", "status", "error", "warning", "converge", "sample_size",
    "samplesize", "n_samples", "ncase", "n_case", "ncontrol", "n_control"
]


def config_id(df, row, config_dir, idx):
    """
    Stable cross-fold configuration identifier.

    The benchmark configuration is encoded by the configuration directory
    (for example 2764_GCTA_genotype_yes).  The previous implementation also
    hashed arbitrary Results.csv columns; some of those values vary by fold,
    which incorrectly split one configuration into multiple stability units.

    Therefore the semantic directory name is the primary ID.  Row index is
    appended only if a single Results.csv genuinely contains multiple h2 rows.
    """
    semantic = re.sub(r"^\d+[_-]*", "", str(config_dir))
    raw = semantic if len(df) == 1 else f"{semantic}||row={idx}"
    return raw, raw


NON_HERITABILITY_DIRS = {"lassosum", "plink"}


def extract_dataset(dataset_dir, dataset_id, kind, trait_type):
    recs = []
    for path in result_files(dataset_dir):
        # These are PRS/preprocessing outputs, not heritability estimators.
        if path.parent.name.lower() in NON_HERITABILITY_DIRS:
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            recs.append({
                "dataset_id": dataset_id, "kind": kind, "trait_type": trait_type,
                "fold": fold_from_path(path), "method": "Unknown", "config_id": "",
                "config_signature": "", "h2": np.nan, "tool_se": np.nan,
                "tool_ci_lower": np.nan, "tool_ci_upper": np.nan,
                "result_file": str(path), "status": f"READ_ERROR:{e}"
            })
            continue

        hcol = detect_h2(df)
        if hcol is None:
            continue
        scol = detect_se(df)
        lcol, ucol = detect_ci(df)

        for idx, row in df.iterrows():
            cid, sig = config_id(df, row, path.parent.name, idx)
            rowtxt = " ".join(clean(row.get(c, "")) for c in df.columns if len(clean(row.get(c, ""))) < 150)
            method = classify_method(f"{path.parent.name} {rowtxt}")
            recs.append({
                "dataset_id": dataset_id,
                "kind": kind,
                "trait_type": trait_type,
                "fold": fold_from_path(path),
                "method": method,
                "config_id": cid,
                "config_signature": sig,
                "h2": fnum(row.get(hcol)),
                "tool_se": fnum(row.get(scol)) if scol else np.nan,
                "tool_ci_lower": fnum(row.get(lcol)) if lcol else np.nan,
                "tool_ci_upper": fnum(row.get(ucol)) if ucol else np.nan,
                "result_file": str(path),
                "status": "OK",
            })
    return recs


def sim_id(x):
    m = re.search(r"Sim\s*0*(\d+)", str(x), re.I)
    return f"Sim{int(m.group(1)):03d}" if m else ""


def find_sim_col(df):
    for c in df.columns:
        vals = df[c].astype(str)
        if vals.str.contains(r"Sim\d+", case=False, regex=True).mean() > 0.2:
            return c
    return None


def load_sim_metadata():
    parts = []
    used = []
    for fn in SIM_METADATA_FILES:
        p = ROOT / fn
        if not p.is_file():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        sc = find_sim_col(df)
        if sc is None:
            continue
        used.append(fn)
        out = pd.DataFrame({"sim_id": df[sc].map(sim_id)})
        keep = out["sim_id"] != ""
        out = out[keep].copy()
        src = df.loc[keep].copy()

        # sample size
        ncol = None
        for c in df.columns:
            if norm(c) in {"n", "sample_size", "samplesize", "target_n", "target_sample_size"}:
                ncol = c
                break
        if ncol is None:
            for c in df.columns:
                x = pd.to_numeric(df[c], errors="coerce").dropna()
                if len(x) and ({750.0, 5000.0} & set(x.astype(float).unique())) == {750.0, 5000.0}:
                    ncol = c
                    break
        out["sample_size"] = pd.to_numeric(src[ncol], errors="coerce").values if ncol else np.nan

        # truth h2
        hcol = None
        for c in df.columns:
            n = norm(c)
            if ("true" in n or "truth" in n) and ("h2" in n or "heritability" in n):
                hcol = c
                break
        if hcol is None:
            for c in df.columns:
                if norm(c) in {"h2", "heritability", "simulated_heritability"}:
                    x = pd.to_numeric(df[c], errors="coerce").dropna()
                    if len(x) and x.between(0, 1).mean() > 0.9:
                        hcol = c
                        break
        out["true_h2"] = pd.to_numeric(src[hcol], errors="coerce").values if hcol else np.nan

        # trait type
        tcol = None
        for c in df.columns:
            n = norm(c)
            if n in {"trait_type", "phenotype_type", "outcome_type", "binary_continuous", "type"}:
                tcol = c
                break
        if tcol is None:
            for c in df.columns:
                s = df[c].astype(str).str.lower()
                if s.str.contains("binary").any() and s.str.contains("cont").any():
                    tcol = c
                    break
        if tcol:
            s = src[tcol].astype(str).str.lower()
            out["sim_trait_type"] = np.where(s.str.contains("bin"), "binary", np.where(s.str.contains("cont"), "continuous", s))
        else:
            out["sim_trait_type"] = ""

        ccol = None
        for c in df.columns:
            n = norm(c)
            if "causal" in n and any(z in n for z in ["snp", "variant", "number", "num"]):
                ccol = c
                break
        out["causal_snps"] = pd.to_numeric(src[ccol], errors="coerce").values if ccol else np.nan

        rcol = None
        for c in df.columns:
            if norm(c) in {"replicate", "rep", "replicate_id", "rep_id"}:
                rcol = c
                break
        out["replicate"] = src[rcol].values if rcol else np.nan
        out["metadata_source"] = fn
        parts.append(out)

    if not parts:
        return pd.DataFrame(), used

    allm = pd.concat(parts, ignore_index=True)
    rows = []
    for sid, g in allm.groupby("sim_id"):
        r = {"sim_id": sid}
        for c in ["sample_size", "true_h2", "sim_trait_type", "causal_snps", "replicate"]:
            vals = g[c].dropna()
            if c == "sim_trait_type":
                vals = [clean(v) for v in vals if clean(v)]
                r[c] = vals[0] if vals else ""
            else:
                r[c] = vals.iloc[0] if len(vals) else np.nan
        r["metadata_source"] = ";".join(sorted(set(g["metadata_source"].astype(str))))
        rows.append(r)

    meta = pd.DataFrame(rows)

    def skey(r):
        tt = clean(r.get("sim_trait_type", ""))
        h = fnum(r.get("true_h2"))
        c = fnum(r.get("causal_snps"))
        rep = clean(r.get("replicate", ""))
        if not tt or not np.isfinite(h) or not np.isfinite(c) or rep == "":
            return ""
        return f"{tt}|h2={h:.6g}|causal={int(c)}|rep={rep}"

    meta["scenario_key"] = meta.apply(skey, axis=1)
    return meta, used


def aggregate(raw, id_col):
    """
    Aggregate one configuration within one dataset across CV folds.

    Stability must be calculated across folds, never across duplicate rows.
    If duplicate numerical rows occur within the same fold/configuration, they
    are first collapsed to one fold-level estimate.
    """
    if raw.empty:
        return pd.DataFrame()

    rows = []
    gcols = [id_col, "trait_type", "method", "config_id", "config_signature"]

    for keys, g in raw.groupby(gcols, dropna=False, sort=False):
        r = dict(zip(gcols, keys))

        # Collapse duplicate rows within a fold before computing stability.
        gg = g.copy()
        gg["h2_num"] = pd.to_numeric(gg["h2"], errors="coerce")
        gg["se_num"] = pd.to_numeric(gg["tool_se"], errors="coerce")
        gg["lo_num"] = pd.to_numeric(gg["tool_ci_lower"], errors="coerce")
        gg["hi_num"] = pd.to_numeric(gg["tool_ci_upper"], errors="coerce")

        fold_rows = []
        for fold, gf in gg.groupby("fold", dropna=True):
            hv = gf["h2_num"].dropna()
            sev = gf["se_num"].dropna()
            lov = gf["lo_num"].dropna()
            hiv = gf["hi_num"].dropna()
            fold_rows.append({
                "fold": fold,
                "h2": float(hv.mean()) if len(hv) else np.nan,
                "tool_se": float(sev.median()) if len(sev) else np.nan,
                "ci_lower": float(lov.median()) if len(lov) else np.nan,
                "ci_upper": float(hiv.median()) if len(hiv) else np.nan,
                "raw_rows_in_fold": len(gf),
            })

        fg = pd.DataFrame(fold_rows)
        h = pd.to_numeric(fg.get("h2", pd.Series(dtype=float)), errors="coerce").dropna()

        r["n_raw_rows"] = len(g)
        r["n_folds_observed"] = int(fg["fold"].nunique()) if not fg.empty else 0
        r["n_valid_h2"] = int(h.notna().sum())
        r["missing_fold_count"] = max(0, EXPECTED_FOLDS - r["n_folds_observed"])
        r["duplicate_rows_collapsed"] = int(max(0, len(g) - r["n_folds_observed"]))

        if len(h):
            r["h2_mean"] = float(h.mean())
            r["h2_median"] = float(h.median())
            r["h2_min"] = float(h.min())
            r["h2_max"] = float(h.max())
            r["fold_sd"] = float(h.std(ddof=1)) if len(h) >= 2 else np.nan
            r["fold_mad"] = MAD(h)
            r["fold_iqr"] = IQR(h)
            r["fold_range"] = float(h.max() - h.min())
            r["negative_fraction"] = float((h < 0).mean())
            r["gt1_fraction"] = float((h > 1).mean())
            r["any_negative"] = int((h < 0).any())
            r["any_gt1"] = int((h > 1).any())
            r["sign_flip"] = int(h.min() < 0 < h.max())
        else:
            for c in [
                "h2_mean", "h2_median", "h2_min", "h2_max", "fold_sd",
                "fold_mad", "fold_iqr", "fold_range", "negative_fraction",
                "gt1_fraction"
            ]:
                r[c] = np.nan
            r["any_negative"] = r["any_gt1"] = r["sign_flip"] = 0

        se = pd.to_numeric(fg.get("tool_se", pd.Series(dtype=float)), errors="coerce").dropna()
        r["tool_se_available_folds"] = len(se)
        r["tool_se_median"] = float(se.median()) if len(se) else np.nan

        lo = pd.to_numeric(fg.get("ci_lower", pd.Series(dtype=float)), errors="coerce")
        hi = pd.to_numeric(fg.get("ci_upper", pd.Series(dtype=float)), errors="coerce")
        ok = lo.notna() & hi.notna()
        r["tool_ci_available_folds"] = int(ok.sum())
        r["tool_ci_contains_zero_fraction"] = (
            float(((lo[ok] <= 0) & (hi[ok] >= 0)).mean()) if ok.any() else np.nan
        )

        # Stability is evaluable only with >=4 numerical folds. Missing units
        # are reported as missing/unassessable, not automatically unstable.
        enough = r["n_valid_h2"] >= 4
        r["stability_evaluable"] = int(enough)
        r["stable_sd_le_0_05"] = (
            int(r["fold_sd"] <= STABLE_SD_005)
            if enough and np.isfinite(r["fold_sd"]) else np.nan
        )
        r["stable_sd_le_0_10"] = (
            int(r["fold_sd"] <= STABLE_SD_010)
            if enough and np.isfinite(r["fold_sd"]) else np.nan
        )
        r["stable_strict"] = (
            int(r["fold_sd"] <= STABLE_SD_005 and r["fold_range"] <= STRICT_RANGE)
            if enough and np.isfinite(r["fold_sd"]) and np.isfinite(r["fold_range"])
            else np.nan
        )

        rows.append(r)

    return pd.DataFrame(rows)


def method_summary(units, simulated=False):
    if units.empty:
        return pd.DataFrame()

    groups = ["method"]
    if simulated:
        groups += ["trait_type", "sample_size"]

    rows = []
    for keys, g in units.groupby(groups, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        r = dict(zip(groups, keys))
        r["n_units"] = len(g)
        r["n_evaluable_units"] = int((g["stability_evaluable"] == 1).sum())
        r["evaluable_pct"] = 100 * r["n_evaluable_units"] / len(g) if len(g) else np.nan

        evalg = g[g["stability_evaluable"] == 1].copy()

        for c in ["fold_sd", "fold_mad", "fold_iqr", "fold_range", "tool_se_median"]:
            x = pd.to_numeric(evalg[c], errors="coerce").dropna()
            r[c + "_median"] = float(x.median()) if len(x) else np.nan
            r[c + "_q25"] = float(x.quantile(.25)) if len(x) else np.nan
            r[c + "_q75"] = float(x.quantile(.75)) if len(x) else np.nan

        for c in ["stable_sd_le_0_05", "stable_sd_le_0_10", "stable_strict"]:
            x = pd.to_numeric(evalg[c], errors="coerce").dropna()
            r[c + "_pct"] = 100 * x.mean() if len(x) else np.nan

        # Boundary behaviour can be summarized over all units with at least one h2.
        validg = g[g["n_valid_h2"] >= 1]
        r["any_negative_pct"] = 100 * validg["any_negative"].mean() if len(validg) else np.nan
        r["any_gt1_pct"] = 100 * validg["any_gt1"].mean() if len(validg) else np.nan
        r["sign_flip_pct"] = 100 * evalg["sign_flip"].mean() if len(evalg) else np.nan
        r["mean_missing_folds"] = float(g["missing_fold_count"].mean())
        rows.append(r)

    return pd.DataFrame(rows)


def accuracy(sim_units):
    if sim_units.empty:
        return pd.DataFrame(), pd.DataFrame()
    x = sim_units.copy()
    x = x[x["trait_type"].astype(str).str.lower().eq("continuous")]
    x["true_h2"] = pd.to_numeric(x["true_h2"], errors="coerce")
    x = x[x["true_h2"].notna() & x["h2_mean"].notna()].copy()
    if x.empty:
        return x, pd.DataFrame()
    x["bias"] = x["h2_mean"] - x["true_h2"]
    x["abs_error"] = x["bias"].abs()
    x["sq_error"] = x["bias"] ** 2
    rows = []
    for (m, n), g in x.groupby(["method", "sample_size"], dropna=False):
        rows.append({
            "method": m, "sample_size": n, "n_units": len(g), "n_simulations": g["sim_id"].nunique(),
            "MAE": float(g["abs_error"].mean()), "median_abs_error": float(g["abs_error"].median()),
            "bias": float(g["bias"].mean()), "RMSE": float(np.sqrt(g["sq_error"].mean())),
            "negative_mean_h2_pct": 100*(g["h2_mean"] < 0).mean(), "gt1_mean_h2_pct": 100*(g["h2_mean"] > 1).mean(),
            "median_fold_sd": float(g.loc[g["stability_evaluable"] == 1, "fold_sd"].median()),
            "median_fold_mad": float(g.loc[g["stability_evaluable"] == 1, "fold_mad"].median()),
            "stable_sd_le_0_05_pct": 100*pd.to_numeric(g.loc[g["stability_evaluable"] == 1, "stable_sd_le_0_05"], errors="coerce").mean(),
            "stable_strict_pct": 100*pd.to_numeric(g.loc[g["stability_evaluable"] == 1, "stable_strict"], errors="coerce").mean(),
        })
    return x, pd.DataFrame(rows)


def sample_size_effect(acc_units):
    if acc_units.empty or "scenario_key" not in acc_units.columns:
        return pd.DataFrame()
    x = acc_units[acc_units["scenario_key"].astype(str) != ""].copy()
    x["sample_size"] = pd.to_numeric(x["sample_size"], errors="coerce")
    x = x[x["sample_size"].isin([750, 5000])]
    if x.empty:
        return pd.DataFrame()
    agg = x.groupby(["scenario_key", "method", "config_id", "sample_size"], as_index=False).agg(
        fold_sd=("fold_sd", "median"), abs_error=("abs_error", "median")
    )
    rows = []
    for m, g in agg.groupby("method"):
        r = {"method": m}
        psd = g.pivot_table(index=["scenario_key", "config_id"], columns="sample_size", values="fold_sd", aggfunc="median")
        perr = g.pivot_table(index=["scenario_key", "config_id"], columns="sample_size", values="abs_error", aggfunc="median")
        if 750 in psd.columns and 5000 in psd.columns:
            d = (psd[5000] - psd[750]).dropna()
            r.update({"n_paired_stability": len(d), "median_delta_fold_sd_N5000_minus_N750": float(d.median()) if len(d) else np.nan,
                      "mean_delta_fold_sd_N5000_minus_N750": float(d.mean()) if len(d) else np.nan,
                      "pct_more_stable_at_N5000": 100*(d < 0).mean() if len(d) else np.nan,
                      "p_stability": safe_wilcoxon(d)})
        else:
            r.update({"n_paired_stability": 0, "median_delta_fold_sd_N5000_minus_N750": np.nan,
                      "mean_delta_fold_sd_N5000_minus_N750": np.nan, "pct_more_stable_at_N5000": np.nan, "p_stability": np.nan})
        if 750 in perr.columns and 5000 in perr.columns:
            d = (perr[5000] - perr[750]).dropna()
            r.update({"n_paired_accuracy": len(d), "median_delta_abs_error_N5000_minus_N750": float(d.median()) if len(d) else np.nan,
                      "mean_delta_abs_error_N5000_minus_N750": float(d.mean()) if len(d) else np.nan,
                      "pct_more_accurate_at_N5000": 100*(d < 0).mean() if len(d) else np.nan,
                      "p_accuracy": safe_wilcoxon(d)})
        else:
            r.update({"n_paired_accuracy": 0, "median_delta_abs_error_N5000_minus_N750": np.nan,
                      "mean_delta_abs_error_N5000_minus_N750": np.nan, "pct_more_accurate_at_N5000": np.nan, "p_accuracy": np.nan})
        rows.append(r)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_stability"] = bh(out["p_stability"].values)
        out["q_accuracy"] = bh(out["p_accuracy"].values)
    return out


def stability_accuracy_assoc(acc_units):
    if acc_units.empty:
        return pd.DataFrame()
    reduced = acc_units.groupby(["sim_id", "method", "sample_size"], as_index=False).agg(
        median_fold_sd=("fold_sd", "median"), median_fold_mad=("fold_mad", "median"), median_abs_error=("abs_error", "median")
    )
    rows = []
    for (m, n), g in reduced.groupby(["method", "sample_size"], dropna=False):
        r1, p1, nn = safe_spearman(g["median_fold_sd"], g["median_abs_error"])
        r2, p2, _ = safe_spearman(g["median_fold_mad"], g["median_abs_error"])
        rows.append({"method": m, "sample_size": n, "n_simulations": nn,
                     "rho_foldSD_vs_absError": r1, "p_foldSD_vs_absError": p1,
                     "rho_foldMAD_vs_absError": r2, "p_foldMAD_vs_absError": p2})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_foldSD_vs_absError"] = bh(out["p_foldSD_vs_absError"].values)
        out["q_foldMAD_vs_absError"] = bh(out["p_foldMAD_vs_absError"].values)
    return out


def real_vs_sim(real_m, sim_m):
    if real_m.empty or sim_m.empty:
        return pd.DataFrame()
    s = sim_m[sim_m["trait_type"].astype(str).str.lower().eq("continuous")].copy()
    keep = ["method", "sample_size", "fold_sd_median", "fold_mad_median", "fold_range_median", "stable_sd_le_0_05_pct", "stable_strict_pct"]
    s = s[[c for c in keep if c in s.columns]].rename(columns={c: "sim_"+c for c in keep if c not in {"method", "sample_size"}})
    rkeep = ["method", "fold_sd_median", "fold_mad_median", "fold_range_median", "stable_sd_le_0_05_pct", "stable_strict_pct"]
    r = real_m[[c for c in rkeep if c in real_m.columns]].rename(columns={c: "real_"+c for c in rkeep if c != "method"})
    out = r.merge(s, on="method", how="outer")
    if "real_fold_sd_median" in out.columns and "sim_fold_sd_median" in out.columns:
        out["sim_to_real_median_SD_ratio"] = out["sim_fold_sd_median"] / out["real_fold_sd_median"]
    return out


def F(x, d=4):
    try:
        x = float(x)
        return f"{x:.{d}f}" if np.isfinite(x) else "NA"
    except Exception:
        return "NA"


def summary_text(real_m, acc_m, n_eff, assoc, real_units):
    L = []
    L += ["METHOD 21 — MAIN MANUSCRIPT SUMMARY", "="*90, "",
          "Cross-fold SD/MAD/IQR/range are stability metrics only because CV training folds overlap.",
          "No inferential SE=fold_SD/sqrt(5) or fold-derived 95% CI is used.", ""]
    if not real_m.empty:
        L += ["REAL-DATA STABILITY BY METHOD", "-"*90]
        for _, r in real_m.sort_values("fold_sd_median").iterrows():
            L.append(f"{r['method']}: median SD={F(r['fold_sd_median'])}; MAD={F(r['fold_mad_median'])}; range={F(r['fold_range_median'])}; SD<=0.05={F(r['stable_sd_le_0_05_pct'],1)}%; strict={F(r['stable_strict_pct'],1)}%; any negative={F(r['any_negative_pct'],1)}%; any >1={F(r['any_gt1_pct'],1)}%.")
        L.append("")
    if not acc_m.empty:
        L += ["CONTINUOUS SIMULATION — ACCURACY + STABILITY", "-"*90]
        for _, r in acc_m.sort_values(["sample_size", "MAE"]).iterrows():
            L.append(f"{r['method']} N={F(r['sample_size'],0)}: MAE={F(r['MAE'])}; bias={F(r['bias'])}; RMSE={F(r['RMSE'])}; median SD={F(r['median_fold_sd'])}; SD<=0.05={F(r['stable_sd_le_0_05_pct'],1)}%.")
        L.append("")
    if not n_eff.empty:
        L += ["PAIRED SAMPLE-SIZE EFFECT (N=5000 minus N=750)", "-"*90,
              "Negative delta SD = more stable at N=5000; negative delta |error| = more accurate at N=5000."]
        for _, r in n_eff.iterrows():
            L.append(f"{r['method']}: delta SD median={F(r['median_delta_fold_sd_N5000_minus_N750'])}, q={F(r['q_stability'])}; delta |error| median={F(r['median_delta_abs_error_N5000_minus_N750'])}, q={F(r['q_accuracy'])}.")
        L.append("")
    if not assoc.empty:
        L += ["STABILITY–ACCURACY ASSOCIATION", "-"*90,
              "Positive rho means greater cross-fold instability tended to accompany larger absolute error."]
        for _, r in assoc.iterrows():
            L.append(f"{r['method']} N={F(r['sample_size'],0)}: rho(SD,|error|)={F(r['rho_foldSD_vs_absError'])}, q={F(r['q_foldSD_vs_absError'])}.")
        L.append("")
    if not real_units.empty:
        evaluable = real_units[real_units["stability_evaluable"] == 1].copy()
        stable = int(pd.to_numeric(evaluable["stable_strict"], errors="coerce").fillna(0).sum())
        denom = len(evaluable)
        pct = 100*stable/denom if denom else np.nan
        incomplete = len(real_units) - denom
        L.append(f"Strict descriptive stability (>=4 numerical folds, SD<={STABLE_SD_005}, range<={STRICT_RANGE}) was met by {stable}/{denom} evaluable real configuration-phenotype units ({pct:.1f}%).")
        L.append(f"An additional {incomplete} units had fewer than four numerical folds and were treated as not evaluable for the stability flag rather than as unstable.")
        L.append("This threshold is descriptive only; manuscript inference should emphasize continuous stability metrics.")
    return "\n".join(L)


print("="*110)
print("METHOD 21 — STABILITY / RELIABILITY + SIMULATION ACCURACY")
print("="*110)
print("Root:", ROOT)
print("SciPy available:", HAVE_SCIPY)
print()

# 1. REAL
print("[1/7] Reading real-data Results.csv ...")
rr = []
for ph in REAL_PHENOTYPES:
    d = ROOT / ph
    if not d.is_dir():
        print("  [MISSING]", ph)
        continue
    tt = "continuous" if ph == "body_mass_index_bmi" else "binary"
    rec = extract_dataset(d, ph, "real", tt)
    rr.extend(rec)
    print(f"  {REAL_LABELS.get(ph, ph)}: {len(rec)} raw rows")
real_raw = pd.DataFrame(rr)
if not real_raw.empty:
    real_raw = real_raw.rename(columns={"dataset_id": "phenotype"})
real_raw.to_csv("Method21_raw_real.csv", index=False)
print("  Wrote Method21_raw_real.csv:", len(real_raw))
print()

# 2. METADATA
print("[2/7] Loading simulation metadata ...")
meta, used = load_sim_metadata()
print("  Metadata files used:", ", ".join(used) if used else "NONE")
print("  Metadata rows:", len(meta))
if not meta.empty:
    print("  N:", sorted(pd.to_numeric(meta["sample_size"], errors="coerce").dropna().unique().tolist()))
    print("  Trait types:", sorted(set(x for x in meta["sim_trait_type"].astype(str) if x)))
    print("  true h2:", sorted(pd.to_numeric(meta["true_h2"], errors="coerce").dropna().unique().tolist()))
print()

# 3. SIM
print("[3/7] Reading simulated Results.csv ...")
sr = []
sdirs = sorted([p for p in ROOT.iterdir() if p.is_dir() and re.fullmatch(r"Sim\d{3}", p.name)])
for i, d in enumerate(sdirs, 1):
    tt = ""
    if not meta.empty:
        m = meta[meta["sim_id"] == d.name]
        if not m.empty:
            tt = clean(m.iloc[0]["sim_trait_type"])
    sr.extend(extract_dataset(d, d.name, "simulated", tt))
    if i % 20 == 0 or i == len(sdirs):
        print(f"  scanned {i}/{len(sdirs)}")
sim_raw = pd.DataFrame(sr)
if not sim_raw.empty:
    sim_raw = sim_raw.rename(columns={"dataset_id": "sim_id"})
sim_raw.to_csv("Method21_raw_simulated.csv", index=False)
print("  Wrote Method21_raw_simulated.csv:", len(sim_raw))
print()

# 4. STABILITY
print("[4/7] Aggregating folds ...")
real_units = aggregate(real_raw, "phenotype")
sim_units = aggregate(sim_raw, "sim_id")
if not sim_units.empty and not meta.empty:
    sim_units = sim_units.merge(meta, on="sim_id", how="left", suffixes=("", "_meta"))
    if "sim_trait_type" in sim_units.columns:
        sim_units["trait_type"] = np.where(sim_units["sim_trait_type"].astype(str) != "", sim_units["sim_trait_type"], sim_units["trait_type"])
real_units.to_csv("Method21_unit_stability_real.csv", index=False)
sim_units.to_csv("Method21_unit_stability_simulated.csv", index=False)
print("  real units:", len(real_units))
print("  simulated units:", len(sim_units))
print()

# 5. METHOD + ACCURACY
print("[5/7] Method summaries + continuous known-truth accuracy ...")
real_m = method_summary(real_units, simulated=False)
sim_m = method_summary(sim_units, simulated=True)
acc_units, acc_m = accuracy(sim_units)
real_m.to_csv("Method21_method_stability_real.csv", index=False)
sim_m.to_csv("Method21_method_stability_simulated.csv", index=False)
acc_m.to_csv("Method21_simulation_accuracy.csv", index=False)
print("  Wrote method summary files")
print()

# 6. SAMPLE SIZE + ASSOCIATION
print("[6/7] Sample-size effects + stability/accuracy association ...")
n_eff = sample_size_effect(acc_units)
assoc = stability_accuracy_assoc(acc_units)
rvss = real_vs_sim(real_m, sim_m)
n_eff.to_csv("Method21_sample_size_effect.csv", index=False)
assoc.to_csv("Method21_stability_accuracy_association.csv", index=False)
rvss.to_csv("Method21_real_vs_sim_stability.csv", index=False)
print("  Wrote comparison files")
print()

# 7. RANKS + SUMMARY
print("[7/7] Ranking configurations + writing manuscript summary ...")
if not real_units.empty:
    real_units[real_units["n_valid_h2"] >= 4].sort_values(["fold_sd", "fold_mad", "fold_range"]).head(100).to_csv("Method21_top_stable_real.csv", index=False)
    real_units[real_units["n_valid_h2"] >= 2].sort_values(["fold_sd", "fold_range"], ascending=False).head(100).to_csv("Method21_top_unstable_real.csv", index=False)
else:
    pd.DataFrame().to_csv("Method21_top_stable_real.csv", index=False)
    pd.DataFrame().to_csv("Method21_top_unstable_real.csv", index=False)

text = summary_text(real_m, acc_m, n_eff, assoc, real_units)
Path("Method21_MAIN_MANUSCRIPT_SUMMARY.txt").write_text(text, encoding="utf-8")
print()
print(text)
print()
print("="*110)
print("AUDIT CHECKS")
print("="*110)
if not real_raw.empty:
    print("Unknown real rows:", int((real_raw["method"] == "Unknown").sum()), "/", len(real_raw))
    if (real_raw["method"] == "Unknown").any():
        print("Unknown real result directories:")
        print(real_raw.loc[real_raw["method"] == "Unknown", "result_file"].map(lambda x: Path(x).parent.name).value_counts().to_string())
if not real_units.empty:
    print("Real configuration-phenotype units:", len(real_units))
    print("Real n_valid_h2 distribution:")
    print(real_units["n_valid_h2"].value_counts(dropna=False).sort_index().to_string())
    print("Real stability-evaluable units:", int((real_units["stability_evaluable"] == 1).sum()), "/", len(real_units))
    print("Units with SD > range (should be 0):", int(((real_units["fold_sd"] > real_units["fold_range"]) & real_units["fold_sd"].notna() & real_units["fold_range"].notna()).sum()))
if not sim_raw.empty:
    print("Unknown simulated rows:", int((sim_raw["method"] == "Unknown").sum()), "/", len(sim_raw))
if not meta.empty:
    print("metadata missing true_h2:", int(pd.to_numeric(meta["true_h2"], errors="coerce").isna().sum()), "/", len(meta))
    print("metadata missing sample_size:", int(pd.to_numeric(meta["sample_size"], errors="coerce").isna().sum()), "/", len(meta))
    print("metadata missing trait type:", int((meta["sim_trait_type"].astype(str).str.strip() == "").sum()), "/", len(meta))
    print("pairable scenario_key:", int((meta["scenario_key"].astype(str) != "").sum()), "/", len(meta))
print()
print("IMPORTANT: fold SD/MAD/IQR/range = stability, NOT inferential uncertainty.")
print("Known-truth accuracy = continuous simulations only until binary scale harmonization.")
print("DONE.")
