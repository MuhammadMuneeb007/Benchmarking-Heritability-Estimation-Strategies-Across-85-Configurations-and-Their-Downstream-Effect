#!/usr/bin/env python3
"""FinalAnalysis16 -- print-only mechanistic audit of EXISTING simulation results.

READS input CSV and optional existing causal-effect CSV / GCTA GRM binary.
DOES NOT save outputs, create directories, plot files, caches, or rerun estimators.

Run from finalheritability/:
    python -B FinalAnalysis16.py --input FinalAnalysis04-XX/FinalAnalysis04-XX_units.csv

Optional:
    python -B FinalAnalysis16.py --grm-prefix /path/to/actual/grm/prefix
    python -B FinalAnalysis16.py --effects /path/to/causal_effects.csv
    python -B FinalAnalysis16.py --case-metadata /path/to/existing/case_counts.csv

Methods references (for interpretation, not fitted claims):
Visscher et al. (2014) PLoS Genet DOI 10.1371/journal.pgen.1004269
Ni et al. (2018) AJHG DOI 10.1016/j.ajhg.2018.03.021 [genetic correlation]
Lee et al. (2011) AJHG DOI 10.1016/j.ajhg.2011.02.002
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
from statistics import NormalDist
import numpy as np
import pandas as pd

REQUIRED = ["Domain", "Dataset", "TraitType", "Method", "ConfigID", "Scale",
            "Estimate", "Reference", "TargetN", "TrueH2", "CausalSNPs"]
GROUPS = {"GCTA-Heritability":"GCTA", "Gemma1-Heritability":"GEMMA",
          "DPR-Gemma-Heritability":"DPR-GEMMA", "LDSC-Heritability":"LDSC",
          "LDSC-Genotype-Heritability":"LDSC-genotype-ref",
          "LDAK-Calculated-Heritability":"LDAK-calculated",
          "LDAK-Precomputed-Heritability":"LDAK-precomputed",
          "Gemma2-Heritability":"GEMMA+GWAS",
          "Gemma-LDSC-Heritability":"GEMMA-LDSC",
          "LDpred-2-Heritability":"LDpred2"}


def section(name):
    print("\n" + "="*92)
    print(name)
    print("="*92)


def tab(frame, digits=5):
    if frame is None or frame.empty:
        print("[No eligible records; result not identified from supplied input]")
        return
    with pd.option_context("display.max_rows", 200, "display.max_columns", 100,
                           "display.width", 250, "display.max_colwidth", 60,
                           "display.expand_frame_repr", False):
        print(frame.to_string(index=False, float_format=lambda v:f"{v:.{digits}f}"))


def load_units(path):
    if not path.is_file():
        raise SystemExit(f"Missing input: {path}\nRun from finalheritability/ or supply --input.")
    d = pd.read_csv(path, low_memory=False)
    miss = [c for c in REQUIRED if c not in d.columns]
    if miss:
        raise SystemExit("Missing columns: " + ", ".join(miss))
    for c in ("Domain","TraitType","Scale"):
        d[c] = d[c].astype(str).str.strip().str.lower()
    for c in ("Dataset","Method","ConfigID"):
        d[c] = d[c].astype(str).str.strip()
    for c in ("Estimate","Reference","TargetN","TrueH2","CausalSNPs"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.loc[d.Domain.eq("simulated") & (
        (d.TraitType.eq("continuous") & d.Scale.eq("native_quantitative")) |
        (d.TraitType.eq("binary") & d.Scale.eq("liability_binary"))
    )].copy()
    d = d.dropna(subset=["Estimate","Reference","TargetN","TrueH2","CausalSNPs"])
    if d.empty:
        raise SystemExit("No simulated rows on correct continuous/binary scale.")
    if not np.allclose(d.Reference, d.TrueH2, atol=1e-8):
        raise SystemExit("Reference does not equal known generating heritability.")
    if d.duplicated(["Dataset","Method","ConfigID"]).any():
        raise SystemExit("Duplicate dataset/method/configuration rows: stop before pairing.")
    for c in ("TraitType","TargetN","TrueH2","CausalSNPs"):
        if (d.groupby("Dataset")[c].nunique() > 1).any():
            raise SystemExit(f"Conflicting {c} metadata inside dataset.")
    d["AE"] = (d.Estimate-d.Reference).abs()
    d["SignedError"] = d.Estimate-d.Reference
    d["MethodShort"] = d.Method.map(GROUPS).fillna(d.Method)
    d["Model"] = d.ConfigID.str.extract(r"h2model_standardized=([^|]+)", expand=False).fillna("").str.strip()
    d["Clump"] = d.ConfigID.str.extract(r"clumpprune=([^|]+)", expand=False).fillna("").str.strip().str.lower()
    d["Algorithm"] = np.select([
        d.Model.str.contains("REML",case=False),
        d.Model.str.contains(r"\bHE\b|Haseman",case=False,regex=True)
    ], ["REML","HE"], default="OTHER")
    d["GRM"] = np.select([
        d.Model.str.contains("standardized|standardised",case=False,regex=True),
        d.Model.str.contains("centered|centred",case=False,regex=True)
    ],["standardised","centred"],default="other/unspecified")
    d["CovSpec"] = np.select([
        d.Model.str.contains("covariate_pca",case=False),
        d.Model.str.contains("covariate",case=False)
    ],["covariates+PCA","covariates"],default="genotype only")
    return d


def stats(d, by, value="AE"):
    return d.groupby(by,dropna=False,as_index=False).agg(
        Datasets=("Dataset","nunique"), MAE=("AE","mean"),
        Bias=("SignedError","mean"), RMSE=("SignedError",lambda s:float(np.sqrt(np.mean(s*s)))),
        PctBoundary=("Estimate",lambda s:100*float(((s<0)|(s>1)).mean()))
    )


def report_method_families(d):
    section("1. COMPLETE METHOD FAMILIES: averaging configurations WITHIN each dataset")
    # Group-family means reflect mix of configs; not estimator-only efficiency.
    per_dataset = d.groupby(["Dataset","TraitType","TargetN","MethodShort"],as_index=False).agg(
        AE=("AE","mean"), SignedError=("SignedError","mean"),
        Configurations=("ConfigID","nunique"))
    output=per_dataset.groupby(["TraitType","TargetN","MethodShort"],as_index=False).agg(
        Datasets=("Dataset","nunique"), Configurations=("Configurations","median"),
        MAE=("AE","mean"), Bias=("SignedError","mean"))
    for (trait,n),part in output.groupby(["TraitType","TargetN"]):
        print(f"\n{trait}, target N={int(n)} (method-family, NOT matched algorithm)")
        tab(part[["MethodShort","Configurations","Datasets","MAE","Bias"]].sort_values("MAE"))
    print("CAUTION: group averages compare different CONFIGURATION MIXES and data sources.")


def report_best_per_n(d):
    section("2. EXACT CONFIGURATIONS: LDSC and GCTA at each target sample size")
    # LDSC simulation-only reference fixed; GCTA top un-clumped geno+cov no PCA.
    chosen=d.loc[((d.MethodShort=="LDSC") & d.Model.str.contains("SIM_REFERENCE",case=False))|
                 ((d.MethodShort=="GCTA") & d.Clump.eq("no") &
                  d.CovSpec.eq("covariates"))].copy()
    chosen=chosen.groupby(["Dataset","TraitType","TargetN","MethodShort"],as_index=False).agg(
        AE=("AE","mean"), SignedError=("SignedError","mean"))
    out=chosen.groupby(["TraitType","TargetN","MethodShort"],as_index=False).agg(
        Datasets=("Dataset","nunique"), MAE=("AE","mean"),Bias=("SignedError","mean"))
    tab(out)
    print("LDSC discovery GWAS N=10,000 remains FIXED as target N changes; target N is not its GWAS N.")
    return out


def subset_equal(d):
    # Constrain to genotype_covariate (NOT _pca), no clumping, REML where encoded,
    # GRM 'standardised' for GEMMA/DPR (GCTA GRM needs external verification).
    keep=(d.Clump.eq("no") & d.CovSpec.eq("covariates") & (
        (d.MethodShort.eq("GCTA") & d.Model.str.contains("^GCTA_",case=False,regex=True)) |
        (d.MethodShort.isin(["GEMMA","DPR-GEMMA"]) &
         d.Algorithm.eq("REML") & d.GRM.eq("standardised"))))
    a=d.loc[keep].copy()
    if a.duplicated(["Dataset","MethodShort"]).any():
        print("MULTIPLE matching configs; inspect below; no arbitrary averaging for like-for-like comparison")
        tab(a.groupby(["MethodShort","Model","Clump"],as_index=False).size())
        raise SystemExit("Ambiguous matched configurations.")
    return a


def report_like_for_like(d):
    section("3. MATCHED GCTA vs GEMMA vs DPR-GEMMA: REML, covariates, no PCA, no CP")
    a=subset_equal(d)
    print("Exact selected configuration models:")
    tab(a.groupby(["MethodShort","Model","Clump","CovSpec"],as_index=False).agg(
        Datasets=("Dataset","nunique")))
    meta=a[["Dataset","TraitType","TargetN","TrueH2","CausalSNPs"]].drop_duplicates("Dataset")
    e=a.pivot(index="Dataset",columns="MethodShort",values="AE")
    h=a.pivot(index="Dataset",columns="MethodShort",values="Estimate")
    missing=[m for m in ["GCTA","GEMMA","DPR-GEMMA"] if m not in e]
    if missing:
        print("MISSING:",missing, "-- skipping matched contrast")
        return
    e=e[["GCTA","GEMMA","DPR-GEMMA"]].dropna().join(meta.set_index("Dataset"))
    h=h[["GCTA","GEMMA","DPR-GEMMA"]].loc[e.index]
    e["Delta_GCTA_GEMMA"]=e.GCTA-e.GEMMA
    e["Delta_DPR_GEMMA"]=e["DPR-GEMMA"]-e.GEMMA
    e["Delta_GCTA_DPR"]=e.GCTA-e["DPR-GEMMA"]
    print("Negative contrast = first method had LOWER absolute error.")
    columns=["Datasets","GCTA_MAE","GEMMA_MAE","DPR_MAE","GCTA_minus_GEMMA","DPR_minus_GEMMA"]
    def grouped(keys):
        return e.groupby(keys,as_index=False).agg(
            Datasets=("GCTA","size"), GCTA_MAE=("GCTA","mean"),
            GEMMA_MAE=("GEMMA","mean"), DPR_MAE=("DPR-GEMMA","mean"),
            GCTA_minus_GEMMA=("Delta_GCTA_GEMMA","mean"),
            DPR_minus_GEMMA=("Delta_DPR_GEMMA","mean"))
    print("\nBy phenotype, target N and architecture:")
    tab(grouped(["TraitType","TargetN","CausalSNPs"]))
    print("\nBy EXACT generating h2 (5 original replicates in each class):")
    tab(grouped(["TraitType","TargetN","TrueH2","CausalSNPs"]))
    print("\nPooled descriptive differences (nested target N and architectures must not be assumed independent):")
    tab(pd.DataFrame([dict(PairedRows=len(e),Mean_GCTA_minus_GEMMA=e["Delta_GCTA_GEMMA"].mean(),
                           Median_GCTA_minus_GEMMA=e["Delta_GCTA_GEMMA"].median(),
                           Mean_DPR_minus_GEMMA=e["Delta_DPR_GEMMA"].mean(),
                           Median_DPR_minus_GEMMA=e["Delta_DPR_GEMMA"].median(),
                           GEMMA_DPR_exact_equal_pct=100*np.mean(np.isclose(
                              h.GEMMA,h["DPR-GEMMA"],rtol=0,atol=1e-9)))]))
    print("NOTE: same named 'standardised' GRM does NOT prove same numeric GRM matrix.")
    print("Only after verifying GRM matrix, SNPs and optimisation can residual differences be attributed to implementation.")


def report_factor_matches(d):
    section("4. ONE-FACTOR MATCHES within GEMMA / DPR: algorithm, GRM, clumping")
    use=d.loc[d.MethodShort.isin(["GEMMA","DPR-GEMMA"]) &
              d.Algorithm.isin(["HE","REML"]) &
              d.GRM.isin(["standardised","centred"])].copy()
    factors=[("Algorithm","HE","REML",["MethodShort","GRM","Clump","CovSpec"]),
             ("GRM","centred","standardised",["MethodShort","Algorithm","Clump","CovSpec"]),
             ("Clump","no","yes",["MethodShort","Algorithm","GRM","CovSpec"])]
    for factor,baseline,changed,other in factors:
        print(f"\n{factor}: AE({changed}) minus AE({baseline}) ; negative = {changed} better")
        idx=["Dataset"]+other
        sub=use.loc[use[factor].isin([baseline,changed])]
        if sub.duplicated(idx+[factor]).any():
            print("Duplicate factor-pair keys; cannot isolate cleanly. Check ConfigIDs.")
            continue
        piv=sub.pivot(index=idx,columns=factor,values="AE")
        if baseline not in piv or changed not in piv:
            print("No exact pairs for",factor)
            continue
        diff=(piv[changed]-piv[baseline]).dropna().reset_index(name="DeltaAE")
        if diff.empty:
            print("No matched pairs")
            continue
        meta=d[["Dataset","TraitType","TargetN","CausalSNPs","TrueH2"]].drop_duplicates("Dataset")
        diff=diff.merge(meta,on="Dataset",how="left",validate="many_to_one")
        tab(diff.groupby(["TraitType","TargetN"],as_index=False).agg(
            MatchedPairs=("DeltaAE","size"),Median_DeltaAE=("DeltaAE","median"),
            Mean_DeltaAE=("DeltaAE","mean"),
            PctChangedBetter=("DeltaAE",lambda x:100*(x<0).mean())))
    print("Matched change effects apply WITHIN GEMMA/DPR only; do not assign them to LDSC/GCTA cross-method differences.")


def report_spectrum(d, selected, gwas_n, train_fraction, grm_prefix, gwas_snp_count, mean_ld_score):
    section("5. SAMPLE-SIZE SPECTRUM and GREML theoretical SE (Visscher et al., 2014)")
    print(f"LDSC discovery-GWAS N = {gwas_n} (fixed; NOT equal to target N).")
    print("Observed comparisons below use TARGET N=750 or 5000; LDSC's GWAS sample remains fixed.")
    nvals=[300,600,750,1000,2000,4000,5000,7500,10000]
    var_grm=None
    if grm_prefix:
        p=Path(grm_prefix)
        vals_path=Path(str(p)+".grm.bin")
        ids_path=Path(str(p)+".grm.id")
        if not (vals_path.is_file() and ids_path.is_file()):
            print("GRM NOT FOUND: expected",vals_path,"and",ids_path)
        else:
            n=sum(1 for _ in ids_path.open("r"))
            z=np.fromfile(vals_path,dtype="<f4")
            if len(z)!=n*(n+1)//2:
                print("GRM size mismatch; skipping", vals_path)
            else:
                diagonal_idx=np.arange(n,dtype=np.int64)*(np.arange(n,dtype=np.int64)+3)//2
                diag=z[diagonal_idx].astype(np.float64)
                pairs=n*(n-1)//2
                mu=(z.sum(dtype=np.float64)-diag.sum())/pairs
                variance=(np.square(z.astype(np.float64)).sum()-np.square(diag).sum())/pairs-mu**2
                var_grm=float(max(variance,0))
                print(f"GRM individuals={n}; offdiagonal GRM variance={var_grm:.9g}; M_e ~ {1/var_grm if var_grm else float('nan'):.2f}")
    rows=[]
    for full_n in nvals:
        training_n=int(round(full_n*train_fraction))
        relative_600=600/training_n
        theory=(np.sqrt(2)/(training_n*np.sqrt(var_grm)) if var_grm and var_grm>0 else np.nan)
        rows.append(dict(TargetN=full_n,TrainingN=training_n,
                         Relative_SE_to_training600=relative_600,
                         Approx_GREML_SE_using_given_GRM=theory,
                         LDSC_GWAS_N=gwas_n))
    tab(pd.DataFrame(rows))
    print("GREML equation: SE ~= sqrt(2) / [N * sqrt(var(off-diagonal GRM))].")
    print("Its regime of validity depends on model, weak relatedness and variance components.")
    print("This curve is expected SE, NOT empirical MAE or a guaranteed observed improvement.")
    print("\nLDSC sample-size spectrum: THEORY ONLY (assuming equal M, mean LD, h2, intercept):")
    # E[chi^2] ~ 1 + (N_GWAS * h2/M) * mean(LD score) + confounding intercept.
    # Relative heritability-slope scaling N / original N is NOT estimator power/precision.
    theoretical=[]
    for n in [1000,2500,5000,7500,gwas_n]:
        row={"GWAS_N":n,"Relative_chi2_LD_slope":n/gwas_n}
        if gwas_snp_count and mean_ld_score:
            for h in (0.05,0.30,0.70):
                row[f"ExpectedChi2_h2_{h}"]=1+n*h*mean_ld_score/gwas_snp_count
        theoretical.append(row)
    tab(pd.DataFrame(theoretical))
    print("This is expected LDSC slope scaling, NOT observed LDSC power, SE, or accuracy.")
    print("Optional: --gwas-snp-count M --mean-ld-score L to print expected chi-square")
    print("under a simplified intercept=1 no-confounding LDSC approximation.")
    print("\nObserved accuracy by target N (same LDSC discovery input):")
    tab(selected.loc[selected.TraitType.eq("continuous"),["TargetN","MethodShort","Datasets","MAE","Bias"]])
    print("\nLDSC GWAS sample-size experiment: NOT available for GWAS N=1000/2500/5000/7500")
    print("from the existing results. Must rerun GWAS/LDSC at those N to measure it.")
    print("Ni et al. 2018: GREML vs LDSC GENETIC CORRELATION; do not quote 150k vs 400k as heritability proof.")


def liability_c(k,p):
    norm=NormalDist()
    t=norm.inv_cdf(1-k)
    z=norm.pdf(t)
    return (k*(1-k))**2/(p*(1-p)*z*z)


def report_binary(d, selected, k, train_fraction, gwas_n, case_meta):
    section("6. BINARY INFORMATION and OBSERVED-to-LIABILITY CONVERSION")
    c=liability_c(k,k)
    print(f"Population prevalence K={k:.5f}; if P=K, liability multiplier C={c:.6f}.")
    print("General C = K^2(1-K)^2 / [P(1-P)*phi(Phi^-1(1-K))^2].")
    print("For actual trait-specific conversion P is the observed case fraction; use the sample supplying the estimator.")
    tab(pd.DataFrame([dict(TargetN=n,Expected_Full_Cases=n*k,
                           Expected_Training_Cases=n*train_fraction*k,
                           LDSC_Expected_Discovery_Cases=gwas_n*k,
                           C_when_P_equals_K=c) for n in [750,5000]]))
    if case_meta:
        p=Path(case_meta)
        if p.suffix.lower()==".csv": cm=pd.read_csv(p)
        elif p.suffix.lower() in (".xlsx",".xls"):
            try:cm=pd.read_excel(p)
            except ImportError as ex:raise SystemExit("Reading .xlsx requires a supported Excel reader in HPC: "+str(ex))
        else:raise SystemExit("--case-metadata must be .csv or .xlsx")
        print("\nProvided case metadata columns:",list(cm.columns))
        columns={x.lower():x for x in cm.columns}
        nc=next((columns[x] for x in ["cases","casen","n_cases","num_cases","case_count","targetcases"] if x in columns),None)
        np_col=next((columns[x] for x in ["n","targetn","sample_size","samplesize","n_total","total"] if x in columns),None)
        if nc and np_col:
            cm["_P"]=pd.to_numeric(cm[nc],errors="coerce")/pd.to_numeric(cm[np_col],errors="coerce")
            cm["_C"]=cm["_P"].apply(lambda p:liability_c(k,p) if np.isfinite(p) and 0<p<1 else np.nan)
            print("Actual case fractions and liability factors from supplied metadata:")
            tab(cm.groupby(np_col,as_index=False).agg(Records=("_P","count"),
                Median_CaseFraction=("_P","median"),Median_ConversionFactor=("_C","median")))
        else: print("Cannot infer case columns; use named case and N columns; no false precision.")
    else:
        print("Actual per-trait case fractions NOT in units CSV; the table above gives expectations only.")
    print("\nObserved BINARY liability-scale MAE (already converted in FinalAnalysis04):")
    tab(selected.loc[selected.TraitType.eq("binary"),["TargetN","MethodShort","Datasets","MAE","Bias"]])
    n5=selected.loc[(selected.TraitType.eq("continuous")) & (selected.TargetN.eq(5000)) &
                    (selected.MethodShort.eq("GCTA"))]
    lb=selected.loc[(selected.TraitType.eq("binary")) & (selected.TargetN.eq(5000)) &
                    (selected.MethodShort.eq("LDSC"))]
    if not n5.empty and not lb.empty:
        a=float(n5.MAE.iloc[0]);b=float(lb.MAE.iloc[0])
        print(f"\nIllustrative (NOT valid as prediction): {a:.5f} continuous GCTA MAE * {c:.4f} = {a*c:.5f}; observed binary LDSC MAE={b:.5f}")
    print("WARNING: multiplying a CONTINUOUS-trait MAE by C does NOT predict BINARY-trait MAE.")
    print("Dichotomisation, sampling information, estimator bias and liability transformation all change.")


def report_architecture(d,effects):
    section("7. SIMULATION ARCHITECTURE / model-assumption diagnostics")
    print("Observed error stratified by genetic architecture, generating heritability, phenotype and N:")
    choices=d.loc[((d.MethodShort=="LDSC") & d.Model.str.contains("SIM_REFERENCE",case=False)) |
                  ((d.MethodShort=="GCTA") & d.Clump.eq("no") & d.CovSpec.eq("covariates"))]
    tab(choices.groupby(["TraitType","TargetN","CausalSNPs","TrueH2","MethodShort"],as_index=False).agg(
        Datasets=("Dataset","nunique"),MAE=("AE","mean"),Bias=("SignedError","mean")))
    if not effects:
        print("\nActual per-SNP effect/MAF/LD distribution cannot be reconstructed from heritability outputs.")
        print("OPTIONAL: --effects existing CSV with beta, MAF and LD_score for actual assumption tests.")
        print("Simulation formula uses standardised genotypes and Normal genetic effects; ")
        print("this is broadly consistent with a standardised-GRM model but does NOT prove")
        print("heritability independence from LD or guarantee either model is exactly specified.")
        return
    e=pd.read_csv(effects,low_memory=False)
    lookup={c.lower().replace("_",""):c for c in e.columns}
    def col(*keys):return next((lookup[k.lower().replace("_","")] for k in keys if k.lower().replace("_","") in lookup),None)
    b=col("beta","effect","effect_size"); maf=col("maf","minor_allele_frequency"); ld=col("ldscore","ld_score","l2")
    if not b:
        print("Effect table missing beta/effect column; columns are:",list(e))
        return
    e["_b2"]=pd.to_numeric(e[b],errors="coerce")**2
    if maf:
        e["_maf"]=pd.to_numeric(e[maf],errors="coerce")
        print(f"beta^2 versus MAF correlation = {e[['_b2','_maf']].corr().iloc[0,1]:.5f}")
        e["_unstandardised_contrib_proxy"]=2*e._maf*(1-e._maf)*e._b2
        print("If effects are coded on RAW dosage scale, 2p(1-p)beta^2 is a diagonal variance proxy.")
        print("Effects for STANDARDISED dosage imply beta^2 proxy instead.")
    if ld:
        e["_ld"]=pd.to_numeric(e[ld],errors="coerce")
        print(f"beta^2 versus LD score correlation = {e[['_b2','_ld']].corr().iloc[0,1]:.5f}")
        if maf:print(f"MAF versus LD score correlation = {e[['_maf','_ld']].corr().iloc[0,1]:.5f}")
    print("NOTE: under correlated SNPs, variance of the total genetic score includes LD covariance cross-terms.")
    print("Correlations alone do not establish that a model is correctly specified.")


def end_summary():
    section("8. WHAT IS IDENTIFIED vs WHAT IS NOT")
    for msg in [
      "IDENTIFIED: GCTA, GEMMA, DPR matched configuration errors on same simulated datasets.",
      "IDENTIFIED: HE/REML, centred/standardised GRM, and clumping contrasts within eligible families.",
      "IDENTIFIED: target-size dependence, binary/continuous and causal-SNP architecture dependence.",
      "THEORY: GREML SE roughly scales as 1/N only under its stated approximations.",
      "NOT IDENTIFIED: LDSC power at GWAS N other than 10,000; no such GWAS runs in source.",
      "NOT IDENTIFIED: equal-input LDSC vs GCTA algorithm-only effect; their inputs differ.",
      "NOT IDENTIFIED: actual LDSC chi-square filtering / reference quality without LDSC logs.",
      "NOT IDENTIFIED: actual MAF/LD causal-effect distribution without genotype/effect data.",
      "CAUTION: REML/HE model and GRM effects explain within-family variation, not every cross-family gap.",
      "NO OUTPUT FILES WERE WRITTEN. All calculations were in memory and printed to stdout."
    ]:print(" - "+msg)


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",type=Path,default=Path("FinalAnalysis04-XX/FinalAnalysis04-XX_units.csv"))
    ap.add_argument("--grm-prefix",default="",help="existing GCTA GRM file prefix, WITHOUT .grm.bin extension")
    ap.add_argument("--effects",default="",help="optional existing CSV with beta/MAF/LD score")
    ap.add_argument("--case-metadata",default="",help="optional existing CSV/XLSX with case counts and N")
    ap.add_argument("--population-prevalence",type=float,default=0.1)
    ap.add_argument("--discovery-n",type=int,default=10000)
    ap.add_argument("--gwas-snp-count",type=int,default=0,help="optional regression M for LDSC expected chi-square")
    ap.add_argument("--mean-ld-score",type=float,default=0,help="optional empirical mean LD score")
    ap.add_argument("--train-fraction",type=float,default=0.8)
    args=ap.parse_args()
    if not (0<args.population_prevalence<1 and 0<args.train_fraction<=1):
        ap.error("invalid prevalence or train fraction")
    print("FINALANALYSIS16 | READ-ONLY, PRINT-ONLY | Source:",args.input)
    print("This program uses EXISTING results. No GWAS/estimator reruns. No file outputs.")
    d=load_units(args.input)
    print(f"Loaded {len(d):,} valid configuration rows across {d.Dataset.nunique()} simulated datasets")
    print("Observed configuration counts per method:")
    tab(d.groupby("MethodShort",as_index=False).agg(
        Configurations=("ConfigID","nunique"),DatasetCount=("Dataset","nunique")))
    report_method_families(d)
    chosen=report_best_per_n(d)
    report_like_for_like(d)
    report_factor_matches(d)
    report_spectrum(d,chosen,args.discovery_n,args.train_fraction,args.grm_prefix,args.gwas_snp_count,args.mean_ld_score)
    report_binary(d,chosen,args.population_prevalence,args.train_fraction,args.discovery_n,args.case_metadata)
    report_architecture(d,args.effects)
    end_summary()

if __name__=="__main__":
    main()
