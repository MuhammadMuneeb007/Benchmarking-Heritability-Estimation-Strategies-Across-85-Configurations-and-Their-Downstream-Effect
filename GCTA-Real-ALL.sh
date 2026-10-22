#!/bin/bash
#SBATCH --job-name=GCTA_REAL
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/GCTA_REAL_%A_%a.out
#SBATCH --error=logs/GCTA_REAL_%A_%a.err

source ~/.bashrc
conda activate leopard-sdm

cd /data/ascher01/uqmmune1/finalheritability || exit 1

PHENOTYPES=(
"asthma"
"body_mass_index_bmi"
"depression"
"gastro_oesophageal_reflux_gord_gastric_reflux"
"high_cholesterol"
"hypertension"
"hypothyroidism_myxoedema"
"irritable_bowel_syndrome"
"migraine"
"osteoarthritis"
)

PHENO=${PHENOTYPES[$((SLURM_ARRAY_TASK_ID-1))]}

if [ "$PHENO" = "body_mass_index_bmi" ]; then
    TYPE="continuous"
else
    TYPE="binary"
fi

echo "============================================================"
echo "GCTA REAL"
echo "Phenotype: $PHENO"
echo "Type: $TYPE"
echo "Start: $(date)"
echo "============================================================"

python - "$PHENO" "$TYPE" <<'PY'
import sys
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

pheno = sys.argv[1]
ptype = sys.argv[2]

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

good = 0
bad = 0
skipped = 0

for fold in range(5):

    fd = Path(pheno) / f"Fold_{fold}"
    config_idx = 0

    print("\n" + "="*80)
    print(pheno, "FOLD", fold)
    print("="*80)

    for method in METHODS:

        hf = fd / method / "Results.csv"

        if not hf.exists():
            continue

        try:
            d = pd.read_csv(hf)
        except Exception as e:
            print("READ ERROR:", hf, e)
            continue

        if d.empty or "h2" not in d.columns:
            continue

        if "h2model" not in d:
            d["h2model"] = "NA"

        if "clumpprune" not in d:
            d["clumpprune"] = "NA"

        if "numberofpca" not in d:
            d["numberofpca"] = 6

        groups = d.groupby(
            ["h2model", "clumpprune", "numberofpca"],
            dropna=False,
            sort=True
        )

        for (model, cp, pca), g in groups:

            hv = pd.to_numeric(g["h2"], errors="coerce").dropna()

            if hv.empty:
                continue

            h2 = float(hv.mean())

            if not np.isfinite(h2) or not (0 < h2 < 1):
                skipped += 1
                continue

            model = str(model)
            cp = str(cp)

            try:
                pca = str(int(float(pca)))
            except Exception:
                pca = "6"

            safe = re.sub(
                r"[^A-Za-z0-9_-]",
                "_",
                f"{model}_{cp}"
            )

            outdir = (
                fd /
                "GCTA-SBLUP-PRS" /
                f"{config_idx}_{safe}"
            )

            rf = outdir / "Results.csv"

            # Resume only genuinely complete files.
            if rf.exists():
                try:
                    x = pd.read_csv(rf)

                    valid = (
                        len(x) == 10
                        and "pvalue_threshold" in x.columns
                        and x["pvalue_threshold"].nunique() == 10
                        and "h2" in x.columns
                        and np.allclose(
                            pd.to_numeric(
                                x["h2"],
                                errors="coerce"
                            ).dropna(),
                            h2,
                            rtol=1e-7,
                            atol=1e-10
                        )
                    )

                    if valid:
                        print(
                            "COMPLETE:",
                            pheno,
                            fold,
                            config_idx,
                            method,
                            h2
                        )
                        good += 1
                        config_idx += 1
                        continue
                except Exception:
                    pass

                print("REMOVING BAD/INCOMPLETE:", outdir)
                shutil.rmtree(outdir, ignore_errors=True)

            cmd = [
                sys.executable,
                "GCTA-PRS-Final.py",
                pheno,
                str(fold),
                str(config_idx),
                ptype,
                str(h2),
                method,
                model,
                cp,
                pca,
            ]

            print(
                "RUN:",
                pheno,
                "fold", fold,
                "config", config_idx,
                "h2", h2,
                method,
                model,
                cp
            )

            r = subprocess.run(cmd)

            valid = False

            if r.returncode == 0 and rf.exists():
                try:
                    x = pd.read_csv(rf)
                    valid = (
                        len(x) == 10
                        and "pvalue_threshold" in x.columns
                        and x["pvalue_threshold"].nunique() == 10
                    )
                except Exception:
                    valid = False

            if valid:
                good += 1
            else:
                bad += 1
                print("FAILED:", outdir)

            config_idx += 1

print("\n============================================================")
print("PHENOTYPE:", pheno)
print("VALID:", good)
print("INVALID H2 SKIPPED:", skipped)
print("FAILED:", bad)
print("============================================================")

if bad:
    sys.exit(2)
PY

echo "End: $(date)"
