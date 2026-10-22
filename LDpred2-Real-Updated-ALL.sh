#!/bin/bash
#SBATCH --job-name=LDPRED2_REAL
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/LDPRED2_REAL_%A_%a.out
#SBATCH --error=logs/LDPRED2_REAL_%A_%a.err

source ~/.bashrc
conda activate leopard-sdm

cd /data/ascher01/uqmmune1/finalheritability || exit 1
mkdir -p logs

# Prevent nested BLAS parallelism.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

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

echo "============================================================"
echo "TRUE LDPRED2 UPDATED ANALYSIS"
echo "PHENOTYPE: $PHENO"
echo "CPUS: $SLURM_CPUS_PER_TASK"
echo "START: $(date)"
echo "============================================================"

python - "$PHENO" <<'PY'
import sys
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

PHENO = sys.argv[1]

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

ptype = "continuous" if PHENO == "body_mass_index_bmi" else "binary"

print("Phenotype type:", ptype)

total_good = 0
total_failed = 0
total_invalid = 0

for fold in range(5):

    fd = Path(PHENO) / f"Fold_{fold}"

    print()
    print("=" * 100)
    print(PHENO, "FOLD", fold)
    print("=" * 100)

    if not fd.exists():
        print("MISSING FOLD:", fd)
        continue

    config_idx = 0

    for method in METHODS:

        hf = fd / method / "Results.csv"

        if not hf.exists():
            print("MISSING HERITABILITY:", hf)
            continue

        try:
            d = pd.read_csv(hf)
        except Exception as e:
            print("READ ERROR:", hf, e)
            continue

        if d.empty or "h2" not in d.columns:
            continue

        if "h2model" not in d.columns:
            d["h2model"] = "NA"

        if "clumpprune" not in d.columns:
            d["clumpprune"] = "NA"

        if "numberofpca" not in d.columns:
            d["numberofpca"] = 6

        groups = d.groupby(
            ["h2model", "clumpprune", "numberofpca"],
            dropna=False,
            sort=True
        )

        for (model, cp, pca), g in groups:

            hv = pd.to_numeric(
                g["h2"],
                errors="coerce"
            ).dropna()

            if hv.empty:
                continue

            h2 = float(hv.mean())

            # Downstream LDpred2 requires valid h2.
            if not np.isfinite(h2) or h2 <= 0 or h2 >= 1:
                print(
                    "SKIP INVALID H2:",
                    method,
                    model,
                    h2
                )
                total_invalid += 1
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
                "LDpred2-PRS-Updated" /
                f"{config_idx}_{safe}"
            )

            rf = outdir / "Results.csv"

            # -------------------------------------------------
            # Resume completed configurations
            # -------------------------------------------------

            if rf.exists():

                try:
                    old = pd.read_csv(rf)

                    good = (
                        len(old) == 10
                        and "grid_p" in old.columns
                        and old["grid_p"].nunique() == 10
                        and "grid_h2" in old.columns
                        and np.allclose(
                            pd.to_numeric(
                                old["grid_h2"],
                                errors="coerce"
                            ).dropna(),
                            h2,
                            atol=1e-10,
                            rtol=0
                        )
                    )

                    if good:
                        print(
                            "ALREADY COMPLETE:",
                            PHENO,
                            "Fold", fold,
                            "Config", config_idx
                        )
                        total_good += 1
                        config_idx += 1
                        continue

                except Exception:
                    pass

                print("REMOVING INCOMPLETE:", outdir)
                shutil.rmtree(outdir, ignore_errors=True)

            cmd = [
                sys.executable,
                "LDpred-2-PRS-Final-real-updated.py",
                PHENO,
                str(fold),
                str(config_idx),
                ptype,
                str(h2),
                method,
                model,
                cp,
                pca,
            ]

            print()
            print(
                "RUN:",
                PHENO,
                "Fold", fold,
                "Config", config_idx,
                "h2 =", h2,
                "method =", method,
                "model =", model
            )

            r = subprocess.run(cmd)

            success = False

            if r.returncode == 0 and rf.exists():

                try:
                    rr = pd.read_csv(rf)

                    success = (
                        len(rr) == 10
                        and "grid_p" in rr.columns
                        and rr["grid_p"].nunique() == 10
                        and "grid_h2" in rr.columns
                        and np.allclose(
                            pd.to_numeric(
                                rr["grid_h2"],
                                errors="coerce"
                            ).dropna(),
                            h2,
                            atol=1e-10,
                            rtol=0
                        )
                    )

                except Exception:
                    success = False

            if success:
                print("SUCCESS")
                total_good += 1
            else:
                print("FAILED")
                total_failed += 1

            config_idx += 1

    print(
        "FINISHED",
        PHENO,
        "Fold",
        fold,
        "valid configs =",
        config_idx
    )

print()
print("=" * 100)
print("PHENOTYPE COMPLETE:", PHENO)
print("GOOD:", total_good)
print("INVALID H2 SKIPPED:", total_invalid)
print("FAILED:", total_failed)
print("=" * 100)

if total_failed:
    sys.exit(2)
PY

echo "END: $(date)"
