#!/bin/bash
#SBATCH --job-name=LDPRED2_SIM
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/LDPRED2_SIM_%A_%a.out
#SBATCH --error=logs/LDPRED2_SIM_%A_%a.err

source ~/.bashrc
conda activate leopard-sdm
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /data/ascher01/uqmmune1/finalheritability || exit 1
mkdir -p logs

SIM=$(printf "Sim%03d" "$SLURM_ARRAY_TASK_ID")

echo "======================================================"
echo "LDpred2 simulated PRS"
echo "SIMULATION: $SIM"
echo "START: $(date)"
echo "======================================================"

python - "$SIM" <<'PY'
import sys
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

SIM = sys.argv[1]

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

simdir = Path(SIM)

if not simdir.exists():
    print("SIMULATION DOES NOT EXIST:", SIM)
    sys.exit(0)

total_complete = 0
total_failed = 0
total_invalid = 0

for fold in range(5):

    fd = simdir / f"Fold_{fold}"

    if not fd.exists():
        print("MISSING:", fd)
        continue

    famfile = fd / "train_data.QC.fam"

    if not famfile.exists():
        print("MISSING:", famfile)
        continue

    # ---------------------------------------------------------
    # Determine phenotype type
    # ---------------------------------------------------------

    fam = pd.read_csv(
        famfile,
        sep=r"\s+",
        header=None
    )

    vals = set(
        pd.to_numeric(
            fam.iloc[:, 5],
            errors="coerce"
        ).dropna().unique()
    )

    if vals and vals.issubset({1, 2, 1.0, 2.0}):
        ptype = "binary"
    else:
        ptype = "continuous"

    print()
    print("======================================================")
    print("SIM:", SIM)
    print("FOLD:", fold)
    print("TYPE:", ptype)
    print("TRAIN N:", len(fam))
    print("======================================================")

    config_idx = 0

    for method in METHODS:

        hfile = fd / method / "Results.csv"

        if not hfile.exists():
            print("MISSING HERITABILITY:", hfile)
            continue

        try:
            df = pd.read_csv(hfile)
        except Exception as e:
            print("READ ERROR:", hfile, e)
            continue

        if df.empty or "h2" not in df.columns:
            continue

        if "h2model" not in df.columns:
            df["h2model"] = "NA"

        if "clumpprune" not in df.columns:
            df["clumpprune"] = "NA"

        if "numberofpca" not in df.columns:
            df["numberofpca"] = 6

        groups = df.groupby(
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

            # LDpred2 downstream h2 must be valid
            if (
                not np.isfinite(h2)
                or h2 <= 0
                or h2 >= 1
            ):
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
                "LDpred2-PRS-v2" /
                f"{config_idx}_{safe}"
            )

            resultfile = outdir / "Results.csv"

            # -------------------------------------------------
            # Resume support:
            # skip ONLY genuinely complete 10-grid results.
            # -------------------------------------------------

            if resultfile.exists():

                try:
                    old = pd.read_csv(resultfile)

                    correct_h2 = (
                        len(old) == 10
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

                    correct_grid = (
                        "grid_p" in old.columns
                        and old["grid_p"].nunique() == 10
                    )

                    if correct_h2 and correct_grid:

                        print(
                            "COMPLETE:",
                            SIM,
                            "Fold", fold,
                            "Config", config_idx,
                            "h2", h2
                        )

                        total_complete += 1
                        config_idx += 1
                        continue

                except Exception:
                    pass

                # incomplete result: rerun configuration
                print("REMOVING INCOMPLETE:", outdir)

                if outdir.exists():
                    shutil.rmtree(outdir)

            # -------------------------------------------------
            # Run patched/cached LDpred2
            # -------------------------------------------------

            cmd = [
                sys.executable,
                "LDpred-2-PRS-Final-cached.py",
                SIM,
                str(fold),
                str(config_idx),
                ptype,
                str(h2),
                method,
                model,
                cp,
                pca
            ]

            print()
            print(
                "RUN:",
                SIM,
                "Fold", fold,
                "Config", config_idx,
                "h2", h2,
                method,
                model,
                cp
            )

            result = subprocess.run(cmd)

            # -------------------------------------------------
            # Validate result
            # -------------------------------------------------

            success = False

            if result.returncode == 0 and resultfile.exists():

                try:
                    rr = pd.read_csv(resultfile)

                    success = (
                        len(rr) == 10
                        and
                        "grid_p" in rr.columns
                        and
                        rr["grid_p"].nunique() == 10
                        and
                        "grid_h2" in rr.columns
                    )

                except Exception:
                    success = False

            if success:

                print(
                    "SUCCESS:",
                    SIM,
                    "Fold", fold,
                    "Config", config_idx
                )

                total_complete += 1

            else:

                print(
                    "FAILED:",
                    SIM,
                    "Fold", fold,
                    "Config", config_idx
                )

                total_failed += 1

            config_idx += 1

    print()
    print(
        "FINISHED",
        SIM,
        "Fold",
        fold,
        "valid config count =",
        config_idx
    )

print()
print("======================================================")
print("PHENOTYPE FINISHED:", SIM)
print("COMPLETE:", total_complete)
print("INVALID H2 SKIPPED:", total_invalid)
print("FAILED:", total_failed)
print("END:", pd.Timestamp.now())
print("======================================================")

if total_failed > 0:
    sys.exit(2)
PY

echo "SLURM JOB FINISHED: $(date)"
