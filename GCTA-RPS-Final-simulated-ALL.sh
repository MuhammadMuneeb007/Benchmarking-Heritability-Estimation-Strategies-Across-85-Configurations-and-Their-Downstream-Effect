#!/bin/bash
#SBATCH --job-name=GCTA_PRS_SIM
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/GCTA_PRS_SIM_%A_%a.out
#SBATCH --error=logs/GCTA_PRS_SIM_%A_%a.err

source ~/.bashrc
conda activate leopard-sdm

cd /data/ascher01/uqmmune1/finalheritability || exit 1

SIM=$(printf "Sim%03d" "$SLURM_ARRAY_TASK_ID")

echo "=============================================="
echo "SIMULATION: $SIM"
echo "START: $(date)"
echo "=============================================="

python - "$SIM" <<'PY'
import sys
import re
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np

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
    print("MISSING SIMULATION:", SIM)
    sys.exit(0)

for fold in range(5):

    folddir = simdir / f"Fold_{fold}"

    if not folddir.exists():
        print("MISSING FOLD:", folddir)
        continue

    # Determine binary / continuous from FAM phenotype
    famfile = folddir / "train_data.QC.fam"

    if not famfile.exists():
        print("MISSING FAM:", famfile)
        continue

    fam = pd.read_csv(famfile, sep=r"\s+", header=None)
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

    print("\n==============================================")
    print("SIM:", SIM, "FOLD:", fold, "TYPE:", ptype)
    print("==============================================")

    config_idx = 0

    for method in METHODS:

        resultfile = folddir / method / "Results.csv"

        if not resultfile.exists():
            print("MISSING:", resultfile)
            continue

        try:
            df = pd.read_csv(resultfile)
        except Exception as e:
            print("READ ERROR:", resultfile, e)
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
            dropna=False
        )

        for (h2model, clumpprune, npca), g in groups:

            h = pd.to_numeric(
                g["h2"],
                errors="coerce"
            ).dropna()

            if h.empty:
                continue

            h2 = float(h.mean())

            # GCTA-SBLUP only for valid h2
            if not np.isfinite(h2) or h2 <= 0 or h2 >= 1:
                print(
                    "SKIP INVALID:",
                    method,
                    h2model,
                    "h2=",
                    h2
                )
                continue

            try:
                npca = str(int(float(npca)))
            except:
                npca = "6"

            h2model = str(h2model)
            clumpprune = str(clumpprune)

            safe = re.sub(
                r"[^A-Za-z0-9_-]",
                "_",
                f"{h2model}_{clumpprune}"
            )

            outfile = (
                folddir /
                "GCTA-SBLUP-PRS" /
                f"{config_idx}_{safe}" /
                "Results.csv"
            )

            # Resume safely
            if outfile.exists():
                try:
                    old = pd.read_csv(outfile)

                    if (
                        len(old) == 10
                        and old["pvalue_threshold"].nunique() == 10
                    ):
                        print(
                            "ALREADY COMPLETE:",
                            SIM,
                            fold,
                            config_idx
                        )
                        config_idx += 1
                        continue
                except:
                    pass

            cmd = [
                sys.executable,
                "GCTA-PRS-Final.py",
                SIM,
                str(fold),
                str(config_idx),
                ptype,
                str(h2),
                method,
                h2model,
                clumpprune,
                npca,
            ]

            print(
                "RUN:",
                SIM,
                "Fold",
                fold,
                "Config",
                config_idx,
                "h2",
                h2,
                method,
                h2model
            )

            r = subprocess.run(cmd)

            if r.returncode != 0:
                print("FAILED:", " ".join(cmd))

            config_idx += 1

    print(
        "FINISHED",
        SIM,
        "Fold",
        fold,
        "configs=",
        config_idx
    )

print("\nDONE:", SIM)
PY

echo "END: $(date)"
