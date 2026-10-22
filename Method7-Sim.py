import os
import sys
import pandas as pd
import numpy as np
import subprocess

filedirec = sys.argv[1]
foldnumber = sys.argv[2] if len(sys.argv) > 2 else "0"

GWAS = filedirec + os.sep + filedirec + ".gz"
df = pd.read_csv(GWAS, compression="gzip", sep=r"\s+")

if "BETA" in df.columns.to_list():
    df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'INFO', 'MAF']]
    df['Z'] = df['BETA'] / df['SE']
else:
    df["BETA"] = np.log(df["OR"])
    df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'INFO', 'MAF']]
    df['Z'] = df['BETA'] / df['SE']

df.to_csv(filedirec + os.sep + filedirec + ".ldsc.txt", sep="\t", index=False)


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def readpve(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if "Total Observed scale h2" in line:
                pve_estimate = line.split(":")[-1].split("(")[0].strip()
                print("PVE Estimate:", pve_estimate)
                return pve_estimate
    return np.nan


def readpve_se(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if "Total Observed scale h2" in line:
                if "(" in line and ")" in line:
                    pve_se = line.split("(")[-1].replace(")", "").strip()
                    print("PVE SE:", pve_se)
                    return pve_se
    return np.nan


def readvariants(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if "After merging with regression SNP LD" in line:
                pve_estimate = line.split(",")[-1].strip().split(" ")[0]
                print("Variants:", pve_estimate)
                return pve_estimate
    return np.nan


reference_dir = "LDSCFILES_SIM" + os.sep + "Reference5000"

missing = []
for chromosome in range(1, 23):
    prefix = reference_dir + os.sep + str(chromosome) + ".l2"
    for suffix in [".ldscore.gz", ".M", ".M_5_50"]:
        if not os.path.exists(prefix + suffix):
            missing.append(prefix + suffix)

if missing:
    raise RuntimeError(
        "Simulation LDSC reference is incomplete. Missing: " +
        ", ".join(missing[:10])
    )

outdir = (
    "LDSC_SIM" + os.sep + filedirec +
    os.sep + "Fold_" + foldnumber +
    os.sep + "Method7"
)
create_directory(outdir)

outprefix = outdir + os.sep + "ldsc"

command = [
    sys.executable,
    "ldsc.py",
    "--h2", filedirec + os.sep + filedirec + ".ldsc.txt",
    "--ref-ld-chr", reference_dir + os.sep,
    "--w-ld-chr", reference_dir + os.sep,
    "--out", outprefix
]

print("Running Method7-Sim:")
print(" ".join(command))

status = subprocess.call(command)
if status != 0:
    raise RuntimeError("Method7-Sim LDSC failed with exit code " + str(status))

log_file = outprefix + ".log"
if not os.path.exists(log_file):
    raise RuntimeError("Missing LDSC log: " + log_file)

h2 = readpve(log_file)
h2_se = readpve_se(log_file)
variants = readvariants(log_file)

print("METHOD7-SIM RESULT:", h2, h2_se, variants)

result_dir = (
    filedirec + os.sep + "Fold_" + foldnumber +
    os.sep + "LDSC-Heritability"
)
create_directory(result_dir)

result = pd.DataFrame([{
    "h2": h2,
    "h2_se": h2_se,
    "h2model": "LDSC_SIM_REFERENCE5000",
    "clumpprune": "no",
    "numberofvariants": variants,
    "numberofpca": "-"
}])

result.to_csv(
    result_dir + os.sep + "Results.csv",
    index=False
)

print("Saved:", result_dir + os.sep + "Results.csv")
exit(0)
