import os
import sys
import pandas as pd
import numpy as np
import subprocess
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar

filedirec = sys.argv[1]
foldnumber = sys.argv[2]

GWAS = filedirec + os.sep + filedirec + ".gz"
df = pd.read_csv(GWAS, compression="gzip", sep=r"\s+")

if "BETA" in df.columns.to_list():
    df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'INFO', 'MAF']]
    df['Z'] = df['BETA'] / df['SE']
else:
    df["BETA"] = np.log(df["OR"])
    df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'INFO', 'MAF']]
    df['Z'] = df['BETA'] / df['SE']

df.to_csv(filedirec + os.sep + filedirec + ".txt", sep="\t", index=False)

df_transformed = pd.DataFrame({
    'Predictor': df['SNP'],
    'A1': df['A1'],
    'A2': df['A2'],
    'n': df['N'],
    'Z': df['BETA'] / df['SE'],
    'SNP': df['SNP']
})

columns_to_check = ['A1', 'A2']

def specified_single_characters(row, cols):
    return all(len(str(row[col])) == 1 for col in cols)

df_transformed = df_transformed[
    df_transformed.apply(lambda row: specified_single_characters(row, columns_to_check), axis=1)
]

# FIX 1: Drop duplicate Predictor (CHR:BP) entries — LDAK errors out if a
# predictor appears more than once in the summary-statistics file.
before = len(df_transformed)
df_transformed = df_transformed.drop_duplicates(subset='Predictor', keep='first')
after = len(df_transformed)
if before != after:
    print(f"Warning: dropped {before - after} duplicate CHR:BP predictor(s) from .ldak file.")

df_transformed.to_csv(filedirec + os.sep + filedirec + ".ldak", sep="\t", index=False)


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def readpve(direc):
    file_path = direc + os.sep + "snpher1.hers"
    with open(file_path, 'r') as file:
        for line in file:
            if "Her_All" in line:
                pve_estimate = line.split(" ")[1].strip()
                print("Heritability:", pve_estimate)
                return pve_estimate
    return np.nan


def readpve_se(direc):
    file_path = direc + os.sep + "snpher1.hers"
    with open(file_path, 'r') as file:
        for line in file:
            if "Her_All" in line:
                parts = line.split()
                if len(parts) >= 3:
                    pve_se = parts[2].strip()
                    print("Heritability SE:", pve_se)
                    return pve_se
    return np.nan


def readvariants(direc):
    file_path = direc + os.sep + "snpher1.overlap"
    with open(file_path, 'r') as file:
        for line in file:
            if "Summary_Statistic_Predictors" in line:
                pve_estimate = line.split(" ")[1].strip()
                print("Variants:", pve_estimate)
                return pve_estimate
    return np.nan


folddirec = filedirec + os.sep + "Fold_" + foldnumber
trainfilename = "train_data"
newtrainfilename = "train_data.QC"
testfilename = "test_data"
newtestfilename = "test_data.QC"

numberofpca = ["6"]
clump_p1 = [1]
clump_r2 = [0.1]
clump_kb = [200]
p_window_size = [200]
p_slide_size = [50]
p_LD_threshold = [0.25]

prs_result = pd.DataFrame(columns=[
    "clump_p1", "clump_r2", "clump_kb",
    "p_window_size", "p_slide_size", "p_LD_threshold",
    "numberofpca", "h2model", "h2", "h2_se",
    "numberofvariants", "clumpprune"
])


def transform_ldak_data(traindirec, newtrainfilename, p, precomputedfile,
                         p1_val, p2_val, p3_val, c1_val, c2_val, c3_val, Name):

    results_file = traindirec + os.sep + Name + os.sep + "Results.csv"
    if os.path.exists(results_file):
        os.remove(results_file)

    hers_file = traindirec + os.sep + "snpher1.hers"
    overlap_file = traindirec + os.sep + "snpher1.overlap"
    if os.path.exists(hers_file):
        os.remove(hers_file)
    if os.path.exists(overlap_file):
        os.remove(overlap_file)

    command = [
        "./ldak",
        "--sum-hers", traindirec + os.sep + "snpher1",
        "--summary", filedirec + os.sep + filedirec + ".ldak",
        "--tagfile", "LDAKFILES_SIM" + os.sep + "Reference5000" + os.sep + precomputedfile,
        "--check-sums", "NO"
    ]
    print(" ".join(command))
    result = subprocess.run(command)

    # FIX 2: Check that LDAK actually succeeded before reading its output.
    # If the command failed (non-zero exit code) or the output file is missing,
    # record NaN values and continue instead of crashing.
    if result.returncode != 0 or not os.path.exists(hers_file):
        print(f"Warning: LDAK failed or did not produce {hers_file} "
              f"(return code {result.returncode}). Skipping this configuration.")
        pve_estimate = np.nan
        pve_se = np.nan
        variants = np.nan
    else:
        pve_estimate = readpve(traindirec)
        pve_se = readpve_se(traindirec)
        variants = readvariants(traindirec)

    print("Heritability:", pve_estimate, "SE:", pve_se, "Variants:", variants)

    global prs_result
    new_row = pd.DataFrame([{
        "clump_p1":       c1_val,
        "clump_r2":       c2_val,
        "clump_kb":       c3_val,
        "p_window_size":  p1_val,
        "p_slide_size":   p2_val,
        "p_LD_threshold": p3_val,
        "numberofpca":    "-",
        "h2":             pve_estimate,
        "h2_se":          pve_se,
        "h2model":        "LDAK_SIM_PRECOMPUTED_" + precomputedfile,
        "clumpprune":     "no",
        "numberofvariants": variants
    }])
    prs_result = pd.concat([prs_result, new_row], ignore_index=True)

    outdir = traindirec + os.sep + Name
    create_directory(outdir)
    prs_result.to_csv(outdir + os.sep + "Results.csv", index=False)
    return


precomputedfiles = [
    "sim_human.tagging",
    "sim_gcta.tagging",
    "sim_alpha_1.0.tagging",
    "sim_alpha_0.5.tagging",
    "sim_alpha_0.0.tagging",
    "sim_alpha_-0.5.tagging"
]

create_directory(folddirec + os.sep + "LDAK-Precomputed-Heritability")

for p1_val in p_window_size:
    for p2_val in p_slide_size:
        for p3_val in p_LD_threshold:
            for c1_val in clump_p1:
                for c2_val in clump_r2:
                    for c3_val in clump_kb:
                        for p in numberofpca:
                            for precomputedfile in precomputedfiles:
                                transform_ldak_data(
                                    folddirec, newtrainfilename, p, precomputedfile,
                                    str(p1_val), str(p2_val), str(p3_val),
                                    str(c1_val), str(c2_val), str(c3_val),
                                    "LDAK-Precomputed-Heritability"
                                )

exit(0)