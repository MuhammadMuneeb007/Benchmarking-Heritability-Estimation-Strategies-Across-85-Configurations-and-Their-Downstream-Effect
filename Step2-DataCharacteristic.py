 

from pathlib import Path
import pandas as pd
import numpy as np
import json
import gzip
import re

OUTPUT_S8 = "Step2_Simulated_Data_Characteristics_Table_S8.csv"
OUTPUT_ALL = "Step2_Simulated_Data_Characteristics_All200.csv"

SIM_RE = re.compile(r"^Sim\d{3}$")


def flatten_dict(d, parent=""):
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        key = f"{parent}.{k}" if parent else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def get_any(flat, *names, default=np.nan):
    """Return first matching key by exact key or final path component."""
    for name in names:
        if name in flat and flat[name] is not None:
            return flat[name]
    for name in names:
        for k, v in flat.items():
            if k.split(".")[-1] == name and v is not None:
                return v
    return default


def count_nonempty_lines(path):
    if not path.exists():
        return np.nan
    try:
        with open(path, "rt") as fh:
            return sum(1 for line in fh if line.strip())
    except Exception:
        return np.nan


def count_gwas_variants(path):
    if not path.exists():
        return np.nan
    try:
        with gzip.open(path, "rt") as fh:
            n = sum(1 for line in fh if line.strip())
        return max(n - 1, 0)  # remove header
    except Exception:
        return np.nan


def inspect_gwas_n(path):
    """Read only the N column; return min, max and representative N."""
    if not path.exists():
        return np.nan, np.nan, np.nan
    try:
        x = pd.read_csv(
            path,
            compression="gzip",
            sep=r"\s+",
            usecols=["N"]
        )
        n = pd.to_numeric(x["N"], errors="coerce").dropna()
        if n.empty:
            return np.nan, np.nan, np.nan
        return int(n.min()), int(n.max()), int(round(n.median()))
    except Exception:
        return np.nan, np.nan, np.nan


def read_target_phenotype(path):
    if not path.exists():
        return None

    df = pd.read_csv(path, sep=r"\s+")
    if "Height" not in df.columns:
        raise ValueError(f"{path} does not contain a Height column")

    y = pd.to_numeric(df["Height"], errors="coerce").dropna()
    n = len(y)
    values = set(y.unique())

    out = {
        "Phenotype_N": n,
        "Cases": np.nan,
        "Controls": np.nan,
        "Sample_Prevalence_pct": np.nan,
        "Phenotype_Type_Detected": "Continuous",
    }

    if values.issubset({1, 2}) and len(values) <= 2:
        controls = int((y == 1).sum())
        cases = int((y == 2).sum())
        out.update({
            "Phenotype_Type_Detected": "Binary",
            "Cases": cases,
            "Controls": controls,
            "Sample_Prevalence_pct": 100.0 * cases / n if n else np.nan,
        })

    elif values.issubset({0, 1}) and len(values) <= 2:
        controls = int((y == 0).sum())
        cases = int((y == 1).sum())
        out.update({
            "Phenotype_Type_Detected": "Binary",
            "Cases": cases,
            "Controls": controls,
            "Sample_Prevalence_pct": 100.0 * cases / n if n else np.nan,
        })

    return out


def fmt_mean_range(series, decimals=2, integer=False):
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return "--"
    if integer:
        return f"{x.mean():.1f} [{int(x.min())}-{int(x.max())}]"
    return f"{x.mean():.{decimals}f} [{x.min():.{decimals}f}-{x.max():.{decimals}f}]"


def fmt_mean_sd(series, decimals=4):
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return "--"
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return f"{x.mean():.{decimals}f} +/- {sd:.{decimals}f}"


rows = []

for d in sorted(Path(".").iterdir()):
    if not d.is_dir() or not SIM_RE.match(d.name):
        continue

    sim = d.name
    truth_file = d / "truth.json"
    pheno_file = d / f"{sim}.height"
    gwas_file = d / f"{sim}.gz"
    bim_file = d / f"{sim}.bim"
    causal_file = d / "causal_snps.tsv"

    if not truth_file.exists() or not pheno_file.exists():
        print(f"WARNING: skipping {sim}: missing truth.json or phenotype file")
        continue

    try:
        with open(truth_file) as fh:
            truth = json.load(fh)
        flat = flatten_dict(truth)
        ph = read_target_phenotype(pheno_file)
    except Exception as e:
        print(f"ERROR: {sim}: {e}")
        continue

    phenotype_type = str(get_any(
        flat,
        "phenotype_type",
        "type",
        default=ph["Phenotype_Type_Detected"]
    )).capitalize()

    target_n = get_any(flat, "target_n", "target_sample_size", default=ph["Phenotype_N"])
    discovery_n_truth = get_any(flat, "discovery_n", "discovery_sample_size")
    master_n = get_any(flat, "master_n", "population_n")

    true_h2 = get_any(flat, "target_h2", "true_h2", "h2")
    causal_snps = get_any(flat, "causal_snps", "n_causal", "causal_snp_count")
    replicate = get_any(flat, "replicate", "rep")
    trait_id = get_any(flat, "trait_id", "underlying_trait_id", default="")

    if pd.isna(causal_snps) and causal_file.exists():
        try:
            cdf = pd.read_csv(causal_file, sep=r"\s+")
            causal_snps = len(cdf)
        except Exception:
            nlines = count_nonempty_lines(causal_file)
            causal_snps = max(nlines - 1, 0) if pd.notna(nlines) else np.nan

    population_prev = get_any(flat, "population_prevalence", "binary_prevalence")
    if pd.notna(population_prev):
        population_prev = float(population_prev)
        if population_prev <= 1:
            population_prev *= 100.0

    realized_master_h2 = get_any(
        flat,
        "realized_master_h2",
        "master_realized_h2",
        "realized_h2_master"
    )
    realized_discovery_h2 = get_any(
        flat,
        "realized_discovery_h2",
        "discovery_realized_h2",
        "realized_h2_discovery"
    )
    realized_target_h2 = get_any(
        flat,
        "realized_target_h2",
        "target_realized_h2",
        "realized_h2_target"
    )
    realized_target750_h2 = get_any(
        flat,
        "realized_target750_h2",
        "target750_realized_h2"
    )
    realized_target5000_h2 = get_any(
        flat,
        "realized_target5000_h2",
        "target5000_realized_h2"
    )

    # Fallback for nested dictionaries with keys such as master.realized_h2.
    if pd.isna(realized_master_h2):
        realized_master_h2 = flat.get("master.realized_h2", np.nan)
    if pd.isna(realized_discovery_h2):
        realized_discovery_h2 = flat.get("discovery.realized_h2", np.nan)
    if pd.isna(realized_target750_h2):
        realized_target750_h2 = flat.get("target750.realized_h2", np.nan)
    if pd.isna(realized_target5000_h2):
        realized_target5000_h2 = flat.get("target5000.realized_h2", np.nan)

    # If only the two target-specific realized values exist, choose the one
    # corresponding to this SimXXX target size.
    if pd.isna(realized_target_h2):
        try:
            tn = int(float(target_n))
            if tn == 750:
                realized_target_h2 = realized_target750_h2
            elif tn == 5000:
                realized_target_h2 = realized_target5000_h2
        except Exception:
            pass

    overlap_n = get_any(
        flat,
        "discovery_target_overlap_n",
        "overlap_n",
        default=0
    )

    gwas_n_min, gwas_n_max, gwas_n = inspect_gwas_n(gwas_file)
    if pd.isna(gwas_n) and pd.notna(discovery_n_truth):
        gwas_n = discovery_n_truth

    rows.append({
        "Simulation": sim,
        "Trait_ID": trait_id,
        "Phenotype_Type": phenotype_type,
        "Master_N": master_n,
        "Discovery_GWAS_N": gwas_n,
        "GWAS_N_Min": gwas_n_min,
        "GWAS_N_Max": gwas_n_max,
        "Pre_QC_Target_N": target_n,
        "Observed_Target_N": ph["Phenotype_N"],
        "Cases": ph["Cases"],
        "Controls": ph["Controls"],
        "Target_Sample_Prevalence_pct": ph["Sample_Prevalence_pct"],
        "Population_Prevalence_pct": population_prev,
        "Target_Genotype_SNPs": count_nonempty_lines(bim_file),
        "GWAS_Variants": count_gwas_variants(gwas_file),
        "True_h2": true_h2,
        "Causal_SNPs": causal_snps,
        "Replicate": replicate,
        "Master_Realized_h2": realized_master_h2,
        "Discovery_Realized_h2": realized_discovery_h2,
        "Target750_Realized_h2": realized_target750_h2,
        "Target5000_Realized_h2": realized_target5000_h2,
        "Target_Realized_h2": realized_target_h2,
        "Discovery_Target_Overlap_N": overlap_n,
    })


raw = pd.DataFrame(rows)

if raw.empty:
    raise SystemExit("ERROR: no Sim### datasets were found.")

raw = raw.sort_values("Simulation").reset_index(drop=True)
raw.to_csv(OUTPUT_ALL, index=False)

# ------------------------------------------------------------------
# Build the publication-facing 40-row Supplementary Table S8.
# ------------------------------------------------------------------

group_cols = [
    "Phenotype_Type",
    "Pre_QC_Target_N",
    "True_h2",
    "Causal_SNPs",
]

s8_rows = []

for keys, g in raw.groupby(group_cols, dropna=False, sort=True):
    phenotype_type, target_n, true_h2, causal_snps = keys

    master_vals = pd.to_numeric(g["Master_N"], errors="coerce").dropna()
    discovery_vals = pd.to_numeric(g["Discovery_GWAS_N"], errors="coerce").dropna()
    target_snp_vals = pd.to_numeric(g["Target_Genotype_SNPs"], errors="coerce").dropna()
    gwas_var_vals = pd.to_numeric(g["GWAS_Variants"], errors="coerce").dropna()
    overlap_vals = pd.to_numeric(g["Discovery_Target_Overlap_N"], errors="coerce").dropna()
    pop_prev_vals = pd.to_numeric(g["Population_Prevalence_pct"], errors="coerce").dropna()

    master_n = int(master_vals.iloc[0]) if not master_vals.empty else "--"
    discovery_n = int(round(discovery_vals.median())) if not discovery_vals.empty else "--"
    overlap_n = int(overlap_vals.max()) if not overlap_vals.empty else "--"

    if not target_snp_vals.empty and target_snp_vals.min() == target_snp_vals.max():
        target_snps = str(int(target_snp_vals.iloc[0]))
    else:
        target_snps = fmt_mean_range(target_snp_vals, decimals=0)

    if not gwas_var_vals.empty and gwas_var_vals.min() == gwas_var_vals.max():
        gwas_variants = str(int(gwas_var_vals.iloc[0]))
    else:
        gwas_variants = fmt_mean_range(gwas_var_vals, decimals=0)

    is_binary = str(phenotype_type).lower() == "binary"

    if is_binary:
        pop_prev = f"{pop_prev_vals.mean():.2f}" if not pop_prev_vals.empty else "--"
        target_cases = fmt_mean_range(g["Cases"], integer=True)
        target_controls = fmt_mean_range(g["Controls"], integer=True)
        target_prev = fmt_mean_range(g["Target_Sample_Prevalence_pct"], decimals=2)
        h2_scale = "Liability"
    else:
        pop_prev = "--"
        target_cases = "--"
        target_controls = "--"
        target_prev = "--"
        h2_scale = "Continuous"

    s8_rows.append({
        "Phenotype_Type": phenotype_type,
        "Pre_QC_Target_N": int(float(target_n)),
        "Discovery_GWAS_N": discovery_n,
        "Master_N": master_n,
        "True_h2": float(true_h2),
        "True_h2_Scale": h2_scale,
        "Causal_SNPs": int(float(causal_snps)),
        "Replicates": len(g),
        "Population_Prevalence_pct": pop_prev,
        "Target_Cases_Mean_Range": target_cases,
        "Target_Controls_Mean_Range": target_controls,
        "Target_Prevalence_pct_Mean_Range": target_prev,
        "Target_Genotype_SNPs": target_snps,
        "GWAS_Variants_Mean_Range": gwas_variants,
        "Master_Realized_h2_Mean_SD": fmt_mean_sd(g["Master_Realized_h2"]),
        "Discovery_Realized_h2_Mean_SD": fmt_mean_sd(g["Discovery_Realized_h2"]),
        "Target_Realized_h2_Mean_SD": fmt_mean_sd(g["Target_Realized_h2"]),
        "Discovery_Target_Overlap_N": overlap_n,
    })

s8 = pd.DataFrame(s8_rows)

type_order = {"Continuous": 0, "Binary": 1}
s8["_order"] = s8["Phenotype_Type"].map(type_order).fillna(99)
s8 = (
    s8.sort_values(
        ["_order", "Pre_QC_Target_N", "True_h2", "Causal_SNPs"]
    )
    .drop(columns="_order")
    .reset_index(drop=True)
)

s8.to_csv(OUTPUT_S8, index=False)

# ------------------------------------------------------------------
# Print the exact information needed for the supplementary text/table.
# ------------------------------------------------------------------

print("\n" + "=" * 180)
print("SUPPLEMENTARY TABLE S8: SIMULATED PHENOTYPE CHARACTERISTICS")
print("=" * 180)
print(s8.to_string(index=False))
print("=" * 180)

print("\n" + "=" * 100)
print("SUMMARY VALUES FOR SUPPLEMENTARY MATERIAL")
print("=" * 100)

print("Simulation datasets:", len(raw))
print("Table S8 design rows:", len(s8))
print("Unique trait IDs:", raw["Trait_ID"].nunique(dropna=True))

print(
    "Phenotype types:",
    sorted(raw["Phenotype_Type"].dropna().astype(str).unique().tolist())
)

print(
    "Pre-QC target N values:",
    sorted(
        pd.to_numeric(raw["Pre_QC_Target_N"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
)

print(
    "Discovery GWAS N values:",
    sorted(
        pd.to_numeric(raw["Discovery_GWAS_N"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
)

print(
    "True h2 values:",
    sorted(
        pd.to_numeric(raw["True_h2"], errors="coerce")
        .dropna()
        .unique()
        .tolist()
    )
)

print(
    "Causal SNP counts:",
    sorted(
        pd.to_numeric(raw["Causal_SNPs"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
)

binary = raw[raw["Phenotype_Type"].astype(str).str.lower() == "binary"].copy()

if not binary.empty:
    case_x = pd.to_numeric(binary["Cases"], errors="coerce").dropna()
    ctrl_x = pd.to_numeric(binary["Controls"], errors="coerce").dropna()
    prev_x = pd.to_numeric(binary["Target_Sample_Prevalence_pct"], errors="coerce").dropna()

    if not case_x.empty:
        print("Binary target cases range:", int(case_x.min()), "to", int(case_x.max()))
    if not ctrl_x.empty:
        print("Binary target controls range:", int(ctrl_x.min()), "to", int(ctrl_x.max()))
    if not prev_x.empty:
        print(
            "Binary target prevalence range (%):",
            round(prev_x.min(), 2),
            "to",
            round(prev_x.max(), 2)
        )

gwas_x = pd.to_numeric(raw["GWAS_Variants"], errors="coerce").dropna()
geno_x = pd.to_numeric(raw["Target_Genotype_SNPs"], errors="coerce").dropna()
overlap_x = pd.to_numeric(raw["Discovery_Target_Overlap_N"], errors="coerce").dropna()

if not gwas_x.empty:
    print("GWAS variant count range:", int(gwas_x.min()), "to", int(gwas_x.max()))
if not geno_x.empty:
    print("Target genotype SNP count range:", int(geno_x.min()), "to", int(geno_x.max()))
if not overlap_x.empty:
    print("Maximum discovery-target overlap:", int(overlap_x.max()))

print("=" * 100)

print("\nCreated:")
print(" ", OUTPUT_S8)
print(" ", OUTPUT_ALL)
 