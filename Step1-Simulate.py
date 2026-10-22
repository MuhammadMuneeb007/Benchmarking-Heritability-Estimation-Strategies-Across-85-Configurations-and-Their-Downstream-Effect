#!/usr/bin/env python3
"""
Step1-Simulate.py

DATA-GENERATION ONLY for the ground-truth heritability simulation benchmark.
This step creates the simulated genotypes, phenotypes, discovery GWAS summary
statistics, known generating parameters, realized finite-sample heritability,
and five-fold target partitions required by the existing heritability/PRS
repository. It deliberately DOES NOT run LDSC, GCTA, GEMMA, LDAK, DPR, or any
other heritability estimator. Those methods belong to the downstream benchmark.

DEFAULT DESIGN
--------------
Genotypes
  Species                 Homo sapiens (HomSap)
  Demographic model       OutOfAfrica_3G09
  Population              CEU
  Genetic map             HapMapII_GRCh38
  Autosomes               1-22
  Simulated region        central 10 Mb per chromosome
  Mutation rate           demographic-model mutation rate
  Common-variant filter   MAF >= 0.05
  Final SNP panel         100,000 common SNPs, allocated proportionally to the
                          eligible common-SNP count across chromosomes

Cohorts
  Master population       N = 15,000
  Discovery GWAS cohort   N = 10,000
  Target cohort           N = 5,000, non-overlapping with discovery
  Small target cohort     N = 750, nested within the N=5,000 target cohort

Traits
  Phenotype type          continuous / binary
  Target h2               0.05, 0.10, 0.30, 0.50, 0.70
  Causal SNPs             100 / 5,000
  Replicates              5
  Unique traits           2 * 5 * 2 * 5 = 100
  Final target datasets   100 traits * 2 target sample sizes = 200 SimXXX folders

For each unique trait, the SAME causal variants and effect sizes are used across
the complete master population. Individuals are then partitioned into a
10,000-person discovery cohort and a non-overlapping 5,000-person target cohort.
The N=750 target set is a fixed nested subset of the N=5,000 target set.

The discovery cohort is used ONLY to create the GWAS summary-statistics file
required by downstream summary-statistic methods. The target cohorts are used
for the individual-level heritability/PRS benchmark. There is zero discovery-
target sample overlap.

Binary traits are generated under a liability-threshold model with population
prevalence K=0.10 and a fixed threshold Phi^{-1}(1-K). The observed prevalence
is therefore allowed to fluctuate naturally in each finite sample.

KNOWN HERITABILITY
------------------
This script records both:
  1. target_h2: the generating population parameter; and
  2. realized_h2: the finite-sample variance ratio computed directly from the
     KNOWN simulated genetic and environmental components.

No statistical estimator is used to obtain these truth values.

OUTPUT PER SimXXX
-----------------
  SimXXX.bed/.bim/.fam        target genotype + phenotype
  SimXXX.height               repository-compatible phenotype file
  SimXXX.cov                  covariate file
  SimXXX.gz                   independent discovery-cohort GWAS summary stats
  simulated_phenotype.tsv     target phenotype + known G/E/liability components
  discovery_phenotype.tsv     discovery phenotype used for the GWAS
  causal_snps.tsv             known causal variants and simulated effects
  truth.json                  full generating parameters and realized truth
  fold_assignments.csv        fold assignment for every target individual
  Fold_0 ... Fold_4           train/test IDs, FAM and covariate files

ROOT-LEVEL OUTPUT
-----------------
  simulation_manifest.csv
  simulation_truth_summary.csv
  All_Phenotypes.txt
  PhenotypeParameters.txt
  FoldParameters.txt
  PhenotypeParametersBinary.txt / FoldParametersBinary.txt
  PhenotypeParametersCont.txt   / FoldParametersCont.txt

Large shared genotype panels are stored under _simulation_shared/. Each SimXXX
folder still receives its own BED/BIM filesystem entries. By default they are
hard links to the relevant target panel, so the simulations remain separately
addressable without multiplying disk use by 200. Use --copy-genotypes only if
physical duplication is specifically required.

Typical execution from the repository root:
    python Step1-Simulate.py

After completion, validate one simulation before launching the downstream
benchmark:
    zcat Sim001/Sim001.gz | head
    cat Sim001/truth.json
    python Step2-PerformQualityControls.py Sim001
    python Method1.py Sim001 0
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import KFold, StratifiedKFold

try:
    import stdpopsim
except ImportError:
    stdpopsim = None


# =============================================================================
# FIXED EXPERIMENT SETTINGS
# =============================================================================

DESIGN_VERSION = "independent_discovery_data_only_v4_20260911"

SPECIES_ID = "HomSap"
DEMOGRAPHIC_MODEL = "OutOfAfrica_3G09"
POPULATION = "CEU"
GENETIC_MAP = "HapMapII_GRCh38"
CHROMOSOMES = list(range(1, 23))
SEGMENT_BP = 10_000_000
MAF_MIN = 0.05
TARGET_COMMON_SNPS = 100_000

MASTER_N = 15_000
DISCOVERY_N = 10_000
TARGET_LARGE_N = 5_000
TARGET_SMALL_N = 750
TARGET_SAMPLE_SIZES = [TARGET_SMALL_N, TARGET_LARGE_N]

PHENOTYPE_TYPES = ["continuous", "binary"]
H2_VALUES = [0.05, 0.10, 0.30, 0.50, 0.70]
CAUSAL_SNP_COUNTS = [100, 5000]
REPLICATES = 5
BINARY_POPULATION_PREVALENCE = 0.10
N_FOLDS = 5

# Deterministic seeds. Do not change these after a production run begins.
GENOTYPE_SEED = 20260911
COHORT_SPLIT_SEED = 20260912
CAUSAL_SEED_BASE = 310000
EFFECT_SEED_BASE = 410000
PHENOTYPE_SEED_BASE = 510000
FOLD_SEED = 42


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class TraitConfig:
    trait_id: str
    phenotype_type: str
    true_h2: float
    causal_snps: int
    replicate: int
    population_prevalence: Optional[float]
    causal_seed: int
    effect_seed: int
    phenotype_seed: int


@dataclass(frozen=True)
class SimulationConfig:
    simulation_id: str
    trait_id: str
    phenotype_type: str
    target_n: int
    true_h2: float
    causal_snps: int
    replicate: int
    population_prevalence: Optional[float]


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def sh(cmd: Sequence[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [str(x) for x in cmd]
    log("+ " + " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
    )


def find_executable(explicit: Optional[str], candidates: Sequence[str]) -> Optional[str]:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return str(p.resolve())
        found = shutil.which(explicit)
        if found:
            return found
        return None

    for candidate in candidates:
        p = Path(candidate)
        if p.exists() and p.is_file():
            return str(p.resolve())
        found = shutil.which(candidate)
        if found:
            return found
    return None


def ensure_python_dependencies() -> None:
    if stdpopsim is None:
        raise ImportError(
            "stdpopsim is not installed. Install stdpopsim, msprime, tskit, "
            "numpy, pandas, scipy and scikit-learn before running."
        )



def hardlink_or_copy(src: Path, dst: Path, force_copy: bool = False) -> None:
    """Create a per-simulation filesystem entry without multiplying storage."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if force_copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def read_table_ws(path: Path, header=None) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=header)


def verify_clean_or_compatible_root(root: Path, shared: Path) -> None:
    """
    Refuse to silently mix the old N=5000-only design with this redesigned run.
    """
    design_file = shared / "simulation_design.json"
    existing_sims = sorted(root.glob("Sim[0-9][0-9][0-9]"))

    if design_file.exists():
        try:
            previous = json.loads(design_file.read_text())
            if previous.get("design_version") != DESIGN_VERSION:
                raise RuntimeError(
                    f"Existing simulation design is {previous.get('design_version')!r}, "
                    f"but this script expects {DESIGN_VERSION!r}. Remove/rename old "
                    "SimXXX directories and _simulation_shared before continuing."
                )
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Cannot parse {design_file}: {exc}")
    elif existing_sims:
        raise RuntimeError(
            "Existing SimXXX directories were found but no matching redesigned "
            "simulation_design.json exists. To avoid mixing old and new simulations, "
            "remove/rename Sim001-Sim200 and _simulation_shared, then rerun."
        )


# =============================================================================
# MANIFESTS
# =============================================================================

def build_trait_manifest(shared: Path) -> pd.DataFrame:
    rows = []
    counter = 1
    for phenotype_type in PHENOTYPE_TYPES:
        for h2 in H2_VALUES:
            for causal_n in CAUSAL_SNP_COUNTS:
                for rep in range(1, REPLICATES + 1):
                    type_offset = 0 if phenotype_type == "continuous" else 10000
                    trait = TraitConfig(
                        trait_id=f"Trait{counter:03d}",
                        phenotype_type=phenotype_type,
                        true_h2=h2,
                        causal_snps=causal_n,
                        replicate=rep,
                        population_prevalence=(
                            BINARY_POPULATION_PREVALENCE
                            if phenotype_type == "binary"
                            else None
                        ),
                        causal_seed=CAUSAL_SEED_BASE + rep,
                        effect_seed=EFFECT_SEED_BASE + rep,
                        phenotype_seed=(
                            PHENOTYPE_SEED_BASE
                            + type_offset
                            + causal_n
                            + rep
                        ),
                    )
                    rows.append(asdict(trait))
                    counter += 1
    df = pd.DataFrame(rows)
    assert len(df) == 100
    df.to_csv(shared / "trait_manifest.csv", index=False)
    return df


def build_simulation_manifest(root: Path, shared: Path, trait_df: pd.DataFrame) -> pd.DataFrame:
    trait_lookup = {
        (
            r.phenotype_type,
            float(r.true_h2),
            int(r.causal_snps),
            int(r.replicate),
        ): r.trait_id
        for r in trait_df.itertuples(index=False)
    }

    rows = []
    counter = 1
    # Preserve the previous SimXXX ordering:
    # continuous N750 1-50, continuous N5000 51-100,
    # binary N750 101-150, binary N5000 151-200.
    for phenotype_type in PHENOTYPE_TYPES:
        for n in TARGET_SAMPLE_SIZES:
            for h2 in H2_VALUES:
                for causal_n in CAUSAL_SNP_COUNTS:
                    for rep in range(1, REPLICATES + 1):
                        key = (phenotype_type, float(h2), int(causal_n), int(rep))
                        cfg = SimulationConfig(
                            simulation_id=f"Sim{counter:03d}",
                            trait_id=trait_lookup[key],
                            phenotype_type=phenotype_type,
                            target_n=n,
                            true_h2=h2,
                            causal_snps=causal_n,
                            replicate=rep,
                            population_prevalence=(
                                BINARY_POPULATION_PREVALENCE
                                if phenotype_type == "binary"
                                else None
                            ),
                        )
                        rows.append(asdict(cfg))
                        counter += 1

    df = pd.DataFrame(rows)
    assert len(df) == 200
    df.to_csv(shared / "simulation_manifest.csv", index=False)
    df.to_csv(root / "simulation_manifest.csv", index=False)
    return df


def write_parameter_files(root: Path, sim_df: pd.DataFrame) -> None:
    names = sim_df["simulation_id"].tolist()
    binary = sim_df.loc[sim_df["phenotype_type"] == "binary", "simulation_id"].tolist()
    continuous = sim_df.loc[sim_df["phenotype_type"] == "continuous", "simulation_id"].tolist()

    with open(root / "All_Phenotypes.txt", "w") as f:
        for name in names:
            f.write(name + "\n")

    def write_fold_params(name_list: Iterable[str], name_path: Path, fold_path: Path) -> None:
        with open(name_path, "w") as fn, open(fold_path, "w") as ff:
            for name in name_list:
                for fold in range(N_FOLDS):
                    fn.write(name + "\n")
                    ff.write(str(fold) + "\n")

    write_fold_params(names, root / "PhenotypeParameters.txt", root / "FoldParameters.txt")
    write_fold_params(binary, root / "PhenotypeParametersBinary.txt", root / "FoldParametersBinary.txt")
    write_fold_params(continuous, root / "PhenotypeParametersCont.txt", root / "FoldParametersCont.txt")


# =============================================================================
# GENOTYPE SIMULATION
# =============================================================================

def filter_common_biallelic_site_ids(ts, maf_min: float) -> np.ndarray:
    keep: List[int] = []
    for var in ts.variants():
        alleles = [a for a in var.alleles if a is not None]
        if len(alleles) != 2:
            continue
        g = var.genotypes
        valid = g[g >= 0]
        if valid.size == 0:
            continue
        counts = np.bincount(valid, minlength=2)
        if counts.sum() == 0:
            continue
        maf = counts.min() / counts.sum()
        if maf >= maf_min:
            keep.append(var.site.id)
    return np.asarray(keep, dtype=np.int64)


def proportional_integer_allocation(total: int, capacities: Sequence[int]) -> List[int]:
    capacities = np.asarray(capacities, dtype=np.int64)
    if total > int(capacities.sum()):
        raise ValueError(
            f"Requested {total:,} SNPs but only {capacities.sum():,} eligible sites exist."
        )
    weights = capacities.astype(float)
    raw = total * weights / weights.sum()
    allocation = np.floor(raw).astype(int)
    remainder = total - int(allocation.sum())
    order = np.argsort(-(raw - allocation))
    for idx in order:
        if remainder <= 0:
            break
        if allocation[idx] < capacities[idx]:
            allocation[idx] += 1
            remainder -= 1
    if remainder != 0:
        # Extremely defensive capacity-filling fallback.
        for idx in np.argsort(-capacities):
            while remainder > 0 and allocation[idx] < capacities[idx]:
                allocation[idx] += 1
                remainder -= 1
    if int(allocation.sum()) != total:
        raise RuntimeError("Could not allocate requested common SNP count across chromosomes.")
    return allocation.tolist()


def simulate_master_genotypes(
    shared: Path,
    plink: str,
    keep_intermediates: bool,
) -> Path:
    ensure_python_dependencies()

    master_dir = shared / "genotypes" / f"N{MASTER_N}_MASTER"
    master_dir.mkdir(parents=True, exist_ok=True)
    master_prefix = master_dir / "genotypes"

    if all(master_prefix.with_suffix(ext).exists() for ext in [".bed", ".bim", ".fam"]):
        log(f"[skip] Master genotype panel already exists: {master_prefix}")
        return master_prefix

    species = stdpopsim.get_species(SPECIES_ID)
    model = species.get_demographic_model(DEMOGRAPHIC_MODEL)
    engine = stdpopsim.get_engine("msprime")

    temp_dir = master_dir / "_chromosomes"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # First pass: simulate and count eligible common variants per chromosome.
    chrom_records = []
    common_counts = []

    for chrom in CHROMOSOMES:
        trees_path = temp_dir / f"chr{chrom}.all.trees"
        common_path = temp_dir / f"chr{chrom}.common.npy"
        meta_path = temp_dir / f"chr{chrom}.meta.json"

        if trees_path.exists() and common_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            common_ids = np.load(common_path)
            common_counts.append(len(common_ids))
            chrom_records.append(meta)
            log(f"[resume] chr{chrom}: eligible common SNPs={len(common_ids):,}")
            continue

        chrom_obj = species.genome.get_chromosome(str(chrom))
        chrom_len = int(chrom_obj.length)
        seg = min(SEGMENT_BP, chrom_len)
        left = max(0, (chrom_len - seg) // 2)
        right = left + seg
        chr_seed = GENOTYPE_SEED + chrom

        log(
            f"\n[simulate] chr{chrom}: {left:,}-{right:,}; "
            f"master N={MASTER_N:,}; seed={chr_seed}"
        )

        contig = species.get_contig(
            chromosome=str(chrom),
            genetic_map=GENETIC_MAP,
            mutation_rate=model.mutation_rate,
            left=left,
            right=right,
        )

        ts = engine.simulate(
            model,
            contig,
            samples={POPULATION: MASTER_N},
            seed=chr_seed,
            msprime_model="dtwf",
            msprime_change_model=[(20, "hudson")],
        )

        common_ids = filter_common_biallelic_site_ids(ts, MAF_MIN)
        log(
            f"[chr{chrom}] total sites={ts.num_sites:,}; "
            f"common biallelic MAF>={MAF_MIN}={len(common_ids):,}"
        )

        ts.dump(str(trees_path))
        np.save(common_path, common_ids)
        meta = {
            "chromosome": chrom,
            "chromosome_length": chrom_len,
            "segment_left": left,
            "segment_right": right,
            "segment_bp": seg,
            "seed": chr_seed,
            "all_sites": int(ts.num_sites),
            "common_biallelic_sites": int(len(common_ids)),
        }
        write_json(meta, meta_path)
        common_counts.append(len(common_ids))
        chrom_records.append(meta)

    quotas = proportional_integer_allocation(TARGET_COMMON_SNPS, common_counts)
    log("\n[allocation] Retained SNPs are proportional to eligible common-SNP density:")
    for chrom, count, quota in zip(CHROMOSOMES, common_counts, quotas):
        log(f"  chr{chrom:>2}: eligible={count:>7,}  retained={quota:>5,}")

    # Second pass: select variants, write VCF, convert each chromosome to PLINK.
    chr_prefixes: List[Path] = []
    final_meta = []

    for chrom, quota, base_meta in zip(CHROMOSOMES, quotas, chrom_records):
        trees_path = temp_dir / f"chr{chrom}.all.trees"
        common_path = temp_dir / f"chr{chrom}.common.npy"
        chr_prefix = temp_dir / f"chr{chrom}"

        if all(Path(str(chr_prefix) + ext).exists() for ext in [".bed", ".bim", ".fam"]):
            chr_prefixes.append(chr_prefix)
            final_meta.append({**base_meta, "retained_sites": int(quota)})
            log(f"[resume] PLINK chr{chrom} already exists")
            continue

        import tskit

        ts = tskit.load(str(trees_path))
        common_ids = np.load(common_path)
        rng = np.random.default_rng(GENOTYPE_SEED * 100 + chrom)
        selected = np.sort(rng.choice(common_ids, size=quota, replace=False))
        selected_set = set(selected.tolist())
        delete_ids = np.fromiter(
            (sid for sid in range(ts.num_sites) if sid not in selected_set),
            dtype=np.int64,
        )
        ts_keep = ts.delete_sites(delete_ids)

        selected_trees = temp_dir / f"chr{chrom}.selected.trees"
        vcf_path = temp_dir / f"chr{chrom}.vcf"
        if keep_intermediates:
            ts_keep.dump(str(selected_trees))

        with open(vcf_path, "w") as out_vcf:
            ts_keep.write_vcf(out_vcf, contig_id=str(chrom))

        sh([
            plink,
            "--vcf", str(vcf_path),
            "--double-id",
            "--set-missing-var-ids", "@:#:$1:$2",
            "--make-bed",
            "--out", str(chr_prefix),
        ])

        # Force globally unique stable variant IDs.
        bim_path = chr_prefix.with_suffix(".bim")
        bim = read_table_ws(bim_path, header=None)
        if bim.shape[1] < 6:
            raise RuntimeError(f"Malformed BIM {bim_path}: {bim.shape[1]} columns")
        bim[1] = (
            bim[0].astype(str)
            + ":" + bim[3].astype(str)
            + ":" + bim[4].astype(str)
            + ":" + bim[5].astype(str)
        )
        if bim[1].duplicated().any():
            raise RuntimeError(f"Duplicate SNP IDs remain on chr{chrom}")
        bim.to_csv(bim_path, sep="\t", header=False, index=False)

        chr_prefixes.append(chr_prefix)
        final_meta.append({**base_meta, "retained_sites": int(quota)})
        vcf_path.unlink(missing_ok=True)

    merge_list = temp_dir / "merge_list.txt"
    with open(merge_list, "w") as f:
        for prefix in chr_prefixes[1:]:
            f.write(str(prefix) + "\n")

    sh([
        plink,
        "--bfile", str(chr_prefixes[0]),
        "--merge-list", str(merge_list),
        "--make-bed",
        "--out", str(master_prefix),
    ])

    # Clean, reproducible sample IDs and random sex assignment.
    fam = read_table_ws(master_prefix.with_suffix(".fam"), header=None)
    if len(fam) != MASTER_N:
        raise RuntimeError(f"Expected {MASTER_N} individuals, found {len(fam)}")
    ids = [f"SIMIND{i+1:06d}" for i in range(MASTER_N)]
    sex_rng = np.random.default_rng(GENOTYPE_SEED + 777)
    sex = sex_rng.integers(1, 3, size=MASTER_N)
    fam[0] = ids
    fam[1] = ids
    fam[2] = 0
    fam[3] = 0
    fam[4] = sex
    fam[5] = -9
    fam.to_csv(master_prefix.with_suffix(".fam"), sep="\t", header=False, index=False)

    pd.DataFrame({"FID": ids, "IID": ids, "Sex": sex}).to_csv(
        master_dir / "sample_metadata.tsv", sep="\t", index=False
    )

    write_json(
        {
            "design_version": DESIGN_VERSION,
            "species": SPECIES_ID,
            "demographic_model": DEMOGRAPHIC_MODEL,
            "population": POPULATION,
            "genetic_map": GENETIC_MAP,
            "mutation_rate": model.mutation_rate,
            "chromosomes": CHROMOSOMES,
            "segment_bp": SEGMENT_BP,
            "maf_min": MAF_MIN,
            "target_common_snps": TARGET_COMMON_SNPS,
            "master_n": MASTER_N,
            "genotype_seed": GENOTYPE_SEED,
            "chromosome_details": final_meta,
        },
        master_dir / "genotype_simulation.json",
    )

    if not keep_intermediates:
        for chrom in CHROMOSOMES:
            for suffix in [
                ".all.trees", ".common.npy", ".meta.json", ".selected.trees",
                ".bed", ".bim", ".fam", ".log", ".nosex",
            ]:
                (temp_dir / f"chr{chrom}{suffix}").unlink(missing_ok=True)
        merge_list.unlink(missing_ok=True)
        try:
            temp_dir.rmdir()
        except OSError:
            pass

    log(f"[done] Master genotype panel: {master_prefix}")
    return master_prefix


# =============================================================================
# COHORT CREATION
# =============================================================================

def make_subset_panel(plink: str, master_prefix: Path, ids_df: pd.DataFrame, out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    if all(out_prefix.with_suffix(ext).exists() for ext in [".bed", ".bim", ".fam"]):
        return
    keep = out_prefix.parent / (out_prefix.name + ".keep")
    ids_df[["FID", "IID"]].to_csv(keep, sep="\t", header=False, index=False)
    sh([
        plink,
        "--bfile", str(master_prefix),
        "--keep", str(keep),
        "--make-bed",
        "--out", str(out_prefix),
    ])


def create_cohorts(shared: Path, plink: str, master_prefix: Path) -> Dict[str, Path]:
    cohort_root = shared / "cohorts"
    cohort_root.mkdir(parents=True, exist_ok=True)

    master_fam = read_table_ws(master_prefix.with_suffix(".fam"), header=None)
    ids = pd.DataFrame({"FID": master_fam[0].astype(str), "IID": master_fam[1].astype(str)})
    if len(ids) != MASTER_N:
        raise RuntimeError("Master FAM size does not match MASTER_N")

    rng = np.random.default_rng(COHORT_SPLIT_SEED)
    perm = rng.permutation(MASTER_N)
    discovery_idx = perm[:DISCOVERY_N]
    target_idx = perm[DISCOVERY_N: DISCOVERY_N + TARGET_LARGE_N]
    if len(set(discovery_idx).intersection(set(target_idx))) != 0:
        raise RuntimeError("Discovery/target sample overlap detected")

    discovery_ids = ids.iloc[discovery_idx].copy()
    target5000_ids = ids.iloc[target_idx].copy()

    small_rng = np.random.default_rng(COHORT_SPLIT_SEED + 1)
    target_small_order = small_rng.permutation(len(target5000_ids))[:TARGET_SMALL_N]
    target750_ids = target5000_ids.iloc[target_small_order].copy()

    discovery_prefix = cohort_root / "discovery_N10000" / "genotypes"
    target5000_prefix = cohort_root / "target_N5000" / "genotypes"
    target750_prefix = cohort_root / "target_N750" / "genotypes"

    make_subset_panel(plink, master_prefix, discovery_ids, discovery_prefix)
    make_subset_panel(plink, master_prefix, target5000_ids, target5000_prefix)
    make_subset_panel(plink, master_prefix, target750_ids, target750_prefix)

    assignments = pd.DataFrame({
        "FID": ids["FID"],
        "IID": ids["IID"],
        "cohort": "unused",
    })
    assignments.loc[discovery_idx, "cohort"] = "discovery"
    assignments.loc[target_idx, "cohort"] = "target5000"
    small_iids = set(target750_ids["IID"].astype(str))
    assignments["in_target750"] = assignments["IID"].astype(str).isin(small_iids).astype(int)
    assignments.to_csv(cohort_root / "cohort_assignments.tsv", sep="\t", index=False)

    if len(set(discovery_ids.IID).intersection(set(target5000_ids.IID))) != 0:
        raise RuntimeError("Discovery/target IDs overlap after panel creation")

    write_json(
        {
            "master_n": MASTER_N,
            "discovery_n": DISCOVERY_N,
            "target_large_n": TARGET_LARGE_N,
            "target_small_n": TARGET_SMALL_N,
            "discovery_target_overlap_n": 0,
            "target_small_nested_within_target_large": True,
            "cohort_split_seed": COHORT_SPLIT_SEED,
        },
        cohort_root / "cohort_design.json",
    )

    return {
        "master": master_prefix,
        "discovery": discovery_prefix,
        "target5000": target5000_prefix,
        "target750": target750_prefix,
    }


# =============================================================================
# PLINK BED READER FOR SELECTED CAUSAL SNPS
# =============================================================================

def read_selected_bed(bed_path: Path, n_samples: int, variant_indices: np.ndarray) -> np.ndarray:
    """Read selected variants from a SNP-major PLINK BED file as A2 dosage."""
    bytes_per_variant = (n_samples + 3) // 4
    out = np.empty((n_samples, len(variant_indices)), dtype=np.float32)
    mapping = np.array([0.0, np.nan, 1.0, 2.0], dtype=np.float32)

    with open(bed_path, "rb") as f:
        header = f.read(3)
        if header != bytes([0x6C, 0x1B, 0x01]):
            raise ValueError(f"{bed_path} is not SNP-major PLINK BED")

        for j, vidx in enumerate(variant_indices):
            f.seek(3 + int(vidx) * bytes_per_variant)
            raw = f.read(bytes_per_variant)
            if len(raw) != bytes_per_variant:
                raise EOFError(f"Unexpected end of BED at variant {vidx}")
            arr = np.frombuffer(raw, dtype=np.uint8)
            codes = np.empty(bytes_per_variant * 4, dtype=np.uint8)
            codes[0::4] = arr & 0b11
            codes[1::4] = (arr >> 2) & 0b11
            codes[2::4] = (arr >> 4) & 0b11
            codes[3::4] = (arr >> 6) & 0b11
            out[:, j] = mapping[codes[:n_samples]]
    return out


def standardize_columns(G: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = np.nanmean(G, axis=0)
    G = np.where(np.isnan(G), means[None, :], G)
    sds = G.std(axis=0, ddof=0)
    polymorphic = sds > 0
    Z = np.zeros_like(G, dtype=np.float32)
    Z[:, polymorphic] = (
        (G[:, polymorphic] - means[polymorphic]) / sds[polymorphic]
    ).astype(np.float32)
    return Z, polymorphic, means, sds


def build_causal_plan(master_prefix: Path, replicate: int) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    bim = read_table_ws(master_prefix.with_suffix(".bim"), header=None)
    max_causal = max(CAUSAL_SNP_COUNTS)
    if len(bim) < max_causal:
        raise RuntimeError(f"Need {max_causal} causal SNPs but only {len(bim)} variants exist")

    rng = np.random.default_rng(CAUSAL_SEED_BASE + replicate)
    indices = rng.choice(len(bim), size=max_causal, replace=False)

    effect_rng = np.random.default_rng(EFFECT_SEED_BASE + replicate)
    beta = effect_rng.normal(0.0, 1.0, size=max_causal)

    causal = bim.iloc[indices].copy()
    causal.columns = ["CHR", "SNP", "CM", "BP", "A1", "A2"]
    causal["variant_index"] = indices
    causal["beta_raw"] = beta
    return indices.astype(np.int64), causal, beta


# =============================================================================
# PHENOTYPE / TRUTH HELPERS
# =============================================================================

def make_standard_normal(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n).astype(np.float64)
    x -= x.mean()
    sd = x.std(ddof=0)
    if sd == 0:
        raise RuntimeError("Environmental noise has zero variance")
    return x / sd


def realized_h2_from_components(g_component: np.ndarray, e_component: np.ndarray) -> float:
    y = g_component + e_component
    vy = float(np.var(y, ddof=0))
    if vy <= 0:
        return float("nan")
    return float(np.var(g_component, ddof=0) / vy)


def component_metrics(
    g_std: np.ndarray,
    e_std: np.ndarray,
    h2: float,
    indices: np.ndarray,
    y_binary: Optional[np.ndarray] = None,
) -> dict:
    gcomp = math.sqrt(h2) * g_std[indices]
    ecomp = math.sqrt(1.0 - h2) * e_std[indices]
    out = {
        "n": int(len(indices)),
        "realized_h2": realized_h2_from_components(gcomp, ecomp),
        "genetic_variance": float(np.var(gcomp, ddof=0)),
        "environmental_variance": float(np.var(ecomp, ddof=0)),
        "genetic_environment_covariance": float(np.cov(gcomp, ecomp, ddof=0)[0, 1]),
        "phenotype_variance": float(np.var(gcomp + ecomp, ddof=0)),
    }
    if y_binary is not None:
        out["sample_prevalence"] = float(y_binary[indices].mean())
        out["cases"] = int(y_binary[indices].sum())
        out["controls"] = int(len(indices) - y_binary[indices].sum())
    return out


def panel_master_indices(master_fam: pd.DataFrame, panel_prefix: Path) -> np.ndarray:
    panel_fam = read_table_ws(panel_prefix.with_suffix(".fam"), header=None)
    lookup = {str(iid): idx for idx, iid in enumerate(master_fam[1].astype(str))}
    try:
        return np.asarray([lookup[str(iid)] for iid in panel_fam[1].astype(str)], dtype=int)
    except KeyError as exc:
        raise RuntimeError(f"Panel IID not found in master FAM: {exc}")


def phenotype_to_pipeline(y_internal: np.ndarray, phenotype_type: str) -> np.ndarray:
    if phenotype_type == "binary":
        return y_internal.astype(int) + 1  # 1=control, 2=case
    return y_internal.astype(float)


def create_target_files(
    sim_dir: Path,
    sim_id: str,
    target_prefix: Path,
    target_indices_master: np.ndarray,
    g_std: np.ndarray,
    e_std: np.ndarray,
    liability: np.ndarray,
    y_internal_master: np.ndarray,
    phenotype_type: str,
    copy_genotypes: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    sim_dir.mkdir(parents=True, exist_ok=True)

    # Separate entries in every SimXXX folder. BED/BIM use hard links by default.
    hardlink_or_copy(target_prefix.with_suffix(".bed"), sim_dir / f"{sim_id}.bed", copy_genotypes)
    hardlink_or_copy(target_prefix.with_suffix(".bim"), sim_dir / f"{sim_id}.bim", copy_genotypes)

    base_fam = read_table_ws(target_prefix.with_suffix(".fam"), header=None)
    y_internal = y_internal_master[target_indices_master]
    y_pipeline = phenotype_to_pipeline(y_internal, phenotype_type)
    if len(base_fam) != len(y_pipeline):
        raise RuntimeError(f"{sim_id}: target FAM/phenotype length mismatch")

    fam = base_fam.copy()
    fam[5] = y_pipeline
    fam.to_csv(sim_dir / f"{sim_id}.fam", sep="\t", header=False, index=False)

    # Legacy repository convention.
    height = pd.DataFrame({
        "IID": fam[1].astype(str),
        "FID": fam[0].astype(str),
        "Height": y_pipeline,
    })
    # Reorder to FID/IID/Height for safer downstream merges while retaining names.
    height = height[["FID", "IID", "Height"]]
    height.to_csv(sim_dir / f"{sim_id}.height", sep="\t", index=False)

    cov = pd.DataFrame({
        "FID": fam[0].astype(str),
        "IID": fam[1].astype(str),
        "Sex": fam[4].astype(int),
    })
    cov.to_csv(sim_dir / f"{sim_id}.cov", sep="\t", index=False)


    phenotype_table = pd.DataFrame({
        "FID": fam[0].astype(str),
        "IID": fam[1].astype(str),
        "phenotype_internal": y_internal,
        "phenotype_pipeline": y_pipeline,
        "liability_or_continuous_value": liability[target_indices_master],
        "genetic_component_standardized": g_std[target_indices_master],
        "environmental_component_standardized": e_std[target_indices_master],
    })
    phenotype_table.to_csv(sim_dir / "simulated_phenotype.tsv", sep="\t", index=False)

    return fam, cov, y_internal


# =============================================================================
# FIVE-FOLD FILES
# =============================================================================

def make_folds(
    sim_dir: Path,
    fam: pd.DataFrame,
    cov: pd.DataFrame,
    phenotype_type: str,
    plink: str,
    genotype_prefix: Path,
    materialize: bool,
) -> None:
    y = fam[5].to_numpy()
    if phenotype_type == "binary":
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
        split_iter = splitter.split(np.zeros(len(y)), y)
    else:
        splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
        split_iter = splitter.split(np.zeros(len(y)))

    assignment = np.empty(len(y), dtype=int)

    for fold_id, (train_idx, test_idx) in enumerate(split_iter):
        assignment[test_idx] = fold_id
        fold_dir = sim_dir / f"Fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_fam = fam.iloc[train_idx].copy()
        test_fam = fam.iloc[test_idx].copy()
        train_cov = cov.iloc[train_idx].copy()
        test_cov = cov.iloc[test_idx].copy()

        train_fam[[0, 1]].to_csv(fold_dir / "train.keep", sep="\t", header=False, index=False)
        test_fam[[0, 1]].to_csv(fold_dir / "test.keep", sep="\t", header=False, index=False)
        train_fam.to_csv(fold_dir / "train_data.fam", sep="\t", header=False, index=False)
        test_fam.to_csv(fold_dir / "test_data.fam", sep="\t", header=False, index=False)
        train_cov.to_csv(fold_dir / "train_data.cov", sep="\t", index=False)
        test_cov.to_csv(fold_dir / "test_data.cov", sep="\t", index=False)

        if materialize:
            sh([
                plink,
                "--bfile", str(genotype_prefix),
                "--keep", str(fold_dir / "train.keep"),
                "--make-bed",
                "--out", str(fold_dir / "train_data"),
            ])
            sh([
                plink,
                "--bfile", str(genotype_prefix),
                "--keep", str(fold_dir / "test.keep"),
                "--make-bed",
                "--out", str(fold_dir / "test_data"),
            ])

            # Put phenotype values back into materialized FAMs.
            pmap = dict(zip(fam[1].astype(str), fam[5]))
            for stem in ["train_data", "test_data"]:
                ff = read_table_ws(fold_dir / f"{stem}.fam", header=None)
                ff[5] = ff[1].astype(str).map(pmap)
                ff.to_csv(fold_dir / f"{stem}.fam", sep="\t", header=False, index=False)

    pd.DataFrame({
        "FID": fam[0].astype(str),
        "IID": fam[1].astype(str),
        "fold": assignment,
    }).to_csv(sim_dir / "fold_assignments.csv", index=False)


# =============================================================================
# GWAS GENERATION USING PLINK 1.9
# =============================================================================

def prepare_discovery_frequency_table(shared: Path, plink: str, discovery_prefix: Path) -> pd.DataFrame:
    outprefix = shared / "cohorts" / "discovery_N10000" / "discovery_freq"
    frq = Path(str(outprefix) + ".frq")
    if not frq.exists():
        sh([
            plink,
            "--bfile", str(discovery_prefix),
            "--freq",
            "--out", str(outprefix),
        ])
    df = pd.read_csv(frq, sep=r"\s+")
    return df


def other_allele_from_bim(assoc: pd.DataFrame, bim: pd.DataFrame) -> pd.Series:
    b = bim[[1, 4, 5]].copy()
    b.columns = ["SNP", "BIM_A1", "BIM_A2"]
    b["SNP"] = b["SNP"].astype(str)
    tmp = assoc[["SNP", "A1"]].copy()
    tmp["SNP"] = tmp["SNP"].astype(str)
    tmp = tmp.merge(b, on="SNP", how="left")

    def choose(row):
        a1 = str(row["A1"])
        x = str(row["BIM_A1"])
        y = str(row["BIM_A2"])
        if a1 == x:
            return y
        if a1 == y:
            return x
        return y

    return tmp.apply(choose, axis=1)


def run_discovery_gwas(
    trait_dir: Path,
    trait_id: str,
    phenotype_type: str,
    discovery_prefix: Path,
    discovery_fam: pd.DataFrame,
    y_internal: np.ndarray,
    plink: str,
    discovery_freq: pd.DataFrame,
) -> Path:
    gwas_gz = trait_dir / f"{trait_id}.gz"
    if gwas_gz.exists():
        log(f"[skip] GWAS exists: {gwas_gz}")
        return gwas_gz

    trait_dir.mkdir(parents=True, exist_ok=True)
    y_pipeline = phenotype_to_pipeline(y_internal, phenotype_type)

    pheno = pd.DataFrame({
        "FID": discovery_fam[0].astype(str),
        "IID": discovery_fam[1].astype(str),
        "PHENO": y_pipeline,
    })
    pheno_path = trait_dir / "discovery.plink.pheno"
    pheno.to_csv(pheno_path, sep="\t", index=False)

    outprefix = trait_dir / "gwas_plink"
    if phenotype_type == "binary":
        cmd = [
            plink,
            "--bfile", str(discovery_prefix),
            "--pheno", str(pheno_path),
            "--allow-no-sex",
            "--logistic", "hide-covar",
            "--out", str(outprefix),
        ]
        result_path = Path(str(outprefix) + ".assoc.logistic")
    else:
        cmd = [
            plink,
            "--bfile", str(discovery_prefix),
            "--pheno", str(pheno_path),
            "--allow-no-sex",
            "--linear", "hide-covar",
            "--out", str(outprefix),
        ]
        result_path = Path(str(outprefix) + ".assoc.linear")

    sh(cmd)
    if not result_path.exists():
        raise FileNotFoundError(f"Expected GWAS output not found: {result_path}")

    assoc = pd.read_csv(result_path, sep=r"\s+")
    if "TEST" in assoc.columns:
        assoc = assoc[assoc["TEST"] == "ADD"].copy()

    assoc["SNP"] = assoc["SNP"].astype(str)
    assoc["A1"] = assoc["A1"].astype(str)

    if phenotype_type == "binary":
        assoc["BETA"] = np.log(pd.to_numeric(assoc["OR"], errors="coerce"))
    else:
        assoc["BETA"] = pd.to_numeric(assoc["BETA"], errors="coerce")

    stat = pd.to_numeric(assoc["STAT"], errors="coerce")
    beta = pd.to_numeric(assoc["BETA"], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        assoc["SE"] = np.abs(beta / stat)

    discovery_bim = read_table_ws(discovery_prefix.with_suffix(".bim"), header=None)
    assoc["A2"] = other_allele_from_bim(assoc, discovery_bim).values

    freq = discovery_freq[["SNP", "MAF"]].copy()
    freq["SNP"] = freq["SNP"].astype(str)
    assoc = assoc.merge(freq, on="SNP", how="left")

    n_col = "NMISS" if "NMISS" in assoc.columns else None
    if n_col is None:
        assoc["N"] = DISCOVERY_N
    else:
        assoc["N"] = pd.to_numeric(assoc[n_col], errors="coerce")

    assoc["P"] = pd.to_numeric(assoc["P"], errors="coerce")
    assoc["CHR"] = pd.to_numeric(assoc["CHR"], errors="coerce")
    assoc["BP"] = pd.to_numeric(assoc["BP"], errors="coerce")
    assoc["INFO"] = 1.0

    out = assoc[["CHR", "BP", "SNP", "A1", "A2", "N", "SE", "P", "BETA", "INFO", "MAF"]].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["CHR", "BP", "SNP", "A1", "A2", "N", "SE", "BETA"])
    out = out[out["SE"] > 0].copy()
    out["CHR"] = out["CHR"].astype(int)
    out["BP"] = out["BP"].astype(int)
    out.to_csv(gwas_gz, sep="\t", index=False, compression="gzip")


    log(f"[done] GWAS {trait_id}: {len(out):,} variants -> {gwas_gz}")
    return gwas_gz


# =============================================================================
# TRAIT / SIMULATION GENERATION
# =============================================================================

def trait_key(row) -> Tuple[str, float, int, int]:
    return (str(row.phenotype_type), float(row.true_h2), int(row.causal_snps), int(row.replicate))


def generate_all_traits_and_simulations(
    root: Path,
    shared: Path,
    trait_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    panels: Dict[str, Path],
    plink: str,
    copy_genotypes: bool,
    materialize_folds: bool,
) -> None:
    master_prefix = panels["master"]
    master_fam = read_table_ws(master_prefix.with_suffix(".fam"), header=None)
    master_iids = master_fam[1].astype(str).tolist()
    master_lookup = {iid: i for i, iid in enumerate(master_iids)}

    discovery_prefix = panels["discovery"]
    target5000_prefix = panels["target5000"]
    target750_prefix = panels["target750"]

    discovery_fam = read_table_ws(discovery_prefix.with_suffix(".fam"), header=None)
    target5000_fam = read_table_ws(target5000_prefix.with_suffix(".fam"), header=None)
    target750_fam = read_table_ws(target750_prefix.with_suffix(".fam"), header=None)

    discovery_idx = np.asarray([master_lookup[str(x)] for x in discovery_fam[1].astype(str)], dtype=int)
    target5000_idx = np.asarray([master_lookup[str(x)] for x in target5000_fam[1].astype(str)], dtype=int)
    target750_idx = np.asarray([master_lookup[str(x)] for x in target750_fam[1].astype(str)], dtype=int)

    if len(set(discovery_idx).intersection(set(target5000_idx))) != 0:
        raise RuntimeError("Discovery-target overlap detected during phenotype generation")

    target_panel_by_n = {
        TARGET_SMALL_N: target750_prefix,
        TARGET_LARGE_N: target5000_prefix,
    }
    target_indices_by_n = {
        TARGET_SMALL_N: target750_idx,
        TARGET_LARGE_N: target5000_idx,
    }

    discovery_freq = prepare_discovery_frequency_table(shared, plink, discovery_prefix)

    sim_lookup: Dict[Tuple[str, int], List[pd.Series]] = {}
    for _, srow in sim_df.iterrows():
        sim_lookup.setdefault((srow["trait_id"], int(srow["target_n"])), []).append(srow)

    # One causal 5000-SNP genotype read per replicate; all h2/type/architecture variants reuse it.
    truth_rows: List[dict] = []

    for rep in range(1, REPLICATES + 1):
        log(f"\n================ REPLICATE {rep}/{REPLICATES} ================")
        causal_indices, causal_all, beta_all = build_causal_plan(master_prefix, rep)
        log(f"[BED] loading master N={MASTER_N:,}, replicate={rep}, 5,000 causal candidates")
        Gmax = read_selected_bed(
            master_prefix.with_suffix(".bed"),
            n_samples=MASTER_N,
            variant_indices=causal_indices,
        )

        for causal_n in CAUSAL_SNP_COUNTS:
            G = Gmax[:, :causal_n]
            beta_raw = beta_all[:causal_n].copy()
            causal = causal_all.iloc[:causal_n].copy()

            Z, polymorphic, geno_means, geno_sds = standardize_columns(G)
            beta_used = beta_raw.copy()
            beta_used[~polymorphic] = 0.0
            g_raw = Z @ beta_used
            g_sd = float(g_raw.std(ddof=0))
            if g_sd <= 0:
                raise RuntimeError(f"rep={rep}, causal={causal_n}: zero genetic-score variance")
            g_std = ((g_raw - g_raw.mean()) / g_sd).astype(np.float64)

            causal["is_polymorphic_master"] = polymorphic
            causal["genotype_mean_master"] = geno_means
            causal["genotype_sd_master"] = geno_sds
            causal["beta_on_standardized_genotype"] = beta_used
            causal["beta_effective_after_score_scaling"] = beta_used / g_sd

            del Z

            for phenotype_type in PHENOTYPE_TYPES:
                type_offset = 0 if phenotype_type == "continuous" else 10000
                phenotype_seed = PHENOTYPE_SEED_BASE + type_offset + causal_n + rep
                e_std = make_standard_normal(MASTER_N, phenotype_seed)

                for h2 in H2_VALUES:
                    matches = trait_df[
                        (trait_df["phenotype_type"] == phenotype_type)
                        & (trait_df["true_h2"] == h2)
                        & (trait_df["causal_snps"] == causal_n)
                        & (trait_df["replicate"] == rep)
                    ]
                    if len(matches) != 1:
                        raise RuntimeError("Trait-manifest lookup is not unique")
                    trait_row = matches.iloc[0]
                    trait_id = str(trait_row["trait_id"])
                    trait_dir = shared / "discovery_traits" / trait_id
                    trait_dir.mkdir(parents=True, exist_ok=True)

                    g_component = math.sqrt(h2) * g_std
                    e_component = math.sqrt(1.0 - h2) * e_std
                    liability = g_component + e_component

                    if phenotype_type == "binary":
                        threshold = float(norm.ppf(1.0 - BINARY_POPULATION_PREVALENCE))
                        y_internal_master = (liability > threshold).astype(np.int8)
                    else:
                        threshold = None
                        y_internal_master = liability.astype(np.float64)

                    master_metrics = component_metrics(
                        g_std, e_std, h2, np.arange(MASTER_N),
                        y_internal_master if phenotype_type == "binary" else None,
                    )
                    discovery_metrics = component_metrics(
                        g_std, e_std, h2, discovery_idx,
                        y_internal_master if phenotype_type == "binary" else None,
                    )
                    target5000_metrics = component_metrics(
                        g_std, e_std, h2, target5000_idx,
                        y_internal_master if phenotype_type == "binary" else None,
                    )
                    target750_metrics = component_metrics(
                        g_std, e_std, h2, target750_idx,
                        y_internal_master if phenotype_type == "binary" else None,
                    )

                    # Save enough shared state to resume/audit the unique biological trait.
                    np.savez_compressed(
                        trait_dir / "master_components.npz",
                        g_std=g_std.astype(np.float32),
                        e_std=e_std.astype(np.float32),
                        liability=liability.astype(np.float32),
                        phenotype_internal=y_internal_master,
                    )
                    causal.to_csv(trait_dir / "causal_snps.tsv", sep="\t", index=False)

                    discovery_y = y_internal_master[discovery_idx]
                    discovery_table = pd.DataFrame({
                        "FID": discovery_fam[0].astype(str),
                        "IID": discovery_fam[1].astype(str),
                        "phenotype_internal": discovery_y,
                        "phenotype_pipeline": phenotype_to_pipeline(discovery_y, phenotype_type),
                        "liability_or_continuous_value": liability[discovery_idx],
                        "genetic_component_standardized": g_std[discovery_idx],
                        "environmental_component_standardized": e_std[discovery_idx],
                    })
                    discovery_table.to_csv(
                        trait_dir / "discovery_phenotype.tsv", sep="\t", index=False
                    )

                    trait_gwas = run_discovery_gwas(
                        trait_dir=trait_dir,
                        trait_id=trait_id,
                        phenotype_type=phenotype_type,
                        discovery_prefix=discovery_prefix,
                        discovery_fam=discovery_fam,
                        y_internal=discovery_y,
                        plink=plink,
                        discovery_freq=discovery_freq,
                    )


                    trait_truth = {
                        "design_version": DESIGN_VERSION,
                        "trait_id": trait_id,
                        "phenotype_type": phenotype_type,
                        "target_h2": h2,
                        "true_h2_scale": "liability" if phenotype_type == "binary" else "continuous",
                        "causal_snps": causal_n,
                        "effective_polymorphic_causal_snps_master": int(polymorphic.sum()),
                        "replicate": rep,
                        "population_prevalence": (
                            BINARY_POPULATION_PREVALENCE if phenotype_type == "binary" else None
                        ),
                        "binary_liability_threshold": threshold,
                        "master_n": MASTER_N,
                        "discovery_n": DISCOVERY_N,
                        "target_large_n": TARGET_LARGE_N,
                        "target_small_n": TARGET_SMALL_N,
                        "discovery_target_overlap_n": 0,
                        "genotype_seed": GENOTYPE_SEED,
                        "cohort_split_seed": COHORT_SPLIT_SEED,
                        "causal_seed": CAUSAL_SEED_BASE + rep,
                        "effect_seed": EFFECT_SEED_BASE + rep,
                        "phenotype_seed": phenotype_seed,
                        "fold_seed": FOLD_SEED,
                        "master": master_metrics,
                        "discovery": discovery_metrics,
                        "target5000": target5000_metrics,
                        "target750": target750_metrics,
                    }
                    write_json(trait_truth, trait_dir / "trait_truth.json")

                    # Create both target-size SimXXX datasets for this one biological trait.
                    for target_n in TARGET_SAMPLE_SIZES:
                        sims = sim_lookup.get((trait_id, target_n), [])
                        if len(sims) != 1:
                            raise RuntimeError(f"Expected one Sim for {trait_id}, N={target_n}; got {len(sims)}")
                        srow = sims[0]
                        sim_id = str(srow["simulation_id"])
                        sim_dir = root / sim_id
                        target_prefix = target_panel_by_n[target_n]
                        target_idx = target_indices_by_n[target_n]

                        fam, cov, target_y = create_target_files(
                            sim_dir=sim_dir,
                            sim_id=sim_id,
                            target_prefix=target_prefix,
                            target_indices_master=target_idx,
                            g_std=g_std,
                            e_std=e_std,
                            liability=liability,
                            y_internal_master=y_internal_master,
                            phenotype_type=phenotype_type,
                            copy_genotypes=copy_genotypes,
                        )

                        # Every simulation folder gets its own GWAS and truth files.
                        copy_file(trait_gwas, sim_dir / f"{sim_id}.gz")
                        copy_file(trait_dir / "discovery_phenotype.tsv", sim_dir / "discovery_phenotype.tsv")
                        causal.to_csv(sim_dir / "causal_snps.tsv", sep="\t", index=False)

                        make_folds(
                            sim_dir=sim_dir,
                            fam=fam,
                            cov=cov,
                            phenotype_type=phenotype_type,
                            plink=plink,
                            genotype_prefix=sim_dir / sim_id,
                            materialize=materialize_folds,
                        )


                        selected_metrics = target750_metrics if target_n == TARGET_SMALL_N else target5000_metrics
                        sample_prev = (
                            float(target_y.mean()) if phenotype_type == "binary" else None
                        )

                        sim_truth = {
                            **trait_truth,
                            "simulation_id": sim_id,
                            "target_n": target_n,
                            "realized_target_h2": selected_metrics["realized_h2"],
                            "target_sample_prevalence": sample_prev,
                            "target_cases": int(target_y.sum()) if phenotype_type == "binary" else None,
                            "target_controls": (
                                int(target_n - target_y.sum()) if phenotype_type == "binary" else None
                            ),
                            "discovery_gwas_file": f"{sim_id}.gz",
                        }
                        write_json(sim_truth, sim_dir / "truth.json")

                        summary_row = {
                            "simulation_id": sim_id,
                            "trait_id": trait_id,
                            "phenotype_type": phenotype_type,
                            "target_n": target_n,
                            "target_h2": h2,
                            "causal_snps": causal_n,
                            "replicate": rep,
                            "population_prevalence": (
                                BINARY_POPULATION_PREVALENCE if phenotype_type == "binary" else np.nan
                            ),
                            "master_realized_h2": master_metrics["realized_h2"],
                            "discovery_realized_h2": discovery_metrics["realized_h2"],
                            "target5000_realized_h2": target5000_metrics["realized_h2"],
                            "target750_realized_h2": target750_metrics["realized_h2"],
                            "realized_target_h2": selected_metrics["realized_h2"],
                            "master_sample_prevalence": master_metrics.get("sample_prevalence", np.nan),
                            "discovery_sample_prevalence": discovery_metrics.get("sample_prevalence", np.nan),
                            "target_sample_prevalence": sample_prev if sample_prev is not None else np.nan,
                            "discovery_target_overlap_n": 0,
                        }
                        truth_rows.append(summary_row)

                        # Persist progressively in case a long HPC job is interrupted.
                        pd.DataFrame(truth_rows).sort_values("simulation_id").to_csv(
                            root / "simulation_truth_summary.csv", index=False
                        )

                        log(
                            f"[done] {sim_id} <- {trait_id}: type={phenotype_type}, "
                            f"N={target_n}, h2={h2}, causal={causal_n}, rep={rep}, "
                            f"realized_target_h2={selected_metrics['realized_h2']:.4f}"
                        )

        del Gmax

    final = pd.DataFrame(truth_rows).sort_values("simulation_id")
    final.to_csv(root / "simulation_truth_summary.csv", index=False)


# =============================================================================
# FINAL OUTPUT VALIDATION
# =============================================================================

def validate_complete_dataset(root: Path) -> None:
    """Fail loudly if any of the 200 simulation folders is incomplete."""
    required_columns = {
        "CHR", "BP", "SNP", "A1", "A2", "N", "SE", "P", "BETA", "INFO", "MAF"
    }
    problems: List[str] = []

    for i in range(1, 201):
        sim_id = f"Sim{i:03d}"
        sim_dir = root / sim_id
        required = [
            sim_dir / f"{sim_id}.bed",
            sim_dir / f"{sim_id}.bim",
            sim_dir / f"{sim_id}.fam",
            sim_dir / f"{sim_id}.height",
            sim_dir / f"{sim_id}.cov",
            sim_dir / f"{sim_id}.gz",
            sim_dir / "simulated_phenotype.tsv",
            sim_dir / "discovery_phenotype.tsv",
            sim_dir / "causal_snps.tsv",
            sim_dir / "truth.json",
            sim_dir / "fold_assignments.csv",
        ]
        for path in required:
            if not path.exists() or path.stat().st_size == 0:
                problems.append(f"{sim_id}: missing/empty {path.name}")

        for fold in range(N_FOLDS):
            fold_dir = sim_dir / f"Fold_{fold}"
            for name in [
                "train.keep", "test.keep",
                "train_data.fam", "test_data.fam",
                "train_data.cov", "test_data.cov",
            ]:
                p = fold_dir / name
                if not p.exists() or p.stat().st_size == 0:
                    problems.append(f"{sim_id}/Fold_{fold}: missing/empty {name}")

        gwas = sim_dir / f"{sim_id}.gz"
        if gwas.exists() and gwas.stat().st_size > 0:
            try:
                hdr = pd.read_csv(gwas, sep=r"\s+", compression="gzip", nrows=2)
                missing = required_columns.difference(hdr.columns)
                if missing:
                    problems.append(
                        f"{sim_id}: GWAS missing columns {sorted(missing)}"
                    )
            except Exception as exc:
                problems.append(f"{sim_id}: cannot read GWAS ({exc})")

    summary = root / "simulation_truth_summary.csv"
    if not summary.exists():
        problems.append("missing simulation_truth_summary.csv")
    else:
        try:
            sdf = pd.read_csv(summary)
            if len(sdf) != 200:
                problems.append(
                    f"simulation_truth_summary.csv has {len(sdf)} rows, expected 200"
                )
        except Exception as exc:
            problems.append(f"cannot read simulation_truth_summary.csv ({exc})")

    if problems:
        preview = "\n".join("  - " + x for x in problems[:50])
        extra = "" if len(problems) <= 50 else f"\n  ... and {len(problems)-50} more"
        raise RuntimeError(
            "Simulation output validation failed:\n" + preview + extra
        )

    log("[validate] All 200 SimXXX directories contain the required data files.")


# =============================================================================
# DESIGN METADATA
# =============================================================================

def write_design_metadata(shared: Path, plink: str) -> None:
    write_json(
        {
            "design_version": DESIGN_VERSION,
            "species": SPECIES_ID,
            "demographic_model": DEMOGRAPHIC_MODEL,
            "population": POPULATION,
            "genetic_map": GENETIC_MAP,
            "chromosomes": CHROMOSOMES,
            "segment_bp_per_chromosome": SEGMENT_BP,
            "maf_min": MAF_MIN,
            "target_common_snps": TARGET_COMMON_SNPS,
            "master_n": MASTER_N,
            "discovery_n": DISCOVERY_N,
            "target_large_n": TARGET_LARGE_N,
            "target_small_n": TARGET_SMALL_N,
            "discovery_target_overlap_n": 0,
            "phenotype_types": PHENOTYPE_TYPES,
            "h2_values": H2_VALUES,
            "causal_snp_counts": CAUSAL_SNP_COUNTS,
            "replicates": REPLICATES,
            "binary_population_prevalence": BINARY_POPULATION_PREVALENCE,
            "n_unique_traits": 100,
            "n_target_datasets": 200,
            "n_folds": N_FOLDS,
            "step1_scope": "data_generation_only",
            "gwas_generation": "PLINK 1.9 discovery-cohort association analysis",
            "truth_h2_source": "known simulated genetic/environmental components",
            "seeds": {
                "genotype": GENOTYPE_SEED,
                "cohort_split": COHORT_SPLIT_SEED,
                "causal_seed_base": CAUSAL_SEED_BASE,
                "effect_seed_base": EFFECT_SEED_BASE,
                "phenotype_seed_base": PHENOTYPE_SEED_BASE,
                "fold": FOLD_SEED,
            },
            "data_generation_tool": {"plink": plink},
            "heritability_estimators_run_in_step1": [],
        },
        shared / "simulation_design.json",
    )


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 200 ground-truth target datasets plus independent "
            "discovery GWAS summary statistics. This step performs data "
            "generation only and runs no heritability estimator."
        )
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository/output root. Default: current directory, so Sim001 is ./Sim001.",
    )
    parser.add_argument(
        "--plink",
        default=None,
        help="PLINK 1.9 executable. Auto-detects ./plink or plink.",
    )
    parser.add_argument(
        "--copy-genotypes",
        action="store_true",
        help="Physically copy BED/BIM into each SimXXX instead of hard-linking them.",
    )
    parser.add_argument(
        "--materialize-folds",
        action="store_true",
        help=(
            "Also create train_data/test_data BED/BIM/FAM in every fold. "
            "This substantially increases disk use; normally Step2 can create them later."
        ),
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep per-chromosome tree-sequence/PLINK intermediates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    shared = root / "_simulation_shared"
    shared.mkdir(parents=True, exist_ok=True)

    verify_clean_or_compatible_root(root, shared)

    plink = find_executable(args.plink, ["./plink", "plink"])
    if not plink:
        raise FileNotFoundError(
            "PLINK 1.9 was not found. Put ./plink in the repository root or pass --plink PATH."
        )

    write_design_metadata(shared, plink)

    log("\n=== STEP 1: DATA GENERATION ONLY ===")
    log(f"Master population        : N={MASTER_N:,}")
    log(f"Discovery GWAS cohort    : N={DISCOVERY_N:,}")
    log(f"Target cohort            : N={TARGET_LARGE_N:,}")
    log(f"Nested small target      : N={TARGET_SMALL_N:,}")
    log("Discovery-target overlap : 0")
    log("Unique biological traits : 100")
    log("Final SimXXX datasets    : 200")
    log(f"PLINK                    : {plink}")
    log("Heritability estimators  : NONE in Step 1")

    trait_df = build_trait_manifest(shared)
    sim_df = build_simulation_manifest(root, shared, trait_df)
    write_parameter_files(root, sim_df)

    master_prefix = simulate_master_genotypes(
        shared=shared,
        plink=plink,
        keep_intermediates=args.keep_intermediates,
    )
    panels = create_cohorts(shared, plink, master_prefix)

    generate_all_traits_and_simulations(
        root=root,
        shared=shared,
        trait_df=trait_df,
        sim_df=sim_df,
        panels=panels,
        plink=plink,
        copy_genotypes=args.copy_genotypes,
        materialize_folds=args.materialize_folds,
    )

    validate_complete_dataset(root)

    log("\n============================================================")
    log("STEP 1 DATA GENERATION COMPLETE")
    log("============================================================")
    log(f"Truth summary : {root / 'simulation_truth_summary.csv'}")
    log(f"Manifest      : {root / 'simulation_manifest.csv'}")
    log("No LDSC/GCTA/GEMMA/LDAK/DPR heritability estimation was run.")
    log("Validation commands:")
    log("  ls Sim001")
    log("  zcat Sim001/Sim001.gz | head")
    log("  cat Sim001/truth.json")
    log("  python Step2-PerformQualityControls.py Sim001")
    log("  python Method1.py Sim001 0")


if __name__ == "__main__":
    main()
