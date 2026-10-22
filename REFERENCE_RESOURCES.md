# External software and reference data: download and placement guide

This guide complements [README.md](README.md). It documents where to obtain third-party software and LD-score/tagging references and **the paths hard-coded in the current benchmark scripts**. Neither third-party binaries nor downloaded human reference panels are bundled with this repository. Download them from their original maintainers, respecting the providers' terms. UK Biobank participant-level genotype/phenotype files must not be redistributed.

All relative paths below are **relative to the working directory from which the benchmark Python scripts are launched** (normally the repository root). Use ancestry-, variant-set- and genome-build-compatible references. The original analyses used their specified historical software/reference versions; downloading a newer program or reference is not a guarantee of numerically identical results.

## 1. Software: official downloads and required local names

| Software | Official download / installation page | Expected by current scripts |
| --- | --- | --- |
| **PLINK 1.9** (not PLINK 2) | [PLINK 1.9, choose the binary for your OS](https://www.cog-genomics.org/plink/) | Executable `./plink` in working directory |
| **GCTA** (GREML / SBLUP) | [GCTA executable downloads](https://yanglab.westlake.edu.cn/software/gcta/) and [official source](https://github.com/JianYang-Lab/GCTA) | Executable `./gcta` |
| **GEMMA** | [GEMMA release binaries](https://github.com/genetics-statistics/GEMMA/releases) and [installation instructions](https://github.com/genetics-statistics/GEMMA/blob/master/INSTALL.md) | Executable `./gemma` |
| **DPR** (latent Dirichlet Process Regression) | [DPR source and build instructions](https://github.com/biostatpzeng/DPR) | Executable `./DPR` (uppercase) |
| **LDSC** | [Official LDSC repository](https://github.com/bulik/ldsc), [installation/wiki](https://github.com/bulik/ldsc/wiki) | `ldsc.py` and its Python `ldscore` package importable from the working directory; run in its compatible Python environment |
| **LDAK / SumHer** | [Official LDAK downloads](https://github.com/dougspeed/LDAK) and [LDAK instructions](https://dougspeed.com/downloads/) | Executable `./ldak` |
| **LDpred2** | [LDpred2 installation and tutorial](https://privefl.github.io/bigsnpr/articles/LDpred2.html) | R packages used by the repository's `LDpred-2-*.R` scripts; `Rscript` on `PATH` |
| **stdpopsim / msprime / tskit** | [stdpopsim installation](https://popsim-consortium.github.io/stdpopsim-docs/stable/) and [msprime](https://tskit.dev/msprime/docs/stable/) | Python dependencies for `Step1-Simulate.py` |
| **R** | [R Project downloads](https://cran.r-project.org/) | `Rscript` available on `PATH` |

The project provides environment specifications such as `leopard-sdm.yml`, `ldsc.yml` and `ldscc.yml`; inspect them before adding packages. Install the **historically compatible tool versions** for exact numerical replication and record the versions actually used.

### Install executable files where the scripts look for them

After downloading and unpacking each official distribution, copy or symlink its actual executable under the expected basename. Examples (replace `/path/to/...` with real installed paths):

```bash
# Run from the benchmark working directory
ln -s /path/to/plink    ./plink
ln -s /path/to/gcta     ./gcta
ln -s /path/to/gemma    ./gemma
ln -s /path/to/DPR      ./DPR
ln -s /path/to/ldak     ./ldak

# Only executable files need execute permission; symbolic links inherit target permissions.
./plink --version
./gemma -h
command -v Rscript
```

If the destination already exists, do not run `ln -s` over it; use the installed executable or replace the link only after checking the version. Some GCTA/LDAK distributions contain versioned filenames (for example, `ldak7.01.linux`); rename or symlink the **executable** to the basename expected above.

For LDSC, the scripts call `python ldsc.py` (or use the active Python interpreter), rather than calling an executable from `PATH`. Make the original `ldsc.py` and accompanying `ldscore/` package available together, and use a compatible LDSC environment. For example, download the upstream source *outside* this tracked repository and make those source files available to the working directory (do not commit the third-party source here). The existing `Method10-Sim.py` specifically uses `sys.executable` to run `ldsc.py`.

To install LDpred2 in R (in the R environment used for PRS):

```r
install.packages("bigsnpr")
# Alternatively, follow the upstream tutorial for the required historical version:
# remotes::install_github("privefl/bigsnpr")
```

Other R dependencies are referenced in the repository's individual `.R` files; install them in the corresponding R environment.

## 2. European LDSC LD-score reference files (empirical Method 7)

**Original sources:** [LDSC heritability / genetic-correlation tutorial](https://github.com/bulik/ldsc/wiki/Heritability-and-Genetic-Correlation), [Broad/Alkes European LD-score archive](https://data.broadinstitute.org/alkesgroup/LDSCORE/eur_w_ld_chr.tar.bz2), and [public Zenodo mirror of the European LD scores](https://doi.org/10.5281/zenodo.8182036). The public mirror can be helpful when Broad/cloud access is costly or unavailable. Confirm the license and provenance for the selected source.

`Method7.py` explicitly requests two reference choices (`eur_ref_ld_chr`, `eur_w_ld_chr`) crossed with two weight choices of the same names:

```text
LDSCFILES/
├── eur_ref_ld_chr/
│   ├── 1.l2.ldscore.gz
│   ├── 1.l2.M
│   ├── 1.l2.M_5_50
│   └── ... through chromosome 22
└── eur_w_ld_chr/
    ├── 1.l2.ldscore.gz
    ├── 1.l2.M
    ├── 1.l2.M_5_50
    └── ... through chromosome 22
```

Get the matching **two original reference directories** from the LDSC/Alkes resources used for your analysis. A widely documented download for `eur_w_ld_chr` is:

```bash
mkdir -p LDSCFILES
wget -c https://data.broadinstitute.org/alkesgroup/LDSCORE/eur_w_ld_chr.tar.bz2
tar -tjf eur_w_ld_chr.tar.bz2 | head
tar -xjf eur_w_ld_chr.tar.bz2 -C LDSCFILES
```

A Zenodo mirror is linked above if the original host is inaccessible. **Check the archive's actual extracted directory structure**; it is not safe to assume that this one archive also supplies `eur_ref_ld_chr`. Obtain `eur_ref_ld_chr` from the original LDSC reference collection separately (the historical resource is named `eur_ref_ld_chr.tar.bz2`), or rebuild a separately identified reference following the [official LDSC LD-score generation tutorial](https://github.com/bulik/ldsc/wiki/LD-Score-Estimation-Tutorial). Do **not** copy or rename the same `eur_w_ld_chr` files to mimic an independent reference; doing so would invalidate the four-reference/weight comparison.

Check both directories before running empirical Method 7:

```bash
for ref in eur_ref_ld_chr eur_w_ld_chr; do
  for chr in $(seq 1 22); do
    for suffix in .l2.ldscore.gz .l2.M .l2.M_5_50; do
      test -s "LDSCFILES/$ref/$chr$suffix" || echo "MISSING: LDSCFILES/$ref/$chr$suffix"
    done
  done
done
```

Use ancestry-appropriate scores for other populations. The public EUR scores are **not** a universal ancestry-independent reference. Do not silently substitute a newer reference when trying to reproduce the benchmark's exact empirical comparisons.

## 3. LDAK precomputed tagging files (empirical Method 9)

**Official source:** [Doug Speed, Pre-computed Taggings](https://dougspeed.com/pre-computed-tagging-files/). The original provider offers GBR, SAS, EAS and AFR variants, with HapMap3 and directly genotyped SNP sets. The empirical `Method8-Precomputed.py` in this repository specifically requests the following **six GBR files directly under `LDAKFILES/`**:

```text
LDAKFILES/
├── bld.ldak.hapmap.gbr.tagging
├── ldak.thin.hapmap.gbr.tagging
├── bld.ldak.lite.alpha.hapmap.gbr.tagging
├── bld.ldak.genotyped.gbr.tagging
├── ldak.thin.genotyped.gbr.tagging
└── bld.ldak.lite.alpha.genotyped.gbr.tagging
```

For convenience, the provider's published naming convention supports downloads such as:

```bash
mkdir -p LDAKFILES
for filename in \
  bld.ldak.hapmap.gbr.tagging \
  ldak.thin.hapmap.gbr.tagging \
  bld.ldak.lite.alpha.hapmap.gbr.tagging \
  bld.ldak.genotyped.gbr.tagging \
  ldak.thin.genotyped.gbr.tagging \
  bld.ldak.lite.alpha.genotyped.gbr.tagging
do
  wget -c -O "LDAKFILES/$filename.gz" \
    "https://genetics.ghpc.au.dk/doug/TaggingFiles/$filename.gz" || exit 1
  gzip -t "LDAKFILES/$filename.gz" || exit 1
  gunzip -f "LDAKFILES/$filename.gz"
done
```

The exact provider filenames and available download mirrors can change. If a download fails, select the matching file using the **official source page** and place the decompressed `.tagging` file at the exact path above. These GBR references may be based on UK Biobank data; use them according to the publisher's terms and do not automatically redistribute them under this repository's MIT license. The LDAK publisher advises checking that most regression SNPs in the tagging file are represented in the supplied GWAS summary statistics.

### Additional resources for LDAK calculated tagging (Method 8)

`Method8-Calculated.py` builds tagging information with `./ldak`, but its BLD-LDAK branch also refers to annotation resources under `LDAKFILES/bld` and `LDAKFILES/bld65`. Refer to the publisher's [BLD-LDAK annotation and genome-build instructions](https://dougspeed.com/bldldak/) and [Calculate Taggings](https://dougspeed.com/calculate-taggings/) to prepare the required corresponding annotations. They are **not** the six precomputed tagging files listed above. Check that the annotation coordinates are compatible with your genotype build.

## 4. Simulation-specific reference resources (not distributed)

`Step1-Simulate.py` generates synthetic genotypes, phenotypes and an independent discovery GWAS; it **does not automatically create every reference file required by every simulated heritability method**. This repository intentionally does not upload the synthetic LD-score or tagging reference files. Users can prepare these from a matching synthetic reference panel using the documented upstream software.

| Simulation script | Required location | How to obtain |
| --- | --- | --- |
| `Method7-Sim.py` | `LDSCFILES_SIM/Reference5000/1.l2.ldscore.gz` through `22.l2.ldscore.gz`, plus `.l2.M` and `.l2.M_5_50` for each chromosome | Generate from a **defined 5,000-person simulation reference** with [LDSC LD-score generation](https://github.com/bulik/ldsc/wiki/LD-Score-Estimation-Tutorial); record the panel's exact individuals, variants and parameters |
| `Method8-Precomputed-Sim.py` | `LDAKFILES_SIM/Reference5000/sim_human.tagging`, `sim_gcta.tagging`, `sim_alpha_1.0.tagging`, `sim_alpha_0.5.tagging`, `sim_alpha_0.0.tagging` and `sim_alpha_-0.5.tagging` | Calculate each corresponding model from the same declared simulated reference using [LDAK Calculate Taggings](https://dougspeed.com/calculate-taggings/); use compatible models and matching SNP IDs |
| `Method10-Sim.py` | `LDSCFILES_SIM/<SimXXX>/Fold_<fold>/Method10/` | **Generated automatically by `Method10-Sim.py`**, using PLINK and `ldsc.py --l2`; no shared `Reference5000` download required |

Example directory structure:

```text
LDSCFILES_SIM/
└── Reference5000/
    ├── 1.l2.ldscore.gz
    ├── 1.l2.M
    ├── 1.l2.M_5_50
    └── ... through chromosome 22

LDAKFILES_SIM/
└── Reference5000/
    ├── sim_human.tagging
    ├── sim_gcta.tagging
    ├── sim_alpha_1.0.tagging
    ├── sim_alpha_0.5.tagging
    ├── sim_alpha_0.0.tagging
    └── sim_alpha_-0.5.tagging
```

**Important reproducibility distinction:** `Method7-Sim.py` and `Method8-Precomputed-Sim.py` **consume** shared precomputed resources; they do not generate those shared references on their own. `Method10-Sim.py` **does** generate its own per-fold LDSC scores. When regenerating shared `Reference5000` resources, document panel selection, ancestry/simulation seed, SNP inclusion, software versions, LD window, and tagging model options. Merely using the same filenames does not guarantee reproduction of the original numerical values.

Verify all expected LDSC files:

```bash
for chr in $(seq 1 22); do
  for suffix in .l2.ldscore.gz .l2.M .l2.M_5_50; do
    test -s "LDSCFILES_SIM/Reference5000/$chr$suffix" ||
      echo "MISSING: LDSCFILES_SIM/Reference5000/$chr$suffix"
  done
done
```

## 5. Optional public data/utility sources

- [GWAS Catalog](https://www.ebi.ac.uk/gwas/) — public GWAS study and summary-statistic access; check each study's own access terms.
- [GWASPokerforPRS](https://github.com/MuhammadMuneeb007/GWASPokerforPRS) — processing GWAS data for PRS workflows.
- [1000 Genomes Project](https://www.internationalgenome.org/data) — genotype reference resources where compatible with the intended build and ancestry.
- [LDpred2 tutorial and HM3/LD reference options](https://privefl.github.io/bigsnpr/articles/LDpred2.html) — use only when compatible with the study and its SNP selection.
- [PLINK documentation](https://www.cog-genomics.org/plink/1.9/) — genotype QC and clumping.
- [Official LDSC LD-score estimation tutorial](https://github.com/bulik/ldsc/wiki/LD-Score-Estimation-Tutorial) — calculate LD-score files rather than redistributing them.

## 6. Validate local file placement before running methods

```bash
# Run in the benchmark working directory.
for exe in plink gcta gemma DPR ldak; do
  if [[ -x "./$exe" ]]; then echo "OK executable: ./$exe"
  else echo "MISSING executable: ./$exe"; fi
done

test -f ldsc.py && echo "OK: ldsc.py" || echo "MISSING: ldsc.py"
test -d ldscore && echo "OK: ldscore/" || echo "CHECK LDSC's ldscore Python package"

for x in eur_ref_ld_chr eur_w_ld_chr; do
  test -d "LDSCFILES/$x" && echo "OK: LDSCFILES/$x" ||
    echo "MISSING: LDSCFILES/$x"
done

for x in \
  bld.ldak.hapmap.gbr.tagging \
  ldak.thin.hapmap.gbr.tagging \
  bld.ldak.lite.alpha.hapmap.gbr.tagging \
  bld.ldak.genotyped.gbr.tagging \
  ldak.thin.genotyped.gbr.tagging \
  bld.ldak.lite.alpha.genotyped.gbr.tagging
do
  test -s "LDAKFILES/$x" && echo "OK: LDAKFILES/$x" ||
    echo "MISSING: LDAKFILES/$x"
done
```

**Provenance and licensing:** Cite the original software and reference creators. A publicly accessible download does not necessarily grant unrestricted redistribution rights. Cloud egress/requester-pays charges, if any, are hosting charges and do not themselves establish a commercial data license. This repository provides download locations and file-layout guidance, not mirrored copies of third-party datasets.