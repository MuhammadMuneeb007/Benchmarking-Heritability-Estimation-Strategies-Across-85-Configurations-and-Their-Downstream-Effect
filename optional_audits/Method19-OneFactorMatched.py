#!/usr/bin/env python3

import os
import sys
import glob
import hashlib
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ============================================================
# USAGE
# ============================================================
#
# python Method19-OneFactorMatched.py real
# python Method19-OneFactorMatched.py simulated
#
# Reads existing Results.csv files only.
# Saves NOTHING.
#
# PURPOSE
# -------
# 1. Compare configurations differing in ONE analytical factor.
# 2. Hold all other controllable factors constant.
# 3. Verify the ACTUAL input SNP set using the .bim file.
# 4. Do NOT use Results.csv numberofvariants to identify input SNP sets.
# 5. Keep numberofvariants as an effective/output quantity.
# 6. Average overlapping CV folds before inference.
# 7. Use phenotype-level effects for inference in real data.
# 8. Apply Benjamini-Hochberg FDR correction.
# 9. Flag suspicious all-zero contrasts for upstream audit.
#
# THREE DECOMPOSITION GROUPS
# --------------------------
#
# 1. DATA / PREPROCESSING
#    - clumping/pruning
#    - covariates
#    - PCA
#    - SNP / variant-set choice
#
# 2. ESTIMATOR / MODEL CONFIGURATION
#    - HE vs REML
#    - centred vs standardised GRM
#    - LDAK tagging/model assumptions
#
# 3. INPUT / REFERENCE RESOURCE
#    - GEMMA vs DPR GRM source
#    - reference strategy where isolatable
#    - LDSC LD-reference choice
#    - LDSC regression-weight choice
#
# ============================================================


# ============================================================
# COMMAND LINE
# ============================================================

if len(sys.argv) != 2:
    print()
    print("USAGE:")
    print("  python Method19-OneFactorMatched.py real")
    print("  python Method19-OneFactorMatched.py simulated")
    print()
    sys.exit(1)


_mode = sys.argv[1].strip().lower()

aliases = {
    "real": "real",
    "sim": "simulated",
    "simulation": "simulated",
    "simulated": "simulated",
}

if _mode not in aliases:
    raise SystemExit(
        "ERROR: mode must be real or simulated"
    )

MODE = aliases[_mode]


# ============================================================
# REAL PHENOTYPES
# ============================================================

REAL_PHENOTYPES = [
    "asthma",
    "body_mass_index_bmi",
    "depression",
    "gastro_oesophageal_reflux_gord_gastric_reflux",
    "high_cholesterol",
    "hypertension",
    "hypothyroidism_myxoedema",
    "irritable_bowel_syndrome",
    "migraine",
    "osteoarthritis",
]


PHENOTYPE_LABELS = {

    "asthma":
        "Asthma",

    "body_mass_index_bmi":
        "Body Mass Index",

    "depression":
        "Depression",

    "gastro_oesophageal_reflux_gord_gastric_reflux":
        "Gastro-Oesophageal Reflux",

    "high_cholesterol":
        "High Cholesterol",

    "hypertension":
        "Hypertension",

    "hypothyroidism_myxoedema":
        "Hypothyroidism",

    "irritable_bowel_syndrome":
        "Irritable Bowel Syndrome",

    "migraine":
        "Migraine",

    "osteoarthritis":
        "Osteoarthritis",
}


# ============================================================
# METHOD METADATA
# ============================================================

METHODS = {

    "LDpred-2-Heritability": {

        "family":
            "LDpred2",

        "engine":
            "LDpred2-LDSC",

        "input_type":
            "GWAS+genotype",

        "uses_genotype":
            True,

        "uses_gwas":
            True,

        "grm_source":
            "none",

        "reference_source":
            "internal_LDpred2",
    },


    "GCTA-Heritability": {

        "family":
            "GCTA",

        "engine":
            "GCTA",

        "input_type":
            "genotype",

        "uses_genotype":
            True,

        "uses_gwas":
            False,

        "grm_source":
            "GCTA",

        "reference_source":
            "none",
    },


    "Gemma1-Heritability": {

        "family":
            "GEMMA",

        "engine":
            "GEMMA",

        "input_type":
            "genotype",

        "uses_genotype":
            True,

        "uses_gwas":
            False,

        "grm_source":
            "GEMMA",

        "reference_source":
            "none",
    },


    "Gemma2-Heritability": {

        "family":
            "GEMMA",

        "engine":
            "GEMMA",

        "input_type":
            "GWAS+genotype",

        "uses_genotype":
            True,

        "uses_gwas":
            True,

        "grm_source":
            "GEMMA",

        "reference_source":
            "none",
    },


    "Gemma-LDSC-Heritability": {

        "family":
            "GEMMA",

        "engine":
            "GEMMA",

        "input_type":
            "GWAS+genotype",

        "uses_genotype":
            True,

        "uses_gwas":
            True,

        "grm_source":
            "LDSC-derived",

        "reference_source":
            "genotype_LD_scores",
    },


    "DPR-Gemma-Heritability": {

        "family":
            "DPR+GEMMA",

        "engine":
            "GEMMA",

        "input_type":
            "genotype",

        "uses_genotype":
            True,

        "uses_gwas":
            False,

        "grm_source":
            "DPR",

        "reference_source":
            "none",
    },


    "LDSC-Heritability": {

        "family":
            "LDSC",

        "engine":
            "LDSC",

        "input_type":
            "GWAS_summary",

        "uses_genotype":
            False,

        "uses_gwas":
            True,

        "grm_source":
            "none",

        "reference_source":
            "external_EUR",
    },


    "LDAK-Calculated-Heritability": {

        "family":
            "LDAK",

        "engine":
            "SumHer",

        "input_type":
            "GWAS+genotype",

        "uses_genotype":
            True,

        "uses_gwas":
            True,

        "grm_source":
            "none",

        "reference_source":
            "calculated_genotype_tagging",
    },


    "LDAK-Precomputed-Heritability": {

        "family":
            "LDAK",

        "engine":
            "SumHer",

        "input_type":
            "GWAS_summary",

        "uses_genotype":
            False,

        "uses_gwas":
            True,

        "grm_source":
            "none",

        "reference_source":
            "precomputed_tagging",
    },


    "LDSC-Genotype-Heritability": {

        "family":
            "LDSC",

        "engine":
            "LDSC",

        "input_type":
            "GWAS+genotype_reference",

        "uses_genotype":
            True,

        "uses_gwas":
            True,

        "grm_source":
            "none",

        "reference_source":
            "genotype_derived_LD",
    },
}


# ============================================================
# FACTOR DEFINITIONS
# ============================================================

FACTOR_DEFINITIONS = [

    # ========================================================
    # 1. DATA / PREPROCESSING
    # ========================================================

    {
        "category":
            "1. DATA / PREPROCESSING",

        "factor":
            "clumpprune",

        "label":
            "Clumping / pruning",

        "scope":
            "within_method",

        # Input SNP set changes by definition.
        "require_same_genotype":
            False,

        "require_same_gwas":
            True,

        "ignore_controls":
            {
                "genotype_fingerprint"
            },
    },


    {
        "category":
            "1. DATA / PREPROCESSING",

        "factor":
            "covariates",

        "label":
            "Covariate inclusion",

        "scope":
            "within_method",

        "require_same_genotype":
            True,

        "require_same_gwas":
            True,

        "ignore_controls":
            set(),
    },


    {
        "category":
            "1. DATA / PREPROCESSING",

        "factor":
            "pca",

        "label":
            "PCA inclusion",

        "scope":
            "within_method",

        "require_same_genotype":
            True,

        "require_same_gwas":
            True,

        "ignore_controls":
            set(),
    },


    {
        "category":
            "1. DATA / PREPROCESSING",

        "factor":
            "snp_set",

        "label":
            "SNP / variant-set choice",

        "scope":
            "within_method",

        # SNP set itself is the factor.
        "require_same_genotype":
            False,

        "require_same_gwas":
            True,

        "ignore_controls":
            {
                "genotype_fingerprint"
            },
    },


    # ========================================================
    # 2. ESTIMATOR / MODEL CONFIGURATION
    # ========================================================

    {
        "category":
            "2. ESTIMATOR / MODEL CONFIGURATION",

        "factor":
            "algorithm",

        "label":
            "Estimation algorithm",

        "scope":
            "within_method",

        "require_same_genotype":
            True,

        "require_same_gwas":
            True,

        "ignore_controls":
            set(),
    },


    {
        "category":
            "2. ESTIMATOR / MODEL CONFIGURATION",

        "factor":
            "grm_type",

        "label":
            "GRM construction",

        "scope":
            "within_method",

        "require_same_genotype":
            True,

        "require_same_gwas":
            True,

        "ignore_controls":
            set(),
    },


    {
        "category":
            "2. ESTIMATOR / MODEL CONFIGURATION",

        "factor":
            "tagging_model",

        "label":
            "LDAK tagging / model assumption",

        "scope":
            "within_method",

        "require_same_genotype":
            True,

        "require_same_gwas":
            True,

        "ignore_controls":
            set(),
    },


    # ========================================================
    # 3. INPUT / REFERENCE RESOURCE
    # ========================================================

    {
        "category":
            "3. INPUT / REFERENCE RESOURCE",

        "factor":
            "grm_source",

        "label":
            "GRM source (e.g. GEMMA vs DPR)",

        "scope":
            "cross_method",

        "require_same_genotype":
            True,

        "require_same_gwas":
            False,

        "ignore_controls":
            {
                "family"
            },
    },


    {
        "category":
            "3. INPUT / REFERENCE RESOURCE",

        "factor":
            "reference_source",

        "label":
            "Reference strategy",

        "scope":
            "cross_method",

        # Reference genotype itself may be what changes.
        "require_same_genotype":
            False,

        "require_same_gwas":
            True,

        "ignore_controls":
            {
                "input_type",
                "family"
            },
    },


    {
        "category":
            "3. INPUT / REFERENCE RESOURCE",

        "factor":
            "ld_reference",

        "label":
            "LDSC LD reference",

        "scope":
            "within_method",

        "require_same_genotype":
            False,

        "require_same_gwas":
            True,

        "ignore_controls":
            {
                "reference_source"
            },
    },


    {
        "category":
            "3. INPUT / REFERENCE RESOURCE",

        "factor":
            "ld_weight",

        "label":
            "LDSC regression weights",

        "scope":
            "within_method",

        "require_same_genotype":
            False,

        "require_same_gwas":
            True,

        "ignore_controls":
            {
                "reference_source"
            },
    },
]


# ============================================================
# STRUCTURALLY CONFOUNDED FACTORS
# ============================================================

STRUCTURALLY_CONFOUNDED = [

    "Input data type "
    "(individual genotype vs GWAS summary statistics)"
]


# ============================================================
# MATCHING VARIABLES
# ============================================================

CONTROL_FIELDS = [

    "family",
    "engine",
    "input_type",

    "clumpprune",
    "covariates",
    "pca",
    "snp_set",

    "algorithm",
    "grm_type",
    "tagging_model",

    "grm_source",
    "reference_source",

    "ld_reference",
    "ld_weight",

    "clump_p1_raw",
    "clump_r2_raw",
    "clump_kb_raw",

    "p_window_size_raw",
    "p_slide_size_raw",
    "p_LD_threshold_raw",

    "numberofpca_raw",

    "genotype_fingerprint",
    "gwas_fingerprint",
]


# ============================================================
# CONTRAST DIRECTION
#
# Delta h2 = Level B - Level A
# ============================================================

LEVEL_ORDER = {

    "clumpprune": [
        "no",
        "yes"
    ],

    "covariates": [
        "no",
        "yes"
    ],

    "pca": [
        "no",
        "yes"
    ],

    "snp_set": [
        "full",
        "all",
        "genotyped",
        "hapmap"
    ],

    "algorithm": [
        "HE",
        "REML",
        "GCTA_REML",
        "LDSC",
        "LDpred2_LDSC",
        "SumHer"
    ],

    "grm_type": [
        "centered",
        "standardized",
        "GCTA",
        "none"
    ],

    "grm_source": [
        "GEMMA",
        "DPR",
        "GCTA",
        "LDSC-derived",
        "none"
    ],
}


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_str(x):

    if pd.isna(x):
        return "NA"

    return str(x).strip()


def safe_float(x):

    try:

        value = float(x)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


def normalise_yes_no(x):

    x = safe_str(x).lower()

    if x in {
        "yes",
        "y",
        "true",
        "1"
    }:
        return "yes"

    if x in {
        "no",
        "n",
        "false",
        "0"
    }:
        return "no"

    return x


def ordered_levels(
    factor,
    levels
):

    levels = list(
        dict.fromkeys(
            levels
        )
    )

    preferred = LEVEL_ORDER.get(
        factor
    )

    if preferred is None:
        return sorted(
            levels
        )

    output = [

        value

        for value in preferred

        if value in levels
    ]

    output.extend(

        sorted(

            value

            for value in levels

            if value not in output
        )
    )

    return output


# ============================================================
# BENJAMINI-HOCHBERG FDR
# ============================================================

def bh_adjust(
    pvalues
):

    pvalues = np.asarray(
        pvalues,
        dtype=float
    )

    output = np.full(
        len(pvalues),
        np.nan
    )

    valid = np.where(
        np.isfinite(
            pvalues
        )
    )[0]

    if len(valid) == 0:
        return output

    p = pvalues[
        valid
    ]

    order = np.argsort(
        p
    )

    ranked = p[
        order
    ]

    m = len(
        ranked
    )

    q = (
        ranked
        * m
        / np.arange(
            1,
            m + 1
        )
    )

    q = np.minimum.accumulate(
        q[::-1]
    )[::-1]

    q = np.minimum(
        q,
        1.0
    )

    temp = np.empty(
        m
    )

    temp[
        order
    ] = q

    output[
        valid
    ] = temp

    return output


# ============================================================
# INPUT SNIFFING
# ============================================================

HASH_CACHE = {}

BIM_COUNT_CACHE = {}


def sha256_file(
    path
):

    """
    SHA-256 hash of actual input file.

    Cached so the same BIM/GWAS is only read once.
    """

    if path in HASH_CACHE:
        return HASH_CACHE[
            path
        ]

    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as handle:

        for chunk in iter(
            lambda:
                handle.read(
                    8 * 1024 * 1024
                ),
            b""
        ):

            h.update(
                chunk
            )

    value = h.hexdigest()

    HASH_CACHE[
        path
    ] = value

    return value


def count_lines(
    path
):

    """
    Count variants in the actual BIM.
    """

    if path in BIM_COUNT_CACHE:
        return BIM_COUNT_CACHE[
            path
        ]

    count = 0

    with open(
        path,
        "rb"
    ) as handle:

        for chunk in iter(
            lambda:
                handle.read(
                    8 * 1024 * 1024
                ),
            b""
        ):

            count += chunk.count(
                b"\n"
            )

    BIM_COUNT_CACHE[
        path
    ] = count

    return count


def choose_best_candidate(
    candidates
):

    existing = [

        os.path.abspath(
            x
        )

        for x in candidates

        if os.path.isfile(
            x
        )
    ]

    if not existing:
        return None

    existing = sorted(

        existing,

        key=lambda x: (
            x.count(
                os.sep
            ),
            len(x),
            x
        )
    )

    return existing[
        0
    ]


# ============================================================
# FIND ACTUAL GENOTYPE INPUT
# ============================================================

def find_bim(
    dataset,
    fold,
    clumpprune
):

    fold_dir = os.path.join(
        dataset,
        f"Fold_{fold}"
    )


    # ========================================================
    # CLUMPED / PRUNED INPUT
    # ========================================================

    if clumpprune == "yes":

        exact = [

            os.path.join(
                fold_dir,
                "train_data.QC.clumped.pruned.bim"
            ),

            os.path.join(
                fold_dir,
                "train_data.clumped.pruned.bim"
            ),
        ]

        found = choose_best_candidate(
            exact
        )

        if found:
            return found


        recursive = glob.glob(

            os.path.join(
                fold_dir,
                "**",
                "*clumped*pruned*.bim"
            ),

            recursive=True
        )

        found = choose_best_candidate(
            recursive
        )

        if found:
            return found


    # ========================================================
    # NON-CLUMPED INPUT
    # ========================================================

    else:

        exact = [

            os.path.join(
                fold_dir,
                "train_data.QC.bim"
            ),

            os.path.join(
                fold_dir,
                "train_data.bim"
            ),
        ]

        found = choose_best_candidate(
            exact
        )

        if found:
            return found


        recursive = [

            x

            for x in glob.glob(

                os.path.join(
                    fold_dir,
                    "**",
                    "train_data*.bim"
                ),

                recursive=True
            )

            if "clumped"
            not in os.path.basename(
                x
            ).lower()

            and "pruned"
            not in os.path.basename(
                x
            ).lower()
        ]

        found = choose_best_candidate(
            recursive
        )

        if found:
            return found


    return None


# ============================================================
# FIND COMMON RAW GWAS SOURCE
# ============================================================

def find_gwas_source(
    dataset
):

    candidates = [

        os.path.join(
            dataset,
            dataset + ".gz"
        ),

        os.path.join(
            dataset,
            dataset + ".txt.gz"
        ),

        os.path.join(
            dataset,
            dataset + ".tsv.gz"
        ),

        os.path.join(
            dataset,
            dataset + ".ldsc.txt"
        ),

        os.path.join(
            dataset,
            dataset + ".ldak"
        ),
    ]

    found = choose_best_candidate(
        candidates
    )

    if found:
        return found


    top_level = []

    for pattern in [
        "*.gz",
        "*.ldsc.txt",
        "*.ldak"
    ]:

        top_level.extend(

            glob.glob(
                os.path.join(
                    dataset,
                    pattern
                )
            )
        )

    return choose_best_candidate(
        top_level
    )


# ============================================================
# SNIFF INPUTS FOR A RESULT
# ============================================================

def sniff_inputs(
    dataset,
    fold,
    method,
    clumpprune
):

    meta = METHODS[
        method
    ]


    # ========================================================
    # GENOTYPE
    # ========================================================

    genotype_path = None

    genotype_fingerprint = (
        "NOT_USED"
    )

    genotype_input_snps = np.nan

    genotype_verified = True


    if meta[
        "uses_genotype"
    ]:

        genotype_path = find_bim(
            dataset,
            fold,
            clumpprune
        )

        if genotype_path is None:

            genotype_fingerprint = (
                "MISSING"
            )

            genotype_verified = (
                False
            )

        else:

            genotype_fingerprint = (
                sha256_file(
                    genotype_path
                )
            )

            genotype_input_snps = (
                count_lines(
                    genotype_path
                )
            )


    # ========================================================
    # GWAS
    # ========================================================

    gwas_path = None

    gwas_fingerprint = (
        "NOT_USED"
    )

    gwas_verified = True


    if meta[
        "uses_gwas"
    ]:

        gwas_path = (
            find_gwas_source(
                dataset
            )
        )

        if gwas_path is None:

            gwas_fingerprint = (
                "MISSING"
            )

            gwas_verified = False

        else:

            gwas_fingerprint = (
                sha256_file(
                    gwas_path
                )
            )


    return {

        "genotype_path":
            genotype_path
            if genotype_path
            else "NA",

        "genotype_fingerprint":
            genotype_fingerprint,

        "genotype_input_snps":
            genotype_input_snps,

        "genotype_verified":
            genotype_verified,

        "gwas_path":
            gwas_path
            if gwas_path
            else "NA",

        "gwas_fingerprint":
            gwas_fingerprint,

        "gwas_verified":
            gwas_verified,
    }


# ============================================================
# SIMULATION METADATA
# ============================================================

def simulation_metadata(
    sim_name
):

    try:

        number = int(
            sim_name.replace(
                "Sim",
                ""
            )
        )

    except Exception:

        return {

            "PhenotypeType":
                "unknown",

            "TargetN":
                np.nan,

            "TrueH2":
                np.nan,

            "CausalSNPs":
                np.nan,

            "Replicate":
                np.nan,

            "Scenario":
                "unknown",
        }


    # ========================================================
    # CONTINUOUS
    # ========================================================

    if 1 <= number <= 100:

        phenotype_type = (
            "Continuous"
        )

        if number <= 50:

            target_n = 750

            local = number

        else:

            target_n = 5000

            local = (
                number - 50
            )


    # ========================================================
    # BINARY
    # ========================================================

    elif 101 <= number <= 200:

        phenotype_type = (
            "Binary"
        )

        if number <= 150:

            target_n = 750

            local = (
                number - 100
            )

        else:

            target_n = 5000

            local = (
                number - 150
            )


    else:

        return {

            "PhenotypeType":
                "unknown",

            "TargetN":
                np.nan,

            "TrueH2":
                np.nan,

            "CausalSNPs":
                np.nan,

            "Replicate":
                np.nan,

            "Scenario":
                "unknown",
        }


    h2_values = [
        0.05,
        0.10,
        0.30,
        0.50,
        0.70
    ]


    true_h2 = h2_values[
        (local - 1) // 10
    ]


    inside = (
        local - 1
    ) % 10


    if inside < 5:

        causal = 100

        replicate = (
            inside + 1
        )

    else:

        causal = 5000

        replicate = (
            inside - 4
        )


    scenario = (

        f"{phenotype_type}"
        f" | N={target_n}"
        f" | h2={true_h2:.2f}"
        f" | causal={causal}"
    )


    return {

        "PhenotypeType":
            phenotype_type,

        "TargetN":
            target_n,

        "TrueH2":
            true_h2,

        "CausalSNPs":
            causal,

        "Replicate":
            replicate,

        "Scenario":
            scenario,
    }


# ============================================================
# CONFIGURATION PARSER
# ============================================================

def parse_configuration(
    method,
    row
):

    info = METHODS[
        method
    ]


    model = safe_str(
        row.get(
            "h2model",
            "NA"
        )
    )


    model_lower = model.lower()


    # ========================================================
    # CLUMPING
    # ========================================================

    clumpprune = normalise_yes_no(

        row.get(
            "clumpprune",
            "NA"
        )
    )


    # ========================================================
    # COVARIATES / PCA
    # ========================================================

    if (
        "genotype_covariate_pca"
        in model_lower
    ):

        covariates = "yes"

        pca = "yes"


    elif (
        "genotype_covariate"
        in model_lower
    ):

        covariates = "yes"

        pca = "no"


    else:

        covariates = "no"

        pca = "no"


    # ========================================================
    # ALGORITHM
    # ========================================================

    if (
        "he regression"
        in model_lower

        or "haseman"
        in model_lower
    ):

        algorithm = "HE"


    elif (
        "reml"
        in model_lower

        or "ai algorithm"
        in model_lower
    ):

        algorithm = "REML"


    elif method == "GCTA-Heritability":

        algorithm = "GCTA_REML"


    elif method in {

        "LDSC-Heritability",
        "LDSC-Genotype-Heritability"

    }:

        algorithm = "LDSC"


    elif method == "LDpred-2-Heritability":

        algorithm = "LDpred2_LDSC"


    elif method in {

        "LDAK-Calculated-Heritability",
        "LDAK-Precomputed-Heritability"

    }:

        algorithm = "SumHer"


    else:

        algorithm = "NA"


    # ========================================================
    # GRM TYPE
    # ========================================================

    if "centered" in model_lower:

        grm_type = "centered"


    elif (
        "standardized"
        in model_lower

        or "standardised"
        in model_lower
    ):

        grm_type = "standardized"


    elif method == "GCTA-Heritability":

        grm_type = "GCTA"


    elif method in {

        "Gemma2-Heritability",
        "Gemma-LDSC-Heritability"

    }:

        grm_type = "centered"


    else:

        grm_type = "none"


    # ========================================================
    # SNP SET
    # ========================================================

    if "hapmap" in model_lower:

        snp_set = "hapmap"


    elif "genotyped" in model_lower:

        snp_set = "genotyped"


    elif (
        method
        == "LDpred-2-Heritability"

        and "full"
        in model_lower
    ):

        snp_set = "full"


    else:

        snp_set = "all"


    # ========================================================
    # LDAK TAGGING MODEL
    # ========================================================

    tagging_model = "NA"


    if method == "LDAK-Calculated-Heritability":


        if "ldak_human" in model_lower:

            tagging_model = (
                "Human"
            )


        elif "ldak_gcta" in model_lower:

            tagging_model = (
                "GCTA"
            )


        elif "bld-ldak" in model_lower:

            tagging_model = (
                "BLD-LDAK"
            )


        elif "alpha_" in model_lower:

            parts = model.split(
                "_"
            )

            lower_parts = [

                x.lower()

                for x in parts
            ]

            try:

                index = lower_parts.index(
                    "alpha"
                )

                tagging_model = (

                    "Alpha_"

                    + parts[
                        index + 1
                    ]
                )

            except Exception:

                tagging_model = model


    elif (
        method
        == "LDAK-Precomputed-Heritability"
    ):

        temp = model_lower

        temp = temp.replace(
            "ldak_sim_precomputed_",
            ""
        )

        temp = temp.replace(
            "ldak_",
            ""
        )

        temp = temp.replace(
            ".tagging",
            ""
        )

        for token in [

            ".hapmap.gbr",
            ".genotyped.gbr",
            "hapmap.gbr",
            "genotyped.gbr"

        ]:

            temp = temp.replace(
                token,
                ""
            )


        tagging_model = (

            temp.strip(
                "._"
            )

            or "precomputed"
        )


    # ========================================================
    # REFERENCE SOURCE
    # ========================================================

    reference_source = info[
        "reference_source"
    ]


    if (
        "sim_reference5000"
        in model_lower
    ):

        reference_source = (
            "simulation_fixed_N5000"
        )


    if (
        "simulationreference"
        in model_lower
    ):

        reference_source = (
            "simulation_genotype_derived"
        )


    if (
        "sim_precomputed"
        in model_lower
    ):

        reference_source = (
            "simulation_precomputed_N5000"
        )


    # ========================================================
    # LDSC REFERENCE / WEIGHTS
    # ========================================================

    ld_reference = "NA"

    ld_weight = "NA"


    if method == "LDSC-Heritability":


        if (
            "sim_reference5000"
            in model_lower
        ):

            ld_reference = (
                "SIM_N5000"
            )

            ld_weight = (
                "SIM_N5000"
            )


        else:

            body = (

                model[5:]

                if model.startswith(
                    "LDSC_"
                )

                else model
            )


            possible = [

                (
                    "eur_ref_ld_chr",
                    "eur_ref_ld_chr"
                ),

                (
                    "eur_ref_ld_chr",
                    "eur_w_ld_chr"
                ),

                (
                    "eur_w_ld_chr",
                    "eur_ref_ld_chr"
                ),

                (
                    "eur_w_ld_chr",
                    "eur_w_ld_chr"
                ),
            ]


            for (
                reference,
                weight
            ) in possible:

                if body == (

                    reference
                    + "_"
                    + weight
                ):

                    ld_reference = (
                        reference
                    )

                    ld_weight = (
                        weight
                    )

                    break


    elif (
        method
        == "LDSC-Genotype-Heritability"
    ):

        ld_reference = (
            "genotype_derived"
        )

        ld_weight = (
            "genotype_derived"
        )


    return {

        "family":
            info["family"],

        "engine":
            info["engine"],

        "input_type":
            info["input_type"],

        "uses_genotype":
            info["uses_genotype"],

        "uses_gwas":
            info["uses_gwas"],

        "clumpprune":
            clumpprune,

        "covariates":
            covariates,

        "pca":
            pca,

        "snp_set":
            snp_set,

        "algorithm":
            algorithm,

        "grm_type":
            grm_type,

        "tagging_model":
            tagging_model,

        "grm_source":
            info["grm_source"],

        "reference_source":
            reference_source,

        "ld_reference":
            ld_reference,

        "ld_weight":
            ld_weight,

        "clump_p1_raw":
            safe_str(
                row.get(
                    "clump_p1",
                    "NA"
                )
            ),

        "clump_r2_raw":
            safe_str(
                row.get(
                    "clump_r2",
                    "NA"
                )
            ),

        "clump_kb_raw":
            safe_str(
                row.get(
                    "clump_kb",
                    "NA"
                )
            ),

        "p_window_size_raw":
            safe_str(
                row.get(
                    "p_window_size",
                    "NA"
                )
            ),

        "p_slide_size_raw":
            safe_str(
                row.get(
                    "p_slide_size",
                    "NA"
                )
            ),

        "p_LD_threshold_raw":
            safe_str(
                row.get(
                    "p_LD_threshold",
                    "NA"
                )
            ),

        "numberofpca_raw":
            safe_str(
                row.get(
                    "numberofpca",
                    "NA"
                )
            ),

        "raw_h2model":
            model,
    }


# ============================================================
# FIND DATASETS
# ============================================================

if MODE == "real":

    DATASETS = [

        x

        for x in REAL_PHENOTYPES

        if os.path.isdir(
            x
        )
    ]


else:

    DATASETS = sorted([

        os.path.basename(
            x
        )

        for x in glob.glob(
            "Sim[0-9][0-9][0-9]"
        )

        if os.path.isdir(
            x
        )
    ])


# ============================================================
# LOAD RESULTS AND SNIFF INPUTS
# ============================================================

print()

print(
    "=" * 140
)

print(
    "METHOD 19 — INPUT-SNIFFED ONE-FACTOR MATCHED DECOMPOSITION"
)

print(
    "=" * 140
)

print(
    "MODE:",
    MODE.upper()
)

print(
    "Datasets found:",
    len(
        DATASETS
    )
)

print(
    "Nothing will be saved."
)

print()


rows = []

files_found = 0

files_missing = 0

INPUT_SNIFF_CACHE = {}


for dataset in DATASETS:

    for fold in range(
        5
    ):

        for method in METHODS:


            path = os.path.join(

                dataset,

                f"Fold_{fold}",

                method,

                "Results.csv"
            )


            if not os.path.exists(
                path
            ):

                files_missing += 1

                continue


            files_found += 1


            try:

                result_df = pd.read_csv(
                    path
                )

            except Exception:

                continue


            if (
                result_df.empty

                or "h2"
                not in result_df.columns
            ):

                continue


            for _, row in result_df.iterrows():


                h2 = safe_float(

                    row.get(
                        "h2",
                        np.nan
                    )
                )


                if not np.isfinite(
                    h2
                ):

                    continue


                metadata = parse_configuration(
                    method,
                    row
                )


                sniff_key = (

                    dataset,
                    fold,
                    method,
                    metadata[
                        "clumpprune"
                    ]
                )


                if (
                    sniff_key
                    not in INPUT_SNIFF_CACHE
                ):

                    INPUT_SNIFF_CACHE[
                        sniff_key
                    ] = sniff_inputs(

                        dataset,
                        fold,
                        method,
                        metadata[
                            "clumpprune"
                        ]
                    )


                sniff = INPUT_SNIFF_CACHE[
                    sniff_key
                ]


                record = {

                    "Dataset":
                        dataset,

                    "Fold":
                        fold,

                    "Method":
                        method,

                    "h2":
                        h2,

                    # IMPORTANT:
                    # this is NOT treated as the input SNP count.
                    "effective_variants":
                        safe_float(

                            row.get(
                                "numberofvariants",
                                np.nan
                            )
                        ),
                }


                record.update(
                    metadata
                )

                record.update(
                    sniff
                )


                if MODE == "real":

                    label = (

                        PHENOTYPE_LABELS.get(
                            dataset,
                            dataset
                        )
                    )

                    record[
                        "DatasetLabel"
                    ] = label

                    record[
                        "Scenario"
                    ] = label


                else:

                    record.update(

                        simulation_metadata(
                            dataset
                        )
                    )

                    record[
                        "DatasetLabel"
                    ] = dataset


                rows.append(
                    record
                )


data = pd.DataFrame(
    rows
)


if data.empty:

    raise SystemExit(
        "ERROR: no valid heritability rows found."
    )


print(
    "Results.csv files found:  ",
    files_found
)

print(
    "Results.csv files missing:",
    files_missing
)

print(
    "Valid h2 rows:            ",
    len(
        data
    )
)

print(
    "Methods represented:      ",
    data[
        "Method"
    ].nunique()
)


# ============================================================
# INPUT-SNIFF AUDIT
# ============================================================

print()

print(
    "=" * 140
)

print(
    "INPUT-SNIFF AUDIT"
)

print(
    "=" * 140
)


genotype_rows = data[

    data[
        "uses_genotype"
    ] == True

].copy()


gwas_rows = data[

    data[
        "uses_gwas"
    ] == True

].copy()


if len(
    genotype_rows
):

    print(

        "Genotype-using rows:",

        len(
            genotype_rows
        ),

        "| verified:",

        int(
            genotype_rows[
                "genotype_verified"
            ].sum()
        ),

        "| unverified:",

        int(
            (
                ~genotype_rows[
                    "genotype_verified"
                ]
            ).sum()
        )
    )


    print(

        "Distinct verified genotype input fingerprints:",

        genotype_rows.loc[

            genotype_rows[
                "genotype_verified"
            ],

            "genotype_fingerprint"

        ].nunique()
    )


if len(
    gwas_rows
):

    print(

        "GWAS-using rows:    ",

        len(
            gwas_rows
        ),

        "| verified:",

        int(
            gwas_rows[
                "gwas_verified"
            ].sum()
        ),

        "| unverified:",

        int(
            (
                ~gwas_rows[
                    "gwas_verified"
                ]
            ).sum()
        )
    )


    print(

        "Distinct verified GWAS source fingerprints:",

        gwas_rows.loc[

            gwas_rows[
                "gwas_verified"
            ],

            "gwas_fingerprint"

        ].nunique()
    )


# ============================================================
# EXACT MATCH BUILDER
# ============================================================

pair_rows = []

factor_audit = []


def factor_match_columns(
    definition
):

    factor = definition[
        "factor"
    ]

    ignore = set(

        definition.get(
            "ignore_controls",
            set()
        )
    )


    columns = [

        "Dataset",
        "Fold"
    ]


    if (
        definition[
            "scope"
        ]
        == "within_method"
    ):

        columns.append(
            "Method"
        )


    else:

        # Cross-method comparisons must at minimum
        # share the underlying estimator engine.
        columns.append(
            "engine"
        )


    for column in CONTROL_FIELDS:


        if column == factor:
            continue


        if column in ignore:
            continue


        if column in {

            "genotype_fingerprint",
            "gwas_fingerprint"

        }:

            continue


        columns.append(
            column
        )


    if definition[
        "require_same_genotype"
    ]:

        columns.append(
            "genotype_fingerprint"
        )


    if definition[
        "require_same_gwas"
    ]:

        columns.append(
            "gwas_fingerprint"
        )


    return list(
        dict.fromkeys(
            columns
        )
    )


def build_pairs(
    definition
):

    factor = definition[
        "factor"
    ]

    label = definition[
        "label"
    ]

    category = definition[
        "category"
    ]


    temp = data.copy()


    temp = temp[

        temp[
            factor
        ].notna()

    ].copy()


    temp = temp[

        ~temp[
            factor
        ]
        .astype(
            str
        )
        .str.lower()
        .isin(
            {
                "na",
                "nan",
                ""
            }
        )

    ].copy()


    # ========================================================
    # STRICT GENOTYPE INPUT VERIFICATION
    # ========================================================

    if definition[
        "require_same_genotype"
    ]:

        temp = temp[

            (
                ~temp[
                    "uses_genotype"
                ]
            )

            |

            (
                temp[
                    "genotype_verified"
                ]
            )

        ].copy()


    # ========================================================
    # STRICT GWAS INPUT VERIFICATION
    # ========================================================

    if definition[
        "require_same_gwas"
    ]:

        temp = temp[

            (
                ~temp[
                    "uses_gwas"
                ]
            )

            |

            (
                temp[
                    "gwas_verified"
                ]
            )

        ].copy()


    match_columns = (
        factor_match_columns(
            definition
        )
    )


    for column in (
        match_columns
        + [factor]
    ):

        temp[
            column
        ] = (

            temp[
                column
            ]

            .fillna(
                "NA"
            )

            .astype(
                str
            )
        )


    candidate_groups = 0

    matched_groups = 0


    grouped = temp.groupby(

        match_columns,

        dropna=False,

        sort=False
    )


    for _, group in grouped:


        levels = ordered_levels(

            factor,

            group[
                factor
            ]
            .unique()
            .tolist()
        )


        if len(
            levels
        ) < 2:

            continue


        candidate_groups += 1


        for (
            level_a,
            level_b
        ) in itertools.combinations(

            levels,
            2
        ):


            group_a = group[

                group[
                    factor
                ] == level_a

            ]


            group_b = group[

                group[
                    factor
                ] == level_b

            ]


            if (
                group_a.empty
                or group_b.empty
            ):

                continue


            matched_groups += 1


            h2_a = group_a[
                "h2"
            ].mean()


            h2_b = group_b[
                "h2"
            ].mean()


            effective_a = (
                group_a[
                    "effective_variants"
                ].mean()
            )


            effective_b = (
                group_b[
                    "effective_variants"
                ].mean()
            )


            if (

                np.isfinite(
                    effective_a
                )

                and np.isfinite(
                    effective_b
                )
            ):

                effective_delta = (
                    effective_b
                    - effective_a
                )

            else:

                effective_delta = np.nan


            input_snps_a = (
                group_a[
                    "genotype_input_snps"
                ].mean()
            )


            input_snps_b = (
                group_b[
                    "genotype_input_snps"
                ].mean()
            )


            method_a = ",".join(

                sorted(

                    group_a[
                        "Method"
                    ]
                    .unique()
                )
            )


            method_b = ",".join(

                sorted(

                    group_b[
                        "Method"
                    ]
                    .unique()
                )
            )


            stratum_bits = []


            for column in match_columns:


                if column in {

                    "Dataset",
                    "Fold"

                }:

                    continue


                stratum_bits.append(

                    f"{column}="
                    f"{safe_str(group.iloc[0][column])}"
                )


            stratum = " | ".join(
                stratum_bits
            )


            pair_rows.append({

                "Category":
                    category,

                "Factor":
                    factor,

                "FactorLabel":
                    label,

                "Dataset":
                    group.iloc[0][
                        "Dataset"
                    ],

                "DatasetLabel":
                    group.iloc[0][
                        "DatasetLabel"
                    ],

                "Scenario":
                    group.iloc[0][
                        "Scenario"
                    ],

                "Fold":
                    int(
                        group.iloc[0][
                            "Fold"
                        ]
                    ),

                "Method_A":
                    method_a,

                "Method_B":
                    method_b,

                "Level_A":
                    level_a,

                "Level_B":
                    level_b,

                "Contrast":
                    f"{level_b} - {level_a}",

                "h2_A":
                    h2_a,

                "h2_B":
                    h2_b,

                "Delta":
                    h2_b - h2_a,

                "AbsDelta":
                    abs(
                        h2_b - h2_a
                    ),

                "EffectiveVariants_A":
                    effective_a,

                "EffectiveVariants_B":
                    effective_b,

                "EffectiveVariantDelta":
                    effective_delta,

                "InputGenotypeSNPs_A":
                    input_snps_a,

                "InputGenotypeSNPs_B":
                    input_snps_b,

                "Stratum":
                    stratum,
            })


    factor_audit.append({

        "Category":
            category,

        "FactorLabel":
            label,

        "CandidateGroups":
            candidate_groups,

        "MatchedComparisons":
            matched_groups,
    })


# ============================================================
# BUILD ALL CONTRASTS
# ============================================================

for definition in FACTOR_DEFINITIONS:

    build_pairs(
        definition
    )


pairs_fold = pd.DataFrame(
    pair_rows
)


# ============================================================
# MATCH CONSTRUCTION AUDIT
# ============================================================

print()

print(
    "=" * 140
)

print(
    "MATCH-CONSTRUCTION AUDIT"
)

print(
    "=" * 140
)


print(

    pd.DataFrame(
        factor_audit
    ).to_string(
        index=False
    )
)


print()

print(
    "STRUCTURALLY CONFOUNDED / "
    "NOT CLAIMED AS ONE-FACTOR EFFECT:"
)


for item in STRUCTURALLY_CONFOUNDED:

    print(
        " -",
        item
    )


if pairs_fold.empty:

    print()

    print(
        "NO valid exact one-factor comparisons were found."
    )

    print(
        "DONE — nothing was saved."
    )

    sys.exit(
        0
    )


# ============================================================
# AVERAGE OVER OVERLAPPING FOLDS
# ============================================================

pair_units = (

    pairs_fold

    .groupby(

        [
            "Category",
            "Factor",
            "FactorLabel",

            "Dataset",
            "DatasetLabel",
            "Scenario",

            "Method_A",
            "Method_B",

            "Contrast",
            "Level_A",
            "Level_B",

            "Stratum",
        ],

        as_index=False,

        dropna=False,
    )

    .agg(

        Delta=(
            "Delta",
            "mean"
        ),

        AbsDelta=(
            "AbsDelta",
            "mean"
        ),

        CommonFolds=(
            "Fold",
            "nunique"
        ),

        h2_A=(
            "h2_A",
            "mean"
        ),

        h2_B=(
            "h2_B",
            "mean"
        ),

        EffectiveVariants_A=(
            "EffectiveVariants_A",
            "mean"
        ),

        EffectiveVariants_B=(
            "EffectiveVariants_B",
            "mean"
        ),

        EffectiveVariantDelta=(
            "EffectiveVariantDelta",
            "mean"
        ),

        InputGenotypeSNPs_A=(
            "InputGenotypeSNPs_A",
            "mean"
        ),

        InputGenotypeSNPs_B=(
            "InputGenotypeSNPs_B",
            "mean"
        ),
    )
)


# ============================================================
# MATCHING SUMMARY
# ============================================================

print()

print(
    "=" * 140
)

print(
    "MATCHING SUMMARY"
)

print(
    "=" * 140
)


print(

    "Fold-level exact matches:                         ",

    len(
        pairs_fold
    )
)


print(

    "Matched configuration pairs after fold averaging:",

    len(
        pair_units
    )
)


print(

    "Datasets represented:                             ",

    pair_units[
        "Dataset"
    ].nunique()
)


# ============================================================
# POST-MATCH INPUT CONTROL CHECK
# ============================================================

print()

print(
    "=" * 140
)

print(
    "POST-MATCH INPUT CONTROL CHECK"
)

print(
    "=" * 140
)


for definition in FACTOR_DEFINITIONS:


    subset = pair_units[

        pair_units[
            "Factor"
        ]
        == definition[
            "factor"
        ]

    ]


    if subset.empty:

        print(

            f"{definition['label']:<42}"
            " : no valid comparisons"
        )

        continue


    if definition[
        "require_same_genotype"
    ]:


        known = (

            subset[
                "InputGenotypeSNPs_A"
            ].notna()

            &

            subset[
                "InputGenotypeSNPs_B"
            ].notna()
        )


        if known.any():


            same = np.isclose(

                subset.loc[
                    known,
                    "InputGenotypeSNPs_A"
                ],

                subset.loc[
                    known,
                    "InputGenotypeSNPs_B"
                ],

                rtol=0,

                atol=0
            )


            print(

                f"{definition['label']:<42}"
                " : same actual input SNP count "
                f"{int(same.sum())}/"
                f"{int(known.sum())}"
            )


        else:

            print(

                f"{definition['label']:<42}"
                " : genotype fingerprint controlled; "
                "count N/A"
            )


    else:

        print(

            f"{definition['label']:<42}"
            " : input SNP set allowed to differ by design"
        )


# ============================================================
# EFFECTIVE / REPORTED VARIANT COUNTS
# ============================================================

print()

print(
    "=" * 140
)

print(
    "EFFECTIVE / REPORTED VARIANT-COUNT CHANGES"
)

print(
    "=" * 140
)


print(

    "These are output/effective counts from Results.csv; "
    "they do NOT define the input SNP set."
)


for definition in FACTOR_DEFINITIONS:


    subset = pair_units[

        pair_units[
            "Factor"
        ]
        == definition[
            "factor"
        ]

    ]


    if subset.empty:
        continue


    values = subset[
        "EffectiveVariantDelta"
    ].dropna()


    if len(
        values
    ) == 0:

        print(

            f"{definition['label']:<42}"
            " : unavailable"
        )

        continue


    different = int(

        (
            np.abs(
                values
            ) > 0
        ).sum()
    )


    print(

        f"{definition['label']:<42}"
        f" : {different}/{len(values)} pairs differ; "
        f"median |Delta variants|="
        f"{np.median(np.abs(values)):.1f}; "
        f"max={np.max(np.abs(values)):.1f}"
    )


# ============================================================
# CONTRAST-LEVEL INFERENCE
# ============================================================

summary_rows = []


grouped_contrasts = pair_units.groupby(

    [
        "Category",
        "Factor",
        "FactorLabel",
        "Contrast",
        "Level_A",
        "Level_B"
    ],

    dropna=False
)


for (
    keys,
    subset
) in grouped_contrasts:


    (
        category,
        factor,
        label,
        contrast,
        level_a,
        level_b

    ) = keys


    # ========================================================
    # ONE EFFECT PER DATASET / PHENOTYPE
    # ========================================================

    dataset_effect = (

        subset

        .groupby(
            "Dataset"
        )[
            "Delta"
        ]

        .mean()

        .dropna()
    )


    dataset_absolute = (

        subset

        .groupby(
            "Dataset"
        )[
            "AbsDelta"
        ]

        .mean()

        .dropna()
    )


    # ========================================================
    # WILCOXON
    # ========================================================

    p_value = np.nan


    if (

        len(
            dataset_effect
        ) >= 2

        and np.any(

            np.abs(
                dataset_effect.values
            ) > 0
        )
    ):

        try:

            wilcoxon_result = stats.wilcoxon(
                dataset_effect.values,
                alternative="two-sided",
                zero_method="wilcox"
            )

            p_value = float(wilcoxon_result.pvalue)

        except Exception:

            p_value = np.nan


    if len(
        dataset_effect
    ):


        percent_positive = (

            100.0
            * np.mean(

                dataset_effect.values
                > 0
            )
        )


        percent_negative = (

            100.0
            * np.mean(

                dataset_effect.values
                < 0
            )
        )


    else:

        percent_positive = np.nan

        percent_negative = np.nan


    # ========================================================
    # EXACT-ZERO AUDIT FLAG
    # ========================================================

    all_zero = bool(

        len(
            dataset_effect
        ) > 0

        and np.all(

            np.isclose(

                dataset_effect.values,

                0.0,

                rtol=0,

                atol=1e-15
            )
        )
    )


    effective_variant_dataset = (

        subset

        .groupby(
            "Dataset"
        )[
            "EffectiveVariantDelta"
        ]

        .mean()

        .dropna()
    )


    summary_rows.append({

        "Category":
            category,

        "Factor":
            factor,

        "FactorLabel":
            label,

        "Contrast":
            contrast,

        "Level_A":
            level_a,

        "Level_B":
            level_b,

        "MatchedPairs":
            len(
                subset
            ),

        "Datasets":
            subset[
                "Dataset"
            ].nunique(),

        "MeanDelta":
            dataset_effect.mean(),

        "MedianDelta":
            dataset_effect.median(),

        "MeanAbsDelta":
            dataset_absolute.mean(),

        "MedianAbsDelta":
            dataset_absolute.median(),

        "Q25Delta":
            dataset_effect.quantile(
                0.25
            ),

        "Q75Delta":
            dataset_effect.quantile(
                0.75
            ),

        "MinDelta":
            dataset_effect.min(),

        "MaxDelta":
            dataset_effect.max(),

        "PctPositive":
            percent_positive,

        "PctNegative":
            percent_negative,

        "WilcoxonP":
            p_value,

        "AllZero":
            all_zero,

        "MedianAbsEffectiveVariantDelta":

            np.median(

                np.abs(
                    effective_variant_dataset.values
                )
            )

            if len(
                effective_variant_dataset
            )

            else np.nan,
    })


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# FDR
# ============================================================

summary[
    "FDR_Q"
] = bh_adjust(

    summary[
        "WilcoxonP"
    ].values
)


summary[
    "FDR_Significant"
] = np.where(

    pd.to_numeric(

        summary[
            "FDR_Q"
        ],

        errors="coerce"

    ) < 0.05,

    "YES",

    "NO"
)


summary.loc[

    summary[
        "FDR_Q"
    ].isna(),

    "FDR_Significant"

] = "NA"


# ============================================================
# MULTIPLE TESTING POLICY
# ============================================================

print()

print(
    "=" * 140
)

print(
    "MULTIPLE-TESTING POLICY"
)

print(
    "=" * 140
)


print(

    "Testing family       : "
    "all valid Method19 one-factor contrasts"
)


print(

    "Raw test             : "
    "two-sided Wilcoxon signed-rank"
)


print(

    "Independent unit     :",

    (
        "phenotype"

        if MODE == "real"

        else "simulation dataset"
    )
)


print(

    "Correction           : "
    "Benjamini-Hochberg FDR"
)


print(

    "Primary significance : "
    "q < 0.05"
)


print(

    "Per-phenotype/scenario tables "
    "are descriptive only."
)


print(

    "Number of valid inferential tests:",

    int(

        summary[
            "WilcoxonP"
        ]
        .notna()
        .sum()
    )
)


# ============================================================
# CATEGORY ORDER
# ============================================================

category_order = []


for definition in FACTOR_DEFINITIONS:

    if (

        definition[
            "category"
        ]
        not in category_order
    ):

        category_order.append(

            definition[
                "category"
            ]
        )


# ============================================================
# PRINT EXACT RESULTS
# ============================================================

print()

print(
    "=" * 140
)

print(
    "EXACT ONE-FACTOR RESULTS"
)

print(
    "Delta h2 = Level B - Level A"
)

print(
    "=" * 140
)


for category in category_order:


    print()

    print(
        "#" * 140
    )

    print(
        category
    )

    print(
        "#" * 140
    )


    definitions = [

        x

        for x in FACTOR_DEFINITIONS

        if x[
            "category"
        ] == category
    ]


    for definition in definitions:


        factor = definition[
            "factor"
        ]


        factor_summary = summary[

            summary[
                "Factor"
            ] == factor

        ].copy()


        print()

        print(
            "FACTOR:",
            definition[
                "label"
            ]
        )

        print(
            "-" * 140
        )


        if factor_summary.empty:

            print(

                "No valid exact one-factor "
                "matched comparison."
            )

            print(

                "This factor could not be isolated "
                "with verified comparable inputs "
                "in the current configuration space."
            )

            continue


        factor_summary = (
            factor_summary.sort_values(

                [
                    "MedianAbsDelta",
                    "MatchedPairs"
                ],

                ascending=[
                    False,
                    False
                ]
            )
        )


        for _, result in factor_summary.iterrows():


            print()

            print(
                "CONTRAST:",
                result[
                    "Contrast"
                ]
            )


            print(

                f"  Matched configuration pairs      : "
                f"{int(result['MatchedPairs'])}"
            )


            print(

                f"  Datasets represented             : "
                f"{int(result['Datasets'])}"
            )


            print(

                f"  Mean Delta h2                    : "
                f"{result['MeanDelta']:.6f}"
            )


            print(

                f"  Median Delta h2                  : "
                f"{result['MedianDelta']:.6f}"
            )


            print(

                f"  Mean |Delta h2|                  : "
                f"{result['MeanAbsDelta']:.6f}"
            )


            print(

                f"  Median |Delta h2|                : "
                f"{result['MedianAbsDelta']:.6f}"
            )


            print(

                f"  IQR Delta h2                     : "
                f"[{result['Q25Delta']:.6f}, "
                f"{result['Q75Delta']:.6f}]"
            )


            print(

                f"  Range Delta h2                   : "
                f"[{result['MinDelta']:.6f}, "
                f"{result['MaxDelta']:.6f}]"
            )


            print(

                f"  Positive dataset effects         : "
                f"{result['PctPositive']:.1f}%"
            )


            print(

                f"  Negative dataset effects         : "
                f"{result['PctNegative']:.1f}%"
            )


            if pd.notna(

                result[
                    "MedianAbsEffectiveVariantDelta"
                ]
            ):

                print(

                    f"  Median |Delta effective variants|: "
                    f"{result['MedianAbsEffectiveVariantDelta']:.1f}"
                )


            if pd.notna(
                result[
                    "WilcoxonP"
                ]
            ):

                print(

                    f"  Raw Wilcoxon P                   : "
                    f"{result['WilcoxonP']:.6g}"
                )


                print(

                    f"  BH-FDR Q                         : "
                    f"{result['FDR_Q']:.6g}"
                )


                print(

                    f"  FDR significant (q<0.05)        : "
                    f"{result['FDR_Significant']}"
                )


            else:

                print(
                    "  Raw Wilcoxon P                   : NA"
                )

                print(
                    "  BH-FDR Q                         : NA"
                )

                print(
                    "  FDR significant (q<0.05)        : NA"
                )


            # =================================================
            # SUSPICIOUS ZERO
            # =================================================

            if result[
                "AllZero"
            ]:

                print(

                    "  *** UPSTREAM AUDIT FLAG           : "
                    "ALL matched dataset effects are exactly zero. "
                    "Verify upstream implementation before "
                    "biological interpretation."
                )


            contrast_data = pair_units[

                (
                    pair_units[
                        "Factor"
                    ] == factor
                )

                &

                (
                    pair_units[
                        "Contrast"
                    ]
                    == result[
                        "Contrast"
                    ]
                )

            ].copy()


            # =================================================
            # METHOD BREAKDOWN
            # =================================================

            method_detail = (

                contrast_data

                .groupby(

                    [
                        "Method_A",
                        "Method_B"
                    ],

                    as_index=False
                )

                .agg(

                    Datasets=(
                        "Dataset",
                        "nunique"
                    ),

                    N_pairs=(
                        "Delta",
                        "size"
                    ),

                    MeanDelta=(
                        "Delta",
                        "mean"
                    ),

                    MedianDelta=(
                        "Delta",
                        "median"
                    ),

                    MeanAbsDelta=(
                        "AbsDelta",
                        "mean"
                    ),
                )
            )


            print()

            print(
                "  BY METHOD / METHOD PAIR"
            )


            print(

                f"  {'Method A':<34}"
                f"{'Method B':<34}"
                f"{'Data':>7}"
                f"{'Pairs':>7}"
                f"{'Mean D':>12}"
                f"{'Median D':>12}"
                f"{'Mean |D|':>12}"
            )


            for _, row in method_detail.iterrows():

                print(

                    f"  {str(row['Method_A']):<34}"
                    f"{str(row['Method_B']):<34}"
                    f"{int(row['Datasets']):>7}"
                    f"{int(row['N_pairs']):>7}"
                    f"{row['MeanDelta']:>12.5f}"
                    f"{row['MedianDelta']:>12.5f}"
                    f"{row['MeanAbsDelta']:>12.5f}"
                )


            # =================================================
            # REAL: PER PHENOTYPE
            # =================================================

            if MODE == "real":


                detail = (

                    contrast_data

                    .groupby(

                        "DatasetLabel",

                        as_index=False
                    )

                    .agg(

                        N_pairs=(
                            "Delta",
                            "size"
                        ),

                        MeanDelta=(
                            "Delta",
                            "mean"
                        ),

                        MedianDelta=(
                            "Delta",
                            "median"
                        ),

                        MeanAbsDelta=(
                            "AbsDelta",
                            "mean"
                        ),

                        MedianAbsDelta=(
                            "AbsDelta",
                            "median"
                        ),

                        MeanEffectiveVariantDelta=(
                            "EffectiveVariantDelta",
                            "mean"
                        ),
                    )
                )


                print()

                print(
                    "  PER PHENOTYPE — DESCRIPTIVE"
                )


                print(

                    f"  {'Phenotype':<36}"
                    f"{'Pairs':>7}"
                    f"{'Mean D':>12}"
                    f"{'Median D':>12}"
                    f"{'Mean |D|':>12}"
                    f"{'Med |D|':>12}"
                    f"{'Mean Dvars':>13}"
                )


                for _, row in detail.iterrows():


                    delta_variants = (
                        row[
                            "MeanEffectiveVariantDelta"
                        ]
                    )


                    delta_variants_string = (

                        f"{delta_variants:.1f}"

                        if pd.notna(
                            delta_variants
                        )

                        else "NA"
                    )


                    print(

                        f"  {row['DatasetLabel']:<36}"
                        f"{int(row['N_pairs']):>7}"
                        f"{row['MeanDelta']:>12.5f}"
                        f"{row['MedianDelta']:>12.5f}"
                        f"{row['MeanAbsDelta']:>12.5f}"
                        f"{row['MedianAbsDelta']:>12.5f}"
                        f"{delta_variants_string:>13}"
                    )


            # =================================================
            # SIMULATION: PER SCENARIO
            # =================================================

            else:


                detail = (

                    contrast_data

                    .groupby(

                        "Scenario",

                        as_index=False
                    )

                    .agg(

                        Simulations=(
                            "Dataset",
                            "nunique"
                        ),

                        N_pairs=(
                            "Delta",
                            "size"
                        ),

                        MeanDelta=(
                            "Delta",
                            "mean"
                        ),

                        MedianDelta=(
                            "Delta",
                            "median"
                        ),

                        MeanAbsDelta=(
                            "AbsDelta",
                            "mean"
                        ),

                        MedianAbsDelta=(
                            "AbsDelta",
                            "median"
                        ),
                    )

                    .sort_values(
                        "Scenario"
                    )
                )


                print()

                print(
                    "  PER SIMULATION SCENARIO — DESCRIPTIVE"
                )


                print(

                    f"  {'Scenario':<58}"
                    f"{'Sims':>6}"
                    f"{'Pairs':>7}"
                    f"{'Mean D':>12}"
                    f"{'Median D':>12}"
                    f"{'Mean |D|':>12}"
                    f"{'Med |D|':>12}"
                )


                for _, row in detail.iterrows():

                    print(

                        f"  {row['Scenario']:<58}"
                        f"{int(row['Simulations']):>6}"
                        f"{int(row['N_pairs']):>7}"
                        f"{row['MeanDelta']:>12.5f}"
                        f"{row['MedianDelta']:>12.5f}"
                        f"{row['MeanAbsDelta']:>12.5f}"
                        f"{row['MedianAbsDelta']:>12.5f}"
                    )


# ============================================================
# FINAL INFERENTIAL TABLE
# ============================================================

print()

print(
    "=" * 140
)

print(
    "FINAL INFERENTIAL TABLE — "
    "USE THIS FOR SIGNIFICANCE CLAIMS"
)

print(
    "=" * 140
)


inferential = summary.sort_values(

    [
        "FDR_Q",
        "MedianAbsDelta"
    ],

    ascending=[
        True,
        False
    ],

    na_position=
        "last"
)


print(

    f"{'Category':<38}"
    f"{'Factor':<34}"
    f"{'Contrast':<40}"
    f"{'N':>5}"
    f"{'Median D':>12}"
    f"{'Med |D|':>12}"
    f"{'Raw P':>12}"
    f"{'FDR Q':>12}"
    f"{'FDR':>7}"
    f"{'FLAG':>8}"
)


print(
    "-" * 180
)


for _, row in inferential.iterrows():


    p_string = (

        f"{row['WilcoxonP']:.4g}"

        if pd.notna(
            row[
                "WilcoxonP"
            ]
        )

        else "NA"
    )


    q_string = (

        f"{row['FDR_Q']:.4g}"

        if pd.notna(
            row[
                "FDR_Q"
            ]
        )

        else "NA"
    )


    flag = (

        "ZERO"

        if row[
            "AllZero"
        ]

        else ""
    )


    print(

        f"{row['Category']:<38}"
        f"{row['FactorLabel']:<34}"
        f"{row['Contrast']:<40}"
        f"{int(row['Datasets']):>5}"
        f"{row['MedianDelta']:>12.5f}"
        f"{row['MedianAbsDelta']:>12.5f}"
        f"{p_string:>12}"
        f"{q_string:>12}"
        f"{row['FDR_Significant']:>7}"
        f"{flag:>8}"
    )


# ============================================================
# DATASET-BALANCED FACTOR RANKING
# ============================================================

factor_dataset = (

    pair_units

    .groupby(

        [
            "Category",
            "Factor",
            "FactorLabel",
            "Dataset",
            "DatasetLabel"
        ],

        as_index=False
    )

    .agg(

        MeanAbsDelta=(
            "AbsDelta",
            "mean"
        ),

        MedianAbsDelta=(
            "AbsDelta",
            "median"
        ),

        MeanSignedDelta=(
            "Delta",
            "mean"
        ),

        MatchedPairs=(
            "Delta",
            "size"
        ),
    )
)


factor_pair_counts = (

    pair_units

    .groupby(

        [
            "Category",
            "Factor",
            "FactorLabel"
        ],

        as_index=False
    )

    .agg(

        TotalMatchedPairs=(
            "Delta",
            "size"
        )
    )
)


overview = (

    factor_dataset

    .groupby(

        [
            "Category",
            "Factor",
            "FactorLabel"
        ],

        as_index=False
    )

    .agg(

        Datasets=(
            "Dataset",
            "nunique"
        ),

        MeanAbsDelta=(
            "MeanAbsDelta",
            "mean"
        ),

        MedianAbsDelta=(
            "MeanAbsDelta",
            "median"
        ),

        MaxAbsDelta=(
            "MeanAbsDelta",
            "max"
        ),
    )

    .merge(

        factor_pair_counts,

        on=[
            "Category",
            "Factor",
            "FactorLabel"
        ],

        how="left"
    )

    .sort_values(

        "MedianAbsDelta",

        ascending=False
    )
)


print()

print(
    "=" * 140
)

print(
    "FINAL RANKING — WHICH CONTROLLED "
    "ANALYTICAL FACTORS CHANGED h2 MOST?"
)

print(
    "=" * 140
)


print(

    f"{'Rank':<6}"
    f"{'Category':<40}"
    f"{'Factor':<40}"
    f"{'Pairs':>9}"
    f"{'Datasets':>10}"
    f"{'Mean |D|':>14}"
    f"{'Median |D|':>16}"
    f"{'Max |D|':>14}"
)


print(
    "-" * 155
)


for rank, (
    _,
    row
) in enumerate(

    overview.iterrows(),

    1
):

    print(

        f"{rank:<6}"
        f"{row['Category']:<40}"
        f"{row['FactorLabel']:<40}"
        f"{int(row['TotalMatchedPairs']):>9}"
        f"{int(row['Datasets']):>10}"
        f"{row['MeanAbsDelta']:>14.5f}"
        f"{row['MedianAbsDelta']:>16.5f}"
        f"{row['MaxAbsDelta']:>14.5f}"
    )


# ============================================================
# BROAD-SOURCE SUMMARY
#
# IMPORTANT:
# This is NOT formal variance partitioning.
# ============================================================

category_dataset = (

    factor_dataset

    .groupby(

        [
            "Category",
            "Dataset",
            "DatasetLabel"
        ],

        as_index=False
    )

    .agg(

        MeanAbsDelta=(
            "MeanAbsDelta",
            "mean"
        ),

        MeanSignedDelta=(
            "MeanSignedDelta",
            "mean"
        ),
    )
)


category_pairs = (

    pair_units

    .groupby(

        "Category",

        as_index=False
    )

    .agg(

        TotalMatchedPairs=(
            "Delta",
            "size"
        )
    )
)


category_summary = (

    category_dataset

    .groupby(

        "Category",

        as_index=False
    )

    .agg(

        Datasets=(
            "Dataset",
            "nunique"
        ),

        MeanAbsDelta=(
            "MeanAbsDelta",
            "mean"
        ),

        MedianAbsDelta=(
            "MeanAbsDelta",
            "median"
        ),

        MaxAbsDelta=(
            "MeanAbsDelta",
            "max"
        ),
    )

    .merge(

        category_pairs,

        on="Category",

        how="left"
    )

    .sort_values(

        "MedianAbsDelta",

        ascending=False
    )
)


print()

print(
    "=" * 140
)

print(
    "BROAD SOURCE SUMMARY — "
    "MATCHED SENSITIVITY, NOT FORMAL VARIANCE PARTITIONING"
)

print(
    "=" * 140
)


print(

    f"{'Source':<48}"
    f"{'Pairs':>10}"
    f"{'Datasets':>10}"
    f"{'Mean |D h2|':>16}"
    f"{'Median |D h2|':>18}"
    f"{'Max |D h2|':>16}"
)


print(
    "-" * 125
)


for _, row in category_summary.iterrows():

    print(

        f"{row['Category']:<48}"
        f"{int(row['TotalMatchedPairs']):>10}"
        f"{int(row['Datasets']):>10}"
        f"{row['MeanAbsDelta']:>16.5f}"
        f"{row['MedianAbsDelta']:>18.5f}"
        f"{row['MaxAbsDelta']:>16.5f}"
    )


# ============================================================
# ALL-ZERO / DUPLICATE OUTPUT FLAGS
# ============================================================

zero_flags = summary[

    summary[
        "AllZero"
    ] == True

].copy()


print()

print(
    "=" * 140
)

print(
    "UPSTREAM ZERO/IDENTICAL-OUTPUT AUDIT FLAGS"
)

print(
    "=" * 140
)


if zero_flags.empty:

    print(

        "No contrast was exactly zero "
        "across every represented dataset."
    )


else:

    print(

        "Treat these as upstream implementation/audit "
        "flags until verified:"
    )


    for _, row in zero_flags.iterrows():

        print(

            f" - {row['FactorLabel']}: "
            f"{row['Contrast']} "
            f"(datasets={int(row['Datasets'])})"
        )


# ============================================================
# INTERPRETATION NOTES
# ============================================================

print()

print(
    "=" * 140
)

print(
    "INTERPRETATION NOTES"
)

print(
    "=" * 140
)


print(
"""
1. HE vs REML, GRM, covariate, PCA, LDAK model and
   GEMMA-vs-DPR comparisons use the SHA-256 fingerprint of
   the actual .bim input whenever genotype input is required.

2. Results.csv numberofvariants is treated as an
   effective/output count, NOT as the identity of the input
   SNP set.

3. Clumping/pruning and SNP-set comparisons deliberately
   allow the input SNP fingerprint to change because that is
   the intervention being tested.

4. LD-reference/weight comparisons hold the raw GWAS source
   constant while allowing the reference resource to change.

5. Overlapping CV folds are averaged before inference.
   Folds are not treated as independent observations.

6. REAL mode uses phenotype-level effects for Wilcoxon
   testing. Per-phenotype rows are descriptive.

7. All valid Method19 contrasts are one BH-FDR
   multiple-testing family.

8. Exactly-zero contrasts across every dataset are audit
   flags, not automatic biological evidence of no effect.

9. Input data type is not claimed as an isolated one-factor
   effect because genotype-vs-summary input is structurally
   coupled to different model and reference requirements in
   this benchmark.

10. The three-category summary is a matched
    sensitivity/decomposition summary, not a formal partition
    of total statistical variance.
"""
)


print(
    "=" * 140
)

print(
    "DONE — NOTHING WAS SAVED TO DISK"
)

print(
    "=" * 140
)

print()