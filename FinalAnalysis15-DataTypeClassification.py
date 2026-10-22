
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinalAnalysis15-DataTypeClassification.py

Classify heritability methods by the data available
to researchers.

Default output:
    Table 1: Method input-data requirements
    Table 2: Available-data scenarios and feasible methods

Optional --rank:
    Best feasible method and configuration based on
    existing simulated known-truth accuracy.

No PRS or empirical phenotype analysis.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. METHOD DEFINITIONS FROM MANUSCRIPT TABLE 2
# ============================================================

# Each method:
# Number, Method, Script, Configurations,
# Requires GWAS, Requires phenotype + genotype,
# Requires genotype reference,
# Requires external LD scores/weights,
# Requires precomputed LDAK tagging,
# Description, Covariates/PCA

METHODS = [
    (
        1, "LDpred-2-Heritability", "Method1.py", 4,
        True, False, True, False, False,
        "GWAS + genotype-derived LD / variant matching",
        "No"
    ),
    (
        2, "GCTA-Heritability", "Method2.py", 6,
        False, True, False, False, False,
        "Genotypes + phenotype; GRM / REML",
        "Optional"
    ),
    (
        3, "Gemma1-Heritability", "Method3.py", 24,
        False, True, False, False, False,
        "Genotypes + phenotype; GEMMA HE / REML",
        "Optional"
    ),
    (
        4, "Gemma2-Heritability", "Method4.py", 6,
        True, True, False, False, False,
        "Genotypes + phenotype + GWAS; GEMMA HE",
        "Optional"
    ),
    (
        5, "Gemma-LDSC-Heritability", "Method5.py", 3,
        True, True, False, False, False,
        "Genotypes + phenotype + GWAS; derived LD scores",
        "Optional"
    ),
    (
        6, "DPR-Gemma-Heritability", "Method6.py", 24,
        False, True, False, False, False,
        "Genotypes + phenotype; DPR GRM + GEMMA",
        "Optional"
    ),
    (
        7, "LDSC-Heritability", "Method7.py", 4,
        True, False, False, True, False,
        "GWAS + external LD scores and regression weights",
        "No"
    ),
    (
        8, "LDAK-Calculated-Heritability",
        "Method8-Calculated.py", 7,
        True, False, True, False, False,
        "GWAS + genotypes to calculate tagging",
        "No"
    ),
    (
        9, "LDAK-Precomputed-Heritability",
        "Method8-Precomputed.py", 6,
        True, False, False, False, True,
        "GWAS + precomputed LDAK tagging",
        "No"
    ),
    (
        10, "LDSC-Genotype-Heritability",
        "Method10.py", 1,
        True, False, True, False, False,
        "GWAS + genotypes to calculate LD scores",
        "No"
    ),
]

METHOD_COLUMNS = [
    "MethodNumber",
    "Method",
    "Script",
    "ModelConfigurations",
    "NeedsGWAS",
    "NeedsPhenotypedGenotypes",
    "NeedsReferenceGenotypes",
    "NeedsExternalLD",
    "NeedsPrecomputedTagging",
    "InputDescription",
    "CovariatesPCA",
]


# ============================================================
# 2. DATA-AVAILABILITY SCENARIOS
# ============================================================

# Fields:
# Scenario, GWAS, genotype + phenotype,
# reference genotypes, external LD scores,
# precomputed LDAK tagging.

SCENARIOS = [
    ("GWAS summary statistics only", 1, 0, 0, 0, 0),
    ("GWAS + external LD scores/weights", 1, 0, 0, 1, 0),
    ("GWAS + precomputed LDAK tagging", 1, 0, 0, 0, 1),
    ("GWAS + external LD + LDAK tagging", 1, 0, 0, 1, 1),

    ("GWAS + reference genotypes (no phenotype)", 1, 0, 1, 0, 0),
    ("GWAS + reference genotypes + external LD", 1, 0, 1, 1, 0),
    ("GWAS + reference genotypes + LDAK tagging", 1, 0, 1, 0, 1),

    ("Genotype + phenotype only", 0, 1, 0, 0, 0),
    ("Genotype + phenotype + external LD", 0, 1, 0, 1, 0),
    ("Genotype + phenotype + LDAK tagging", 0, 1, 0, 0, 1),

    ("Genotype + phenotype + GWAS", 1, 1, 0, 0, 0),
    ("Genotype + phenotype + GWAS + external LD", 1, 1, 0, 1, 0),
    ("Genotype + phenotype + GWAS + tagging", 1, 1, 0, 0, 1),

    ("Genotype + phenotype + GWAS + LD + tagging", 1, 1, 0, 1, 1),

    ("Reference genotypes only (no GWAS)", 0, 0, 1, 0, 0),
]

SCENARIO_COLUMNS = [
    "DataScenario",
    "GWAS",
    "PhenotypedGenotypes",
    "ReferenceGenotypes",
    "ExternalLDScoresWeights",
    "PrecomputedLDAKTagging",
]


def get_methods():

    df = pd.DataFrame(
        METHODS,
        columns=METHOD_COLUMNS
    )

    assert len(df) == 10
    assert df["ModelConfigurations"].sum() == 85

    conditions = [
        df.NeedsGWAS & df.NeedsPhenotypedGenotypes,
        df.NeedsGWAS & df.NeedsReferenceGenotypes,
        df.NeedsGWAS & df.NeedsExternalLD,
        df.NeedsGWAS & df.NeedsPrecomputedTagging,
        ~df.NeedsGWAS & df.NeedsPhenotypedGenotypes,
    ]

    labels = [
        "GWAS + genotype + phenotype",
        "GWAS + genotype-derived reference",
        "GWAS + external LD",
        "GWAS + precomputed tagging",
        "Genotype + phenotype",
    ]

    df["DataTypeClass"] = np.select(
        conditions,
        labels,
        default="Other / review"
    )

    return df


def get_scenarios():

    return pd.DataFrame(
        SCENARIOS,
        columns=SCENARIO_COLUMNS
    )


# ============================================================
# 3. DETERMINE WHICH METHODS CAN BE USED
# ============================================================

def feasible(method, scenario):

    # A phenotyped genotype cohort can also supply
    # genotypes for reference construction.

    reference_available = bool(
        scenario["PhenotypedGenotypes"]
        or scenario["ReferenceGenotypes"]
    )

    checks = [
        not method.NeedsGWAS or bool(scenario.GWAS),

        not method.NeedsPhenotypedGenotypes
        or bool(scenario.PhenotypedGenotypes),

        not method.NeedsReferenceGenotypes
        or reference_available,

        not method.NeedsExternalLD
        or bool(scenario.ExternalLDScoresWeights),

        not method.NeedsPrecomputedTagging
        or bool(scenario.PrecomputedLDAKTagging),
    ]

    return all(checks)


def classify_scenarios(methods, scenarios):

    rows = []
    matrix = []

    for _, scenario in scenarios.iterrows():

        available = []

        for _, method in methods.iterrows():

            compatible = feasible(method, scenario)

            matrix.append({
                "DataScenario": scenario.DataScenario,
                "MethodNumber": method.MethodNumber,
                "Method": method.Method,
                "Feasible": compatible,
            })

            if compatible:
                available.append(method)

        rows.append({
            **scenario.to_dict(),

            "FeasibleMethodCount": len(available),

            "FeasibleConfigUpperBound": sum(
                int(m.ModelConfigurations)
                for m in available
            ),

            "MethodNumbers": ", ".join(
                f"M{int(m.MethodNumber)}"
                for m in available
            ) or "NONE",

            "FeasibleMethods": "; ".join(
                m.Method for m in available
            ) or "NONE",
        })

    return pd.DataFrame(rows), pd.DataFrame(matrix)


# ============================================================
# 4. OPTIONAL SIMULATION ACCURACY RECOMMENDATIONS
# ============================================================

def h2_band(value):

    if value <= 0.10:
        return "Low (0.05-0.10)"

    if value < 0.50:
        return "Moderate (0.30)"

    return "High (0.50-0.70)"


def load_simulation_accuracy(path, scope):

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. "
            "Run FinalAnalysis05-Accuracy.py first."
        )

    df = pd.read_csv(path, low_memory=False)

    required = [
        "Domain", "Dataset", "TraitType", "Method",
        "ConfigID", "Scale", "Estimate", "Reference",
        "TargetN", "TrueH2", "CausalSNPs",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for c in ["Domain", "TraitType", "Scale"]:
        df[c] = df[c].astype(str).str.lower().str.strip()

    for c in ["Dataset", "Method", "ConfigID"]:
        df[c] = df[c].astype(str).str.strip()

    for c in [
        "Estimate", "Reference", "TargetN",
        "TrueH2", "CausalSNPs",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    scale_ok = (
        (
            (df.TraitType == "binary")
            & (df.Scale == "liability_binary")
        )
        |
        (
            (df.TraitType == "continuous")
            & (df.Scale == "native_quantitative")
        )
    )

    df = df[
        (df.Domain == "simulated")
        & scale_ok
        & df.TargetN.isin([750, 5000])
        & df.CausalSNPs.isin([100, 5000])
        & df.TrueH2.apply(
            lambda h: any(
                np.isclose(h, v)
                for v in [0.05, 0.10, 0.30, 0.50, 0.70]
            )
        )
        & np.isfinite(df.Estimate)
        & np.isfinite(df.Reference)
    ].copy()

    if df.empty:
        raise ValueError("No valid simulated results.")

    if not np.allclose(
        df.Reference,
        df.TrueH2,
        atol=1e-8
    ):
        raise ValueError(
            "Generating TrueH2 and Reference disagree."
        )

    if df.duplicated([
        "Dataset", "TargetN", "Method", "ConfigID"
    ]).any():
        raise ValueError("Duplicate configuration results.")

    df["AE"] = (df.Estimate - df.Reference).abs()

    df["H2Class"] = df.TrueH2.map(h2_band)

    if scope == "overall":

        df["H2Class"] = "All"
        df["Architecture"] = "All"
        df["ExactH2"] = "All"

    elif scope == "band":

        df["Architecture"] = "All"
        df["ExactH2"] = "All"

    else:

        df["Architecture"] = (
            df.CausalSNPs.astype(int).astype(str)
        )

        df["ExactH2"] = df.TrueH2.map(
            lambda h: f"{h:.2f}"
        )

    return df


def recommend(methods, scenarios, sim, minimum_coverage):

    results = []

    keys = [
        "TraitType",
        "TargetN",
        "H2Class",
        "ExactH2",
        "Architecture",
    ]

    for _, scenario in scenarios.iterrows():

        eligible = set(
            methods.loc[
                methods.apply(
                    lambda m: feasible(m, scenario),
                    axis=1
                ),
                "Method"
            ]
        )

        if not eligible:
            continue

        data = sim[
            sim.Method.isin(eligible)
        ].copy()

        for group_key, group in data.groupby(
            keys,
            sort=True
        ):

            meta = dict(zip(keys, group_key))

            selection = (sim[keys] == pd.Series(meta)).all(axis=1)

            n_expected = sim.loc[
                selection, "Dataset"
            ].nunique()

            if n_expected == 0:
                continue

            # One error observation per dataset/method
            # after averaging across its configurations.

            method_units = (
                group.groupby(
                    ["Dataset", "Method"],
                    as_index=False
                )
                .agg(AE=("AE", "mean"))
            )

            methods_ranked = (
                method_units.groupby("Method")
                .agg(
                    MAE=("AE", "mean"),
                    Datasets=("Dataset", "nunique")
                )
                .reset_index()
            )

            configs_ranked = (
                group.groupby(["Method", "ConfigID"])
                .agg(
                    MAE=("AE", "mean"),
                    Datasets=("Dataset", "nunique")
                )
                .reset_index()
            )

            minimum_n = np.ceil(
                n_expected * minimum_coverage
            )

            methods_ranked = methods_ranked[
                methods_ranked.Datasets >= minimum_n
            ]

            configs_ranked = configs_ranked[
                configs_ranked.Datasets >= minimum_n
            ]

            if methods_ranked.empty or configs_ranked.empty:
                continue

            best_method = (
                methods_ranked.sort_values(
                    ["MAE", "Method"]
                ).iloc[0]
            )

            best_config = (
                configs_ranked.sort_values(
                    ["MAE", "Method", "ConfigID"]
                ).iloc[0]
            )

            results.append({
                "DataScenario": scenario.DataScenario,
                **meta,

                "BestFeasibleMethod": best_method.Method,
                "MethodAverageMAE": best_method.MAE,
                "MethodN": int(best_method.Datasets),

                "BestFeasibleConfigurationMethod":
                    best_config.Method,

                "BestFeasibleConfiguration":
                    best_config.ConfigID,

                "ConfigurationMAE":
                    best_config.MAE,

                "ConfigurationN":
                    int(best_config.Datasets),

                "ExpectedDatasets":
                    int(n_expected),
            })

    return pd.DataFrame(results)


# ============================================================
# 5. PRINT AND SAVE
# ============================================================

def print_table(title, table):

    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.max_colwidth", 160,
        "display.width", 240,
    ):
        print(table.to_string(index=False))


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out",
        default="FinalAnalysis15-DataTypeClassification"
    )

    parser.add_argument(
        "--rank",
        action="store_true",
        help="Rank feasible methods using simulated accuracy"
    )

    parser.add_argument(
        "--input",
        default="FinalAnalysis04-XX/FinalAnalysis04-XX_units.csv"
    )

    parser.add_argument(
        "--scope",
        choices=["overall", "band", "exact"],
        default="band"
    )

    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.80
    )

    args = parser.parse_args()

    if not 0 < args.min_coverage <= 1:
        parser.error("--min-coverage must be between 0 and 1")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    methods = get_methods()
    scenarios = get_scenarios()

    feasible_table, matrix = classify_scenarios(
        methods,
        scenarios
    )

    print_table(
        "TABLE 1: HERITABILITY METHOD INPUT-DATA CLASSIFICATION",
        methods
    )

    print_table(
        "TABLE 2: DATA AVAILABILITY AND FEASIBLE METHODS",
        feasible_table
    )

    methods.to_csv(
        out / "Table1_MethodDataRequirements.csv",
        index=False
    )

    feasible_table.to_csv(
        out / "Table2_DataAvailabilityScenarios.csv",
        index=False
    )

    matrix.to_csv(
        out / "MethodScenarioFeasibilityMatrix.csv",
        index=False
    )

    if args.rank:

        simulated = load_simulation_accuracy(
            Path(args.input),
            args.scope
        )

        rankings = recommend(
            methods,
            scenarios,
            simulated,
            args.min_coverage
        )

        print_table(
            "OPTIONAL TABLE 3: SIMULATION-BASED RECOMMENDATIONS",
            rankings
        )

        rankings.to_csv(
            out / "Optional_SimulationRecommendationsByDataAvailability.csv",
            index=False
        )

    print("\nResults saved in:", out.resolve())
    print("GWAS alone requires additional LD or tagging information.")


if __name__ == "__main__":
    main()
