#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulation-only guidance: exactly TWO tables, using existing heritability results.

Run from finalheritability/: 
  python FinalAnalysis14-DataClassificationAndPerformance.py --bootstrap 2000

Input: FinalAnalysis04-XX/FinalAnalysis04-XX_units.csv
Outputs: two printed tables and two CSVs. Does not rerun estimation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

H2_LEVELS = (0.05, 0.10, 0.30, 0.50, 0.70)
SAMPLE_SIZES = (750, 5000)
CAUSAL_COUNTS = (100, 5000)
GROUP = ["Scope", "PhenotypeType", "TargetN", "HeritabilityClass", "TrueH2", "CausalSNPs"]
REQUIRED = ("Domain", "Dataset", "TraitType", "Method", "ConfigID", "Scale",
            "Estimate", "Reference", "TargetN", "TrueH2", "CausalSNPs")


def h2_class(x):
    if x <= 0.10 + 1e-9:
        return "Low (0.05-0.10)"
    if x < 0.50 - 1e-9:
        return "Moderate (0.30)"
    return "High (0.50-0.70)"


def bootstrap_mean_ci(values, rng, iterations):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    samples = rng.integers(0, len(x), size=(iterations, len(x)))
    return tuple(np.quantile(x[samples].mean(axis=1), (0.025, 0.975)))


def load_simulation(path):
    if not path.is_file():
        raise SystemExit(f"Cannot find {path}. Run the existing final accuracy analysis first.")
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")
    for col in ("Domain", "TraitType", "Scale"):
        df[col] = df[col].astype(str).str.strip().str.lower()
    df = df.loc[df["Domain"].eq("simulated")].copy()
    for col in ("Dataset", "Method", "ConfigID"):
        df[col] = df[col].astype(str).str.strip()
    for col in ("Estimate", "Reference", "TargetN", "TrueH2", "CausalSNPs"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    correct_scale = (
        (df["TraitType"].eq("binary") & df["Scale"].eq("liability_binary")) |
        (df["TraitType"].eq("continuous") & df["Scale"].eq("native_quantitative"))
    )
    valid_h2 = np.logical_or.reduce([
        np.isclose(df["TrueH2"], h, atol=1e-8) for h in H2_LEVELS
    ])
    df = df.loc[
        correct_scale & valid_h2 & df["TargetN"].isin(SAMPLE_SIZES) &
        df["CausalSNPs"].isin(CAUSAL_COUNTS) &
        np.isfinite(df["Estimate"]) & np.isfinite(df["Reference"])
    ].copy()
    if df.empty:
        raise SystemExit("No eligible known-truth simulation records were found.")
    if not np.allclose(df["Reference"], df["TrueH2"], atol=1e-8):
        raise SystemExit("Simulation reference does not match generating TrueH2.")
    if df.duplicated(["Dataset", "Method", "ConfigID"]).any():
        raise SystemExit("Repeated dataset/method/configuration rows: inspect input.")
    for col in ("TraitType", "TargetN", "TrueH2", "CausalSNPs"):
        if (df.groupby("Dataset")[col].nunique() > 1).any():
            raise SystemExit(f"Dataset carries conflicting {col} metadata.")
    df["PhenotypeType"] = df["TraitType"].str.capitalize()
    df["HeritabilityClass"] = df["TrueH2"].map(h2_class)
    df["SignedError"] = df["Estimate"] - df["Reference"]
    df["AE"] = df["SignedError"].abs()
    df["SquaredError"] = df["SignedError"] ** 2
    return df


def build_scopes(sim):
    exact = sim.copy()
    exact["Scope"] = "Exact"
    band = sim.copy()
    band["Scope"] = "Heritability band"
    band["TrueH2"] = np.nan
    band["CausalSNPs"] = np.nan
    overall = sim.copy()
    overall["Scope"] = "Overall"
    overall["TrueH2"] = np.nan
    overall["CausalSNPs"] = np.nan
    overall["HeritabilityClass"] = "All"
    return pd.concat((exact, band, overall), ignore_index=True)


def expected_datasets(meta):
    if meta["Scope"] == "Exact":
        return 5
    if meta["Scope"] == "Heritability band":
        return 10 if meta["HeritabilityClass"].startswith("Moderate") else 20
    return 50


def method_dataset_units(frame):
    # Within each simulation, average errors across configurations of a method.
    # This ranks method-family ROBUSTNESS, not a particular runnable configuration.
    return (frame.groupby(["Dataset", "Method"], as_index=False)
            .agg(AE=("AE", "mean"), SignedError=("SignedError", "mean"),
                 SquaredError=("SquaredError", "mean")))


def rank_choices(frame, choice_col, expected, rng, iterations):
    summaries = []
    for name, g in frame.groupby(choice_col, sort=True):
        if g["Dataset"].duplicated().any():
            raise SystemExit(f"Duplicate dataset for {choice_col}: {name}")
        n = g["Dataset"].nunique()
        coverage = n / expected
        if coverage < 0.80 or n < 4:
            continue
        ci_l, ci_u = bootstrap_mean_ci(g["AE"], rng, iterations)
        summaries.append(dict(Name=name, MAE=float(g["AE"].mean()),
                              MAE_CI95_Lower=ci_l, MAE_CI95_Upper=ci_u,
                              RMSE=float(np.sqrt(g["SquaredError"].mean())),
                              Bias=float(g["SignedError"].mean()),
                              N_Datasets=n, Coverage=coverage))
    if not summaries:
        return pd.DataFrame()
    return pd.DataFrame(summaries).sort_values(
        ["MAE", "Name"], kind="stable").reset_index(drop=True)


def paired_test(frame, choice_col, winner, runner, rng, iterations):
    if runner is None:
        return dict(PairedN=np.nan, DeltaMAE=np.nan,
                    DeltaMAE_CI95_Lower=np.nan, DeltaMAE_CI95_Upper=np.nan)
    pivot = frame.loc[frame[choice_col].isin((winner, runner))].pivot(
        index="Dataset", columns=choice_col, values="AE")
    diff = (pivot[winner] - pivot[runner]).dropna().to_numpy(dtype=float)
    low, high = bootstrap_mean_ci(diff, rng, iterations)
    return dict(PairedN=len(diff), DeltaMAE=float(diff.mean()) if len(diff) else np.nan,
                DeltaMAE_CI95_Lower=low, DeltaMAE_CI95_Upper=high)


def make_tables(sim, rng, iterations):
    methods, configurations = [], []
    for key, frame in build_scopes(sim).groupby(GROUP, dropna=False, sort=True):
        meta = dict(zip(GROUP, key))
        expected = expected_datasets(meta)
        observed = frame["Dataset"].nunique()
        if observed != expected:
            raise SystemExit(f"Incomplete simulation class: {meta}; "
                             f"found {observed}, expected {expected} datasets")
        method_frame = method_dataset_units(frame)
        mt = rank_choices(method_frame, "Method", expected, rng, iterations)
        ct = rank_choices(frame, "ConfigID", expected, rng, iterations)
        if mt.empty or ct.empty:
            raise SystemExit(f"No eligible method/configuration for {meta}")
        base = {**meta, "ExpectedDatasets": expected}
        method_best = mt.iloc[0]
        config_best = ct.iloc[0]
        next_method = str(mt.iloc[1]["Name"]) if len(mt) > 1 else None
        next_config = str(ct.iloc[1]["Name"]) if len(ct) > 1 else None
        # Fully specified class label, readable in exported CSV and console.
        base["PhenotypeClass"] = (
            f"{meta['PhenotypeType']} / {meta['HeritabilityClass']} / "
            + (f"{int(meta['CausalSNPs'])} causal SNPs" if pd.notna(meta["CausalSNPs"])
               else "all architectures"))
        method_configs = frame.loc[frame["Method"].eq(method_best["Name"]), "ConfigID"].nunique()
        methods.append({**base, "BestMethod": method_best["Name"],
                        "MethodMetric": "Mean AE across configurations per dataset",
                        "ConfigurationsInMethod": method_configs,
                        **{c:method_best[c] for c in ("MAE", "MAE_CI95_Lower",
                            "MAE_CI95_Upper", "RMSE", "Bias", "N_Datasets", "Coverage")},
                        "RunnerUpMethod": next_method or "",
                        "RunnerUpMAE": float(mt.iloc[1]["MAE"]) if next_method else np.nan,
                        **paired_test(method_frame, "Method", method_best["Name"],
                                      next_method, rng, iterations)})
        configuration_method = frame.loc[
            frame["ConfigID"].eq(config_best["Name"]), "Method"].iloc[0]
        configurations.append({**base, "BestMethod": configuration_method,
                               "BestConfiguration": config_best["Name"],
                               **{c:config_best[c] for c in ("MAE", "MAE_CI95_Lower",
                                   "MAE_CI95_Upper", "RMSE", "Bias", "N_Datasets", "Coverage")},
                               "RunnerUpConfiguration": next_config or "",
                               "RunnerUpMAE": float(ct.iloc[1]["MAE"]) if next_config else np.nan,
                               **paired_test(frame, "ConfigID", config_best["Name"],
                                             next_config, rng, iterations)})
    return pd.DataFrame(methods), pd.DataFrame(configurations)


def display_and_save(name, frame, destination):
    print("\n" + "="*140 + "\n" + name + "\n" + "="*140)
    with pd.option_context("display.max_rows", None, "display.max_columns", None,
                           "display.max_colwidth", None, "display.width", 10000,
                           "display.expand_frame_repr", False):
        print(frame.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    frame.to_csv(destination, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("FinalAnalysis04-XX/FinalAnalysis04-XX_units.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("FinalAnalysis14-DataClassificationAndPerformance"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap must be >=100")
    rng = np.random.default_rng(args.seed)
    methods, configurations = make_tables(load_simulation(args.input), rng, args.bootstrap)
    args.output.mkdir(parents=True, exist_ok=True)
    display_and_save("TABLE 1: BEST HERITABILITY METHODS BY SIMULATED DATASET CLASS",
                     methods, args.output / "Table1_Simulated_BestMethods.csv")
    display_and_save("TABLE 2: BEST HERITABILITY CONFIGURATIONS BY SIMULATED DATASET CLASS",
                     configurations, args.output / "Table2_Simulated_BestConfigurations.csv")


if __name__ == "__main__":
    main()
