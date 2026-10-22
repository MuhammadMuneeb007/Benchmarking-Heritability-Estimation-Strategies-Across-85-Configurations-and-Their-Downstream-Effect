 
# -*- coding: utf-8 -*-

"""
Method20B-SimulationConfigurationAccuracy.py

Goal
----
Rank individual heritability CONFIGURATIONS by closeness to the known simulated
heritability value.

This analysis uses ONLY continuous simulated phenotypes because their estimated
h2 values and simulated truth are currently on the same scale.

Statistical unit
----------------
1. First average h2 across available CV folds within each:
       simulation dataset x configuration
2. Compare that fold-averaged estimate with the known simulated h2.
3. Rank configurations using:
       MAE               = mean absolute error (PRIMARY)
       MedianAbsError    = median absolute error
       Bias              = mean(estimate - truth)
       RMSE
       PctWithin0.025
       PctWithin0.05
       PctWithin0.10
4. Report N=750 and N=5000 separately.
5. Also create an overall equal-weighted rank across the two sample sizes.

Why MAE is primary
------------------
MAE directly answers: "How close was this configuration to the known h2?"
Lower MAE = better.

Input
-----
Method21_unit_stability_simulated.csv

Run
---
python Method20B-SimulationConfigurationAccuracy.py | tee Method20B_output.txt

Outputs
-------
Method20B_configuration_accuracy_byN.csv
Method20B_configuration_accuracy_overall.csv
Method20B_best_configuration_per_method.csv
Method20B_accuracy_by_true_h2.csv
Method20B_top20_N750.csv
Method20B_top20_N5000.csv
Method20B_top20_overall.csv
Method20B_MAIN_RESULTS.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".").resolve()
INPUT = ROOT / "Method21_unit_stability_simulated.csv"

MIN_COVERAGE = 0.80  # ranking requires >=80% of eligible simulations


def fmt(x, d=4):
    try:
        x = float(x)
        if not np.isfinite(x):
            return "NA"
        return f"{x:.{d}f}"
    except Exception:
        return "NA"


def safe_rmse(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square(x))))


if not INPUT.exists():
    raise SystemExit(
        "ERROR: Method21_unit_stability_simulated.csv not found.\n"
        "Run the corrected Method21-StabilityReliability.py first."
    )

d = pd.read_csv(INPUT)

required = [
    "sim_id", "trait_type", "sample_size", "true_h2",
    "method", "config_id", "h2_mean"
]

missing = [c for c in required if c not in d.columns]
if missing:
    raise SystemExit("ERROR: Missing required columns: " + ", ".join(missing))

# ---------------------------------------------------------------------
# 1. Continuous simulations only
# ---------------------------------------------------------------------

x = d[
    d["trait_type"].astype(str).str.lower().eq("continuous")
].copy()

for c in ["sample_size", "true_h2", "h2_mean", "fold_sd"]:
    if c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")

x = x[
    np.isfinite(x["true_h2"])
    & np.isfinite(x["h2_mean"])
].copy()

x["error"] = x["h2_mean"] - x["true_h2"]
x["abs_error"] = x["error"].abs()
x["sq_error"] = x["error"] ** 2

print("=" * 110)
print("METHOD 20B — SIMULATION CONFIGURATION ACCURACY")
print("=" * 110)
print("Input:", INPUT)
print("Continuous configuration × simulation units:", len(x))
print("Sample sizes:", sorted(x["sample_size"].dropna().unique().tolist()))
print("True h2 values:", sorted(x["true_h2"].dropna().unique().tolist()))
print()

# Number of eligible simulation datasets at each N.
eligible_sim_counts = (
    x.groupby("sample_size")["sim_id"]
    .nunique()
    .to_dict()
)

print("Eligible continuous simulations per N:")
for n, k in sorted(eligible_sim_counts.items()):
    print(f"  N={int(n)}: {k}")
print()

# ---------------------------------------------------------------------
# 2. Configuration-level accuracy by sample size
# ---------------------------------------------------------------------

rows = []

group_cols = ["sample_size", "method", "config_id"]

for (n, method, config), g in x.groupby(group_cols, dropna=False):
    total_expected = eligible_sim_counts.get(n, np.nan)
    n_sims = g["sim_id"].nunique()

    err = pd.to_numeric(g["error"], errors="coerce")
    ae = pd.to_numeric(g["abs_error"], errors="coerce")

    coverage = (
        n_sims / total_expected
        if np.isfinite(total_expected) and total_expected > 0
        else np.nan
    )

    rows.append({
        "sample_size": n,
        "method": method,
        "config_id": config,
        "n_simulations": n_sims,
        "expected_simulations": total_expected,
        "coverage": coverage,

        "MAE": float(ae.mean()),
        "MedianAbsError": float(ae.median()),
        "Bias": float(err.mean()),
        "AbsBias": float(abs(err.mean())),
        "RMSE": safe_rmse(err),

        "PctWithin0.025": 100 * float((ae <= 0.025).mean()),
        "PctWithin0.05": 100 * float((ae <= 0.05).mean()),
        "PctWithin0.10": 100 * float((ae <= 0.10).mean()),

        "MeanEstimate": float(g["h2_mean"].mean()),
        "MeanTruth": float(g["true_h2"].mean()),

        "MedianFoldSD": (
            float(pd.to_numeric(g["fold_sd"], errors="coerce").median())
            if "fold_sd" in g.columns else np.nan
        ),

        "NegativeEstimatePct": 100 * float((g["h2_mean"] < 0).mean()),
        "AboveOneEstimatePct": 100 * float((g["h2_mean"] > 1).mean()),
    })

by_n = pd.DataFrame(rows)

# Rank only configurations with good coverage.
by_n["eligible_for_rank"] = by_n["coverage"] >= MIN_COVERAGE

by_n["MAE_rank_within_N"] = np.nan

for n, idx in by_n.groupby("sample_size").groups.items():
    idx = list(idx)
    eligible_idx = [
        i for i in idx
        if bool(by_n.loc[i, "eligible_for_rank"])
    ]

    if eligible_idx:
        ranks = (
            by_n.loc[eligible_idx, "MAE"]
            .rank(method="min", ascending=True)
        )
        by_n.loc[eligible_idx, "MAE_rank_within_N"] = ranks

by_n = by_n.sort_values(
    ["sample_size", "eligible_for_rank", "MAE", "AbsBias"],
    ascending=[True, False, True, True]
)

by_n.to_csv(
    "Method20B_configuration_accuracy_byN.csv",
    index=False
)

# ---------------------------------------------------------------------
# 3. Overall equal-weighted ranking across N=750 and N=5000
# ---------------------------------------------------------------------

eligible_by_n = by_n[
    by_n["eligible_for_rank"]
].copy()

overall_rows = []

for (method, config), g in eligible_by_n.groupby(
    ["method", "config_id"], dropna=False
):
    # Equal weight to each available sample-size stratum rather than allowing
    # one N to dominate by number of rows.
    overall_rows.append({
        "method": method,
        "config_id": config,
        "n_sample_sizes": g["sample_size"].nunique(),

        "Overall_MAE_equalN": float(g["MAE"].mean()),
        "Overall_MedianAbsError_equalN": float(g["MedianAbsError"].mean()),
        "Overall_AbsBias_equalN": float(g["AbsBias"].mean()),
        "Overall_RMSE_equalN": float(g["RMSE"].mean()),

        "MeanCoverage": float(g["coverage"].mean()),
        "MeanPctWithin0.05": float(g["PctWithin0.05"].mean()),
        "MeanMedianFoldSD": float(g["MedianFoldSD"].mean()),

        "MAE_N750": (
            float(g.loc[g["sample_size"] == 750, "MAE"].iloc[0])
            if (g["sample_size"] == 750).any()
            else np.nan
        ),
        "MAE_N5000": (
            float(g.loc[g["sample_size"] == 5000, "MAE"].iloc[0])
            if (g["sample_size"] == 5000).any()
            else np.nan
        ),
    })

overall = pd.DataFrame(overall_rows)

# Primary overall rank: must be represented at both sample sizes.
overall["eligible_overall"] = (
    (overall["n_sample_sizes"] == 2)
    & (overall["MeanCoverage"] >= MIN_COVERAGE)
)

overall["OverallRank"] = np.nan

idx = overall.index[overall["eligible_overall"]].tolist()
if idx:
    overall.loc[idx, "OverallRank"] = (
        overall.loc[idx, "Overall_MAE_equalN"]
        .rank(method="min", ascending=True)
    )

overall = overall.sort_values(
    ["eligible_overall", "Overall_MAE_equalN", "Overall_AbsBias_equalN"],
    ascending=[False, True, True]
)

overall.to_csv(
    "Method20B_configuration_accuracy_overall.csv",
    index=False
)

# ---------------------------------------------------------------------
# 4. Best configuration per method
# ---------------------------------------------------------------------

best_rows = []

for method, g in overall[
    overall["eligible_overall"]
].groupby("method"):
    z = g.sort_values(
        ["Overall_MAE_equalN", "Overall_AbsBias_equalN"]
    ).iloc[0]

    best_rows.append(z.to_dict())

best = pd.DataFrame(best_rows).sort_values(
    "Overall_MAE_equalN"
)

best.to_csv(
    "Method20B_best_configuration_per_method.csv",
    index=False
)

# ---------------------------------------------------------------------
# 5. Accuracy by true h2, configuration and N
# ---------------------------------------------------------------------

truth_rows = []

for (n, method, config, truth), g in x.groupby(
    ["sample_size", "method", "config_id", "true_h2"],
    dropna=False
):
    ae = g["abs_error"]
    err = g["error"]

    truth_rows.append({
        "sample_size": n,
        "method": method,
        "config_id": config,
        "true_h2": truth,
        "n": len(g),
        "MAE": float(ae.mean()),
        "MedianAbsError": float(ae.median()),
        "Bias": float(err.mean()),
        "RMSE": safe_rmse(err),
        "PctWithin0.05": 100 * float((ae <= 0.05).mean()),
    })

truth_df = pd.DataFrame(truth_rows).sort_values(
    ["sample_size", "true_h2", "MAE"]
)

truth_df.to_csv(
    "Method20B_accuracy_by_true_h2.csv",
    index=False
)

# ---------------------------------------------------------------------
# 6. Top 20 tables
# ---------------------------------------------------------------------

def top_for_n(n):
    z = by_n[
        (by_n["sample_size"] == n)
        & by_n["eligible_for_rank"]
    ].sort_values(
        ["MAE", "AbsBias", "RMSE"]
    ).head(20)
    return z

top750 = top_for_n(750)
top5000 = top_for_n(5000)

topoverall = overall[
    overall["eligible_overall"]
].sort_values(
    ["Overall_MAE_equalN", "Overall_AbsBias_equalN"]
).head(20)

top750.to_csv("Method20B_top20_N750.csv", index=False)
top5000.to_csv("Method20B_top20_N5000.csv", index=False)
topoverall.to_csv("Method20B_top20_overall.csv", index=False)

# ---------------------------------------------------------------------
# 7. Text summary
# ---------------------------------------------------------------------

lines = []

lines.append("METHOD 20B — SIMULATION CONFIGURATION ACCURACY")
lines.append("=" * 100)
lines.append("")
lines.append(
    "Primary metric: MAE between the fold-averaged configuration estimate "
    "and the known simulated heritability."
)
lines.append(
    "Lower MAE = closer to known truth. Binary simulations are excluded "
    "until liability-scale harmonization is completed."
)
lines.append("")

for n, table in [(750, top750), (5000, top5000)]:
    lines.append(f"TOP 10 CONFIGURATIONS — N={n}")
    lines.append("-" * 100)

    for rank, (_, r) in enumerate(table.head(10).iterrows(), start=1):
        lines.append(
            f"{rank}. {r['method']} | {r['config_id']} | "
            f"MAE={fmt(r['MAE'])}; "
            f"median|error|={fmt(r['MedianAbsError'])}; "
            f"bias={fmt(r['Bias'])}; "
            f"RMSE={fmt(r['RMSE'])}; "
            f"within 0.05={fmt(r['PctWithin0.05'],1)}%; "
            f"coverage={fmt(100*r['coverage'],1)}%"
        )

    lines.append("")

lines.append("TOP 10 CONFIGURATIONS — OVERALL EQUAL-WEIGHTED ACROSS N")
lines.append("-" * 100)

for rank, (_, r) in enumerate(topoverall.head(10).iterrows(), start=1):
    lines.append(
        f"{rank}. {r['method']} | {r['config_id']} | "
        f"overall MAE={fmt(r['Overall_MAE_equalN'])}; "
        f"MAE N750={fmt(r['MAE_N750'])}; "
        f"MAE N5000={fmt(r['MAE_N5000'])}; "
        f"mean within 0.05={fmt(r['MeanPctWithin0.05'],1)}%; "
        f"mean fold SD={fmt(r['MeanMedianFoldSD'])}"
    )

lines.append("")
lines.append("BEST CONFIGURATION WITHIN EACH METHOD")
lines.append("-" * 100)

for _, r in best.iterrows():
    lines.append(
        f"{r['method']}: {r['config_id']} | "
        f"overall MAE={fmt(r['Overall_MAE_equalN'])}; "
        f"N750={fmt(r['MAE_N750'])}; "
        f"N5000={fmt(r['MAE_N5000'])}"
    )

lines.append("")
lines.append(
    "IMPORTANT: configuration ranking is based on accuracy against known "
    "simulation truth, not on stability alone. A configuration can be very "
    "stable but systematically biased."
)

summary = "\n".join(lines)

Path("Method20B_MAIN_RESULTS.txt").write_text(
    summary,
    encoding="utf-8"
)

print(summary)

print()
print("=" * 110)
print("AUDIT")
print("=" * 110)
print("Continuous units:", len(x))
print("Configurations by N:")
print(
    by_n.groupby("sample_size")["config_id"]
    .nunique()
    .to_string()
)
print()
print("Eligible configs (coverage >= 80%) by N:")
print(
    by_n[by_n["eligible_for_rank"]]
    .groupby("sample_size")["config_id"]
    .nunique()
    .to_string()
)
print()
print("DONE.")
 