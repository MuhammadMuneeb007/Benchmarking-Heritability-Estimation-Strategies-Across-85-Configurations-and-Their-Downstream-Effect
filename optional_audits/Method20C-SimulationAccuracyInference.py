 
# -*- coding: utf-8 -*-

"""
Method20C-SimulationAccuracyInference.py

Inferential companion to Method20B.

Purpose
-------
Method20B ranks configurations descriptively by MAE against known simulated h2.
This script adds formal paired statistical comparisons while respecting the fact
that all configurations are evaluated on the SAME simulated traits.

Analyses
--------
1) Configuration-level accuracy, separately at N=750 and N=5000:
   - absolute error = |estimated h2 - true h2|
   - global Friedman test across configurations
   - all pairwise two-sided Wilcoxon signed-rank tests
   - Benjamini-Hochberg FDR correction across ALL pairwise comparisons within N
   - median paired absolute-error difference as an effect size

2) Overall configuration-level accuracy:
   - first average absolute error across N=750 and N=5000 within each underlying
     simulation scenario, so the nested sample sizes are NOT treated as 100
     independent traits
   - Friedman + all-pairwise Wilcoxon + BH-FDR

3) Method-level accuracy:
   - within each simulation, average absolute error across configurations belonging
     to the same method
   - Friedman + all-pairwise Wilcoxon + BH-FDR
   - performed separately for N=750, N=5000, and overall paired scenarios

Interpretation
--------------
Lower absolute error = closer to known truth.

FDR correction is appropriate HERE because multiple formal hypothesis tests are
performed. FDR is NOT needed merely to calculate/rank MAE in Method20B.

Input
-----
Method21_unit_stability_simulated.csv

Run
---
python Method20C-SimulationAccuracyInference.py | tee Method20C_output.txt

Outputs
-------
Method20C_config_pairwise_N750.csv
Method20C_config_pairwise_N5000.csv
Method20C_config_pairwise_overall.csv
Method20C_method_pairwise_N750.csv
Method20C_method_pairwise_N5000.csv
Method20C_method_pairwise_overall.csv
Method20C_global_tests.csv
Method20C_top_config_significance.csv
Method20C_method_accuracy_summary.csv
Method20C_MAIN_RESULTS.txt
"""

from pathlib import Path
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from scipy.stats import friedmanchisquare, wilcoxon
except Exception as e:
    raise SystemExit("ERROR: scipy is required: " + str(e))

ROOT = Path(".").resolve()
INPUT = ROOT / "Method21_unit_stability_simulated.csv"
MIN_COVERAGE = 0.80


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    q = np.full(len(p), np.nan)

    ok = np.where(np.isfinite(p))[0]
    if len(ok) == 0:
        return q

    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]

    adj = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)

    q[ok[order]] = adj
    return q


def safe_wilcoxon(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a = a[keep]
    b = b[keep]

    if len(a) < 5:
        return np.nan, len(a)

    diff = a - b

    if np.allclose(diff, 0):
        return 1.0, len(a)

    try:
        p = wilcoxon(
            a,
            b,
            alternative="two-sided",
            zero_method="wilcox",
            correction=False,
            method="auto",
        ).pvalue
        return float(p), len(a)
    except Exception:
        return np.nan, len(a)


def friedman_from_matrix(mat, label):
    """
    mat: rows = paired simulation units, cols = configs/methods.
    Uses only rows complete across all included columns.
    """
    complete = mat.dropna(axis=0, how="any")

    rec = {
        "analysis": label,
        "n_paired_units": len(complete),
        "n_groups": mat.shape[1],
        "friedman_chi2": np.nan,
        "friedman_p": np.nan,
    }

    if len(complete) < 5 or mat.shape[1] < 3:
        return rec

    arrays = [complete[c].values for c in complete.columns]

    try:
        stat, p = friedmanchisquare(*arrays)
        rec["friedman_chi2"] = float(stat)
        rec["friedman_p"] = float(p)
    except Exception:
        pass

    return rec


def pairwise_tests(mat, group_type, analysis_label):
    """
    All-pairwise paired Wilcoxon. BH-FDR is applied over the full family.
    Effect is median(error_A - error_B):
      negative => A has lower error / is more accurate than B
      positive => B is more accurate than A
    """
    rows = []
    cols = list(mat.columns)

    for a, b in itertools.combinations(cols, 2):
        z = mat[[a, b]].dropna()

        if len(z) < 5:
            continue

        p, n = safe_wilcoxon(z[a].values, z[b].values)
        diff = z[a] - z[b]

        rows.append({
            "analysis": analysis_label,
            "group_type": group_type,
            "group_A": a,
            "group_B": b,
            "n_paired": n,
            "mean_abs_error_A": float(z[a].mean()),
            "mean_abs_error_B": float(z[b].mean()),
            "median_abs_error_A": float(z[a].median()),
            "median_abs_error_B": float(z[b].median()),
            "median_delta_A_minus_B": float(diff.median()),
            "mean_delta_A_minus_B": float(diff.mean()),
            "pct_A_lower_error": 100.0 * float((diff < 0).mean()),
            "p": p,
        })

    out = pd.DataFrame(rows)

    if not out.empty:
        out["q_BH_FDR"] = bh_fdr(out["p"].values)
        out["significant_FDR_0_05"] = out["q_BH_FDR"] < 0.05

    return out


def config_label(method, config):
    return f"{method} | {config}"


def make_overall_scenario_key(df):
    """
    Prefer scenario_key created by Method21. If absent, construct a conservative
    key from known design variables.
    """
    if "scenario_key" in df.columns:
        k = df["scenario_key"].fillna("").astype(str)
        if (k.str.strip() != "").any():
            return k

    needed = ["true_h2", "causal_snps", "replicate"]
    if all(c in df.columns for c in needed):
        return (
            "h2=" + df["true_h2"].astype(str)
            + "|causal=" + df["causal_snps"].astype(str)
            + "|rep=" + df["replicate"].astype(str)
        )

    raise SystemExit(
        "ERROR: Cannot identify paired N=750/N=5000 simulation scenarios. "
        "Need scenario_key or true_h2/causal_snps/replicate columns."
    )


if not INPUT.exists():
    raise SystemExit(
        "ERROR: Method21_unit_stability_simulated.csv not found. "
        "Run corrected Method21 first."
    )

d = pd.read_csv(INPUT)

required = [
    "sim_id", "trait_type", "sample_size", "true_h2",
    "method", "config_id", "h2_mean"
]

missing = [c for c in required if c not in d.columns]
if missing:
    raise SystemExit("ERROR: Missing columns: " + ", ".join(missing))

# Continuous only: estimate and truth currently share the same scale.
x = d[d["trait_type"].astype(str).str.lower().eq("continuous")].copy()

for c in ["sample_size", "true_h2", "h2_mean"]:
    x[c] = pd.to_numeric(x[c], errors="coerce")

x = x[
    np.isfinite(x["true_h2"])
    & np.isfinite(x["h2_mean"])
].copy()

x["abs_error"] = (x["h2_mean"] - x["true_h2"]).abs()
x["config_label"] = [
    config_label(m, c)
    for m, c in zip(x["method"], x["config_id"])
]
x["scenario_key_for_pairing"] = make_overall_scenario_key(x)

print("=" * 110)
print("METHOD 20C — SIMULATION ACCURACY INFERENCE WITH BH-FDR")
print("=" * 110)
print("Continuous configuration × simulation units:", len(x))
print("Configurations:", x["config_label"].nunique())
print("Methods:", x["method"].nunique())
print()

global_rows = []
config_pairwise_outputs = {}
method_pairwise_outputs = {}

# =============================================================================
# CONFIGURATION-LEVEL ANALYSIS BY N
# =============================================================================

for N in [750, 5000]:
    z = x[x["sample_size"] == N].copy()

    expected = z["sim_id"].nunique()
    coverage = (
        z.groupby("config_label")["sim_id"].nunique()
        / expected
    )
    keep_configs = coverage[coverage >= MIN_COVERAGE].index

    z = z[z["config_label"].isin(keep_configs)].copy()

    mat = z.pivot_table(
        index="sim_id",
        columns="config_label",
        values="abs_error",
        aggfunc="mean",
    )

    label = f"configuration_N{N}"

    global_rows.append(friedman_from_matrix(mat, label))

    pw = pairwise_tests(
        mat,
        group_type="configuration",
        analysis_label=label,
    )

    config_pairwise_outputs[N] = pw
    pw.to_csv(
        f"Method20C_config_pairwise_N{N}.csv",
        index=False
    )

# =============================================================================
# OVERALL CONFIGURATION ANALYSIS: AVERAGE N750/N5000 WITHIN SAME SCENARIO
# =============================================================================

scenario_config = (
    x.groupby(
        ["scenario_key_for_pairing", "config_label"],
        as_index=False
    )
    .agg(
        abs_error=("abs_error", "mean"),
        n_sample_sizes=("sample_size", "nunique"),
    )
)

# Require both N values for the overall paired analysis.
scenario_config = scenario_config[
    scenario_config["n_sample_sizes"] == 2
].copy()

expected_scenarios = scenario_config[
    "scenario_key_for_pairing"
].nunique()

coverage_overall = (
    scenario_config.groupby("config_label")[
        "scenario_key_for_pairing"
    ].nunique()
    / expected_scenarios
)

keep_configs = coverage_overall[
    coverage_overall >= MIN_COVERAGE
].index

scenario_config = scenario_config[
    scenario_config["config_label"].isin(keep_configs)
].copy()

mat_overall = scenario_config.pivot_table(
    index="scenario_key_for_pairing",
    columns="config_label",
    values="abs_error",
    aggfunc="mean",
)

global_rows.append(
    friedman_from_matrix(
        mat_overall,
        "configuration_overall_equalN"
    )
)

pw_overall = pairwise_tests(
    mat_overall,
    group_type="configuration",
    analysis_label="configuration_overall_equalN",
)

pw_overall.to_csv(
    "Method20C_config_pairwise_overall.csv",
    index=False
)

# =============================================================================
# METHOD-LEVEL ANALYSIS
# Average across configurations within a method for each simulation first.
# =============================================================================

method_unit = (
    x.groupby(
        ["sim_id", "scenario_key_for_pairing", "sample_size", "method"],
        as_index=False
    )
    .agg(abs_error=("abs_error", "mean"))
)

method_accuracy_rows = []

for N in [750, 5000]:
    z = method_unit[method_unit["sample_size"] == N].copy()

    mat = z.pivot_table(
        index="sim_id",
        columns="method",
        values="abs_error",
        aggfunc="mean",
    )

    label = f"method_N{N}"

    global_rows.append(friedman_from_matrix(mat, label))

    pw = pairwise_tests(
        mat,
        group_type="method",
        analysis_label=label,
    )

    method_pairwise_outputs[N] = pw
    pw.to_csv(
        f"Method20C_method_pairwise_N{N}.csv",
        index=False
    )

    for method, g in z.groupby("method"):
        method_accuracy_rows.append({
            "analysis": label,
            "sample_size": N,
            "method": method,
            "n_simulations": g["sim_id"].nunique(),
            "mean_abs_error": float(g["abs_error"].mean()),
            "median_abs_error": float(g["abs_error"].median()),
        })

# Overall method-level: average N within paired scenario.
method_overall = (
    method_unit.groupby(
        ["scenario_key_for_pairing", "method"],
        as_index=False
    )
    .agg(
        abs_error=("abs_error", "mean"),
        n_sample_sizes=("sample_size", "nunique"),
    )
)

method_overall = method_overall[
    method_overall["n_sample_sizes"] == 2
].copy()

mat_method_overall = method_overall.pivot_table(
    index="scenario_key_for_pairing",
    columns="method",
    values="abs_error",
    aggfunc="mean",
)

global_rows.append(
    friedman_from_matrix(
        mat_method_overall,
        "method_overall_equalN"
    )
)

pw_method_overall = pairwise_tests(
    mat_method_overall,
    group_type="method",
    analysis_label="method_overall_equalN",
)

pw_method_overall.to_csv(
    "Method20C_method_pairwise_overall.csv",
    index=False
)

for method, g in method_overall.groupby("method"):
    method_accuracy_rows.append({
        "analysis": "method_overall_equalN",
        "sample_size": "overall",
        "method": method,
        "n_simulations": g["scenario_key_for_pairing"].nunique(),
        "mean_abs_error": float(g["abs_error"].mean()),
        "median_abs_error": float(g["abs_error"].median()),
    })

method_summary = pd.DataFrame(method_accuracy_rows)
method_summary = method_summary.sort_values(
    ["analysis", "mean_abs_error"]
)
method_summary.to_csv(
    "Method20C_method_accuracy_summary.csv",
    index=False
)

global_df = pd.DataFrame(global_rows)
global_df["global_q_BH_FDR"] = bh_fdr(
    global_df["friedman_p"].values
)
global_df.to_csv(
    "Method20C_global_tests.csv",
    index=False
)

# =============================================================================
# TOP-CONFIG SIGNIFICANCE SUMMARY
# Use all-pairwise FDR results, not post-hoc unadjusted best-vs-rest tests.
# =============================================================================

top_rows = []

def summarize_top(mat, pairwise_df, label):
    means = mat.mean(axis=0).sort_values()
    if len(means) == 0:
        return

    best = means.index[0]

    relevant = pairwise_df[
        (pairwise_df["group_A"] == best)
        | (pairwise_df["group_B"] == best)
    ].copy()

    sig_better = 0
    sig_worse = 0
    not_sig = 0

    for _, r in relevant.iterrows():
        if not bool(r.get("significant_FDR_0_05", False)):
            not_sig += 1
            continue

        # delta = error_A - error_B
        delta = r["median_delta_A_minus_B"]

        if r["group_A"] == best:
            if delta < 0:
                sig_better += 1
            elif delta > 0:
                sig_worse += 1
            else:
                not_sig += 1
        else:
            # best is B. Positive A-B means B has lower error.
            if delta > 0:
                sig_better += 1
            elif delta < 0:
                sig_worse += 1
            else:
                not_sig += 1

    top_rows.append({
        "analysis": label,
        "best_configuration": best,
        "mean_abs_error": float(means.iloc[0]),
        "comparators": len(relevant),
        "significantly_better_than_n_FDR": sig_better,
        "significantly_worse_than_n_FDR": sig_worse,
        "not_significantly_different_n_FDR": not_sig,
    })

for N in [750, 5000]:
    z = x[x["sample_size"] == N]
    mat = z.pivot_table(
        index="sim_id",
        columns="config_label",
        values="abs_error",
        aggfunc="mean",
    )
    summarize_top(
        mat,
        config_pairwise_outputs[N],
        f"configuration_N{N}",
    )

summarize_top(
    mat_overall,
    pw_overall,
    "configuration_overall_equalN",
)

top_df = pd.DataFrame(top_rows)
top_df.to_csv(
    "Method20C_top_config_significance.csv",
    index=False
)

# =============================================================================
# TEXT SUMMARY
# =============================================================================

lines = []
lines.append("METHOD 20C — SIMULATION ACCURACY INFERENCE WITH BH-FDR")
lines.append("=" * 100)
lines.append("")
lines.append(
    "Formal hypothesis testing was performed on paired absolute errors against known truth."
)
lines.append(
    "All pairwise Wilcoxon signed-rank tests within each analysis family were corrected using Benjamini-Hochberg FDR."
)
lines.append(
    "The overall N analysis first averaged N=750 and N=5000 within the same underlying simulation scenario, avoiding treatment of nested sample sizes as independent traits."
)
lines.append("")

lines.append("GLOBAL FRIEDMAN TESTS")
lines.append("-" * 100)
for _, r in global_df.iterrows():
    lines.append(
        f"{r['analysis']}: n={int(r['n_paired_units'])}, "
        f"groups={int(r['n_groups'])}, "
        f"chi2={r['friedman_chi2']:.4f}, "
        f"p={r['friedman_p']:.4g}, "
        f"FDR q={r['global_q_BH_FDR']:.4g}"
    )
lines.append("")

lines.append("TOP CONFIGURATION — FDR-CORRECTED PAIRWISE SUPPORT")
lines.append("-" * 100)
for _, r in top_df.iterrows():
    lines.append(
        f"{r['analysis']}: {r['best_configuration']} | "
        f"mean absolute error={r['mean_abs_error']:.4f}; "
        f"significantly more accurate than "
        f"{int(r['significantly_better_than_n_FDR'])}/"
        f"{int(r['comparators'])} comparators after BH-FDR; "
        f"not significantly different from "
        f"{int(r['not_significantly_different_n_FDR'])}."
    )
lines.append("")

lines.append("METHOD-LEVEL MEAN ABSOLUTE ERROR")
lines.append("-" * 100)
for analysis, g in method_summary.groupby("analysis", sort=False):
    lines.append(analysis)
    for _, r in g.sort_values("mean_abs_error").iterrows():
        lines.append(
            f"  {r['method']}: mean |error|={r['mean_abs_error']:.4f}; "
            f"median |error|={r['median_abs_error']:.4f}"
        )
lines.append("")
lines.append(
    "Interpretation: Method20B provides the descriptive ranking; Method20C provides inferential support for differences in paired accuracy. "
    "FDR correction should be cited for Method20C and the exact one-factor matched analysis, not for the simple MAE ranking itself."
)

summary = "\n".join(lines)
Path("Method20C_MAIN_RESULTS.txt").write_text(summary, encoding="utf-8")

print(summary)
print()
print("Outputs written successfully.")
 