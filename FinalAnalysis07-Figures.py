# -*- coding: utf-8 -*-

"""
FinalAnalysis08-PrimaryFigures-Uncertainty.py

Purpose
-------
Create THREE manuscript-quality figures using the PRIMARY reporting scale:
- binary phenotypes: LIABILITY scale
- BMI: NATIVE / OBSERVED quantitative scale

Figures created
---------------
1. Figure08_Heatmap_PrimaryScale.png
   - Panel A: mean h2 heatmap (primary scale)
   - Panel B: median estimator SE heatmap (same rows/columns)

2. Figure08_Distribution_Uncertainty.png
   - Panel A: distribution of mean h2 across configurations by method
   - Panel B: method-level median h2 with uncertainty bars
              using median estimator 95% CI half-width
   - Panel C: method-level median estimator SE

3. Figure08_InterMethodCorrelation.png
   - Pearson inter-method correlation heatmap
   - computed using method-level phenotype profiles on the primary scale

Exports
-------
- Figure08_Heatmap_PrimaryScale.png / .pdf
- Figure08_Distribution_Uncertainty.png / .pdf
- Figure08_InterMethodCorrelation.png / .pdf
- Figure08_PrimaryScale_ConfigTable.csv
- Figure08_MethodSummary.csv
- Figure08_CorrelationMatrix_Pearson.csv
- Figure08_FigureCaptions.txt

Input
-----
Requires:
    FinalAnalysis06-UncertaintyReliability/
        FinalAnalysis06_unit_stability.csv
        FinalAnalysis06_raw_scale_rows.csv

Run
---
cd /data/ascher01/uqmmune1/finalheritability
python FinalAnalysis08-PrimaryFigures-Uncertainty.py

Notes
-----
- Matplotlib only. No seaborn.
- Negative h2 and h2 > 1 are retained.
- Estimator SE / CI are taken from the underlying tool output where available.
- Cross-fold SD is NOT used as estimator SE.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# =============================================================================
# PATHS
# =============================================================================

def find_root() -> Path:
    env = os.environ.get("HERITABILITY_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "FinalAnalysis06-UncertaintyReliability").exists():
            return p

    candidates = [
        Path.cwd().resolve(),
        Path.cwd().resolve().parent,
        Path("/data/ascher01/uqmmune1/finalheritability"),
    ]

    for p in candidates:
        if (p / "FinalAnalysis06-UncertaintyReliability").exists():
            return p

    raise FileNotFoundError(
        "Could not locate FinalAnalysis06-UncertaintyReliability. "
        "Run from finalheritability or set HERITABILITY_ROOT."
    )


ROOT = find_root()
IN_DIR = ROOT / "FinalAnalysis06-UncertaintyReliability"
OUT_DIR = ROOT / "FinalAnalysis08-PrimaryFigures-Uncertainty"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# GLOBALS
# =============================================================================

PHENOTYPE_ORDER = [
    "Asthma",
    "Body Mass Index",
    "Depression",
    "Gastro-Oesophageal Reflux",
    "High Cholesterol",
    "Hypertension",
    "Hypothyroidism",
    "Irritable Bowel Syndrome",
    "Migraine",
    "Osteoarthritis",
]

METHOD_ORDER = [
    "LDpred-2-Heritability",
    "GCTA-Heritability",
    "Gemma1-Heritability",
    "Gemma2-Heritability",
    "Gemma-LDSC-Heritability",
    "DPR-Gemma-Heritability",
    "LDSC-Heritability",
    "LDAK-Calculated-Heritability",
    "LDAK-Precomputed-Heritability",
    "LDSC-Genotype-Heritability",
]

METHOD_SHORT = {
    "LDpred-2-Heritability": "LDpred2",
    "GCTA-Heritability": "GCTA",
    "Gemma1-Heritability": "GEMMA1",
    "Gemma2-Heritability": "GEMMA2",
    "Gemma-LDSC-Heritability": "GEMMA-LDSC",
    "DPR-Gemma-Heritability": "DPR-GEMMA",
    "LDSC-Heritability": "LDSC",
    "LDAK-Calculated-Heritability": "LDAK-Calc",
    "LDAK-Precomputed-Heritability": "LDAK-Pre",
    "LDSC-Genotype-Heritability": "LDSC-Gen",
}


# =============================================================================
# HELPERS
# =============================================================================

def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return path


def to_num(x):
    try:
        y = float(x)
        return y if np.isfinite(y) else np.nan
    except Exception:
        return np.nan


def parse_config_id(config_id: str):
    """
    Parse visible configuration fields from ConfigID.
    """
    text = str(config_id)
    fields = {}
    for piece in text.split("|"):
        piece = piece.strip()
        if "=" in piece:
            k, v = piece.split("=", 1)
            fields[k.strip().lower()] = v.strip()

    h2model = (
        fields.get("h2model_standardized")
        or fields.get("h2model")
        or fields.get("model")
        or ""
    )
    clumpprune = fields.get("clumpprune") or fields.get("clump") or ""
    numberofpca = fields.get("numberofpca") or fields.get("number_of_pca") or fields.get("pca") or ""

    if not h2model:
        skip = {"clumpprune", "clump", "numberofpca", "number_of_pca", "pca"}
        rest = []
        for k, v in fields.items():
            if k in skip:
                continue
            rest.append(f"{k}={v}")
        h2model = "; ".join(rest)

    return h2model, clumpprune, numberofpca


def safe_rank_order(df: pd.DataFrame, column: str, order: list[str]):
    df[column] = pd.Categorical(df[column], categories=order, ordered=True)
    return df.sort_values(column, kind="stable")


# =============================================================================
# LOAD + BUILD PRIMARY SCALE TABLE
# =============================================================================

def load_primary_long():
    units = pd.read_csv(require(IN_DIR / "FinalAnalysis06_unit_stability.csv"))
    raw = pd.read_csv(require(IN_DIR / "FinalAnalysis06_raw_scale_rows.csv"))

    # --------------------
    # primary scale
    # binary => liability
    # continuous BMI => native quantitative
    # --------------------
    u = units[units["Domain"] == "empirical"].copy()
    u["Phenotype"] = u["DatasetLabel"].astype(str)

    mask_u = (
        ((u["TraitType"] == "binary") & (u["Scale"] == "liability_binary"))
        |
        ((u["TraitType"] == "continuous") & (u["Scale"] == "native_quantitative"))
    )
    u = u[mask_u].copy()

    r = raw[raw["Domain"] == "empirical"].copy()
    r["Phenotype"] = r["DatasetLabel"].astype(str)

    mask_r = (
        ((r["TraitType"] == "binary") & (r["Scale"] == "liability_binary"))
        |
        ((r["TraitType"] == "continuous") & (r["Scale"] == "native_quantitative"))
    )
    r = r[mask_r].copy()

    for c in ["ToolSE", "ToolCILower", "ToolCIUpper"]:
        if c in r.columns:
            r[c] = pd.to_numeric(r[c], errors="coerce")
        else:
            r[c] = np.nan

    uncertainty = (
        r.groupby(["Phenotype", "Method", "ConfigID", "Scale"], dropna=False)
         .agg(
             EstimatorSE=("ToolSE", "median"),
             EstimatorSE_n=("ToolSE", lambda z: int(pd.Series(z).notna().sum())),
             EstimatorCILower=("ToolCILower", "median"),
             EstimatorCIUpper=("ToolCIUpper", "median"),
             EstimatorCI_n=("ToolCILower", lambda z: int(pd.Series(z).notna().sum())),
         )
         .reset_index()
    )
    uncertainty["EstimatorCIWidth"] = uncertainty["EstimatorCIUpper"] - uncertainty["EstimatorCILower"]
    uncertainty["EstimatorCIHalfWidth"] = uncertainty["EstimatorCIWidth"] / 2.0

    keep_cols = [
        "Phenotype", "TraitType", "Scale", "Method", "MethodFamily", "ConfigID",
        "ValidH2Folds", "MeanH2", "MedianH2", "MinH2", "MaxH2",
        "FoldSD", "FoldMAD", "FoldIQR", "FoldRange"
    ]
    keep_cols = [c for c in keep_cols if c in u.columns]

    long = (
        u[keep_cols]
        .merge(uncertainty, on=["Phenotype", "Method", "ConfigID", "Scale"], how="left")
        .copy()
    )

    parsed = long["ConfigID"].apply(parse_config_id)
    long["h2model"] = [x[0] for x in parsed]
    long["clumpprune"] = [x[1] for x in parsed]
    long["numberofpca"] = [x[2] for x in parsed]

    configs = (
        long[["Method", "ConfigID", "h2model", "clumpprune", "numberofpca"]]
        .drop_duplicates()
        .copy()
    )
    configs = safe_rank_order(configs, "Method", METHOD_ORDER)
    configs = configs.sort_values(
        ["Method", "h2model", "clumpprune", "numberofpca", "ConfigID"],
        kind="stable"
    ).reset_index(drop=True)

    configs["Configuration"] = np.arange(1, len(configs) + 1)

    long = long.merge(
        configs[["Method", "ConfigID", "Configuration"]],
        on=["Method", "ConfigID"],
        how="left"
    )

    long = safe_rank_order(long, "Method", METHOD_ORDER)
    long["Phenotype"] = pd.Categorical(long["Phenotype"], categories=PHENOTYPE_ORDER, ordered=True)
    long = long.sort_values(["Configuration", "Phenotype"], kind="stable").reset_index(drop=True)

    return long, configs


# =============================================================================
# EXPORT TABLES
# =============================================================================

def export_tables(long, configs):
    export_cols = [
        "Configuration", "Method", "h2model", "clumpprune", "numberofpca",
        "Phenotype", "TraitType", "Scale", "MeanH2", "FoldSD",
        "EstimatorSE", "EstimatorCILower", "EstimatorCIUpper",
        "EstimatorCIWidth", "EstimatorCIHalfWidth", "ValidH2Folds"
    ]
    export_cols = [c for c in export_cols if c in long.columns]

    table = long[export_cols].copy()
    table.to_csv(OUT_DIR / "Figure08_PrimaryScale_ConfigTable.csv", index=False)

    summary = (
        long.groupby("Method", dropna=False)
            .agg(
                N=("MeanH2", lambda z: int(pd.Series(z).notna().sum())),
                MedianH2=("MeanH2", "median"),
                MeanH2=("MeanH2", "mean"),
                SDofMeans=("MeanH2", "std"),
                MedianEstimatorSE=("EstimatorSE", "median"),
                MedianEstimatorCIWidth=("EstimatorCIWidth", "median"),
                MedianEstimatorCIHalfWidth=("EstimatorCIHalfWidth", "median"),
                MedianFoldSD=("FoldSD", "median"),
                EstimatorSECoverage=("EstimatorSE", lambda z: 100 * pd.Series(z).notna().mean()),
            )
            .reset_index()
    )
    summary = safe_rank_order(summary, "Method", METHOD_ORDER)
    summary.to_csv(OUT_DIR / "Figure08_MethodSummary.csv", index=False)

    return table, summary


# =============================================================================
# FIGURE 1: HEATMAP
# =============================================================================

def draw_heatmap(ax, mat, row_labels, col_labels, title, cbar_label):
    # Replace all-NA rows with nan; imshow handles.
    arr = np.asarray(mat, dtype=float)
    im = ax.imshow(arr, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label(cbar_label)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title(title)
    ax.set_xlabel("Phenotype")
    ax.set_ylabel("Configuration")

    # light gridlines
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.2, alpha=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def figure_heatmap(long, configs):
    mean_pivot = (
        long.pivot_table(
            index="Configuration",
            columns="Phenotype",
            values="MeanH2",
            aggfunc="first"
        )
        .reindex(index=configs["Configuration"], columns=PHENOTYPE_ORDER)
    )

    se_pivot = (
        long.pivot_table(
            index="Configuration",
            columns="Phenotype",
            values="EstimatorSE",
            aggfunc="first"
        )
        .reindex(index=configs["Configuration"], columns=PHENOTYPE_ORDER)
    )

    row_labels = [
        f"{i:02d}. {METHOD_SHORT.get(m, m)}"
        for i, m in zip(configs["Configuration"], configs["Method"])
    ]

    fig = plt.figure(figsize=(14, 18))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.22)

    ax1 = fig.add_subplot(gs[0])
    draw_heatmap(
        ax1,
        mean_pivot.to_numpy(),
        row_labels,
        PHENOTYPE_ORDER,
        "A. Mean SNP heritability on the primary reporting scale\n(binary phenotypes on liability scale; BMI on native quantitative scale)",
        r"Mean $h^2$"
    )

    ax2 = fig.add_subplot(gs[1])
    draw_heatmap(
        ax2,
        se_pivot.to_numpy(),
        row_labels,
        PHENOTYPE_ORDER,
        "B. Median estimator-specific standard error\n(NA values indicate tool-specific SE was unavailable)",
        "Median estimator SE"
    )

    fig.savefig(OUT_DIR / "Figure08_Heatmap_PrimaryScale.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Figure08_Heatmap_PrimaryScale.pdf", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# FIGURE 2: DISTRIBUTION + UNCERTAINTY
# =============================================================================

def figure_distribution_uncertainty(long, method_summary):
    methods = [m for m in METHOD_ORDER if m in set(long["Method"])]
    x = np.arange(len(methods))

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.45, 1.0, 0.85], hspace=0.32)

    # -------------------------------------------------------------------------
    # Panel A: distribution of point estimates by method
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])

    box_data = []
    for m in methods:
        vals = pd.to_numeric(
            long.loc[long["Method"] == m, "MeanH2"],
            errors="coerce"
        ).dropna().to_numpy()
        box_data.append(vals)

    ax1.boxplot(box_data, positions=x, widths=0.5, showfliers=False)

    rng = np.random.default_rng(20260916)
    for i, m in enumerate(methods):
        vals = pd.to_numeric(
            long.loc[long["Method"] == m, "MeanH2"],
            errors="coerce"
        ).dropna().to_numpy()
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        ax1.scatter(np.full(len(vals), i) + jitter, vals, s=12, alpha=0.55)

    ax1.axhline(0, linestyle="--", linewidth=1)
    ax1.axhline(1, linestyle=":", linewidth=1)
    ax1.set_ylabel(r"Configuration-level mean $h^2$")
    ax1.set_title("A. Distribution of primary-scale SNP-heritability estimates across configurations")
    ax1.set_xticks(x)
    ax1.set_xticklabels([METHOD_SHORT.get(m, m) for m in methods], rotation=35, ha="right")
    ax1.grid(axis="y", alpha=0.22)

    # -------------------------------------------------------------------------
    # Panel B: method-level median h2 with median estimator 95% CI half-width
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1])

    ms = method_summary.set_index("Method").reindex(methods)
    y = pd.to_numeric(ms["MedianH2"], errors="coerce").to_numpy()
    yerr = pd.to_numeric(ms["MedianEstimatorCIHalfWidth"], errors="coerce").to_numpy()

    # If CI is unavailable, plot point without error bar.
    for i, (yy, ee) in enumerate(zip(y, yerr)):
        if np.isfinite(yy) and np.isfinite(ee):
            ax2.errorbar(i, yy, yerr=ee, fmt='o', capsize=3)
        elif np.isfinite(yy):
            ax2.plot(i, yy, 'o')

    ax2.axhline(0, linestyle="--", linewidth=1)
    ax2.axhline(1, linestyle=":", linewidth=1)
    ax2.set_ylabel(r"Median configuration-level $h^2$")
    ax2.set_title("B. Method-level median heritability with median estimator-specific 95% CI half-width")
    ax2.set_xticks(x)
    ax2.set_xticklabels([METHOD_SHORT.get(m, m) for m in methods], rotation=35, ha="right")
    ax2.grid(axis="y", alpha=0.22)

    # -------------------------------------------------------------------------
    # Panel C: method-level median estimator SE
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[2])
    y3 = pd.to_numeric(ms["MedianEstimatorSE"], errors="coerce").to_numpy()
    ax3.plot(x, y3, marker='o')
    ax3.set_ylabel("Median estimator SE")
    ax3.set_title("C. Method-level median estimator-specific standard error")
    ax3.set_xticks(x)
    ax3.set_xticklabels([METHOD_SHORT.get(m, m) for m in methods], rotation=35, ha="right")
    ax3.grid(axis="y", alpha=0.22)
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=6))

    fig.savefig(OUT_DIR / "Figure08_Distribution_Uncertainty.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Figure08_Distribution_Uncertainty.pdf", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# FIGURE 3: INTER-METHOD CORRELATION
# =============================================================================

def figure_correlation(long):
    # Average across configurations within method and phenotype.
    prof = (
        long.groupby(["Method", "Phenotype"], dropna=False)
            .agg(ProfileH2=("MeanH2", "mean"))
            .reset_index()
    )

    pivot = (
        prof.pivot(index="Phenotype", columns="Method", values="ProfileH2")
            .reindex(index=PHENOTYPE_ORDER, columns=METHOD_ORDER)
    )

    pearson = pivot.corr(method="pearson")
    pearson.to_csv(OUT_DIR / "Figure08_CorrelationMatrix_Pearson.csv")

    arr = pearson.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 9))
    im = ax.imshow(arr, vmin=-1, vmax=1, aspect="equal")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")

    labels = [METHOD_SHORT.get(m, m) for m in pearson.columns]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    # annotate matrix
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    ax.set_title(
        "Inter-method correlation of phenotype-level heritability profiles\n"
        "(primary scale: liability for binary phenotypes; native scale for BMI)"
    )

    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.3, alpha=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.savefig(OUT_DIR / "Figure08_InterMethodCorrelation.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Figure08_InterMethodCorrelation.pdf", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# CAPTIONS
# =============================================================================

def write_captions():
    text = """
Figure 1. Primary-scale heatmap of configuration-level SNP heritability and estimator uncertainty. Panel A shows the mean SNP-heritability estimate for each of the 86 analytical configurations across the 10 empirical phenotypes, using the primary reporting scale defined as liability scale for binary phenotypes and the native quantitative scale for body mass index. Panel B shows the corresponding median estimator-specific standard error across available fold outputs. Negative estimates and estimates greater than one were retained without truncation. Empty cells indicate unavailable estimator-specific uncertainty rather than unavailable heritability estimates.

Figure 2. Distribution and uncertainty of primary-scale SNP-heritability estimates across methods. Panel A shows the configuration-level distribution of mean h2 values for each method. Panel B shows the method-level median h2 together with the median estimator-specific 95% confidence-interval half-width. Panel C shows the method-level median estimator-specific standard error. These uncertainty summaries describe the typical size of method-level estimator uncertainty across the benchmark and complement the full configuration-level values in Supplementary Table S1.

Figure 3. Inter-method correlation of phenotype-level heritability profiles on the primary scale. For each method, phenotype-level profiles were obtained by averaging configuration-level primary-scale heritability estimates across configurations within method. The heatmap shows pairwise Pearson correlations between methods across the 10 phenotypes. Correlations close to one indicate that two methods assign similar relative heritability rankings across phenotypes, whereas lower correlations indicate weaker agreement.
""".strip()
    (OUT_DIR / "Figure08_FigureCaptions.txt").write_text(text, encoding="utf-8")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*110)
    print("FinalAnalysis08 - PRIMARY FIGURES WITH UNCERTAINTY")
    print("="*110)
    print("ROOT   :", ROOT)
    print("INPUT  :", IN_DIR)
    print("OUTPUT :", OUT_DIR)
    print()

    long, configs = load_primary_long()

    print("Empirical primary-scale rows:", len(long))
    print("Unique configurations:", configs.shape[0])

    table, method_summary = export_tables(long, configs)

    print("Exported:")
    print(" ", OUT_DIR / "Figure08_PrimaryScale_ConfigTable.csv")
    print(" ", OUT_DIR / "Figure08_MethodSummary.csv")

    print("\nCreating figures ...")
    figure_heatmap(long, configs)
    figure_distribution_uncertainty(long, method_summary)
    figure_correlation(long)
    write_captions()

    print("\nDone. Created:")
    for f in [
        "Figure08_Heatmap_PrimaryScale.png",
        "Figure08_Distribution_Uncertainty.png",
        "Figure08_InterMethodCorrelation.png",
        "Figure08_PrimaryScale_ConfigTable.csv",
        "Figure08_MethodSummary.csv",
        "Figure08_CorrelationMatrix_Pearson.csv",
        "Figure08_FigureCaptions.txt",
    ]:
        print(" ", OUT_DIR / f)

    print("\nInterpretation:")
    print(" - Heatmap now uses LIABILITY scale for binary phenotypes and native BMI scale.")
    print(" - Distribution figure explicitly incorporates estimator SE and CI.")
    print(" - Correlation figure restores inter-method agreement analysis.")
    print(" - Cross-fold SD is not mislabelled as estimator SE.")


if __name__ == "__main__":
    main()
