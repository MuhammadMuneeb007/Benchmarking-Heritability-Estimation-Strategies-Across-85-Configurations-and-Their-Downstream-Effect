 
# -*- coding: utf-8 -*-

"""
Method23-ParticipantOverlapAudit.py

Purpose
-------
Address the reviewer question:
"were the different empirical phenotypes analysed using the same subset of individuals?"

This script:
1) identifies one top-level analytic .fam file per phenotype;
2) extracts participant IDs from FID/IID;
3) reports sample size per phenotype;
4) computes pairwise overlap counts and Jaccard similarity;
5) reports the intersection shared across ALL phenotypes and the union.

No FDR correction is needed because this is a descriptive sample-composition
audit, not a family of hypothesis tests.

Run
---
python Method23-ParticipantOverlapAudit.py | tee Method23_output.txt

Outputs
-------
Method23_selected_fam_files.csv
Method23_pairwise_overlap.csv
Method23_pairwise_jaccard.csv
Method23_MAIN_RESULTS.txt
"""

from pathlib import Path
import itertools
import numpy as np
import pandas as pd

ROOT = Path(".").resolve()

PHENOTYPES = {
    "asthma": "Asthma",
    "body_mass_index_bmi": "BMI",
    "depression": "Depression",
    "gastro_oesophageal_reflux_gord_gastric_reflux": "GORD",
    "high_cholesterol": "High cholesterol",
    "hypertension": "Hypertension",
    "hypothyroidism_myxoedema": "Hypothyroidism",
    "irritable_bowel_syndrome": "IBS",
    "migraine": "Migraine",
    "osteoarthritis": "Osteoarthritis",
}


def read_fam(path):
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        dtype=str,
        engine="python",
    )
    if df.shape[1] < 2:
        raise ValueError("FAM has fewer than 2 columns")
    ids = set(zip(df.iloc[:, 0], df.iloc[:, 1]))
    return df, ids


def score_fam(path):
    """
    Prefer top-level QC/analysis FAM with ~full cohort.
    Avoid fold files and obvious raw/unphenotyped FAM where possible.
    """
    name = path.name.lower()
    score = 0

    if "qc" in name:
        score += 100
    if "final" in name:
        score += 40
    if "analysis" in name:
        score += 30
    if "clean" in name:
        score += 20

    try:
        df, ids = read_fam(path)
        score += min(len(ids), 1000) / 1000

        if df.shape[1] >= 6:
            ph = df.iloc[:, 5].astype(str)
            usable = (~ph.isin(["-9", "0", "NA", "nan"])).mean()
            score += 25 * usable
    except Exception:
        score -= 1000

    return score


def select_fam(folder):
    # Top-level only: do not accidentally use Fold_0 etc.
    candidates = sorted(folder.glob("*.fam"))

    if not candidates:
        return None

    scored = sorted(
        [(score_fam(p), p) for p in candidates],
        key=lambda x: x[0],
        reverse=True,
    )

    return scored[0][1]


selected_rows = []
id_sets = {}

print("=" * 100)
print("METHOD 23 — EMPIRICAL PARTICIPANT OVERLAP AUDIT")
print("=" * 100)

for folder_name, label in PHENOTYPES.items():
    folder = ROOT / folder_name

    if not folder.is_dir():
        print(f"[MISSING DIR] {label}: {folder}")
        continue

    fam = select_fam(folder)

    if fam is None:
        print(f"[NO FAM] {label}")
        continue

    df, ids = read_fam(fam)
    id_sets[label] = ids

    phenotype_counts = ""
    if df.shape[1] >= 6:
        phenotype_counts = str(
            df.iloc[:, 5].value_counts(dropna=False).to_dict()
        )

    selected_rows.append({
        "phenotype": label,
        "folder": folder_name,
        "selected_fam": str(fam),
        "N_unique_IDs": len(ids),
        "phenotype_code_counts": phenotype_counts,
    })

    print(
        f"{label}: N={len(ids)} | {fam.name} | "
        f"phenotype codes={phenotype_counts}"
    )

selected_df = pd.DataFrame(selected_rows)
selected_df.to_csv(
    "Method23_selected_fam_files.csv",
    index=False
)

labels = list(id_sets.keys())

overlap = pd.DataFrame(
    index=labels,
    columns=labels,
    dtype=float,
)
jaccard = pd.DataFrame(
    index=labels,
    columns=labels,
    dtype=float,
)

pair_rows = []

for a in labels:
    for b in labels:
        A = id_sets[a]
        B = id_sets[b]
        inter = len(A & B)
        union = len(A | B)

        overlap.loc[a, b] = inter
        jaccard.loc[a, b] = inter / union if union else np.nan

for a, b in itertools.combinations(labels, 2):
    A = id_sets[a]
    B = id_sets[b]
    inter = len(A & B)
    union = len(A | B)

    pair_rows.append({
        "phenotype_A": a,
        "N_A": len(A),
        "phenotype_B": b,
        "N_B": len(B),
        "intersection": inter,
        "pct_of_A_shared": 100 * inter / len(A) if A else np.nan,
        "pct_of_B_shared": 100 * inter / len(B) if B else np.nan,
        "pct_of_smaller_set_shared": (
            100 * inter / min(len(A), len(B))
            if min(len(A), len(B)) else np.nan
        ),
        "jaccard": inter / union if union else np.nan,
    })

pair_df = pd.DataFrame(pair_rows)

pair_df.to_csv(
    "Method23_pairwise_overlap.csv",
    index=False
)

overlap.to_csv(
    "Method23_pairwise_overlap_matrix.csv"
)

jaccard.to_csv(
    "Method23_pairwise_jaccard.csv"
)

if id_sets:
    shared_all = set.intersection(*id_sets.values())
    union_all = set.union(*id_sets.values())
else:
    shared_all = set()
    union_all = set()

if not pair_df.empty:
    min_shared = pair_df["pct_of_smaller_set_shared"].min()
    max_shared = pair_df["pct_of_smaller_set_shared"].max()
    min_j = pair_df["jaccard"].min()
    max_j = pair_df["jaccard"].max()
else:
    min_shared = max_shared = min_j = max_j = np.nan

lines = []
lines.append("METHOD 23 — PARTICIPANT OVERLAP RESULTS")
lines.append("=" * 90)
lines.append("")
lines.append(f"Phenotypes with usable FAM files: {len(id_sets)}/{len(PHENOTYPES)}")
lines.append(f"Participants present in ALL analysed phenotypes: {len(shared_all)}")
lines.append(f"Unique participants in the UNION of phenotypes: {len(union_all)}")
lines.append(
    f"Pairwise overlap relative to the smaller phenotype sample: "
    f"{min_shared:.1f}% to {max_shared:.1f}%"
)
lines.append(
    f"Pairwise Jaccard similarity: {min_j:.4f} to {max_j:.4f}"
)
lines.append("")
lines.append("SELECTED ANALYTIC FAM FILES")
lines.append("-" * 90)

for _, r in selected_df.iterrows():
    lines.append(
        f"{r['phenotype']}: N={int(r['N_unique_IDs'])}; "
        f"{Path(r['selected_fam']).name}"
    )

lines.append("")
lines.append(
    "No hypothesis test or FDR correction is applied because participant overlap "
    "is a descriptive property of the analytic samples."
)

summary = "\n".join(lines)
Path("Method23_MAIN_RESULTS.txt").write_text(
    summary,
    encoding="utf-8"
)

print()
print(summary)
print()
print("DONE.")
 
