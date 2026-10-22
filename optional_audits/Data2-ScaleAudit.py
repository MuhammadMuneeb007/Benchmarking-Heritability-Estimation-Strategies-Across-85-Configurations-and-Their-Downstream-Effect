 
"""
Data2-ScaleAudit.py

Purpose
-------
Audit the ORIGINAL heritability code and available logs to determine whether
each method produced:
    1) observed / untransformed binary-trait heritability,
    2) an explicitly liability-transformed estimate, or
    3) an estimate whose scale cannot yet be proven from the available code.

Run from the finalheritability project root:

    python Data2-ScaleAudit.py

No arguments are required.

Outputs
-------
Data2_Method_Scale_Audit.csv
Data2_Method_Scale_Audit.txt

This script DOES NOT transform any heritability values.
It only establishes what the original pipeline actually did.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path


ROOT = Path.cwd()
OUT_CSV = ROOT / "Data2_Method_Scale_Audit.csv"
OUT_TXT = ROOT / "Data2_Method_Scale_Audit.txt"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""


def compact(s: str, n: int = 220) -> str:
    s = " ".join(str(s).split())
    return s[:n]


def find_lines(text: str, patterns):
    out = []
    lines = text.splitlines()

    for i, line in enumerate(lines, 1):
        low = line.lower()
        if any(re.search(p, low, flags=re.I) for p in patterns):
            out.append((i, line.strip()))

    return out


def evidence_string(matches, limit=10):
    if not matches:
        return ""

    pieces = []
    for line_no, line in matches[:limit]:
        pieces.append(f"L{line_no}: {compact(line, 180)}")
    return " || ".join(pieces)


def method_number(path: Path):
    m = re.search(r"method\s*([0-9]+)", path.name, flags=re.I)
    if m:
        return int(m.group(1))
    return 999


def discover_method_files():
    """
    Find Method*.py scripts in the current directory.

    Also find the external LDpred2 R script if present.
    """
    py_files = sorted(
        [p for p in ROOT.glob("Method*.py") if p.is_file()],
        key=lambda p: (method_number(p), p.name.lower())
    )

    r_candidates = []

    for pattern in (
        "LDpred-2-Heritability.R",
        "*LDpred*Heritability*.R",
        "*heritability*.R",
        "*Heritability*.R",
    ):
        for p in ROOT.glob(pattern):
            if p.is_file() and p not in r_candidates:
                r_candidates.append(p)

    return py_files, r_candidates


# ---------------------------------------------------------------------
# Generic scale evidence
# ---------------------------------------------------------------------
EXPLICIT_LIABILITY_PATTERNS = [
    r"--samp-prev",
    r"--pop-prev",
    r"--prevalence",
    r"liability.scale",
    r"liability_scale",
    r"liability scale",
    r"observed.*liability",
    r"liability.*observed",
    r"samp[_\. ]?prev",
    r"pop[_\. ]?prev",
    r"population[_\. ]?prevalence",
]

EXPLICIT_OBSERVED_PATTERNS = [
    r"total observed scale h2",
    r"observed scale h2",
    r"observed-scale",
    r"observed_scale",
]

PREVALENCE_WORD_PATTERNS = [
    r"preval",
    r"liabil",
    r"samp[_\. -]?prev",
    r"pop[_\. -]?prev",
]


def contains_any(text, patterns):
    return any(re.search(p, text, flags=re.I) for p in patterns)


# ---------------------------------------------------------------------
# Classification of each uploaded/original pipeline family
# ---------------------------------------------------------------------
def classify_python_method(path: Path, text: str, r_candidates):
    name = path.name.lower()

    liability_matches = find_lines(text, EXPLICIT_LIABILITY_PATTERNS)
    observed_matches = find_lines(text, EXPLICIT_OBSERVED_PATTERNS)

    # Anything explicit wins.
    if liability_matches:
        return {
            "classification": "EXPLICIT_LIABILITY_OR_PREVALENCE_HANDLING_FOUND",
            "can_transform_again": "NO - inspect exact code first",
            "reason": (
                "The source code contains explicit prevalence/liability handling. "
                "This method must be inspected before any further transformation "
                "to avoid double conversion."
            ),
            "evidence": evidence_string(liability_matches),
        }

    # Method 7 and Method 10 explicitly parse the LDSC observed-scale line.
    if observed_matches:
        return {
            "classification": "OBSERVED_SCALE",
            "can_transform_again": "YES, for eligible binary traits",
            "reason": (
                "The script explicitly reads 'Total Observed scale h2' and no "
                "sample/population prevalence option was found."
            ),
            "evidence": evidence_string(observed_matches),
        }

    # Method 1 calls an external R script. The Python wrapper alone cannot prove
    # the scale because the R script creates the h2 output files.
    if re.search(r"method1", name):
        external_call = find_lines(
            text,
            [r"ldpred-2-heritability\.r", r"ldpred_h2"]
        )

        if not r_candidates:
            return {
                "classification": "NEEDS_EXTERNAL_R_SCRIPT_AUDIT",
                "can_transform_again": "NOT YET",
                "reason": (
                    "Method1 does not itself calculate or convert h2. It calls "
                    "LDpred-2-Heritability.R and then reads the generated h2 files. "
                    "The R script is therefore required to prove the scale."
                ),
                "evidence": evidence_string(external_call),
            }

        # If R script exists, inspect it.
        combined_r = "\n".join(read_text(p) for p in r_candidates)
        r_liab = find_lines(combined_r, EXPLICIT_LIABILITY_PATTERNS)

        if r_liab:
            return {
                "classification": "METHOD1_R_SCRIPT_HAS_PREVALENCE_OR_LIABILITY_CODE",
                "can_transform_again": "NO - inspect exact R operation first",
                "reason": (
                    "The external LDpred2 R script contains prevalence/liability "
                    "logic, so Method1 must not be blindly transformed again."
                ),
                "evidence": evidence_string(r_liab),
            }

        return {
            "classification": "NO_LIABILITY_CONVERSION_FOUND_IN_METHOD1_OR_R_SCRIPT",
            "can_transform_again": "YES, subject to confirming LDpred2 h2 definition",
            "reason": (
                "Neither the Python wrapper nor the available LDpred2 R script "
                "contains explicit population/sample prevalence or liability-scale "
                "conversion code."
            ),
            "evidence": evidence_string(external_call),
        }

    # Method 2: GCTA individual-level REML, no --prevalence option.
    if re.search(r"method2", name):
        cmd = find_lines(
            text,
            [r"\./gcta", r"--reml", r"v\(g\)/vp", r"--prevalence"]
        )
        return {
            "classification": "OBSERVED_OR_UNTRANSFORMED_BINARY_SCALE",
            "can_transform_again": "YES, for binary traits",
            "reason": (
                "GCTA REML is run on the target phenotype and the script does not "
                "supply a prevalence/liability option. The stored V(G)/Vp value "
                "therefore has not been harmonised to a population liability scale."
            ),
            "evidence": evidence_string(cmd),
        }

    # Methods 3-6: GEMMA/DPR PVE. No K/P conversion appears in the code.
    if re.search(r"method[3-6]", name):
        cmd = find_lines(
            text,
            [r"\./gemma", r"\./dpr", r"pve estimates", r"se\(pve\)", r"-vc"]
        )
        return {
            "classification": "OBSERVED_OR_UNTRANSFORMED_BINARY_SCALE",
            "can_transform_again": "YES, for binary traits",
            "reason": (
                "The pipeline reads GEMMA/DPR PVE estimates and contains no "
                "population prevalence, sample prevalence, or liability-scale "
                "transformation. Binary-trait estimates were therefore not "
                "harmonised to liability scale."
            ),
            "evidence": evidence_string(cmd),
        }

    # Old Method8 script runs LDSC and exits before the later code.
    if path.name.lower().startswith("method8("):
        cmd = find_lines(
            text,
            [r"python ldsc\.py", r"--h2", r"exit\(0\)"]
        )
        return {
            "classification": "OBSERVED_SCALE_LDSC",
            "can_transform_again": "YES, for binary traits",
            "reason": (
                "This script runs LDSC --h2 without --samp-prev/--pop-prev and "
                "then exits. No liability conversion is supplied."
            ),
            "evidence": evidence_string(cmd),
        }

    # LDAK SumHer scripts.
    if "method8-calculated" in name or "method8-precomputed" in name:
        cmd = find_lines(
            text,
            [r"--sum-hers", r"--summary", r"--tagfile", r"her_all"]
        )
        return {
            "classification": "UNTRANSFORMED_SUMMARY_STATISTIC_H2",
            "can_transform_again": "YES, for binary traits",
            "reason": (
                "LDAK SumHer is run without any phenotype-specific population "
                "prevalence or sample prevalence in the supplied code. Therefore "
                "the result has not been converted by this pipeline to a common "
                "liability scale."
            ),
            "evidence": evidence_string(cmd),
        }

    # Generic LDSC fallback.
    if "ldsc.py" in text and "--h2" in text:
        cmd = find_lines(text, [r"ldsc\.py", r"--h2"])
        return {
            "classification": "OBSERVED_SCALE_LDSC",
            "can_transform_again": "YES, for binary traits",
            "reason": (
                "LDSC --h2 is called without sample/population prevalence options."
            ),
            "evidence": evidence_string(cmd),
        }

    return {
        "classification": "UNRESOLVED_FROM_AUTOMATIC_RULES",
        "can_transform_again": "NOT YET",
        "reason": "No safe automatic scale classification rule matched this file.",
        "evidence": "",
    }


# ---------------------------------------------------------------------
# Scan local logs for contradictory evidence
# ---------------------------------------------------------------------
def scan_logs(max_files=3000, max_bytes_per_file=2_000_000):
    """
    Search lightweight text/log/out files for observed/liability evidence.
    This is deliberately bounded so it remains fast on HPC.
    """
    patterns = (
        "*.log",
        "*.out",
        "*.txt",
        "*.hers",
        "*.hsq",
    )

    candidates = []
    for pattern in patterns:
        candidates.extend(ROOT.glob(f"*/**/{pattern}"))

    # Unique paths, bounded.
    seen = set()
    unique = []
    for p in candidates:
        try:
            rp = str(p.resolve())
        except Exception:
            rp = str(p)
        if rp not in seen:
            seen.add(rp)
            unique.append(p)

    unique = unique[:max_files]

    liability_hits = []
    observed_hits = []

    for p in unique:
        try:
            if p.stat().st_size > max_bytes_per_file:
                continue
        except Exception:
            continue

        txt = read_text(p)
        if not txt:
            continue

        if contains_any(txt, EXPLICIT_LIABILITY_PATTERNS):
            m = find_lines(txt, EXPLICIT_LIABILITY_PATTERNS)
            if m:
                liability_hits.append(
                    f"{p}: " + evidence_string(m, limit=3)
                )

        if contains_any(txt, EXPLICIT_OBSERVED_PATTERNS):
            m = find_lines(txt, EXPLICIT_OBSERVED_PATTERNS)
            if m:
                observed_hits.append(
                    f"{p}: " + evidence_string(m, limit=3)
                )

    return liability_hits, observed_hits


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("=" * 110)
    print("DATA2 — ORIGINAL HERITABILITY SCALE AUDIT")
    print("=" * 110)
    print(f"Current directory: {ROOT}")
    print("No heritability values will be modified.")
    print("")

    method_files, r_candidates = discover_method_files()

    print(f"Method Python files found: {len(method_files)}")
    for p in method_files:
        print(f"  {p.name}")

    if r_candidates:
        print("External R script(s) found:")
        for p in r_candidates:
            print(f"  {p.name}")
    else:
        print("External LDpred2 R script not found in current directory.")

    print("")

    rows = []

    for p in method_files:
        text = read_text(p)
        result = classify_python_method(p, text, r_candidates)

        rows.append({
            "Source_File": p.name,
            "Classification": result["classification"],
            "Safe_to_Liability_Transform": result["can_transform_again"],
            "Reason": result["reason"],
            "Code_Evidence": result["evidence"],
            "Explicit_Prevalence_or_Liability_Term_Found": (
                "YES" if contains_any(text, EXPLICIT_LIABILITY_PATTERNS) else "NO"
            ),
            "Explicit_Observed_Scale_Term_Found": (
                "YES" if contains_any(text, EXPLICIT_OBSERVED_PATTERNS) else "NO"
            ),
        })

        print("-" * 110)
        print(p.name)
        print(f"  classification: {result['classification']}")
        print(f"  transform?:     {result['can_transform_again']}")
        print(f"  reason:         {result['reason']}")

    # Scan logs only for additional evidence.
    print("")
    print("=" * 110)
    print("SCANNING EXISTING LOGS FOR SCALE KEYWORDS")
    print("=" * 110)

    liability_log_hits, observed_log_hits = scan_logs()

    print(f"Liability/prevalence evidence in logs: {len(liability_log_hits)}")
    for x in liability_log_hits[:20]:
        print("  " + compact(x, 500))

    print(f"Observed-scale evidence in logs: {len(observed_log_hits)}")
    for x in observed_log_hits[:20]:
        print("  " + compact(x, 500))

    # Save CSV.
    fieldnames = [
        "Source_File",
        "Classification",
        "Safe_to_Liability_Transform",
        "Reason",
        "Code_Evidence",
        "Explicit_Prevalence_or_Liability_Term_Found",
        "Explicit_Observed_Scale_Term_Found",
    ]

    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Human-readable report.
    with OUT_TXT.open("w") as fh:
        fh.write("DATA2 — ORIGINAL HERITABILITY SCALE AUDIT\n")
        fh.write("=" * 110 + "\n\n")

        for r in rows:
            fh.write(f"Source: {r['Source_File']}\n")
            fh.write(f"Classification: {r['Classification']}\n")
            fh.write(
                f"Safe to transform: "
                f"{r['Safe_to_Liability_Transform']}\n"
            )
            fh.write(f"Reason: {r['Reason']}\n")
            fh.write(f"Evidence: {r['Code_Evidence']}\n")
            fh.write("\n")

        fh.write("\nLOG EVIDENCE\n")
        fh.write("-" * 110 + "\n")

        fh.write("\nLiability/prevalence hits:\n")
        if liability_log_hits:
            for x in liability_log_hits:
                fh.write(x + "\n")
        else:
            fh.write("NONE FOUND\n")

        fh.write("\nObserved-scale hits:\n")
        if observed_log_hits:
            for x in observed_log_hits:
                fh.write(x + "\n")
        else:
            fh.write("NONE FOUND\n")

    print("")
    print("=" * 110)
    print("DONE")
    print("=" * 110)
    print(f"Wrote: {OUT_CSV.name}")
    print(f"Wrote: {OUT_TXT.name}")
    print("")
    print("Next:")
    print("  cat Data2_Method_Scale_Audit.txt")


if __name__ == "__main__":
    main()
 
