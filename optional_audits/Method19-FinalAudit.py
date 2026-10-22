#!/usr/bin/env python3

import os
import re
import sys
import glob
import shutil
import hashlib
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PHENOTYPES = [
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

# Only two phenotypes are required for the fresh LDAK diagnostic.
# The code path is identical across phenotypes.
LDAK_TEST_PHENOTYPES = [
    "asthma",
    "depression",
]

LDSC_ROOT = Path("LDSCFILES")
LDAK_ROOT = Path("LDAKFILES")
LDSC_SCRIPT = Path("ldsc.py")
LDAK_EXE = Path("./ldak")

EPS = 1e-12


# ============================================================
# UTILITIES
# ============================================================

def banner(text):
    print()
    print("=" * 110)
    print(text)
    print("=" * 110)


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def run_command(cmd, cwd=None, timeout=1800):
    """
    Run command safely.
    Nothing in the original benchmark directories is modified.
    """

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )

        return result.returncode, result.stdout

    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or "") + "\nTIMEOUT"

    except Exception as e:
        return 999, str(e)


def parse_ldsc_log(path):
    h2 = np.nan
    se = np.nan
    variants = np.nan

    if not os.path.exists(path):
        return h2, se, variants

    with open(path, "r", errors="ignore") as f:
        for line in f:

            if "Total Observed scale h2" in line:

                m = re.search(
                    r"Total Observed scale h2:\s*"
                    r"([-+0-9.eE]+)\s*"
                    r"\(([-+0-9.eE]+)\)",
                    line,
                )

                if m:
                    h2 = float(m.group(1))
                    se = float(m.group(2))

            if "After merging with regression SNP LD" in line:

                m = re.search(
                    r"After merging with regression SNP LD,\s*"
                    r"([0-9]+)\s+SNP",
                    line,
                )

                if m:
                    variants = int(m.group(1))

    return h2, se, variants


def parse_ldak_hers(path):
    h2 = np.nan
    se = np.nan

    if not os.path.exists(path):
        return h2, se

    with open(path, "r", errors="ignore") as f:
        for line in f:

            parts = line.split()

            if parts and parts[0] == "Her_All":

                if len(parts) >= 2:
                    try:
                        h2 = float(parts[1])
                    except Exception:
                        pass

                if len(parts) >= 3:
                    try:
                        se = float(parts[2])
                    except Exception:
                        pass

                break

    return h2, se


def valid_plink(prefix):
    prefix = str(prefix)

    return all(
        os.path.exists(prefix + ext)
        for ext in [".bed", ".bim", ".fam"]
    )


def find_clumped_bfile(phenotype, fold=0):
    fold_dir = Path(phenotype) / f"Fold_{fold}"

    candidates = [
        fold_dir / "train_data.QC.clumped.pruned",
        fold_dir / "train_data.clumped.pruned",
    ]

    for p in candidates:
        if valid_plink(p):
            return p

    matches = glob.glob(
        str(fold_dir / "**" / "*clumped*pruned*.bim"),
        recursive=True,
    )

    for bim in sorted(matches):
        prefix = bim[:-4]

        if valid_plink(prefix):
            return Path(prefix)

    return None


def find_ldak_summary(phenotype):
    candidates = [
        Path(phenotype) / f"{phenotype}.ldak",
    ]

    for p in candidates:
        if p.exists():
            return p

    matches = glob.glob(
        str(Path(phenotype) / "*.ldak")
    )

    if matches:
        return Path(sorted(matches)[0])

    return None


# ============================================================
# 1. AUDIT STORED ZERO RESULTS
# ============================================================

banner("AUDIT 1 — STORED ZERO / DUPLICATE RESULTS")

ldsc_rows = []
ldak_rows = []

for phenotype in PHENOTYPES:

    for fold in range(5):

        # LDSC
        f = (
            Path(phenotype)
            / f"Fold_{fold}"
            / "LDSC-Heritability"
            / "Results.csv"
        )

        if f.exists():

            try:
                df = pd.read_csv(f)

                for _, r in df.iterrows():

                    ldsc_rows.append({
                        "phenotype": phenotype,
                        "fold": fold,
                        "model": str(r.get("h2model", "")),
                        "h2": pd.to_numeric(
                            r.get("h2", np.nan),
                            errors="coerce",
                        ),
                        "se": pd.to_numeric(
                            r.get("h2_se", np.nan),
                            errors="coerce",
                        ),
                        "variants": pd.to_numeric(
                            r.get("numberofvariants", np.nan),
                            errors="coerce",
                        ),
                    })

            except Exception as e:
                print("Could not read", f, e)

        # LDAK Calculated
        f = (
            Path(phenotype)
            / f"Fold_{fold}"
            / "LDAK-Calculated-Heritability"
            / "Results.csv"
        )

        if f.exists():

            try:
                df = pd.read_csv(f)

                for _, r in df.iterrows():

                    model = str(r.get("h2model", ""))

                    if (
                        "LDAK_GCTA_" in model
                        or "LDAK_BLD-LDAK_" in model
                    ):

                        ldak_rows.append({
                            "phenotype": phenotype,
                            "fold": fold,
                            "model": model,
                            "h2": pd.to_numeric(
                                r.get("h2", np.nan),
                                errors="coerce",
                            ),
                            "se": pd.to_numeric(
                                r.get("h2_se", np.nan),
                                errors="coerce",
                            ),
                            "variants": pd.to_numeric(
                                r.get("numberofvariants", np.nan),
                                errors="coerce",
                            ),
                        })

            except Exception as e:
                print("Could not read", f, e)


ldsc_df = pd.DataFrame(ldsc_rows)
ldak_df = pd.DataFrame(ldak_rows)


print()
print("LDSC STORED RESULTS")
print("-" * 80)

if len(ldsc_df):

    stored_ldsc_summary = []

    for (phenotype, fold), g in ldsc_df.groupby(
        ["phenotype", "fold"]
    ):

        vals = g["h2"].dropna()

        if len(vals):

            stored_ldsc_summary.append({
                "phenotype": phenotype,
                "fold": fold,
                "rows": len(g),
                "unique_h2": vals.nunique(),
                "min": vals.min(),
                "max": vals.max(),
                "range": vals.max() - vals.min(),
            })

    temp = pd.DataFrame(stored_ldsc_summary)

    print("Phenotype/fold groups:", len(temp))
    print(
        "Groups with identical h2:",
        int((temp["range"].abs() <= EPS).sum()),
        "/",
        len(temp),
    )
    print(
        "Maximum stored within-group range:",
        temp["range"].max()
    )

else:
    print("NO LDSC RESULTS FOUND")


print()
print("LDAK GCTA vs BLD-LDAK STORED RESULTS")
print("-" * 80)

if len(ldak_df):

    gcta = ldak_df[
        ldak_df["model"].str.contains(
            "LDAK_GCTA_",
            regex=False,
        )
    ].copy()

    bld = ldak_df[
        ldak_df["model"].str.contains(
            "LDAK_BLD-LDAK_",
            regex=False,
        )
    ].copy()

    merged = gcta.merge(
        bld,
        on=["phenotype", "fold"],
        suffixes=("_GCTA", "_BLD"),
    )

    if len(merged):

        merged["delta"] = (
            merged["h2_BLD"]
            - merged["h2_GCTA"]
        )

        print("Matched stored rows:", len(merged))
        print(
            "Exact stored h2 equality:",
            int(
                np.isclose(
                    merged["h2_GCTA"],
                    merged["h2_BLD"],
                    rtol=0,
                    atol=0,
                ).sum()
            ),
            "/",
            len(merged),
        )
        print(
            "Maximum stored |delta h2|:",
            merged["delta"].abs().max(),
        )

    else:
        print("No stored GCTA/BLD pairs found.")

else:
    print("NO LDAK CALCULATED RESULTS FOUND")


# ============================================================
# 2. SOURCE-CODE RISK AUDIT
# ============================================================

banner("AUDIT 2 — SOURCE-CODE COLLISION / STALE-FILE RISK")

method7 = Path("Method7.py")
method8 = Path("Method8-Calculated.py")

if method7.exists():

    text7 = method7.read_text(
        errors="ignore"
    )

    same_out = (
        '"--out", "LDSC" + os.sep + filedirec'
        in text7
    )

    print(
        "Method7 reuses one LDSC output prefix:",
        "YES" if same_out else "NOT DETECTED"
    )

    print(
        "Interpretation: reuse is not automatically wrong "
        "because each run is read immediately, but the fresh "
        "audit below removes this ambiguity completely."
    )

else:
    print("Method7.py not found in current directory.")


print()

if method8.exists():

    text8 = method8.read_text(
        errors="ignore"
    )

    humdef_count = text8.count(
        "HumDef"
    )

    cleanup_tagging = (
        "HumDef.tagging"
        in re.search(
            r"def delete_previous_results.*?(?=\ndef |\Z)",
            text8,
            flags=re.S,
        ).group(0)
        if re.search(
            r"def delete_previous_results.*?(?=\ndef |\Z)",
            text8,
            flags=re.S,
        )
        else False
    )

    check_true_count = text8.count(
        "check=True"
    )

    print(
        "Method8 references shared HumDef output:",
        humdef_count,
        "times"
    )

    print(
        "Method8 cleanup deletes HumDef.tagging:",
        "YES" if cleanup_tagging else "NO"
    )

    print(
        "Method8 subprocess calls using check=True:",
        check_true_count
    )

    if (
        humdef_count > 1
        and not cleanup_tagging
    ):

        print()
        print(
            "*** RISK DETECTED: a failed later calc-tagging "
            "command could leave an earlier HumDef.tagging "
            "file available for reuse."
        )

else:
    print(
        "Method8-Calculated.py not found in current directory."
    )


# ============================================================
# 3. LDSC REFERENCE FILE CONTENT AUDIT
# ============================================================

banner("AUDIT 3 — ARE LDSC REFERENCE AND WEIGHT FILES ACTUALLY DIFFERENT?")

ref_dir = LDSC_ROOT / "eur_ref_ld_chr"
weight_dir = LDSC_ROOT / "eur_w_ld_chr"

if not ref_dir.exists() or not weight_dir.exists():

    print(
        "Cannot inspect LDSCFILES directories."
    )

else:

    ref_files = {
        p.name: p
        for p in ref_dir.iterdir()
        if p.is_file()
    }

    weight_files = {
        p.name: p
        for p in weight_dir.iterdir()
        if p.is_file()
    }

    common = sorted(
        set(ref_files)
        & set(weight_files)
    )

    identical = 0
    different = 0

    examples = []

    for name in common:

        try:

            h1 = sha256(
                ref_files[name]
            )

            h2 = sha256(
                weight_files[name]
            )

            if h1 == h2:
                identical += 1
            else:
                different += 1

                if len(examples) < 10:
                    examples.append(name)

        except Exception:
            pass

    print("Common filenames:", len(common))
    print("Byte-identical files:", identical)
    print("Different files:", different)

    if examples:
        print(
            "Examples of differing files:",
            ", ".join(examples)
        )

    if different == 0 and common:
        print()
        print(
            "IMPORTANT: the two LDSC directories are byte-identical "
            "for all common files. Exact identical h2 values would "
            "therefore be expected."
        )


# ============================================================
# 4. FRESH LDSC RERUN WITH UNIQUE OUTPUT PREFIXES
# ============================================================

banner("AUDIT 4 — FRESH LDSC 4-WAY RERUN WITH UNIQUE OUTPUTS")

ldsc_status = "INCONCLUSIVE"
ldsc_fresh_rows = []

conda = shutil.which("conda")

can_run_ldsc = (
    LDSC_SCRIPT.exists()
    and ref_dir.exists()
    and weight_dir.exists()
)


if not can_run_ldsc:

    print(
        "Cannot rerun LDSC: ldsc.py or LDSCFILES missing."
    )

elif conda is None:

    print(
        "Cannot find conda command. Run this section manually "
        "inside the ldscc environment."
    )

else:

    scratch = tempfile.mkdtemp(
        prefix=".method19_ldsc_audit_",
        dir=".",
    )

    try:

        refs = [
            "eur_ref_ld_chr",
            "eur_w_ld_chr",
        ]

        weights = [
            "eur_ref_ld_chr",
            "eur_w_ld_chr",
        ]

        for phenotype in PHENOTYPES:

            gwas = Path(
                phenotype
            ) / f"{phenotype}.ldsc.txt"

            if not gwas.exists():

                print(
                    phenotype,
                    ": missing",
                    gwas,
                )

                continue

            phenotype_values = []

            for ref in refs:

                for weight in weights:

                    safe_name = (
                        phenotype
                        + "__REF_"
                        + ref
                        + "__W_"
                        + weight
                    )

                    out_prefix = (
                        Path(scratch)
                        / safe_name
                    )

                    cmd = [
                        conda,
                        "run",
                        "-n",
                        "ldscc",
                        "python",
                        str(
                            LDSC_SCRIPT.resolve()
                        ),
                        "--h2",
                        str(
                            gwas.resolve()
                        ),
                        "--ref-ld-chr",
                        str(
                            (
                                LDSC_ROOT
                                / ref
                            ).resolve()
                        )
                        + os.sep,
                        "--w-ld-chr",
                        str(
                            (
                                LDSC_ROOT
                                / weight
                            ).resolve()
                        )
                        + os.sep,
                        "--out",
                        str(
                            out_prefix.resolve()
                        ),
                    ]

                    rc, output = run_command(
                        cmd
                    )

                    log_path = (
                        str(out_prefix)
                        + ".log"
                    )

                    h2, se, variants = (
                        parse_ldsc_log(
                            log_path
                        )
                    )

                    ldsc_fresh_rows.append({
                        "phenotype": phenotype,
                        "ref": ref,
                        "weight": weight,
                        "returncode": rc,
                        "h2": h2,
                        "se": se,
                        "variants": variants,
                    })

                    if np.isfinite(h2):
                        phenotype_values.append(
                            h2
                        )

                    if rc != 0:

                        print()
                        print(
                            "*** LDSC COMMAND FAILED"
                        )
                        print(
                            phenotype,
                            ref,
                            weight,
                            "returncode=",
                            rc,
                        )
                        print(
                            output[-2000:]
                        )

            if phenotype_values:

                r = (
                    max(phenotype_values)
                    - min(phenotype_values)
                )

                print(
                    f"{phenotype:<45}"
                    f" range across four fresh runs = "
                    f"{r:.10f}"
                )

        fresh = pd.DataFrame(
            ldsc_fresh_rows
        )

        if len(fresh):

            failures = int(
                (
                    fresh[
                        "returncode"
                    ] != 0
                ).sum()
            )

            valid = fresh[
                np.isfinite(
                    fresh[
                        "h2"
                    ]
                )
            ]

            ranges = []

            for phenotype, g in valid.groupby(
                "phenotype"
            ):

                if len(g) == 4:

                    ranges.append(
                        g["h2"].max()
                        - g["h2"].min()
                    )

            print()
            print("Fresh LDSC commands:", len(fresh))
            print("Failed commands:", failures)

            if ranges:

                print(
                    "Maximum fresh 4-way h2 range:",
                    max(ranges),
                )

            if (
                failures == 0
                and len(ranges) == len(PHENOTYPES)
            ):

                if max(ranges) <= EPS:

                    ldsc_status = (
                        "VERIFIED_IDENTICAL_AT_LDSC_OUTPUT_PRECISION"
                    )

                else:

                    ldsc_status = (
                        "CURRENT_RESULTS_ARE_STALE_OR_INCORRECT"
                    )

    finally:

        shutil.rmtree(
            scratch,
            ignore_errors=True,
        )


print()
print(
    "LDSC FINAL AUDIT STATUS:",
    ldsc_status
)


# ============================================================
# 5. FRESH LDAK GCTA vs BLD-LDAK RERUN
#    UNIQUE TAGGING PREFIXES + check return codes
# ============================================================

banner("AUDIT 5 — FRESH LDAK GCTA vs BLD-LDAK WITH UNIQUE TAGGING FILES")

ldak_status = "INCONCLUSIVE"

ldak_fresh_rows = []

if not LDAK_EXE.exists():

    print("./ldak executable not found.")

elif not LDAK_ROOT.exists():

    print("LDAKFILES directory not found.")

else:

    scratch = tempfile.mkdtemp(
        prefix=".method19_ldak_audit_",
        dir=".",
    )

    try:

        scratch = Path(
            scratch
        ).resolve()

        successful_tests = 0
        differing_h2_tests = 0
        differing_tag_tests = 0
        failed_tests = 0

        for phenotype in LDAK_TEST_PHENOTYPES:

            print()
            print("-" * 110)
            print(
                "Testing:",
                phenotype,
                "Fold_0"
            )
            print("-" * 110)

            bfile = find_clumped_bfile(
                phenotype,
                fold=0,
            )

            summary = find_ldak_summary(
                phenotype
            )

            if bfile is None:
                print(
                    "Missing clumped BFILE."
                )
                failed_tests += 1
                continue

            if summary is None:
                print(
                    "Missing .ldak GWAS summary."
                )
                failed_tests += 1
                continue

            test_dir = (
                scratch
                / phenotype
            )

            test_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            bfile = bfile.resolve()
            summary = summary.resolve()

            # -----------------------------------------------
            # GCTA tagging model
            # -----------------------------------------------

            gcta_tag = (
                test_dir
                / "GCTA_UNIQUE"
            )

            cmd = [
                str(
                    LDAK_EXE.resolve()
                ),
                "--calc-tagging",
                str(gcta_tag),
                "--bfile",
                str(bfile),
                "--power",
                "-1",
            ]

            rc_gcta_tag, out = run_command(
                cmd
            )

            print(
                "GCTA calc-tagging return code:",
                rc_gcta_tag
            )

            if rc_gcta_tag != 0:

                print(
                    out[-3000:]
                )

                failed_tests += 1
                continue

            gcta_tagfile = Path(
                str(gcta_tag)
                + ".tagging"
            )

            if not gcta_tagfile.exists():

                print(
                    "GCTA tagging file was not created."
                )
                failed_tests += 1
                continue

            gcta_hers = (
                test_dir
                / "GCTA_HERS"
            )

            cmd = [
                str(
                    LDAK_EXE.resolve()
                ),
                "--sum-hers",
                str(gcta_hers),
                "--summary",
                str(summary),
                "--tagfile",
                str(gcta_tagfile),
                "--check-sums",
                "NO",
            ]

            rc_gcta_hers, out = run_command(
                cmd
            )

            print(
                "GCTA sum-hers return code:",
                rc_gcta_hers
            )

            if rc_gcta_hers != 0:

                print(
                    out[-3000:]
                )

                failed_tests += 1
                continue

            gcta_h2, gcta_se = (
                parse_ldak_hers(
                    str(gcta_hers)
                    + ".hers"
                )
            )


            # -----------------------------------------------
            # Build BLD-LDAK annotations IN SCRATCH.
            # Do not touch original LDAKFILES/bld65.
            # -----------------------------------------------

            sections = (
                test_dir
                / "sections"
            )

            cmd = [
                str(
                    LDAK_EXE.resolve()
                ),
                "--cut-weights",
                str(sections),
                "--bfile",
                str(bfile),
            ]

            rc_cut, out = run_command(
                cmd
            )

            print(
                "BLD cut-weights return code:",
                rc_cut
            )

            if rc_cut != 0:

                print(
                    out[-3000:]
                )

                failed_tests += 1
                continue


            cmd = [
                str(
                    LDAK_EXE.resolve()
                ),
                "--calc-weights-all",
                str(sections),
                "--bfile",
                str(bfile),
            ]

            rc_weights, out = run_command(
                cmd
            )

            print(
                "BLD calc-weights-all return code:",
                rc_weights
            )

            if rc_weights != 0:

                print(
                    out[-3000:]
                )

                failed_tests += 1
                continue


            weights_short = (
                sections
                / "weights.short"
            )

            if not weights_short.exists():

                print(
                    "weights.short not created."
                )

                failed_tests += 1
                continue


            ann_dir = (
                test_dir
                / "annotations"
            )

            ann_dir.mkdir(
                exist_ok=True
            )


            # Copy/symlink the existing BLD annotation files,
            # except bld65 which must be generated for this BFILE.
            copied = 0

            for source in sorted(
                LDAK_ROOT.glob(
                    "bld*"
                )
            ):

                if not source.is_file():
                    continue

                if source.name == "bld65":
                    continue

                target = (
                    ann_dir
                    / source.name
                )

                if not target.exists():

                    os.symlink(
                        source.resolve(),
                        target,
                    )

                copied += 1


            shutil.copy2(
                weights_short,
                ann_dir / "bld65",
            )


            annotation_prefix = (
                ann_dir
                / "bld"
            )


            # -----------------------------------------------
            # BLD tagging model with UNIQUE prefix
            # -----------------------------------------------

            bld_tag = (
                test_dir
                / "BLD_UNIQUE"
            )

            cmd = [
                str(
                    LDAK_EXE.resolve()
                ),
                "--calc-tagging",
                str(bld_tag),
                "--bfile",
                str(bfile),
                "--power",
                "-.25",
                "--annotation-number",
                "65",
                "--annotation-prefix",
                str(annotation_prefix),
            ]

            rc_bld_tag, out = run_command(
                cmd
            )

            print(
                "BLD calc-tagging return code:",
                rc_bld_tag
            )

            if rc_bld_tag != 0:

                print()
                print(
                    "*** BLD CALC-TAGGING FAILED"
                )
                print(
                    out[-5000:]
                )

                print()
                print(
                    "*** THIS IS IMPORTANT: the original Method8 "
                    "does not remove HumDef.tagging before this step "
                    "and does not require subprocess success. "
                    "A failed BLD calculation could therefore reuse "
                    "the earlier GCTA HumDef.tagging file."
                )

                failed_tests += 1
                continue


            bld_tagfile = Path(
                str(bld_tag)
                + ".tagging"
            )

            if not bld_tagfile.exists():

                print(
                    "BLD tagging file was not created."
                )

                failed_tests += 1
                continue


            bld_hers = (
                test_dir
                / "BLD_HERS"
            )

            cmd = [
                str(
                    LDAK_EXE.resolve()
                ),
                "--sum-hers",
                str(bld_hers),
                "--summary",
                str(summary),
                "--tagfile",
                str(bld_tagfile),
                "--check-sums",
                "NO",
            ]

            rc_bld_hers, out = run_command(
                cmd
            )

            print(
                "BLD sum-hers return code:",
                rc_bld_hers
            )

            if rc_bld_hers != 0:

                print(
                    out[-3000:]
                )

                failed_tests += 1
                continue


            bld_h2, bld_se = (
                parse_ldak_hers(
                    str(bld_hers)
                    + ".hers"
                )
            )


            # -----------------------------------------------
            # Compare
            # -----------------------------------------------

            gcta_hash = sha256(
                gcta_tagfile
            )

            bld_hash = sha256(
                bld_tagfile
            )

            tags_same = (
                gcta_hash
                == bld_hash
            )

            h2_delta = (
                bld_h2
                - gcta_h2
            )


            print()
            print(
                "GCTA fresh h2:",
                gcta_h2,
                "SE:",
                gcta_se
            )

            print(
                "BLD  fresh h2:",
                bld_h2,
                "SE:",
                bld_se
            )

            print(
                "Fresh BLD-GCTA delta:",
                h2_delta
            )

            print(
                "Tagging files byte-identical:",
                tags_same
            )


            successful_tests += 1


            if not tags_same:
                differing_tag_tests += 1


            if (
                np.isfinite(
                    h2_delta
                )
                and abs(
                    h2_delta
                ) > EPS
            ):

                differing_h2_tests += 1


            ldak_fresh_rows.append({
                "phenotype":
                    phenotype,

                "gcta_h2":
                    gcta_h2,

                "bld_h2":
                    bld_h2,

                "delta":
                    h2_delta,

                "tagfiles_identical":
                    tags_same,
            })


        print()
        print("=" * 110)
        print("FRESH LDAK SUMMARY")
        print("=" * 110)

        print(
            "Successful fresh phenotype tests:",
            successful_tests
        )

        print(
            "Tests with different tagging files:",
            differing_tag_tests
        )

        print(
            "Tests with different fresh h2:",
            differing_h2_tests
        )

        print(
            "Failed/incomplete tests:",
            failed_tests
        )


        if successful_tests > 0:

            if differing_h2_tests > 0:

                ldak_status = (
                    "STORED_GCTA_BLD_ZERO_IS_INVALID"
                )

            elif differing_tag_tests > 0:

                ldak_status = (
                    "TAGGING_DIFFERS_BUT_H2_EQUAL"
                )

            else:

                ldak_status = (
                    "GCTA_AND_BLD_FRESHLY_IDENTICAL"
                )


    finally:

        shutil.rmtree(
            scratch,
            ignore_errors=True,
        )


print()
print(
    "LDAK FINAL AUDIT STATUS:",
    ldak_status
)


# ============================================================
# FINAL DECISION
# ============================================================

banner("FINAL DECISION")

print(
    "LDSC:",
    ldsc_status
)

print(
    "LDAK:",
    ldak_status
)

print()


need_method7_rerun = (
    ldsc_status
    == "CURRENT_RESULTS_ARE_STALE_OR_INCORRECT"
)

need_method8_rerun = (
    ldak_status
    == "STORED_GCTA_BLD_ZERO_IS_INVALID"
    or ldak_status == "INCONCLUSIVE"
)


if need_method7_rerun:

    print(
        "ACTION: METHOD 7 MUST BE FIXED/RERUN "
        "before freezing Method19."
    )

else:

    print(
        "ACTION: No Method7 rerun is indicated by this audit "
        "unless LDSC status is INCONCLUSIVE."
    )


if need_method8_rerun:

    print(
        "ACTION: METHOD 8 CALCULATED MUST BE FIXED/RERUN "
        "before freezing Method19."
    )

else:

    print(
        "ACTION: No automatic Method8 rerun required by this "
        "audit result, but interpret exact-zero results cautiously."
    )


print()

if (
    ldsc_status
    == "VERIFIED_IDENTICAL_AT_LDSC_OUTPUT_PRECISION"
    and ldak_status
    not in {
        "STORED_GCTA_BLD_ZERO_IS_INVALID",
        "INCONCLUSIVE",
    }
):

    print(
        "FINAL: upstream zero audits passed sufficiently to "
        "rerun Method19 once and freeze the empirical "
        "one-factor analysis."
    )

else:

    print(
        "FINAL: DO NOT FREEZE METHOD19 YET. "
        "Resolve the upstream method(s) identified above, "
        "then rerun Method19 one final time."
    )


print()
print(
    "Audit scratch directories were deleted. "
    "Original benchmark files were not modified."
)
print()

