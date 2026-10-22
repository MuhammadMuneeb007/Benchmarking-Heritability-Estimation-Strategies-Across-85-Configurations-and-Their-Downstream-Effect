 
"""
Data1-Information.py

REAL EMPIRICAL DATA ONLY.

Run from:
    /data/ascher01/uqmmune1/finalheritability

Simply:
    python Data1-Information.py

No arguments are required.

For each real phenotype folder this script gets:
  - GWAS accession from local GCST* filename
  - PMID / DOI / study title from GWAS Catalog API when available
  - target N / cases / controls / sample prevalence P from *_QC.fam
  - discovery GWAS N from <phenotype>.gz
  - discovery GWAS cases / controls / P from GWAS Catalog metadata
  - common SNPs between <phenotype>.gz and *_QC.bim

IMPORTANT
---------
For liability-scale conversion:
    P = sample prevalence = cases / (cases + controls)
    K = population prevalence

This script obtains P where possible.
It DOES NOT invent K.

The GWAS Catalog metadata parser:
  1. queries both study and ancestry/sample endpoints;
  2. looks for structured case/control fields;
  3. parses free-text sample descriptions such as
       "102,084 cases and 771,257 controls";
  4. if several candidate pairs are found, chooses the pair whose
     cases+controls is closest to the N column in the local processed GWAS file.

Simulation folders are ignored.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path.cwd()
OUT_CSV = ROOT / "Data1_GWAS_Information_REAL.csv"
OUT_TXT = ROOT / "Data1_GWAS_Information_REAL.txt"
OUT_CANDIDATES = ROOT / "Data1_GWAS_CaseControl_Candidates.csv"

API_TIMEOUT = 20


# ---------------------------------------------------------------------
# REAL empirical phenotype folders only.
# ---------------------------------------------------------------------
REAL_PHENOTYPES = {
    "asthma": [
        "asthma",
    ],
    "body_mass_index": [
        "body_mass_index",
        "body_mass_index_bmi",
        "bmi",
    ],
    "depression": [
        "depression",
    ],
    "gastro_oesophageal_reflux": [
        "gastro_oesophageal_reflux",
        "gastroesophageal_reflux",
        "gastro_oesophageal_reflux_gord_gastric_reflux",
        "gastroesophageal_reflux_gord_gastric_reflux",
        "gord",
        "gerd",
        "gastric_reflux",
    ],
    "high_cholesterol": [
        "high_cholesterol",
    ],
    "hypertension": [
        "hypertension",
    ],
    "hypothyroidism": [
        "hypothyroidism",
        "hypothyroidism_myxoedema",
        "myxoedema",
    ],
    "irritable_bowel_syndrome": [
        "irritable_bowel_syndrome",
        "ibs",
    ],
    "migraine": [
        "migraine",
    ],
    "osteoarthritis": [
        "osteoarthritis",
        "oa",
    ],
}


# Optional fallback if a real phenotype does not have a GCST filename locally.
# Example:
# MANUAL_ACCESSION = {"depression": "GCSTXXXXXXXX"}
MANUAL_ACCESSION = {}


def nrm(x: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def safe_int(x):
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)

    s = str(x).strip().replace(",", "")
    if re.fullmatch(r"\d+", s):
        try:
            return int(s)
        except Exception:
            return None
    return None


def fmt_p(x):
    if x in ("", None):
        return ""
    try:
        return f"{float(x):.6f}"
    except Exception:
        return str(x)


# ---------------------------------------------------------------------
# REAL phenotype folder discovery
# ---------------------------------------------------------------------
def canonical_real_name(folder_name: str):
    name = nrm(folder_name)

    for canonical, aliases in REAL_PHENOTYPES.items():
        valid = {nrm(canonical)} | {nrm(a) for a in aliases}
        if name in valid:
            return canonical

    return None


def find_real_phenotype_folders(root: Path):
    """
    Immediate children only. This makes the script fast and guarantees that
    simulation directories are ignored.
    """
    out = []

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue

        canonical = canonical_real_name(d.name)
        if canonical is None:
            continue

        if not any(d.glob("*_QC.fam")):
            continue

        if not any(d.glob("*_QC.bim")):
            continue

        out.append((canonical, d))

    return out


def pick_qc_file(folder: Path, ext: str):
    exact = folder / f"{folder.name}_QC.{ext}"
    if exact.exists():
        return exact

    hits = sorted(folder.glob(f"*_QC.{ext}"))
    return hits[0] if hits else None


# ---------------------------------------------------------------------
# Target sample counts
# ---------------------------------------------------------------------
def read_target_fam(fam_path: Path):
    values = []

    with fam_path.open("rt", errors="ignore") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                values.append(float(parts[5]))
            except Exception:
                pass

    if not values:
        return "", "", "", "", "NO_PHENOTYPE_VALUES"

    unique = set(values)

    # Standard PLINK binary coding.
    if (
        1.0 in unique
        and 2.0 in unique
        and unique.issubset({-9.0, 0.0, 1.0, 2.0})
    ):
        controls = sum(v == 1.0 for v in values)
        cases = sum(v == 2.0 for v in values)
        n = cases + controls
        p = cases / n if n else ""
        return n, cases, controls, p, "PLINK_1_CONTROL_2_CASE"

    # 0/1 binary coding fallback.
    if (
        0.0 in unique
        and 1.0 in unique
        and 2.0 not in unique
        and unique.issubset({-9.0, 0.0, 1.0})
    ):
        controls = sum(v == 0.0 for v in values)
        cases = sum(v == 1.0 for v in values)
        n = cases + controls
        p = cases / n if n else ""
        return n, cases, controls, p, "0_CONTROL_1_CASE"

    # Continuous trait, e.g. BMI.
    valid = [v for v in values if v not in {-9.0, -99.0}]
    return len(valid), "", "", "", "CONTINUOUS_OR_UNRESOLVED"


# ---------------------------------------------------------------------
# Local processed GWAS + common SNPs
# ---------------------------------------------------------------------
def get_qc_rsids(bim_path: Path):
    rsids = set()

    with bim_path.open("rt", errors="ignore") as fh:
        for line in fh:
            parts = line.split(None, 3)
            if len(parts) >= 2:
                rsids.add(parts[1])

    return rsids


def find_processed_gwas(folder: Path):
    """
    Prefer <foldername>.gz, e.g. asthma/asthma.gz.

    Do NOT use the huge raw GCST*.tsv.gz for this step.
    """
    exact = folder / f"{folder.name}.gz"
    if exact.exists():
        return exact

    # Try canonical aliases against local *.gz.
    non_gcst = [
        p for p in folder.glob("*.gz")
        if not p.name.upper().startswith("GCST")
    ]

    return sorted(non_gcst)[0] if non_gcst else None


def read_processed_gwas(processed_gz: Path | None, qc_rsids: set[str]):
    """
    Stream local processed GWAS:
      CHR BP SNP A1 A2 N SE P BETA INFO MAF

    Returns:
      row count
      number of common unique rsIDs
      first valid N
      status
    """
    if processed_gz is None or not processed_gz.exists():
        return "", "", "", "PROCESSED_GZ_NOT_FOUND"

    common = set()
    local_n = ""
    rows = 0

    try:
        with gzip.open(processed_gz, "rt", errors="ignore") as fh:
            header = fh.readline().strip()

            if not header:
                return "", "", "", "EMPTY_GZ"

            cols = header.split("\t")
            if len(cols) == 1:
                cols = header.split()

            normalized = [nrm(c) for c in cols]

            snp_idx = None
            for alias in ("snp", "rsid", "rs_id", "variant_id"):
                if alias in normalized:
                    snp_idx = normalized.index(alias)
                    break

            n_idx = normalized.index("n") if "n" in normalized else None

            if snp_idx is None:
                return "", "", "", f"SNP_COLUMN_NOT_FOUND:{header[:120]}"

            needed = max(snp_idx, n_idx if n_idx is not None else 0)

            for line in fh:
                parts = line.rstrip("\n").split("\t", needed + 1)

                if len(parts) <= needed:
                    parts = line.split(None, needed + 1)

                if len(parts) <= snp_idx:
                    continue

                rows += 1
                snp = parts[snp_idx]

                if snp in qc_rsids:
                    common.add(snp)

                if local_n == "" and n_idx is not None and len(parts) > n_idx:
                    raw_n = parts[n_idx].strip()

                    if raw_n:
                        try:
                            n_val = float(raw_n)
                            local_n = int(n_val) if n_val.is_integer() else n_val
                        except Exception:
                            local_n = raw_n

        return rows, len(common), local_n, "OK"

    except Exception as e:
        return "", "", "", f"ERROR:{type(e).__name__}:{e}"


# ---------------------------------------------------------------------
# GWAS accession
# ---------------------------------------------------------------------
def find_accession(folder: Path):
    found = []

    for p in folder.iterdir():
        if not p.is_file():
            continue

        m = re.search(r"(GCST\d+)", p.name, flags=re.I)
        if m:
            found.append(m.group(1).upper())

    found = sorted(set(found))

    if found:
        return found[0], "FROM_LOCAL_GCST_FILENAME"

    manual = MANUAL_ACCESSION.get(folder.name.lower(), "")
    if manual:
        return manual, "MANUAL_OVERRIDE"

    return "", "NOT_FOUND"


# ---------------------------------------------------------------------
# GWAS Catalog API
# ---------------------------------------------------------------------
def fetch_json(url: str):
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "heritability-benchmark-liability-audit/1.0",
        },
    )

    with urllib.request.urlopen(req, timeout=API_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8", errors="ignore"))


def fetch_catalog_payloads(accession: str):
    """
    Query several GWAS Catalog study/sample endpoints because field nesting
    varies between API versions.
    """
    if not accession:
        return []

    urls = [
        f"https://www.ebi.ac.uk/gwas/rest/api/studies/{accession}",
        f"https://www.ebi.ac.uk/gwas/rest/api/studies/{accession}/ancestries",
        f"https://www.ebi.ac.uk/gwas/rest/api/v2/studies/{accession}",
        f"https://www.ebi.ac.uk/gwas/rest/api/v2/studies/{accession}/ancestries",
        (
            "https://www.ebi.ac.uk/gwas/rest/api/v2/studies?"
            + urllib.parse.urlencode({"accession_id": accession})
        ),
    ]

    payloads = []

    for url in urls:
        try:
            data = fetch_json(url)
            payloads.append((url, data))
        except Exception:
            continue

    return payloads


def walk_json(obj, path=""):
    if isinstance(obj, dict):
        yield path, obj

        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else str(key)
            yield from walk_json(value, new_path)

    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from walk_json(value, f"{path}[{i}]")

    else:
        yield path, obj


# ---------------------------------------------------------------------
# Case/control extraction
# ---------------------------------------------------------------------
def parse_case_control_pairs_from_text(text: str):
    """
    Return ALL ancestry/stage-specific case/control pairs found in GWAS Catalog
    sample text.

    Handles examples such as:
      38,369 European ancestry cases, 411,131 European ancestry controls,
      13,015 East Asian ancestry cases, 162,933 East Asian ancestry controls

    It deliberately keeps those as TWO candidate pairs instead of summing them.
    """
    if not text:
        return []

    text = " ".join(str(text).split())
    pairs = []

    # Typical GWAS Catalog wording:
    #   <N> European ancestry cases, <N> European ancestry controls
    forward = re.compile(
        r"(?P<cases>\d[\d,]*)\s+"
        r"(?P<case_desc>[^,;]{0,180}?\b(?:case subjects|cases|affected individuals|affected subjects|affected participants)\b)"
        r"\s*(?:,|;|\band\b)?\s*"
        r"(?P<controls>\d[\d,]*)\s+"
        r"(?P<control_desc>[^,;]{0,180}?\b(?:control subjects|controls|unaffected individuals|unaffected subjects|unaffected participants)\b)",
        flags=re.I,
    )

    for m in forward.finditer(text):
        try:
            cases = int(m.group("cases").replace(",", ""))
            controls = int(m.group("controls").replace(",", ""))
        except ValueError:
            continue

        pairs.append((cases, controls, m.group(0)))

    # Reverse wording fallback:
    #   <N> controls, <N> cases
    reverse = re.compile(
        r"(?P<controls>\d[\d,]*)\s+"
        r"(?P<control_desc>[^,;]{0,180}?\b(?:control subjects|controls|unaffected individuals|unaffected subjects|unaffected participants)\b)"
        r"\s*(?:,|;|\band\b)?\s*"
        r"(?P<cases>\d[\d,]*)\s+"
        r"(?P<case_desc>[^,;]{0,180}?\b(?:case subjects|cases|affected individuals|affected subjects|affected participants)\b)",
        flags=re.I,
    )

    for m in reverse.finditer(text):
        try:
            cases = int(m.group("cases").replace(",", ""))
            controls = int(m.group("controls").replace(",", ""))
        except ValueError:
            continue

        pairs.append((cases, controls, m.group(0)))

    # Labelled fallback:
    #   cases: 102084 controls: 771257
    labelled = re.compile(
        r"(?:cases|case subjects|affected individuals)\s*[:=]\s*(?P<cases>\d[\d,]*)"
        r".{0,180}?"
        r"(?:controls|control subjects|unaffected individuals)\s*[:=]\s*(?P<controls>\d[\d,]*)",
        flags=re.I,
    )

    for m in labelled.finditer(text):
        try:
            cases = int(m.group("cases").replace(",", ""))
            controls = int(m.group("controls").replace(",", ""))
        except ValueError:
            continue

        pairs.append((cases, controls, m.group(0)))

    # De-duplicate pairs extracted from the same string.
    seen = set()
    unique = []

    for cases, controls, evidence in pairs:
        key = (cases, controls)
        if key in seen:
            continue
        seen.add(key)
        unique.append((cases, controls, evidence))

    return unique


def structured_case_control_from_dict(d: dict):
    """
    If one JSON object contains structured case and control counts, return them.
    """
    cases = []
    controls = []
    stage = ""
    description = ""

    for key, value in d.items():
        nk = nrm(key)

        if "stage" in nk:
            stage = str(value)

        if any(x in nk for x in ("description", "sample_description", "initial_sample")):
            if isinstance(value, str):
                description = value

        numeric = safe_int(value)

        if numeric is None:
            continue

        if (
            "case" in nk
            and "control" not in nk
        ):
            cases.append(numeric)

        if "control" in nk:
            controls.append(numeric)

    if not cases or not controls:
        return []

    out = []

    for ca in cases:
        for co in controls:
            out.append((ca, co, stage, description))

    return out


def extract_case_control_candidates(payloads):
    """
    Extract all plausible discovery GWAS case/control pairs from:
      - structured API fields
      - free-text sample descriptions

    Each candidate carries provenance.
    """
    candidates = []

    for url, payload in payloads:
        for path, value in walk_json(payload):

            # Structured dictionary fields.
            if isinstance(value, dict):
                for cases, controls, stage, description in structured_case_control_from_dict(value):
                    candidates.append({
                        "cases": cases,
                        "controls": controls,
                        "total": cases + controls,
                        "source": "STRUCTURED_JSON",
                        "api_url": url,
                        "json_path": path,
                        "stage": stage,
                        "evidence_text": description,
                    })

            # Free-text sample descriptions.
            elif isinstance(value, str):
                for cases, controls, evidence in parse_case_control_pairs_from_text(value):
                    candidates.append({
                        "cases": cases,
                        "controls": controls,
                        "total": cases + controls,
                        "source": "PARSED_SAMPLE_TEXT",
                        "api_url": url,
                        "json_path": path,
                        "stage": "",
                        "evidence_text": evidence,
                    })

    # De-duplicate exact candidates while preserving provenance preference.
    seen = set()
    unique = []

    for c in candidates:
        key = (
            c["cases"],
            c["controls"],
            c["api_url"],
            c["json_path"],
            c["source"],
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(c)

    return unique


def discovery_bonus(candidate):
    """
    Prefer candidates that explicitly look like discovery/initial-stage samples.
    """
    text = " ".join(
        str(candidate.get(k, ""))
        for k in ("stage", "json_path", "evidence_text")
    ).lower()

    score = 0

    if "initial" in text:
        score += 4

    if "discovery" in text:
        score += 4

    if "replication" in text:
        score -= 5

    if "replicate" in text:
        score -= 5

    if candidate.get("source") == "STRUCTURED_JSON":
        score += 2

    return score


def choose_case_control_candidate(candidates, local_n):
    """
    Select the case/control candidate corresponding to the local processed GWAS.

    Priority:
      1. exact cases + controls == local GWAS N
      2. discovery / initial-stage evidence
      3. smallest difference from local GWAS N
      4. structured JSON over free-text parsing when tied

    This prevents multi-ancestry studies from being incorrectly summed.
    """
    if not candidates:
        return None

    local_n_int = safe_int(local_n)

    def sort_key(c):
        stage_score = discovery_bonus(c)

        if local_n_int is not None:
            diff = abs(c["total"] - local_n_int)
            exact_penalty = 0 if diff == 0 else 1
            relative_diff = diff / max(local_n_int, 1)
        else:
            exact_penalty = 1
            diff = 10**18
            relative_diff = 10**18

        source_penalty = 0 if c["source"] == "STRUCTURED_JSON" else 1

        return (
            exact_penalty,
            -stage_score,
            relative_diff,
            diff,
            source_penalty,
        )

    return sorted(candidates, key=sort_key)[0]


# ---------------------------------------------------------------------
# Other study metadata: PMID / DOI / title / trait
# ---------------------------------------------------------------------
def extract_general_metadata(payloads):
    doi = ""
    pmid = ""
    title = ""
    trait = ""
    all_sample_text = []

    for url, payload in payloads:
        for path, value in walk_json(payload):
            if value is None or isinstance(value, (dict, list)):
                continue

            sval = str(value).strip()
            lpath = path.lower()

            if not doi:
                m = re.search(
                    r"(10\.\d{4,9}/[^\s\"'<>]+)",
                    sval,
                    flags=re.I,
                )
                if m:
                    doi = m.group(1).rstrip(".,;)")

            if not pmid and ("pmid" in lpath or "pubmed" in lpath):
                m = re.search(r"\b(\d{6,9})\b", sval)
                if m:
                    pmid = m.group(1)

            if not title and lpath.endswith("title"):
                title = sval

            if not trait and (
                "diseasetrait" in lpath
                or "reportedtrait" in lpath
                or lpath.endswith(".trait")
            ):
                trait = sval

            if any(
                token in lpath
                for token in (
                    "sample",
                    "ancestr",
                    "initial",
                    "discovery",
                )
            ):
                if (
                    isinstance(value, str)
                    and len(sval) > 5
                    and sval not in all_sample_text
                ):
                    all_sample_text.append(sval)

    return {
        "doi": doi,
        "pmid": pmid,
        "title": title,
        "trait": trait,
        "sample_text": " | ".join(all_sample_text[:20]),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("=" * 108)
    print("DATA1 — REAL GWAS ACCESSION / DOI / CASE-CONTROL AUDIT")
    print("=" * 108)
    print(f"Current directory: {ROOT}")
    print("REAL empirical phenotype folders only.")
    print("Simulation folders are ignored.")
    print("No arguments are required.")
    print("")

    folders = find_real_phenotype_folders(ROOT)

    if not folders:
        print("ERROR: No real phenotype folders were found.")
        sys.exit(1)

    print(f"REAL phenotype folders found: {len(folders)}")
    print("  " + ", ".join(d.name for _, d in folders))
    print("")

    results = []
    all_candidates = []

    for i, (canonical, folder) in enumerate(folders, 1):
        print("-" * 108)
        print(f"[{i}/{len(folders)}] {folder.name}")

        bim = pick_qc_file(folder, "bim")
        fam = pick_qc_file(folder, "fam")
        processed_gz = find_processed_gwas(folder)

        target_n, target_cases, target_controls, target_p, phenotype_type = (
            read_target_fam(fam)
        )

        accession, accession_source = find_accession(folder)

        # Fast local processed GWAS scan.
        qc_rsids = get_qc_rsids(bim)

        (
            processed_rows,
            common_snps,
            local_gwas_n,
            processed_status,
        ) = read_processed_gwas(
            processed_gz,
            qc_rsids,
        )

        # GWAS Catalog API.
        payloads = fetch_catalog_payloads(accession)

        metadata = extract_general_metadata(payloads)

        candidates = extract_case_control_candidates(payloads)

        selected = choose_case_control_candidate(
            candidates,
            local_gwas_n,
        )

        # Save all candidates for audit.
        for rank, c in enumerate(
            sorted(
                candidates,
                key=lambda x: (
                    -discovery_bonus(x),
                    abs(x["total"] - safe_int(local_gwas_n))
                    if safe_int(local_gwas_n) is not None
                    else 10**18,
                ),
            ),
            1,
        ):
            all_candidates.append({
                "Phenotype": folder.name,
                "Accession": accession,
                "Local_GWAS_N": local_gwas_n,
                "Candidate_Rank": rank,
                "Cases": c["cases"],
                "Controls": c["controls"],
                "Total": c["total"],
                "Difference_From_Local_N": (
                    abs(c["total"] - safe_int(local_gwas_n))
                    if safe_int(local_gwas_n) is not None
                    else ""
                ),
                "Discovery_Bonus": discovery_bonus(c),
                "Source": c["source"],
                "Stage": c["stage"],
                "Evidence_Text": c["evidence_text"],
                "JSON_Path": c["json_path"],
                "API_URL": c["api_url"],
            })

        if selected:
            gwas_cases = selected["cases"]
            gwas_controls = selected["controls"]
            gwas_total = selected["total"]
            gwas_p = (
                gwas_cases / gwas_total
                if gwas_total
                else ""
            )

            case_control_source = selected["source"]
            case_control_stage = selected["stage"]
            case_control_evidence = selected["evidence_text"]
            case_control_url = selected["api_url"]

            if safe_int(local_gwas_n) is not None:
                n_difference = gwas_total - safe_int(local_gwas_n)
                n_exact_match = gwas_total == safe_int(local_gwas_n)
            else:
                n_difference = ""
                n_exact_match = ""

            case_control_status = (
                "FOUND_EXACT_N_MATCH"
                if n_exact_match is True
                else "FOUND_BUT_N_MISMATCH"
            )

        else:
            gwas_cases = ""
            gwas_controls = ""
            gwas_total = ""
            gwas_p = ""
            case_control_source = ""
            case_control_stage = ""
            case_control_evidence = ""
            case_control_url = ""
            n_difference = ""
            n_exact_match = ""
            case_control_status = "NOT_FOUND_IN_GWAS_CATALOG_API"

        print(f"  accession:        {accession or 'NOT FOUND'}")
        print(f"  PMID:             {metadata['pmid'] or 'NOT FOUND'}")
        print(f"  DOI:              {metadata['doi'] or 'NOT FOUND'}")

        print(
            f"  target:           N={target_n}, "
            f"cases={target_cases}, controls={target_controls}, "
            f"P={fmt_p(target_p)}"
        )

        print(f"  local GWAS N:     {local_gwas_n}")

        print(
            f"  GWAS Catalog:     cases={gwas_cases}, "
            f"controls={gwas_controls}, "
            f"N={gwas_total}, "
            f"P={fmt_p(gwas_p)}"
        )

        if selected:
            print(
                f"  case/control src: {case_control_source}; "
                f"stage={case_control_stage or 'not labelled'}"
            )
            print(
                f"  N comparison:     Catalog N - local N = "
                f"{n_difference}; exact={n_exact_match}"
            )
        else:
            print("  case/control src: NOT FOUND")

        print(
            f"  common SNPs:      {common_snps} "
            f"({processed_gz.name if processed_gz else 'NO .gz'})"
        )

        results.append({
            "Phenotype": folder.name,
            "Canonical_Phenotype": canonical,
            "Data_Type": "REAL_EMPIRICAL",

            "Accession": accession,
            "Accession_Source": accession_source,
            "PMID": metadata["pmid"],
            "DOI": metadata["doi"],
            "Catalog_Trait": metadata["trait"],
            "Catalog_Title": metadata["title"],

            "Target_N": target_n,
            "Target_Cases": target_cases,
            "Target_Controls": target_controls,
            "Target_P": target_p,
            "Target_Phenotype_Type": phenotype_type,

            "Processed_GWAS_File": (
                processed_gz.name if processed_gz else ""
            ),
            "Processed_GWAS_Rows": processed_rows,
            "GWAS_N_Local_File": local_gwas_n,

            "GWAS_Cases_Catalog": gwas_cases,
            "GWAS_Controls_Catalog": gwas_controls,
            "GWAS_N_CasesPlusControls": gwas_total,
            "GWAS_P_Catalog": gwas_p,

            "Catalog_N_Minus_Local_N": n_difference,
            "Catalog_N_Exact_Local_N": n_exact_match,

            "GWAS_CaseControl_Status": case_control_status,
            "GWAS_CaseControl_Source": case_control_source,
            "GWAS_CaseControl_Stage": case_control_stage,
            "GWAS_CaseControl_Evidence": case_control_evidence,
            "GWAS_CaseControl_API_URL": case_control_url,

            "QC_BIM_File": bim.name,
            "Common_SNPs": common_snps,

            "Catalog_Sample_Text": metadata["sample_text"],
            "Number_of_CaseControl_Candidates": len(candidates),
            "Processed_GZ_Status": processed_status,
        })

    # -----------------------------------------------------------------
    # Save main output.
    # -----------------------------------------------------------------
    fieldnames = list(results[0].keys())

    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Save all candidate pairs so ambiguous API results can be inspected.
    candidate_fields = [
        "Phenotype",
        "Accession",
        "Local_GWAS_N",
        "Candidate_Rank",
        "Cases",
        "Controls",
        "Total",
        "Difference_From_Local_N",
        "Discovery_Bonus",
        "Source",
        "Stage",
        "Evidence_Text",
        "JSON_Path",
        "API_URL",
    ]

    with OUT_CANDIDATES.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=candidate_fields,
        )
        writer.writeheader()
        writer.writerows(all_candidates)

    # -----------------------------------------------------------------
    # Human-readable report.
    # -----------------------------------------------------------------
    with OUT_TXT.open("w") as fh:
        fh.write("DATA1 — REAL GWAS CASE/CONTROL INFORMATION\n")
        fh.write("=" * 120 + "\n\n")

        for r in results:
            fh.write(f"Phenotype: {r['Phenotype']}\n")
            fh.write(
                f"  Accession: {r['Accession'] or 'NOT FOUND'}\n"
            )
            fh.write(
                f"  PMID: {r['PMID'] or 'NOT FOUND'}\n"
            )
            fh.write(
                f"  DOI: {r['DOI'] or 'NOT FOUND'}\n"
            )

            fh.write(
                f"  Target: N={r['Target_N']}, "
                f"cases={r['Target_Cases']}, "
                f"controls={r['Target_Controls']}, "
                f"P={fmt_p(r['Target_P'])}\n"
            )

            fh.write(
                f"  Discovery GWAS: local N={r['GWAS_N_Local_File']}, "
                f"cases={r['GWAS_Cases_Catalog']}, "
                f"controls={r['GWAS_Controls_Catalog']}, "
                f"cases+controls={r['GWAS_N_CasesPlusControls']}, "
                f"P={fmt_p(r['GWAS_P_Catalog'])}\n"
            )

            fh.write(
                f"  Case/control status: "
                f"{r['GWAS_CaseControl_Status']}\n"
            )

            fh.write(
                f"  Case/control source: "
                f"{r['GWAS_CaseControl_Source']}\n"
            )

            fh.write(
                f"  Case/control stage: "
                f"{r['GWAS_CaseControl_Stage']}\n"
            )

            fh.write(
                f"  Evidence: "
                f"{r['GWAS_CaseControl_Evidence']}\n"
            )

            fh.write(
                f"  Catalog N - local N: "
                f"{r['Catalog_N_Minus_Local_N']}\n"
            )

            fh.write(
                f"  Common SNPs "
                f"({r['Processed_GWAS_File']} vs {r['QC_BIM_File']}): "
                f"{r['Common_SNPs']}\n"
            )

            fh.write("\n")

    print("")
    print("=" * 108)
    print("DONE — REAL DATA ONLY")
    print("=" * 108)
    print(f"Wrote: {OUT_CSV.name}")
    print(f"Wrote: {OUT_TXT.name}")
    print(f"Wrote: {OUT_CANDIDATES.name}")
    print("")
    print("Now run:")
    print("  cat Data1_GWAS_Information_REAL.txt")


if __name__ == "__main__":
    main()
 
