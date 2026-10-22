 
# -*- coding: utf-8 -*-

"""
Method22-LiabilityScaleMetadataAudit.py

Purpose
-------
Build the metadata needed to convert observed-scale binary-trait SNP heritability
to the liability scale.

The script does FOUR things:

1. LOCAL TARGET SAMPLE AUDIT
   - Reads each phenotype's QC .fam file.
   - Reads fold-level training .fam files.
   - Reports cases, controls, N, and sample prevalence P.

2. GWAS ACCESSION + SAMPLE-SIZE AUDIT
   - Finds GCST accessions in phenotype folders.
   - Inspects accession-named summary-statistic files for an N column.
   - Queries the NHGRI-EBI GWAS Catalog APIs.
   - Tries to recover study trait, PubMed ID, title, sample description,
     ancestry, cases, controls, discovery N, and GWAS sample prevalence P.

3. POPULATION PREVALENCE K CANDIDATE SEARCH
   - Uses the linked PubMed record through Europe PMC.
   - Searches abstract and, when available, open-access full text for passages
     mentioning liability-scale/population prevalence/prevalence assumptions.
   - Extracts candidate prevalence values and surrounding text.
   - DOES NOT silently accept a value as K. Every K candidate is flagged
     REVIEW_REQUIRED because population prevalence is a scientific assumption.

4. STATUS TABLE
   - Tells you, phenotype by phenotype, which liability-scale inputs are
     available locally, which were recovered online, and what remains missing.

IMPORTANT
---------
P = sample prevalence = cases / (cases + controls).
K = population prevalence in the underlying population.

For individual-level genotype methods:
    P must come from the exact analytic target sample/fold.

For summary-statistic methods:
    P should come from the GWAS discovery sample.

K is NOT inferred from the target .fam file and is NOT the GWAS case fraction.
The script only proposes K candidates if a publication explicitly reports a
prevalence assumption in relevant context. Those candidates must be reviewed.

Run
---
From:
    /data/ascher01/uqmmune1/finalheritability

Use:
    python Method22-LiabilityScaleMetadataAudit.py

Optional:
    python Method22-LiabilityScaleMetadataAudit.py | tee Method22_AUDIT.txt

Outputs
-------
Method22_local_prevalence.csv
Method22_gwas_metadata.csv
Method22_k_candidates.csv
Method22_liability_status.csv
Method22_metadata_cache/

Dependencies
------------
Python standard library + pandas + numpy.
No API key is required.
Internet access is required for GWAS Catalog / Europe PMC lookups.
"""

import os
import re
import sys
import json
import time
import gzip
import hashlib
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(".").resolve()

BINARY_PHENOTYPES = [
    "asthma",
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
    "asthma": "Asthma",
    "depression": "Depression",
    "gastro_oesophageal_reflux_gord_gastric_reflux": "GORD",
    "high_cholesterol": "High cholesterol",
    "hypertension": "Hypertension",
    "hypothyroidism_myxoedema": "Hypothyroidism",
    "irritable_bowel_syndrome": "Irritable bowel syndrome",
    "migraine": "Migraine",
    "osteoarthritis": "Osteoarthritis",
}

CACHE_DIR = ROOT / "Method22_metadata_cache"
CACHE_DIR.mkdir(exist_ok=True)

MAX_LOCAL_TEXT_BYTES = 10_000_000
MAX_GWAS_ROWS_FOR_N_AUDIT = 100_000
HTTP_TIMEOUT = 30
REQUEST_SLEEP = 0.12  # GWAS Catalog documents a 15 requests/s throttle.

USER_AGENT = (
    "Method22-LiabilityScaleMetadataAudit/1.0 "
    "(academic reproducibility audit)"
)

GCST_RE = re.compile(r"(?<![A-Za-z0-9])GCST\d+", re.I)


# =============================================================================
# SMALL HELPERS
# =============================================================================

def clean_text(x):
    if x is None:
        return ""
    x = str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def safe_float(x):
    try:
        y = float(x)
        return y if np.isfinite(y) else np.nan
    except Exception:
        return np.nan


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def json_dump(path, obj):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def unique_preserve(values):
    seen = set()
    out = []
    for x in values:
        key = str(x)
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


# =============================================================================
# HTTP + CACHE
# =============================================================================

def http_get(url, params=None, accept="application/json", cache_name=None):
    if params:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)

    if cache_name:
        cache_path = CACHE_DIR / cache_name
        if cache_path.exists():
            try:
                return cache_path.read_bytes()
            except Exception:
                pass
    else:
        cache_path = None

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": accept,
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as response:
            raw = response.read()

        if cache_path is not None:
            try:
                cache_path.write_bytes(raw)
            except Exception:
                pass

        time.sleep(REQUEST_SLEEP)
        return raw

    except Exception as exc:
        print(f"[HTTP WARNING] {url} -> {exc}", file=sys.stderr)
        return None


def get_json(url, params=None, cache_name=None):
    raw = http_get(
        url,
        params=params,
        accept="application/json",
        cache_name=cache_name,
    )
    if raw is None:
        return None

    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except Exception as exc:
        print(f"[JSON WARNING] {url} -> {exc}", file=sys.stderr)
        return None


# =============================================================================
# RECURSIVE JSON FLATTENING
# =============================================================================

def flatten_json(obj, prefix=""):
    """
    Return [(path, scalar_value), ...] for arbitrary JSON.
    This deliberately avoids depending on one exact GWAS Catalog JSON schema.
    """
    out = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.extend(flatten_json(value, path))

    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            path = f"{prefix}[{i}]"
            out.extend(flatten_json(value, path))

    else:
        out.append((prefix, obj))

    return out


def values_by_key_pattern(obj, patterns):
    patterns = [p.lower() for p in patterns]
    vals = []

    for path, value in flatten_json(obj):
        low = path.lower()
        if any(p in low for p in patterns):
            text = clean_text(value)
            if text:
                vals.append(text)

    return unique_preserve(vals)


# =============================================================================
# LOCAL FAM AUDIT
# =============================================================================

def read_fam_counts(path):
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None)
    except Exception as exc:
        return {
            "fam_file": str(path),
            "status": f"READ_ERROR: {exc}",
        }

    if df.shape[1] < 6:
        return {
            "fam_file": str(path),
            "status": "INVALID_FAM_LT_6_COLUMNS",
        }

    y = pd.to_numeric(df.iloc[:, 5], errors="coerce")
    vals = set(y.dropna().unique())

    result = {
        "fam_file": str(path),
        "fam_rows": len(df),
        "phenotype_counts": json.dumps(
            {
                str(k): int(v)
                for k, v in y.value_counts(dropna=False).sort_index().items()
            }
        ),
        "cases": np.nan,
        "controls": np.nan,
        "analytic_n": np.nan,
        "sample_prevalence_P": np.nan,
        "coding": "UNKNOWN",
        "status": "OK",
    }

    # Standard PLINK binary coding: 1=control, 2=case; 0/-9 missing.
    if 2 in vals and vals.issubset({-9, 0, 1, 2}):
        cases = int((y == 2).sum())
        controls = int((y == 1).sum())
        n = cases + controls

        result.update({
            "cases": cases,
            "controls": controls,
            "analytic_n": n,
            "sample_prevalence_P": cases / n if n else np.nan,
            "coding": "PLINK_1_CONTROL_2_CASE",
        })

    elif vals.issubset({0, 1}) and len(vals) > 1:
        cases = int((y == 1).sum())
        controls = int((y == 0).sum())
        n = cases + controls

        result.update({
            "cases": cases,
            "controls": controls,
            "analytic_n": n,
            "sample_prevalence_P": cases / n if n else np.nan,
            "coding": "POSSIBLE_0_CONTROL_1_CASE_VERIFY",
        })

    return result


def sample_signature_from_fam(path):
    """
    Signature based on IID + phenotype, allowing copied FAMs to be deduplicated.
    """
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, usecols=[1, 5])
        payload = "\n".join(
            f"{a}\t{b}"
            for a, b in zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str))
        )
        return sha256_text(payload)
    except Exception:
        return "NA"


def choose_full_fam(phenotype_dir, phenotype):
    candidates = [
        phenotype_dir / f"{phenotype}_QC.fam",
        phenotype_dir / f"{phenotype}.QC.fam",
        phenotype_dir / f"{phenotype}.fam",
    ]

    for path in candidates:
        if path.is_file():
            counts = read_fam_counts(path)
            if np.isfinite(safe_float(counts.get("sample_prevalence_P"))):
                return path

    # Fallback: shallow top-level FAM with usable binary phenotype.
    for path in sorted(phenotype_dir.glob("*.fam")):
        counts = read_fam_counts(path)
        if np.isfinite(safe_float(counts.get("sample_prevalence_P"))):
            return path

    return None


def fold_fam_candidates(fold_dir):
    """
    Prefer canonical fold-level files. If only nested copied files exist,
    deduplicate identical sample/phenotype sets and report the shallowest path.
    """
    exact = [
        fold_dir / "train_data.QC.fam",
        fold_dir / "train_data.fam",
        fold_dir / "train_data.QC.clumped.pruned.fam",
        fold_dir / "train_data.clumped.pruned.fam",
    ]

    for path in exact:
        if path.is_file():
            return [path]

    recursive = sorted(
        fold_dir.rglob("*.fam"),
        key=lambda p: (len(p.parts), len(str(p)), str(p)),
    )

    usable = []
    seen_signatures = set()

    for path in recursive:
        counts = read_fam_counts(path)
        p = safe_float(counts.get("sample_prevalence_P"))
        if not np.isfinite(p):
            continue

        sig = sample_signature_from_fam(path)
        if sig in seen_signatures:
            continue

        seen_signatures.add(sig)
        usable.append(path)

    return usable


# =============================================================================
# GCST ACCESSION DISCOVERY
# =============================================================================

TEXT_EXTENSIONS = {
    ".txt", ".csv", ".tsv", ".py", ".sh", ".log", ".out", ".err",
    ".ldsc", ".ldak", ".ma", ".json", ".yaml", ".yml",
}


def discover_gcst_accessions(phenotype_dir):
    """
    Find GCST IDs in:
      - top-level filenames
      - top-level/small text files
      - filenames up to two directory levels down
    We do NOT crawl huge Fold_* result trees for text content.
    """
    found = {}

    def add(acc, source):
        acc = acc.upper()
        found.setdefault(acc, set()).add(str(source))

    # Top-level filenames.
    for path in phenotype_dir.iterdir():
        for acc in GCST_RE.findall(path.name):
            add(acc, f"filename:{path.name}")

    # Filenames up to depth 2.
    base_depth = len(phenotype_dir.parts)
    for root, dirs, files in os.walk(phenotype_dir):
        root_path = Path(root)
        depth = len(root_path.parts) - base_depth

        # Do not recursively explode through PRS/fold result trees.
        if depth >= 2:
            dirs[:] = []

        for name in files:
            for acc in GCST_RE.findall(name):
                add(acc, f"filename:{root_path / name}")

    # Small top-level text-like files.
    for path in phenotype_dir.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue

        try:
            if path.stat().st_size > MAX_LOCAL_TEXT_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for acc in GCST_RE.findall(text):
            add(acc, f"text:{path.name}")

    return {
        acc: sorted(sources)
        for acc, sources in sorted(found.items())
    }


# =============================================================================
# LOCAL SUMMARY-STATISTIC N AUDIT
# =============================================================================

def accession_files(phenotype_dir, accession):
    out = []

    for path in phenotype_dir.iterdir():
        if path.is_file() and accession.upper() in path.name.upper():
            out.append(path)

    return sorted(out)


def read_header_and_n_values(path, max_rows=MAX_GWAS_ROWS_FOR_N_AUDIT):
    """
    Lightweight reader for accession-named summary statistics.
    Reports N min/median/max from first max_rows data rows.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    mode = "rt"

    try:
        handle = opener(path, mode, encoding="utf-8", errors="ignore")
    except TypeError:
        handle = opener(path, mode, errors="ignore")

    with handle:
        header = handle.readline().strip()

        if not header:
            return {
                "local_gwas_file": str(path),
                "local_header": "",
                "local_N_column": "",
                "local_N_values_read": 0,
                "local_N_min": np.nan,
                "local_N_median": np.nan,
                "local_N_max": np.nan,
            }

        if "\t" in header:
            cols = header.split("\t")
            delimiter = "\t"
        elif "," in header:
            cols = header.split(",")
            delimiter = ","
        else:
            cols = re.split(r"\s+", header)
            delimiter = None

        norm = [c.strip().lower() for c in cols]

        n_index = None
        for candidate in ["n", "sample_size", "samplesize", "total_n"]:
            if candidate in norm:
                n_index = norm.index(candidate)
                break

        values = []

        if n_index is not None:
            for i, line in enumerate(handle):
                if i >= max_rows:
                    break

                if delimiter is None:
                    fields = re.split(r"\s+", line.strip())
                else:
                    fields = line.rstrip("\n").split(delimiter)

                if n_index >= len(fields):
                    continue

                v = safe_float(fields[n_index])
                if np.isfinite(v):
                    values.append(v)

        result = {
            "local_gwas_file": str(path),
            "local_header": header[:1000],
            "local_N_column": cols[n_index] if n_index is not None else "",
            "local_N_values_read": len(values),
            "local_N_min": np.nan,
            "local_N_median": np.nan,
            "local_N_max": np.nan,
        }

        if values:
            result["local_N_min"] = float(np.min(values))
            result["local_N_median"] = float(np.median(values))
            result["local_N_max"] = float(np.max(values))

        return result


# =============================================================================
# GWAS CATALOG LOOKUP
# =============================================================================

GWAS_V2_STUDIES = "https://www.ebi.ac.uk/gwas/rest/api/v2/studies"
GWAS_V2_ANCESTRY = "https://www.ebi.ac.uk/gwas/rest/api/v2/studies/{accession}/ancestries"
GWAS_V1_DIRECT = "https://www.ebi.ac.uk/gwas/rest/api/studies/{accession}"
GWAS_SUMSTATS_STUDY = (
    "https://www.ebi.ac.uk/gwas/summary-statistics/api/studies/{accession}"
)


def lookup_gwas_catalog(accession):
    """
    Query several official GWAS Catalog resources.
    The outputs are kept raw + flattened because schemas can differ.
    """
    result = {
        "v2": None,
        "v2_ancestry": None,
        "v1": None,
        "sumstats": None,
    }

    result["v2"] = get_json(
        GWAS_V2_STUDIES,
        params={"accession_id": accession},
        cache_name=f"{accession}_gwas_v2.json",
    )

    result["v2_ancestry"] = get_json(
        GWAS_V2_ANCESTRY.format(accession=accession),
        cache_name=f"{accession}_ancestry_v2.json",
    )

    result["v1"] = get_json(
        GWAS_V1_DIRECT.format(accession=accession),
        cache_name=f"{accession}_gwas_v1.json",
    )

    result["sumstats"] = get_json(
        GWAS_SUMSTATS_STUDY.format(accession=accession),
        cache_name=f"{accession}_sumstats.json",
    )

    return result


def combine_json_strings(api_bundle):
    strings = []
    for source_name, obj in api_bundle.items():
        if obj is None:
            continue
        for path, value in flatten_json(obj):
            text = clean_text(value)
            if text:
                strings.append((source_name, path, text))
    return strings


def extract_pubmed_ids(api_bundle):
    ids = []

    for source_name, obj in api_bundle.items():
        if obj is None:
            continue

        for path, value in flatten_json(obj):
            low = path.lower()

            if "pubmed" not in low and "pmid" not in low:
                continue

            text = clean_text(value)
            for match in re.findall(r"\b\d{6,9}\b", text):
                ids.append(match)

    return unique_preserve(ids)


def extract_field_text(api_bundle, key_patterns):
    vals = []

    for source_name, obj in api_bundle.items():
        if obj is None:
            continue

        for path, value in flatten_json(obj):
            low = path.lower()
            if any(pattern.lower() in low for pattern in key_patterns):
                text = clean_text(value)
                if text:
                    vals.append(text)

    return unique_preserve(vals)


def parse_case_control_candidates(strings):
    """
    Search every text scalar from API metadata for phrases containing
    explicit case/control counts.
    """
    candidates = []

    case_patterns = [
        r"(\d[\d,]*)\s+(?:[\w\-]+\s+){0,8}cases?\b",
        r"cases?\s*[:=]\s*(\d[\d,]*)",
    ]
    control_patterns = [
        r"(\d[\d,]*)\s+(?:[\w\-]+\s+){0,8}controls?\b",
        r"controls?\s*[:=]\s*(\d[\d,]*)",
    ]

    for source_name, path, text in strings:
        low = text.lower()

        if "case" not in low or "control" not in low:
            continue

        case_vals = []
        control_vals = []

        for pattern in case_patterns:
            case_vals.extend(re.findall(pattern, text, flags=re.I))

        for pattern in control_patterns:
            control_vals.extend(re.findall(pattern, text, flags=re.I))

        case_nums = []
        control_nums = []

        for x in case_vals:
            try:
                case_nums.append(int(x.replace(",", "")))
            except Exception:
                pass

        for x in control_vals:
            try:
                control_nums.append(int(x.replace(",", "")))
            except Exception:
                pass

        if case_nums and control_nums:
            # Prefer the largest values in the same metadata string.
            cases = max(case_nums)
            controls = max(control_nums)

            candidates.append({
                "source": source_name,
                "json_path": path,
                "cases": cases,
                "controls": controls,
                "n": cases + controls,
                "P": cases / (cases + controls),
                "context": text[:2000],
            })

    # Highest-N candidate first.
    candidates = sorted(candidates, key=lambda x: x["n"], reverse=True)
    return candidates


def extract_numeric_sample_size(api_bundle):
    """
    Extract likely discovery/initial sample-size values from structured keys.
    These are supportive metadata only. Case/control counts are preferred
    for calculating P.
    """
    vals = []

    patterns = [
        "initialsamplesize",
        "initial_sample_size",
        "discoverysamplesize",
        "discovery_sample_size",
        "sample_size",
        "samplesize",
    ]

    for source_name, obj in api_bundle.items():
        if obj is None:
            continue

        for path, value in flatten_json(obj):
            low = path.lower().replace("-", "_")

            if not any(p in low for p in patterns):
                continue

            text = clean_text(value)

            # Extract all sufficiently large integers.
            numbers = re.findall(r"\b\d[\d,]{2,}\b", text)
            for n in numbers:
                try:
                    vals.append(int(n.replace(",", "")))
                except Exception:
                    pass

    return max(vals) if vals else np.nan


# =============================================================================
# EUROPE PMC LOOKUP + K-CANDIDATE EXTRACTION
# =============================================================================

EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPE_PMC_FULLTEXT = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
)


def lookup_europe_pmc(pmid):
    data = get_json(
        EUROPE_PMC_SEARCH,
        params={
            "query": f"EXT_ID:{pmid} AND SRC:MED",
            "format": "json",
            "resultType": "core",
            "pageSize": 1,
        },
        cache_name=f"PMID_{pmid}_europepmc.json",
    )

    if not data:
        return {
            "pmid": pmid,
            "title": "",
            "abstract": "",
            "pmcid": "",
            "fulltext": "",
        }

    results = (
        data.get("resultList", {})
        .get("result", [])
    )

    if not results:
        return {
            "pmid": pmid,
            "title": "",
            "abstract": "",
            "pmcid": "",
            "fulltext": "",
        }

    rec = results[0]

    title = clean_text(rec.get("title", ""))
    abstract = clean_text(rec.get("abstractText", ""))
    pmcid = clean_text(rec.get("pmcid", ""))

    fulltext = ""

    if pmcid:
        raw = http_get(
            EUROPE_PMC_FULLTEXT.format(pmcid=pmcid),
            accept="application/xml,text/xml",
            cache_name=f"{pmcid}_fulltext.xml",
        )

        if raw:
            try:
                root = ET.fromstring(raw)
                fulltext = clean_text(" ".join(root.itertext()))
            except Exception:
                fulltext = ""

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "pmcid": pmcid,
        "fulltext": fulltext,
    }


def split_sentences(text):
    text = clean_text(text)
    if not text:
        return []

    # Good enough for prevalence-context extraction.
    pieces = re.split(r"(?<=[.!?;])\s+", text)
    return [p.strip() for p in pieces if p.strip()]


def prevalence_values_in_sentence(sentence):
    values = []

    # Percentages.
    for x in re.findall(r"(?<!\d)(\d{1,2}(?:\.\d+)?)\s*%", sentence):
        try:
            k = float(x) / 100.0
            if 0 < k < 0.80:
                values.append(k)
        except Exception:
            pass

    # Decimal prevalence proportions.
    for x in re.findall(r"(?<!\d)(0\.\d{1,4})(?!\d)", sentence):
        try:
            k = float(x)
            if 0 < k < 0.80:
                values.append(k)
        except Exception:
            pass

    return unique_preserve(values)


def score_k_context(sentence):
    s = sentence.lower()
    score = 0

    if "population prevalence" in s:
        score += 5

    if (
        "assumed prevalence" in s
        or "prevalence was assumed" in s
        or "assuming a prevalence" in s
        or "assuming prevalence" in s
    ):
        score += 5

    if "liability scale" in s or "liability-scale" in s:
        score += 4

    if "prevalence" in s:
        score += 1

    if "sample prevalence" in s:
        score -= 5

    if "case-control ratio" in s or "case control ratio" in s:
        score -= 3

    if "incidence" in s and "prevalence" not in s:
        score -= 3

    return score


def extract_k_candidates(article):
    """
    Search abstract/full text for candidate K values.

    IMPORTANT: Output is a candidate list, NOT an accepted K.
    """
    source_texts = []

    if article.get("abstract"):
        source_texts.append(("abstract", article["abstract"]))

    if article.get("fulltext"):
        source_texts.append(("fulltext", article["fulltext"]))

    candidates = []

    for source, text in source_texts:
        sentences = split_sentences(text)

        for i, sentence in enumerate(sentences):
            low = sentence.lower()

            if (
                "prevalence" not in low
                and "liability" not in low
            ):
                continue

            # Include neighbouring sentence to preserve definitions.
            context_parts = []
            if i > 0:
                context_parts.append(sentences[i - 1])
            context_parts.append(sentence)
            if i + 1 < len(sentences):
                context_parts.append(sentences[i + 1])

            context = clean_text(" ".join(context_parts))
            values = prevalence_values_in_sentence(context)

            if not values:
                continue

            score = score_k_context(sentence)

            for k in values:
                candidates.append({
                    "source": source,
                    "K_candidate": k,
                    "score": score,
                    "context": context[:2500],
                })

    # Deduplicate same K + context.
    seen = set()
    out = []

    for row in sorted(candidates, key=lambda x: x["score"], reverse=True):
        key = (
            round(row["K_candidate"], 6),
            row["context"][:500],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)

    return out


# =============================================================================
# MAIN: LOCAL P
# =============================================================================

print("=" * 110)
print("METHOD 22 — LIABILITY-SCALE METADATA AUDIT")
print("=" * 110)
print("Root:", ROOT)
print()

local_rows = []

for phenotype in BINARY_PHENOTYPES:
    pdir = ROOT / phenotype

    if not pdir.is_dir():
        print(f"[LOCAL] Missing phenotype directory: {phenotype}")
        continue

    label = PHENOTYPE_LABELS[phenotype]

    full_fam = choose_full_fam(pdir, phenotype)

    if full_fam:
        info = read_fam_counts(full_fam)
        local_rows.append({
            "phenotype": phenotype,
            "phenotype_label": label,
            "sample_scope": "FULL_QC_SAMPLE",
            "fold": "",
            **info,
        })

    for fold in range(5):
        fold_dir = pdir / f"Fold_{fold}"

        if not fold_dir.is_dir():
            continue

        candidates = fold_fam_candidates(fold_dir)

        if not candidates:
            local_rows.append({
                "phenotype": phenotype,
                "phenotype_label": label,
                "sample_scope": "FOLD_TRAINING",
                "fold": fold,
                "fam_file": "",
                "fam_rows": np.nan,
                "phenotype_counts": "",
                "cases": np.nan,
                "controls": np.nan,
                "analytic_n": np.nan,
                "sample_prevalence_P": np.nan,
                "coding": "",
                "status": "NO_USABLE_FAM_FOUND",
            })
            continue

        # If multiple unique sample signatures exist, retain all;
        # normally there should be one exact training sample per fold.
        for j, fam in enumerate(candidates, start=1):
            info = read_fam_counts(fam)

            local_rows.append({
                "phenotype": phenotype,
                "phenotype_label": label,
                "sample_scope": (
                    "FOLD_TRAINING"
                    if len(candidates) == 1
                    else f"FOLD_TRAINING_SIGNATURE_{j}"
                ),
                "fold": fold,
                **info,
            })

local_df = pd.DataFrame(local_rows)
local_out = ROOT / "Method22_local_prevalence.csv"
local_df.to_csv(local_out, index=False)

print("[LOCAL] Wrote:", local_out.name)
print("[LOCAL] Rows :", len(local_df))
print()


# =============================================================================
# MAIN: ACCESSIONS + API METADATA + K CANDIDATES
# =============================================================================

metadata_rows = []
k_rows = []

for phenotype in BINARY_PHENOTYPES:
    pdir = ROOT / phenotype

    if not pdir.is_dir():
        continue

    label = PHENOTYPE_LABELS[phenotype]
    accessions = discover_gcst_accessions(pdir)

    print(f"[GWAS] {label}: {len(accessions)} GCST accession(s)")

    if not accessions:
        metadata_rows.append({
            "phenotype": phenotype,
            "phenotype_label": label,
            "accession": "",
            "accession_sources": "",
            "local_gwas_file": "",
            "local_N_min": np.nan,
            "local_N_median": np.nan,
            "local_N_max": np.nan,
            "catalog_pubmed_ids": "",
            "catalog_title": "",
            "catalog_trait": "",
            "catalog_sample_description": "",
            "catalog_discovery_N": np.nan,
            "catalog_cases": np.nan,
            "catalog_controls": np.nan,
            "catalog_sample_prevalence_P": np.nan,
            "catalog_ancestry": "",
            "metadata_status": "NO_GCST_ACCESSION_FOUND",
        })
        continue

    for accession, sources in accessions.items():
        print(f"       {accession}")

        # -----------------------------
        # Local accession-named GWAS N
        # -----------------------------
        local_file_records = []

        for path in accession_files(pdir, accession):
            try:
                local_file_records.append(
                    read_header_and_n_values(path)
                )
            except Exception as exc:
                local_file_records.append({
                    "local_gwas_file": str(path),
                    "local_header": "",
                    "local_N_column": "",
                    "local_N_values_read": 0,
                    "local_N_min": np.nan,
                    "local_N_median": np.nan,
                    "local_N_max": np.nan,
                })

        if local_file_records:
            # Prefer file with most N observations.
            local_best = sorted(
                local_file_records,
                key=lambda r: r.get("local_N_values_read", 0),
                reverse=True,
            )[0]
        else:
            local_best = {
                "local_gwas_file": "",
                "local_header": "",
                "local_N_column": "",
                "local_N_values_read": 0,
                "local_N_min": np.nan,
                "local_N_median": np.nan,
                "local_N_max": np.nan,
            }

        # -----------------------------
        # Online GWAS Catalog
        # -----------------------------
        api_bundle = lookup_gwas_catalog(accession)
        strings = combine_json_strings(api_bundle)

        pubmed_ids = extract_pubmed_ids(api_bundle)

        titles = extract_field_text(
            api_bundle,
            ["title", "publication.title"],
        )

        traits = extract_field_text(
            api_bundle,
            [
                "diseasetrait",
                "disease_trait",
                "reportedtrait",
                "reported_trait",
                ".trait",
            ],
        )

        sample_descriptions = extract_field_text(
            api_bundle,
            [
                "initialsamplesize",
                "initial_sample_size",
                "discoverysample",
                "discovery_sample",
                "sampledescription",
                "sample_description",
            ],
        )

        ancestry_vals = []
        ancestry_obj = api_bundle.get("v2_ancestry")

        if ancestry_obj is not None:
            for path, value in flatten_json(ancestry_obj):
                if any(
                    key in path.lower()
                    for key in [
                        "ancestry",
                        "broad",
                        "country",
                        "population",
                    ]
                ):
                    text = clean_text(value)
                    if text:
                        ancestry_vals.append(text)

        ancestry_vals = unique_preserve(ancestry_vals)

        cc_candidates = parse_case_control_candidates(strings)

        if cc_candidates:
            best_cc = cc_candidates[0]
            catalog_cases = best_cc["cases"]
            catalog_controls = best_cc["controls"]
            catalog_p = best_cc["P"]
            catalog_n = best_cc["n"]
            cc_context = best_cc["context"]
            cc_source = (
                f"{best_cc['source']}:{best_cc['json_path']}"
            )
        else:
            catalog_cases = np.nan
            catalog_controls = np.nan
            catalog_p = np.nan
            catalog_n = extract_numeric_sample_size(api_bundle)
            cc_context = ""
            cc_source = ""

        metadata_rows.append({
            "phenotype": phenotype,
            "phenotype_label": label,
            "accession": accession,
            "accession_sources": " || ".join(sources),
            "local_gwas_file": local_best.get("local_gwas_file", ""),
            "local_header": local_best.get("local_header", ""),
            "local_N_column": local_best.get("local_N_column", ""),
            "local_N_values_read": local_best.get("local_N_values_read", 0),
            "local_N_min": local_best.get("local_N_min", np.nan),
            "local_N_median": local_best.get("local_N_median", np.nan),
            "local_N_max": local_best.get("local_N_max", np.nan),
            "catalog_pubmed_ids": ";".join(pubmed_ids),
            "catalog_title": " || ".join(titles[:5]),
            "catalog_trait": " || ".join(traits[:10]),
            "catalog_sample_description": " || ".join(sample_descriptions[:10]),
            "catalog_discovery_N": catalog_n,
            "catalog_cases": catalog_cases,
            "catalog_controls": catalog_controls,
            "catalog_sample_prevalence_P": catalog_p,
            "case_control_metadata_source": cc_source,
            "case_control_context": cc_context,
            "catalog_ancestry": " || ".join(ancestry_vals[:20]),
            "metadata_status": (
                "CASE_CONTROL_RECOVERED"
                if np.isfinite(safe_float(catalog_p))
                else "CASE_CONTROL_NOT_RECOVERED"
            ),
        })

        # -----------------------------
        # Linked publication -> K search
        # -----------------------------
        if not pubmed_ids:
            k_rows.append({
                "phenotype": phenotype,
                "phenotype_label": label,
                "accession": accession,
                "pmid": "",
                "pmcid": "",
                "paper_title": "",
                "K_candidate": np.nan,
                "candidate_score": np.nan,
                "candidate_source": "",
                "candidate_context": "",
                "K_status": "NO_LINKED_PUBMED_ID",
            })
            continue

        for pmid in pubmed_ids:
            article = lookup_europe_pmc(pmid)
            candidates = extract_k_candidates(article)

            if not candidates:
                k_rows.append({
                    "phenotype": phenotype,
                    "phenotype_label": label,
                    "accession": accession,
                    "pmid": pmid,
                    "pmcid": article.get("pmcid", ""),
                    "paper_title": article.get("title", ""),
                    "K_candidate": np.nan,
                    "candidate_score": np.nan,
                    "candidate_source": "",
                    "candidate_context": "",
                    "K_status": "NO_K_CANDIDATE_FOUND",
                })

            else:
                for cand in candidates[:20]:
                    k_rows.append({
                        "phenotype": phenotype,
                        "phenotype_label": label,
                        "accession": accession,
                        "pmid": pmid,
                        "pmcid": article.get("pmcid", ""),
                        "paper_title": article.get("title", ""),
                        "K_candidate": cand["K_candidate"],
                        "candidate_score": cand["score"],
                        "candidate_source": cand["source"],
                        "candidate_context": cand["context"],
                        "K_status": "REVIEW_REQUIRED",
                    })


metadata_df = pd.DataFrame(metadata_rows)
metadata_out = ROOT / "Method22_gwas_metadata.csv"
metadata_df.to_csv(metadata_out, index=False)

k_df = pd.DataFrame(k_rows)
k_out = ROOT / "Method22_k_candidates.csv"
k_df.to_csv(k_out, index=False)

print()
print("[GWAS] Wrote:", metadata_out.name)
print("[K]    Wrote:", k_out.name)
print()


# =============================================================================
# STATUS TABLE
# =============================================================================

status_rows = []

for phenotype in BINARY_PHENOTYPES:
    label = PHENOTYPE_LABELS[phenotype]

    loc = local_df[
        local_df["phenotype"] == phenotype
    ].copy() if not local_df.empty else pd.DataFrame()

    met = metadata_df[
        metadata_df["phenotype"] == phenotype
    ].copy() if not metadata_df.empty else pd.DataFrame()

    kk = k_df[
        k_df["phenotype"] == phenotype
    ].copy() if not k_df.empty else pd.DataFrame()

    full = (
        loc[loc["sample_scope"] == "FULL_QC_SAMPLE"]
        if not loc.empty
        else pd.DataFrame()
    )

    if not full.empty:
        full_cases = safe_float(full.iloc[0].get("cases"))
        full_controls = safe_float(full.iloc[0].get("controls"))
        full_n = safe_float(full.iloc[0].get("analytic_n"))
        full_p = safe_float(full.iloc[0].get("sample_prevalence_P"))
    else:
        full_cases = full_controls = full_n = full_p = np.nan

    fold_rows = (
        loc[loc["sample_scope"].astype(str).str.startswith("FOLD_TRAINING")]
        if not loc.empty
        else pd.DataFrame()
    )

    fold_ps = (
        pd.to_numeric(
            fold_rows["sample_prevalence_P"],
            errors="coerce",
        ).dropna().tolist()
        if not fold_rows.empty
        else []
    )

    gwas_p_available = False
    accessions = []

    if not met.empty:
        accessions = [
            x for x in met["accession"].astype(str).tolist()
            if x
        ]

        gwas_ps = pd.to_numeric(
            met["catalog_sample_prevalence_P"],
            errors="coerce",
        )

        gwas_p_available = bool(gwas_ps.notna().any())

    strong_k = pd.DataFrame()

    if not kk.empty:
        strong_k = kk[
            pd.to_numeric(
                kk["candidate_score"],
                errors="coerce",
            ) >= 4
        ].copy()

    status_rows.append({
        "phenotype": phenotype,
        "phenotype_label": label,
        "full_cases": full_cases,
        "full_controls": full_controls,
        "full_N": full_n,
        "full_target_P": full_p,
        "fold_P_min": min(fold_ps) if fold_ps else np.nan,
        "fold_P_max": max(fold_ps) if fold_ps else np.nan,
        "GCST_accessions": ";".join(unique_preserve(accessions)),
        "GWAS_discovery_P_recovered": "YES" if gwas_p_available else "NO",
        "strong_K_candidate_found": "YES" if not strong_k.empty else "NO",
        "K_candidates_need_manual_review": (
            int(len(strong_k))
            if not strong_k.empty
            else 0
        ),
        "liability_status": (
            "TARGET_P_AVAILABLE__GWAS_P_AVAILABLE__K_REVIEW_NEEDED"
            if np.isfinite(full_p) and gwas_p_available
            else
            "TARGET_P_AVAILABLE__GWAS_P_MISSING__K_REVIEW_NEEDED"
            if np.isfinite(full_p)
            else
            "TARGET_P_MISSING__REVIEW_REQUIRED"
        ),
    })


status_df = pd.DataFrame(status_rows)
status_out = ROOT / "Method22_liability_status.csv"
status_df.to_csv(status_out, index=False)

print("[STATUS] Wrote:", status_out.name)
print()
print("=" * 110)
print("STATUS SUMMARY")
print("=" * 110)

display_cols = [
    "phenotype_label",
    "full_cases",
    "full_controls",
    "full_N",
    "full_target_P",
    "GCST_accessions",
    "GWAS_discovery_P_recovered",
    "strong_K_candidate_found",
    "K_candidates_need_manual_review",
]

print(
    status_df[display_cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print()
print("=" * 110)
print("INTERPRETATION")
print("=" * 110)
print("""
1. full_target_P and fold-level P values are valid for individual-level
   genotype analyses using those exact samples.

2. catalog_sample_prevalence_P is used only when explicit discovery
   cases/controls were recovered from GWAS metadata.

3. K_candidate is NOT automatically accepted as population prevalence K.
   Open Method22_k_candidates.csv and review the candidate_context column.

4. A strong candidate usually has text such as:
       "assuming a population prevalence of 16%"
   or:
       "liability-scale heritability ... prevalence 0.16"

5. If the linked GWAS publication does not report a liability/population
   prevalence assumption, K must be obtained from a separate epidemiological
   or heritability reference. That is scientifically normal.

6. Do NOT use the target case fraction P as K.

7. Do NOT convert BMI; liability-scale transformation is only for the
   binary phenotypes.

NEXT STEP:
After K is manually validated for each phenotype, create a small curated file:

    Phenotype,K,K_source,PMID_or_DOI,Notes

and only then apply the observed-to-liability transformation.
""")

print("DONE.")
 
