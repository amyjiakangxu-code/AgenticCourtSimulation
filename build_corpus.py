#!/usr/bin/env python3
"""
build_corpus.py

Converts three Canadian legal data sources into a single JSONL corpus:
  1. HuggingFace  a2aj/canadian-laws      (statutes, one chunk per section)
  2. HuggingFace  a2aj/canadian-case-law  (cases, 10% stratified sample,
                                           chunked by semantic section)
  3. [DISABLED]   SCC "Cases in Brief" PDFs — superseded by canadian-case-law

Output : legal_corpus.jsonl
Sanity : sanity_check.json  (one statute + one case law record, full text)
"""

import hashlib
import json
import logging
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

# import pdfplumber  # uncomment if re-enabling PDF processing
from datasets import load_dataset

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
OUTPUT_FILE       = "legal_corpus.jsonl"
SANITY_FILE       = "sanity_check.json"
CASES_DIR         = Path("Case in Brief")   # kept for reference; not used in main
HF_STATUTES       = "a2aj/canadian-laws"
HF_CASELAW        = "a2aj/canadian-case-law"
CASELAW_SAMPLE    = 0.10                    # 10 % stratified sample
CASELAW_SEED      = 42                      # fixed seed for reproducibility

# ── Jurisdiction maps ──────────────────────────────────────────────────────────

# For a2aj/canadian-laws  (dataset tag → province/territory)
STATUTE_JURISDICTION_MAP: dict[str, str] = {
    "LEGISLATION-BC":      "British Columbia",
    "LEGISLATION-AB":      "Alberta",
    "LEGISLATION-ON":      "Ontario",
    "LEGISLATION-QC":      "Quebec",
    "LEGISLATION-FEDERAL": "Canada",
}

# For a2aj/canadian-case-law  (court code → province/territory)
CASELAW_JURISDICTION_MAP: dict[str, str] = {
    # British Columbia
    "BCCA": "British Columbia", "BCSC": "British Columbia",
    # Ontario
    "ONCA": "Ontario",  "ONSC": "Ontario",  "ONCJ": "Ontario",
    # Nova Scotia
    "NSSC": "Nova Scotia", "NSCA": "Nova Scotia",
    "NSSM": "Nova Scotia", "NSPC": "Nova Scotia", "NSFC": "Nova Scotia",
    # Alberta
    "ABCA": "Alberta",  "ABQB": "Alberta",
    # Saskatchewan
    "SKCA": "Saskatchewan", "SKQB": "Saskatchewan",
    # Manitoba
    "MBCA": "Manitoba", "MBQB": "Manitoba",
    # New Brunswick
    "NBCA": "New Brunswick",
    # Newfoundland & Labrador
    "NLCA": "Newfoundland and Labrador",
    # Prince Edward Island
    "PECA": "Prince Edward Island",
    # Yukon
    "YKCA": "Yukon",
    # Federal / Canada-wide bodies
    "SCC":     "Canada", "FC":      "Canada", "FCA":    "Canada",
    "TCC":     "Canada", "RAD":     "Canada", "RPD":    "Canada",
    "SST":     "Canada", "FPSLREB": "Canada", "CIRB":   "Canada",
    "CHRT":    "Canada", "OHSTC":   "Canada", "CT":     "Canada",
    "CITT":    "Canada", "OIC":     "Canada", "CMAC":   "Canada",
    "PSDPT":   "Canada", "RLLR":    "Canada",
}

# ── Case law section heading regexes ──────────────────────────────────────────
#
# Each pattern matches a WHOLE stripped line (no surrounding words).
# Matches are case-insensitive. Optional trailing colon is allowed.
# Chunk index assignment:  opening→0  analysis→1  conclusion→2

_OPENING_HEADING_RE = re.compile(
    r"""^(?:
        overview | background(?:\s+and\s+\S+)? | the\s+background |
        introduction | context | the\s+context |
        facts?(?:\s+(?:of\s+the\s+case|and\s+\S+))? | the\s+facts? |
        agreed\s+facts? | undisputed\s+facts? | factual\s+background |
        procedural\s+(?:history|background) | history |
        parties?(?:\s+and\s+\S+)? | summary |
        statutory\s+(?:framework|context|background) |
        legislative\s+(?:framework|context|background) |
        the\s+legislation |
        (?:relevant|applicable)\s+(?:provisions?|legislation|law) |
        issues?(?:\s+(?:raised|to\s+be\s+decided|in\s+(?:dispute|issue)))? |
        questions?\s+(?:in|at)\s+issue |
        preliminary\s+(?:matters?|issues?)? | preamble |
        nature\s+of\s+(?:the\s+)?(?:proceeding|application|appeal)
    )\s*:?$""",
    re.IGNORECASE | re.VERBOSE,
)

_ANALYSIS_HEADING_RE = re.compile(
    r"""^(?:
        analysis | the\s+analysis | discussion |
        reasons?\s+for\s+(?:judgment|decision|appeal) |
        reasons?\s+for\s+the\s+(?:judgment|decision) |
        legal\s+(?:framework|analysis|principles?|issues?) |
        standard\s+of\s+review |
        (?:applicable\s+)?(?:legal\s+)?(?:framework|test|standard) |
        (?:the\s+)?law | consideration | merits |
        merits\s+of\s+the\s+appeal | determinative\s+issues? |
        scope\s+of\s+the\s+appeal | assessment |
        (?:my|our|the)\s+(?:analysis|reasons?|findings?) |
        findings? | credibility |
        submissions?\s*(?:and\s+analysis)?
    )\s*:?$""",
    re.IGNORECASE | re.VERBOSE,
)

_CONCLUSION_HEADING_RE = re.compile(
    r"""^(?:
        conclusion | disposition | result |
        orders? | remedy | remedies | relief | costs? | award
    )\s*:?$""",
    re.IGNORECASE | re.VERBOSE,
)

# Maps bucket name → chunk_index and human-readable label stored in metadata
_SECTION_CHUNK_INDEX: dict[str, int] = {
    "opening": 0, "analysis": 1, "conclusion": 2
}
_SECTION_LABEL: dict[str, str] = {
    "opening": "opening_background", "analysis": "analysis", "conclusion": "conclusion"
}

# ── ID Generation ──────────────────────────────────────────────────────────────

def make_id(seed: str) -> int:
    """Deterministic numeric ID via SHA-256 (first 15 hex digits → int)."""
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


# ── JSON Serialization Safety ──────────────────────────────────────────────────

def sanitize(obj: Any) -> Any:
    """
    Recursively convert Arrow/numpy scalars to native Python so json.dump()
    never raises TypeError on HuggingFace dataset values.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if hasattr(obj, "item"):    # numpy scalar
        return obj.item()
    if hasattr(obj, "as_py"):   # pyarrow scalar
        return obj.as_py()
    if not isinstance(obj, (str, int, float, bool)):
        return str(obj)
    return obj


# ── Text Cleaning ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Strip whitespace and collapse runs of 3+ blank lines to 2."""
    if not text:
        return ""
    return re.sub(r"\n{3,}", "\n\n", text.strip())


# ── Jurisdiction & Year Helpers ────────────────────────────────────────────────

def map_statute_jurisdiction(dataset_tag: Optional[str]) -> str:
    if not dataset_tag:
        return "Canada"
    return STATUTE_JURISDICTION_MAP.get(str(dataset_tag).strip().upper(), "Canada")


def map_caselaw_jurisdiction(court_code: Optional[str]) -> str:
    if not court_code:
        return "Canada"
    return CASELAW_JURISDICTION_MAP.get(str(court_code).strip().upper(), "Canada")


_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


def extract_year(text: Optional[str]) -> Optional[int]:
    """Return the first plausible 4-digit year found in text, else None."""
    if not text:
        return None
    m = _YEAR_RE.search(str(text))
    return int(m.group(1)) if m else None


# ── Case Law: Semantic Section Splitting ───────────────────────────────────────

def split_into_legal_sections(text: str) -> Optional[dict[str, str]]:
    """
    Attempt to split a case text into semantic sections by detecting standalone
    section-heading lines.

    Returns a dict with up to three keys ('opening', 'analysis', 'conclusion'),
    each containing the concatenated text for that section.
    Returns None when no analysis/conclusion heading is found (caller should
    emit the whole text as a single chunk instead).

    Strategy:
      - Walk lines; when a line matches a heading pattern and triggers a
        bucket transition, switch the active bucket.
      - Opening headings (Background, Facts, etc.) are noted but do NOT
        themselves trigger found_transition — we require at least one
        analysis or conclusion heading to confirm the case has structure.
    """
    lines = text.split("\n")
    buckets: dict[str, list[str]] = {"opening": [], "analysis": [], "conclusion": []}
    current = "opening"
    found_transition = False  # True only when analysis or conclusion heading seen

    for line in lines:
        stripped = line.strip()

        if 2 < len(stripped) < 80:
            if _CONCLUSION_HEADING_RE.match(stripped) and current != "conclusion":
                current = "conclusion"
                found_transition = True
            elif _ANALYSIS_HEADING_RE.match(stripped) and current != "conclusion":
                if current == "opening":
                    found_transition = True
                current = "analysis"
            # Opening headings: stay in current bucket (don't change state
            # if we've already moved past opening)

        buckets[current].append(line)

    if not found_transition:
        return None

    # Return only non-empty buckets
    return {
        k: clean_text("\n".join(v))
        for k, v in buckets.items()
        if any(l.strip() for l in v)
    }


# ── Case Law: Stratified Sampling ─────────────────────────────────────────────

def stratified_sample_indices(
    split, sample_rate: float = 0.10, seed: int = 42
) -> list[int]:
    """
    Return a sorted list of row indices sampled at sample_rate from every
    'dataset' group (court code). Uses a fixed seed for reproducibility.
    """
    rng = random.Random(seed)

    # Build groups using the Arrow column (fast, no per-row dicts)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, val in enumerate(split["dataset"]):
        groups[str(val) if val else "unknown"].append(i)

    sampled: list[int] = []
    for group_key in sorted(groups):
        indices = groups[group_key]
        k = min(max(1, round(len(indices) * sample_rate)), len(indices))
        chosen = rng.sample(indices, k)
        sampled.extend(chosen)
        log.info("  %-12s %5d rows → %d sampled", group_key, len(indices), k)

    log.info("Total sampled: %d / %d (%.0f%%)",
             len(sampled), sum(len(v) for v in groups.values()), sample_rate * 100)
    return sorted(sampled)


# ── Statute Processing ─────────────────────────────────────────────────────────

def _first_split(ds) -> Any:
    """Return the train split, or the first available split."""
    return ds["train"] if "train" in ds else ds[next(iter(ds.keys()))]


def iter_statute_records(ds) -> Any:
    """
    Yield one corpus dict per statute section from a2aj/canadian-laws.
    Skips rows with missing or unparseable unofficial_sections_en.
    """
    split = _first_split(ds)
    total = len(split)
    log.info("Processing %d statute rows...", total)

    for i, row in enumerate(split):
        if i % 500 == 0 and i > 0:
            log.info("  ... %d / %d rows done", i, total)

        title       = row.get("name_en")               or ""
        citation    = row.get("citation_en")            or ""
        dataset_tag = row.get("dataset")               or ""
        date_str    = row.get("document_date_en")       or ""
        source_url  = row.get("source_url_en")          or ""
        sections_raw = row.get("unofficial_sections_en") or ""

        try:
            sections: dict[str, str] = json.loads(sections_raw) if sections_raw else {}
        except (json.JSONDecodeError, TypeError, ValueError):
            log.warning("Skipping statute (bad sections JSON): %s", title)
            continue

        if not sections:
            continue

        jurisdiction = map_statute_jurisdiction(dataset_tag)
        year         = extract_year(date_str) or extract_year(citation)
        num_sections = len(sections)

        base_meta = {
            "dataset":          dataset_tag,
            "document_date_en": date_str,
            "source_url_en":    source_url,
            "num_sections_en":  num_sections,
        }

        for section_key, section_text in sections.items():
            try:
                chunk_index = int(section_key)
            except (ValueError, TypeError):
                chunk_index = 0

            text = clean_text(str(section_text))
            if not text:
                continue

            yield sanitize({
                "id":           make_id(f"statute|{citation}|{section_key}"),
                "source_type":  "statute",
                "title":        title,
                "citation":     citation,
                "jurisdiction": jurisdiction,
                "year":         year,
                "chunk_index":  chunk_index,
                "text":         text,
                "metadata":     base_meta,
            })


# ── Case Law Processing ────────────────────────────────────────────────────────

def iter_caselaw_records(ds_caselaw, sample_rate: float = CASELAW_SAMPLE) -> Any:
    """
    Yield corpus dicts from a2aj/canadian-case-law.

    Sampling: 10 % stratified by court ('dataset' column), fixed seed.

    Chunking strategy (best-effort):
      - If section headings are found → up to 3 chunks:
          chunk_index 0 = opening/background
          chunk_index 1 = analysis
          chunk_index 2 = conclusion
      - If no headings detected → 1 chunk (chunk_index 0, full text).
    """
    split = _first_split(ds_caselaw)

    log.info("Building stratified sample from %d case law rows...", len(split))
    indices = stratified_sample_indices(split, sample_rate=sample_rate, seed=CASELAW_SEED)
    log.info("Processing %d sampled case law rows...", len(indices))

    for count, idx in enumerate(indices):
        if count % 500 == 0 and count > 0:
            log.info("  ... %d / %d done", count, len(indices))

        row = split[idx]

        title       = row.get("name_en")            or ""
        citation    = row.get("citation_en")         or ""
        court       = row.get("dataset")            or ""
        date_str    = row.get("document_date_en")    or ""
        url         = row.get("url_en")              or ""
        raw_text    = row.get("unofficial_text_en")  or ""
        cases_cited = row.get("cases_cited_en")      or []
        citing_cnt  = row.get("citing_cases_count")  or 0

        if not raw_text.strip():
            continue

        jurisdiction = map_caselaw_jurisdiction(court)
        year         = extract_year(date_str) or extract_year(citation)
        text         = clean_text(raw_text)

        base_meta = {
            "dataset":            court,
            "document_date_en":   date_str,
            "url_en":             url,
            "cases_cited":        list(cases_cited) if cases_cited else [],
            "citing_cases_count": int(citing_cnt) if citing_cnt else 0,
        }

        sections = split_into_legal_sections(text)

        if sections is None:
            # No recognisable headings → emit whole text as single chunk
            yield sanitize({
                "id":           make_id(f"caselaw|{citation}|full"),
                "source_type":  "case",
                "title":        title,
                "citation":     citation,
                "jurisdiction": jurisdiction,
                "year":         year,
                "chunk_index":  0,
                "text":         text,
                "metadata":     {**base_meta,
                                 "section_label":    "full",
                                 "chunking_method":  "single_chunk"},
            })
        else:
            for bucket, chunk_text in sections.items():
                if not chunk_text:
                    continue
                yield sanitize({
                    "id":           make_id(f"caselaw|{citation}|{bucket}"),
                    "source_type":  "case",
                    "title":        title,
                    "citation":     citation,
                    "jurisdiction": jurisdiction,
                    "year":         year,
                    "chunk_index":  _SECTION_CHUNK_INDEX[bucket],
                    "text":         chunk_text,
                    "metadata":     {**base_meta,
                                     "section_label":   _SECTION_LABEL[bucket],
                                     "chunking_method": "section_headings"},
                })


# ── [DISABLED] SCC PDF Case Processing ────────────────────────────────────────
# Superseded by a2aj/canadian-case-law.  Code kept for reference only.
#
# def extract_pdf_text(pdf_path): ...
# def process_case_pdf(pdf_path): ...
# def iter_case_records(cases_dir): ...


# ── Sanity Check ───────────────────────────────────────────────────────────────

def run_sanity_check(ds_statutes, ds_caselaw) -> None:
    """
    Write one statute record + all chunks for one case law record to
    sanity_check.json (full text, no truncation) for manual inspection.
    """
    sanity: list[dict] = []

    # --- One statute (first section of first statute with sections) ---
    for record in iter_statute_records(ds_statutes):
        sanity.append(record)
        log.info("Sanity statute : %s  (chunk %s)", record["title"], record["chunk_index"])
        break

    # --- One case law record (first sampled row) ---
    split = _first_split(ds_caselaw)
    indices = stratified_sample_indices(split, sample_rate=CASELAW_SAMPLE, seed=CASELAW_SEED)
    first_idx = indices[0]
    row = split[first_idx]

    title    = row.get("name_en")           or ""
    citation = row.get("citation_en")        or ""
    court    = row.get("dataset")           or ""
    date_str = row.get("document_date_en")   or ""
    url      = row.get("url_en")             or ""
    raw_text = row.get("unofficial_text_en") or ""
    cases_cited = row.get("cases_cited_en")  or []
    citing_cnt  = row.get("citing_cases_count") or 0

    text         = clean_text(raw_text)
    jurisdiction = map_caselaw_jurisdiction(court)
    year         = extract_year(date_str) or extract_year(citation)

    base_meta = {
        "dataset":            court,
        "document_date_en":   date_str,
        "url_en":             url,
        "cases_cited":        list(cases_cited) if cases_cited else [],
        "citing_cases_count": int(citing_cnt) if citing_cnt else 0,
    }

    sections = split_into_legal_sections(text)

    if sections is None:
        rec = sanitize({
            "id": make_id(f"caselaw|{citation}|full"),
            "source_type": "case", "title": title, "citation": citation,
            "jurisdiction": jurisdiction, "year": year, "chunk_index": 0,
            "text": text,
            "metadata": {**base_meta, "section_label": "full",
                         "chunking_method": "single_chunk"},
        })
        sanity.append(rec)
        log.info("Sanity case    : %s  → single chunk  (%d chars)",
                 citation, len(text))
    else:
        for bucket, chunk_text in sections.items():
            if not chunk_text:
                continue
            rec = sanitize({
                "id": make_id(f"caselaw|{citation}|{bucket}"),
                "source_type": "case", "title": title, "citation": citation,
                "jurisdiction": jurisdiction, "year": year,
                "chunk_index": _SECTION_CHUNK_INDEX[bucket],
                "text": chunk_text,
                "metadata": {**base_meta,
                             "section_label":   _SECTION_LABEL[bucket],
                             "chunking_method": "section_headings"},
            })
            sanity.append(rec)
        log.info("Sanity case    : %s  → %d section chunk(s): %s",
                 citation, len(sections), list(sections.keys()))

    with open(SANITY_FILE, "w", encoding="utf-8") as fh:
        json.dump(sanity, fh, indent=2, ensure_ascii=False)

    log.info("Sanity check → %s  (%d record(s))", SANITY_FILE, len(sanity))


# ── Output Writer ──────────────────────────────────────────────────────────────

def append_jsonl(records, output_path: str) -> int:
    """Append an iterable of dicts to a JSONL file. Returns count written."""
    count = 0
    with open(output_path, "a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load datasets ──────────────────────────────────────────────────────────
    log.info("Loading %s ...", HF_STATUTES)
    ds_statutes = load_dataset(HF_STATUTES)
    log.info("Statute splits: %s", list(ds_statutes.keys()))

    log.info("Loading %s ...", HF_CASELAW)
    ds_caselaw = load_dataset(HF_CASELAW)
    log.info("Case law splits: %s", list(ds_caselaw.keys()))

    # ── Sanity check: one statute + one case law, full text ────────────────────
    log.info("=== Sanity check ===")
    run_sanity_check(ds_statutes, ds_caselaw)

    # ── Clear output file ──────────────────────────────────────────────────────
    if os.path.exists(OUTPUT_FILE):
        log.info("Removing existing %s", OUTPUT_FILE)
        os.remove(OUTPUT_FILE)

    # ── 1. Statutes ────────────────────────────────────────────────────────────
    log.info("=== Processing statutes ===")
    n_statutes = append_jsonl(iter_statute_records(ds_statutes), OUTPUT_FILE)
    log.info("Statute records written: %d", n_statutes)

    # ── 2. Case law (10 % stratified sample) ───────────────────────────────────
    log.info("=== Processing case law (%.0f%% stratified sample) ===",
             CASELAW_SAMPLE * 100)
    n_cases = append_jsonl(iter_caselaw_records(ds_caselaw), OUTPUT_FILE)
    log.info("Case law records written: %d", n_cases)

    # ── [DISABLED] 3. SCC PDFs ─────────────────────────────────────────────────
    # log.info("=== Processing PDF cases ===")
    # n_pdfs = append_jsonl(iter_case_records(CASES_DIR), OUTPUT_FILE)
    # log.info("PDF case records written: %d", n_pdfs)

    # ── Summary ────────────────────────────────────────────────────────────────
    total = n_statutes + n_cases
    log.info("=== Done. Total records in %s: %d ===", OUTPUT_FILE, total)


if __name__ == "__main__":
    main()
