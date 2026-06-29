#!/usr/bin/env python3
"""Generate legal training questions from Canadian *case* excerpts using Gemini.

Companion to questiongen.py (which handles source_type == "statute"). This script
handles source_type == "case". For each case it pulls the in-text "Summary:" block
out of the decision header, treats that as a CASE_SYNOPSIS for orientation, and
pairs it with each chunk of that case (the verbatim EXTRACT) when prompting the
model. Only cases that actually contain a Summary block are processed.

Mechanics (rate limiting, concurrency, --resume, per-excerpt fsync'd writes) are
shared with questiongen.py and imported from it.

Usage:
    export GOOGLE_API_KEY=...        # or GEMINI_API_KEY
    ./venv/bin/python questiongen_cases.py
    ./venv/bin/python questiongen_cases.py --limit 5 --workers 8 --rpm 10
"""

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import google.generativeai as genai

# Reuse the throttling / parsing / resume machinery from the statute script.
from questiongen import (
    HERE,
    RateLimiter,
    extract_json,
    iter_records,
    load_done_keys,
    record_key,
)

CASE_PROMPT_TEMPLATE = """You are a legal question generator for a Canadian legal AI training dataset. You are given a STRUCTURED EXTRACT from a Canadian court decision (a single issue, treated by a single opinion — majority, concurrence, or dissent), plus a short CASE SYNOPSIS for orientation. Your job is to write realistic underlying legal questions that this extract is relevant to. You generate ONLY questions — never answers, reasoning, or holdings.

## INPUT
- CASE_SYNOPSIS: a brief orientation describing the overall case and where this issue sits. CONTEXT ONLY — questions must be answerable from the EXTRACT, not the synopsis.
- EXTRACT: verbatim text covering one issue from one opinion. This is the citable grounding source.
- CITATION: the case citation (e.g. "2022 SCC 30")
- OPINION_TYPE: majority | concurrence | dissent
- JURISDICTION: e.g. federal, ON, QC
- COURT_LEVEL: e.g. SCC, ONCA

## TASK
Generate exactly {N} questions distributed across the three TYPES below ({N_GROUNDED} grounded, {N_REASONING} reasoning, {N_UNCERTAINTY} uncertainty). Output only the type label and the question text.

### TYPE 1 — GROUNDED
Questions the EXTRACT squarely answers — where the legal rule or holding in the extract is a correct, direct answer.
- Target the legal principle, rule, or test the extract sets out or applies.
- Write the UNDERLYING legal question a real user would type — NOT a question about the case.

### TYPE 2 — REASONING
Questions that apply the extract's principle to a NEW, concrete fact pattern not contained in the extract.
- Include a brief, specific hypothetical.
- The extract's rule should be the governing authority, but answering requires applying it to the new facts, not restating it.

### TYPE 3 — UNCERTAINTY
Questions the extract is RELEVANT to but does NOT actually resolve — where the honest answer is "this decision doesn't settle that." Use one of these concrete reasons:
- The question pushes the principle beyond the facts or issue actually decided in the extract.
- The question concerns a related point the opinion expressly left open or did not address.
- The question depends on facts, another issue, or another jurisdiction's law the extract doesn't cover.
- The question seeks a definitive rule where the extract is narrow or fact-specific.
- Keep these plausible and on-topic — a real user might reasonably ask them about this area, but the extract alone cannot fully answer them. Do NOT make them absurd or off-topic.

## CRITICAL RULES
- Generate questions ONLY. No answers, reasoning, holdings, or citations in the output.
- Do NOT name or reference the case, the court, the judge, or "this decision/extract" in the question text. Write the bare underlying legal question a user would actually type.
- Questions must be self-contained — name the legal subject rather than referring to "the above" or "this provision."
- Ground TYPE 1 and TYPE 2 in what the EXTRACT actually supports. Do not rely on the synopsis for substance, and do not invent rules not present in the extract.

## OUTPUT FORMAT
Return a JSON array. Each element:
{{"type": "grounded" | "reasoning" | "uncertainty", "question": "..."}}

## PROVIDED
CASE_SYNOPSIS: {case_synopsis}
EXTRACT: {extract}
CITATION: {citation}
OPINION_TYPE: {opinion_type}
JURISDICTION: {jurisdiction}
COURT_LEVEL: {court_level}
"""

# The Summary block runs from the "Summary:" label until the first numbered
# paragraph marker ("[1]") or the "Reasons for Judgment/Decision" heading.
SUMMARY_RE = re.compile(
    r"Summary:\s*(.+?)(?=\n\s*\[\d+\]|\n\s*Reasons for (?:Judgment|Decision)\b|\Z)",
    re.DOTALL,
)


def extract_synopsis(text):
    """Pull the decision's Summary block out of the raw text, or None if absent."""
    m = SUMMARY_RE.search(text)
    if not m:
        return None
    # Collapse internal whitespace/newlines into a single-paragraph synopsis.
    syn = " ".join(m.group(1).split())
    return syn or None


def build_synopsis_map(path):
    """Map citation -> synopsis for every case whose text contains a Summary block."""
    synopses = {}
    for rec in iter_records(path):
        if rec.get("source_type") != "case":
            continue
        cite = rec.get("citation")
        if not cite or cite in synopses:
            continue
        if "Summary:" not in rec.get("text", ""):
            continue
        syn = extract_synopsis(rec["text"])
        if syn:
            synopses[cite] = syn
    return synopses


def iter_case_records(path):
    """Yield only source_type == 'case' records."""
    for rec in iter_records(path):
        if rec.get("source_type") == "case":
            yield rec


def court_level_for(rec):
    """Best-effort court level: the dataset code (e.g. BCCA, SCC, ONCA)."""
    return rec.get("metadata", {}).get("dataset") or ""


def build_case_prompt(rec, synopsis, n_grounded, n_reasoning, n_uncertainty,
                      opinion_type):
    n = n_grounded + n_reasoning + n_uncertainty
    return CASE_PROMPT_TEMPLATE.format(
        N=n,
        N_GROUNDED=n_grounded,
        N_REASONING=n_reasoning,
        N_UNCERTAINTY=n_uncertainty,
        case_synopsis=synopsis,
        extract=rec["text"],
        citation=rec["citation"],
        opinion_type=opinion_type,
        jurisdiction=rec.get("jurisdiction", ""),
        court_level=court_level_for(rec),
    )


def generate_for_case(model, rec, synopsis, n_grounded, n_reasoning, n_uncertainty,
                      opinion_type, limiter=None):
    """Call Gemini for one case extract and return the parsed list of questions."""
    prompt = build_case_prompt(
        rec, synopsis, n_grounded, n_reasoning, n_uncertainty, opinion_type
    )
    if limiter is not None:
        limiter.acquire()
    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"},
    )
    return extract_json(response.text)


def main():
    parser = argparse.ArgumentParser(
        description="Generate legal questions from case excerpts via Gemini."
    )
    parser.add_argument("--input", default=os.path.join(HERE, "legal_corpus.jsonl"),
                        help="Path to the corpus (JSON array or JSONL).")
    parser.add_argument("--model", default="gemini-3.1-flash-lite",
                        help="Gemini model name.")
    parser.add_argument("--grounded", type=int, default=1)
    parser.add_argument("--reasoning", type=int, default=2)
    parser.add_argument("--uncertainty", type=int, default=1)
    parser.add_argument("--include-opening-background", action="store_true",
                        help="Also generate from 'opening_background' chunks. By "
                             "default these are skipped because they contain the "
                             "Summary block itself (which is already the synopsis).")
    parser.add_argument("--opinion-type", default="majority",
                        choices=["majority", "concurrence", "dissent"],
                        help="OPINION_TYPE label for every extract. Chunks are not "
                             "split by opinion in the corpus, so this defaults to "
                             "majority.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max number of case excerpts to process (0 = all).")
    parser.add_argument("--output", default=os.path.join(HERE, "questions_cases.jsonl"),
                        help="Path to write results, one JSON object per line "
                             "(default: questions_cases.jsonl). Written incrementally.")
    parser.add_argument("--resume", action="store_true",
                        help="Append to an existing output file and skip excerpts "
                             "already present in it (instead of overwriting).")
    parser.add_argument("--rpm", type=float, default=10,
                        help="Max API requests per minute (default 10). "
                             "Requests are spaced evenly to stay under this rate.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of concurrent request threads (default 8). "
                             "Hides per-call latency so the --rpm budget is "
                             "actually saturated. Use 1 for sequential.")
    args = parser.parse_args()

    limiter = RateLimiter(args.rpm)

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Set GOOGLE_API_KEY (or GEMINI_API_KEY) in your environment.")
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(args.model)

    # Pass 1: collect the Summary block for every case that has one.
    synopses = build_synopsis_map(args.input)
    print(f"Found synopses for {len(synopses)} case(s).", file=sys.stderr)

    done = load_done_keys(args.output) if args.resume else set()
    if args.resume:
        print(f"Resuming: {len(done)} excerpt(s) already in {args.output}",
              file=sys.stderr)

    # Pass 2: select case chunks whose case has a synopsis (respecting resume/limit).
    skipped = 0
    no_synopsis = 0
    skipped_opening = 0
    todo = []
    for rec in iter_case_records(args.input):
        cite = rec.get("citation")
        if cite not in synopses:
            no_synopsis += 1
            continue
        # The opening_background chunk holds the Summary block itself, so its
        # extract would just duplicate the synopsis; skip it unless asked.
        if (not args.include_opening_background
                and rec.get("metadata", {}).get("section_label") == "opening_background"):
            skipped_opening += 1
            continue
        if args.resume and record_key(rec) in done:
            skipped += 1
            continue
        if args.limit and len(todo) >= args.limit:
            break
        todo.append(rec)

    print(f"Skipping {no_synopsis} case chunk(s) without a synopsis.", file=sys.stderr)
    print(f"Skipping {skipped_opening} opening_background chunk(s).", file=sys.stderr)

    def label_for(rec, i):
        return rec.get("citation") or rec.get("title") or rec.get("id") or f"#{i}"

    def work(rec):
        return generate_for_case(
            model, rec, synopses[rec["citation"]],
            args.grounded, args.reasoning, args.uncertainty,
            args.opinion_type, limiter,
        )

    processed = 0     # results successfully written this run
    total_q = 0
    write_lock = threading.Lock()
    # Append when resuming, otherwise overwrite. Flush + fsync after every
    # excerpt so progress survives an interruption (each line = one extract).
    mode = "a" if args.resume else "w"
    with open(args.output, mode, encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(work, rec): (i, rec)
                   for i, rec in enumerate(todo, start=1)}
        for fut in as_completed(futures):
            i, rec = futures[fut]
            label = label_for(rec, i)
            try:
                questions = fut.result()
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[{i}] SKIP {label}: could not parse model output ({e})",
                      file=sys.stderr)
                continue
            entry = {
                "id": rec.get("id"),
                "citation": rec.get("citation"),
                "title": rec.get("title"),
                "jurisdiction": rec.get("jurisdiction"),
                "court_level": court_level_for(rec),
                "opinion_type": args.opinion_type,
                "chunk_index": rec.get("chunk_index"),
                "questions": questions,
            }
            line = json.dumps(entry, ensure_ascii=False) + "\n"
            # Serialize writes so concurrent results never interleave on disk.
            with write_lock:
                out.write(line)
                out.flush()
                os.fsync(out.fileno())
                processed += 1
                total_q += len(questions)
            print(f"[{i}] {label}: {len(questions)} questions", file=sys.stderr)

    msg = f"\nProcessed {processed} case excerpt(s), {total_q} questions total"
    if args.resume:
        msg += f" (skipped {skipped} already done)"
    print(f"{msg} -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
