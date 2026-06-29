#!/usr/bin/env python3
"""Generate legal training questions from a source excerpt using Gemini.

Loads the "statute"-labeled excerpt from sanity_check.json, fills it into the
question-generation prompt, sends it to Gemini, and prints/saves the resulting
JSON array of questions.

Usage:
    export GOOGLE_API_KEY=...        # or GEMINI_API_KEY
    python questiongen.py
    python questiongen.py --n 12 --grounded 5 --reasoning 4 --uncertainty 3
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import google.generativeai as genai

PROMPT_TEMPLATE = """You are a legal question generator for a Canadian legal AI training dataset. Your job is to write realistic questions a user might ask, grounded in a provided legal source excerpt. You generate ONLY questions — never answers.

## INPUT
You will receive:
- SOURCE_EXCERPT: verbatim text from a Canadian legal source
- CITATION: the source's citation
- JURISDICTION: e.g. federal, ON, BC, QC
- COURT_LEVEL: (if a case) e.g. SCC, ONCA; (if statute) "statute"

## TASK
Generate exactly {N} questions distributed across the three TYPES below. For each question, output the type label, the question text, and nothing else (no answers, no reasoning, no citations).

### TYPE 1 — GROUNDED
Questions that the SOURCE_EXCERPT can substantively answer on its own. A knowledgeable reader with only this excerpt should be able to give a correct, citable answer.
- Vary the task: plain-language explanation, applying the provision to a short hypothetical, interpreting a specific term or condition, comparing two parts within the excerpt.
- Phrase them the way a real user would (a paralegal, law student, or layperson) — not as exam questions about the text.

### TYPE 2 — REASONING
Questions that use the excerpt as the governing rule but require applying it to a fact pattern NOT contained in the excerpt — issue-spotting or analysis the excerpt doesn't spell out.
- Include a brief, concrete fact scenario in the question.
- The excerpt should be the relevant rule, but answering requires reasoning beyond restating it.

### TYPE 3 — UNCERTAINTY
Questions where the SOURCE_EXCERPT is genuinely INSUFFICIENT to give a complete, safe answer — the honest answer is "this depends on more than the provided source." Generate questions that are insufficient for one of these concrete reasons:
- The answer turns on facts not stated and not inferable from the excerpt.
- The answer requires a multi-factor analysis (e.g. case law factors) the excerpt only gestures at.
- The answer depends on another jurisdiction's law or a provision not in the excerpt.
- The question seeks a definitive yes/no on something the excerpt only partially governs.
Do NOT make these absurd or off-topic — they must be plausible questions a real user would ask about this excerpt, that simply can't be fully resolved from it alone.

## RULES
- Generate questions ONLY. No answers, no explanations, no citations in the output.
- Every question must be answerable (or appropriately unanswerable, for Type 3) in reference to THIS excerpt — do not invent provisions or facts about other sources.
- Make questions self-contained: a downstream reader sees the question + excerpt, so don't reference "the above" or "this section" ambiguously — name the subject.
- Use natural, realistic phrasing across a range of user sophistication.
- Where the jurisdiction matters (especially QC civil law vs common law), you may write questions that surface that nuance.
- Do not duplicate questions or write trivial rephrasings of each other.

## OUTPUT FORMAT
Return a JSON array. Each element:
{{"type": "grounded" | "reasoning" | "uncertainty", "question": "..."}}

Distribution for this batch: {N_GROUNDED} grounded, {N_REASONING} reasoning, {N_UNCERTAINTY} uncertainty.

## INPUT
SOURCE_EXCERPT: {source_excerpt}
CITATION: {citation}
JURISDICTION: {jurisdiction}
COURT_LEVEL: {court_level}
"""

HERE = os.path.dirname(os.path.abspath(__file__))


def iter_records(path):
    """Yield records from either a JSON array file or a JSONL file (one obj/line)."""
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            # Whole-file JSON array.
            for rec in json.load(f):
                yield rec
        else:
            # JSONL: one JSON object per line (streamed, memory-safe for big files).
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def iter_statute_records(path):
    """Yield only the records whose source_type == 'statute'."""
    for rec in iter_records(path):
        if rec.get("source_type") == "statute":
            yield rec


def build_prompt(rec, n_grounded, n_reasoning, n_uncertainty):
    n = n_grounded + n_reasoning + n_uncertainty
    # For a statute, COURT_LEVEL is the literal string "statute".
    court_level = "statute" if rec.get("source_type") == "statute" else rec.get("source_type", "")
    return PROMPT_TEMPLATE.format(
        N=n,
        N_GROUNDED=n_grounded,
        N_REASONING=n_reasoning,
        N_UNCERTAINTY=n_uncertainty,
        source_excerpt=rec["text"],
        citation=rec["citation"],
        jurisdiction=rec["jurisdiction"],
        court_level=court_level,
    )


def extract_json(text):
    """Best-effort extraction of a JSON array from the model output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


def record_key(rec):
    """Stable identity for an excerpt, used for resume/dedup."""
    return (rec.get("id"), rec.get("chunk_index"))


def load_done_keys(path):
    """Return the set of record_keys already present in an existing JSONL output."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue  # ignore a half-written trailing line
            done.add((entry.get("id"), entry.get("chunk_index")))
    return done


class RateLimiter:
    """Thread-safe limiter that spaces request *starts* by a minimum interval.

    Shared across worker threads so concurrent in-flight calls still respect a
    global requests-per-minute cap (the spacing bounds the request *rate*, while
    the thread pool hides per-call latency).
    """

    def __init__(self, rpm):
        self._interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self):
        if not self._interval:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_allowed)
            self._next_allowed = slot + self._interval
        wait = slot - time.monotonic()
        if wait > 0:
            time.sleep(wait)


def generate_for_record(model, rec, n_grounded, n_reasoning, n_uncertainty,
                        limiter=None):
    """Call Gemini for one excerpt and return the parsed list of questions."""
    prompt = build_prompt(rec, n_grounded, n_reasoning, n_uncertainty)
    if limiter is not None:
        limiter.acquire()
    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"},
    )
    return extract_json(response.text)


def main():
    parser = argparse.ArgumentParser(description="Generate legal questions via Gemini.")
    parser.add_argument("--input", default=os.path.join(HERE, "legal_corpus.jsonl"),
                        help="Path to the corpus (JSON array or JSONL).")
    parser.add_argument("--model", default="gemini-3.1-flash-lite", help="Gemini model name.")
    parser.add_argument("--grounded", type=int, default=1)
    parser.add_argument("--reasoning", type=int, default=2)
    parser.add_argument("--uncertainty", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0,
                        help="Max number of statute excerpts" \
                        " to process (0 = all). "
                             "Default 10 to avoid accidental huge/expensive runs.")
    parser.add_argument("--output", default=os.path.join(HERE, "questions.jsonl"),
                        help="Path to write results, one JSON object per line "
                             "(default: questions.jsonl). Written incrementally.")
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

    done = load_done_keys(args.output) if args.resume else set()
    if args.resume:
        print(f"Resuming: {len(done)} excerpt(s) already in {args.output}", file=sys.stderr)

    # Select the excerpts to process up front (respecting --resume and --limit).
    skipped = 0
    todo = []
    for rec in iter_statute_records(args.input):
        if args.resume and record_key(rec) in done:
            skipped += 1
            continue
        if args.limit and len(todo) >= args.limit:
            break
        todo.append(rec)

    def label_for(rec, i):
        return rec.get("citation") or rec.get("title") or rec.get("id") or f"#{i}"

    def work(rec):
        return generate_for_record(
            model, rec, args.grounded, args.reasoning, args.uncertainty, limiter
        )

    processed = 0     # results successfully written this run
    total_q = 0
    write_lock = threading.Lock()
    # Append when resuming, otherwise overwrite. Flush + fsync after every
    # excerpt so progress survives an interruption (each line = one statute).
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

    msg = f"\nProcessed {processed} statute excerpt(s), {total_q} questions total"
    if args.resume:
        msg += f" (skipped {skipped} already done)"
    print(f"{msg} -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

