#!/usr/bin/env python3
"""Generate grounded reasoning + answers for statute questions using Gemini.

Companion to questiongen.py. Reads questions.jsonl (produced by questiongen.py,
one object per excerpt with a "questions": [...] array), and for each question
calls Gemini to produce a "reasoning", "answer", and a structured "evidence_used"
list grounded in the source statute.

Evidence retrieval: for a given question, the exact source chunk (matched by id)
is always included, plus nearby chunks from the same citation (by chunk_index
proximity, up to --window chunks each side and --max-evidence-chars total) so
the model has surrounding statutory context, not just one isolated chunk.

Because legal_corpus.jsonl is large (~230k lines / ~900MB), this script builds a
one-time index (citation -> byte offsets) on first run and caches it to
"<corpus>.idx.pkl" next to the corpus file, keyed on the corpus file's
(size, mtime). Later runs (including answergen_cases.py) reuse that cache.

Output: one flattened JSON object per question, written to answers.jsonl:
    {"question_id", "source_id", "citation", "title", "jurisdiction",
     "chunk_index", "type", "question", "reasoning", "answer",
     "evidence_used": [{"chunk_id", "paragraph_index", "quote", "verified"}],
     "model"}
"chunk_index" (top-level) is which corpus chunk the question was generated
from. Within "evidence_used", "chunk_id" is the exact corpus record (id) a
quote is grounded in (verified by substring match against the evidence
actually sent — corrected if the model mislabels it), and "paragraph_index"
is the paragraph/section number the source text itself assigns near that
quote (e.g. a bracketed marker like [43] in case text), not a corpus index.

Usage:
    export GOOGLE_API_KEY=...        # or GEMINI_API_KEY
    # Smoke test on 5 questions first:
    python answergen.py --limit 5
    # Full run once the smoke test looks right:
    python answergen.py --resume
    # Different model:
    python answergen.py --model gemini-2.5-pro --limit 5
"""

import argparse
import json
import os
import pickle
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import google.generativeai as genai

from questiongen import HERE, RateLimiter, iter_records

ANSWER_PROMPT_TEMPLATE = """You are a Canadian legal research assistant answering a question that was generated from a specific statutory excerpt. Ground your answer strictly in the EVIDENCE below — verbatim excerpts from the same statute (the originating excerpt plus surrounding chunks for context).

## QUESTION_TYPE
- grounded: the evidence should directly and fully answer the question.
- reasoning: the evidence supplies the governing rule; apply it to the hypothetical facts stated in the question.
- uncertainty: the evidence only partially resolves the question. Give an honest answer that states what the evidence does and does not establish — do not fabricate certainty.

## TASK
1. Write "reasoning": a concise legal analysis (2-5 sentences) showing how the evidence leads to your answer, or, for uncertainty questions, exactly what is missing or unresolved.
2. Write "answer": the direct answer a user would receive, as one short paragraph.
3. Write "evidence_used": the excerpt(s) that actually support your reasoning/answer, each as {{"chunk_id": <id>, "paragraph_index": <int or null>, "quote": "<verbatim substring from that excerpt's text>"}}.
   - "chunk_id" must be the chunk_id shown in the EVIDENCE header the quote came from.
   - "paragraph_index" is the paragraph/section number the SOURCE TEXT ITSELF assigns to the quoted passage (e.g. a numbered section like "1" or a bracketed marker like [43]) if one appears near the quote in that excerpt; use null if the source has no such numbering there.
   - Quotes must be exact substrings, not paraphrases. Cite only excerpts provided below. If nothing meaningfully supports the answer, return an empty list.

## RULES
- Do not invent statutory text, provisions, or facts that are not in the EVIDENCE.
- Do not rely on outside legal knowledge, even if you believe you know the answer — only what's in the EVIDENCE.
- For uncertainty questions, "the evidence does not resolve this" is the correct and expected answer — do not force a definitive answer the evidence doesn't support.
- Each EVIDENCE excerpt below is labeled "--- EVIDENCE (chunk_id=..., corpus_chunk_index=...) ---". corpus_chunk_index is only for your own orientation (this excerpt's position within the statute) — never report it as paragraph_index. paragraph_index refers only to numbering that appears inside the excerpt's own text.

## OUTPUT FORMAT
Return a single JSON object exactly like:
{{"reasoning": "...", "answer": "...", "evidence_used": [{{"chunk_id": ..., "paragraph_index": ..., "quote": "..."}}]}}

## PROVIDED
QUESTION_TYPE: {question_type}
QUESTION: {question}
CITATION: {citation}
TITLE: {title}
JURISDICTION: {jurisdiction}
EVIDENCE (excerpts from this statute):
{evidence_block}
"""

# Duplicated (deliberately small) from questiongen_cases.py's synopsis logic so
# the corpus index can capture case synopses in the same single pass that
# indexes chunk offsets, instead of scanning the 900MB corpus file twice.
SUMMARY_RE = re.compile(
    r"Summary:\s*(.+?)(?=\n\s*\[\d+\]|\n\s*Reasons for (?:Judgment|Decision)\b|\Z)",
    re.DOTALL,
)


def extract_synopsis(text):
    m = SUMMARY_RE.search(text)
    if not m:
        return None
    syn = " ".join(m.group(1).split())
    return syn or None


def extract_json_object(text):
    """Best-effort extraction of a JSON object from the model output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


def build_or_load_corpus_index(corpus_path, cache_path=None, force_rebuild=False):
    """Return {"chunks": citation -> [(chunk_index, id, offset), ...] sorted,
    "synopses": citation -> synopsis} built (or loaded from cache) from the corpus.
    """
    cache_path = cache_path or corpus_path + ".idx.pkl"
    st = os.stat(corpus_path)
    sig = (st.st_size, int(st.st_mtime))

    if not force_rebuild and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached_sig, index = pickle.load(f)
            if cached_sig == sig:
                print(f"Loaded corpus index from cache ({cache_path})", file=sys.stderr)
                return index
        except Exception:
            pass  # fall through and rebuild

    print(f"Building corpus index from {corpus_path} (one-time scan)...", file=sys.stderr)
    chunks_by_citation = {}
    synopses = {}
    n = 0
    with open(corpus_path, "r", encoding="utf-8") as f:
        offset = f.tell()
        line = f.readline()
        while line:
            stripped = line.strip()
            if stripped:
                try:
                    rec = json.loads(stripped)
                except json.JSONDecodeError:
                    rec = None
                if rec is not None:
                    citation = rec.get("citation")
                    if citation:
                        chunks_by_citation.setdefault(citation, []).append(
                            (rec.get("chunk_index"), rec.get("id"), offset)
                        )
                        if (rec.get("source_type") == "case"
                                and citation not in synopses
                                and "Summary:" in rec.get("text", "")):
                            syn = extract_synopsis(rec["text"])
                            if syn:
                                synopses[citation] = syn
            n += 1
            if n % 50000 == 0:
                print(f"  indexed {n} lines...", file=sys.stderr)
            offset = f.tell()
            line = f.readline()

    for lst in chunks_by_citation.values():
        lst.sort(key=lambda t: (t[0] if t[0] is not None else 0))

    index = {"chunks": chunks_by_citation, "synopses": synopses}
    try:
        with open(cache_path, "wb") as f:
            pickle.dump((sig, index), f)
    except OSError as e:
        print(f"Warning: could not write index cache: {e}", file=sys.stderr)

    print(f"Index built: {len(chunks_by_citation)} citations, "
          f"{len(synopses)} case synopses -> {cache_path}", file=sys.stderr)
    return index


class CorpusReader:
    """Thread-safe random-access reader for the corpus file, keyed by byte offset.

    Keeps one open file handle per thread (via threading.local) so concurrent
    workers can seek/read independently without contending on a shared handle.
    """

    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self._local = threading.local()

    def _handle(self):
        fh = getattr(self._local, "fh", None)
        if fh is None:
            fh = open(self.corpus_path, "r", encoding="utf-8")
            self._local.fh = fh
        return fh

    def read_at(self, offset):
        fh = self._handle()
        fh.seek(offset)
        line = fh.readline()
        return json.loads(line)


def get_evidence_chunks(index, reader, citation, source_id, source_chunk_index,
                        window=3, max_chars=12000):
    """Return the source chunk plus up to `window` neighbors each side (by
    chunk_index proximity within the same citation), capped at max_chars total.
    The source chunk is always included regardless of the budget.
    """
    entries = index["chunks"].get(citation, [])
    if not entries:
        return []

    pos = next((i for i, (ci, rid, off) in enumerate(entries) if rid == source_id), None)
    if pos is None:
        pos = next((i for i, (ci, rid, off) in enumerate(entries) if ci == source_chunk_index), 0)

    def fetch(i):
        ci, rid, off = entries[i]
        rec = reader.read_at(off)
        return {"id": rid, "chunk_index": ci, "text": rec.get("text", "")}

    chunks = [fetch(pos)]
    total_chars = len(chunks[0]["text"])

    lo, hi = pos - 1, pos + 1
    added = 0
    while added < 2 * window and (lo >= 0 or hi < len(entries)):
        if lo >= 0:
            c = fetch(lo)
            if total_chars + len(c["text"]) <= max_chars:
                chunks.append(c)
                total_chars += len(c["text"])
            lo -= 1
            added += 1
        if hi < len(entries) and added < 2 * window:
            c = fetch(hi)
            if total_chars + len(c["text"]) <= max_chars:
                chunks.append(c)
                total_chars += len(c["text"])
            hi += 1
            added += 1

    chunks.sort(key=lambda c: (c["chunk_index"] if c["chunk_index"] is not None else 0))
    return chunks


def format_evidence_block(chunks):
    parts = []
    for c in chunks:
        parts.append(
            f"--- EVIDENCE (chunk_id={c['id']}, corpus_chunk_index={c['chunk_index']}) ---\n{c['text']}"
        )
    return "\n\n".join(parts) if parts else "(no evidence found for this citation)"


def _normalize_ws(s):
    return " ".join(s.split())


def verify_evidence_used(evidence_used, evidence_chunks):
    """Check each cited quote is actually a substring of *some* evidence chunk
    (whitespace-normalized), regardless of which chunk_id the model claimed.

    Chunks can be uneven in size (a single corpus chunk can run to 100k+
    chars for cases), and models sometimes mislabel which chunk_id a quote
    came from — especially since case text has its own internal paragraph
    numbering (e.g. "[43]") that looks similar to our chunk_id/index labels.
    So rather than trusting the claimed chunk_id, we search every chunk
    actually provided as evidence and correct chunk_id to wherever the quote
    is really found. Only flag verified=False if the quote isn't found
    anywhere in the evidence. paragraph_index is passed through as-is (it's
    the model's read of the source's own internal numbering, not something we
    can independently check against corpus structure).
    """
    out = []
    for item in evidence_used or []:
        quote = (item.get("quote") or "").strip()
        claimed_id = item.get("chunk_id")
        matched_id = claimed_id
        ok = False
        if quote:
            norm_quote = _normalize_ws(quote)
            # Check the claimed chunk first, then fall back to the rest.
            ordered = sorted(evidence_chunks, key=lambda c: c["id"] != claimed_id)
            for c in ordered:
                if quote in c["text"] or norm_quote in _normalize_ws(c["text"]):
                    ok = True
                    matched_id = c["id"]
                    break
        out.append({
            "chunk_id": matched_id,
            "paragraph_index": item.get("paragraph_index"),
            "quote": item.get("quote"),
            "verified": ok,
        })
    return out


def build_answer_prompt(question_type, question, citation, title, jurisdiction, evidence_chunks):
    return ANSWER_PROMPT_TEMPLATE.format(
        question_type=question_type,
        question=question,
        citation=citation,
        title=title,
        jurisdiction=jurisdiction,
        evidence_block=format_evidence_block(evidence_chunks),
    )


def generate_answer(model, prompt, limiter=None):
    if limiter is not None:
        limiter.acquire()
    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"},
    )
    return extract_json_object(response.text)


def iter_question_tasks(path):
    """Yield one task dict per individual question in a questions(.jsonl) file."""
    for entry in iter_records(path):
        questions = entry.get("questions") or []
        for j, q in enumerate(questions):
            yield {
                "question_id": f"{entry.get('id')}:{j}",
                "source_id": entry.get("id"),
                "citation": entry.get("citation"),
                "title": entry.get("title"),
                "jurisdiction": entry.get("jurisdiction"),
                "chunk_index": entry.get("chunk_index"),
                "type": q.get("type"),
                "question": q.get("question"),
            }


def load_done_question_ids(path):
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
                continue
            qid = entry.get("question_id")
            if qid:
                done.add(qid)
    return done


def make_entry(task, model_name, result, evidence_used):
    entry = {
        "question_id": task["question_id"],
        "source_id": task["source_id"],
        "citation": task["citation"],
        "title": task["title"],
        "jurisdiction": task["jurisdiction"],
        "chunk_index": task["chunk_index"],
    }
    if "court_level" in task:
        entry["court_level"] = task["court_level"]
    if "opinion_type" in task:
        entry["opinion_type"] = task["opinion_type"]
    entry.update({
        "type": task["type"],
        "question": task["question"],
        "reasoning": result.get("reasoning"),
        "answer": result.get("answer"),
        "evidence_used": evidence_used,
        "model": model_name,
    })
    return entry


def add_common_args(parser, default_questions_input, default_output):
    parser.add_argument("--questions-input", default=default_questions_input,
                        help="Path to the questions file to answer (JSON array or JSONL).")
    parser.add_argument("--corpus", default=os.path.join(HERE, "legal_corpus.jsonl"),
                        help="Path to legal_corpus.jsonl.")
    parser.add_argument("--model", default="gemini-3.1-flash-lite",
                        help="Gemini model name to use for reasoning/answer generation.")
    parser.add_argument("--output", default=default_output,
                        help="Path to write results, one JSON object per question "
                             "(written incrementally).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max number of QUESTIONS to answer (0 = all). "
                             "Use a small number (e.g. 5) for a smoke test first.")
    parser.add_argument("--resume", action="store_true",
                        help="Append to an existing output file and skip questions "
                             "already present in it (instead of overwriting).")
    parser.add_argument("--rpm", type=float, default=10,
                        help="Max API requests per minute (default 10).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of concurrent request threads (default 8).")
    parser.add_argument("--window", type=int, default=3,
                        help="Number of neighboring chunks (same citation, by "
                             "chunk_index) to include on each side of the source "
                             "chunk as extra evidence context (default 3).")
    parser.add_argument("--max-evidence-chars", type=int, default=12000,
                        help="Max total characters of evidence text sent per "
                             "question (default 12000). The source chunk is "
                             "always included regardless of this budget.")
    parser.add_argument("--force-reindex", action="store_true",
                        help="Rebuild the corpus index cache even if a valid one exists.")


def run(args, build_prompt_fn, iter_tasks_fn=iter_question_tasks):
    """Shared driver.

    build_prompt_fn(task, evidence_chunks, index) -> prompt string.
    iter_tasks_fn(path) -> yields task dicts (defaults to the statute-shaped
    iterator; answergen_cases.py supplies its own to carry court_level/opinion_type).
    """
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Set GOOGLE_API_KEY (or GEMINI_API_KEY) in your environment.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    limiter = RateLimiter(args.rpm)
    index = build_or_load_corpus_index(args.corpus, force_rebuild=args.force_reindex)
    reader = CorpusReader(args.corpus)

    done = load_done_question_ids(args.output) if args.resume else set()
    if args.resume:
        print(f"Resuming: {len(done)} question(s) already in {args.output}", file=sys.stderr)

    skipped = 0
    todo = []
    for task in iter_tasks_fn(args.questions_input):
        if args.resume and task["question_id"] in done:
            skipped += 1
            continue
        if args.limit and len(todo) >= args.limit:
            break
        todo.append(task)

    if not todo:
        print("Nothing to do.", file=sys.stderr)
        return

    def work(task):
        evidence_chunks = get_evidence_chunks(
            index, reader, task["citation"], task["source_id"], task["chunk_index"],
            window=args.window, max_chars=args.max_evidence_chars,
        )
        prompt = build_prompt_fn(task, evidence_chunks, index)
        result = generate_answer(model, prompt, limiter)
        evidence_used = verify_evidence_used(result.get("evidence_used", []), evidence_chunks)
        return result, evidence_used

    processed = 0
    unverified = 0
    write_lock = threading.Lock()
    mode = "a" if args.resume else "w"
    with open(args.output, mode, encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(work, task): (i, task) for i, task in enumerate(todo, start=1)}
        for fut in as_completed(futures):
            i, task = futures[fut]
            label = f"{task['citation']} [{task['type']}]"
            try:
                result, evidence_used = fut.result()
            except Exception as e:
                print(f"[{i}] SKIP {label}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            entry = make_entry(task, args.model, result, evidence_used)
            line = json.dumps(entry, ensure_ascii=False) + "\n"
            with write_lock:
                out.write(line)
                out.flush()
                os.fsync(out.fileno())
                processed += 1
                if evidence_used and not any(e["verified"] for e in evidence_used):
                    unverified += 1
            flag = "" if not evidence_used or any(e["verified"] for e in evidence_used) else " (UNVERIFIED evidence)"
            print(f"[{i}] {label}{flag}", file=sys.stderr)

    msg = f"\nProcessed {processed} question(s)"
    if args.resume:
        msg += f" (skipped {skipped} already done)"
    if unverified:
        msg += f", {unverified} with unverified evidence quotes"
    print(f"{msg} -> {args.output}", file=sys.stderr)


def statute_build_prompt(task, evidence_chunks, index):
    return build_answer_prompt(
        task["type"], task["question"], task["citation"], task["title"],
        task["jurisdiction"], evidence_chunks,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate grounded reasoning + answers for statute questions via Gemini."
    )
    add_common_args(
        parser,
        default_questions_input=os.path.join(HERE, "questions.jsonl"),
        default_output=os.path.join(HERE, "answers.jsonl"),
    )
    args = parser.parse_args()
    run(args, statute_build_prompt)


if __name__ == "__main__":
    main()
