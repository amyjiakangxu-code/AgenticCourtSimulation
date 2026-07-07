#!/usr/bin/env python3
"""Build a LoRA fine-tuning dataset from answers_cases.jsonl.

Reads the grounded question/reasoning/answer records produced by
answergen_cases.py, attaches a CASE_SYNOPSIS per citation (reusing the same
three-tier fallback index built in answergen.py), and writes mlx-lm-compatible
chat-format JSONL files.

Training examples use question + verified evidence quotes (when available) as
input, matching what answergen_cases.py actually grounded its reasoning/answer
in. The held-out test set is written twice: once in the same format as
training (test.jsonl, for mlx-lm's own loss tracking), and once in the
evidence-free "deployment" format (test_eval.jsonl) — case description +
question only, no evidence — since that's the actual downstream condition the
distilled model needs to work under, and what evaluate_distilled.py scores.

Split is by citation (case), not by individual question, so no case's
questions are split across train/valid/test.

Usage:
    python prepare_distill_data.py
    python prepare_distill_data.py --train-frac 0.7 --valid-frac 0.15
"""

import argparse
import json
import os
import random

from questiongen import HERE
from answergen import build_or_load_corpus_index

SYSTEM_PROMPT = (
    "You are a Canadian legal reasoning assistant. Given a description of a "
    "case and a legal question, provide clear legal reasoning followed by a "
    "direct answer, grounded in Canadian law."
)

MAX_SYNOPSIS_CHARS = 1500
MAX_QUOTES_PER_EXAMPLE = 3
MAX_QUOTE_CHARS = 600


def iter_answer_records(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def verified_quotes(rec):
    quotes = []
    for e in rec.get("evidence_used") or []:
        if e.get("verified") and e.get("quote"):
            quotes.append(e["quote"].strip())
    return quotes[:MAX_QUOTES_PER_EXAMPLE]


def case_header(rec):
    return (
        f"CASE: {rec.get('title')} ({rec.get('citation')})\n"
        f"JURISDICTION: {rec.get('jurisdiction')}  "
        f"COURT_LEVEL: {rec.get('court_level')}  "
        f"OPINION_TYPE: {rec.get('opinion_type')}"
    )


def build_user_content(rec, synopsis, include_evidence):
    parts = [case_header(rec)]
    if synopsis:
        parts.append(f"CASE DESCRIPTION: {synopsis[:MAX_SYNOPSIS_CHARS]}")
    parts.append(f"QUESTION: {rec['question']}")
    if include_evidence:
        quotes = verified_quotes(rec)
        if quotes:
            bullets = "\n".join(f"- {q[:MAX_QUOTE_CHARS]}" for q in quotes)
            parts.append(f"EVIDENCE:\n{bullets}")
    return "\n\n".join(parts)


def build_assistant_content(rec):
    return f"Reasoning: {rec['reasoning']}\n\nAnswer: {rec['answer']}"


def make_chat_example(rec, synopsis, include_evidence):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_content(rec, synopsis, include_evidence)},
            {"role": "assistant", "content": build_assistant_content(rec)},
        ]
    }


def make_eval_example(rec, synopsis):
    """Deployment-style (no evidence) prompt + reference target, for scoring."""
    return {
        "question_id": rec.get("question_id"),
        "citation": rec.get("citation"),
        "title": rec.get("title"),
        "type": rec.get("type"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_content(rec, synopsis, include_evidence=False)},
        ],
        "reference_reasoning": rec.get("reasoning"),
        "reference_answer": rec.get("answer"),
    }


def split_citations(citations, train_frac, valid_frac, seed):
    citations = sorted(citations)  # sort first for determinism, then shuffle
    rng = random.Random(seed)
    rng.shuffle(citations)
    n = len(citations)
    n_train = round(n * train_frac)
    n_valid = round(n * valid_frac)
    train = set(citations[:n_train])
    valid = set(citations[n_train:n_train + n_valid])
    test = set(citations[n_train + n_valid:])
    return train, valid, test


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build LoRA distillation dataset from answers_cases.jsonl.")
    parser.add_argument("--input", default=os.path.join(HERE, "answers_cases.jsonl"))
    parser.add_argument("--corpus", default=os.path.join(HERE, "legal_corpus.jsonl"))
    parser.add_argument("--output-dir", default=os.path.join(HERE, "distill_data"))
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--valid-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    records = list(iter_answer_records(args.input))
    for i, rec in enumerate(records):
        rec.setdefault("question_id", f"{rec.get('source_id') or rec.get('citation')}:{i}")

    index = build_or_load_corpus_index(args.corpus)
    synopses = index["synopses"]

    citations = {rec["citation"] for rec in records}
    train_c, valid_c, test_c = split_citations(citations, args.train_frac, args.valid_frac, args.seed)

    train, valid, test = [], [], []
    for rec in records:
        synopsis = synopses.get(rec["citation"])
        cite = rec["citation"]
        if cite in train_c:
            train.append(make_chat_example(rec, synopsis, include_evidence=True))
        elif cite in valid_c:
            valid.append(make_chat_example(rec, synopsis, include_evidence=True))
        else:
            test.append(make_chat_example(rec, synopsis, include_evidence=True))

    test_eval = [
        make_eval_example(rec, synopses.get(rec["citation"]))
        for rec in records if rec["citation"] in test_c
    ]

    write_jsonl(train, os.path.join(args.output_dir, "train.jsonl"))
    write_jsonl(valid, os.path.join(args.output_dir, "valid.jsonl"))
    write_jsonl(test, os.path.join(args.output_dir, "test.jsonl"))
    write_jsonl(test_eval, os.path.join(args.output_dir, "test_eval.jsonl"))

    print(f"Citations: {len(train_c)} train / {len(valid_c)} valid / {len(test_c)} test "
          f"(of {len(citations)} total)")
    print(f"Examples:  {len(train)} train / {len(valid)} valid / {len(test)} test "
          f"({len(test_eval)} test_eval, evidence-free)")
    print(f"-> {args.output_dir}/")


if __name__ == "__main__":
    main()
