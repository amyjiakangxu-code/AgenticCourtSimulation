#!/usr/bin/env python3
"""Generate grounded reasoning + answers for case questions using Gemini.

Companion to answergen.py (which handles questions.jsonl / statutes). This
script handles questions_cases.jsonl (produced by questiongen_cases.py). Reuses
the corpus index, evidence retrieval, verification, and CLI/driver machinery
from answergen.py, and adds the case-specific prompt (CASE_SYNOPSIS,
OPINION_TYPE, COURT_LEVEL).

Usage:
    export GOOGLE_API_KEY=...        # or GEMINI_API_KEY
    # Smoke test on 5 questions first:
    python answergen_cases.py --limit 5
    # Full run once the smoke test looks right:
    python answergen_cases.py --resume
    # Different model:
    python answergen_cases.py --model gemini-2.5-pro --limit 5
"""

import argparse
import os

from questiongen import HERE, iter_records
from answergen import add_common_args, format_evidence_block, run

CASE_ANSWER_PROMPT_TEMPLATE = """You are a Canadian legal research assistant answering an underlying legal question that was generated from a specific extract of a Canadian court decision. Ground your answer strictly in the EVIDENCE below — verbatim extracts from the same case (the originating extract plus surrounding chunks for context). CASE_SYNOPSIS is background orientation only; it is not itself citable evidence.

## QUESTION_TYPE
- grounded: the evidence should directly answer the question (the rule/holding in the evidence is a correct, direct answer).
- reasoning: the evidence supplies the governing principle; apply it to the hypothetical facts stated in the question.
- uncertainty: the evidence is relevant but does not actually resolve the question. Give an honest answer that states what the evidence does and does not settle — do not fabricate a holding the evidence doesn't support.

## TASK
1. Write "reasoning": a concise legal analysis (2-5 sentences) showing how the evidence leads to your answer, or, for uncertainty questions, exactly what the decision leaves open or doesn't address.
2. Write "answer": the direct answer a user would receive, as one short paragraph. Do not refer to "this case" or "this decision" in the answer — state the underlying legal answer directly, as questiongen_cases.py's questions do.
3. Write "evidence_used": the excerpt(s) that actually support your reasoning/answer, each as {{"chunk_id": <id>, "paragraph_index": <int or null>, "quote": "<verbatim substring from that excerpt's text>"}}.
   - "chunk_id" must be the chunk_id shown in the EVIDENCE header the quote came from.
   - "paragraph_index" is the paragraph number the DECISION ITSELF assigns to the quoted passage (e.g. a bracketed marker like [43] appearing in the text near the quote), if one is present; use null if none appears there.
   - Quotes must be exact substrings, not paraphrases. Cite only excerpts provided below. If nothing meaningfully supports the answer, return an empty list.

## RULES
- Do not invent holdings, rules, or facts that are not in the EVIDENCE.
- Do not rely on outside legal knowledge, even if you believe you know the answer — only what's in the EVIDENCE.
- For uncertainty questions, "this decision doesn't settle that" is the correct and expected answer — do not force a definitive rule the evidence doesn't support.
- Each EVIDENCE excerpt below is labeled "--- EVIDENCE (chunk_id=..., corpus_chunk_index=...) ---". corpus_chunk_index is only for your own orientation (this excerpt's position within the case) — never report it as paragraph_index. paragraph_index refers only to numbering that appears inside the excerpt's own text (the decision's own paragraph numbers).

## OUTPUT FORMAT
Return a single JSON object exactly like:
{{"reasoning": "...", "answer": "...", "evidence_used": [{{"chunk_id": ..., "paragraph_index": ..., "quote": "..."}}]}}

## PROVIDED
QUESTION_TYPE: {question_type}
QUESTION: {question}
CASE_SYNOPSIS: {case_synopsis}
CITATION: {citation}
TITLE: {title}
JURISDICTION: {jurisdiction}
COURT_LEVEL: {court_level}
OPINION_TYPE: {opinion_type}
EVIDENCE (extracts from this case):
{evidence_block}
"""


def iter_case_question_tasks(path):
    """Yield one task dict per individual question in a questions_cases(.jsonl) file."""
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
                "court_level": entry.get("court_level"),
                "opinion_type": entry.get("opinion_type"),
                "type": q.get("type"),
                "question": q.get("question"),
            }


def build_case_answer_prompt(task, evidence_chunks, index):
    synopsis = index["synopses"].get(task["citation"], "(no synopsis available)")
    return CASE_ANSWER_PROMPT_TEMPLATE.format(
        question_type=task["type"],
        question=task["question"],
        case_synopsis=synopsis,
        citation=task["citation"],
        title=task["title"],
        jurisdiction=task["jurisdiction"],
        court_level=task.get("court_level") or "",
        opinion_type=task.get("opinion_type") or "",
        evidence_block=format_evidence_block(evidence_chunks),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate grounded reasoning + answers for case questions via Gemini."
    )
    add_common_args(
        parser,
        default_questions_input=os.path.join(HERE, "questions_cases.jsonl"),
        default_output=os.path.join(HERE, "answers_cases.jsonl"),
    )
    args = parser.parse_args()
    run(args, build_case_answer_prompt, iter_tasks_fn=iter_case_question_tasks)


if __name__ == "__main__":
    main()
