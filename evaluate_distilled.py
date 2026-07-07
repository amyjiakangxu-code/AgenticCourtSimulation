#!/usr/bin/env python3
"""Evaluate a LoRA-distilled model against the base model on held-out cases.

Runs both models (base, and base+adapter) on distill_data/test_eval.jsonl —
the evidence-free, deployment-style prompts (case description + question only,
no retrieved evidence) — and scores the generated reasoning+answer against the
original Gemini-generated reference using ROUGE and BERTScore.

Caveat: these are textual/semantic similarity metrics, not correctness
checkers. A model can phrase a legally correct answer differently from the
reference and score low, or phrase a wrong answer similarly and score high.
Treat scores as a relative signal (did fine-tuning move the needle vs. the
base model) rather than an absolute quality bar.

Usage:
    # Quick sanity check on a handful of examples first:
    python evaluate_distilled.py --limit 10
    # Full run:
    python evaluate_distilled.py --adapter-path ./distill_adapters_smoketest
"""

import argparse
import json
import os

from questiongen import HERE

MAX_NEW_TOKENS = 300


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def reference_text(rec):
    return f"Reasoning: {rec['reference_reasoning']}\n\nAnswer: {rec['reference_answer']}"


def generate_predictions(model_path, adapter_path, examples, max_tokens):
    from mlx_lm import generate, load

    model, tokenizer = load(model_path, adapter_path=adapter_path)
    preds = []
    for i, rec in enumerate(examples, start=1):
        prompt = tokenizer.apply_chat_template(
            rec["messages"], add_generation_prompt=True, tokenize=False
        )
        text = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        preds.append(text.strip())
        print(f"  [{i}/{len(examples)}] generated ({len(text)} chars)", end="\r")
    print()
    return preds


def score_predictions(preds, refs):
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score_fn

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    per_example = []
    for pred, ref in zip(preds, refs):
        scores = scorer.score(ref, pred)
        entry = {k: scores[k].fmeasure for k in rouge_totals}
        for k in rouge_totals:
            rouge_totals[k] += entry[k]
        per_example.append(entry)

    n = len(preds)
    rouge_avg = {k: v / n for k, v in rouge_totals.items()}

    print("  computing BERTScore (downloads a scoring model on first run)...")
    P, R, F1 = bert_score_fn(preds, refs, lang="en", verbose=False)
    bert_avg = {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}
    for i in range(n):
        per_example[i]["bertscore_f1"] = F1[i].item()

    return rouge_avg, bert_avg, per_example


def run_eval(label, model_path, adapter_path, examples, refs, max_tokens):
    print(f"\n=== {label} ===")
    preds = generate_predictions(model_path, adapter_path, examples, max_tokens)
    rouge_avg, bert_avg, per_example = score_predictions(preds, refs)
    print(f"  ROUGE-1 F1: {rouge_avg['rouge1']:.4f}  "
          f"ROUGE-2 F1: {rouge_avg['rouge2']:.4f}  "
          f"ROUGE-L F1: {rouge_avg['rougeL']:.4f}")
    print(f"  BERTScore  P: {bert_avg['precision']:.4f}  "
          f"R: {bert_avg['recall']:.4f}  F1: {bert_avg['f1']:.4f}")
    return preds, rouge_avg, bert_avg, per_example


def main():
    parser = argparse.ArgumentParser(description="Evaluate distilled model vs. base model.")
    parser.add_argument("--test-file", default=os.path.join(HERE, "distill_data", "test_eval.jsonl"))
    parser.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--adapter-path", default=os.path.join(HERE, "distill_adapters_smoketest"))
    parser.add_argument("--limit", type=int, default=0, help="Max test examples to evaluate (0 = all).")
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Only evaluate the distilled model, skip the base-model comparison.")
    parser.add_argument("--output", default=os.path.join(HERE, "eval_results.jsonl"))
    args = parser.parse_args()

    examples = list(iter_jsonl(args.test_file))
    if args.limit:
        examples = examples[:args.limit]
    refs = [reference_text(rec) for rec in examples]
    print(f"Evaluating on {len(examples)} test example(s) from {args.test_file}")

    results = {}
    if not args.skip_baseline:
        results["base"] = run_eval("base model (no adapter)", args.model, None, examples, refs, args.max_tokens)
    results["distilled"] = run_eval("distilled model (with adapter)", args.model, args.adapter_path,
                                     examples, refs, args.max_tokens)

    with open(args.output, "w", encoding="utf-8") as out:
        for i, rec in enumerate(examples):
            entry = {
                "question_id": rec.get("question_id"),
                "citation": rec.get("citation"),
                "type": rec.get("type"),
                "reference": refs[i],
            }
            for label, (preds, _, _, per_example) in results.items():
                entry[f"{label}_prediction"] = preds[i]
                entry[f"{label}_scores"] = per_example[i]
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nPer-example predictions + scores -> {args.output}")

    if "base" in results and "distilled" in results:
        print("\n=== Summary (distilled vs. base) ===")
        _, base_rouge, base_bert, _ = results["base"]
        _, dist_rouge, dist_bert, _ = results["distilled"]
        for k in ["rouge1", "rouge2", "rougeL"]:
            print(f"  {k}: base {base_rouge[k]:.4f} -> distilled {dist_rouge[k]:.4f}")
        print(f"  bertscore_f1: base {base_bert['f1']:.4f} -> distilled {dist_bert['f1']:.4f}")


if __name__ == "__main__":
    main()
