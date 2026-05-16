#!/usr/bin/env python
"""
Phase 0: Generate baseline traces on MATH level-3 problems.

Run:
    pdm run python experiments/phase0_baseline.py
"""

import json
import re
import time
from pathlib import Path

import torch as t
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# ── config ───────────────────────────────────────────────────────────────────
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEVICE = "cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu"
MAX_NEW_TOKENS = 4096
LIMIT = 1  # set to None to run all level-3 problems
OUTPUT_PATH = Path("data/baseline_traces.json")

# ── setup ────────────────────────────────────────────────────────────────────
OUTPUT_PATH.parent.mkdir(exist_ok=True)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=t.float16).to(DEVICE)
model.eval()
print("Model loaded.\n")

# ── dataset ──────────────────────────────────────────────────────────────────
print("Loading MATH-500 level 3...")
full_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
dataset = full_dataset.filter(
    lambda x: (x["level"] == 3) or (x["level"] == "3") or (x["level"] == "Level 3")
)
print(f"{len(dataset)} level-3 problems loaded.\n")


def format_problem(row: dict) -> tuple[str, str]:
    """Return (prompt, ground_truth_answer)."""
    prompt = (
        "Solve the following math problem. Show your reasoning step by step.\n\n"
        f"{row['problem']}\n\n"
        r"Give your final answer inside \boxed{}."
    )
    # Extract answer from solution's \boxed{}
    m = re.search(r'\\boxed\{(.+?)\}', row["solution"])
    ground_truth = normalize(m.group(1)) if m else row["solution"].strip()
    return prompt, ground_truth


def normalize(answer: str) -> str:
    answer = answer.strip()
    answer = answer.replace(r"\dfrac", r"\frac")
    answer = answer.replace(r"\tfrac", r"\frac")
    answer = answer.replace(r"\left", "").replace(r"\right", "")
    answer = re.sub(r"\s+", "", answer)
    return answer


def parse_answer(text: str) -> str | None:
    """Extract the content of the last \\boxed{} in text."""
    matches = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', text)
    return normalize(matches[-1]) if matches else None


# ── load existing results for resumability ───────────────────────────────────
results: list[dict] = []
done_indices: set[int] = set()
if OUTPUT_PATH.exists():
    with open(OUTPUT_PATH) as f:
        results = json.load(f)
    done_indices = {r["index"] for r in results}
    print(f"Resuming: {len(done_indices)} already done.\n")

# ── main loop ────────────────────────────────────────────────────────────────
subset = dataset.select(range(LIMIT)) if LIMIT else dataset
for idx, row in enumerate(tqdm(subset, desc="MATH-L3")):
    if idx in done_indices:
        continue

    prompt, ground_truth = format_problem(row)

    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )["input_ids"].to(DEVICE)

    streamer = TextStreamer(tokenizer, skip_prompt=True)
    t0 = time.time()
    with t.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )
    elapsed = time.time() - t0

    generated_ids = output_ids[0, input_ids.shape[1]:]
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    think_match = re.search(r'<think>(.*?)</think>(.*)', full_text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer_section = think_match.group(2).strip()
    else:
        thinking = ""
        answer_section = full_text

    model_answer = parse_answer(answer_section) or parse_answer(thinking)
    is_correct = (model_answer == ground_truth) if model_answer else False

    results.append({
        "index": idx,
        "problem": row["problem"],
        "type": row.get("type", row.get("subject")),
        "ground_truth": ground_truth,
        "model_answer": model_answer,
        "is_correct": is_correct,
        "thinking": thinking,
        "answer_section": answer_section,
        "elapsed_s": round(elapsed, 1),
    })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

# ── summary ──────────────────────────────────────────────────────────────────
n_total = len(results)
n_correct = sum(r["is_correct"] for r in results)
n_parsed = sum(r["model_answer"] is not None for r in results)

print(f"\n{'='*50}")
print(f"Total:    {n_total}")
print(f"Parsed:   {n_parsed}/{n_total}")
print(f"Correct:  {n_correct}/{n_total}" + (f" ({100*n_correct/n_total:.1f}%)" if n_total else ""))
print(f"Saved to: {OUTPUT_PATH}")
