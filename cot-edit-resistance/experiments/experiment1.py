#!/usr/bin/env python
"""
Experiment 1: Inject a fake "Final Answer: 0" at the nearest sentence boundary
to the 30% mark of the reasoning trace. Let the model continue and measure
whether it repairs back to the correct answer.

Strategy: truncate-and-prepend (not logits processor).
  1. Generate reasoning up to ~30% of baseline thinking token count.
  2. Find the nearest sentence boundary (. or newline) in the decoded text.
  3. Append injection text mimicking the model's natural final answer format.
  4. Feed the full sequence back, generate continuation.
  5. Measure repair.

Reads:  data/experiment0_results.json  (correct traces from experiment0)
Writes: data/experiment1_results.json
"""

import json
import re
from pathlib import Path

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

DEVICE = "cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MAX_NEW_TOKENS = 4096
INJECTION_DEPTH = 0.30
OUTPUT_PATH = Path("data/experiment1_results.json")

# mirrors the model's natural conclusion format, but wrong
INJECTION_TEXT = (
    "\n\nSo, I think the answer is 0.\n\n"
    "**Final Answer**\n"
    "The answer is $\\boxed{0}$.\n\n"
)


def normalize(a: str) -> str:
    a = a.strip()
    a = a.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    a = a.replace(r"\left", "").replace(r"\right", "")
    a = re.sub(r"\s+", "", a)
    return a


def parse_answer(text: str) -> str | None:
    matches = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', text)
    return normalize(matches[-1]) if matches else None


def find_sentence_boundary(text: str, target_char: int) -> int:
    """Return the index of the last sentence-ending char at or before target_char."""
    # look for '. ' or '.\n' or '\n\n' boundaries
    candidates = []
    for m in re.finditer(r'\.(?:\s)', text[:target_char + 50]):
        if m.start() <= target_char:
            candidates.append(m.end())
    for m in re.finditer(r'\n\n', text[:target_char + 50]):
        if m.start() <= target_char:
            candidates.append(m.end())
    return max(candidates) if candidates else target_char


# ── load model ────────────────────────────────────────────────────────────────
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=t.float16).to(DEVICE)
model.eval()
print(f"Model loaded on {DEVICE}.\n")

# ── load correct baseline traces ──────────────────────────────────────────────
baseline_path = Path("data/baseline_traces.json")
if not baseline_path.exists():
    raise FileNotFoundError("Run experiment0.py first.")

with open(baseline_path) as f:
    baseline = json.load(f)

correct = [r for r in baseline if r["is_correct"]]
print(f"Correct traces to inject into: {len(correct)}/{len(baseline)}\n")

# ── load existing results ─────────────────────────────────────────────────────
results: list[dict] = []
done_indices: set[int] = set()
if OUTPUT_PATH.exists():
    with open(OUTPUT_PATH) as f:
        results = json.load(f)
    done_indices = {r["index"] for r in results}
    print(f"Resuming: {len(done_indices)} already done.\n")

# ── main loop ─────────────────────────────────────────────────────────────────
for row in correct:
    idx = row["index"]
    if idx in done_indices:
        continue

    print(f"[{idx}] {row['problem'][:70]}...")

    prompt = (
        "Solve the following math problem. Show your reasoning step by step.\n\n"
        f"{row['problem']}\n\n"
        r"Give your final answer inside \boxed{}."
    )
    messages = [{"role": "user", "content": prompt}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )["input_ids"].to(DEVICE)

    # step 1: find sentence boundary at ~30% of the baseline thinking text
    # thinking is empty if <think> was prepended by template; fall back to answer_section
    # strip </think> and everything after so we only have the raw reasoning
    raw = row["thinking"] if row["thinking"] else row["answer_section"]
    thinking_text = raw.split("</think>")[0].strip()
    target_char = int(len(thinking_text) * INJECTION_DEPTH)
    boundary_char = find_sentence_boundary(thinking_text, target_char)
    prefix_trimmed = thinking_text[:boundary_char]

    # step 2: append injection text and encode
    injected_text = prefix_trimmed + INJECTION_TEXT
    injected_ids = tokenizer.encode(injected_text, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    full_input_ids = t.cat([prompt_ids, injected_ids], dim=1)

    print(f"  prefix ends: ...{repr(prefix_trimmed[-80:])}")
    print(f"  injecting:   {repr(INJECTION_TEXT[:60])}...")

    # step 4: generate continuation
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    with t.no_grad():
        output_ids = model.generate(
            full_input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

    continuation = tokenizer.decode(
        output_ids[0, full_input_ids.shape[1]:], skip_special_tokens=True
    )

    # step 5: measure repair on the full post-injection trajectory.
    # Using continuation alone misses the case where the model accepts the
    # injected boxed answer and emits little/no additional text.
    post_injection_text = injected_text + continuation
    ground_truth = normalize(row["ground_truth"])
    model_answer = parse_answer(post_injection_text)
    repaired = (model_answer == ground_truth) if model_answer else False
    fake_answer = normalize("0")
    accepted_fake = (model_answer == fake_answer) and (ground_truth != fake_answer)

    backtrack_phrases = ["wait", "actually", "that's not right", "i made an error",
                         "let me reconsider", "that's wrong", "i think i made"]
    explicit_repair = any(p in continuation.lower() for p in backtrack_phrases)

    print(f"  ground truth:    {ground_truth}")
    print(f"  model answer:    {model_answer}")
    print(f"  accepted fake:   {accepted_fake}")
    print(f"  repaired:        {repaired}  |  explicit backtrack: {explicit_repair}\n")

    results.append({
        "index": idx,
        "problem": row["problem"],
        "ground_truth": ground_truth,
        "model_answer": model_answer,
        "accepted_fake": accepted_fake,
        "repaired": repaired,
        "explicit_repair": explicit_repair,
        "prefix_trimmed": prefix_trimmed,
        "injection_text": INJECTION_TEXT,
        "post_injection_text": post_injection_text,
        "continuation": continuation,
        "injection_char": boundary_char,
    })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

# ── summary ───────────────────────────────────────────────────────────────────
n = len(results)
n_repaired = sum(r["repaired"] for r in results)
n_accepted_fake = sum(r.get("accepted_fake", False) for r in results)
n_explicit = sum(r["explicit_repair"] for r in results)

print(f"\n{'='*50}")
print(f"Total:           {n}")
if n:
    print(f"Accepted fake:   {n_accepted_fake}/{n} ({100*n_accepted_fake/n:.1f}%)")
    print(f"Repaired:        {n_repaired}/{n} ({100*n_repaired/n:.1f}%)")
    print(f"Explicit repair: {n_explicit}/{n} ({100*n_explicit/n:.1f}%)")
print(f"Saved to:        {OUTPUT_PATH}")
