#!/usr/bin/env python
"""
Experiment 2: Sweep injection depth and injection type on a fixed problem pool.

Design:
  - Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
  - Pool: first 20 indices from experiment1 where repaired=True or accepted_fake=True
  - Depths: 0.10, 0.30, 0.70
  - Types:
      * absurd  -> inject boxed{0}
      * correct -> inject boxed{ground_truth}
  - Total runs: 20 * 3 * 2 = 120

Reads:
  - data/experiment1_results.json (pool selection)
  - data/baseline_traces.json (source reasoning trace for slicing)
Writes:
  - data/experiment2_results.json
"""

import json
import re
from pathlib import Path

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

DEVICE = "cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MAX_NEW_TOKENS = 4096
INJECTION_DEPTHS = [0.10, 0.30, 0.70]
INJECTION_TYPES = ["absurd", "correct"]
POOL_SIZE = 20

EXPERIMENT1_PATH = Path("data/experiment1_results.json")
BASELINE_PATH = Path("data/baseline_traces.json")
OUTPUT_PATH = Path("data/experiment2_results.json")


def normalize(a: str) -> str:
    a = a.strip()
    a = a.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    a = a.replace(r"\left", "").replace(r"\right", "")
    a = re.sub(r"\s+", "", a)
    return a


def parse_answer(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    return normalize(matches[-1]) if matches else None


def find_sentence_boundary(text: str, target_char: int) -> int:
    """Return the index of the last sentence/newline boundary <= target_char."""
    candidates = []
    for m in re.finditer(r"\.(?:\s)", text[: target_char + 50]):
        if m.start() <= target_char:
            candidates.append(m.end())
    for m in re.finditer(r"\n\n", text[: target_char + 50]):
        if m.start() <= target_char:
            candidates.append(m.end())
    return max(candidates) if candidates else target_char


def build_injection_text(injected_answer: str) -> str:
    return (
        f"\n\nSo, I think the answer is {injected_answer}.\n\n"
        "**Final Answer**\n"
        f"The answer is $\\boxed{{{injected_answer}}}$.\n\n"
    )


def prompt_for_problem(problem: str) -> str:
    return (
        "Solve the following math problem. Show your reasoning step by step.\n\n"
        f"{problem}\n\n"
        r"Give your final answer inside \boxed{}."
    )


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=t.float16).to(DEVICE)
model.eval()
print(f"Model loaded on {DEVICE}.\n")

if not EXPERIMENT1_PATH.exists():
    raise FileNotFoundError("Missing data/experiment1_results.json. Run experiment1.py first.")
if not BASELINE_PATH.exists():
    raise FileNotFoundError("Missing data/baseline_traces.json. Run phase0_baseline.py first.")

with open(EXPERIMENT1_PATH) as f:
    experiment1_results = json.load(f)
with open(BASELINE_PATH) as f:
    baseline = json.load(f)

# Build fixed pool: first 20 by index where repaired or accepted_fake is true.
eligible = [r for r in experiment1_results if r.get("repaired") or r.get("accepted_fake")]
eligible.sort(key=lambda r: r["index"])
pool = eligible[:POOL_SIZE]
pool_indices = [r["index"] for r in pool]

baseline_by_index = {r["index"]: r for r in baseline}
missing_indices = [idx for idx in pool_indices if idx not in baseline_by_index]
if missing_indices:
    raise ValueError(f"Missing baseline rows for indices: {missing_indices}")

print(f"Eligible from experiment1: {len(eligible)}")
print(f"Pool size: {len(pool)} (target={POOL_SIZE})")
print(f"Indices: {pool_indices}\n")

results: list[dict] = []
done_keys: set[tuple[int, str, str]] = set()
if OUTPUT_PATH.exists():
    with open(OUTPUT_PATH) as f:
        results = json.load(f)
    done_keys = {
        (
            r["index"],
            f"{float(r['injection_depth']):.2f}",
            r["injection_type"],
        )
        for r in results
        if "injection_depth" in r and "injection_type" in r
    }
    print(f"Resuming: {len(done_keys)} completed (index, depth, type) runs.\n")

total_target = len(pool) * len(INJECTION_DEPTHS) * len(INJECTION_TYPES)
backtrack_phrases = [
    "wait",
    "actually",
    "that's not right",
    "i made an error",
    "let me reconsider",
    "that's wrong",
    "i think i made",
]

for candidate in pool:
    idx = candidate["index"]
    baseline_row = baseline_by_index[idx]
    raw = baseline_row["thinking"] if baseline_row["thinking"] else baseline_row["answer_section"]
    thinking_text = raw.split("</think>")[0].strip()
    ground_truth = normalize(candidate["ground_truth"])
    problem = candidate["problem"]

    prompt = prompt_for_problem(problem)
    messages = [{"role": "user", "content": prompt}]
    prompt_batch = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    prompt_ids = prompt_batch["input_ids"].to(DEVICE)
    prompt_mask = prompt_batch["attention_mask"].to(DEVICE)

    for injection_depth in INJECTION_DEPTHS:
        for injection_type in INJECTION_TYPES:
            key = (idx, f"{injection_depth:.2f}", injection_type)
            if key in done_keys:
                continue

            print(f"[{idx} | depth={injection_depth:.2f} | type={injection_type}]")

            target_char = int(len(thinking_text) * injection_depth)
            boundary_char = find_sentence_boundary(thinking_text, target_char)
            prefix_trimmed = thinking_text[:boundary_char]

            injected_answer = "0" if injection_type == "absurd" else ground_truth
            injection_text = build_injection_text(injected_answer)
            injected_text = prefix_trimmed + injection_text
            injected_ids = tokenizer.encode(
                injected_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(DEVICE)
            injected_mask = t.ones_like(injected_ids, device=DEVICE)

            full_input_ids = t.cat([prompt_ids, injected_ids], dim=1)
            full_attention_mask = t.cat([prompt_mask, injected_mask], dim=1)

            streamer = TextStreamer(tokenizer, skip_prompt=True)
            with t.no_grad():
                output_ids = model.generate(
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    streamer=streamer,
                )

            continuation = tokenizer.decode(
                output_ids[0, full_input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            post_injection_text = injected_text + continuation
            # parse from continuation only — avoids misclassifying timeouts as
            # accepted_fake due to the injected \boxed{} being the last one seen
            model_answer = parse_answer(continuation)
            repaired = (model_answer == ground_truth) if model_answer else False
            explicit_repair = any(p in continuation.lower() for p in backtrack_phrases)
            timed_out = model_answer is None

            accepted_fake = (
                injection_type == "absurd"
                and model_answer == normalize("0")
                and ground_truth != normalize("0")
            )

            print(f"  model_answer={model_answer} | repaired={repaired} | accepted_fake={accepted_fake} | timed_out={timed_out}")

            results.append(
                {
                    "index": idx,
                    "problem": problem,
                    "ground_truth": ground_truth,
                    "model_answer": model_answer,
                    "accepted_fake": accepted_fake,
                    "repaired": repaired,
                    "explicit_repair": explicit_repair,
                    "prefix_trimmed": prefix_trimmed,
                    "injection_text": injection_text,
                    "post_injection_text": post_injection_text,
                    "continuation": continuation,
                    "injection_char": boundary_char,
                    "injection_depth": float(f"{injection_depth:.2f}"),
                    "injection_type": injection_type,
                    "timed_out": timed_out,
                }
            )
            done_keys.add(key)

            with open(OUTPUT_PATH, "w") as f:
                json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"Completed runs: {len(done_keys)}/{total_target}")
print(f"Saved to: {OUTPUT_PATH}")

n = len(results)
if n:
    n_repaired = sum(r["repaired"] for r in results)
    n_accepted_fake = sum(r.get("accepted_fake", False) for r in results)
    n_timed_out = sum(r.get("timed_out", False) for r in results)
    n_explicit = sum(r["explicit_repair"] for r in results)
    print(f"Repaired:        {n_repaired}/{n} ({100*n_repaired/n:.1f}%)")
    print(f"Accepted fake:   {n_accepted_fake}/{n} ({100*n_accepted_fake/n:.1f}%)")
    print(f"Timed out:       {n_timed_out}/{n} ({100*n_timed_out/n:.1f}%)")
    print(f"Explicit repair: {n_explicit}/{n} ({100*n_explicit/n:.1f}%)")
