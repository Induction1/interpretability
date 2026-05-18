#!/usr/bin/env python
"""
Experiment 3: Commitment probe + logit lens at each reasoning depth.

For each correct trace, at depths 0.1-0.9:
  1. Truncate reasoning at that depth.
  2. Append a commitment prompt that forces the model into \boxed{...} format.
  3. Run one forward pass (output_hidden_states=True) for logit lens.
  4. Run model.generate (30 tokens) to get the probe answer.
  5. Save per-layer correct-token probabilities + probe answer.

Reads:  data/baseline_traces.json, data/experiment1_results.json
Writes: data/experiment3_results.json
"""

import json
import re
from pathlib import Path

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu"
MODEL_NAME  = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
OUTPUT_PATH = Path("data/experiment3_results.json")
DEPTHS      = [round(i / 10, 1) for i in range(1, 10)]
POOL_SIZE   = 3   # set to 55 for full run
PROBE_TOKENS = 30

# Ends with \boxed{ so the model is forced to complete inside the box.
COMMITMENT_PROMPT = "\n\nFinal answer: $\\boxed{"


def normalize(a: str) -> str:
    a = a.strip()
    a = a.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    a = a.replace(r"\left", "").replace(r"\right", "")
    a = re.sub(r"\s+", "", a)
    return a


def find_sentence_boundary(text: str, target: int) -> int:
    candidates = []
    for m in re.finditer(r"\.(?:\s)", text[: target + 50]):
        if m.start() <= target:
            candidates.append(m.end())
    for m in re.finditer(r"\n\n", text[: target + 50]):
        if m.start() <= target:
            candidates.append(m.end())
    return max(candidates) if candidates else target


def parse_probe(generated: str) -> str | None:
    # generated is text after \boxed{ — grab everything before the first }
    m = re.match(r"([^}]+)", generated)
    if not m:
        return None
    return normalize(m.group(1))


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=t.float16).to(DEVICE)
model.eval()
n_layers = len(model.model.layers)
print(f"Model loaded on {DEVICE}. Layers: {n_layers}\n")

with open("data/baseline_traces.json") as f:
    baseline = json.load(f)
with open("data/experiment1_results.json") as f:
    exp1 = json.load(f)

baseline_by_index = {r["index"]: r for r in baseline if r["is_correct"]}
pool_indices = sorted({r["index"] for r in exp1} & baseline_by_index.keys())[:POOL_SIZE]
print(f"Pool: {pool_indices}\n")

results: list[dict] = []
done_keys: set[tuple] = set()
if OUTPUT_PATH.exists():
    with open(OUTPUT_PATH) as f:
        results = json.load(f)
    done_keys = {(r["index"], r["depth"]) for r in results}
    print(f"Resuming from {len(done_keys)} completed runs.\n")

for idx in pool_indices:
    row = baseline_by_index[idx]
    raw = row["thinking"] if row["thinking"] else row["answer_section"]
    thinking = raw.split("</think>")[0].strip()
    ground_truth = normalize(row["ground_truth"])

    answer_ids = tokenizer(ground_truth, add_special_tokens=False)["input_ids"]
    if not answer_ids:
        print(f"[idx={idx}] skipping — empty answer tokenization")
        continue
    answer_token_id = int(answer_ids[0])
    answer_token = tokenizer.decode([answer_token_id])

    prompt = (
        "Solve the following math problem. Show your reasoning step by step.\n\n"
        f"{row['problem']}\n\n"
        r"Give your final answer inside \boxed{}."
    )
    messages = [{"role": "user", "content": prompt}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )["input_ids"].to(DEVICE)

    n_tokens = len(tokenizer.encode(thinking, add_special_tokens=False))

    for depth in DEPTHS:
        key = (idx, depth)
        if key in done_keys:
            continue

        boundary = find_sentence_boundary(thinking, int(len(thinking) * depth))
        prefix_text = thinking[:boundary]

        reasoning_prefix_ids = tokenizer.encode(
            prefix_text, add_special_tokens=False, return_tensors="pt"
        ).to(DEVICE)
        commit_prompt_ids = tokenizer.encode(
            COMMITMENT_PROMPT, add_special_tokens=False, return_tensors="pt"
        ).to(DEVICE)
        prefix_ids = t.cat([reasoning_prefix_ids, commit_prompt_ids], dim=1)
        input_ids = t.cat([prompt_ids, prefix_ids], dim=1)

        # logit lens at the last token of the reasoning prefix (before commitment prompt)
        # this position encodes "what the model knows after reasoning up to this depth"
        last_reasoning_pos = prompt_ids.shape[1] + reasoning_prefix_ids.shape[1] - 1

        # --- logit lens ---
        with t.no_grad():
            fwd = model(input_ids=input_ids, output_hidden_states=True)

        layers_out = []
        for li in range(n_layers):
            h = fwd.hidden_states[li + 1][:, last_reasoning_pos, :]
            logits = model.lm_head(model.model.norm(h))
            probs = t.softmax(logits, dim=-1)[0]
            correct_prob = float(probs[answer_token_id].item())
            top1_prob, top1_id = probs.max(dim=-1)
            layers_out.append({
                "layer": li,
                "correct_prob": correct_prob,
                "top1_token": tokenizer.decode([int(top1_id)]),
                "top1_prob": float(top1_prob.item()),
            })

        # --- commitment probe ---
        with t.no_grad():
            gen_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=PROBE_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(gen_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        probe_answer = parse_probe(generated)
        probe_correct = (normalize(probe_answer) == ground_truth) if probe_answer else False

        print(f"[idx={idx} | depth={depth} | probe={probe_answer!r} | correct={probe_correct}]")

        results.append({
            "index": idx,
            "depth": depth,
            "ground_truth": ground_truth,
            "answer_token": answer_token,
            "answer_token_id": answer_token_id,
            "probe_output": generated,
            "probe_answer": probe_answer,
            "probe_correct": probe_correct,
            "layers": layers_out,
        })
        done_keys.add(key)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

print(f"\nDone. {len(done_keys)} runs saved to {OUTPUT_PATH}")
