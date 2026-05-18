#!/usr/bin/env python
"""
Experiment 3: Incremental cache pass with logit lens + commitment probes.

For each correct trace:
  1. Tokenize the full reasoning trace and compute 10%-90% token checkpoints.
  2. Run the reasoning trace in chunks with use_cache=True, carrying KV cache.
  3. At each checkpoint:
     - Logit lens from hidden_states at last token for all 28 layers.
     - Commitment probe via generate() using the saved KV cache.
  4. Save one result per (index, depth).

Reads:
  - data/baseline_traces.json
  - data/experiment1_results.json
Writes:
  - data/experiment3_results.json
"""

import json
import re
from pathlib import Path

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
OUTPUT_PATH = Path("data/experiment3_results.json")
BASELINE_PATH = Path("data/baseline_traces.json")
EXPERIMENT1_PATH = Path("data/experiment1_results.json")
DEPTHS = [i / 10 for i in range(1, 10)]
COMMITMENT_PROMPT = "\n\nWait, I should state my current best answer. I'm confident the answer is:"
PROBE_MAX_NEW_TOKENS = 50


def normalize(a: str) -> str:
    a = a.strip()
    a = a.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    a = a.replace(r"\left", "").replace(r"\right", "")
    a = re.sub(r"\s+", "", a)
    return a


def prompt_for_problem(problem: str) -> str:
    return (
        "Solve the following math problem. Show your reasoning step by step.\n\n"
        f"{problem}\n\n"
        r"Give your final answer inside \boxed{}."
    )


def get_thinking_text(row: dict) -> str:
    raw = row["thinking"] if row["thinking"] else row["answer_section"]
    return raw.split("</think>")[0].strip()


def parse_answer(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    return normalize(matches[-1]) if matches else None


def greedy_probe_from_cache(
    model,
    tokenizer,
    base_past_key_values,
    prefix_ids: t.Tensor,
    max_new_tokens: int,
) -> t.Tensor:
    """Greedy probe rollout from a cached prefix context."""
    if max_new_tokens <= 0:
        return t.empty((1, 0), dtype=prefix_ids.dtype, device=prefix_ids.device)

    with t.no_grad():
        out = model(
            input_ids=prefix_ids,
            past_key_values=base_past_key_values,
            use_cache=True,
        )
    probe_past = out.past_key_values
    next_token = t.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    generated_tokens = [next_token]

    eos_id = tokenizer.eos_token_id
    for _ in range(max_new_tokens - 1):
        if eos_id is not None and int(next_token.item()) == eos_id:
            break
        with t.no_grad():
            out = model(
                input_ids=next_token,
                past_key_values=probe_past,
                use_cache=True,
            )
        probe_past = out.past_key_values
        next_token = t.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens.append(next_token)

    return t.cat(generated_tokens, dim=1)


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=t.float16).to(DEVICE)
model.eval()
print(f"Model loaded on {DEVICE}.\n")

n_layers = len(model.model.layers)
print(f"Detected layers: {n_layers}")
if n_layers != 28:
    raise ValueError(f"Expected 28 layers for this model, got {n_layers}.")

if not BASELINE_PATH.exists():
    raise FileNotFoundError("Missing data/baseline_traces.json. Run phase0_baseline.py first.")
if not EXPERIMENT1_PATH.exists():
    raise FileNotFoundError("Missing data/experiment1_results.json. Run experiment1.py first.")

with open(BASELINE_PATH) as f:
    baseline = json.load(f)
with open(EXPERIMENT1_PATH) as f:
    experiment1_results = json.load(f)

baseline_correct = [r for r in baseline if r["is_correct"]]
baseline_by_index = {r["index"]: r for r in baseline_correct}

# Use experiment1 indices as the canonical 55-trace set, then pull full rows from baseline.
candidate_indices = sorted({r["index"] for r in experiment1_results})
POOL_SIZE = 55
pool_indices = [idx for idx in candidate_indices if idx in baseline_by_index][:POOL_SIZE]
if len(pool_indices) < POOL_SIZE:
    raise ValueError(
        f"Expected at least {POOL_SIZE} usable indices from experiment1+baseline, got {len(pool_indices)}."
    )

print(f"Correct traces in baseline: {len(baseline_correct)}")
print(f"Pool indices from experiment1: {len(pool_indices)}\n")

results: list[dict] = []
done_keys: set[tuple[int, str]] = set()
if OUTPUT_PATH.exists():
    with open(OUTPUT_PATH) as f:
        results = json.load(f)
    done_keys = {(r["index"], f"{float(r['depth']):.2f}") for r in results if "depth" in r}
    print(f"Resuming: {len(done_keys)} completed (index, depth) runs.\n")

total_target = len(pool_indices) * len(DEPTHS)
commitment_ids = tokenizer.encode(
    COMMITMENT_PROMPT,
    add_special_tokens=False,
    return_tensors="pt",
).to(DEVICE)

for idx in pool_indices:
    row = baseline_by_index[idx]
    problem = row["problem"]
    ground_truth = normalize(row["ground_truth"])
    thinking_text = get_thinking_text(row)

    answer_ids = tokenizer(ground_truth, add_special_tokens=False)["input_ids"]
    if not answer_ids:
        print(f"[idx={idx}] skipping: empty answer tokenization for ground_truth={ground_truth!r}")
        continue
    answer_token_id = int(answer_ids[0])
    answer_token = tokenizer.decode([answer_token_id])

    reasoning_ids = tokenizer.encode(
        thinking_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(DEVICE)
    n_reason_tokens = int(reasoning_ids.shape[1])
    if n_reason_tokens == 0:
        print(f"[idx={idx}] skipping: empty reasoning tokenization")
        continue

    checkpoint_positions: list[int] = []
    prev_pos = 0
    for depth in DEPTHS:
        pos = int(n_reason_tokens * depth)
        pos = max(1, pos)
        pos = min(n_reason_tokens, pos)
        pos = max(prev_pos + 1, pos) if prev_pos < n_reason_tokens else n_reason_tokens
        checkpoint_positions.append(pos)
        prev_pos = pos

    prompt = prompt_for_problem(problem)
    messages = [{"role": "user", "content": prompt}]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )["input_ids"].to(DEVICE)

    with t.no_grad():
        prompt_out = model(
            input_ids=prompt_ids,
            use_cache=True,
        )
    past_key_values = prompt_out.past_key_values

    prev_token_pos = 0
    for depth, token_pos in zip(DEPTHS, checkpoint_positions):
        key = (idx, f"{depth:.2f}")
        chunk_ids = reasoning_ids[:, prev_token_pos:token_pos]
        prev_token_pos = token_pos
        if chunk_ids.shape[1] == 0:
            continue

        with t.no_grad():
            outputs = model(
                input_ids=chunk_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
        past_key_values = outputs.past_key_values

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden_states.")
        if len(hidden_states) != n_layers + 1:
            raise RuntimeError(
                f"Expected {n_layers + 1} hidden states (embedding + layers), got {len(hidden_states)}."
            )

        # Always run the incremental trace forward to keep KV cache aligned,
        # even when resuming and skipping this (index, depth) result.
        if key in done_keys:
            continue

        probe_new = greedy_probe_from_cache(
            model=model,
            tokenizer=tokenizer,
            base_past_key_values=past_key_values,
            prefix_ids=commitment_ids,
            max_new_tokens=PROBE_MAX_NEW_TOKENS,
        )[0]
        probe_output = tokenizer.decode(probe_new, skip_special_tokens=True)
        probe_answer = parse_answer(probe_output)
        probe_correct = (
            normalize(probe_answer) == ground_truth if probe_answer is not None else False
        )

        layers_out = []
        for layer_idx in range(n_layers):
            # hidden_states[0] is embedding; hidden_states[1 + i] is layer i output.
            h = hidden_states[layer_idx + 1][:, -1, :].detach()
            normed_h = model.model.norm(h)
            logits = model.lm_head(normed_h)
            probs = t.softmax(logits, dim=-1)

            correct_prob = float(probs[0, answer_token_id].item())
            top1_prob, top1_id = t.max(probs[0], dim=-1)
            top1_token_id = int(top1_id.item())
            top1_token = tokenizer.decode([top1_token_id])

            layers_out.append(
                {
                    "layer": layer_idx,
                    "correct_prob": correct_prob,
                    "top1_token": top1_token,
                    "top1_prob": float(top1_prob.item()),
                }
            )

        print(
            f'[idx={idx} | depth={depth:.2f} | probe_answer="{probe_answer}" | correct={probe_correct}]'
        )

        results.append(
            {
                "index": idx,
                "depth": float(f"{depth:.2f}"),
                "ground_truth": ground_truth,
                "answer_token": answer_token,
                "answer_token_id": answer_token_id,
                "probe_output": probe_output,
                "probe_answer": probe_answer,
                "probe_correct": probe_correct,
                "layers": layers_out,
            }
        )
        done_keys.add(key)

        with open(OUTPUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"Completed runs: {len(done_keys)}/{total_target}")
print(f"Saved to: {OUTPUT_PATH}")
