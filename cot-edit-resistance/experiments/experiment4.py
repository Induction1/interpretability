#!/usr/bin/env python
"""
Experiment 4: "Wait" token injection before commit on easy problems.

Research question: Is "wait" (and similar hedging tokens) a genuine trigger for
reconsideration, or just a surface marker that correlates with it? If we inject
"Wait, that doesn't seem right." immediately before the model commits to an
answer it is already confident about, does it change its answer?

Protocol (per problem × injection condition):
  1. Generate a base reasoning trace; locate the first \\boxed{ in the output.
  2. Truncate there — this is the natural commit point.
  3. Append the injection text and regenerate freely (no forced \\boxed{).
  4. Parse the first \\boxed{...} in the continuation.
  5. Record: did the answer change? Was it still correct?
  6. Run logit lens at the last token of the full input (end of injection text)
     to measure whether commitment at layer 26 was disrupted.

The "none" injection is the control: truncate at \\boxed{ and regenerate
immediately, confirming the model would have written the correct answer.

Reads:  nothing — problems are hardcoded below
Writes: data/experiment4_results.json
"""

import json
import re
from pathlib import Path

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu"
MODEL_NAME    = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
OUTPUT_PATH   = Path("data/experiment4_results.json")
BASE_MAX_TOKENS = 400   # for generating the initial reasoning trace
CONT_MAX_TOKENS = 200   # for post-injection continuation

# Five injection conditions. "none" is the unperturbed control.
INJECTIONS = {
    "none":       "",
    "wait":       "Wait, ",
    "wait_check": "Wait, let me double-check. ",
    "wait_wrong": "Wait, that doesn't seem right. ",
    "hmm":        "Hmm. ",
}

# Hand-curated easy problems. All answers are integers or simple fractions
# that a 7B reasoning model should answer in a short trace with near-certainty.
PROBLEMS = [
    # --- arithmetic ---
    {"id": "arith_01", "problem": r"What is $7 \times 8$?",                                          "answer": "56",            "category": "arithmetic"},
    {"id": "arith_02", "problem": r"What is $6 \times 9$?",                                          "answer": "54",            "category": "arithmetic"},
    {"id": "arith_03", "problem": r"What is $144 \div 12$?",                                         "answer": "12",            "category": "arithmetic"},
    {"id": "arith_04", "problem": r"What is $23 + 47$?",                                             "answer": "70",            "category": "arithmetic"},
    {"id": "arith_05", "problem": r"What is $100 - 37$?",                                            "answer": "63",            "category": "arithmetic"},
    {"id": "arith_06", "problem": r"What is $8^2$?",                                                 "answer": "64",            "category": "arithmetic"},
    {"id": "arith_07", "problem": r"What is $3^3$?",                                                 "answer": "27",            "category": "arithmetic"},
    {"id": "arith_08", "problem": r"What is $\sqrt{144}$?",                                          "answer": "12",            "category": "arithmetic"},
    {"id": "arith_09", "problem": r"What is $2^7$?",                                                 "answer": "128",           "category": "arithmetic"},
    {"id": "arith_10", "problem": r"What is $13 \times 4$?",                                         "answer": "52",            "category": "arithmetic"},
    # --- number theory ---
    {"id": "numth_01", "problem": r"What is $\gcd(24, 36)$?",                                        "answer": "12",            "category": "number_theory"},
    {"id": "numth_02", "problem": r"What is $\text{lcm}(4, 6)$?",                                    "answer": "12",            "category": "number_theory"},
    {"id": "numth_03", "problem": r"How many positive divisors does $36$ have?",                      "answer": "9",             "category": "number_theory"},
    {"id": "numth_04", "problem": r"What is the largest prime less than $20$?",                       "answer": "19",            "category": "number_theory"},
    {"id": "numth_05", "problem": r"What is $17 \mod 5$?",                                           "answer": "2",             "category": "number_theory"},
    # --- algebra ---
    {"id": "alg_01",   "problem": r"If $3x = 21$, what is $x$?",                                     "answer": "7",             "category": "algebra"},
    {"id": "alg_02",   "problem": r"If $x + 5 = 13$, what is $x$?",                                  "answer": "8",             "category": "algebra"},
    {"id": "alg_03",   "problem": r"If $2x - 3 = 11$, what is $x$?",                                 "answer": "7",             "category": "algebra"},
    # --- fractions ---
    {"id": "frac_01",  "problem": r"What is $\frac{1}{2} + \frac{1}{3}$? Express as a fraction.",    "answer": r"\frac{5}{6}",  "category": "fractions"},
    {"id": "frac_02",  "problem": r"What is $\frac{3}{4} - \frac{1}{4}$?",                           "answer": r"\frac{1}{2}",  "category": "fractions"},
    # --- geometry ---
    {"id": "geo_01",   "problem": r"What is the area of a rectangle with length $7$ and width $4$?", "answer": "28",            "category": "geometry"},
    {"id": "geo_02",   "problem": r"What is the perimeter of a square with side length $5$?",        "answer": "20",            "category": "geometry"},
    {"id": "geo_03",   "problem": r"What is the area of a right triangle with legs $6$ and $8$?",    "answer": "24",            "category": "geometry"},
    # --- sums / sequences ---
    {"id": "sum_01",   "problem": r"What is the sum of all integers from $1$ to $10$?",              "answer": "55",            "category": "sequences"},
    {"id": "sum_02",   "problem": r"What is $1 + 2 + 3 + \cdots + 20$?",                             "answer": "210",           "category": "sequences"},
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def normalize(a: str) -> str:
    a = a.strip()
    a = a.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    a = a.replace(r"\left", "").replace(r"\right", "")
    a = re.sub(r"\s+", "", a)
    return a


def parse_boxed(text: str) -> str | None:
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    return normalize(m.group(1)) if m else None


def run_logit_lens(model, tokenizer, input_ids, answer_token_id, n_layers):
    """Logit lens at the last token of input_ids across all layers."""
    with t.no_grad():
        fwd = model(input_ids=input_ids, output_hidden_states=True)
    layers_out = []
    for li in range(n_layers):
        h = fwd.hidden_states[li + 1][:, -1, :]
        logits = model.lm_head(model.model.norm(h))
        probs = t.softmax(logits, dim=-1)[0]
        top1_prob, top1_id = probs.max(dim=-1)
        layers_out.append({
            "layer": li,
            "correct_prob": float(probs[answer_token_id]),
            "top1_token": tokenizer.decode([int(top1_id)]),
            "top1_prob": float(top1_prob),
        })
    return layers_out


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=t.float16).to(DEVICE)
model.eval()
n_layers = len(model.model.layers)
print(f"Model loaded on {DEVICE}. Layers: {n_layers}\n")

results: list[dict] = []
done_keys: set[str] = set()
if OUTPUT_PATH.exists():
    with open(OUTPUT_PATH) as f:
        results = json.load(f)
    done_keys = {r["problem_id"] for r in results}
    print(f"Resuming from {len(done_keys)} completed problems.\n")


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

for prob in PROBLEMS:
    pid = prob["id"]
    if pid in done_keys:
        continue

    ground_truth = normalize(prob["answer"])
    answer_ids = tokenizer(ground_truth, add_special_tokens=False)["input_ids"]
    if not answer_ids:
        print(f"[{pid}] skipping — empty answer tokenization")
        continue
    answer_token_id = int(answer_ids[0])

    prompt_text = (
        "Solve the following math problem. Show your reasoning step by step.\n\n"
        f"{prob['problem']}\n\n"
        r"Give your final answer inside \boxed{}."
    )
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )["input_ids"].to(DEVICE)

    # --- step 1: generate base trace ---
    with t.no_grad():
        base_gen = model.generate(
            input_ids=prompt_ids,
            max_new_tokens=BASE_MAX_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    base_trace = tokenizer.decode(base_gen[0, prompt_ids.shape[1]:], skip_special_tokens=True)
    base_answer = parse_boxed(base_trace)
    base_correct = (base_answer == ground_truth) if base_answer else False

    print(f"[{pid}] base_answer={base_answer!r} correct={base_correct}")

    # --- step 2: find injection point ---
    BOXED = r"\boxed{"
    if BOXED not in base_trace:
        print(f"  no \\boxed{{ in base trace, skipping")
        continue

    prefix_text = base_trace[: base_trace.index(BOXED)]
    prefix_ids = tokenizer.encode(
        prefix_text, add_special_tokens=False, return_tensors="pt"
    ).to(DEVICE)

    # --- step 3: injection conditions ---
    injection_results = {}
    for cond, inj_text in INJECTIONS.items():
        if inj_text:
            inj_ids = tokenizer.encode(
                inj_text, add_special_tokens=False, return_tensors="pt"
            ).to(DEVICE)
            full_input = t.cat([prompt_ids, prefix_ids, inj_ids], dim=1)
        else:
            full_input = t.cat([prompt_ids, prefix_ids], dim=1)

        # logit lens at last token (= end of injection text, or end of prefix for "none")
        layers = run_logit_lens(model, tokenizer, full_input, answer_token_id, n_layers)

        # generate continuation freely
        with t.no_grad():
            cont_gen = model.generate(
                input_ids=full_input,
                max_new_tokens=CONT_MAX_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        continuation = tokenizer.decode(
            cont_gen[0, full_input.shape[1]:], skip_special_tokens=True
        )
        answer = parse_boxed(continuation)
        correct = (answer == ground_truth) if answer else False
        changed = (answer != base_answer)

        print(f"  [{cond:12s}] answer={answer!r} correct={correct} changed={changed}")

        injection_results[cond] = {
            "injected_text": inj_text,
            "continuation": continuation,
            "answer": answer,
            "correct": correct,
            "changed": changed,
            "layers": layers,
        }

    results.append({
        "problem_id": pid,
        "problem": prob["problem"],
        "ground_truth": ground_truth,
        "answer_token_id": answer_token_id,
        "category": prob["category"],
        "base_trace": base_trace,
        "base_answer": base_answer,
        "base_correct": base_correct,
        "prefix_text": prefix_text,
        "injections": injection_results,
    })
    done_keys.add(pid)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

print(f"\nDone. {len(done_keys)} problems saved to {OUTPUT_PATH}")
