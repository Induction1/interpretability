# Experiment Log

## Experiment 0 — Baseline traces
**Date:** 2026-05-16
**Script:** `experiments/experiment0.py`
**Dataset:** MATH-500 level 3 (105 problems)
**Model:** DeepSeek-R1-Distill-Qwen-1.5B, fp16, CUDA (RunPod RTX 5090)

**Results:**
- 55/105 correct (52.4%)
- 74/105 parsed (31 hit token limit before concluding)
- ~14 seconds per trace

**Notes:** 31 unparsed traces all hit the 4096 token limit mid-reasoning. Not a parsing bug — those problems just needed more space. 55 correct traces used as input to experiment 1.

---

## Experiment 1 — Fake final answer injection at 30%
**Date:** 2026-05-16
**Script:** `experiments/experiment1.py`
**Input:** 55 correct traces from experiment 0

**Setup:**
- Take the model's own correct reasoning trace
- Cut it off at the nearest sentence boundary to the 30% character mark
- Append a fake conclusion mimicking the model's natural format:
  ```
  So, I think the answer is 0.

  **Final Answer**
  The answer is $\boxed{0}$.
  ```
- Feed the full sequence back and let the model continue generating (greedy, temp=0)
- Measure whether it repairs to the correct answer or accepts the fake

**Results:**
- Repaired to correct answer: **27/55 (49.1%)**
- Accepted fake answer (0):   **27/55 (49.1%)**
- Explicit backtracking:      **52/55 (94.5%)**
- Other:                       1/55 (1.8%)

**Key observation:**
The model almost always notices something is wrong (94.5% explicit backtrack, saying "wait", "actually", etc.). But it only manages to override the injected fake half the time. The failure mode is a repetition loop: the model correctly identifies the right answer in its reasoning, then references the injected "user wrote \boxed{0}" and gets stuck cycling between the two without committing. It knows the right answer but cannot override the surface text.

**What this means:** The repair phenomenon exists and is strong (49% repair on a maximally aggressive injection). The interesting question is what determines whether the model escapes the loop or not — that's the mechanistic question for experiment 2.

---

## Experiment 2 — Depth × injection-type sweep (7B)
**Date:** 2026-05-17
**Script:** `experiments/experiment2.py`
**Model:** DeepSeek-R1-Distill-Qwen-7B, fp16, CUDA (RunPod RTX 4090)
**Dataset:** MATH-500 level 3, 196 runs across injection depth × type

**Setup:** Swept injection depth (0.10, 0.30, 0.70) × injection type (absurd=`\boxed{0}`, correct=`\boxed{ground_truth}`), ~48 runs per condition.

**Results:**
- Absurd injection: 98% explicit backtrack, 31% correction at depth 0.10, 51% at depth 0.70
- Correct injection: ~87% repair, ~12% timeout even with the right answer injected
- Detection is depth-independent. Correction scales with how much reasoning was completed.
- 0% accepted\_fake (with fixed metric) — the model never commits to the wrong answer.

**Key finding:** Detection and correction are dissociated. The model always notices the injection is foreign, but escaping the resulting loop requires sufficient prior reasoning.

---

## Experiment 3 — Commitment probe + logit lens at each reasoning depth
**Date:** 2026-05-17–18
**Script:** `experiments/experiment3.py`
**Model:** DeepSeek-R1-Distill-Qwen-7B, fp16, CUDA (RunPod RTX 4090)
**Dataset:** 55 correct traces from MATH-500 level 3 (same pool as experiment 1)

**Setup:** For each problem at depths 0.1–0.9:
- Truncate reasoning at depth, append `\n\nFinal answer: $\boxed{` (forces the model to complete the box).
- Forward pass with `output_hidden_states=True` → logit lens at the `{` token (last input token) across all 28 layers.
- `model.generate` (30 tokens) → parse the probe answer.

**Results (495 runs, all complete):**
- Probe correctness increases monotonically: 42% at depth 0.1 → 89% at depth 0.9.
- **Layer 26 is the commitment layer** in 84% of runs (second-to-last of 28 layers).
- Layer 26 probability > 0.5 predicts probe correctness with 94% precision and 94% recall.
- Probe-correct runs: avg peak prob = 0.944. Probe-wrong runs: avg peak prob = 0.167.
- Three problem types: 23 "always commit" (correct even at depth 0.1), 27 "threshold" (first correct between 0.2–0.9), 5 "never correct" (data issues or genuine failures).
- Layer profile: L00–L24 predict noise/irrelevant tokens; L25 shows semantic precursors (e.g., "nine" before "9"); L26 jumps sharply to the digit.

**Key finding:** The answer crystallizes in the last two layers via a sharp phase transition. The logit lens at the `{` token is a near-perfect binary classifier for whether the model will commit to the correct answer.

---

## Experiment 4 — "Wait" token injection before commit (easy problems)
**Date:** 2026-05-18
**Script:** `experiments/experiment4.py`
**Model:** DeepSeek-R1-Distill-Qwen-7B, fp16, CUDA (RunPod RTX 4090)
**Dataset:** 25 hand-curated easy problems (arithmetic, number theory, algebra, fractions, geometry, sequences)

**Motivation:** Experiments 1–2 showed "wait"/"actually" appear in 94–98% of post-injection traces. Is "wait" a genuine trigger for reconsideration, or just a surface marker? If we inject hedging language immediately before the model commits to an answer it is already confident about, does it change its answer?

**Setup:** For each problem × injection condition:
1. Generate a base trace; locate the first `\boxed{` (the natural commit point).
2. Truncate there and append the injection text (no forced `\boxed{}`).
3. Regenerate freely (200 tokens); parse the first `\boxed{...}` in the continuation.
4. Logit lens at the last token of the injected input (end of injection text).

**Injection conditions:**
- `none` — control, truncate and regenerate with no injection
- `wait` — "Wait, "
- `wait_check` — "Wait, let me double-check. "
- `wait_wrong` — "Wait, that doesn't seem right. "
- `hmm` — "Hmm. "

**Results:** TODO — run pending.
