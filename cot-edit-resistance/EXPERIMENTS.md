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

## Next: Experiment 2 — White-box analysis
Look at what's happening internally in the repair vs. non-repair cases. Starting point: logit lens at layers 12-20 during the post-injection window. Does the correct answer stay represented in the residual stream even in non-repair cases? If so, what's preventing it from making it to the surface?
