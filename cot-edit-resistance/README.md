# CoT Edit Resistance

**Last updated:** 2026-05-16

## Research Question

When a reasoning model's chain-of-thought is corrupted mid-generation — by injecting a fake final answer — does the model repair back to its original answer? And what is happening internally that causes (or prevents) that repair?

This is an open question from Neel Nanda's research problems doc. Nobody has done the truncate-and-prepend + regeneration experiment with activation-level analysis on a distilled reasoning model.

## Key Result (Experiment 1)

We injected a fake `\boxed{0}` conclusion at the 30% mark of correct reasoning traces on MATH-500 level 3. DeepSeek-R1-Distill-Qwen-1.5B:

- **Repaired to correct answer: 49.1%**
- **Accepted fake answer: 49.1%**
- **Explicitly backtracked ("wait", "actually"): 94.5%**

The model almost always notices the injection is wrong. But it only escapes half the time. The failure mode is a repetition loop — the model correctly states the right answer, then references the injected text, then cycles between the two until it hits the token limit.

## The Experiment

**Dataset:** MATH-500 level 3 (105 problems, 55 correct baseline traces).

**Model:** DeepSeek-R1-Distill-Qwen-1.5B, fp16, CUDA.

**Injection method (truncate-and-prepend):**
1. Take the model's own correct reasoning trace
2. Cut at nearest sentence boundary to the 30% character mark
3. Append a fake conclusion in the model's natural format (`So, I think the answer is 0. **Final Answer** \boxed{0}`)
4. Feed full sequence back, generate continuation (greedy, temp=0)
5. Measure repair rate and explicit backtracking

This is a generation-time intervention, not prompt-level editing. The model cannot distinguish the prefix from tokens it generated itself.

## What's Next

White-box analysis: logit lens at layers 12-20 during the post-injection window. Does the correct answer stay represented in the residual stream in non-repair cases? What determines whether the model escapes the loop?

See `EXPERIMENTS.md` for the full experiment log.

## File Structure

```
experiments/
  experiment0.py    — baseline traces on MATH-500 level 3
  experiment1.py    — fake final answer injection, repair measurement
  experiment2.py    — white-box: logit lens on repair vs. non-repair cases (TODO)
data/
  baseline_traces.json
  experiment1_results.json
papers/             — reference PDFs
```

## Prior Work

**Zhang 2025** — Attention analysis of R1-Qwen-1.5B. Found Reasoning-Focus Heads (layers 12-20). Did residual stream patching on synthetic tasks. Did not study regeneration.

**Boppana 2026** — Showed R1-Distill-1.5B does genuine reasoning on hard tasks. Motivated use of MATH over GPQA-Diamond (too hard for 1.5B).

**Lanham 2023** — Prompt-level mistake injection. Showed edits don't always change answers. Did not use generation-time intervention or look at mechanisms.
