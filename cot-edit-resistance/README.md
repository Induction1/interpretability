# CoT Edit Resistance

**Last updated:** 2026-05-18

## Research Question

When a reasoning model's chain-of-thought is corrupted mid-generation — by injecting a fake final answer — does the model repair back to its original answer? And what is happening internally that causes (or prevents) that repair?

This is an open question from Neel Nanda's research problems doc. Nobody has done the truncate-and-prepend + regeneration experiment with activation-level analysis on a distilled reasoning model.

## Key Results

**Experiment 1 — Fake answer injection at 30% depth (1.5B)**

We injected a fake `\boxed{0}` conclusion at the 30% mark of correct reasoning traces on MATH-500 level 3. DeepSeek-R1-Distill-Qwen-1.5B:

- **Repaired to correct answer: 49.1%**
- **Accepted fake answer: 49.1%**
- **Explicitly backtracked ("wait", "actually"): 94.5%**

The model almost always notices the injection is wrong. But it only escapes half the time. The failure mode is a repetition loop — the model correctly states the right answer, then references the injected text, then cycles between the two until it hits the token limit.

**Experiment 3 — Commitment probe + logit lens (7B)**

- Layer 26 is the commitment layer in 84% of runs (second-to-last of 28).
- Layer 26 probability > 0.5 predicts answer correctness with 94% precision and recall.
- Sharp phase transition: layers 0–24 predict noise; L25 shows semantic precursors; L26 jumps to the digit.

**Experiment 5 — Mid-generation harmful CoT injection (Qwen3-8B)**

A pivot toward safety: can you jailbreak a safety-aligned model by injecting harmful reasoning mid-generation into its chain-of-thought?

## The Injection Method

1. Take the model's own reasoning trace in progress
2. Cut at the nearest sentence boundary to the target depth
3. Append injected text (a fake conclusion, or harmful reasoning)
4. Feed the full sequence back and let the model continue (greedy, temp=0)
5. Measure repair/compliance rate and inspect internal activations

This is a generation-time intervention — the model cannot distinguish injected tokens from its own.

## File Structure

```
experiments/
  experiment0.py    — baseline traces on MATH-500 level 3
  experiment1.py    — fake final answer injection at 30%, repair measurement
  experiment2.py    — depth × injection-type sweep (7B)
  experiment3.py    — commitment probe + logit lens across all reasoning depths
  experiment4.py    — "Wait" token injection before commit
  experiment5.py    — mid-think harmful CoT injection on Qwen3-8B (safety pivot)
  utils.py          — shared parsing helpers
data/
  advbench.csv
  baseline_traces.json
  experiment1_results.json
  experiment2_results.json
  experiment3_results.json
  experiment4_results.json
```

## Setup

Python 3.11, [PDM](https://pdm-project.org/) for dependency management.

```bash
cd cot-edit-resistance
pdm install
```

Experiments were run on a RunPod RTX 4090. Models used: DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-7B, Qwen3-8B.

## Prior Work

**Zhang 2025** — Attention analysis of R1-Qwen-1.5B. Found Reasoning-Focus Heads (layers 12-20). Did residual stream patching on synthetic tasks. Did not study regeneration.

**Boppana 2026** — Showed R1-Distill-1.5B does genuine reasoning on hard tasks. Motivated use of MATH over GPQA-Diamond (too hard for 1.5B).

**Lanham 2023** — Prompt-level mistake injection. Showed edits don't always change answers. Did not use generation-time intervention or look at mechanisms.

**Chua et al. 2025 (H-CoT)** — Injected harmful reasoning into the *prompt* before generation. We inject mid-stream during generation — the model cannot distinguish injected tokens from its own.
