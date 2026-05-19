# CoT Edit Resistance

Experiments on mid-generation injection into reasoning model chain-of-thought traces. Two phases: math domain (experiments 0-4) and safety domain (experiment 5).

Write-up: [induction1.github.io/research/cot-safety-injection](https://induction1.github.io/research/cot-safety-injection)

## The injection method

Take a model's reasoning trace in progress, truncate at some depth, append injected text, let the model continue generating. Measure whether it repairs back to its original answer (math) or complies with a harmful request (safety).

The key property: the model cannot structurally distinguish injected tokens from its own prior reasoning.

## Results

**Experiments 1-2 — Fake answer injection (DeepSeek-R1-Distill-Qwen-7B, MATH-500)**

Detection is near-perfect and depth-independent (98% explicit backtracking). Correction scales with reasoning depth completed before injection (31% at 10% depth, 51% at 70% depth). The model never accepts the fake — the failure mode is looping, not acceptance.

**Experiments 3-4 — Commitment layer (logit lens)**

Layer 26 (second-to-last of 28) is the commitment layer in 84% of runs. Layer 26 probability > 0.5 predicts answer correctness with 94% precision and recall — a near-binary phase transition. Injecting hedging language ("Wait, that doesn't seem right.") before the commit point disrupts layer 26 and causes looping rather than a wrong answer.

**Experiment 5 — Safety CoT injection (Qwen3-8B)**

Injected pro-compliance reasoning mid-think into a safety-aligned model across 24 AdvBench behaviors, three injection strengths (50 / 250 / 550 tokens). Zero genuine compliance across 72 runs. Counterintuitively, longer injections produced more decisive refusals (0 loops at 550 tokens vs. 3 at 250 tokens). The model treats injected argumentation as content to evaluate and refute, not as its own reasoning to continue from.

## File structure

```
experiments/
  experiment0.py          — baseline traces, MATH-500 level 3
  experiment1.py          — fake answer injection at 30% depth (1.5B)
  experiment2.py          — depth x injection-type sweep (7B)
  experiment3.py          — commitment probe + logit lens across all depths
  experiment4.py          — hedging token injection before commit
  experiment5.py          — harmful CoT injection on Qwen3-8B
  test_experiment5.py     — smoke tests for experiment5 (no GPU required)
outputs/
  experiment5/            — injection results, refusal direction plots, heatmaps
EXPERIMENTS.md            — full experiment log with raw numbers
RUNPOD_SETUP.txt          — RunPod setup instructions
```

## Setup

Experiments 0-4 run locally on Apple Silicon (MPS). Experiment 5 requires a GPU with 24GB+ VRAM (tested on RunPod RTX 4090). See `RUNPOD_SETUP.txt` for remote setup instructions.

```bash
pip install "transformers==4.51.3" accelerate requests matplotlib numpy hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 python experiments/experiment5.py
```

## Prior work

**Lanham et al. 2023** — prompt-level mistake injection, no generation-time intervention.

**H-CoT / Kuo et al. 2025 (2502.12893)** — injects harmful reasoning into the prompt before generation. We inject mid-stream during the model's own generation.

**CoT Hijacking / Zhao et al. 2025 (2510.26418)** — long benign prefix in the prompt passively dilutes the safety signal. Works at the prompt level, not mid-generation.

**Yamaguchi et al. 2025 (2507.03167)** — linear "caution" direction in DeepSeek-R1-Distill activation space during CoT generation. Closest to the mechanistic approach in experiment 5.
