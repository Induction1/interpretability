# CoT Edit Resistance

**Last updated:** 2026-05-16

## Research Question

When a reasoning model's chain-of-thought is corrupted mid-generation — by token-forcing a wrong continuation — does the model repair back to its original answer upon regeneration? And what is happening internally that causes (or prevents) that repair?

This is an open question from Neel Nanda's research problems doc. Nobody has done the token-forcing + regeneration experiment with activation-level analysis on a distilled reasoning model.

## The Experiment

**Dataset:** GPQA-Diamond (198 graduate-level science questions). We use this because Boppana 2026 confirmed that R1-Distill-1.5B builds its answer gradually on hard questions rather than knowing from token 1 — which means "repair" is a real phenomenon, not just a strong prior surviving noise.

**Model:** DeepSeek-R1-Distill-Qwen-1.5B, fp16, MPS.

### Step 1 — Baseline (Phase 0, current)
Run the model on all GPQA-Diamond questions. Save full traces. Keep only the ones the model gets right — these are our experimental cases.

### Step 2 — Inject and measure repair (Phase 1)
For each correct trace, token-force a wrong continuation at some point mid-reasoning, then let the model regenerate freely. Measure: does it recover the correct answer?

Token-forcing means using a custom `LogitsProcessor` to override the model's output distribution at specific positions — not prompt editing, not greedy tricks. This is a genuine generation-time intervention.

### Step 3 — White-box analysis (Phase 2)
If repair exists, look at what's happening internally. Details TBD based on what the data shows. Starting point: Zhang 2025 identified Reasoning-Focus Heads in layers 12-20 of R1-Qwen-1.5B — that's where to look first.

## Key Prior Work

**Zhang 2025** — Attention analysis of R1-Qwen-1.5B. Found Reasoning-Focus Heads (layers 12-20) that carry reasoning content to answer tokens. Did residual stream patching on a synthetic task. Did not study regeneration.

**Boppana 2026** — Showed R1-Distill-1.5B does genuine reasoning on hard tasks (answer builds gradually). Easy tasks: answer committed from token 1. This is why we use GPQA-Diamond.

**Lanham 2023** — Behavioral mistake injection at the prompt level. Showed edits don't always change the answer. Did not use token-forcing or look at mechanisms.

## Kill Criterion

Repair rate < 20% on GPQA-Diamond → phenomenon too weak at 1.5B. Fallback: rent an A100, run on R1-Distill-7B.

## File Structure

```
experiments/
  phase0_baseline.py      — generate and save baseline traces
  phase1_injection.py     — token-forcing + repair measurement
  phase2_activations.py   — white-box analysis
data/
  baseline_traces.json
  repair_results.json
```
