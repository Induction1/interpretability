# CoT Edit Resistance — Cursor Agent Guide

## What this project is

We token-force wrong continuations into a reasoning model's chain-of-thought mid-generation, let it regenerate, and study whether and how it repairs back to the correct answer. See README.md for full context.

## Rules

- `import torch as t` everywhere
- Python 3.11, PDM venv at `.venv/`
- Model: DeepSeek-R1-Distill-Qwen-1.5B, fp16, MPS
- Do NOT use TransformerLens (Qwen2.5 not supported)
- Do NOT use nnsight unless raw hooks become unworkable
- For activations: `model.model.layers[i].register_forward_hook(fn)`

## Do these phases in order — do not skip ahead

### Phase 0 — Baseline (experiments/phase0_baseline.py) ← START HERE
Run the model on GPQA-Diamond. Save full traces to `data/baseline_traces.json`. Only move on once you have traces and have read a few of them manually.

### Phase 1 — Injection (experiments/phase1_injection.py)
Token-force a wrong continuation mid-trace using a custom `LogitsProcessor`. Let the model regenerate freely. Measure repair rate. Save to `data/repair_results.json`.

```python
from transformers import LogitsProcessor

class ForceTokensProcessor(LogitsProcessor):
    def __init__(self, forced_token_ids: list[int], start_position: int):
        self.forced = forced_token_ids
        self.start = start_position
        self.step = 0

    def __call__(self, input_ids, scores):
        pos = self.step
        self.step += 1
        if self.start <= pos < self.start + len(self.forced):
            forced_id = self.forced[pos - self.start]
            scores[:] = -1e9
            scores[:, forced_id] = 0
        return scores
```

Kill criterion: repair rate < 20% → stop, consider moving to 7B.

### Phase 2 — White-box (experiments/phase2_activations.py)
Only start if Phase 1 shows interesting repair. Details TBD — read the data first. Starting point: layers 12-20 (Zhang 2025 RFHs). Use logit lens to check if correct answer is still internally represented during forced wrong content.
