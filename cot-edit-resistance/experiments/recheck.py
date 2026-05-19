"""Re-score baseline_traces.json using the canonical normalize() from utils."""

import json
from pathlib import Path

from utils import normalize


def recheck(path: Path = Path("data/baseline_traces.json")) -> None:
    results = json.load(path.open())

    for r in results:
        gt = normalize(r["ground_truth"])
        pred = normalize(r["model_answer"]) if r["model_answer"] else None
        r["is_correct"] = pred == gt
        print(f"gt={gt!r}  pred={pred!r}  correct={r['is_correct']}")

    json.dump(results, path.open("w"), indent=2)


if __name__ == "__main__":
    recheck()
