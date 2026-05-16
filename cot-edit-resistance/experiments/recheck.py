import json
import re
from pathlib import Path


def normalize(a):
    a = a.strip()
    a = a.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    a = a.replace(r"\left", "").replace(r"\right", "")
    a = re.sub(r"\s+", "", a)
    return a


path = Path("data/baseline_traces.json")
results = json.load(path.open())

for r in results:
    gt = normalize(r["ground_truth"])
    pred = normalize(r["model_answer"]) if r["model_answer"] else None
    r["is_correct"] = pred == gt
    print(f"gt={gt!r}  pred={pred!r}  correct={r['is_correct']}")

json.dump(results, path.open("w"), indent=2)
