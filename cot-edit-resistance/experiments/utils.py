import re


def normalize(a: str) -> str:
    a = a.strip()
    a = a.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    a = a.replace(r"\left", "").replace(r"\right", "")
    a = re.sub(r"\s+", "", a)
    return a


def parse_answer(text: str) -> str | None:
    """Extract the last \\boxed{} answer from text, normalized."""
    matches = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', text)
    return normalize(matches[-1]) if matches else None


def find_sentence_boundary(text: str, target_char: int) -> int:
    """Return char index of the last sentence boundary at or before target_char."""
    candidates = []
    for m in re.finditer(r'\.(?:\s)', text[:target_char + 50]):
        if m.start() <= target_char:
            candidates.append(m.end())
    for m in re.finditer(r'\n\n', text[:target_char + 50]):
        if m.start() <= target_char:
            candidates.append(m.end())
    return max(candidates) if candidates else target_char


def get_thinking(row: dict) -> str:
    """Extract raw reasoning text from a saved trace row, stripping </think> and after."""
    raw = row["thinking"] if row["thinking"] else row["answer_section"]
    return raw.split("</think>")[0].strip()
