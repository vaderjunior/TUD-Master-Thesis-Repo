"""Extract JSON, validate fields and apply bounded repair requests."""

import json
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from src.hsrag.schema import Result, normalise_raw

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def extract_json(raw: str) -> str:
    """Extract an object from fences or surrounding prose, respecting quoted braces."""
    text = _THINK.sub("", raw or "").strip()
    text = _FENCE.sub("", text).strip()

    start = text.find("{")
    if start == -1:
        return text

    depth, in_str, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


@dataclass
class RunResult:
    """One sampled response and its validation/repair records."""
    result: Result | None
    raw_outputs: list[str] = field(default_factory=list)
    repairs: int = 0
    normalisations: int = 0
    gate_normalised: bool = False
    parse_failure: bool = False
    error: str | None = None
    latency_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.result is not None


def parse_response(raw: str) -> tuple[Result | None, int, str | None]:
    """(result, normalisations, error). No API calls."""
    try:
        obj = json.loads(extract_json(raw))
    except json.JSONDecodeError as e:
        return None, 0, f"invalid JSON: {e}"
    if not isinstance(obj, dict):
        return None, 0, f"expected a JSON object, got {type(obj).__name__}"

    obj, n = normalise_raw(obj)
    try:
        return Result(**obj), n, None
    except ValidationError as e:
        return None, n, f"schema violation: {e.errors()[0]['msg']} " \
                        f"at {e.errors()[0]['loc']}"

REPAIR_TEMPLATE = (
    "Your previous output failed validation: {error}\n"
    "Return ONLY the corrected JSON object, with no prose and no code fences."
)


def run_once(client, messages: list[dict], max_repairs: int = 2) -> RunResult:
    """Request one response and retry invalid JSON up to the repair limit."""
    import time

    out = RunResult(result=None)
    convo = list(messages)

    for attempt in range(max_repairs + 1):
        t0 = time.time()
        raw = client.complete(convo)
        out.latency_s += time.time() - t0
        out.raw_outputs.append(raw)

        result, n, err = parse_response(raw)
        out.normalisations += n

        if result is not None:
            out.result = result
            out.gate_normalised = result.gate_normalised
            out.repairs = attempt
            return out

        out.error = err
        if attempt == max_repairs:
            break
        convo = convo + [
            {"role": "assistant", "text": raw},
            {"role": "user", "text": REPAIR_TEMPLATE.format(error=err)},
        ]

    out.repairs = max_repairs
    out.parse_failure = True
    return out
