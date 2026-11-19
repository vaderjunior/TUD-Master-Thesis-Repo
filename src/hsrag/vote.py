"""Combine valid sampled responses into one prediction."""

from dataclasses import dataclass, field

from src.hsrag.schema import SEVERITY_ORDER, Result


@dataclass
class VoteResult:
    result: Result | None
    n_valid: int = 0
    n_runs: int = 0
    uncertain: bool = False
    agreement: dict = field(default_factory=dict)


def _median_severity(values: list[str]) -> str | None:
    """Return the lower median of emitted ordered severity values."""
    if not values:
        return None
    ranks = sorted(SEVERITY_ORDER[v] for v in values)
    rank = ranks[(len(ranks) - 1) // 2]
    return next(k for k, v in SEVERITY_ORDER.items() if v == rank)


def aggregate(runs: list) -> VoteResult:
    """Use strict-majority hate/labels and lower-median severity; legal stays independent."""
    valid = [r.result for r in runs if getattr(r, "ok", False)]
    out = VoteResult(result=None, n_valid=len(valid), n_runs=len(runs))
    if not valid:
        return out

    n = len(valid)
    yes = sum(1 for r in valid if r.hate)
    uncertain = (yes * 2 == n)
    hate = yes * 2 > n

    out.uncertain = uncertain
    out.agreement["hate"] = max(yes, n - yes) / n

    def majority_labels(field_name: str) -> list[str]:
        counts = {}
        for r in valid:
            for v in getattr(r, field_name):
                counts[v] = counts.get(v, 0) + 1
        kept = sorted(v for v, c in counts.items() if c * 2 > n)
        out.agreement[field_name] = (
            min(c / n for c in counts.values()) if counts else 1.0)
        return kept

    target_group = majority_labels("target_group")
    hate_type = majority_labels("hate_type")
    legal = majority_labels("legal")

    sevs = [r.severity for r in valid if r.severity is not None]
    severity = _median_severity(sevs)
    out.agreement["severity"] = len(sevs) / n if sevs else 1.0

    if uncertain:
        hate = False

    if not hate:
        target_group, hate_type, severity = [], [], None

    agreeing = [r for r in valid if r.hate == hate]
    reasoning = (agreeing[0] if agreeing else valid[0]).reasoning

    out.result = Result(
        reasoning=reasoning,
        hate=hate, target_group=target_group,
        hate_type=hate_type, severity=severity, legal=legal,
    )

    out.agreement["_dropped"] = [
        f for f in ("target_group", "hate_type", "legal")
        if any(getattr(r, f) for r in valid) and not getattr(out.result, f)]

    return out
