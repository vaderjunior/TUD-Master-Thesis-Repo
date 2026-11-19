"""Policy quantile precision."""

from collections import Counter
import json
import numpy as np
from scipy.stats import beta
import yaml
from analysis.knowledge.policy import robustness as p

c = p.c


def precision():
    results = []
    for view in c.read(p.BASE / "sensitivity_index.json"):
        v = c.read(c.ROOT / view["path"])
        boot = dict(np.load(c.ROOT / v["bootstrap_source"], allow_pickle=False))
        for r in v["contrasts"]:
            if r["direction"] == "unresolved":
                continue
            a = p.NAMES.index(r["left"])
            b = p.NAMES.index(r["right"])
            d = boot[r["metric"]][:, a] - boot[r["metric"]][:, b]
            n = len(d)
            k = int((d <= 0).sum()) if r["direction"] == "positive" else int((d >= 0).sum())
            tail = (1 - r["family_confidence"]) / 2
            lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
            hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
            out = {
                k0: r[k0]
                for k0 in (
                    "endpoint",
                    "mask",
                    "metric",
                    "left",
                    "right",
                    "direction",
                    "family",
                    "family_interval",
                )
            }
            out.update(
                opposite_or_zero_draws=k,
                draws=n,
                tail_threshold=tail,
                opposite_tail_probability=k / n,
                monte_carlo95_binomial_probability_interval=[lo, hi],
                threshold_within_mc_band=lo <= tail <= hi,
            )
            results.append(out)
    value = dict(
        method="Exact binomial interval for the conditional bootstrap probability of an opposite-or-zero difference. This describes Monte Carlo sampling precision, not a new scientific confidence interval. No new draws or changed seed. Only originally nonzero saved intervals checked.",
        rows=results,
    )
    path = p.BASE / "monte_carlo_precision.json"
    if path.exists():
        c.require(c.read(path) == value, "Saved precision calculation differs")
    else:
        p.save(path.name, value)
    return len(results)


def missing():
    roster, arrays, rows = p.load()
    selected = [r for r in rows if r["result"] is None]
    contexts = {r["key"]: r for r in c.rows(p.P4 / "context_features.jsonl")}
    taxonomy = yaml.safe_load((c.ROOT / "config/taxonomy.yaml").read_text(encoding="utf8"))[
        "dimensions"
    ]
    parser = p.reader.tb.bundle(c.ROOT / "config/taxonomy.yaml").parser
    out = []
    errors = Counter()
    unknown = Counter()
    per_prediction = Counter()
    bindings = {}
    for row in selected:
        path = c.OUTPUT / "live/items" / row["stamp"] / "result.json"
        v = p.reader.checked(path, row["result_sha256"])
        bindings[c.ref(path)] = row["result_sha256"]
        chains = []
        seen = set()
        for i, chain in enumerate(v["raw_runs"]):
            details = []
            for j, raw in enumerate(chain):
                parsed, _, error = parser.parse_response(raw)
                assert parsed is None and error
                errors[error] += 1
                try:
                    obj = json.loads(parser.extract_json(raw))
                except Exception:
                    obj = None
                invalid = {}
                if isinstance(obj, dict):
                    for field in ("hate_type", "target_group", "legal"):
                        values = obj.get(field, [])
                        if isinstance(values, list):
                            bad = sorted(
                                {str(x) for x in values if x not in taxonomy[field]["labels"]}
                            )
                            if bad:
                                invalid[field] = bad
                                for x in bad:
                                    unknown[field + "/" + x] += 1
                                    seen.add(field + "/" + x)
                details.append(dict(turn=j, error=error, unknown_labels=invalid))
            chains.append(dict(vote=i, turns=details))
        per_prediction.update(seen)
        ctx = contexts[row["key"]]
        out.append(
            dict(
                key=row["key"],
                repeat=row["repeat"],
                source=roster[row["key"]]["source"],
                gold_gate=roster[row["key"]]["gate"],
                definitions_A=ctx["definitions_A"],
                guidelines_A=ctx["guidelines_A"],
                legal_guidelines_A=ctx["legal_guidelines_A"],
                chains=chains,
                result_sha256=row["result_sha256"],
            )
        )
    p.save(
        "null_taxonomy_review.json",
        dict(
            predictions=len(selected),
            unique_inputs=len({r["key"] for r in selected}),
            raw_texts=sum((len(ch["turns"]) for r in out for ch in r["chains"])),
            errors=errors,
            unknown_labels_per_response=unknown,
            unknown_labels_per_prediction=per_prediction,
            null_predictions_with_insult_definition=sum(
                ("def-legal-insult_defamation" in r["definitions_A"] for r in out)
            ),
            null_predictions_with_any_legal_guideline=sum(
                (r["legal_guidelines_A"] > 0 for r in out)
            ),
            rows=out,
            bindings=bindings,
            caution="A schema-invalid whole answer is unavailable even when other fields look usable. The original parser is retained. Policy association is descriptive and does not isolate which rule caused a field mix-up.",
        ),
    )
    return len(selected)


if __name__ == "__main__":
    with c.offline():
        n = precision()
        z = missing()
        p.save(
            "precision_missing_checks.json",
            dict(
                status="passed",
                nonzero_intervals=n,
                null_predictions=z,
                sources=c.bindings(
                    [
                        __file__,
                        p.BASE / "monte_carlo_precision.json",
                        p.BASE / "null_taxonomy_review.json",
                    ]
                ),
            ),
        )
        print(f"{n} saved interval precision checks; {z} null answers traced.", flush=True)
