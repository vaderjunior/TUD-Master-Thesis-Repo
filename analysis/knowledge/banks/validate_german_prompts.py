from collections import Counter
import json
from pathlib import Path

from analysis.knowledge.banks import german_resume as run

p, u = run.p, run.u


def main():
    plan = run.validate()
    p.require(run.check_setup(), "Setup is incomplete")
    import pandas as pd

    expected = {}
    for subset in p.SCOPES.values():
        for row in pd.read_parquet(
            p.ROOT / f"data/processed/{subset}.parquet", columns=["id", "text", "lang"]
        ).itertuples(index=False):
            expected[(subset, str(row.id))] = (row.text, row.lang)
    originals = {r["id"]: r for r in u.read_lines(p.ROOT / "kb/records.jsonl")}
    pools = {
        name: {r["id"]: r for r in u.read_lines(p.BASE / f"kb/{name}/records.jsonl")}
        for name in p.POOLS
    }
    p.require(
        all(originals[rid] == r for rid, r in pools["original_small"].items()),
        "Original small records changed",
    )
    p.require(
        all(
            pools["canonical_full"][rid] == r
            for rid, r in pools["canonical_small"].items()
        ),
        "Small/full annotations differ",
    )
    outputs, prompts = {}, {}
    for condition, (pool, variant) in p.CONDITIONS.items():
        rows = u.read_lines(p.BASE / f"prompts_{condition}.jsonl")
        bykey = {(r["subset"], r["item_id"]): r for r in rows}
        p.require(
            len(rows) == len(bykey) == len(expected) and set(bykey) == set(expected),
            "Wrong dev roster",
        )
        summaries = {}
        for subset in p.SCOPES.values():
            stats = Counter()
            used = set()
            lengths = []
            for key, entry in bykey.items():
                if key[0] != subset:
                    continue
                p.require(
                    (entry["text"], entry["lang"]) == expected[key],
                    "Input text/language changed",
                )
                ctx = entry["context"]
                examples = [h for _, group in ctx["example_groups"] for h in group]
                p.require(
                    len(examples) == len({h["id"] for h in examples}) == 5,
                    "Wrong example budget",
                )
                for h in ctx["definitions"] + ctx["guidelines"] + examples:
                    record = pools[pool][h["id"]]
                    p.require(
                        h["text"] == record["text"], "Delivered text differs from KB"
                    )
                    meta = {
                        k: record[k]
                        for k in ("kind", "lang", "source", "dimension", "label")
                    }
                    meta.update(record["meta"])
                    p.require(
                        all(h["meta"].get(k) == v for k, v in meta.items()),
                        "Delivered annotation differs from KB",
                    )
                    p.require(
                        set(h["meta"]) <= set(meta) | {"_ranks"},
                        "Unexpected annotation keys",
                    )
                for h in examples:
                    meta = h["meta"]
                    p.require(meta["lang"] == "de", "English example delivered")
                    stats["example_slots"] += 1
                    stats["gate_" + str(meta.get("gate"))] += 1
                    stats["legal_annotated"] += meta.get("legal") is not None
                    stats["legal_positive"] += bool(meta.get("legal"))
                    stats["illustration_slots"] += bool(meta.get("illustrative_only"))
                    stats["auxiliary_paragraph_slots"] += bool(
                        meta.get("astra_aux_stgb")
                    )
                    used.add(h["id"])
                rendered = run.rendering.render_messages(
                    entry["text"], entry["lang"], ctx, variant
                )
                p.require(
                    rendered == entry["messages"], "Saved prompt differs from renderer"
                )
                p.require(
                    run.signature(rendered) == entry["messages_sha256"],
                    "Prompt hash differs",
                )
                p.require(
                    run.signature(ctx) == entry["context_sha256"],
                    "Context hash differs",
                )
                p.require(
                    run.rendering.hit_ids(ctx) == entry["retrieved"],
                    "Retrieved roster differs",
                )
                lengths.append(sum(len(m["text"]) for m in rendered))
                stats["items"] += 1
            summaries[subset] = {
                **dict(stats),
                "unique_examples": len(used),
                "prompt_chars_min": min(lengths),
                "prompt_chars_max": max(lengths),
                "prompt_chars_mean": sum(lengths) / len(lengths),
            }
        prompts[condition] = bykey
        outputs[condition] = summaries
    contrasts = Counter()
    for key in expected:
        aa, bb, cc, dd = (prompts[k][key] for k in p.CONDITIONS)
        p.require(aa["context"] == bb["context"], "Renderer contrast changed contexts")
        p.require(
            all(e["messages"][0] == aa["messages"][0] for e in (bb, cc, dd)),
            "System prompt changed",
        )
        for part in ("definitions", "guidelines"):
            p.require(
                all(e["context"][part] == aa["context"][part] for e in (bb, cc, dd)),
                "Abstract context changed",
            )
        ids = lambda e: [
            h["id"] for _, hh in e["context"]["example_groups"] for h in hh
        ]
        contrasts["a_b_changed_messages"] += aa["messages"] != bb["messages"]
        contrasts["b_c_same_example_set"] += set(ids(bb)) == set(ids(cc))
        contrasts["b_c_same_example_order"] += ids(bb) == ids(cc)
        contrasts["c_d_changed_example_set"] += set(ids(cc)) != set(ids(dd))
        contrasts["c_d_changed_messages"] += cc["messages"] != dd["messages"]
    p.require(
        contrasts["b_c_same_example_set"] == len(expected),
        "Annotation control changed selected example set",
    )
    p.require(
        not (p.BASE / "api_log.jsonl").exists(),
        "API activity exists; classify status separately",
    )
    p.require(
        not list((p.BASE / "runs").glob("*/items/*.json")), "Prediction files exist"
    )
    artifact = {
        "checked_utc": u.now(),
        "status": "prepared_not_executed",
        "dev_inputs": len(expected),
        "frozen_prompt_entries": sum(len(v) for v in prompts.values()),
        "conditions": outputs,
        "contrasts": dict(contrasts),
        "checks": [
            "All actual dev text/language rosters",
            "Every delivered example text and observed annotation",
            "Five distinct German examples per prompt",
            "Frozen messages reconstructed exactly",
            "System and abstract prompts equal across conditions",
            "A/B exact contexts equal",
            "B/C selected example sets equal; grouping order reported separately",
            "Canonical small byte-identical subset of full",
            "Original eligible records unchanged",
            "Frozen plan, source, cache, execution and setup hashes",
            "No classifier results or API log",
        ],
        "input_hashes": {
            u.rel(f): u.digest(f)
            for f in [
                p.PLAN,
                run.AMENDMENT,
                p.BASE / "setup_complete.json",
                Path(__file__),
            ]
        },
    }
    path = p.BASE / "setup_review.json"
    p.write(path, p.js(artifact))
    print(
        json.dumps(
            {
                "review": u.rel(path),
                "sha256": u.digest(path),
                "contrasts": dict(contrasts),
                "conditions": outputs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
