import argparse
from collections import Counter
from pathlib import Path
from types import FunctionType, SimpleNamespace
from analysis.common import study as c
from analysis.classification.broad import preparation_retrieval as r
from analysis.common import prompt as prompt
from analysis.common import feedback_inputs as token_helper


def captured_check(module):

    captured = []

    def sink(path, value):
        c.require(
            Path(path).parent == c.PREP
            and Path(path).name in ("verification.json", "coverage_verification.json"),
            "Unexpected check output",
        )
        captured.append(value)

    proxy = SimpleNamespace(**vars(module.c) | {"save": sink})
    FunctionType(module.main.__code__, vars(module) | {"c": proxy})()
    c.require(len(captured) == 1, "Missing independent check result")
    return captured[0]


def main(study):
    dest = c.BASE / "prompt_checks" / f"{study}.json"
    if dest.exists():
        done = c.read(dest)
        c.verify(done["inputs"])
        c.verify(done["source_hashes"])
        print(study + " prompt check already complete.")
        return
    bindings = c.scientific_inputs()
    details = {}
    paths = []
    sources = [
        Path(__file__),
        Path(c.__file__),
        Path(prompt.__file__),
        Path(token_helper.__file__),
    ]
    if study == "matched":
        from analysis.classification import validate_preparation as original_check
        from analysis.classification import validate_coverage as coverage_check

        details["original_manipulation"] = captured_check(original_check)
        details["coverage_manipulation"] = captured_check(coverage_check)
        sources += [
            Path(original_check.__file__),
            Path(coverage_check.__file__),
            Path(original_check.c.__file__),
        ]
        paths = [
            c.PREP / "matched_prompts.jsonl",
            c.PREP / "coverage_matched_prompts.jsonl",
        ]
    else:
        done = c.read(r.BASE / "complete.json")
        c.verify(done["outputs"])
        bindings.update(done["outputs"])
        bindings[c.rel(r.BASE / "complete.json")] = c.sha(r.BASE / "complete.json")
        bindings[c.rel(r.PLAN)] = c.sha(r.PLAN)
        gold = {x["key"]: x for x in c.rows(c.PREP / "broad_roster.jsonl")}
        known = {
            pool: {x["id"]: x for x in c.rows(c.CORE / f"kb/{pool}/records.jsonl")}
            for pool in ("small", "full")
        }
        paths = [
            r.BASE / "small/prompts.jsonl",
            r.BASE / "full/prompts.jsonl",
            r.BASE / "zero_shot.jsonl",
        ]
        sources += [Path(r.__file__), Path(r.old.__file__)]
    tok = token_helper.tokenizer()
    counts = Counter()
    ranges = {}
    seen = set()
    for path in paths:
        bindings[c.rel(path)] = c.sha(path)
        for row in c.rows(path):
            ident = (row["key"], row["condition"])
            c.require(ident not in seen, "Duplicate prompt identity")
            seen.add(ident)
            c.require(
                c.signature(row["messages"]) == row["messages_sha256"],
                "Message hash mismatch",
            )
            if study == "broad":
                source = gold[row["key"]]
                c.require(
                    row["population"] == source["phase"]
                    and row["item_id"] == str(source["id"])
                    and (row["lang"] == source["lang"]),
                    "Query identity differs",
                )
                context = row.get(
                    "context",
                    {"definitions": [], "guidelines": [], "example_groups": []},
                )
                if row["condition"].startswith("rag_"):
                    r.old.validate_context(
                        context,
                        known[row["condition"].removeprefix("rag_")],
                        source["lang"],
                    )
                    c.require(
                        c.signature(context) == row["context_sha256"]
                        and prompt.hit_ids(context) == row["retrieved"],
                        "Context provenance differs",
                    )
                else:
                    c.require(
                        row["condition"] == "zero_shot", "Unknown broad condition"
                    )
                c.require(
                    prompt.render_messages(
                        source["text"], source["lang"], context, "corrected"
                    )
                    == row["messages"],
                    "Original prompt renderer differs",
                )
            n = token_helper.token_count(tok, row["messages"])
            counts[row["condition"]] += 1
            lo, hi = ranges.get(row["condition"], (n, n))
            ranges[row["condition"]] = (min(lo, n), max(hi, n))
    expected = 12233 if study == "matched" else 64881
    c.require(len(seen) == expected, "Missing prompt matrix entries")
    bindings[c.rel(token_helper.TOK / "complete.json")] = c.sha(
        token_helper.TOK / "complete.json"
    )
    for name, v in c.read(token_helper.TOK / "complete.json")["files"].items():
        bindings[c.rel(token_helper.TOK / name)] = v["sha256"]
    c.save(
        dest,
        {
            "status": "all_prompts_independently_verified",
            "study": study,
            "created_utc": c.utc(),
            "prompts": len(seen),
            "by_condition": dict(counts),
            "local_reference_token_ranges": ranges,
            "token_limit": 8192,
            "token_scope": "Pinned local ChatML token estimate, not a measurement of server-injected templates or vision processing. Text only.",
            "independent_checks": details,
            "inputs": bindings,
            "source_hashes": {c.rel(p): c.sha(p) for p in sources},
            "new_Qwen_calls": 0,
        },
    )
    print(study, "prompt check passed:", len(seen), "messages;", ranges, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--study", required=True, choices=["matched", "broad"])
    args = p.parse_args()
    with c.offline():
        main(args.study)
