from collections import Counter
from copy import deepcopy
import time
from analysis.feedback.reserved import common as c
from analysis.common import prompt as renderer
from analysis.common import feedback_inputs as tok_util
from analysis.feedback.balanced.validate import SyntheticTransport


def expected_context(base, mode, record=None):

    definitions = deepcopy(base["definitions"])
    guidelines = deepcopy(base["guidelines"])
    hits = deepcopy([h for _, hs in base["example_groups"] for h in hs])
    c.require(len(hits) == 5, "Independent base count")
    if mode == "reference4":
        hits.pop()
    elif mode in ("reserved5", "extended6", "masked5"):
        if mode != "extended6":
            hits.pop()
        added = {
            "id": record["id"],
            "text": record["text"],
            "meta": deepcopy(record["meta"]),
            "via": "rrf",
        }
        if mode == "masked5":
            added["meta"] = {k: None for k in added["meta"]}
        hits.append(added)
    else:
        c.require(mode == "reference5", "Unknown independent transform")
    return {
        "definitions": definitions,
        "guidelines": guidelines,
        "example_groups": [["Labelled examples", hits]],
    }


def check_prompts():
    inputs = c.read(c.BASE / "preparation_plan.json")["inputs"]
    c.verify(inputs)
    prepared = c.read(c.BASE / "prepared.json")
    c.verify(prepared["outputs"])
    if (c.BASE / "complete.json").exists():
        receipt = c.read(c.BASE / "complete.json")
        c.require(
            receipt["prepared_sha256"] == c.sha(c.BASE / "prepared.json"),
            "Prepared binding changed",
        )
        c.require(
            receipt["verification_sha256"] == c.sha(c.ROOT / receipt["verification"]),
            "Verification binding changed",
        )
        c.verify(c.read(c.ROOT / receipt["verification"])["checker_sources"])
        print("Existing A15 independent preparation receipt verified.")
        return
    roster, pools, records, references, natural, corrections, choices = c.load_sources()
    current_roster = c.rows(c.BASE / "roster.jsonl")
    c.require(current_roster == roster, "Roster/gold changed")
    science = c.read(c.BASE / "scientific_plan_v1.json")
    conditions = c.condition_list(pools)
    c.require(
        science["conditions"] == conditions and science["ordinary_calls"] == 449550,
        "Scientific matrix differs",
    )
    c.require(
        science["primary"]["confidence"] == 0.975
        and science["mechanism"]["confidence"] == 1 - 0.05 / 6,
        "Inference family changed",
    )
    c.require(
        science["mandatory_excluded_keys"]
        == c.read(c.A14 / "cold_replay_review_v1/sensitivity_plan.json")[
            "excluded_keys"
        ],
        "Replay exclusion changed",
    )
    c.require(
        c.rows(c.BASE / "choices.jsonl") == [choices[k] for k in sorted(choices)],
        "Source-choice ledger changed",
    )
    poolmap = {p["condition"]: p for p in pools}
    bykey = {r["key"]: r for r in roster}
    tok = tok_util.tokenizer()
    seen = set()
    counts = Counter()
    tokens = []
    checks = Counter()
    science_sha = c.sha(c.BASE / "scientific_plan_v1.json")
    for p in c.stream(c.BASE / "prompts.jsonl"):
        ident = (p["key"], p["condition"])
        c.require(ident not in seen, "Duplicate prepared prompt")
        seen.add(ident)
        r = bykey[p["key"]]
        base = references[p["key"]]
        c.require(
            p["text"] == r["text"]
            and p["item_id"] == r["id"]
            and (p["population"] == r["population"])
            and (p["lang"] == "en"),
            "Query attribution changed",
        )
        c.require(p["scientific_plan_sha256"] == science_sha, "Prompt/plan mismatch")
        if "__" in p["condition"]:
            mode, pool = p["condition"].split("__", 1)
            chosen = corrections[p["key"], pool]
            sid = chosen["source_id"]
            record = records[sid]
            c.require(
                sid == p["source_id"] == choices[p["key"], pool]["source_id"]
                and sid in poolmap[pool]["ids"],
                "Independent source mismatch",
            )
            c.require(
                record["meta"]["gate"] == choices[p["key"], pool]["source_gate"],
                "Independent source gate mismatch",
            )
            c.require(
                p["selected_correction_cosine"] == choices[p["key"], pool]["cosine"],
                "Independent cosine mismatch",
            )
            c.require(
                record["provenance"]["source_id"] == sid and sid != r["id"],
                "Source provenance/query leak",
            )
            expected = (
                deepcopy(natural[p["key"], pool]["context"])
                if mode == "natural5"
                else expected_context(base["context"], mode, record)
            )
            if mode == "natural5":
                c.require(
                    p["messages"] == natural[p["key"], pool]["messages"],
                    "Natural message reuse failed",
                )
                checks["natural_messages"] += 1
            else:
                c.require(
                    all((v is None for k, v in record["meta"].items() if k != "gate")),
                    "Source fine annotation present",
                )
                checks["correction_record_transforms"] += 1
                if mode == "masked5":
                    c.require(
                        renderer.corrected_gold(
                            expected["example_groups"][0][1][-1]["meta"]
                        )
                        == "",
                        "Masked label still rendered",
                    )
                    checks["masked_all_annotations_unknown"] += 1
        else:
            mode = p["condition"]
            expected = expected_context(base["context"], mode)
            c.require(
                p["source_id"] is None and p["selected_correction_cosine"] is None,
                "Reference acquired correction",
            )
            if mode == "reference5":
                c.require(
                    p["messages"] == base["messages"], "Reference message reuse failed"
                )
                checks["reference_messages"] += 1
        c.require(
            p["context"] == expected,
            "Independent context reconstruction failed: " + p["condition"],
        )
        c.require(
            expected["definitions"] == base["context"]["definitions"]
            and expected["guidelines"] == base["context"]["guidelines"],
            "Independent abstract mismatch",
        )
        n_examples = sum((len(hs) for _, hs in expected["example_groups"]))
        c.require(
            n_examples == {"reference4": 4, "extended6": 6}.get(mode, 5),
            "Example count wrong",
        )
        messages = renderer.render_messages(r["text"], "en", expected, "corrected")
        c.require(
            p["messages"] == messages and messages[0] == base["messages"][0],
            "Independent prompt rendering mismatch",
        )
        c.require(
            p["messages_sha256"] == c.sig(messages)
            and p["context_sha256"] == c.sig(expected),
            "Prompt digest mismatch",
        )
        n = tok_util.token_count(tok, messages)
        c.require(n == p["reference_input_tokens"], "Token count mismatch")
        tokens.append(n)
        counts[mode] += 1
        if len(seen) % 5000 == 0:
            print(
                f"A15 independently checked {len(seen)}/{c.N_PROMPTS} prompts.",
                flush=True,
            )
    c.require(
        seen == {(r["key"], cond) for r in roster for cond in conditions},
        "Independent matrix incomplete",
    )
    c.require(len(seen) == c.N_PROMPTS, "Independent total mismatch")
    c.verify(inputs)
    c.verify(prepared["outputs"])
    path = c.BASE / "independent_verification.json"
    c.save(
        path,
        {
            "status": "all_A15_interventions_sources_labels_messages_and_tokens_checked",
            "created_utc": c.utc(),
            "prompts": len(seen),
            "counts": dict(counts),
            "checks": dict(checks),
            "token_min": min(tokens),
            "token_max": max(tokens),
            "prepared_sha256": c.sha(c.BASE / "prepared.json"),
            "new_model_calls": 0,
            "CUDA_used": False,
            "checker_sources": c.bindings(
                [c.ROOT / "analysis/feedback/reserved/validate.py"]
            ),
        },
    )
    c.save(
        c.BASE / "complete.json",
        {
            "status": "offline_preparation_independently_verified",
            "prepared_sha256": c.sha(c.BASE / "prepared.json"),
            "verification": c.rel(path),
            "verification_sha256": c.sha(path),
            "new_model_calls": 0,
        },
    )
    print("A15 all 49,950 prompt constructions independently verified.", flush=True)


def check_schedule(schedule, catalogs, b):

    a = b.a
    seen = set()
    owners = {}
    total = Counter()
    stage_ids = set()
    c.require(len(schedule["stages"]) == c.N_BATCHES, "Wrong number of batches")
    for stage in schedule["stages"]:
        study = stage["study"]
        info = b.STUDIES[study]
        cat = catalogs[study]
        c.require(stage["id"] not in stage_ids, "Duplicate stage")
        stage_ids.add(stage["id"])
        keys = stage["input_keys"]
        c.require(
            len(keys) == len(set(keys)) and len(keys) in (42, 43), "Stage roster wrong"
        )
        c.require(
            len(stage["jobs"]) == stage["items"] == len(keys) * c.N_CONDITIONS * 3
            and stage["ordinary_calls"] == stage["items"] * 3,
            "Stage budget wrong",
        )
        local = Counter()
        for job in stage["jobs"]:
            ident = (study, job["key"], job["condition"], job["repeat"])
            c.require(
                ident not in seen
                and job["study"] == study
                and (job["repeat"] in (0, 1, 2)),
                "Job duplicated or malformed",
            )
            seen.add(ident)
            c.require(
                job["key"] in keys
                and owners.setdefault((study, job["key"]), stage["id"]) == stage["id"],
                "Input comparisons split",
            )
            entry = cat[job["key"], job["condition"]]
            c.require(
                all(
                    (
                        job[k] == entry[k]
                        for k in (
                            "messages_sha256",
                            "schema_path",
                            "schema_sha256",
                            "population",
                        )
                    )
                ),
                "Job payload attribution changed",
            )
            c.require(
                a.signature(entry["messages"]) == job["messages_sha256"],
                "Schedule message hash wrong",
            )
            wire = a.body(entry["messages"])
            c.require(
                [m["content"] for m in wire["messages"]]
                == [m["text"] for m in entry["messages"]],
                "Wire conversion changed",
            )
            local[job["key"]] += 1
            total[study] += 1
        c.require(
            set(local) == set(keys) and set(local.values()) == {222},
            "Incomplete per-input comparison",
        )
    expected = {
        (st, k, cond, rep)
        for st, cat in catalogs.items()
        for k, cond in cat
        for rep in range(3)
    }
    c.require(
        seen == expected and len(seen) == c.N_PREDICTIONS, "Complete queue differs"
    )
    c.require(
        total == Counter({k: v["predictions"] for k, v in b.STUDIES.items()}),
        "Study budget differs",
    )
    return {
        "predictions": len(seen),
        "ordinary_calls": len(seen) * 3,
        "batches": len(stage_ids),
        "conditions": c.N_CONDITIONS,
    }


def rehearse(b, schedule, catalogs, plan_sha):

    root = b.BASE / "offline_check" / f"rehearsal_{time.time_ns()}"
    selected = []
    conditions = sorted({cond for cat in catalogs.values() for _, cond in cat})
    covered = set()
    for i, stage in enumerate(schedule["stages"]):
        cat = catalogs[stage["study"]]
        populations = {}
        for key in stage["input_keys"]:
            population = cat[key, conditions[0]]["population"]
            populations.setdefault(population, key)
        c.require(
            set(populations) == {"benign", "hateful_types"},
            "Two rehearsal populations absent",
        )
        selected_conditions = {
            cond for n, cond in enumerate(conditions) if n % c.N_BATCHES == i
        }
        keys = set(populations.values())
        jobs = [
            j
            for j in stage["jobs"]
            if j["key"] in keys and j["condition"] in selected_conditions
        ]
        covered.update((j["condition"] for j in jobs))
        selected.append(
            {
                **stage,
                "input_keys": sorted(keys),
                "jobs": jobs,
                "items": len(jobs),
                "ordinary_calls": len(jobs) * 3,
            }
        )
    c.require(covered == set(conditions), "Rehearsal misses conditions")
    transport = SyntheticTransport()
    for stage in selected:
        b.execute_stage(
            stage,
            catalogs[stage["study"]],
            root / stage["study"],
            plan_sha,
            mode="mock",
            client_factory=lambda: transport,
            sleep=lambda _: None,
        )
        b.audit.audit_stage(
            stage,
            catalogs[stage["study"]],
            root / stage["study"],
            plan_sha,
            mode="mock",
        )
    reduced = {**schedule, "stages": selected}
    first = b.audit.finalize_queue(reduced, root, plan_sha, mode="mock")
    before = transport.count
    for stage in selected:
        b.execute_stage(
            stage,
            catalogs[stage["study"]],
            root / stage["study"],
            plan_sha,
            mode="mock",
            client_factory=c.forbidden,
        )
        b.verify_review(stage, root / stage["study"], plan_sha, "mock")
    second = b.audit.finalize_queue(reduced, root, plan_sha, mode="mock")
    expected = sum((s["items"] for s in selected))
    c.require(
        first == second
        and transport.count == before == expected * 3
        and (first["counts"]["predictions"] == expected),
        "Rehearsal/restart mismatch",
    )
    return {
        "root": b.rel(root),
        "completion_sha256": c.sha(root / "complete.json"),
        "predictions": expected,
        "mock_replies": before,
        "conditions": len(covered),
        "batches": len(selected),
        "new_replies_on_restart": 0,
        "actual_API_calls": 0,
    }


if __name__ == "__main__":
    with c.offline():
        check_prompts()
