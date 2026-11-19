"""Common execution."""

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import socket
import time
from types import SimpleNamespace
from unittest.mock import patch
from analysis.common import study as c
from analysis.common import api as a
from analysis.common import batch as core
from analysis.common import schema as isolated
from analysis.common import vote_api as template
from analysis.common import cached_retrieval as retrieval

SIZES = {"matched": (941, 13, 36699), "broad": (21627, 3, 194643)}
COPIED = [
    "rel",
    "save",
    "forbidden",
    "bind",
    "verify",
    "test_receipt",
    "engines",
    "identity",
    "item_dir",
    "scan_jobs",
    "availability",
    "checked_root",
    "execute_stage",
    "verify_review",
    "run_sequence",
]
AUDITOR = c.ROOT / "analysis/common/vote_audit.py"
SOURCE_FILES = list(
    dict.fromkeys(
        [
            *core.SOURCE_FILES,
            "analysis/common/vote_api.py",
            "analysis/common/vote_audit.py",
            "tests/feedback/summary/test_balanced.py",
            "analysis/common/scoring.py",
            "analysis/common/schema.py",
            "analysis/common/prompt_archive.py",
            "analysis/common/study.py",
            "analysis/common/execution.py",
            "analysis/common/result_reader.py",
            "tests/common/test_execution.py",
            "analysis/common/cached_retrieval.py",
            "analysis/common/checkpoints.py",
            "training/original/validate.py",
            "scripts/prepare_comparison.ps1",
            "analysis/classification/validate_prompts.py",
            "analysis/classification/validate_execution.py",
            "scripts/run_comparison.ps1",
        ]
    )
)


def load_catalog(study):
    check = c.BASE / "prompt_checks" / f"{study}.json"
    checked = c.read(check)
    c.require(
        checked["status"] == "all_prompts_independently_verified",
        "Prompt check required",
    )
    c.verify(checked["inputs"])
    c.verify(checked["source_hashes"])
    paths = (
        [c.PREP / "matched_prompts.jsonl", c.PREP / "coverage_matched_prompts.jsonl"]
        if study == "matched"
        else [
            retrieval.BASE / "small/prompts.jsonl",
            retrieval.BASE / "full/prompts.jsonl",
            retrieval.BASE / "zero_shot.jsonl",
        ]
    )
    cat = {}
    schema = c.ROOT / "config/taxonomy.yaml"
    schema_sha = c.sha(schema)
    for path in paths:
        for r in c.rows(path):
            ident = (r["key"], r["condition"])
            c.require(
                ident not in cat and c.signature(r["messages"]) == r["messages_sha256"],
                "Duplicate/changed prompt",
            )
            population = r.get("population") or r["scopes"][0]["subset"]
            cat[ident] = {
                "messages": r["messages"],
                "messages_sha256": r["messages_sha256"],
                "schema_path": c.rel(schema),
                "schema_sha256": schema_sha,
                "population": population,
                "entry_sha256": c.signature(r),
            }
    n, settings, _ = SIZES[study]
    keys = {k for k, _ in cat}
    conds = {cond for _, cond in cat}
    c.require(
        len(keys) == n
        and len(conds) == settings
        and (set(cat) == {(k, s) for k in keys for s in conds}),
        "Incomplete study matrix",
    )
    return cat


def make_schedule(study, cat):
    conditions = sorted({cond for _, cond in cat})
    by_population = defaultdict(list)
    for key, cond in cat:
        if cond == conditions[0]:
            by_population[cat[key, cond]["population"]].append(key)
    width = 12000 // (len(conditions) * 9)
    stages = []
    for population, raw in sorted(by_population.items()):
        keys = sorted(
            raw,
            key=lambda k: hashlib.sha256(
                ("20261008-later|" + study + "|" + k).encode()
            ).hexdigest(),
        )
        for start in range(0, len(keys), width):
            group = keys[start : start + width]
            jobs = []
            for ordinal, key in enumerate(group, start):
                for block in range(3):
                    repeat = (ordinal + block) % 3
                    for offset in range(len(conditions)):
                        cond = conditions[(ordinal + repeat + offset) % len(conditions)]
                        entry = cat[key, cond]
                        jobs.append(
                            {
                                "study": study,
                                "key": key,
                                "condition": cond,
                                "repeat": repeat,
                                **{
                                    k: entry[k]
                                    for k in [
                                        "messages_sha256",
                                        "schema_path",
                                        "schema_sha256",
                                        "population",
                                    ]
                                },
                            }
                        )
            stages.append(
                {
                    "id": f"{study}_{len(stages) + 1:03d}",
                    "study": study,
                    "population": population,
                    "input_keys": group,
                    "items": len(jobs),
                    "ordinary_calls": len(jobs) * 3,
                    "jobs": jobs,
                }
            )
    return {
        "ordering": "Stable hashed keys within population; rotate condition and repeat order. All comparisons for one key remain together. At most 12000 ordinary calls per automatic audit batch.",
        "stages": stages,
    }


def queue(study):
    c.require(study in SIZES, "Unknown study")
    base = c.BASE / "execution" / study
    ns = {
        **vars(template),
        "BASE": base,
        "PLAN": base / "plan.json",
        "WORKERS": 8,
        "SOURCE_FILES": SOURCE_FILES,
        "STUDIES": {
            study: {
                "inputs": SIZES[study][0],
                "conditions": SIZES[study][1],
                "predictions": SIZES[study][2],
            }
        },
    }
    tree = ast.parse(Path(template.__file__).read_text(encoding="utf8"))
    selected = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in COPIED
    ]
    c.require(
        {n.name for n in selected} == set(COPIED),
        "Execution template functions missing",
    )
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), template.__file__, "exec"),
        ns,
    )
    b = SimpleNamespace(
        **{
            k: ns[k]
            for k in [
                "ROOT",
                "BASE",
                "PLAN",
                "WORKERS",
                "SOURCE_FILES",
                "STUDIES",
                "a",
                "core",
                "isolated",
                *COPIED,
            ]
        }
    )
    audit_tree = ast.parse(AUDITOR.read_text(encoding="utf8"))
    audit_tree.body = [
        n
        for n in audit_tree.body
        if not (
            isinstance(n, ast.ImportFrom) and any((x.asname == "b" for x in n.names))
        )
    ]
    audit_ns = {
        "b": b,
        "__name__": __name__ + "_audit_" + study,
        "__package__": __package__,
    }
    exec(compile(audit_tree, str(AUDITOR), "exec"), audit_ns)
    audit = SimpleNamespace(
        **{
            n.name: audit_ns[n.name]
            for n in audit_tree.body
            if isinstance(n, ast.FunctionDef)
        }
    )
    b.audit = audit
    b.validate = lambda ready=False: validate(study, ready)
    b.run_all = lambda: run_all(study)
    return b


def inputs(study):
    result = c.scientific_inputs()
    check = c.BASE / "prompt_checks" / f"{study}.json"
    result[c.rel(check)] = c.sha(check)
    result.update(c.read(check)["inputs"])
    if study == "broad":
        done = c.read(retrieval.BASE / "complete.json")
        result.update(done["outputs"])
        result[c.rel(retrieval.BASE / "complete.json")] = c.sha(
            retrieval.BASE / "complete.json"
        )
        result[c.rel(retrieval.PLAN)] = c.sha(retrieval.PLAN)
    return result


def prepare(study):
    b = queue(study)
    if b.PLAN.exists():
        validate(study)
        print("Existing execution plan verified.")
        return
    c.require(not (b.BASE / "live").exists(), "Preserve live execution")
    tested = a.read(b.test_receipt())
    c.require(
        tested["passed"] and tested["environment"] == core.environment(),
        "Current passing tests required",
    )
    c.verify(tested["source_hashes"])
    cat = load_catalog(study)
    schedule = make_schedule(study, cat)
    count = sum((s["items"] for s in schedule["stages"]))
    c.require(count == SIZES[study][2], "Wrong prediction budget")
    b.save(b.BASE / "schedule.json", schedule)
    bindings = inputs(study)
    bindings[c.rel(b.BASE / "schedule.json")] = c.sha(b.BASE / "schedule.json")
    bindings[c.rel(b.test_receipt())] = c.sha(b.test_receipt())
    b.save(
        b.PLAN,
        {
            "created_utc": c.utc(),
            "study": study,
            "status": "frozen_offline_rehearsal_required",
            "input_hashes": bindings,
            "source_hashes": {n: c.sha(c.ROOT / n) for n in SOURCE_FILES},
            "environment": core.environment(),
            "api_contract": a.CONTRACT,
            "workers": 8,
            "budget": {
                "inputs": SIZES[study][0],
                "settings": SIZES[study][1],
                "predictions": count,
                "ordinary_calls": count * 3,
                "automatic_audit_batches": len(schedule["stages"]),
            },
            "derivation": {
                "template": c.rel(Path(template.__file__)),
                "unchanged_execution_functions": COPIED,
                "auditor": c.rel(AUDITOR),
                "auditor_change": "Only locally bound queue namespace; function AST unchanged.",
            },
            "scope": "New decoder phase only; no old prediction reuse, model substitution, retraining, retrieval or accuracy-based stopping during API execution.",
            "readiness": "Requires passing synthetic rehearsal and explicit live mode. Pinned service availability is checked at each batch; unavailable model stops without substitution.",
        },
    )
    print(study + " execution frozen; synthetic rehearsal required.")


def validate(study, ready=False):
    b = queue(study)
    p = c.read(b.PLAN)
    c.require(
        p["study"] == study
        and p["environment"] == core.environment()
        and (p["api_contract"] == a.CONTRACT)
        and (p["workers"] == 8),
        "Execution contract/environment changed",
    )
    c.verify(p["input_hashes"])
    c.verify(p["source_hashes"])
    if ready:
        r = c.read(b.BASE / "ready.json")
        c.require(
            r["status"] == "offline_verified_ready_for_user"
            and r["plan_sha256"] == c.sha(b.PLAN),
            "Offline rehearsal required",
        )
        c.require(
            c.sha(c.ROOT / r["evidence"]) == r["evidence_sha256"],
            "Readiness evidence changed",
        )
    return p


class SyntheticTransport:

    def __init__(self):
        import threading, uuid

        self.lock = threading.Lock()
        self.count = 0
        self.prefix = uuid.uuid4().hex

    def __call__(self, payload):
        with self.lock:
            self.count += 1
            number = self.count
        value = {
            "reasoning": "SYNTHETIC OFFLINE FIXTURE",
            "hate": False,
            "target_group": [],
            "hate_type": [],
            "severity": None,
            "legal": [],
        }
        reply = {
            "id": f"offline-{self.prefix}-{number}",
            "model": a.MODEL,
            "choices": [
                {"message": {"content": json.dumps(value)}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        return a.WireReply(
            200, json.dumps(reply).encode(), {"Content-Type": "application/json"}
        )


def check_schedule(study, schedule, cat):
    seen = set()
    owners = {}
    expected = {(k, cond, r) for k, cond in cat for r in range(3)}
    for stage in schedule["stages"]:
        c.require(
            stage["items"] == len(stage["jobs"])
            and stage["ordinary_calls"] == stage["items"] * 3 <= 12000,
            "Wrong stage budget",
        )
        c.require(
            set(stage["input_keys"]) == {j["key"] for j in stage["jobs"]},
            "Stage key roster differs",
        )
        for j in stage["jobs"]:
            ident = (j["key"], j["condition"], j["repeat"])
            c.require(
                ident not in seen and ident in expected,
                "Duplicate/unexpected scheduled job",
            )
            seen.add(ident)
            c.require(
                owners.setdefault(j["key"], stage["id"]) == stage["id"],
                "Input comparisons split across stages",
            )
            e = cat[j["key"], j["condition"]]
            c.require(
                all(
                    (
                        j[k] == e[k]
                        for k in [
                            "messages_sha256",
                            "schema_path",
                            "schema_sha256",
                            "population",
                        ]
                    )
                ),
                "Job metadata changed",
            )
            c.require(
                j["study"] == study
                and a.signature(e["messages"]) == j["messages_sha256"],
                "Study/wire identity changed",
            )
            a.body(e["messages"])
    c.require(
        seen == expected and len(seen) == SIZES[study][2], "Incomplete full matrix"
    )
    return {
        "predictions": len(seen),
        "ordinary_calls": len(seen) * 3,
        "batches": len(schedule["stages"]),
    }


def rehearse(study):
    b = queue(study)
    p = validate(study)
    plan_sha = c.sha(b.PLAN)
    if (b.BASE / "ready.json").exists():
        validate(study, True)
        return
    cat = load_catalog(study)
    schedule = c.read(b.BASE / "schedule.json")
    matrix = check_schedule(study, schedule, cat)
    selected = []
    populations = set()
    for stage in schedule["stages"]:
        if stage["population"] in populations:
            continue
        populations.add(stage["population"])
        keys = set(stage["input_keys"][:1])
        jobs = [j for j in stage["jobs"] if j["key"] in keys]
        selected.append(
            {
                **stage,
                "input_keys": sorted(keys),
                "jobs": jobs,
                "items": len(jobs),
                "ordinary_calls": len(jobs) * 3,
            }
        )
    root = b.BASE / "offline_check" / f"rehearsal_{time.time_ns()}"
    transport = SyntheticTransport()

    def run(factory):
        for stage in selected:
            b.execute_stage(
                stage,
                cat,
                root / study,
                plan_sha,
                mode="mock",
                client_factory=factory,
                sleep=lambda _: None,
            )
            b.audit.audit_stage(stage, cat, root / study, plan_sha, mode="mock")
        return b.audit.finalize_queue({"stages": selected}, root, plan_sha, mode="mock")

    first = run(lambda: transport)
    before = transport.count
    second = run(b.forbidden)
    c.require(
        first == second
        and transport.count == before == sum((s["ordinary_calls"] for s in selected)),
        "Restart sent new requests",
    )
    evidence = b.BASE / "development" / f"readiness_{time.time_ns()}.json"
    b.save(
        evidence,
        {
            "status": "synthetic_matrix_resume_audit_pass",
            "plan_sha256": plan_sha,
            "matrix": matrix,
            "fixture_root": c.rel(root),
            "fixture_completion_sha256": c.sha(root / "complete.json"),
            "synthetic_predictions": sum((s["items"] for s in selected)),
            "synthetic_replies": before,
            "new_replies_on_restart": 0,
            "actual_Qwen_calls": 0,
            "research_predictions_generated": 0,
            "populations": sorted(populations),
        },
    )
    b.save(
        b.BASE / "ready.json",
        {
            "status": "offline_verified_ready_for_user",
            "plan_sha256": plan_sha,
            "evidence": c.rel(evidence),
            "evidence_sha256": c.sha(evidence),
        },
    )
    validate(study, True)
    print(study.upper() + " offline execution rehearsal passed; no live requests.")


def run_all(study):
    b = queue(study)
    with a.item_lock(a.storage_path(b.BASE / "combined_launcher.lock")):
        validate(study, True)
        plan_sha = c.sha(b.PLAN)
        cat = load_catalog(study)
        schedule = c.read(b.BASE / "schedule.json")
        for stage in schedule["stages"]:
            validate(study, True)
            b.execute_stage(
                stage,
                cat,
                b.BASE / "live" / study,
                plan_sha,
                mode="live",
                client_factory=core.live_client,
            )
            b.audit.audit_stage(
                stage, cat, b.BASE / "live" / study, plan_sha, mode="live"
            )
        b.audit.finalize_queue(schedule, b.BASE / "live", plan_sha, mode="live")
    print(
        study.upper()
        + " complete with offline audits. Ask Astra to inspect before interpreting results."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["status", "prepare", "rehearse", "validate", "run"])
    p.add_argument("--study", choices=sorted(SIZES), required=True)
    args = p.parse_args()
    if args.mode == "run":
        run_all(args.study)
        return
    with c.offline():
        if args.mode == "status":
            b = queue(args.study)
            print(
                json.dumps(
                    {
                        "study": args.study,
                        "plan_exists": b.PLAN.exists(),
                        "offline_ready": (b.BASE / "ready.json").exists(),
                        "budget_ordinary_calls": SIZES[args.study][2] * 3,
                        "live_mode_requested": False,
                    }
                )
            )
        elif args.mode == "prepare":
            prepare(args.study)
        elif args.mode == "rehearse":
            rehearse(args.study)
        else:
            validate(args.study, True)
            print("Validated offline; no service access.")


if __name__ == "__main__":
    main()
