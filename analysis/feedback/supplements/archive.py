from src.hsrag.paths import PROJECT_ROOT
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
from unittest.mock import patch

ROOT = PROJECT_ROOT
EXEC = ROOT / "experiments/astra/sq3_additions_20261004_v1/execution"
DEST = ROOT / "experiments/astra/sq3_additions_results_20261006_v1"
REVIEW = DEST / "review"
PLAN_SHA = "f231e8124c030efe6bb92be8a304e83f0ab06f6193ec3035f013438cdff99965"
LABELS = [
    "explicit",
    "incitement",
    "inferiority",
    "irony",
    "stereotyping",
    "threatening",
    "grievance",
]
FIELDS = ["hate", "hate_type", "target_group", "severity", "legal"]
SEEDS = [202609281, 202609282, 202609283]
STUDIES = {
    s: EXEC.parent / name
    for s, name in [
        ("factorial", "factorial"),
        ("human_meaning", "human_meaning_v2"),
        ("benign_selection", "benign_selection"),
    ]
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def io_path(p):
    p = Path(p).absolute()
    s = str(p)
    return (
        Path("\\\\?\\" + s) if os.name == "nt" and (not s.startswith("\\\\?\\")) else p
    )


def read(p):
    return json.loads(io_path(p).read_text(encoding="utf8"))


def rows(p):
    with io_path(p).open(encoding="utf8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sha(p):
    return hashlib.sha256(io_path(p).read_bytes()).hexdigest()


def canonical(v):
    return (
        json.dumps(v, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf8")


def sig(v):
    return hashlib.sha256(canonical(v)).hexdigest()


def rel(p):
    return Path(str(p).removeprefix("\\\\?\\")).resolve().relative_to(ROOT).as_posix()


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def save(p, v):
    p = Path(p)
    require(p.resolve().is_relative_to(DEST), "Write outside analysis directory")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf8", newline="\n") as f:
        json.dump(v, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def verify(bindings):
    for n, h in bindings.items():
        require(sha(ROOT / n) == h, "Changed input: " + n)


def forbid(*args, **kwargs):
    raise AssertionError("Offline analysis: network forbidden")


def vote(predictions):
    valid = [v for v in predictions if v is not None]
    if not valid:
        return (None, 0, False)
    n = len(valid)
    yes = sum((v["hate"] for v in valid))
    out = {"hate": 2 * yes > n}
    for field in ["hate_type", "target_group", "legal"]:
        c = Counter((l for v in valid for l in set(v[field])))
        out[field] = sorted((l for l, count in c.items() if 2 * count > n))
    order = ["low", "medium", "high"]
    sev = sorted(
        (order.index(v["severity"]) for v in valid if v["severity"] is not None)
    )
    out["severity"] = order[sev[(len(sev) - 1) // 2]] if sev else None
    if not out["hate"]:
        out.update(hate_type=[], target_group=[], severity=None)
    return (out, n, 2 * yes == n)


def plan():
    require(sha(EXEC / "plan.json") == PLAN_SHA, "Execution plan changed")
    inputs = {
        rel(EXEC / n): sha(EXEC / n)
        for n in [
            "plan.json",
            "schedule.json",
            "live/complete.json",
            "recovery_20261005_v1/manifest.json",
            "recovery_20261005_v1/ready.json",
        ]
    }
    for study, base in STUDIES.items():
        for name in [
            "scientific_plan_v1.json",
            "STUDY_PROTOCOL.md",
            "roster.jsonl",
            "prompts.jsonl",
            "summary.json",
            "complete.json",
            "prepared.json",
            "preparation_plan.json",
            "availability.jsonl",
            "source_review_cards.jsonl",
        ]:
            path = base / name
            if path.exists():
                inputs[rel(path)] = sha(path)
    inputs["analysis/feedback/balanced/selection_pools.py"] = sha(
        ROOT / "analysis/feedback/balanced/selection_pools.py"
    )
    save(
        DEST / "analysis_plan.json",
        {
            "status": "analysis_details_fixed_before_new_scores",
            "created_utc": utc(),
            "inputs": inputs,
            "factorial_primary": "Immutable protocol: target-label F1 within each of seven categories on hateful pairs; mean three repeats then equal seven-category average. All 12 outputs valid per pair. Marginal E and D effects with 97.5% intervals; 10000 paired whole-text-overlap-cluster draws, PCG64 seed 202610041. All category pairs and repeats stay together. Zero-denominator F1=0; no retained positive support in any category makes primary unestimable.",
            "human_primary": "Immutable protocol: 350 original hateful texts, seven-type macro F1 mean of three pools and three repeats. All 18 human/plain outputs valid per text. Human minus plain with 95% interval, 10000 text-cluster draws, PCG64 seed 202610042.",
            "sensitivities": "Both studies: common all-three-valid-votes mask; one deterministic original-text representative per overlap cluster, preserving all factorial pairs of that selected text; missing-output worst/best bounds on full original rosters. Use original primary seeds for sensitivity bootstrap.",
            "representative_hash": "Minimum SHA256 of 20261006-additions-representative|original_key within each cluster, selected before masks; retain every pair for that text.",
            "factorial_supporting": "Joint restoration, simple E/D effects, interaction, each category precision/recall/support/TP/FP/FN, other-six-label macro F1 on the category subset, benign FP rates. Supporting intervals are 95%, not new primaries. Bootstrap seed 202610061 for benign; primary draws for other factorial diagnostics.",
            "human_supporting": "All-policy comparisons use a 30-output common mask per population. Reference remains its three real repeats, never three independent pools. Paired 95% intervals; seeds 202610062 hateful and 202610063 benign (source-specific seed + source sorted index + 1). Individual label diagnostics, false-negative rates, all pool/repeat values and costs. Conditional availability, old-assistant intersection and flagged-source strata are fixed from source cards; each pool uses its own common all-nine-output mask and equally averages pool-specific scores. No causal authorship inference.",
            "secondary_coverage_seeds": {
                "available": 202610064,
                "old_assistant_and_human": 202610065,
                "flagged": 202610066,
                "fallback": 202610067,
            },
            "missing_bounds": "For fine labels, every missing final output gets complement-of-gold (worst) or gold (best), then F1 is computed per repeat/pool/category before averaging. Linear contrast bounds select opposite extremes according to coefficient sign. Gate error bounds put missing at error/non-error. These are conservative bounds, not filled-in research predictions.",
            "benign_selection": "Audit all 3072 baseline predictions; apply unchanged classify_candidates/select_pools functions and verify independently. Any null among three repeats excludes source; majority gate sets correct/error. Choose largest budget 40,20 satisfying all four polarity/status buckets. Same fixed candidate universe; no more sources or altered criteria. Publish actual pools and gate-only records if supply passes; no update model run.",
            "review": "Recheck completed raw-audit chain, rehash every result/receipt, independently parse every raw reply and field vote, confirm all retry recovery and the eight-call amendment. Original finalizer already hashed full raw files; retain that provenance and its limits.",
            "interpretation": "Previously exposed DEV; additions have separate study-specific inference families. Original primaries remain closed. Conditional source strata and supporting comparisons are exploratory. Two-human annotation audit remains pending. No new classifier calls.",
        },
    )
    print("Analysis details frozen before new scores.", flush=True)


def review():
    from analysis.common import schema as isolated
    from analysis.common import api as api

    require(not (REVIEW / "complete.json").exists(), "Preserve completed review")
    frozen = read(DEST / "analysis_plan.json")
    verify(frozen["inputs"])
    from analysis.feedback.supplements import recovery_support as recovery

    amendment = recovery.validate()
    allowed_extra = {
        (
            e["job"]["study"],
            e["job"]["key"],
            e["job"]["condition"],
            e["job"]["repeat"],
            e["vote"],
            e["turn"],
        )
        for e in amendment["calls"]
    }
    ep = read(EXEC / "plan.json")
    verify(ep["input_hashes"])
    verify(ep["source_hashes"])
    completed = read(EXEC / "live/complete.json")
    require(
        completed["status"] == "all_studies_executed_and_audited"
        and completed["plan_sha256"] == PLAN_SHA,
        "Incomplete run",
    )
    verify(completed["review_hashes"])
    schedule = read(EXEC / "schedule.json")
    stages = {}
    hashes = {}
    remote_ids = set()
    totals = defaultdict(Counter)
    inputs = {
        **frozen["inputs"],
        rel(DEST / "analysis_plan.json"): sha(DEST / "analysis_plan.json"),
        rel(__file__): sha(__file__),
    }
    for st in schedule["stages"]:
        root = EXEC / "live" / st["study"]
        folder = root / "stages" / st["id"]
        rv = read(folder / "review.json")
        require(
            sha(folder / "complete.json") == rv["complete_sha256"],
            "Stage completion changed",
        )
        for n, h in rv["artifacts"].items():
            require(sha(folder / n) == h, "Audit artifact changed")
        c = read(folder / "complete.json")
        report = read(folder / rv["report"])
        manifest = read(folder / rv["raw_hashes"])
        ids = read(folder / rv["response_ids"])
        require(
            report["counts"]["predictions"] == st["items"]
            and report["stage_complete_sha256"] == sha(folder / "complete.json"),
            "Stage audit identity differs",
        )
        require(
            len(ids) == len(set(ids)) and (not remote_ids.intersection(ids)),
            "Response IDs repeated",
        )
        remote_ids.update(ids)
        for n, h in report["attempt_hashes"].items():
            require(sha(root / n) == h, "Attempt record changed")
        require(
            sha(ROOT / c["final_attempt"] / "summary.json") == c["summary_sha256"],
            "Final summary changed",
        )
        for n, h in c["item_receipt_hashes"].items():
            require(manifest[n] == h, "Receipt/manifest mismatch")
        hashes[st["id"]] = manifest
        stages[st["id"]] = {"stage": st, "completion": c, "report": report}
        for k in ["counts", "accounting", "observed_usage"]:
            totals[k].update(report[k])
        inputs[rel(folder / "review.json")] = sha(folder / "review.json")
        for n, h in rv["artifacts"].items():
            inputs[rel(folder / n)] = h
    require(
        len(remote_ids) == completed["distinct_remote_ids"], "Global remote IDs differ"
    )
    for k in totals:
        require(dict(totals[k]) == completed[k], "Final totals differ: " + k)
    count = Counter()
    duplicates = Counter()
    failures = []
    retries = []
    costs = defaultdict(Counter)
    stage_counts = defaultdict(Counter)
    exported = []
    backends = {}
    rosters = {
        s: {r["key"]: r for r in rows(d / "roster.jsonl")} for s, d in STUDIES.items()
    }
    promptmeta = {}
    for study, base in STUDIES.items():
        for p in rows(base / "prompts.jsonl"):
            require(
                sig(p["messages"]) == p["messages_sha256"], "Prompt signature differs"
            )
            promptmeta[study, p["key"], p["condition"]] = {
                k: p[k] for k in ["messages_sha256", "context_sha256"]
            }
    for st in schedule["stages"]:
        manifest = hashes[st["id"]]
        root = EXEC / "live" / st["study"]
        cc = stage_counts[st["id"]]
        for ordinal, j in enumerate(st["jobs"], 1):
            ident = {
                "experiment_sha256": PLAN_SHA,
                "condition": j["study"] + "::" + j["condition"],
                "repeat": j["repeat"],
                "key": j["key"],
            }
            folder = root / "items" / sig(ident)
            result_path = folder / "result.json"
            receipt_path = folder / "complete.json"
            require(
                sha(receipt_path)
                == manifest[
                    rel(receipt_path).split("/live/" + st["study"] + "/", 1)[1]
                ],
                "Item receipt changed",
            )
            receipt = read(receipt_path)
            rn = result_path.relative_to(root).as_posix()
            require(
                sha(result_path) == manifest[rn] == receipt["hashes"][rn],
                "Result changed",
            )
            v = read(result_path)
            pm = promptmeta[j["study"], j["key"], j["condition"]]
            require(
                v["identity"] == ident
                and v["messages_sha256"]
                == j["messages_sha256"]
                == pm["messages_sha256"]
                and (v["execution_mode"] == "live"),
                "Wrong result attribution",
            )
            require(
                v["active_model"]
                == v["requested_model"]
                == "Qwen/Qwen3-VL-32B-Instruct"
                and v["n_runs"] == 3,
                "Model or vote contract differs",
            )
            if j["schema_path"] not in backends:
                require(
                    sha(ROOT / j["schema_path"]) == j["schema_sha256"], "Schema changed"
                )
                backends[j["schema_path"]] = isolated.adapter(ROOT / j["schema_path"])
            parser = backends[j["schema_path"]].bundle.parser
            parsed_votes = []
            terminal = []
            for vi, raw_run in enumerate(v["raw_runs"]):
                parsed = []
                for ti, raw in enumerate(raw_run):
                    obj, norm, err = parser.parse_response(raw)
                    parsed.append(obj)
                    cc["raw_turns"] += 1
                    cc["normalisations"] += norm
                    if obj is None:
                        cc["invalid_raw_turns"] += 1
                    if ti == len(raw_run) - 1 and obj is None:
                        terminal.append({"vote": vi, "error": err})
                require(
                    1 <= len(parsed) <= 3 and all((o is None for o in parsed[:-1])),
                    "Repair after a success",
                )
                last = parsed[-1].model_dump() if parsed[-1] is not None else None
                require(
                    last == v["run_predictions"][vi],
                    "Saved vote differs from raw parsing",
                )
                if last is None:
                    require(len(parsed) == 3, "Invalid vote terminated early")
                    cc["parse_failures"] += 1
                else:
                    cc["gate_normalised"] += parsed[-1].gate_normalised
                    for field in ["hate_type", "target_group", "legal"]:
                        duplicates[field] += len(last[field]) != len(set(last[field]))
                parsed_votes.append(last)
            result, n, tie = vote(parsed_votes)
            require(
                n == v["n_valid"]
                and tie == v["uncertain"]
                and (
                    result
                    == ({k: v["result"][k] for k in FIELDS} if v["result"] else None)
                ),
                "Independent vote differs",
            )
            cc["predictions"] += 1
            cc["nulls"] += result is None
            cc["valid_votes_" + str(n)] += 1
            cc["repairs"] += v["repairs"]
            cc["gate_ties"] += tie
            if terminal:
                failures.append(
                    {
                        **j,
                        "terminal_errors": terminal,
                        "null": result is None,
                        "source": rel(result_path),
                    }
                )
            cost = Counter(v["accounting"])
            cost["vote_seconds"] = sum(v["latencies"])
            for ev in v["events"]:
                extra = (
                    j["study"],
                    j["key"],
                    j["condition"],
                    j["repeat"],
                    ev["vote"],
                    ev["turn"],
                ) in allowed_extra
                require(
                    0 <= ev["attempt"] <= (9 if extra else 4),
                    "Unapproved retry extension",
                )
                if ev["action"] == "ok":
                    for field in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                        cost[field] += ev["model_response_metadata"]["usage"][field]
                else:
                    call = (
                        folder
                        / "calls"
                        / f"vote_{ev['vote']}_turn_{ev['turn']}"
                        / f"attempt_{ev['attempt']}"
                    )
                    for name in ["intent.json", "outcome.json"]:
                        f = call / name
                        require(
                            sha(f) == manifest[f.relative_to(root).as_posix()],
                            "Retry record changed",
                        )
                    intent = read(call / "intent.json")
                    out = read(call / "outcome.json")
                    later = [
                        e
                        for e in v["events"]
                        if e["vote"] == ev["vote"]
                        and e["turn"] == ev["turn"]
                        and (e["attempt"] > ev["attempt"])
                    ]
                    require(
                        any((e["action"] == "ok" for e in later)), "Unrecovered retry"
                    )
                    retries.append(
                        {
                            **j,
                            "vote": ev["vote"],
                            "turn": ev["turn"],
                            "attempt": ev["attempt"],
                            "elapsed_s": ev["elapsed_s"],
                            "response_class": ev["response_class"],
                            "category": out.get("category"),
                            "started_utc": intent["started_utc"],
                            "finished_utc": out["finished_utc"],
                            "recovered": True,
                        }
                    )
            policy = (
                j["condition"].split("_s20260928")[0]
                if st["study"] == "human_meaning"
                else j["condition"]
            )
            group = st["study"] + "/" + policy
            costs[group].update(cost)
            costs[group]["predictions"] += 1
            costs[group]["nulls"] += result is None
            r = rosters[st["study"]][j["key"]]
            exported.append(
                {
                    **j,
                    "stage": st["id"],
                    "policy": policy,
                    "id": r["id"],
                    "target_label": r.get("target_label"),
                    "cluster": r["cluster"],
                    "gold": r["gold"],
                    "data_source": r["source"],
                    "result": result,
                    "n_valid": n,
                    "gate_tie": tie,
                    "run_predictions": [
                        {k: p[k] for k in FIELDS} if p else None for p in parsed_votes
                    ],
                    "context_sha256": pm["context_sha256"],
                    "costs": dict(cost),
                    "source": rel(result_path),
                    "source_sha256": manifest[rn],
                }
            )
            if ordinal % 1500 == 0 or ordinal == len(st["jobs"]):
                print(
                    f"{st['id']}: independently parsed/voted {ordinal}/{len(st['jobs'])}",
                    flush=True,
                )
        report = stages[st["id"]]["report"]
        for key, n in report["counts"].items():
            require(cc[key] == n, "Independent stage count mismatch: " + key)
        require(
            cc["raw_turns"] == report["accounting"]["http_success_with_text"]
            and cc["invalid_raw_turns"] == cc["repairs"] + cc["parse_failures"],
            "Raw counts differ",
        )
        count.update(cc)
    require(
        count["predictions"] == 22536
        and len(retries) == completed["accounting"]["retryable_failures"],
        "Export totals differ",
    )
    REVIEW.mkdir(parents=True, exist_ok=True)
    out = REVIEW / "predictions.jsonl"
    with out.open("x", encoding="utf8", newline="\n") as f:
        for r in sorted(
            exported, key=lambda x: (x["study"], x["condition"], x["repeat"], x["id"])
        ):
            f.write(
                json.dumps(r, sort_keys=True, ensure_ascii=False, allow_nan=False)
                + "\n"
            )
    summary = {
        "created_utc": utc(),
        "status": "all_saved_outputs_independently_reparsed_and_voted",
        "counts": dict(count),
        "stage_counts": {s: dict(v) for s, v in stage_counts.items()},
        "operational": completed,
        "stage_operational": {s: v["report"] for s, v in stages.items()},
        "within_vote_duplicate_label_fields": dict(duplicates),
        "by_policy_costs": {s: dict(v) for s, v in costs.items()},
        "terminal_errors": failures,
        "retry_events": retries,
        "null_jobs": [
            {k: v for k, v in r.items() if k not in ["run_predictions", "costs"]}
            for r in exported
            if r["result"] is None
        ],
        "review_scope": "All result and receipt hashes rechecked, all raw text reparsed and votes independently recomputed. Full raw intent/outcome/payload hash replay was already completed by the finalizer; audit manifests and retry records rechecked here.",
        "new_model_calls": 0,
    }
    save(REVIEW / "summary.json", summary)
    verify(inputs)
    save(
        REVIEW / "complete.json",
        {
            "status": summary["status"],
            "created_utc": utc(),
            "inputs": inputs,
            "outputs": {rel(p): sha(p) for p in [out, REVIEW / "summary.json"]},
            "new_model_calls": 0,
        },
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "counts": dict(count),
                "recovered_retries": len(retries),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["plan", "review"])
    args = p.parse_args()
    with patch.object(socket.socket, "connect", forbid), patch.object(
        socket, "create_connection", forbid
    ):
        {"plan": plan, "review": review}[args.mode]()
