from src.hsrag.paths import PROJECT_ROOT
import argparse
import base64
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import time
import traceback
from unittest.mock import patch

import numpy as np

ROOT = PROJECT_ROOT
STUDY = ROOT / "experiments/astra/sq3_full_kb_relevance_20260930_v1"
API = STUDY / "api_execution_v1"
REVIEW = STUDY / "review_v1"
RESULTS = STUDY / "results_v1"
SCIENCE = STUDY / "scientific_plan_v1.json"
LABELS = [
    "explicit",
    "incitement",
    "inferiority",
    "irony",
    "stereotyping",
    "threatening",
    "grievance",
]
POLICIES = [
    "natural",
    "relevant_full",
    "relevant_labels",
    "disjoint_full",
    "disjoint_labels",
    "reference",
]
SEEDS = [202609281, 202609282, 202609283]
FIELDS = ["hate", "target_group", "hate_type", "severity", "legal"]
SCIENCE_SHA = "5665da9d37f4694b767003059ea086e0c4f60c9c499574fcb8129e9f57e6a785"
API_SHA = "d4d34f90adf144131e3e59a13cb9c6cd351b7eefc97a093ee71613ee75a0babb"
ROSTER = (
    ROOT
    / "experiments/astra/sq3_full_kb_feedback_20260928_v1/feasibility_v1/roster.jsonl"
)


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf8"))


def rows(p):
    return [
        json.loads(x)
        for x in Path(p).read_text(encoding="utf8").splitlines()
        if x.strip()
    ]


def canonical(v):
    return (
        json.dumps(v, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf8")


def signature(v):
    return hashlib.sha256(canonical(v)).hexdigest()


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("xb") as f:
        f.write(canonical(value))


def forbid(*args, **kwargs):
    raise AssertionError("Network forbidden in offline relevance review")


def merge(target, source):
    for name, sha in source.items():
        require(
            name not in target or target[name] == sha, "Conflicting binding: " + name
        )
        target[name] = sha


def verify(bindings, remote=False):

    ids = Counter()
    usage = Counter()
    kinds = Counter()

    def one(pair):
        name, expected = pair
        raw = (ROOT / name).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected, "Changed file: " + name)
        if remote and name.endswith("/outcome.json"):
            outcome = json.loads(raw)
            require(
                outcome["kind"] != "transport_error" and outcome["status"] == 200,
                "Unexpected transport failure",
            )
            body = base64.b64decode(outcome["body_base64"], validate=True)
            require(
                hashlib.sha256(body).hexdigest() == outcome["body_sha256"],
                "Response body hash differs",
            )
            value = json.loads(body)
            require(
                value["model"] == "Qwen/Qwen3-VL-32B-Instruct"
                and value["choices"][0]["finish_reason"] == "stop",
                "Remote model/finish differs",
            )
            require(value.get("id"), "Missing remote response ID")
            u = {
                k: value["usage"][k]
                for k in ["prompt_tokens", "completion_tokens", "total_tokens"]
            }
            require(
                all(type(x) is int and x >= 0 for x in u.values())
                and u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"],
                "Usage arithmetic differs",
            )
            return value["id"], u
        return None

    items = list(bindings.items())
    with ThreadPoolExecutor(max_workers=4) as executor:
        for start in range(0, len(items), 1000):
            for result in executor.map(one, items[start : start + 1000]):
                if result:
                    ids[result[0]] += 1
                    usage.update(result[1])
                    kinds["remote_responses"] += 1
            end = min(start + 1000, len(items))
            if remote and (end % 20000 == 0 or end == len(items)):
                print(f"Hash-checked {end}/{len(items)} saved files", flush=True)
    require(
        not ids or max(ids.values()) == 1, "Duplicate remote response across batches"
    )
    return {
        "files": len(items),
        "remote_responses": sum(ids.values()),
        "distinct_remote_ids": len(ids),
        "usage": dict(usage),
    }


def vote(predictions):
    valid = [p for p in predictions if p is not None]
    if not valid:
        return None, 0, False
    n = len(valid)
    yes = sum(p["hate"] for p in valid)
    out = {"hate": yes * 2 > n}
    for field in ("target_group", "hate_type", "legal"):
        counts = Counter(label for p in valid for label in p[field])
        out[field] = sorted(k for k, v in counts.items() if v * 2 > n)
    ranks = sorted(
        ["low", "medium", "high"].index(p["severity"])
        for p in valid
        if p["severity"] is not None
    )
    out["severity"] = (
        ["low", "medium", "high"][ranks[(len(ranks) - 1) // 2]] if ranks else None
    )
    if not out["hate"]:
        out.update(target_group=[], hate_type=[], severity=None)
    return out, n, yes * 2 == n


def review():
    from src.hsrag.parse import parse_response, extract_json
    from analysis.feedback.relevance import run as b

    require(not (REVIEW / "complete.json").exists(), "Preserve completed review")
    b.validate()
    require(
        digest(SCIENCE) == SCIENCE_SHA and digest(API / "plan.json") == API_SHA,
        "Frozen plan changed",
    )
    science = read(SCIENCE)
    schedule = read(API / "schedule.json")
    inputs = {
        rel(__file__): digest(__file__),
        rel(SCIENCE): SCIENCE_SHA,
        rel(API / "plan.json"): API_SHA,
    }
    outputs = {}
    aggregate = defaultdict(Counter)
    stages = []
    for st in schedule["stages"]:
        rp = API / "stage_reviews" / f"{st['id']}.json"
        rec = read(rp)
        ep = ROOT / rec["evidence"]
        e = read(ep)
        cp = API / "live/stages" / st["id"] / "complete.json"
        require(
            rec["status"] == "audited"
            and rec["plan_sha256"] == API_SHA
            and digest(ep) == rec["evidence_sha256"]
            and digest(cp) == rec["complete_sha256"],
            "Audit receipt chain differs",
        )
        require(
            e["status"] == "completed_relevance_stage_verified"
            and e["semantic_replay_items"] == st["items"]
            and e["independent_field_vote_checks"] == st["items"]
            and e["completed_stage_restart_new_POSTs"]
            == e["actual_audit_API_calls"]
            == 0,
            "Stage audit incomplete",
        )
        merge(inputs, e["verified_input_hashes"])
        merge(outputs, e["verified_output_hashes"])
        for p in [rp, ep, cp]:
            inputs[rel(p)] = digest(p)
        fields = [
            "counts",
            "accounting",
            "observed_usage_sums",
            "http_statuses",
            "reported_models",
            "finish_reasons",
            "first_response_validation_errors",
            "terminal_invalid_vote_errors",
        ]
        for field in fields:
            aggregate[field].update(e[field])
        stages.append(
            {
                k: e[k]
                for k in [
                    "stage",
                    "start_utc",
                    "completed_utc",
                    "created_utc",
                    "wall_seconds_including_final_save",
                    "counts",
                    "accounting",
                    "latency",
                ]
            }
        )
    save(
        REVIEW / "plan.json",
        {
            "created_utc": utc(),
            "inputs": inputs,
            "raw_file_count": len(outputs),
            "scope": "All three completed automatic audits consolidated; all bound files rehashed, remote IDs checked across stages, raw answer strings reparsed and independent field voting.",
            "new_model_calls": 0,
        },
    )
    verify(inputs)
    checked = verify(outputs, True)
    require(
        checked["remote_responses"] == aggregate["accounting"]["http_success_with_text"]
        and checked["usage"] == dict(aggregate["observed_usage_sums"]),
        "Global accounting differs",
    )
    jobs = [j for st in schedule["stages"] for j in st["jobs"]]
    expected = {
        signature(
            {
                "experiment_sha256": API_SHA,
                "condition": j["condition"],
                "repeat": j["repeat"],
                "key": j["key"],
            }
        )
        for j in jobs
    }
    require(
        len(expected) == len(jobs) == 16800
        and {p.name for p in (API / "live/items").iterdir() if p.is_dir()} == expected,
        "Live inventory differs",
    )
    gold = {r["id"]: r for r in rows(ROSTER)}
    evaluated = science["rosters"]["evaluation"]
    require(
        len({gold[i]["cluster"] for i in evaluated}) == len(evaluated),
        "Non-singleton transfer clusters",
    )
    count = Counter()
    invalid_labels = Counter()
    terminal_labels = Counter()
    failure_rows = []
    exported = []
    cost = defaultdict(Counter)
    for ordinal, job in enumerate(jobs, 1):
        ident = {
            "experiment_sha256": API_SHA,
            "condition": job["condition"],
            "repeat": job["repeat"],
            "key": job["key"],
        }
        folder = API / "live/items" / signature(ident)
        p = folder / "result.json"
        v = read(p)
        require(
            v["identity"] == ident
            and v["messages_sha256"] == job["messages_sha256"]
            and v["execution_mode"] == "live",
            "Saved result identity differs",
        )
        require(
            v["active_model"] == "Qwen/Qwen3-VL-32B-Instruct" and v["n_runs"] == 3,
            "Result model/repeat count differs",
        )
        parsed_votes = []
        for vi, raw_run in enumerate(v["raw_runs"]):
            parsed = []
            for turn, raw in enumerate(raw_run):
                obj, norm, err = parse_response(raw)
                parsed.append(obj)
                count["raw_turns"] += 1
                count["normalisations"] += norm
                if obj is None:
                    count["invalid_raw_turns"] += 1
                    try:
                        loose = json.loads(extract_json(raw))
                        bad = set(loose.get("hate_type", [])) - set(LABELS)
                    except (ValueError, TypeError, AttributeError):
                        bad = set()
                    invalid_labels.update(bad)
                    if turn == len(raw_run) - 1:
                        terminal_labels.update(bad)
                        failure_rows.append(
                            {
                                "job": job,
                                "vote": vi,
                                "error": err,
                                "invalid_type_labels": sorted(bad),
                                "source": rel(p),
                            }
                        )
            require(
                1 <= len(parsed) <= 3 and all(x is None for x in parsed[:-1]),
                "Repair continued after success",
            )
            last = parsed[-1].model_dump() if parsed[-1] is not None else None
            require(
                last == v["run_predictions"][vi],
                "Raw reparsing differs from saved vote",
            )
            if last is None:
                require(len(parsed) == 3, "Premature invalid vote")
                count["terminal_invalid_votes"] += 1
            else:
                count["gate_normalised"] += parsed[-1].gate_normalised
            parsed_votes.append(last)
        result, n, tie = vote(parsed_votes)
        require(
            n == v["n_valid"]
            and tie == v["uncertain"]
            and result
            == (
                {k: v["result"][k] for k in FIELDS} if v["result"] is not None else None
            ),
            "Independent vote differs",
        )
        count["predictions"] += 1
        count["valid_votes_" + str(n)] += 1
        count["repairs"] += v["repairs"]
        count["gate_ties"] += tie
        policy = job["policy"]
        iid = job["key"].split("/", 1)[1]
        require(iid in evaluated, "Unexpected evaluation text")
        costs = Counter(v["accounting"])
        costs["vote_seconds"] = sum(v["latencies"])
        for event in v["events"]:
            if event["action"] == "ok":
                u = event["model_response_metadata"]["usage"]
                for field in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                    costs[field] += u[field]
        cost[policy].update(costs)
        cost[policy]["predictions"] += 1
        cost[policy]["nulls"] += result is None
        cost[policy]["invalid_votes"] += 3 - n
        exported.append(
            {
                **job,
                "id": iid,
                "cluster": gold[iid]["cluster"],
                "gold": gold[iid]["gold"],
                "result": result,
                "n_valid": n,
                "gate_tie": tie,
                "run_predictions": [
                    {k: q[k] for k in FIELDS} if q else None for q in parsed_votes
                ],
                "costs": dict(costs),
                "source": rel(p),
                "source_sha256": outputs[rel(p)],
            }
        )
        if ordinal % 2000 == 0 or ordinal == len(jobs):
            print(
                f"Reparsed and independently voted {ordinal}/{len(jobs)} predictions",
                flush=True,
            )
    require(
        count["raw_turns"] == checked["remote_responses"]
        and count["terminal_invalid_votes"] == aggregate["counts"]["parse_failures"]
        and count["repairs"] == aggregate["counts"]["repairs"],
        "Raw validation totals differ",
    )
    for key in [
        "valid_votes_0",
        "valid_votes_1",
        "valid_votes_2",
        "valid_votes_3",
        "gate_ties",
        "gate_normalised",
        "normalisations",
    ]:
        require(
            count[key] == aggregate["counts"][key],
            "Stage/independent count differs: " + key,
        )
    out = REVIEW / "predictions.jsonl"
    with out.open("x", encoding="utf8", newline="\n") as f:
        for row in sorted(
            exported, key=lambda x: (x["condition"], x["repeat"], x["id"])
        ):
            f.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )
    summary = {
        "created_utc": utc(),
        "status": "all_relevance_predictions_independently_reviewed",
        "stages": stages,
        "operational": {k: dict(v) for k, v in aggregate.items()},
        "rehash_and_global_envelopes": checked,
        "independent_raw_counts": dict(count),
        "invalid_type_labels_all_turns": dict(invalid_labels),
        "terminal_invalid_type_labels": dict(terminal_labels),
        "failed_vote_details": failure_rows,
        "null_details": [x for x in exported if x["result"] is None],
        "by_policy_costs": {k: dict(v) for k, v in cost.items()},
        "new_model_calls": 0,
    }
    save(REVIEW / "summary.json", summary)
    verify(inputs)
    save(
        REVIEW / "complete.json",
        {
            "status": summary["status"],
            "inputs": inputs,
            "outputs": {
                rel(p): digest(p)
                for p in [REVIEW / "plan.json", out, REVIEW / "summary.json"]
            },
            "new_model_calls": 0,
        },
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "counts": dict(count),
                "raw_files": len(outputs),
            }
        ),
        flush=True,
    )


def f1(counts):
    tp, fp, fn = (counts[..., k] for k in range(3))
    den = 2 * tp + fp + fn
    per = np.divide(2 * tp, den, out=np.zeros_like(den, dtype=float), where=den != 0)
    return per.mean(axis=-1), per


def data(ids, run, gold):
    y = np.array([[lab in gold[i] for lab in LABELS] for i in ids], dtype=bool)
    valid = np.array([run[i]["result"] is not None for i in ids])
    x = np.array(
        [
            [
                lab in run[i]["result"]["hate_type"] if valid[k] else False
                for lab in LABELS
            ]
            for k, i in enumerate(ids)
        ],
        dtype=bool,
    )
    gate = np.array(
        [run[i]["result"]["hate"] if valid[k] else False for k, i in enumerate(ids)],
        dtype=bool,
    )
    require(not np.any(x[valid & ~gate]), "Labels despite false gate")
    counts = np.stack([y & x, ~y & x, y & ~x], axis=-1).astype(np.int32)
    rates = np.stack(
        [np.all(~y | x, axis=1), np.all(y == x, axis=1), x.sum(axis=1), ~gate], axis=-1
    ).astype(float)
    return y, x, valid, counts, rates


def metric(ids, run, gold):
    from sklearn.metrics import f1_score

    y, x, valid, counts, rates = data(ids, run, gold)
    n = int(valid.sum())
    require(n > 0, "No valid predictions")
    score, per = f1(counts[valid].sum(axis=0))
    check = f1_score(y[valid], x[valid], average=None, zero_division=0)
    require(
        np.allclose(per, check, rtol=0, atol=1e-13), "Independent sklearn F1 differs"
    )
    missing = ~valid
    extremes = []
    for imputed in [~y[missing], y[missing]]:
        all_x = x.copy()
        all_x[missing] = imputed
        cc = np.stack([y & all_x, ~y & all_x, y & ~all_x], axis=-1).sum(axis=0)
        extremes.append(float(f1(cc)[0]))
    return {
        "n": len(ids),
        "valid": n,
        "null_ids": [i for i, ok in zip(ids, valid) if not ok],
        "macro_f1": float(score),
        "per_label_f1": dict(zip(LABELS, per.tolist())),
        "contains": float(rates[valid, 0].mean()),
        "exact": float(rates[valid, 1].mean()),
        "predicted_labels": float(rates[valid, 2].mean()),
        "missed_hate_rate": float(rates[valid, 3].mean()),
        "full_roster_missing_prediction_bounds": extremes,
    }


def groups_for(policies):
    return {
        p: [p] if p == "reference" else [f"{p}_s{s}" for s in SEEDS] for p in policies
    }


def comparison(ids, runs, gold, policies, draws=10000):
    groups = groups_for(policies)
    keys = [(c, r) for members in groups.values() for c in members for r in range(3)]
    common = [i for i in ids if all(runs[k][i]["result"] is not None for k in keys)]
    require(common, "Empty common-valid roster")
    arrays = [data(common, runs[k], gold) for k in keys]
    counts = np.stack([v[3] for v in arrays])
    rates = np.stack([v[4] for v in arrays])
    point, per = f1(counts.sum(axis=1))
    n = len(common)
    picks = np.random.default_rng(20260930).integers(0, n, size=(draws, n))
    weights = np.stack([np.bincount(row, minlength=n) for row in picks]).astype(float)
    sums = (weights @ counts.transpose(1, 0, 2, 3).reshape(n, -1)).reshape(
        draws, len(keys), 7, 3
    )
    boot = f1(sums)[0]

    for setting in range(len(keys)):
        for start in range(0, draws, 128):
            expanded = counts[setting][picks[start : start + 128]].sum(axis=1)
            den = 2 * expanded[..., 0] + expanded[..., 1] + expanded[..., 2]
            alt = np.divide(
                2 * expanded[..., 0],
                den,
                out=np.zeros_like(den, dtype=float),
                where=den > 0,
            ).mean(axis=-1)
            require(
                np.allclose(
                    boot[start : start + 128, setting], alt, rtol=0, atol=1e-13
                ),
                "Expanded bootstrap differs",
            )
    setting_results = {}
    group_results = {}
    group_boot = {}
    for index, (c, rep) in enumerate(keys):
        own = metric(ids, runs[c, rep], gold)
        matched = metric(common, runs[c, rep], gold)
        require(
            abs(matched["macro_f1"] - point[index]) < 1e-13, "Point estimates differ"
        )
        setting_results[f"{c}_rep{rep}"] = {"own_valid": own, "common_valid": matched}
    for policy, members in groups.items():
        positions = [[keys.index((c, rep)) for rep in range(3)] for c in members]
        flat = [i for pp in positions for i in pp]
        means = [float(point[pp].mean()) for pp in positions]
        group_boot[policy] = np.mean(
            [boot[:, pp].mean(axis=1) for pp in positions], axis=0
        )
        own = [
            setting_results[f"{c}_rep{rep}"]["own_valid"]
            for c in members
            for rep in range(3)
        ]
        group_results[policy] = {
            "macro_f1": float(np.mean(means)),
            "pool_means": means,
            "pool_sample_sd": float(np.std(means, ddof=1)) if len(means) > 1 else None,
            "per_pool_decoder_sample_sd": [
                float(np.std(point[pp], ddof=1)) for pp in positions
            ],
            "per_label_f1": dict(zip(LABELS, per[flat].mean(axis=0).tolist())),
            "contains": float(rates[flat, :, 0].mean()),
            "exact": float(rates[flat, :, 1].mean()),
            "predicted_labels": float(rates[flat, :, 2].mean()),
            "missed_hate_rate": float(rates[flat, :, 3].mean()),
            "own_valid_macro_f1": float(np.mean([m["macro_f1"] for m in own])),
            "full_roster_missing_prediction_bounds": np.mean(
                [m["full_roster_missing_prediction_bounds"] for m in own], axis=0
            ).tolist(),
        }
    contrasts = {}
    specs = [
        ("relevant_full", "natural"),
        ("relevant_full", "disjoint_full"),
        ("relevant_full", "relevant_labels"),
        ("disjoint_full", "disjoint_labels"),
        ("relevant_labels", "disjoint_labels"),
        ("natural", "reference"),
        ("relevant_full", "reference"),
        ("relevant_labels", "reference"),
    ]
    for left, right in specs:
        if left not in groups or right not in groups:
            continue
        diff = group_boot[left] - group_boot[right]
        a = group_results[left]
        b = group_results[right]
        ab = a["full_roster_missing_prediction_bounds"]
        bb = b["full_roster_missing_prediction_bounds"]
        contrasts[left + "_minus_" + right] = {
            "difference": a["macro_f1"] - b["macro_f1"],
            "ci95": np.quantile(diff, [0.025, 0.975]).tolist(),
            "ci9875": np.quantile(diff, [0.00625, 0.99375]).tolist(),
            "full_roster_missing_prediction_difference_bounds": [
                ab[0] - bb[1],
                ab[1] - bb[0],
            ],
        }
    if all(
        x in groups
        for x in [
            "relevant_full",
            "relevant_labels",
            "disjoint_full",
            "disjoint_labels",
        ]
    ):
        diff = (
            group_boot["relevant_full"]
            - group_boot["relevant_labels"]
            - group_boot["disjoint_full"]
            + group_boot["disjoint_labels"]
        )
        value = sum(
            sign * group_results[g]["macro_f1"]
            for sign, g in [
                (1, "relevant_full"),
                (-1, "relevant_labels"),
                (-1, "disjoint_full"),
                (1, "disjoint_labels"),
            ]
        )
        contrasts["text_by_label_overlap_interaction"] = {
            "difference": value,
            "ci95": np.quantile(diff, [0.025, 0.975]).tolist(),
        }
    return {
        "original_n": len(ids),
        "common_valid_n": n,
        "common_valid_ids": common,
        "excluded_ids": sorted(set(ids) - set(common)),
        "groups": group_results,
        "settings": setting_results,
        "contrasts": contrasts,
        "bootstrap": {
            "draws": draws,
            "seed": 20260930,
            "independently_checked_setting_draws": draws * len(keys),
            "sampling": "Same singleton text clusters across every setting. Fixed pools and decoder repeats; one reference with three actual repeats.",
            "primary_interval": 0.9875,
            "secondary_intervals": "Pointwise descriptive 95%; additional stored 98.75% endpoints do not create additional primary claims.",
        },
    }


def tests():
    gold = {"a": {"explicit"}, "b": {"irony"}, "c": {"explicit"}}
    pred = lambda labels, hate=True: {"result": {"hate": hate, "hate_type": labels}}
    perfect = {i: pred(sorted(g)) for i, g in gold.items()}
    zero = {i: pred([], False) for i in gold}
    require(
        abs(metric(list(gold), perfect, gold)["macro_f1"] - 2 / 7) < 1e-13,
        "Fixed label average",
    )
    null = dict(perfect)
    null["c"] = {"result": None}
    m = metric(list(gold), null, gold)
    require(
        m["valid"] == 2
        and m["full_roster_missing_prediction_bounds"][0]
        < m["full_roster_missing_prediction_bounds"][1],
        "Missing bounds",
    )
    extra = {i: pred(sorted(g | {"grievance"})) for i, g in gold.items()}
    require(
        metric(list(gold), extra, gold)["contains"] == 1
        and metric(list(gold), extra, gold)["exact"] == 0,
        "Contains/exact distinction",
    )
    runs = {
        (c, r): perfect
        for cc in groups_for(POLICIES).values()
        for c in cc
        for r in range(3)
    }
    z = comparison(list(gold), runs, gold, POLICIES, 31)
    require(
        all(
            abs(v["difference"]) < 1e-13 and np.allclose(v["ci95"], [0, 0])
            for v in z["contrasts"].values()
        ),
        "Paired identical fixture",
    )
    require(
        len(z["groups"]["reference"]["pool_means"]) == 1, "Shared reference duplication"
    )
    for s in SEEDS:
        for r in range(3):
            runs[f"natural_s{s}", r] = zero
    z = comparison(list(gold), runs, gold, POLICIES, 31)
    require(
        z["contrasts"]["relevant_full_minus_natural"]["ci9875"][0] > 0,
        "Positive contrast",
    )
    runs["natural_s202609281", 0] = null
    z = comparison(list(gold), runs, gold, ["natural", "relevant_full"], 31)
    require(
        z["excluded_ids"] == ["c"] and z["common_valid_n"] == 2,
        "Common mask across all pools/repeats",
    )
    p = {
        "hate": True,
        "hate_type": ["explicit"],
        "target_group": ["religion"],
        "severity": "high",
        "legal": ["insult_defamation"],
    }
    q = {**p, "hate": False, "hate_type": [], "target_group": [], "severity": None}
    v, n, tie = vote([p, q, None])
    require(
        n == 2
        and tie
        and not v["hate"]
        and v["hate_type"] == []
        and v["legal"] == ["insult_defamation"],
        "Tie and legal independence",
    )
    require(vote([None] * 3) == (None, 0, False), "All invalid")
    out = RESULTS / "development" / f"tests_{digest(__file__)[:16]}.json"
    save(
        out,
        {
            "passed": True,
            "fixture_checks": 9,
            "source_sha256": digest(__file__),
            "created_utc": utc(),
            "new_model_calls": 0,
        },
    )
    print("Nine scoring/voting/masking fixtures passed.", flush=True)


def score():
    require(not (RESULTS / "complete.json").exists(), "Preserve completed results")
    tested = RESULTS / "development" / f"tests_{digest(__file__)[:16]}.json"
    require(read(tested)["passed"], "Current tests required")
    complete = read(REVIEW / "complete.json")
    verify(complete["outputs"])
    science = read(SCIENCE)
    require(
        digest(SCIENCE) == SCIENCE_SHA and science["labels"] == LABELS,
        "Science differs",
    )
    inputs = {
        **complete["outputs"],
        rel(REVIEW / "complete.json"): digest(REVIEW / "complete.json"),
        rel(SCIENCE): SCIENCE_SHA,
        rel(__file__): digest(__file__),
        rel(tested): digest(tested),
        rel(ROSTER): digest(ROSTER),
    }
    save(
        RESULTS / "plan.json",
        {
            "created_utc": utc(),
            "inputs": inputs,
            "primary": science["primary"],
            "inference": science["inference"],
            "secondary_mask": "Shared-valid across all 48 settings/repeats for each secondary roster; primary uses only its 18 settings/repeats. Own-valid and full-roster failure bounds retained.",
            "null_bounds": "For each missing prediction, complement of all seven gold indicators versus gold indicators gives attainable per-system F1 extremes; subtract opposite extremes for conservative contrasts.",
            "new_model_calls": 0,
        },
    )
    rr = rows(REVIEW / "predictions.jsonl")
    runs = defaultdict(dict)
    roster = {r["id"]: r for r in rows(ROSTER)}
    gold = {i: set(v["gold"]["hate_type"]) for i, v in roster.items()}
    for r in rr:
        require(
            r["id"] not in runs[r["condition"], r["repeat"]],
            "Repeated exported prediction",
        )
        runs[r["condition"], r["repeat"]][r["id"]] = r
    require(
        len(runs) == 48
        and all(set(v) == set(science["rosters"]["evaluation"]) for v in runs.values()),
        "Incomplete assembled matrix",
    )
    scopes = [
        ("primary", "primary_full_cover", ["natural", "relevant_full"]),
        ("primary_controls", "primary_full_cover", POLICIES),
        ("all_transfer", "evaluation", POLICIES),
        ("selection_holdout", "holdout_from_original_selection", POLICIES),
        ("historical_sentinels", "historical_sentinels", POLICIES),
    ]
    summary = {}
    for endpoint, roster_name, policies in scopes:
        out = comparison(science["rosters"][roster_name], runs, gold, policies)
        save(RESULTS / f"{endpoint}.json", out)
        summary[endpoint] = {
            k: out[k]
            for k in [
                "original_n",
                "common_valid_n",
                "excluded_ids",
                "groups",
                "contrasts",
                "bootstrap",
            ]
        }
        print(
            json.dumps(
                {
                    "endpoint": endpoint,
                    "n": out["common_valid_n"],
                    "scores": {k: v["macro_f1"] for k, v in out["groups"].items()},
                    "primary_contrast": out["contrasts"]["relevant_full_minus_natural"],
                }
            ),
            flush=True,
        )
    partial = {
        f"{c}_rep{r}": metric(
            science["rosters"]["secondary_only_partial_cover"], run, gold
        )
        for (c, r), run in runs.items()
    }
    save(RESULTS / "partial_cover_seven.json", partial)
    save(
        RESULTS / "summary.json",
        {
            "created_utc": utc(),
            "status": "frozen_relevance_endpoints_scored",
            "endpoints": summary,
            "limits": [
                "Artificial query-gold diagnostic; not deployable retrieval accuracy.",
                "Full versus labels-only changes content and length; no pure semantic-mechanism claim.",
                "Previously inspected DEV and overlapping diagnostic subsets, not untouched confirmation.",
                "All 350 inputs are hateful; no benign false-positive result.",
                "Intervals condition on these three pools and decoder repeats; no extra independent texts or reference copies.",
            ],
            "new_model_calls": 0,
        },
    )
    verify(inputs)
    artifacts = [RESULTS / f"{ep}.json" for ep, _, _ in scopes] + [
        RESULTS / "partial_cover_seven.json",
        RESULTS / "summary.json",
        RESULTS / "plan.json",
    ]
    save(
        RESULTS / "complete.json",
        {
            "status": "frozen_relevance_endpoints_scored",
            "created_utc": utc(),
            "inputs": inputs,
            "outputs": {rel(p): digest(p) for p in artifacts},
            "new_model_calls": 0,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["tests", "review", "score"])
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", forbid), patch.object(
        socket, "create_connection", forbid
    ):
        {"tests": tests, "review": review, "score": score}[args.mode]()


if __name__ == "__main__":
    main()
