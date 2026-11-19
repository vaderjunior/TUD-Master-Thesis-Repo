"""Policy setup."""

from src.hsrag.paths import PROJECT_ROOT
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys
from unittest.mock import patch

ROOT = PROJECT_ROOT
BASE = ROOT / "experiments/astra/complete_policy_setup_20261114_v1"
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
BROAD = ROOT / "experiments/astra/later_evaluation_20261008_v1"
GOLD = (
    ROOT
    / "experiments/astra/offline_closure_20261006_v1/evaluation_preparation/broad_roster.jsonl"
)
PROMPTS = BROAD / "retrieval_v3/full/prompts.jsonl"
RECORDS = CORE / "kb/full/records.jsonl"
TOKENIZER = ROOT / "experiments/astra/qwen_tokenizer_assets_20261001_v1"
CONDITIONS = ("retrieved_policy", "complete_definitions", "complete_policy")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(
        (
            json.dumps(
                value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()


def ref(path):
    return Path(path).relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def save(name, value):
    path = BASE / name
    require(
        path.resolve().is_relative_to(BASE.resolve()), "Output outside new setup folder"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            value, handle, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False
        )
        handle.write("\n")


def write_rows(name, values):
    path = BASE / name
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(
                json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
                + "\n"
            )


def forbidden(*args, **kwargs):
    raise RuntimeError("Offline policy preparation: network access is forbidden")


@contextmanager
def offline():
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
        CUDA_VISIBLE_DEVICES="",
    )
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket.socket, "connect_ex", forbidden
    ), patch.object(socket, "create_connection", forbidden), patch.object(
        socket, "getaddrinfo", forbidden
    ):
        yield


def sources():
    return [
        RECORDS,
        GOLD,
        PROMPTS,
        BROAD / "retrieval_v3/complete.json",
        BROAD / "replay_review_20261009_v1/sensitivity_plan.json",
        ROOT / "experiments/astra/german_guidelines_20261112_v1/plan.json",
        ROOT / "config/taxonomy.yaml",
        ROOT / "prompts/classify_v1.txt",
        ROOT / "src/hsrag/prompt.py",
        ROOT / "src/hsrag/parse.py",
        ROOT / "src/hsrag/vote.py",
        ROOT / "src/hsrag/schema.py",
        ROOT / "analysis/common/prompt.py",
        ROOT / "analysis/common/api.py",
        TOKENIZER / "complete.json",
        TOKENIZER / "tokenizer.json",
        TOKENIZER / "chat_template.json",
        TOKENIZER / "config.json",
        ROOT
        / "Thesis_Log/astra_result/german_guidelines_part7_synthesis_20261113_v2.md",
        ROOT / "Thesis_Log/astra_result/complete_policy_experiment_plan_20261114_v1.md",
    ]


def inspect():
    BASE.mkdir(exist_ok=False)
    before = {ref(p): {"sha256": sha(p), "bytes": p.stat().st_size} for p in sources()}
    old = read(ROOT / "experiments/astra/german_guidelines_20261112_v1/plan.json")
    for path in [RECORDS, GOLD, PROMPTS]:
        require(
            before[ref(path)]["sha256"] == old["input_hashes"][ref(path)],
            "Frozen source changed: " + ref(path),
        )
    for name in ["tokenizer.json", "chat_template.json", "config.json"]:
        require(
            sha(TOKENIZER / name)
            == read(TOKENIZER / "complete.json")["files"][name]["sha256"],
            "Tokenizer changed",
        )
    save(
        "source_snapshot.json",
        {"created_utc": datetime.now(timezone.utc).isoformat(), "sources": before},
    )
    records = []
    kinds = Counter()
    for index, row in enumerate(rows(RECORDS)):
        kinds[row["kind"]] += 1
        if row["kind"] in {"definition", "guideline"}:
            records.append(
                {
                    "bank_index": index,
                    "record": row,
                    "record_sha256": signature(row),
                    "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                }
            )
    require(
        Counter((r["record"]["kind"] for r in records))
        == {"definition": 21, "guideline": 19},
        "Policy inventory differs",
    )
    save("policy_inventory.json", records)
    roster = list(rows(GOLD))
    groups = defaultdict(list)
    for row in roster:
        groups[row["phase"]].append(row)
    summary = {}
    for phase, rr in groups.items():
        summary[phase] = {
            "n": len(rr),
            "source": dict(Counter((r["source"] for r in rr))),
            "endpoints": dict(Counter((d for r in rr for d in r["endpoints"]))),
            "row_fields": sorted(rr[0]),
            "sample_without_text": {k: v for k, v in rr[0].items() if k != "text"},
        }
    first = next(rows(PROMPTS))
    save(
        "intake_summary.json",
        {
            "bank_kinds": dict(kinds),
            "policy_records": len(records),
            "roster": summary,
            "prompt_fields": sorted(first),
            "prompt_summary": {
                k: v
                for k, v in first.items()
                if k not in {"text", "messages", "context"}
            },
            "context_keys": list(first["context"]),
            "disk": {
                drive: {
                    "free_bytes": shutil.disk_usage(drive).free,
                    "total_bytes": shutil.disk_usage(drive).total,
                }
                for drive in ["C:/", "D:/"]
            },
        },
    )
    print(
        json.dumps(
            {
                "policy_records": len(records),
                "bank_kinds": dict(kinds),
                "roster": {
                    k: {
                        key: val
                        for key, val in v.items()
                        if key != "sample_without_text"
                    }
                    for k, v in summary.items()
                },
                "prompt_fields": sorted(first),
            },
            indent=2,
        )
    )


def truth(row, dimension):
    gate = row.get("gate")
    if dimension == "hate":
        return gate
    if dimension != "legal" and gate is not True:
        return None
    value = row.get(
        {
            "hate_type": "hate_types",
            "target_group": "target_groups",
            "severity": "severity",
            "legal": "legal",
        }[dimension]
    )
    return sorted(set(value)) if isinstance(value, list) else value


SELECTION_SEED = "complete-policy-20261114-v1"


def sampled_keys(records, target):
    groups = defaultdict(list)
    for row in records:
        if type(row.get("gate")) is bool:
            groups[row["source"], row["gate"]].append(row)
    total = sum(map(len, groups.values()))
    quotas = {group: target * len(rr) // total for group, rr in groups.items()}
    remaining = target - sum(quotas.values())
    order = sorted(groups, key=lambda g: (-(target * len(groups[g]) % total), g))
    for group in order[:remaining]:
        quotas[group] += 1
    selected = []
    report = []
    for group, rr in sorted(groups.items()):
        chosen = sorted(
            rr,
            key=lambda r: hashlib.sha256(
                (SELECTION_SEED + "|" + r["key"]).encode()
            ).hexdigest(),
        )[: quotas[group]]
        selected.extend((r["key"] for r in chosen))
        report.append(
            {
                "source": group[0],
                "gate": group[1],
                "eligible": len(rr),
                "selected": len(chosen),
            }
        )
    require(
        len(selected) == target and len(set(selected)) == target,
        "Invalid stratified sample",
    )
    return (sorted(selected), report)


def roster_for(en_target, de_target):
    original = list(rows(GOLD))
    groups = defaultdict(list)
    for row in original:
        groups[row["phase"]].append(row)
    en, en_strata = sampled_keys(groups["en_test"], en_target)
    de, de_strata = sampled_keys(groups["de_test"], de_target)
    endpoints = {"en_hate": en, "de_hate": de}
    for dimension in ("hate_type", "target_group", "severity"):
        endpoints["en_" + dimension] = sorted(
            (r["key"] for r in groups["en_test"] if truth(r, dimension) is not None)
        )
    endpoints["de_legal"] = sorted(
        (r["key"] for r in groups["de_legal_test"] if truth(r, "legal") is not None)
    )
    selected = set().union(*map(set, endpoints.values()))
    lookup = {r["key"]: r for r in original}
    require(len(lookup) == len(original), "Duplicate original roster key")
    return {
        "name": f"en{en_target}_de{de_target}_all_fine",
        "selected_keys": sorted(selected),
        "endpoints": endpoints,
        "strata": {"en_hate": en_strata, "de_hate": de_strata},
        "records": [lookup[k] for k in sorted(selected)],
    }


def counter():
    from tokenizers import Tokenizer
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    token = Tokenizer.from_file(str(TOKENIZER / "tokenizer.json"))
    token.no_truncation()
    token.no_padding()
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(
        ValueError(message)
    )
    template = env.from_string(read(TOKENIZER / "chat_template.json")["chat_template"])

    def count(messages):
        formatted = template.render(
            messages=[{"role": m["role"], "content": m["text"]} for m in messages],
            add_generation_prompt=True,
            tools=None,
        )
        return len(token.encode(formatted, add_special_tokens=False).ids)

    return count


def hit(row):
    meta = {key: row[key] for key in ("kind", "lang", "source", "dimension", "label")}
    meta.update(row.get("meta", {}))
    return {"id": row["id"], "text": row["text"], "meta": meta, "via": "fixed_policy"}


def policy_hits():
    records = [r["record"] for r in read(BASE / "policy_inventory.json")]
    return (
        [hit(r) for r in records if r["kind"] == "definition"],
        [hit(r) for r in records if r["kind"] == "guideline"],
    )


def messages_for(old, text):
    from analysis.common import prompt as prompt

    definitions, guidelines = policy_hits()
    require(
        prompt.render_messages(text, old["lang"], old["context"], "corrected")
        == old["messages"],
        "Original prompt does not reproduce: " + old["key"],
    )
    a, b, c = [deepcopy(old["context"]) for _ in range(3)]
    b["definitions"] = definitions
    c["definitions"] = definitions
    c["guidelines"] = guidelines
    return [
        (CONDITIONS[0], old["messages"], a),
        (CONDITIONS[1], prompt.render_messages(text, old["lang"], b, "corrected"), b),
        (CONDITIONS[2], prompt.render_messages(text, old["lang"], c, "corrected"), c),
    ]


def profile():
    additional = [
        "analysis/common/scoring.py",
        "analysis/common/result_reader.py",
        "analysis/knowledge/german_guidelines/prepare.py",
    ]
    save(
        "additional_source_bindings.json",
        {name: sha(ROOT / name) for name in additional},
    )
    candidates = [
        roster_for(en, de) for en, de in [(2000, 1200), (2400, 1500), (3000, 2000)]
    ]
    for candidate in candidates:
        save(
            "candidates/" + candidate["name"] + ".json",
            {k: v for k, v in candidate.items() if k != "records"},
        )
    selected = set().union(*(set(c["selected_keys"]) for c in candidates))
    lookup = {r["key"]: r for r in rows(GOLD) if r["key"] in selected}
    count_tokens = counter()
    token_rows = []
    seen = set()
    for line_no, old in enumerate(rows(PROMPTS), 1):
        key = old["key"]
        if key not in selected:
            continue
        require(key not in seen, "Duplicate prompt key")
        seen.add(key)
        require(
            signature(old["messages"]) == old["messages_sha256"]
            and signature(old["context"]) == old["context_sha256"],
            "Original stored prompt hash differs",
        )
        require(
            len(old["context"]["definitions"]) == 2
            and len(old["context"]["guidelines"]) == 2
            and (sum((len(rr) for _, rr in old["context"]["example_groups"])) == 5),
            "Original channel budget differs",
        )
        values = messages_for(old, lookup[key]["text"])
        token_rows.append(
            {
                "key": key,
                "source_prompt_line": line_no,
                "population": old["population"],
                "original_messages_sha256": old["messages_sha256"],
                "original_context_sha256": old["context_sha256"],
                "counts": {
                    cond: count_tokens(messages) for cond, messages, ctx in values
                },
                "message_hashes": {
                    cond: signature(messages) for cond, messages, ctx in values
                },
            }
        )
        if len(seen) % 1000 == 0:
            print(
                f"Checked {len(seen)}/{len(selected)} input prompts offline.",
                flush=True,
            )
    require(seen == selected, "Missing original prompt")
    write_rows("candidate_prompt_counts.jsonl", token_rows)
    from statistics import mean, median

    out = []
    for candidate in candidates:
        keep = set(candidate["selected_keys"])
        rr = [r for r in token_rows if r["key"] in keep]
        item = {
            k: v
            for k, v in candidate.items()
            if k not in {"records", "selected_keys", "endpoints"}
        }
        item.update(
            inputs=len(rr),
            prompts=3 * len(rr),
            predictions=9 * len(rr),
            ordinary_calls=27 * len(rr),
            endpoint_counts={k: len(v) for k, v in candidate["endpoints"].items()},
            tokens={
                cond: {
                    "mean": mean((r["counts"][cond] for r in rr)),
                    "median": median((r["counts"][cond] for r in rr)),
                    "min": min((r["counts"][cond] for r in rr)),
                    "max": max((r["counts"][cond] for r in rr)),
                    "ordinary_input_tokens": 9 * sum((r["counts"][cond] for r in rr)),
                }
                for cond in CONDITIONS
            },
        )
        out.append(item)
    save("candidate_profiles.json", out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["inspect", "profile"])
    args = parser.parse_args()
    with offline():
        {"inspect": inspect, "profile": profile}[args.mode]()
