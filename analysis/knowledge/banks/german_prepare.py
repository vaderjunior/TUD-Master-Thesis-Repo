"""Banks german prepare."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
from analysis.knowledge.banks import build as u
from analysis.knowledge.banks.german_results import username_norm
from analysis.common.prompt import corrected_gold, module

ROOT = u.ROOT
BASE = ROOT / "experiments/astra/german_kb_comparison_20260913_v1"
PLAN = BASE / "plan.json"
AUDIT = ROOT / "experiments/astra/german_kb_audit_20260913_v1"
SCOPES = {"gate": "de_dev_eval", "legal": "de_legal_dev_eval"}
CONDITIONS = {
    "a_legacy": ("original_small", "legacy"),
    "b_fixed": ("original_small", "corrected"),
    "c_canonical": ("canonical_small", "corrected"),
    "d_full": ("canonical_full", "corrected"),
}
POOLS = ("original_small", "canonical_small", "canonical_full")
KEEP_BOUNDARY = {
    "detox-1406639349408616454": "Handles/link dominate; emoji content and URL differ; no shared substantive sentence.",
    "detox-1394065387009024013": "Handles and different links without a substantive comment; insufficient copy evidence.",
    "detox-1398572002508775426": "Short generic lead-in, added named person and different URL; insufficient copy evidence.",
    "detox-1403991288672927744": "Same handles but different emoji-only messages; potentially different meaning.",
}


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def owned(path):
    p = Path(path).resolve()
    require(
        p.is_relative_to(BASE.resolve()) and p != BASE.resolve(),
        "Write outside comparison refused",
    )
    return p


def js(obj):
    return (
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def lines(rows):
    return "".join(
        (
            json.dumps(r, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            for r in rows
        )
    ).encode("utf-8")


def write(path, content):
    p = owned(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        require(p.read_bytes() == content, f"Existing file differs: {p}")
        return
    tmp = owned(p.with_name(p.name + f".partial_{time.time_ns()}"))
    with tmp.open("xb") as f:
        f.write(content)
    tmp.rename(p)


def canon_id(rid):
    return rid.replace("ex-detox-legal-", "detox-").removeprefix("ex-")


def merged_meta(rows):
    result = {}
    for field in ("gate", "target_groups", "hate_types", "severity", "legal"):
        observed = [u.clean(r.get(field)) for r in rows]
        observed = [v for v in observed if v is not None]
        keys = {json.dumps(sorted(v) if isinstance(v, list) else v) for v in observed}
        require(len(keys) <= 1, f"Conflicting observed {field} labels")
        result[field] = observed[0] if observed else None
    flags = set()
    for row in rows:
        if row["source"] == "detox":
            raw = json.loads(row["raw"])
            flags.update(
                (
                    k.removeprefix("p_")
                    for k, v in raw.get("legal_paragraphs", {}).items()
                    if v > 0.5
                )
            )
    result["illustrative_only"] = False
    if flags:
        result["astra_aux_stgb"] = ",".join(sorted(flags))
    result["astra_train_ids"] = sorted((str(r["id"]) for r in rows))
    return result


def build_pools(raw, original, excluded):
    by_id = {str(r["id"]): r for r in raw}
    old_de = [r for r in original if r["kind"] == "example" and r["lang"] == "de"]
    eligible_old = [r for r in old_de if canon_id(r["id"]) not in excluded]
    require(
        all((r["text"] == by_id[canon_id(r["id"])]["text"] for r in old_de)),
        "Original text differs",
    )
    background = [r for r in original if r not in old_de]
    groups = defaultdict(list)
    for row in raw:
        if str(row["id"]) not in excluded:
            groups[u.canonical_text(row["text"])].append(row)
    selected = {u.canonical_text(r["text"]): r for r in eligible_old}
    require(
        len(selected) == len(eligible_old),
        "Small pool contains repeated canonical text",
    )
    order = list(selected) + sorted(set(groups) - set(selected))
    canonical = []
    provenance = []
    for key in order:
        rr = sorted(groups[key], key=lambda r: str(r["id"]))
        old = selected.get(key)
        representative = by_id[canon_id(old["id"])] if old else rr[0]
        rid = old["id"] if old else "ex-" + str(representative["id"])
        rec = {
            "id": rid,
            "kind": "example",
            "dimension": None,
            "label": None,
            "lang": "de",
            "text": representative["text"],
            "source": "+".join(sorted({r["source"] for r in rr})) + "-train",
            "meta": merged_meta(rr),
        }
        canonical.append(rec)
        provenance.append(
            {
                "record_id": rid,
                "train_ids": rec["meta"]["astra_train_ids"],
                "in_small": bool(old),
                "representative_train_id": str(representative["id"]),
                "old_record": old,
                "canonical_record": rec,
            }
        )
    pools = {
        "original_small": [r for r in original if r not in old_de or r in eligible_old],
        "canonical_small": background + canonical[: len(eligible_old)],
        "canonical_full": background + canonical,
    }
    for name, rr in pools.items():
        require(len({r["id"] for r in rr}) == len(rr), f"Duplicate IDs: {name}")
        require(
            len({u.canonical_text(r["text"]) for r in rr}) == len(rr),
            f"Duplicate texts: {name}",
        )
        require(
            [r for r in rr if not (r["lang"] == "de" and r["kind"] == "example")]
            == background,
            "English/abstract background changed",
        )
    full = {r["id"]: r for r in pools["canonical_full"]}
    require(
        all((full[r["id"]] == r for r in pools["canonical_small"])),
        "Canonical small not nested identically",
    )
    return (pools, provenance)


def summarize(rr):
    de = [r for r in rr if r["kind"] == "example" and r["lang"] == "de"]
    return {
        "total_records": len(rr),
        "german_examples": len(de),
        "source": dict(Counter((r["source"] for r in de))),
        "gate": dict(Counter((str(r["meta"].get("gate")) for r in de))),
        "illustrations": sum((r["meta"].get("illustrative_only", False) for r in de)),
        "legal_annotated": sum((r["meta"].get("legal") is not None for r in de)),
        "positive_legal": sum((bool(r["meta"].get("legal")) for r in de)),
        "aux_flags": sum((bool(r["meta"].get("astra_aux_stgb")) for r in de)),
    }


def prepare():
    import pandas as pd
    import yaml

    require(not PLAN.exists(), "Plan already exists; use validate, not re-prepare")
    result_path = (
        ROOT / "Thesis_Log/astra_result/evidence/german_kb_results_20260913_v1.json"
    )
    follow_path = (
        ROOT
        / "Thesis_Log/astra_result/evidence/german_kb_overlap_followup_20260913_v1.json"
    )
    evidence = u.read_json(result_path)
    follow = u.read_json(follow_path)
    for e in (evidence, follow):
        for p, h in e["input_hashes"].items():
            require(u.digest(ROOT / p) == h, f"Audit source changed: {p}")
    raw = []
    for n in ("de_train", "de_legal_train"):
        raw += pd.read_parquet(ROOT / f"data/processed/{n}.parquet").to_dict("records")
    original = u.read_lines(ROOT / "kb/records.jsonl")
    base_excluded = set(u.read_json(AUDIT / "report.json")["proposed_excluded_ids"])
    user_ids = {
        r["train_id"] for r in evidence["retained_username_normalized_candidates"]
    }
    decisions = []
    manual = set()
    for r in evidence["retained_094_to_095"]:
        rid = r["train_id"]
        if rid in user_ids:
            decision, reason = (
                "exclude_username_match",
                "Exact normalized body after handle replacement; uniform automatic rule.",
            )
        elif rid in KEEP_BOUNDARY:
            decision, reason = ("retain", KEEP_BOUNDARY[rid])
        else:
            decision = "exclude_related_text"
            reason = (
                "Closely shared sentence template or local rewrite; exclude related TRAIN member even if the edit can change its label."
                if rid.startswith("gahd-")
                else "Shared substantive comment body with prefix/suffix truncation or added/removed handles; anonymization compatible."
            )
            manual.add(rid)
        decisions.append({**r, "decision": decision, "reason": reason})
    require(
        len(decisions) == 47 and len(manual) == 38, "Unexpected manual-review roster"
    )
    excluded = base_excluded | user_ids | manual
    equiv = {username_norm(r["text"]) for r in raw if r["id"] in excluded}
    excluded |= {str(r["id"]) for r in raw if username_norm(r["text"]) in equiv}
    held = u.read_lines(AUDIT / "heldout_texts.jsonl")
    held_keys = {username_norm(r["text"]) for r in held}
    require(
        not any(
            (
                username_norm(r["text"]) in held_keys
                for r in raw
                if r["id"] not in excluded
            )
        ),
        "Username overlap remains",
    )
    pools, provenance = build_pools(raw, original, excluded)
    policy = {
        "created_utc": u.now(),
        "original_scan_exclusions": sorted(base_excluded),
        "username_exclusions": sorted(user_ids),
        "manual_related_text_exclusions": sorted(manual),
        "review_decisions": decisions,
        "equivalence_propagation": sorted(excluded - base_excluded - user_ids - manual),
        "final_excluded_ids": sorted(excluded),
        "policy": "Additive, post-hoc, text-only overlap policy. Keep .95 scan; add username-normalized exact matches, reviewed derived templates/comment copies in the predefined .94-.95 band, and propagate equivalent TRAIN bodies. Do not infer identical labels. Four ambiguous link/emoji-dominated pairs retained explicitly. Within-KB merging uses NFKC/casefold/whitespace text equality, not handle stripping. Merge complementary observed TRAIN annotations, reject conflicts; DeTox paragraph flags remain auxiliary. Canonical small uses full-source annotation union for its selected texts, explicitly a distinct condition. No claim of exhaustive semantic-overlap removal below the screened range.",
        "counts": {
            "raw_train": len(raw),
            "scan": len(base_excluded),
            "username_additions": len(user_ids),
            "reviewed_boundary": len(decisions),
            "manual_additions": len(manual),
            "propagated_additions": len(excluded - base_excluded - user_ids - manual),
            "excluded_total": len(excluded),
            "remaining_train_rows": len(raw) - len(excluded),
        },
    }
    generated = []
    for name, records in pools.items():
        p = BASE / f"kb/{name}/records.jsonl"
        write(p, lines(records))
        generated.append(p)
    for name, value in (
        ("policy.json", policy),
        ("record_provenance.json", provenance),
    ):
        p = BASE / name
        write(p, js(value))
        generated.append(p)
    en = [r for r in original if r["kind"] == "example" and r["lang"] == "en"]
    require(
        all(
            (
                module("legacy")._render_gold(r["meta"]) == corrected_gold(r["meta"])
                for r in en
            )
        ),
        "English rendering changed",
    )
    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))
    retrieval = dict(cfg["retrieval"])
    require(
        retrieval["k_examples"] == 5
        and retrieval.get("k_examples_feedback", 0) == 0
        and (retrieval["strategy"] == "hybrid"),
        "Unexpected retrieval defaults",
    )
    runs = []
    for rep in (1, 2, 3):
        order = list(CONDITIONS)
        if rep == 2:
            order.reverse()
        for scope, subset in SCOPES.items():
            for condition in order:
                pool, renderer = CONDITIONS[condition]
                name = f"astra_de_{condition}_{scope}_r{rep}"
                runs.append(
                    {
                        "name": name,
                        "condition": condition,
                        "pool": pool,
                        "renderer": renderer,
                        "scope": scope,
                        "subset": subset,
                        "replicate": rep,
                        "n_items": len(
                            pd.read_parquet(
                                ROOT / f"data/processed/{subset}.parquet",
                                columns=["id"],
                            )
                        ),
                    }
                )
    files = {**evidence["input_hashes"], **follow["input_hashes"]}
    needed = [
        result_path,
        follow_path,
        ROOT / "config/config.yaml",
        ROOT / "prompts/classify_v1.txt",
        ROOT / "config/taxonomy.yaml",
    ]
    needed += list((ROOT / "src/hsrag").rglob("*.py"))
    needed += [
        ROOT / p
        for p in (
            "analysis/knowledge/banks/german_prepare.py",
            "analysis/common/prompt.py",
            "analysis/knowledge/banks/german_run.py",
            "tests/knowledge/banks/test_german_preparation.py",
            "scripts/run_german_bank.ps1",
        )
    ]
    files.update({u.rel(p): u.digest(p) for p in needed})
    plan = {
        "run_id": BASE.name,
        "created_utc": u.now(),
        "input_hashes": files,
        "generated_hashes": {u.rel(p): u.digest(p) for p in generated},
        "environment": u.environment(),
        "embedding_cache": u.cache_info(),
        "pool_counts": {k: summarize(v) for k, v in pools.items()},
        "policy_counts": policy["counts"],
        "runs": runs,
        "planned_calls": sum((r["n_items"] * 3 for r in runs)),
        "provider": "peasec",
        "model": "Qwen/Qwen3-VL-32B-Instruct",
        "temperature": 1.0,
        "votes": 3,
        "max_repairs": 2,
        "workers": 8,
        "retrieval": retrieval,
        "code_version": u.code_stamp(),
        "prompt_version": "classify_v1-"
        + u.digest(ROOT / "prompts/classify_v1.txt")[:8],
        "taxonomy_version": "taxonomy-" + u.digest(ROOT / "config/taxonomy.yaml")[:8],
        "renderer_sha256": u.digest(ROOT / "analysis/common/prompt.py"),
        "scope": "24 RAG-only runs: four conditions, two DEV scopes, three repeats. B-A isolates rendering on eligible original small; C-B changes annotation presentation/union; D-C expands the canonical pool with five retrieved examples fixed. No new test scoring; no RAG-vs-few-shot claim. Class/source/annotation mix is part of pool expansion. Primary endpoint gate strict macro-F1, secondary legal three-label macro-F1; report FP/FN, per-label precision/recall, failures, repeat variation and paired uncertainty. No selecting or resampling the best run. Further few-shot controls belong to broader SQ2.",
    }
    require(
        all((u.digest(ROOT / p) == h for p, h in files.items())),
        "Input changed during preparation",
    )
    write(PLAN, js(plan))
    print(
        json.dumps(
            {
                "policy": policy["counts"],
                "pools": plan["pool_counts"],
                "runs": len(runs),
                "planned_calls": plan["planned_calls"],
                "plan": u.rel(PLAN),
                "sha256": u.digest(PLAN),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    prepare()
