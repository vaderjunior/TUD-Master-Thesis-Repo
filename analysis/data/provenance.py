from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import re
import pandas as pd
from analysis.common import replay as c

BASE = c.BASE / "documentation"
PARENTS = [
    f"{lang}_{part}"
    for lang in ("en", "de", "de_legal")
    for part in ("train", "dev", "test")
]
FIELDS = ["gate", "target_groups", "hate_types", "severity", "legal"]
AUDITS = {
    "en": c.ROOT / "experiments/astra/english_kb_audit_20260914_v1",
    "de": c.ROOT / "experiments/astra/german_kb_audit_20260913_v1",
}


def oldnorm(s):
    return re.sub("\\s+", " ", s.lower()).strip()


def records(p):
    return [
        {k: c.clean(v) for k, v in r.items()}
        for r in pd.read_parquet(p).to_dict("records")
    ]


def census(rr):
    out = {
        "rows": len(rr),
        "unique_ids": len({r["id"] for r in rr}),
        "sources": dict(Counter((r.get("source", "unknown") for r in rr))),
        "normalized_texts": len({c.normalized(r["text"]) for r in rr}),
        "empty_texts": sum((not c.normalized(r["text"]) for r in rr)),
        "fields": {},
    }
    assert out["unique_ids"] == len(rr)
    for f in FIELDS:
        values = [r.get(f) for r in rr]
        known = [v for v in values if v is not None]
        eligible = [
            r
            for r in rr
            if r.get(f) is not None
            and (f in ("gate", "legal") or r.get("gate") is True)
        ]
        out["fields"][f] = {
            "annotated": len(known),
            "missing": len(rr) - len(known),
            "observed_empty": sum((v == [] for v in known)),
            "scorable_rows": len(eligible),
            "positive_support": dict(
                Counter(
                    (
                        str(x)
                        for r in eligible
                        for x in (r[f] if isinstance(r[f], list) else [r[f]])
                    )
                )
            ),
        }
    return out


def main():
    paths = list((c.ROOT / "data/processed").glob("*.parquet"))
    source_paths = [
        Path(__file__).resolve(),
        Path(c.__file__).resolve(),
        c.CORE / "kb/full/records.jsonl",
        c.CORE / "kb/small/records.jsonl",
    ]
    source_paths += list((c.ROOT / "data/mappings").glob("*"))
    source_paths += [
        c.ROOT / s
        for s in [
            "scripts/make_splits.py",
            "scripts/make_botox_splits.py",
            "scripts/make_eval_subsets.py",
            "scripts/make_hatecheck_subset.py",
            "analysis/data/score_distribution.py",
            "config/taxonomy.yaml",
        ]
    ]
    source_paths += list((c.ROOT / "src/hsrag/data").glob("*.py"))
    for a in AUDITS.values():
        source_paths += [a / "plan.json", a / "heldout_texts.jsonl"]
    source_paths += [
        c.ROOT / "experiments/astra/english_kb_cleanup_20260914_v1/policy.json",
        c.ROOT / "experiments/astra/german_kb_comparison_20260913_v1/policy.json",
    ]
    c.save(
        BASE / "plan.json",
        {
            "created_utc": c.utc(),
            "status": "frozen_before_census",
            "inputs": {c.rel(p): c.sha(p) for p in paths + source_paths if p.is_file()},
            "scope": "All local processed parquets: counts, annotation support, split membership and exact-text overlaps. No new prediction scores.",
            "exposure": "A documented minimum footprint from existing live JSONL files; missing output is NOT evidence of untouched data. Record ID exposure without accuracy scoring.",
            "audit_reuse": "Verify old held-out parquet hashes and generated roster hashes, then check file/ID coverage. Reuses prior near-duplicate policy; does not redo embeddings or certify semantic independence.",
            "normalization": "Original lower/whitespace and stronger NFKC/casefold/whitespace reported separately.",
            "no_external_activity": True,
        },
    )
    allrows = {p.stem: records(p) for p in paths}
    stats = {k: census(v) for k, v in allrows.items()}
    parent_sets = {k: {str(r["id"]) for r in allrows[k]} for k in PARENTS}
    membership = {}
    for name, rr in allrows.items():
        ids = {str(r["id"]) for r in rr}
        membership[name] = {p: len(ids & s) for p, s in parent_sets.items() if ids & s}
        if name not in PARENTS and name != "en_hatecheck_eval":
            assert sum(membership[name].values()) == len(ids), name
    overlaps = {}
    for method, fun in [("original", oldnorm), ("nfkc_casefold", c.normalized)]:
        sets = {
            k: {fun(r["text"]) for r in allrows[k] if fun(r["text"])}
            for k in PARENTS + ["en_hatecheck_eval"]
        }
        overlaps[method] = {
            a + " / " + b: len(sets[a] & sets[b])
            for a, b in combinations(sets, 2)
            if sets[a] & sets[b]
        }
    audit = {}
    for lang, base in AUDITS.items():
        plan = c.read(base / "plan.json")
        held = base / "heldout_texts.jsonl"
        assert c.sha(held) == plan["generated_hashes"][c.rel(held)]
        refs = {
            (r["file"], str(r["id"])) for row in c.rows(held) for r in row["references"]
        }
        check = {}
        for name, rr in allrows.items():
            if not name.startswith(lang + "_") or name.endswith("_train"):
                continue
            key = f"data/processed/{name}.parquet"
            if key not in plan["input_hashes"]:
                continue
            assert c.sha(c.ROOT / key) == plan["input_hashes"][key]
            missing = [str(r["id"]) for r in rr if (key, str(r["id"])) not in refs]
            nonempty = [
                str(r["id"])
                for r in rr
                if str(r["id"]) in set(missing) and c.normalized(r["text"])
            ]
            assert not nonempty, (name, nonempty[:3])
            check[name] = {
                "rows": len(rr),
                "covered_references": len(rr) - len(missing),
                "missing_empty_ids": missing,
                "nonempty_missing": len(nonempty),
            }
        audit[lang] = {
            "old_plan_sha256": c.sha(base / "plan.json"),
            "heldout_sha256": c.sha(held),
            "datasets": check,
        }
    kb = {}
    hold = {
        k: {c.normalized(r["text"]) for r in allrows[k] if c.normalized(r["text"])}
        for k in PARENTS
        if not k.endswith("_train")
    }
    hold["en_hatecheck_eval"] = {
        c.normalized(r["text"]) for r in allrows["en_hatecheck_eval"]
    }
    for pool in ("full", "small"):
        rr = c.rows(c.CORE / f"kb/{pool}/records.jsonl")
        examples = [r for r in rr if r["kind"] == "example"]
        bylang = {}
        for lang in ("en", "de"):
            ex = [r for r in examples if r["lang"] == lang]
            con = []
            for r in ex:
                con.append(
                    {
                        "id": r["id"],
                        "text": r["text"],
                        "source": r["source"],
                        **r["meta"],
                    }
                )
            st = census(con)
            texts = {c.normalized(r["text"]) for r in ex if c.normalized(r["text"])}
            st["exact_heldout_overlap"] = {
                k: len(texts & v) for k, v in hold.items() if k.startswith(lang + "_")
            }
            bylang[lang] = st
        kb[pool] = {
            "total_records": len(rr),
            "abstract_records": len(rr) - len(examples),
            "languages": bylang,
        }
    exposure = {}
    seen = defaultdict(set)
    files = []
    exposed_sets = {
        name: {str(r["id"]) for r in allrows[name]}
        for name in ["en_test", "de_test", "de_legal_test", "en_hatecheck_eval"]
    }
    for p in sorted((c.ROOT / "experiments/results").glob("*live.jsonl")):
        local = Counter()
        n = 0
        with p.open(encoding="utf8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = __import__("json").loads(line)
                n += 1
                rid = r.get("id", r.get("item_id"))
                if rid is not None:
                    rid = str(rid)
                    for name, ids in exposed_sets.items():
                        if rid in ids:
                            seen[name].add(rid)
                            local[name] += 1
        files.append(
            {"path": c.rel(p), "sha256": c.sha(p), "rows": n, "id_matches": dict(local)}
        )
    for name in ["en_test", "de_test", "de_legal_test", "en_hatecheck_eval"]:
        exposure[name] = {
            "rows": len(allrows[name]),
            "known_prior_live_id_count": len(seen[name]),
            "known_prior_ids": sorted(seen[name]),
            "remaining_status": "unknown / not declared untouched",
        }
    result = {
        "created_utc": c.utc(),
        "status": "local_census_and_provenance_pass",
        "plan_sha256": c.sha(BASE / "plan.json"),
        "datasets": stats,
        "parent_membership": membership,
        "nonzero_exact_split_overlaps": overlaps,
        "old_audit_coverage": audit,
        "kb": kb,
        "minimum_documented_exposure": exposure,
        "historical_live_files": files,
        "limits": [
            "No licensing certification or dataset re-download",
            "No near-neighbour model rerun",
            "No new score or test outcome inspection",
            "All earlier DEV and benchmark exposure must remain disclosed",
        ],
    }
    c.save(BASE / "data_provenance.json", result)
    c.save(
        BASE / "complete.json",
        {
            "status": "complete",
            "plan_sha256": c.sha(BASE / "plan.json"),
            "data_sha256": c.sha(BASE / "data_provenance.json"),
        },
    )
    print(
        "Census, prior held-out coverage and historical exposure inventory complete.",
        flush=True,
    )
    for k in PARENTS:
        print(k, stats[k]["rows"], stats[k]["sources"])
    print("Exposure:", {k: v["known_prior_live_id_count"] for k, v in exposure.items()})


if __name__ == "__main__":
    with c.network_guard():
        main()
