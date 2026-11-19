from src.hsrag.paths import PROJECT_ROOT
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = PROJECT_ROOT
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
OUT = ROOT / "Thesis_Log/astra_result/evidence/core_encoder_followup_20260915_v5a.json"
HASHES = {}


def sha(p):
    value = hashlib.sha256(p.read_bytes()).hexdigest()
    HASHES[p.relative_to(ROOT).as_posix()] = value
    return value


def read(p):
    sha(p)
    return json.loads(p.read_text(encoding="utf8"))


def rows(p):
    sha(p)
    return list(map(json.loads, p.read_text(encoding="utf8").splitlines()))


def main():
    assert not OUT.exists(), "Preserve existing follow-up"
    sha(Path(__file__))
    audit = read(
        ROOT / "Thesis_Log/astra_result/evidence/core_encoder_results_20260915_v5.json"
    )
    batch = read(CORE / "encoders_v5/batch_receipts/1789481983165297000.json")
    records = []
    diagnostics = []
    for seed in range(3):
        old = {
            x["item_id"]: x
            for x in rows(
                ROOT / f"experiments/results/encoder_kb_seed{seed}_live.jsonl"
            )
            if x["encoder_meta"]["subset"].startswith("en_dev_eval")
        }
        for head in ("target_group", "hate_type", "severity"):
            e = next(
                e
                for e in batch["jobs"]
                if e["mode"] == "reuse" and e["job"] == f"kb/{head}/seed{seed}"
            )
            raw = rows(ROOT / e["attempt"] / "dev_head_logits.jsonl")
            tally = Counter()
            for x in raw:
                p = old[x["key"].split("/", 1)[1]]["run_predictions"][0][head]
                if head == "severity":
                    labels = [k for k, v in p.items() if v == max(p.values())]
                    tally[
                        (
                            "ambiguous_rounded"
                            if len(labels) > 1
                            else (
                                "different"
                                if x["labels"][x["decision"]] != labels[0]
                                else "same"
                            )
                        )
                    ] += 1
                else:
                    for lab, d in zip(x["labels"], x["decision"]):
                        tally[
                            (
                                "ambiguous_rounded"
                                if p[lab] == 0.5
                                else "different" if int(p[lab] > 0.5) != d else "same"
                            )
                        ] += 1
            assert (
                tally["different"] == 0
            ), "Unexpected definite reused-head decision difference"
            records.append(
                {
                    "job": e["job"],
                    "rows": len(raw),
                    "unit": (
                        "row argmax" if head == "severity" else "row-label threshold"
                    ),
                    **dict(tally),
                }
            )
        for arm in ("full", "kb"):
            assembled = rows(
                CORE / f"encoder_results_v5/en_{arm}_seed{seed}_assembled.jsonl"
            )
            for subset, head in [
                ("en_dev_eval_targets", "target_group"),
                ("en_dev_eval_sq3_types", "hate_type"),
                ("en_dev_eval_severity", "severity"),
            ]:
                selected = [
                    x
                    for x in assembled
                    if any(s["subset"] == subset for s in x["scopes"])
                ]
                d = {
                    "arm": arm,
                    "seed": seed,
                    "subset": subset,
                    "n": len(selected),
                    "gate_negative": sum(not x["result"]["hate"] for x in selected),
                }
                if head != "severity":
                    d.update(
                        raw_nonempty=sum(bool(x["raw_fields"][head]) for x in selected),
                        assembled_nonempty=sum(
                            bool(x["result"][head]) for x in selected
                        ),
                    )
                if arm == "kb":
                    d["assembled_head_changed_vs_original"] = sum(
                        x["result"][head] != old[x["item_id"]]["result"][head]
                        for x in selected
                    )
                diagnostics.append(d)
    gender = audit["english_dev"]["kb/en_dev_eval_targets"]["per_seed"][2]
    assert (
        gender["old"]["per_label"]["gender"]["tp"]
        == gender["new"]["per_label"]["gender"]["tp"]
        == 0
    )
    assert (
        gender["new"]["per_label"]["gender"]["fp"]
        == gender["old"]["per_label"]["gender"]["fp"] + 1
    )
    prior = read(
        ROOT / "Thesis_Log/astra_result/evidence/core_encoder_followup_20260915_v5.json"
    )
    report = {
        "supersedes_interpretation_of": "core_encoder_followup_20260915_v5.json; preliminary wording incorrectly equated unchanged target F1 with identical predictions. Counts/main audit remain correct; correction follows explicit per-label count verification.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "scope": "Saved DEV predictions only; no neural model, training, API or TEST scoring.",
        "reuse_decision_comparison": records,
        "assembly_diagnostics": diagnostics,
        "interpretation": "No definite decision differences in reused fine heads versus historical rounded probabilities across all 970 DEV texts per seed/head. Five comparisons are ambiguous because historical probabilities were rounded (three binary thresholds and two severity ties); this is not proof of exact historical logit identity. Compact type predictions are identical item by item. Compact targets differ on one seed-2 item: an extra gender false positive with zero gender true positives leaves F1 unchanged. Thus zero bootstrap F1-delta intervals do not prove identical predictions or zero population uncertainty. Compact type output is almost always empty before gating. Full type gate misses rise 107/135/149 to 134/145/164; this demonstrates suppression, not an isolated causal attribution of the entire score change.",
        "main_audit_status": audit["status"],
        "verified_hashes": dict(sorted(HASHES.items())),
    }
    with OUT.open("x", encoding="utf8", newline="\n") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(
        json.dumps(
            {
                "evidence": str(OUT),
                "sha256": sha(OUT),
                "reuse_checks": records,
                "assembly": diagnostics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
