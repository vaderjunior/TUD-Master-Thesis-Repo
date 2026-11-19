from collections import Counter
from pathlib import Path
from analysis.feedback.insertion import archive as r
from analysis.feedback.insertion import common as c


def main():
    dest = r.DEST / "mechanism_check_v2"
    target = dest / "complete.json"
    if target.exists():
        d = r.read(target)
        r.verify(d["inputs"])
        r.verify(d["outputs"])
        print("Corrected mechanism summary verified.")
        return
    old = r.DEST / "mechanism_check_v1"
    done = r.read(old / "complete.json")
    r.verify(done["inputs"])
    r.verify(done["outputs"])
    material = c.CORE / "kb/full/records.jsonl"
    pools = r.read(c.SELECTION / "actual_pools.json")
    source = r.read(c.SELECTION / "gate_only_records.json")
    records = list(r.rows(material))
    inventory = Counter((v["kind"], v["lang"]) for v in records)
    english = sum(v["kind"] == "example" and v["lang"] == "en" for v in records)
    r.require(
        english > 50000 and sum(inventory.values()) == 71131,
        "Corpus inventory unexpectedly differs",
    )
    value = r.read(old / "results.json")
    value["original_inventory"] = dict(
        records=len(records),
        english_examples=english,
        kind_language_counts={k + "/" + l: n for (k, l), n in inventory.items()},
    )
    value["annotation"] = dict(
        prior_v1_count_error="Only original_inventory.english_examples was incorrectly read from meta. Canonical records store kind/lang at top level. All RRF, message, exposure, insertion, score and failure checks are unaffected.",
        source=r.rel(old / "results.json"),
        source_sha256=r.sha(old / "results.json"),
        new_model_calls=0,
    )
    value["source_pool_inventory"] = dict(
        distinct_sources=pools["distinct_sources"],
        source_prefix_counts=dict(Counter(k.split("-")[0] for k in source)),
        added_fraction_of_english_pool=40 / (english + 40),
    )

    roster = {v["key"]: v for v in r.rows(c.BASE / "roster.jsonl")}
    metadata = {
        (v["key"], v["condition"]): v
        for v in r.rows(r.DEST / "diagnostics_v2/prompt_metadata.jsonl")
    }
    rows = {
        (v["key"], v["condition"], v["repeat"]): v
        for v in r.rows(r.REVIEW / "predictions.jsonl")
    }
    proofs = {}
    for truth, pop, ep in [
        (False, "benign", "benign_false_positive_rate"),
        (True, "hateful", "hateful_false_negative_rate"),
    ]:
        ids = r.read(r.DEST / "full/primary.json")["primary"][ep + "/primary"][
            "common_ids"
        ]
        cnt = Counter()
        for key in ids:
            r.require(
                roster[key]["gold"]["hate"] == truth, "Primary population differs"
            )
            for seed in r.SEEDS:
                cond = f"error_balanced_s{seed}"
                if not metadata[key, cond]["message_equals_reference"]:
                    continue
                for rep in range(3):
                    a = rows[key, "reference", rep]["result"]["hate"]
                    b = rows[key, cond, rep]["result"]["hate"]
                    cnt["pairs"] += 1
                    cnt["fixed"] += a != truth and b == truth
                    cnt["harmed"] += a == truth and b != truth
        expected = r.read(r.DEST / "diagnostics_v2/full_transitions.json")[
            pop + "/error_balanced/primary_common"
        ]["categories"]["no_delivery_identical"]
        r.require(
            all(cnt[k] == expected[k] for k in ["pairs", "fixed", "harmed"]),
            "Identical-message transition reconstruction differs",
        )
        proofs[pop] = dict(cnt)
    value["independent_identical_message_proof"] = proofs
    path = dest / "results.json"
    r.save(path, value)
    r.save(
        target,
        dict(
            status="corrected_inventory_and_all_prior_mechanism_checks_verified",
            created_utc=r.utc(),
            inputs={
                r.rel(p): r.sha(p)
                for p in [
                    Path(__file__),
                    old / "complete.json",
                    old / "results.json",
                    material,
                    c.SELECTION / "actual_pools.json",
                    c.SELECTION / "gate_only_records.json",
                ]
            },
            outputs={r.rel(path): r.sha(path)},
            new_model_calls=0,
        ),
    )
    print(value["original_inventory"])
    print(value["source_pool_inventory"])
    print(proofs)


if __name__ == "__main__":
    main()
