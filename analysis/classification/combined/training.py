from collections import Counter
from analysis.classification.combined import prepare as a
from analysis.classification.combined.native_schema import native_row


def training_labels(target, spec, head):
    if spec["kind"] in ("binary", "ordinal", "multiclass"):
        a.require(
            type(target) is int and 0 <= target < len(spec["labels"]),
            "Invalid categorical training target",
        )
        positive = {spec["labels"][target]}
    else:
        a.require(
            spec["kind"] == "multilabel"
            and isinstance(target, list)
            and len(target) == len(spec["labels"]),
            "Invalid multilabel training shape",
        )
        a.require(all(v in (0, 1) for v in target), "Nonbinary training indicator")
        positive = {lab for lab, v in zip(spec["labels"], target) if v}
    if head == "hate":
        a.require(positive <= {"false", "true"}, "Unknown gate class")
        positive = {"hate" if x == "true" else "not_hate" for x in positive}
    return positive


def run():
    a.verify(a.read(a.BASE / "analysis_contract.json")["sources"])
    banks = {
        n: {
            r["id"]: r
            for r in a.rows(a.CORE / "kb" / n / "records.jsonl")
            if r["kind"] == "example"
        }
        for n in ("small", "full")
    }
    a.require(
        set(banks["small"]) <= set(banks["full"])
        and all(r == banks["full"][k] for k, r in banks["small"].items()),
        "Shared KB source changed",
    )
    en = {
        n: {k: r for k, r in b.items() if r["lang"] == "en"} for n, b in banks.items()
    }
    eligible = {
        n: {
            h: {k for k, r in b.items() if a.labels(r["meta"], h) is not None}
            for h in a.g.HEADS
        }
        for n, b in en.items()
    }
    inventory = [
        dict(
            name="Qwen_" + n,
            rows=len(b),
            languages=dict(Counter(r["lang"] for r in b.values())),
            english_head_counts={h: len(eligible[n][h]) for h in a.g.HEADS},
            distinct_normalized_texts=len(
                {a.normalized_sha(r["text"]) for r in b.values()}
            ),
        )
        for n, b in banks.items()
    ]
    native = []
    observations = []
    for arm, n in [("kb", "small"), ("full", "full")]:
        for h in a.g.HEADS:
            parts = {}
            for split in ("train", "val"):
                p = a.CORE / "encoder_data" / arm / (h + "_" + split + ".parquet")
                raw = a.parquet(p)
                parts[split] = [native_row(r, en[n], h) for r in raw]
                observations.append(
                    dict(
                        path=a.rel(p),
                        rows=len(raw),
                        ids_with_added_prefix=sum(
                            r["id"] != s["id"] for r, s in zip(raw, parts[split])
                        ),
                        authoritative_gates_joined=sum("gate" not in r for r in raw),
                        head=h,
                        arm=arm,
                    )
                )
            joined = {r["id"]: r for rr in parts.values() for r in rr}
            a.require(
                len(joined) == sum(map(len, parts.values()))
                and set(joined) == eligible[n][h],
                "Native membership differs",
            )
            a.require(
                all(
                    a.labels(r, h) == a.labels(en[n][k]["meta"], h)
                    for k, r in joined.items()
                ),
                "Native labels differ",
            )
            native.append(
                dict(
                    arm=arm,
                    head=h,
                    gradient_train=len(parts["train"]),
                    internal_validation=len(parts["val"]),
                    all_permitted=len(joined),
                    kb_membership_and_gold_equal=True,
                    train_ids=sorted(r["id"] for r in parts["train"]),
                    validation_ids=sorted(r["id"] for r in parts["val"]),
                )
            )
    pools = {}
    stats = []
    for n in a.g.POOLS:
        d = a.read(a.g.PACKAGE / "data/train" / f"{n}.json")
        a.require(
            d == a.read(a.g.LINEAR / "data/train" / f"{n}.json"),
            "GPU and linear TRAIN sources differ",
        )
        ids = set(d["gate_ids"])
        a.require(
            len(ids) == len(d["gate_ids"]) == len(d["texts"])
            and ids <= set(en["full"]),
            "Invalid pool IDs",
        )
        a.require(
            all(t == en["full"][k]["text"] for k, t in zip(d["gate_ids"], d["texts"])),
            "Pool text differs",
        )
        counts = {}
        support = {}
        for h, spec in d["heads"].items():
            rr = spec["rows"]
            hid = {r["id"] for r in rr}
            a.require(
                len(hid) == len(rr) and hid == ids & eligible["full"][h],
                "Head mask differs",
            )
            a.require(
                all(
                    training_labels(r["target"], spec, h)
                    == a.labels(en["full"][r["id"]]["meta"], h)
                    for r in rr
                ),
                "Training labels differ",
            )
            counts[h] = len(rr)
            support[h] = dict(
                Counter(lab for k in hid for lab in a.labels(en["full"][k]["meta"], h))
            )
        pools[n] = ids
        shared = ids & set(en["small"])
        stats.append(
            dict(
                pool=n,
                total_texts=len(ids),
                annotated_heads=counts,
                positive_support=support,
                curated_overlap=len(shared),
                curated_shared_ids=sorted(shared),
                source_counts=dict(Counter(en["full"][k]["source"] for k in ids)),
                training_ids=sorted(ids),
                same_GPU_and_linear_data=True,
            )
        )
    nested = []
    between = []
    for s in (42, 43, 44):
        for x, y in [("compact", "1000"), ("1000", "5000")]:
            l, r = f"uniform_{x}_s{s}", f"uniform_{y}_s{s}"
            a.require(pools[l] <= pools[r], "Pool nesting differs")
            nested.append(
                dict(
                    left=l,
                    right=r,
                    shared=len(pools[l] & pools[r]),
                    left_subset_of_right=True,
                )
            )
    for n in ("compact", "1000", "5000"):
        for x, y in ((42, 43), (42, 44), (43, 44)):
            l, r = f"uniform_{n}_s{x}", f"uniform_{n}_s{y}"
            between.append(dict(left=l, right=r, shared=len(pools[l] & pools[r])))
    evaluation = list(a.rows(a.ROSTER)) + list(a.rows(a.MATCHED / "gold_rosters.jsonl"))
    for n in a.read(a.g.LINEAR / "data/endpoints.json")["core"]:
        evaluation += a.parquet(a.ROOT / f"data/processed/{n}.parquet")
    et = {a.normalized_sha(r["text"]) for r in evaluation}
    leak = {
        n: len(et & {a.normalized_sha(r["text"]) for r in b.values()})
        for n, b in banks.items()
    }
    a.require(not any(leak.values()), "Exact normalized evaluation/training overlap")
    gate = {
        arm: set(
            next(
                r["train_ids"]
                for r in native
                if r["arm"] == arm and r["head"] == "hate"
            )
        )
        for arm in ("kb", "full")
    }
    overlap = [
        dict(
            arm=r["arm"],
            head=r["head"],
            fine_validation=len(r["validation_ids"]),
            overlaps_gate_training=len(set(r["validation_ids"]) & gate[r["arm"]]),
        )
        for r in native
        if r["head"] != "hate"
    ]
    a.save(
        a.BASE / "training_information.json",
        dict(
            candidate_banks=inventory,
            native_partitions=native,
            uniform_pools=stats,
            nested_chains=nested,
            same_budget_pair_overlap=between,
            native_fine_validation_in_gate_training=overlap,
            exact_normalized_evaluation_overlap=leak,
            meaning="Total texts, observed labels, fitting rows, selection information and retrieved slots are separate budgets. No new fitting or retrieval.",
        ),
    )
    a.save(
        a.BASE / "native_format_review.json",
        dict(
            status="native_head_schema_and_categorical_encoding_verified",
            tables=observations,
            sources=a.bind(
                [
                    __file__,
                    a.ROOT / "analysis/classification/combined/native_schema.py",
                    a.BASE / "native_schema_amendment.json",
                    a.BASE / "training_schema_amendment_v2.json",
                ]
            ),
            note="Fine head tables omit the already-filtered gate; its true value was verified on the same exact-text KB record. Ordinal severity uses an integer class index. Neither correction changes the evidence.",
        ),
    )
    a.save(
        a.BASE / "training_complete.json",
        dict(
            status="training_membership_annotation_and_budget_alignment_passed",
            created_utc=a.utc(),
            pools=len(stats),
            native_heads=len(native),
            new_fits=0,
            producer=a.rel(__file__),
            outputs=a.bind(
                [
                    a.BASE / "training_information.json",
                    a.BASE / "native_format_review.json",
                ]
            ),
        ),
    )
    print(
        "Verified nine equal-budget pools and eight native head partitions.", flush=True
    )


if __name__ == "__main__":
    with a.g.offline():
        run()
