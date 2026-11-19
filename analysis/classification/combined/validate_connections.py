from collections import Counter
from analysis.classification.combined import prepare as a
from analysis.classification.combined.training import training_labels


def run():
    a.verify(a.read(a.BASE / "analysis_contract.json")["sources"])
    cpu = a.ROOT / "experiments/astra/sq2_cpu_baselines_20261011_v1"
    tune = a.ROOT / "experiments/astra/compact_roberta_tuning_20261010_v1"
    short = a.ROOT / "experiments/astra/compact_roberta_short_20261012_v1"
    compact = a.ROOT / "experiments/astra/compact_bge_20261008_v1"
    stage = a.ROOT / "experiments/astra/matched_type_pools_20261011_v1"
    sources = [
        __file__,
        cpu / "data/evaluation.json",
        cpu / "data/scopes.json",
        cpu / "data/full_heads.json",
        cpu / "plan.json",
        tune / "data/train.json",
        tune / "data/inference.json",
        tune / "data/scopes.json",
        tune / "plan.json",
        short / "plan.json",
        compact / "train.json",
        stage / "data/inventory.json",
        stage / "data/endpoints.json",
        stage / "plan.json",
        a.ROOT
        / "experiments/astra/roberta_short_review_20261020_v1/analysis_plan.json",
        a.ROOT
        / "experiments/astra/roberta_short_review_20261020_v1/selection_review.json",
    ]
    inventory = a.read(stage / "data/inventory.json")
    sources += [stage / "data" / (n + ".json") for n in inventory]
    before = a.bind(sources)
    a.save(
        a.BASE / "connections_scope_v2.json",
        dict(
            status="identity_bridge_scope_recorded",
            created_utc=a.utc(),
            sources=before,
            scope="Verify compact training, ten matched-type unions, native full-data CPU partitions, input order and original core pooled fine annotations. No performance calculation.",
        ),
    )
    full = {
        r["id"]: r
        for r in a.rows(a.CORE / "kb/full/records.jsonl")
        if r["kind"] == "example" and r["lang"] == "en"
    }
    small = {
        r["id"]: r
        for r in a.rows(a.CORE / "kb/small/records.jsonl")
        if r["kind"] == "example" and r["lang"] == "en"
    }
    cd = a.read(compact / "train.json")
    td = a.read(tune / "data/train.json")
    a.require(
        set(cd["gate_ids"]) == set(td["gate_ids"]) == set(small),
        "Curated TRAIN membership changed",
    )
    for h in a.g.HEADS:
        for data in (cd, td):
            spec = data["heads"][h]
            rr = spec["rows"]
            expected = {
                k for k, v in small.items() if a.labels(v["meta"], h) is not None
            }
            a.require(
                {r["id"] for r in rr} == expected and len(rr) == len(expected),
                "Compact head mask differs",
            )
            a.require(
                all(
                    training_labels(r["target"], spec, h)
                    == a.labels(small[r["id"]]["meta"], h)
                    for r in rr
                ),
                "Compact training labels differ",
            )
    master = a.read(a.g.LINEAR / "data/inference.json")
    scopes = a.read(a.g.LINEAR / "data/scopes.json")
    same = []
    for name, p in [
        ("CPU full/compact", cpu / "data/evaluation.json"),
        ("compact neural tuning", tune / "data/inference.json"),
    ]:
        rr = a.read(p)
        a.require(
            [(r["key"], r["text"]) for r in rr]
            == [(r["key"], r["text"]) for r in master],
            "Input order or text differs",
        )
        sp = p.parent / "scopes.json"
        a.require(a.read(sp) == scopes, "Evaluation scopes differ")
        same.append(
            dict(
                study=name,
                rows=len(rr),
                same_order_and_exact_text=True,
                scope_memberships_identical=True,
            )
        )
    native = a.read(a.BASE / "training_information.json")["native_partitions"]
    fh = a.read(cpu / "data/full_heads.json")
    for h, spec in fh.items():
        ref = next(v for v in native if v["arm"] == "full" and v["head"] == h)
        for split, saved in [("train", "train_ids"), ("val", "validation_ids")]:
            rr = spec[split]
            a.require(
                {r["id"] for r in rr} == set(ref[saved]),
                "CPU native fitting/validation rows differ",
            )
            a.require(
                all(
                    training_labels(r["target"], spec, h)
                    == a.labels(full[r["id"]]["meta"], h)
                    for r in rr
                ),
                "CPU native annotation differs",
            )
    stage_rows = []
    for name, expected in inventory.items():
        v = a.read(stage / "data" / (name + ".json"))
        spec = v["heads"]["hate_type"]
        rr = spec["rows"]
        ids = {r["id"] for r in rr}
        a.require(
            len(ids) == len(rr) == 67 and ids <= set(full),
            "Matched-type roster differs",
        )
        a.require(
            all(
                training_labels(r["target"], spec, "hate_type")
                == a.labels(full[r["id"]]["meta"], "hate_type")
                for r in rr
            ),
            "Matched-type label differs",
        )
        a.require(
            all(text == full[r["id"]]["text"] for r, text in zip(rr, v["texts"])),
            "Matched-type exact text differs",
        )
        union = ids | set(small)
        overlap = ids & set(small)
        a.require(
            len(union) == expected["union_training_ids"]
            and len(overlap) == expected["gate_overlap_groups"],
            "Matched-type whole-budget accounting differs",
        )
        stage_rows.append(
            dict(
                pool=name,
                type_annotations=67,
                gate_annotations=len(small),
                union_texts=len(union),
                shared_gate_type_ids=len(overlap),
                type_ids=sorted(ids),
            )
        )
    ep = a.read(a.g.LINEAR / "data/endpoints.json")
    for name, wrapper in a.read(stage / "data/endpoints.json").items():
        scope, endpoint_name = name.split("/", 1)
        expected = ep[scope][endpoint_name]
        a.require(
            wrapper["endpoint"] == expected, "Matched-type endpoint contract differs"
        )
        a.require(
            wrapper["keys"]
            == [
                k if k.startswith("en/") else "en/" + k
                for k in expected["scorable_ids"]
            ],
            "Matched-type canonical keys differ",
        )

    sh = a.read(short / "plan.json")
    short_bindings = sh.get("inputs", {})
    wanted = {
        a.rel(p)
        for p in [
            tune / "plan.json",
            tune / "complete.json",
            tune / "data/train.json",
            tune / "data/inference.json",
            tune / "data/scopes.json",
            compact / "train.json",
            compact / "plan.json",
        ]
    }
    relevant = {k: v for k, v in short_bindings.items() if k in wanted}
    a.require(relevant, "No compact-source provenance in shorter-training plan")
    a.verify(relevant)
    core = []
    by_phase = {}
    for name in ep["core"]:
        rr = a.parquet(a.ROOT / f"data/processed/{name}.parquet")
        by_phase[name] = rr
        core += rr
    broad = a.index(list(a.rows(a.ROSTER)))
    pooled = []
    for h in ("target_group", "severity", "hate_type"):
        rr = [r for r in core if a.labels(r, h) is not None]
        ids = [a.key(r) for r in rr]
        pooled.append(
            dict(
                head=h,
                eligible=len(rr),
                phase_counts={
                    n: sum(a.labels(r, h) is not None for r in rows)
                    for n, rows in by_phase.items()
                },
                ids=ids,
                in_broader_ids=sorted(set(ids) & set(broad)),
                broader_overlap=len(set(ids) & set(broad)),
            )
        )
    a.verify(before)
    a.save(
        a.BASE / "earlier_study_connections.json",
        dict(
            status="earlier_study_identity_bridges_verified",
            sources=before,
            evaluation_bridges=same,
            curated_train_identical=True,
            curated_head_counts={h: len(cd["heads"][h]["rows"]) for h in a.g.HEADS},
            full_CPU_native_heads_identical=True,
            matched_type_pools=stage_rows,
            shorter_existing_source_bindings=relevant,
            pooled_core_annotations=pooled,
            limits="Identical evaluation inputs or reused probabilities are not independent replication. No scoring, refitting or inference.",
        ),
    )
    print(
        "Verified compact/CPU input bridges, ten type pools and original pooled fine-field membership.",
        flush=True,
    )


if __name__ == "__main__":
    with a.g.offline():
        run()
