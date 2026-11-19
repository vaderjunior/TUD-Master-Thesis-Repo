from collections import Counter, defaultdict
from pathlib import Path
from analysis.feedback.rounds import contexts_inputs as a
from analysis.feedback.rounds.validate_contexts_context import check
from analysis.feedback.rounds.validate_contexts_context import equal


def main():
    v = a.read(a.OUT / "details.json")
    for p, digest in v["inputs"].items():
        check(a.sha(a.ROOT / p) == digest, "Detail input changed")
    deliveries = [
        d for d in a.rows(a.OUT / "deliveries.jsonl") if d["role"] == "evaluation"
    ]
    selections = a.read(a.OUT / "selections.json")
    candidates = list(a.rows(a.OUT / "candidate_selection_ledger.jsonl"))
    members = list(a.rows(a.OUT / "pool_memberships.jsonl"))
    contexts = {r["key"]: r for r in a.rows(a.OUT / "query_contexts.jsonl")}
    catalog = a.read(a.OUT / "source_catalog.json")
    roster = list(a.rows(a.STUDY / "evaluation.jsonl"))
    counts, first, by_pool, pools, index = (
        Counter(),
        {},
        defaultdict(Counter),
        defaultdict(set),
        {},
    )
    global_count, popcount, polcount = (
        Counter(),
        defaultdict(Counter),
        defaultdict(Counter),
    )
    for m in members:
        pools[m["round"], m["stream"], m["policy"]].add(m["source_id"])
    for d in deliveries:
        k = d["stream"], d["policy"], d["source_id"]
        counts[*k, d["round"]] += 1
        first[k] = min(first.get(k, d["round"]), d["round"])
        by_pool[d["round"], d["stream"], d["policy"]][d["source_id"]] += 1
        index[d["round"], d["stream"], d["policy"], d["key"]] = d
        global_count[d["source_id"]] += 1
        popcount[d["source_id"]][d["population"]] += 1
        polcount[d["source_id"]][d["policy"]] += 1
    for r in v["correction_intake"]:
        k = r["stream"], r["policy"], r["source_id"]
        rnd = r["added_round"]
        chosen = next(
            s for s in selections if (s["round"], s["stream"]) == (rnd, r["stream"])
        )
        check(
            r["source_id"] in chosen["additions"][r["policy"]], "Actual source intake"
        )
        per_round = {str(i): counts[*k, i] for i in range(rnd, 5)}
        equal(
            r,
            dict(
                stream=k[0],
                policy=k[1],
                source_id=k[2],
                added_round=rnd,
                gold_hate=catalog[k[2]]["record"]["meta"]["gate"],
                first_shown_round=first.get(k),
                assignments_by_round=per_round,
                total_assignments=sum(per_round.values()),
                immediate_assignments=per_round[str(rnd)],
            ),
            "Every intake and first-use field",
        )
    check(len(v["correction_intake"]) == 360, "Complete intake ledger")
    for r in v["per_pool_use"]:
        k = r["round"], r["stream"], r["policy"]
        c = by_pool[k]
        freq = sorted(c.values(), reverse=True)
        birth = {
            x["source_id"]: x["added_round"]
            for x in v["correction_intake"]
            if (x["stream"], x["policy"]) == k[1:]
        }
        equal(
            r,
            dict(
                round=k[0],
                stream=k[1],
                policy=k[2],
                pool_size=len(pools[k]),
                used=len(c),
                unused=sorted(pools[k] - set(c)),
                assignments=sum(c.values()),
                newest=sum(n for sid, n in c.items() if birth[sid] == k[0]),
                oldest=sum(n for sid, n in c.items() if birth[sid] == 1),
                top1_count=freq[0],
                top5_count=sum(freq[:5]),
                effective_sources=sum(freq) ** 2 / sum(n * n for n in freq),
            ),
            "Per-pool usage",
        )
    for r in v["cross_stream_overlap"]:
        rnd, p, s, t, pop = [
            r[k]
            for k in ["round", "policy", "left_stream", "right_stream", "population"]
        ]
        x, y = pools[rnd, s, p], pools[rnd, t, p]
        keys = [q["key"] for q in roster if q["population"] == pop]
        equal(
            r,
            dict(
                round=rnd,
                policy=p,
                left_stream=s,
                right_stream=t,
                population=pop,
                intersection=len(x & y),
                union=len(x | y),
                jaccard=len(x & y) / len(x | y),
                inputs=len(keys),
                same_source=sum(
                    index[rnd, s, p, key]["source_id"]
                    == index[rnd, t, p, key]["source_id"]
                    for key in keys
                ),
                same_prompt=sum(
                    index[rnd, s, p, key]["messages_sha256"]
                    == index[rnd, t, p, key]["messages_sha256"]
                    for key in keys
                ),
            ),
            "Cross-stream overlap",
        )
    for r in v["selected_error_history"] + v["source_screening"]:
        subset = [
            x
            for x in candidates
            if x["round"] == r["round"]
            and x["gold_hate"] == r["gold_hate"]
            and (x["selected"][r["policy"]] if "policy" in r else x["common_valid"])
        ]
        equal(
            r["transition_counts"] if "policy" in r else r["transitions"],
            dict(Counter(x["error_transition"] for x in subset)),
            "Source transitions",
        )
        check(
            r.get("selected", r.get("eligible")) == len(subset)
            and r["initial_errors"]
            == sum(x["initial_majority"] != r["gold_hate"] for x in subset)
            and r["current_errors"]
            == sum(x["current_majority"] != r["gold_hate"] for x in subset),
            "Source selection error counts",
        )
    for pop, r in v["reference_contexts"].items():
        values = [contexts[q["key"]] for q in roster if q["population"] == pop]
        hits = [q["context"]["example_groups"][0][1] for q in values]
        fifth = [x[4] for x in hits]
        expected = dict(
            queries=len(values),
            fifth_hateful=sum(h["meta"]["gate"] for h in fifth),
            fifth_is_lowest_cosine=sum(q["fifth_is_lowest_cosine"] for q in values),
            fifth_cosine_rank=dict(
                Counter(str(q["fifth_cosine_rank"]) for q in values)
            ),
            ordinary_unique_ids=len({h["id"] for hs in hits for h in hs}),
            replaced_unique_ids=len({h["id"] for h in fifth}),
            replaced_sources=dict(Counter(h["meta"]["source"] for h in fifth)),
            removed_any_known=sum(
                any(h["meta"].get(f) is not None for f in a.FIELDS) for h in fifth
            ),
            fields={
                f: dict(
                    known_removed=sum(h["meta"].get(f) is not None for h in fifth),
                    known_nonempty_removed=sum(bool(h["meta"].get(f)) for h in fifth),
                    known_empty_removed=sum(
                        h["meta"].get(f) is not None and not h["meta"].get(f)
                        for h in fifth
                    ),
                    last_known_example_removed=sum(
                        hs[4]["meta"].get(f) is not None
                        and all(h["meta"].get(f) is None for h in hs[:4])
                        for hs in hits
                    ),
                )
                for f in a.FIELDS
            },
        )
        equal(r, expected, "Original fifth annotation coverage")
    check(
        v["total_selected_source_ids"] == len(catalog)
        and v["evaluation_delivered_source_ids"] == len(global_count)
        and v["never_evaluation_delivered_source_ids"]
        == sorted(set(catalog) - set(global_count)),
        "Unique selected/delivered sources",
    )
    for r, (sid, n) in zip(v["top_delivered_sources"], global_count.most_common(12)):
        equal(
            r,
            dict(
                source_id=sid,
                assignments=n,
                share=n / len(deliveries),
                selected_memberships=sum(
                    x["source_id"] == sid for x in v["correction_intake"]
                ),
                by_population=dict(popcount[sid]),
                by_policy=dict(polcount[sid]),
                gate=catalog[sid]["record"]["meta"]["gate"],
                text=catalog[sid]["record"]["text"],
            ),
            "Most-delivered source provenance",
        )
    equal(
        v["invalid_candidates"],
        [r for r in candidates if not r["common_valid"]],
        "All invalid source opportunities",
    )
    duplicates = []
    for d in deliveries:
        source = " ".join(catalog[d["source_id"]]["record"]["text"].split()).casefold()
        pos = [
            i + 1
            for i, h in enumerate(contexts[d["key"]]["context"]["example_groups"][0][1])
            if " ".join(h["text"].split()).casefold() == source
        ]
        if pos:
            duplicates.append(
                dict(
                    phase=d["phase"],
                    key=d["key"],
                    condition=d["condition"],
                    original_positions=pos,
                )
            )
    equal(
        v["exact_normalized_source_text_duplicates_with_original_examples"],
        duplicates,
        "Exact duplicate examples",
    )
    a.write(
        a.OUT / "details_independent_verification.json",
        dict(
            status="passed",
            created_utc=a.pure.utc(),
            source_sha256=a.sha(Path(__file__)),
            inputs={a.rel(a.OUT / "details.json"): a.sha(a.OUT / "details.json")},
            intake_records=360,
            pool_records=36,
            cross_stream_records=len(v["cross_stream_overlap"]),
            new_Qwen_calls=0,
        ),
    )
    print("Independent additional-descriptive checks passed.", flush=True)


if __name__ == "__main__":
    with a.pure.offline():
        main()
