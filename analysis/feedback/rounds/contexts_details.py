from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np
from analysis.feedback.rounds import contexts_inputs as a


def main():
    a.need(
        a.read(a.OUT / "numerical_independent_verification.json")["status"] == "passed",
        "Verified analysis required",
    )
    result = a.read(a.OUT / "analysis.json")
    ds = [d for d in a.rows(a.OUT / "deliveries.jsonl") if d["role"] == "evaluation"]
    selections = a.read(a.OUT / "selections.json")
    catalog = a.read(a.OUT / "source_catalog.json")
    candidates = list(a.rows(a.OUT / "candidate_selection_ledger.jsonl"))
    contexts = list(a.rows(a.OUT / "query_contexts.jsonl"))
    matrix = list(a.rows(a.READER / "matrix.jsonl"))
    by_key = {r["key"]: r for r in contexts}
    pools = {
        (s["round"], s["stream"], p): set(s["cumulative"][p])
        for s in selections
        for p in a.POLICIES
    }
    index = {(d["round"], d["stream"], d["policy"], d["key"]): d for d in ds}
    cross_stream = []
    for rnd in range(1, 5):
        for p in a.POLICIES:
            for left, right in combinations(a.STREAMS, 2):
                x, y = pools[rnd, left, p], pools[rnd, right, p]
                for pop in ["en_test", "en_hatecheck_eval"]:
                    keys = [r["key"] for r in matrix if r["population"] == pop]
                    paired = [
                        (index[rnd, left, p, key], index[rnd, right, p, key])
                        for key in keys
                    ]
                    cross_stream.append(
                        dict(
                            round=rnd,
                            policy=p,
                            left_stream=left,
                            right_stream=right,
                            population=pop,
                            intersection=len(x & y),
                            union=len(x | y),
                            jaccard=len(x & y) / len(x | y),
                            inputs=len(keys),
                            same_source=sum(
                                u["source_id"] == v["source_id"] for u, v in paired
                            ),
                            same_prompt=sum(
                                u["messages_sha256"] == v["messages_sha256"]
                                for u, v in paired
                            ),
                        )
                    )
    pool_use, intake = [], []
    for rnd in range(1, 5):
        for stream in a.STREAMS:
            for p in a.POLICIES:
                values = [
                    d
                    for d in ds
                    if (d["round"], d["stream"], d["policy"]) == (rnd, stream, p)
                ]
                counts = Counter(d["source_id"] for d in values)
                a.need(
                    sum(counts.values()) == 2077
                    and set(counts) <= pools[rnd, stream, p],
                    "Full per-pool evaluation matrix",
                )
                freq = sorted([counts[s] for s in pools[rnd, stream, p]], reverse=True)
                pool_use.append(
                    dict(
                        round=rnd,
                        stream=stream,
                        policy=p,
                        pool_size=len(pools[rnd, stream, p]),
                        used=len(counts),
                        unused=sorted(pools[rnd, stream, p] - set(counts)),
                        assignments=len(values),
                        newest=sum(d["added_round"] == rnd for d in values),
                        oldest=sum(d["added_round"] == 1 for d in values),
                        top1_count=freq[0],
                        top5_count=sum(freq[:5]),
                        effective_sources=sum(freq) ** 2 / sum(n * n for n in freq),
                    )
                )
                selection = next(
                    s for s in selections if (s["round"], s["stream"]) == (rnd, stream)
                )
                for sid in selection["additions"][p]:
                    exposure = [
                        d
                        for d in ds
                        if d["stream"] == stream
                        and d["policy"] == p
                        and d["source_id"] == sid
                        and d["round"] >= rnd
                    ]
                    counts_by_round = {
                        str(r): sum(d["round"] == r for d in exposure)
                        for r in range(rnd, 5)
                    }
                    intake.append(
                        dict(
                            stream=stream,
                            policy=p,
                            source_id=sid,
                            added_round=rnd,
                            gold_hate=catalog[sid]["record"]["meta"]["gate"],
                            first_shown_round=min(
                                (d["round"] for d in exposure), default=None
                            ),
                            assignments_by_round=counts_by_round,
                            total_assignments=len(exposure),
                            immediate_assignments=counts_by_round[str(rnd)],
                        )
                    )
    selection_errors = []
    for rnd in range(1, 5):
        for p in a.POLICIES:
            for g in [False, True]:
                chosen = [
                    r
                    for r in candidates
                    if r["round"] == rnd and r["gold_hate"] == g and r["selected"][p]
                ]
                selection_errors.append(
                    dict(
                        round=rnd,
                        policy=p,
                        gold_hate=g,
                        selected=len(chosen),
                        transition_counts=dict(
                            Counter(r["error_transition"] for r in chosen)
                        ),
                        initial_errors=sum(r["initial_majority"] != g for r in chosen),
                        current_errors=sum(r["current_majority"] != g for r in chosen),
                    )
                )
    source_screens = []
    for rnd in range(1, 5):
        for g in [False, True]:
            eligible = [
                r
                for r in candidates
                if r["round"] == rnd and r["gold_hate"] == g and r["common_valid"]
            ]
            source_screens.append(
                dict(
                    round=rnd,
                    gold_hate=g,
                    eligible=len(eligible),
                    initial_errors=sum(r["initial_majority"] != g for r in eligible),
                    current_errors=sum(r["current_majority"] != g for r in eligible),
                    transitions=dict(Counter(r["error_transition"] for r in eligible)),
                )
            )
    context_summary = {}
    for pop in ["en_test", "en_hatecheck_eval"]:
        keys = {r["key"] for r in matrix if r["population"] == pop}
        rows = [r for r in contexts if r["key"] in keys]
        fifths = [r["context"]["example_groups"][0][1][4] for r in rows]
        context_summary[pop] = dict(
            queries=len(rows),
            fifth_hateful=sum(r["removed_gate"] for r in rows),
            fifth_is_lowest_cosine=sum(r["fifth_is_lowest_cosine"] for r in rows),
            fifth_cosine_rank=dict(Counter(str(r["fifth_cosine_rank"]) for r in rows)),
            ordinary_unique_ids=len(
                {h["id"] for r in rows for h in r["context"]["example_groups"][0][1]}
            ),
            replaced_unique_ids=len({r["removed_id"] for r in rows}),
            replaced_sources=dict(Counter(r["removed_source"] for r in rows)),
            removed_any_known=sum(any(r["removed_known_fine"].values()) for r in rows),
            fields={
                f: dict(
                    known_removed=sum(r["removed_known_fine"][f] for r in rows),
                    known_nonempty_removed=sum(
                        h["meta"].get(f) is not None and bool(h["meta"][f])
                        for h in fifths
                    ),
                    known_empty_removed=sum(
                        h["meta"].get(f) is not None and not bool(h["meta"][f])
                        for h in fifths
                    ),
                    last_known_example_removed=sum(
                        r["removed_known_fine"][f]
                        and r["remaining_known_fine_counts"][f] == 0
                        for r in rows
                    ),
                )
                for f in a.FIELDS
            },
        )
        for f, v in context_summary[pop]["fields"].items():
            a.need(
                v["known_removed"]
                == v["known_nonempty_removed"] + v["known_empty_removed"],
                "Known/empty annotation partition",
            )
            a.need(
                v["known_removed"] * 36
                == result["population_totals"][pop]["known_fine_removed"][f],
                "Per-query versus assignment coverage",
            )
            a.need(
                v["last_known_example_removed"] * 36
                == result["population_totals"][pop]["last_known_example_removed"][f],
                "Last-known coverage reconciliation",
            )
    exact_duplicate = []
    for d in ds:
        text = " ".join(catalog[d["source_id"]]["record"]["text"].casefold().split())
        hits = by_key[d["key"]]["context"]["example_groups"][0][1]
        matching = [
            i + 1
            for i, h in enumerate(hits)
            if text == " ".join(h["text"].casefold().split())
        ]
        if matching:
            exact_duplicate.append(
                dict(
                    phase=d["phase"],
                    key=d["key"],
                    condition=d["condition"],
                    original_positions=matching,
                )
            )
    global_counts = Counter(d["source_id"] for d in ds)
    top = []
    for sid, n in global_counts.most_common(12):
        selected = [r for r in intake if r["source_id"] == sid]
        top.append(
            dict(
                source_id=sid,
                assignments=n,
                share=n / len(ds),
                selected_memberships=len(selected),
                by_population=dict(
                    Counter(d["population"] for d in ds if d["source_id"] == sid)
                ),
                by_policy=dict(
                    Counter(d["policy"] for d in ds if d["source_id"] == sid)
                ),
                gate=catalog[sid]["record"]["meta"]["gate"],
                text=catalog[sid]["record"]["text"],
            )
        )
    summary = dict(
        status="descriptive_details_complete",
        created_utc=a.pure.utc(),
        cross_stream_overlap=cross_stream,
        per_pool_use=pool_use,
        correction_intake=intake,
        selected_error_history=selection_errors,
        source_screening=source_screens,
        reference_contexts=context_summary,
        exact_normalized_source_text_duplicates_with_original_examples=exact_duplicate,
        total_selected_source_ids=len(catalog),
        evaluation_delivered_source_ids=len(global_counts),
        never_evaluation_delivered_source_ids=sorted(set(catalog) - set(global_counts)),
        top_delivered_sources=top,
        invalid_candidates=[r for r in candidates if not r["common_valid"]],
        inputs={
            a.rel(p): a.sha(p)
            for p in [
                a.OUT / "analysis.json",
                a.OUT / "numerical_independent_verification.json",
                a.OUT / "context_independent_verification.json",
                a.OUT / "deliveries.jsonl",
                a.OUT / "selections.json",
                a.OUT / "candidate_selection_ledger.jsonl",
                a.OUT / "query_contexts.jsonl",
            ]
        },
        source_sha256=a.sha(Path(__file__)),
        new_Qwen_calls=0,
    )
    a.write(a.OUT / "details.json", summary)
    print(
        "Detailed cross-stream overlap, source uptake and annotation-loss views saved.",
        flush=True,
    )


if __name__ == "__main__":
    with a.pure.offline():
        main()
