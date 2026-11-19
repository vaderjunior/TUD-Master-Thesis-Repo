from pathlib import Path
from analysis.feedback.reserved import transitions as a

NAT = "supporting/natural5__error_balanced-reference5"
MASK = "mechanism/masked5__error_balanced"
EXT = "mechanism/extended6__error_balanced"
SLOTS = [
    ("01_benign_fewer", a.MAIN, "fix", "Benign text, fewer binary errors"),
    ("02_benign_more", a.MAIN, "regression", "Benign text, more binary errors"),
    ("03_hateful_fewer", a.MAIN, "fix", "Hateful text, fewer binary errors"),
    ("04_hateful_more", a.MAIN, "regression", "Hateful text, more binary errors"),
    ("05_offsetting", a.MAIN, "fix", "Equal binary averages with offsetting changes"),
    (
        "06_persistent_error",
        a.MAIN,
        "unchanged_error",
        "Persistent binary error in both arms",
    ),
    ("07_type_fix", a.MAIN, "type_fix", "Exact type-set fix with both gates positive"),
    (
        "08_type_regression",
        a.MAIN,
        "type_regression",
        "Exact type-set regression with both gates positive",
    ),
    ("09_identical_fix", NAT, "fix", "Identical-message binary fix"),
    (
        "10_identical_regression",
        NAT,
        "regression",
        "Identical-message binary regression",
    ),
    ("11_label_fix", MASK, "fix", "Label-visibility comparison: binary fix"),
    (
        "12_label_regression",
        MASK,
        "regression",
        "Label-visibility comparison: binary regression",
    ),
    ("13_retention_fix", EXT, "fix", "Retention/position comparison: binary fix"),
    (
        "14_retention_regression",
        EXT,
        "regression",
        "Retention/position comparison: binary regression",
    ),
]


def qualifies(slot, pair, text):
    if not pair["common_mask_bits"] & 8 or pair["status"] != "valid":
        return False
    sid, contrast, event, _ = slot
    if pair["contrast"] != contrast:
        return False
    if event.startswith("type_"):
        return pair.get("type_exact") == event[5:] and pair.get(
            "both_gates_positive", False
        )
    if pair["binary"] != event:
        return False
    if sid.startswith("01_"):
        return text["endpoint"] == "FPR" and text.get("direction") == "fewer_errors"
    if sid.startswith("02_"):
        return text["endpoint"] == "FPR" and text.get("direction") == "more_errors"
    if sid.startswith("03_"):
        return text["endpoint"] == "FNR" and text.get("direction") == "fewer_errors"
    if sid.startswith("04_"):
        return text["endpoint"] == "FNR" and text.get("direction") == "more_errors"
    if sid.startswith("05_"):
        return text.get("direction") == "same_average_errors" and text.get(
            "offsetting_binary_changes"
        )
    if sid.startswith("06_"):
        return text["counts"].get("unchanged_error", 0) == text["counts"].get(
            "valid", 0
        )
    if sid.startswith(("09_", "10_")):
        return pair["same_messages"]
    return True


def main():
    done = a.io.read(a.OUT / "analysis_complete.json")
    a.io.verify(a.io.ROOT, done["artifacts"])
    a.need(
        (a.OUT / "independent_readback.json").exists()
        and not (a.OUT / "cases_complete.json").exists(),
        "Cases require complete independently checked counts",
    )
    roster, science, predictions, meta = a.load()
    bykey = {r["key"]: r for r in roster}
    bases = {s[1] for s in SLOTS}
    texts = {
        (r["contrast"], r["key"]): r
        for r in a.rows(a.OUT / "per_text_transitions.jsonl.gz")
        if r["contrast"] in bases
    }
    candidates = {s[0]: {} for s in SLOTS}
    for pair in a.rows(a.OUT / "transition_pairs.jsonl.gz"):
        if pair["contrast"] not in bases:
            continue
        for slot in SLOTS:
            if qualifies(slot, pair, texts[pair["contrast"], pair["key"]]):
                value = "|".join(
                    str(pair[k]) for k in ("key", "treatment", "control", "repeat")
                )
                rank = a.core.rank(slot[0], value)
                old = candidates[slot[0]].get(pair["key"])
                if old is None or rank < old["pair_hash"]:
                    candidates[slot[0]][pair["key"]] = {"pair_hash": rank, "pair": pair}
    selected = []
    used = set()
    for slot in SLOTS:
        sid, contrast, event, title = slot
        possible = candidates[sid]
        key = a.core.select_key(sid, possible, used)
        fallback = False
        if key is None and possible:
            key = a.core.select_key(sid, possible, set())
            fallback = True
        if key:
            used.add(key)
        selected.append(
            {
                "slot": sid,
                "title": title,
                "comparison": contrast,
                "candidate_keys": sorted(possible),
                "candidate_count": len(possible),
                "selected_key": key,
                "reused_because_no_unused_candidate": fallback,
                "selection_key_hash": a.core.rank(sid, key) if key else None,
                "selected_pair": possible[key] if key else None,
            }
        )
    a.write(
        a.OUT / "case_selection.json",
        {
            "method": "All eligible keys retained. Minimum prespecified SHA-256 by slot, then minimum pair hash; outcomes define the announced diagnostic stratum, not strength, rhetorical appeal or a new optimal policy.",
            "scope": "Ordinary common-valid, same-six excluded; selected after the exhaustive ledger and independent readback.",
            "slots": selected,
        },
    )
    catalog = a.io.read(a.OUT / "example_catalog.json")
    context = {
        (r["key"], r["condition"]): r
        for r in a.rows(a.OUT / "contexts.jsonl.gz")
        if r["key"] in used
    }
    source_records = a.io.read(a.SELECTION / "gate_only_records.json")
    statuses = a.io.read(a.SELECTION / "candidate_statuses.json")
    cases = []
    for choice in selected:
        key = choice["selected_key"]
        if key is None:
            continue
        pair = choice["selected_pair"]["pair"]
        t, c = pair["treatment"], pair["control"]
        conditions = {"reference5", "reference4", t, c}
        for condition in (t, c):
            if "__" in condition:
                pool = condition.split("__", 1)[1]
                conditions.update(mode + "__" + pool for mode in a.P4.MODES)
        displayed = {}
        for condition in sorted(conditions):
            ctx = context[key, condition]
            es = [
                {
                    "id": eid,
                    "text": catalog[eid]["text"],
                    "meta": catalog[eid]["metadata_variants"][h],
                    "meta_sha256": h,
                }
                for eid, h in zip(ctx["example_order"], ctx["example_meta_sha256"])
            ]
            source = ctx["source_id"]
            displayed[condition] = {
                "context": ctx,
                "examples": es,
                "selected_source_record": source_records[source] if source else None,
                "selected_screening": statuses[source] if source else None,
            }
        cases.append(
            {
                "selection": choice,
                "query": bykey[key],
                "focal_pair": pair,
                "text_comparison": texts[pair["contrast"], key],
                "displayed_contexts": displayed,
                "all_condition_outcomes": {
                    condition: [predictions[key, condition, rep] for rep in range(3)]
                    for condition in science["conditions"]
                },
                "interpretation_boundary": "Illustration selected mechanically within a declared outcome stratum. Does not identify internal reasoning, representative prevalence, human semantic correctness, or a new feedback policy.",
            }
        )
    a.write(a.OUT / "case_packets.json", cases)

    wanted = {
        (c["query"]["key"], condition)
        for c in cases
        for condition in c["displayed_contexts"]
    }
    frozen = {}
    for p in a.io.rows(a.io.STUDY / "prompts.jsonl"):
        identity = p["key"], p["condition"]
        if identity in wanted:
            frozen[identity] = p
    a.need(set(frozen) == wanted, "Missing selected frozen prompt")
    for case in cases:
        for condition, d in case["displayed_contexts"].items():
            p = frozen[case["query"]["key"], condition]
            es = p["context"]["example_groups"][0][1]
            a.need(
                [{k: e[k] for k in ("id", "text", "meta")} for e in d["examples"]]
                == [{k: e[k] for k in ("id", "text", "meta")} for e in es],
                "Case example packet differs",
            )
            a.need(
                p["messages_sha256"] == d["context"]["messages_sha256"]
                and p["text"] == case["query"]["text"],
                "Case prompt link differs",
            )
    a.write(
        a.OUT / "case_frozen_messages.json",
        [
            {
                "key": k,
                "condition": c,
                "messages_sha256": p["messages_sha256"],
                "messages": p["messages"],
            }
            for (k, c), p in sorted(frozen.items())
        ],
    )
    paths = [
        a.OUT / n
        for n in (
            "case_selection.json",
            "case_packets.json",
            "case_frozen_messages.json",
        )
    ]
    a.write(
        a.OUT / "cases_complete.json",
        {
            "status": "deterministic_case_packets_checked_semantic_discussion_pending",
            "created_utc": a.io.utc(),
            "artifacts": {a.io.rel(p): a.io.sha(p) for p in paths},
            "bindings": {
                a.io.rel(Path(__file__)): a.io.sha(Path(__file__)),
                a.io.rel(a.OUT / "analysis_complete.json"): a.io.sha(
                    a.OUT / "analysis_complete.json"
                ),
                a.io.rel(a.OUT / "independent_readback.json"): a.io.sha(
                    a.OUT / "independent_readback.json"
                ),
            },
            "slots": len(selected),
            "selected_cases": len(cases),
            "unique_case_texts": len(used),
            "frozen_case_contexts_checked": len(frozen),
            "new_model_calls": 0,
        },
    )
    print(
        "Deterministic case packets checked; qualitative discussion remains.",
        flush=True,
    )


if __name__ == "__main__":
    with a.io.offline():
        main()
