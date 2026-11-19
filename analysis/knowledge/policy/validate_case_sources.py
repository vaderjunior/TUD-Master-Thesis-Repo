from collections import Counter, defaultdict
import json
import pandas as pd
from analysis.knowledge.policy import labels as p

c = p.c
B = p.BASE


def run():
    contract, root, roster, endpoints, arrays = p.load_inputs()
    cards = c.read(B / "case_cards.json")
    contexts = {
        v["key"]: v["context"] for v in c.rows(B / "selected_original_contexts.jsonl")
    }
    raw = pd.read_csv(
        c.ROOT / "data/raw/gahd/gahd.csv",
        dtype={"gahd_id": str, "contrastive_gahd_id": str},
    )
    links = defaultdict(set)
    for row in raw.to_dict("records"):
        other = row["contrastive_gahd_id"]
        if not isinstance(other, str):
            continue
        links["gahd-" + row["gahd_id"]].add("gahd-" + other)
        links["gahd-" + other].add("gahd-" + row["gahd_id"])
    linked = []
    eligible = []
    for key in endpoints["de_hate"]["keys"]:
        if roster[key]["source"] != "gahd":
            continue
        eligible.append(key)
        ex = [x for _, group in contexts[key]["example_groups"] for x in group]
        delivered = [
            dict(example_id=x["id"], gate=x["meta"].get("gate"), text=x["text"])
            for x in ex
            if x["id"].removeprefix("ex-") in links[roster[key]["id"]]
        ]
        if delivered:
            linked.append(dict(key=key, gate=roster[key]["gate"], examples=delivered))
    selected = []
    for caseid, cond, rep in [
        ("CP4-40", "complete_definitions", 0),
        ("CP4-41", "complete_definitions", 0),
        ("CP4-46", "complete_policy", 0),
    ]:
        card = next((v for v in cards if v["case_id"] == caseid))
        answer = card["answers"][cond][rep]
        path = c.ROOT / answer["result_path"]
        c.require(
            c.sha(path) == answer["result_sha256"],
            "Selected explanation source changed",
        )
        raw = c.read(path)
        valid = [r for r in raw["run_predictions"] if r is not None]
        agreeing = [r for r in valid if r["hate"] == raw["result"]["hate"]]
        inherited = (agreeing or valid)[0]
        c.require(
            inherited["reasoning"] == raw["result"]["reasoning"],
            "Explanation origin differs",
        )
        selected.append(
            dict(
                case_id=caseid,
                condition=cond,
                outer_repeat=rep,
                result=raw["result"],
                inherited_inner_answer=inherited,
                all_inner_answers=valid,
                mismatching_fields=[
                    k
                    for k in ("hate_type", "target_group", "severity", "legal")
                    if inherited[k] != raw["result"][k]
                ],
                source_sha256=answer["result_sha256"],
            )
        )
    features = {v["key"]: v for v in c.rows(B / "context_features.jsonl")}
    strata = {
        str(i): dict(
            Counter(
                (
                    roster[k]["source"]
                    for k in endpoints["de_hate"]["keys"]
                    if features[k]["legal_guidelines_A"] == i
                )
            )
        )
        for i in range(3)
    }
    p.save(
        B / "context_followup_checks.json",
        dict(
            status="PASS",
            gahd_direct_contrastive_links=dict(
                endpoint_n=len(eligible),
                inputs_with_declared_partner_shown=len(linked),
                shown_partner_records=sum((len(v["examples"]) for v in linked)),
                opposite_gold_gate_partner_records=sum(
                    (x["gate"] != v["gate"] for v in linked for x in v["examples"])
                ),
                cases=linked,
                scope="Only direct official contrastive links, in either direction, in the actual five examples. This is not an exhaustive semantic-overlap test and no exclusion sensitivity is run here.",
            ),
            explanation_provenance=selected,
            german_hate_legal_guide_count_by_source=strata,
        ),
    )
    print(
        json.dumps(
            dict(
                partner_inputs=len(linked),
                partner_records=sum((len(v["examples"]) for v in linked)),
                examples_checked=len(selected),
                legal_guide_strata=strata,
            ),
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    with c.offline():
        run()
