from collections import Counter, defaultdict
import json
from pathlib import Path
from analysis.common import replay as c

BASE = c.BASE / "evaluation_preparation"


def label(meta):
    out = []
    if meta.get("gate") is not None:
        out.append("hate=" + ("true" if meta["gate"] else "false"))
    for field, name in [
        ("target_groups", "target_group"),
        ("hate_types", "hate_type"),
        ("severity", "severity"),
        ("legal", "legal"),
    ]:
        value = meta.get(field)
        if value is not None:
            out.append(
                name
                + "="
                + ("[" + ", ".join(value) + "]" if isinstance(value, list) else value)
            )
    text = ", ".join(out)
    if meta.get("astra_aux_stgb"):
        text += (
            "\n  Auxiliary DeTox paragraph flags (illustrative, not legal gold): §"
            + meta["astra_aux_stgb"]
        )
    return text


def main():
    c.verify(c.read(BASE / "plan.json")["inputs"])
    c.verify(c.read(BASE / "complete.json")["outputs"])
    rr = c.rows(BASE / "broad_roster.jsonl")
    phases = defaultdict(list)
    for r in rr:
        phases[r["phase"]].append(r)
    ids = {}
    checks = 0
    for phase, actual in phases.items():
        source = "en_dev" if phase == "larger_dev" else phase
        allgold = c.gold_rows(source)
        wanted = (
            [r for r in allgold if r["gate"] is True and r["target_groups"] is not None]
            if phase == "larger_dev"
            else allgold
        )
        assert {r["id"] for r in wanted} == {r["id"] for r in actual}
        gold = {r["id"]: r for r in wanted}
        for r in actual:
            assert all((r.get(k) == v for k, v in gold[r["id"]].items())), r["id"]
            checks += 1
        ids[phase] = {r["id"] for r in actual}
    hc = {r["id"]: r for r in phases["en_hatecheck_eval"]}
    templates = defaultdict(list)
    links = defaultdict(set)
    norms = defaultdict(list)
    for rid, r in hc.items():
        raw = json.loads(r["raw"])
        templates[raw["templ_id"]].append(rid)
        norms[c.normalized(r["text"])].append(rid)
        if raw["ref_case_id"] is not None:
            ref = "hatecheck-" + str(raw["ref_case_id"])
            links[rid].add(ref)
            links[ref].add(rid)
    for group in list(templates.values()) + list(norms.values()):
        for rid in group[1:]:
            links[group[0]].add(rid)
            links[rid].add(group[0])
    remaining = set(hc)
    components = 0
    while remaining:
        start = min(remaining)
        stack = [start]
        component = set()
        while stack:
            rid = stack.pop()
            if rid in component:
                continue
            component.add(rid)
            stack += list(links[rid] - component)
        remaining -= component
        components += 1
        assert all(
            (hc[rid]["functional_cluster"] == min(component) for rid in component)
        )
    kb = {
        r["id"]: r
        for r in c.rows(c.CORE / "kb/full/records.jsonl")
        if r["kind"] == "example"
    }
    original = {r["key"]: r for r in c.rows(c.CORE / "retrieval_v1/full/prompts.jsonl")}
    fixed = c.read(BASE / "fixed_sets.json")
    rows = c.rows(BASE / "matched_prompts.jsonl")
    groups = defaultdict(dict)
    slot_checks = 0
    unchanged = 0
    coverage = defaultdict(Counter)
    for r in rows:
        key = r["key"]
        cond = r["condition"]
        s = original[key]
        assert cond not in groups[key]
        groups[key][cond] = r
        assert r["messages_sha256"] == c.signature(r["messages"])
        assert r["messages"][0] == s["messages"][0]
        assert all(
            (
                rid in kb and kb[rid]["lang"] == r["lang"]
                for rid in r["example_text_ids"]
            )
        )
        if cond == "original_bridge":
            assert r["messages"] == s["messages"]
            continue
        old = s["messages"][1]["text"]
        new = r["messages"][1]["text"]
        abstracts = old.split("## Labelled examples\n", 1)[0].rstrip()
        assert abstracts in new
        target = old.split("## Text to classify", 1)[1]
        assert new.endswith("## Text to classify" + target)
        assert len(r["example_text_ids"]) == len(set(r["example_text_ids"])) == 5
        if cond.startswith("fixed_"):
            assert r["example_text_ids"] == fixed[r["lang"] + "/" + cond.split("_")[1]]
        else:
            assert r["example_text_ids"] == [
                x["id"] for x in s["context"]["example_groups"][0][1]
            ]
        if cond.startswith("permuted_"):
            assert sorted(r["permutation"]) == list(range(5)) and all(
                (i != v for i, v in enumerate(r["permutation"]))
            )
        else:
            assert r["annotation_source_ids"] == r["example_text_ids"]
        expected = []
        for i, (tid, lid) in enumerate(
            zip(r["example_text_ids"], r["annotation_source_ids"])
        ):
            text = " ".join(kb[tid]["text"].split())
            parts = []
            if cond != "retrieved_labels_only":
                parts.append('"' + text + '"')
            if cond != "retrieved_text_only":
                parts.append("labels: " + label(kb[lid]["meta"]))
            expected.append(str(i + 1) + ". " + "\n   ".join(parts))
            slot_checks += 1
            for scope in r["scopes"]:
                dimension = (
                    "hate_type"
                    if scope["dimension"].startswith("hate_type")
                    else scope["dimension"]
                )
                field = c.original.GOLD_FIELDS[dimension]
                coverage[scope["subset"] + "/" + cond]["slots"] += 1
                coverage[scope["subset"] + "/" + cond]["annotated_slots"] += int(
                    kb[lid]["meta"].get(field) is not None
                    and cond != "retrieved_text_only"
                )
        actual = new.split("## Example context\n", 1)[1].split(
            "\n\n## Text to classify", 1
        )[0]
        assert actual == "\n".join(expected), (key, cond)
    for key, group in groups.items():
        assert len(group) == 10
        complete = group["retrieved_complete"]
        texts = complete["example_text_ids"]
        blocks = [label(kb[rid]["meta"]) for rid in texts]
        permutations = []
        for k in range(3):
            row = group[f"permuted_{k}"]
            new = [label(kb[rid]["meta"]) for rid in row["annotation_source_ids"]]
            assert Counter(blocks) == Counter(new)
            changed = sum((a != b for a, b in zip(blocks, new)))
            assert changed == row["changed_annotation_slots"]
            if not changed:
                assert row["messages"] == complete["messages"]
                unchanged += 1
            permutations.append(tuple(row["permutation"]))
        assert len(set(permutations)) == 3
    assert len(rows) == 9410 and len(groups) == 941
    preparation = c.read(BASE / "preparation.json")
    assert unchanged == preparation["matched_zero_slot_changes"]
    assert (
        len(rr) * 27 == preparation["broad_ordinary_calls"] and len(rows) * 9 == 84690
    )
    c.save(
        BASE / "verification.json",
        {
            "created_utc": c.utc(),
            "status": "independent_offline_preparation_pass",
            "checked_by_sha256": c.sha(Path(__file__)),
            "all_roster_gold_rows_checked": checks,
            "functional_components_independent_graph_check": components,
            "matched_prompts_checked": len(rows),
            "matched_slots_independently_rendered": slot_checks,
            "permutation_identical_message_pairs": unchanged,
            "annotation_delivery_by_endpoint_condition": dict(coverage),
            "new_model_calls": 0,
            "execution_readiness": "No launcher or new predictions; broad retrieval/inference preflight still required",
        },
    )
    print(
        "Independent preparation checks pass:",
        checks,
        "roster rows;",
        len(rows),
        "prompts;",
        slot_checks,
        "slots;",
        components,
        "functional clusters;",
        unchanged,
        "unchanged permutation pairs",
    )


if __name__ == "__main__":
    with c.network_guard():
        main()
