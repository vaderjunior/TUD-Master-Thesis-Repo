from collections import Counter, defaultdict
import hashlib
import json
import numpy as np
import pandas as pd
from analysis.knowledge.policy import labels as p
from analysis.knowledge.policy.verify_results import labels
from analysis.knowledge.policy.verify_results import require_close

c = p.c
B = p.BASE


def run():
    contract, root, roster, endpoints, arrays = p.load_inputs()
    done = c.read(p.PART3 / "reader_complete.json")
    c.verify(done["reader_files"])
    predictions = {
        (v["key"], v["condition"], v["repeat"]): v["result"]
        for name in done["reader_files"]
        for v in c.read(c.ROOT / name)
    }
    features = {v["key"]: v for v in c.rows(B / "context_features.jsonl")}
    policy = {
        v["record"]["id"]: v["record"]
        for v in c.read(c.SETUP / "policy_inventory.json")
    }
    data = c.read(B / "groups_and_sources.json")
    checks = Counter()
    direct = {}
    diagnostics = {}
    for endpoint, meta in endpoints.items():
        field = meta["dimension"]
        fine = field not in ("hate", "legal")
        state = {}
        correct = {}
        selected = [
            k
            for k in meta["keys"]
            if all(
                (
                    predictions[k, cond, r] is not None
                    for cond in p.CONDITIONS
                    for r in range(3)
                )
            )
        ]
        for key in selected:
            gold = labels(roster[key], field, True)
            state[key] = [
                [labels(predictions[key, cond, r], field) for r in range(3)]
                for cond in p.CONDITIONS
            ]
            correct[key] = [
                [
                    q == gold and (not fine or predictions[key, cond, r]["hate"])
                    for r, q in enumerate(state[key][ci])
                ]
                for ci, cond in enumerate(p.CONDITIONS)
            ]
        direct[endpoint] = dict(keys=selected, correct=correct, state=state)
        for source in sorted({roster[k]["source"] for k in selected}):
            keys = [k for k in selected if roster[k]["source"] == source]
            old = data[endpoint]["groups"]["source/" + source]
            c.require(len(keys) == old["n"], "Source N differs")
            for ci, cond in enumerate(p.CONDITIONS):
                for rep in range(3):
                    counts = []
                    for label in meta["all_labels"]:
                        yy = [label in labels(roster[k], field, True) for k in keys]
                        pp = [label in state[k][ci][rep] for k in keys]
                        counts.append(
                            [
                                sum((a and b for a, b in zip(yy, pp))),
                                sum((not a and b for a, b in zip(yy, pp))),
                                sum((a and (not b) for a, b in zip(yy, pp))),
                            ]
                        )
                    require_close(
                        counts, old["counts"][ci][rep], "Direct source TP/FP/FN differs"
                    )
                    checks["source_condition_repeat_counts"] += 1
                    f1 = [
                        2 * t / (2 * t + f + n) if 2 * t + f + n else 0.0
                        for t, f, n in counts
                    ]
                    require_close(
                        sum(f1) / len(f1),
                        old["macro_repeats"][ci][rep],
                        "Direct source macro differs",
                    )
            for j, (left, right) in enumerate(p.CONTRASTS):
                fixed = [
                    sum(
                        (
                            correct[k][left][r] and (not correct[k][right][r])
                            for k in keys
                        )
                    )
                    for r in range(3)
                ]
                broken = [
                    sum(
                        (not correct[k][left][r] and correct[k][right][r] for k in keys)
                    )
                    for r in range(3)
                ]
                require_close(
                    fixed, old["transitions"][j]["fixes_repeats"], "Source fixes differ"
                )
                require_close(
                    broken,
                    old["transitions"][j]["regressions_repeats"],
                    "Source regressions differ",
                )
                checks["source_contrasts"] += 1
        for group in ("none", "some", "all", "empty_gold"):
            kk = []
            for key in selected:
                wanted = (
                    {"hate"} if field == "hate" else labels(roster[key], field, True)
                )
                seen = {
                    "hate" if field == "hate" else policy[i].get("label")
                    for i in features[key]["definitions_A"]
                    if policy[i]["dimension"] == field
                }
                found = seen & wanted
                actual = (
                    "empty_gold"
                    if not wanted
                    else "all" if wanted <= seen else "some" if found else "none"
                )
                if actual == group:
                    kk.append(key)
            saved = data[endpoint]["groups"]["gold_definition_coverage/" + group]
            c.require(len(kk) == saved["n"], "Coverage grouping differs")
            if kk:
                require_close(
                    [
                        sum((sum(correct[k][ci]) for k in kk)) / (3 * len(kk))
                        for ci in range(3)
                    ],
                    saved["exact"],
                    "Coverage exact differs",
                )
            checks["definition_coverage_groups"] += 1
        if field in ("hate_type", "target_group", "severity", "legal"):
            paired = []
            for left, right in p.CONTRASTS:
                fixes = Counter()
                regressions = Counter()
                for rep in range(3):
                    for key in selected:
                        a, b = (
                            predictions[key, p.CONDITIONS[left], rep],
                            predictions[key, p.CONDITIONS[right], rep],
                        )
                        transition = str(b["hate"]) + "->" + str(a["hate"])
                        if correct[key][left][rep] and (not correct[key][right][rep]):
                            fixes[transition] += 1
                        if not correct[key][left][rep] and correct[key][right][rep]:
                            regressions[transition] += 1
                paired.append(
                    dict(
                        left=p.CONDITIONS[left],
                        right=p.CONDITIONS[right],
                        fixes_gate_transitions_pooled_three_repeats=dict(fixes),
                        regressions_gate_transitions_pooled_three_repeats=dict(
                            regressions
                        ),
                    )
                )
            diagnostics[endpoint] = dict(paired_gate_transitions=paired)
    for v in c.read(B / "label_definition_exposure.json"):
        meta = endpoints[v["endpoint"]]
        field = meta["dimension"]
        lab = v["label"]
        defids = {
            i
            for i, rec in policy.items()
            if rec["kind"] == "definition"
            and rec["dimension"] == field
            and (field == "hate" or rec.get("label") == lab)
        }
        keys = [
            k
            for k in direct[v["endpoint"]]["keys"]
            if (lab in labels(roster[k], field, True)) == v["gold_positive"]
            and bool(defids & set(features[k]["definitions_A"]))
            == v["definition_A_present"]
        ]
        c.require(len(keys) == v["n"], "Exposure N differs")
        counts = [
            [
                sum((lab in direct[v["endpoint"]]["state"][k][ci][r] for k in keys))
                for r in range(3)
            ]
            for ci in range(3)
        ]
        require_close(
            counts, v["predicted_positive_counts"], "Exposure positive counts differ"
        )
        checks["label_exposure_cells"] += 1
    selection = c.read(B / "case_selection.json")
    selected_types = Counter()
    for slot in selection:
        d = direct[slot["endpoint"]]
        left = p.CONDITIONS.index(slot["left"])
        right = p.CONDITIONS.index(slot["right"])
        keys = [
            k
            for k in d["keys"]
            if slot["source"] == "all" or roster[k]["source"] == slot["source"]
        ]
        if slot["stratum"] in ("True", "False"):
            keys = [k for k in keys if roster[k]["gate"] == (slot["stratum"] == "True")]
        if slot["stratum"] in ("empty", "nonempty"):
            keys = [
                k
                for k in keys
                if bool(roster[k]["legal"]) == (slot["stratum"] == "nonempty")
            ]
        if slot["direction"] == "persistent_wrong":
            keys = [
                k
                for k in keys
                if not any(d["correct"][k][0])
                and (not any(d["correct"][k][2]))
                and all(
                    (
                        d["state"][k][ci][r] == d["state"][k][0][0]
                        and (
                            endpoints[slot["endpoint"]]["dimension"]
                            in ("hate", "legal")
                            or predictions[k, p.CONDITIONS[ci], r]["hate"]
                            == predictions[k, p.CONDITIONS[0], 0]["hate"]
                        )
                        for ci in (0, 2)
                        for r in range(3)
                    )
                )
            ]
        else:
            sign = 1 if slot["direction"] == "fix" else -1
            strength = {
                k: sign * (sum(d["correct"][k][left]) - sum(d["correct"][k][right]))
                for k in keys
            }
            keys = [k for k in keys if strength[k] > 0]
            c.require(len(keys) == slot["candidates"], "Candidate count differs")
            stable = [k for k in keys if strength[k] == 3]
            c.require(
                len(stable) == slot["stable_candidates"],
                "Stable candidate count differs",
            )
            if keys:
                top = max((strength[k] for k in keys))
                keys = [k for k in keys if strength[k] == top]
        expected = (
            min(
                keys,
                key=lambda k: (
                    hashlib.sha256((p.CASE_SEED + "|" + k).encode()).hexdigest(),
                    k,
                ),
            )
            if keys
            else None
        )
        c.require(expected == slot["key"], "Independent case selection differs")
        checks["case_slots"] += 1
        selected_types[slot["selection"]] += 1
    cards = c.read(B / "case_cards.json")
    input_paths = {v["input"]["source_file"] for v in cards}
    source_bindings = {}
    for name in sorted(input_paths):
        path = c.ROOT / name
        source_bindings[c.ref(path)] = c.sha(path)
        frame = pd.read_parquet(path)
        index = {row["id"]: row for row in frame.to_dict("records")}
        for card in cards:
            r = card["input"]
            if r["source_file"] != name:
                continue
            original = index[r["id"]]
            for field in (
                "text",
                "source",
                "lang",
                "gate",
                "hate_types",
                "target_groups",
                "severity",
                "legal",
            ):
                x = original.get(field)
                y = r.get(field)
                if isinstance(x, np.ndarray):
                    x = x.tolist()
                if isinstance(x, np.generic):
                    x = x.item()
                if isinstance(x, float) and np.isnan(x):
                    x = None
                c.require(
                    x == y,
                    "Selected case processed label/text differs: "
                    + card["key"]
                    + " "
                    + field,
                )
            checks["case_processed_source_rows"] += 1
    path = c.ROOT / "data/raw/gahd/gahd.csv"
    source_bindings[c.ref(path)] = c.sha(path)
    frame = pd.read_csv(path, dtype={"gahd_id": str, "contrastive_gahd_id": str})
    wanted = {"94c95084", "35c7cee5", "c9605a02"}
    gahd_rows = json.loads(
        frame[frame["gahd_id"].isin(wanted)].to_json(
            orient="records", force_ascii=False
        )
    )
    for card in cards:
        if card["key"] not in ("de/gahd-94c95084", "de/gahd-35c7cee5"):
            continue
        row = next((v for v in gahd_rows if "de/gahd-" + v["gahd_id"] == card["key"]))
        c.require(
            row["text"] == card["input"]["text"]
            and bool(row["label"]) == card["input"]["gate"],
            "GAHD original case differs",
        )
        checks["original_gahd_cases"] += 1
    for name in (
        "src/hsrag/vote.py",
        "src/hsrag/data/detox.py",
        "src/hsrag/data/gahd.py",
        "src/hsrag/data/mhs.py",
    ):
        source_bindings[name] = c.sha(c.ROOT / name)
    c.verify(source_bindings)
    p.save(
        B / "case_annotation_provenance.json",
        dict(
            source_bindings=source_bindings,
            gahd_source_rows=gahd_rows,
            warning="Unchanged original annotations; no semantic relabelling. GAHD official split differs from the project split as already documented.",
        ),
    )
    p.save(B / "gate_transition_diagnostics.json", diagnostics)
    p.save(
        B / "independent_checks.json",
        dict(
            status="PASS",
            checks=dict(checks),
            case_selection_types=dict(selected_types),
            original_inputs_unchanged=True,
            no_new_model_requests=True,
        ),
    )
    print(
        json.dumps(
            dict(status="PASS", checks=checks, case_selection_types=selected_types),
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    with c.offline():
        run()
