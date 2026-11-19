from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from sklearn.metrics import f1_score
from analysis.knowledge.policy import common as c
from analysis.knowledge.policy import reader as reader


def require_close(left, right, message):
    c.require(np.allclose(left, right, atol=1e-12, rtol=0), message)


def labels(value, dimension, gold=False):
    if dimension == "hate":
        return {"hate" if value["gate" if gold else "hate"] else "not_hate"}
    if dimension == "severity":
        return set() if value["severity"] is None else {value["severity"]}
    field = (
        {"hate_type": "hate_types", "target_group": "target_groups"}.get(
            dimension, dimension
        )
        if gold
        else dimension
    )
    c.require(value[field] is not None, "Unknown gold in fixed endpoint")
    return set(value[field])


def run():
    base = reader.BASE
    root = c.ROOT / c.read(base / "analysis_complete.json")["run"]
    receipt = c.read(root / "complete.json")
    c.verify(receipt["artifacts"])
    old_bindings = c.read(base / "analysis_contract.json")["original_bindings"]
    c.verify(old_bindings)
    rr = c.read(base / "reader_complete.json")
    c.verify(rr["reader_files"])
    c.verify(rr["reader_sources"])
    roster = {x["key"]: x for x in c.rows(c.SETUP / "roster.jsonl")}
    predictions = {
        (x["key"], x["condition"], x["repeat"]): x
        for path in rr["reader_files"]
        for x in c.read(c.ROOT / path)
    }
    endpoints = c.read(c.SETUP / "endpoints.json")
    tested = {}
    first_draw_values = 0
    for endpoint, meta in endpoints.items():
        result = c.read(root / endpoint / "results.json")
        point = result["point"]
        field = meta["dimension"]
        selected = [
            key
            for key in meta["keys"]
            if all(
                (
                    predictions[key, condition, repeat]["result"] is not None
                    for condition in c.CONDITIONS
                    for repeat in range(3)
                )
            )
        ]
        c.require(len(selected) == point["n"], "Direct JSON common mask differs")
        yy = np.array(
            [
                [
                    label in labels(roster[key], field, True)
                    for label in meta["all_labels"]
                ]
                for key in selected
            ]
        )
        pp = np.zeros((3, 3, len(selected), len(meta["all_labels"])), bool)
        if field == "severity":
            common_emitted = [
                key
                for key in selected
                if all(
                    (
                        predictions[key, cond, rep]["result"]["severity"] is not None
                        for cond in c.CONDITIONS
                        for rep in range(3)
                    )
                )
            ]
            c.require(
                len(common_emitted) == point["severity"]["common_emitted_n"],
                "Direct common emitted cohort differs",
            )
            for ci, cond in enumerate(c.CONDITIONS):
                for rep in range(3):
                    distances = [
                        abs(
                            meta["all_labels"].index(
                                predictions[key, cond, rep]["result"]["severity"]
                            )
                            - meta["all_labels"].index(roster[key]["severity"])
                        )
                        for key in common_emitted
                    ]
                    require_close(
                        sum(distances) / len(distances),
                        point["severity"]["common_emitted_mae_repeats"][ci][rep],
                        "Direct common-emitted severity MAE differs",
                    )
        state_lists = []
        sample_checks = 0
        for ci, condition in enumerate(c.CONDITIONS):
            states = []
            correct = []
            for repeat in range(3):
                cc = Counter()
                exact = overlap = 0.0
                signed = Counter()
                emitted = closed = open_missing = 0
                answers = []
                right = []
                for i, key in enumerate(selected):
                    answer = predictions[key, condition, repeat]["result"]
                    g = labels(roster[key], field, True)
                    p = labels(answer, field)
                    fine = field not in ("hate", "legal")
                    hit = p == g and (answer["hate"] or not fine)
                    exact += hit
                    right.append(hit)
                    answers.append(
                        (answer["hate"], tuple(sorted(p))) if fine else tuple(sorted(p))
                    )
                    overlap += len(g & p) / len(g | p) if g | p else 1
                    for j, label in enumerate(meta["all_labels"]):
                        pp[ci, repeat, i, j] = label in p
                        cc[label, "tp"] += label in p and label in g
                        cc[label, "fp"] += label in p and label not in g
                        cc[label, "fn"] += label not in p and label in g
                    if field == "severity":
                        closed += not answer["hate"]
                        if answer["severity"] is not None:
                            signed[
                                meta["all_labels"].index(answer["severity"])
                                - meta["all_labels"].index(roster[key]["severity"])
                            ] += 1
                            emitted += 1
                        elif answer["hate"]:
                            open_missing += 1
                counts = [
                    [cc[label, "tp"], cc[label, "fp"], cc[label, "fn"]]
                    for label in meta["all_labels"]
                ]
                require_close(
                    counts,
                    point["counts"][ci][repeat],
                    "Direct JSON label counts differ",
                )
                require_close(
                    exact / len(selected),
                    point["exact_repeats"][ci][repeat],
                    "Direct JSON gate-aware exactness differs",
                )
                if "jaccard_repeats" in point:
                    require_close(
                        overlap / len(selected),
                        point["jaccard_repeats"][ci][repeat],
                        "Direct JSON Jaccard differs",
                    )
                if field == "severity":
                    s = point["severity"]
                    c.require(
                        emitted == s["emitted_n_repeats"][ci][repeat]
                        and closed == point["gate_closed_counts"][ci][repeat]
                        and (
                            open_missing == s["gate_open_missing_n_repeats"][ci][repeat]
                        ),
                        "Direct severity coverage differs",
                    )
                    c.require(
                        all(
                            (
                                signed[d]
                                == s["signed_error_counts_repeats"][str(d)][ci][repeat]
                                for d in (-2, -1, 0, 1, 2)
                            )
                        ),
                        "Signed severity error differs",
                    )
                    require_close(
                        sum((abs(d) * v for d, v in signed.items())) / emitted,
                        s["mae_emitted_repeats"][ci][repeat],
                        "Direct conditional severity MAE differs",
                    )
                states.append(answers)
                correct.append(right)
                sample_checks += len(selected)
            pair_agreement = np.array(
                [
                    [
                        sum(
                            (
                                states[a][i] == states[b][i]
                                for a, b in ((0, 1), (0, 2), (1, 2))
                            )
                        )
                        / 3
                        for i in range(len(selected))
                    ]
                ]
            )
            require_close(
                pair_agreement.mean(),
                point["repeatability"]["pairwise_agreement"][ci],
                "Direct JSON agreement differs",
            )
            state_lists.append(states)
        source_members = defaultdict(set)
        for key in meta["keys"]:
            source_members[roster[key]["text_cluster"]].add(roster[key]["source"])
        members = defaultdict(list)
        for i, key in enumerate(selected):
            members[roster[key]["text_cluster"]].append(i)
        clusters = sorted(members)
        strata = defaultdict(list)
        for i, cl in enumerate(clusters):
            strata[tuple(sorted(source_members[cl]))].append(i)
        rng = np.random.default_rng(20261114)
        weights = np.zeros((256, len(clusters)), int)
        for name in sorted(strata):
            ids = strata[name]
            weights[:, ids] = rng.multinomial(
                len(ids), np.full(len(ids), 1 / len(ids)), size=256
            )
        with np.load(root / endpoint / "bootstrap.npz", allow_pickle=False) as bs:
            for draw in range(3):
                picked = [
                    i
                    for cluster, copies in enumerate(weights[draw])
                    for _ in range(int(copies))
                    for i in members[clusters[cluster]]
                ]
                for ci in range(3):
                    mac = np.mean(
                        [
                            f1_score(
                                yy[picked],
                                pp[ci, r, picked],
                                average="macro",
                                zero_division=0,
                            )
                            for r in range(3)
                        ]
                    )
                    mic = np.mean(
                        [
                            f1_score(
                                yy[picked],
                                pp[ci, r, picked],
                                average="micro",
                                zero_division=0,
                            )
                            for r in range(3)
                        ]
                    )
                    st = state_lists[ci]
                    agreement = sum(
                        (
                            sum(
                                (
                                    st[a][i] == st[b][i]
                                    for a, b in ((0, 1), (0, 2), (1, 2))
                                )
                            )
                            for i in picked
                        )
                    ) / (len(picked) * 3)
                    for name, value in [
                        ("macro_f1", mac),
                        ("micro_f1_all_labels", mic),
                        ("mean_pairwise_repeat_agreement", agreement),
                    ]:
                        require_close(
                            bs[name][draw, ci],
                            value,
                            "Independent actual-data bootstrap draw differs",
                        )
                        first_draw_values += 1
        tested[endpoint] = sample_checks
    live = c.read(reader.LIVE / "complete.json")
    stages = c.read(c.BASE / "schedule.json")["stages"]
    c.require(
        set(live["review_hashes"])
        == {c.ref(reader.LIVE / "batches" / s["id"] / "review.json") for s in stages},
        "Extra/missing batch review",
    )
    c.require(
        set(live["final_integrity_hashes"])
        == {c.ref(reader.LIVE / "final_checks" / f"{s['id']}.json") for s in stages},
        "Extra/missing final check",
    )
    actual_items = {
        p.name for p in c.a.storage_path(reader.LIVE / "items").iterdir() if p.is_dir()
    }
    expected_items = {v["stamp"] for v in predictions.values()}
    c.require(
        actual_items == expected_items and len(actual_items) == 62154,
        "Unexpected live item directory",
    )
    c.verify(old_bindings)
    before = c.read(base / "analysis_contract.json")["thesis_before"]
    after = reader.snapshot_thesis()
    changes = [name for name, sha in before.items() if after.get(name) != sha]
    additions = sorted(set(after) - set(before))
    reader.save(
        base / "final_independent_checks.json",
        dict(
            status="passed",
            direct_json_decision_checks=tested,
            actual_bootstrap_values_reproduced=first_draw_values,
            item_directory_count=len(actual_items),
            frozen_original_bindings_verified=len(old_bindings),
            thesis_files_before=len(before),
            thesis_files_after=len(after),
            thesis_changed_or_removed=changes,
            thesis_added=additions,
            thesis_action="No thesis writes; any concurrent writer changes are listed separately.",
            verification_source_sha256=c.sha(Path(__file__)),
        ),
    )
    print(
        "Final independent checks passed; actual bootstrap values:",
        first_draw_values,
        "; thesis changes:",
        len(changes),
        flush=True,
    )


if __name__ == "__main__":
    with c.offline():
        run()
