from analysis.feedback import micro_f1 as p
from collections import defaultdict
import numpy as np

c, h, B = p.c, p.h, p.B


def units(study):
    return [
        u for u in c.read(p.BASE / "analysis_plan.json")["units"] if u["study"] == study
    ]


def original(u):
    path = c.ROOT / u["source"]
    c.require(c.sha(path) == u["source_sha256"], "Frozen result changed: " + u["id"])
    return c.read(path)


def raw_rows(path, key="key", study=None, baseline=False):
    data = {}
    for r in c.lines(path):
        if study is not None and r.get("study") != study:
            continue
        if baseline:
            c.require(
                r["condition"] == "reference_baseline", "Unexpected baseline condition"
            )
            k = ("reference", r["replicate"], r[key])
        else:
            k = (r["condition"], r["repeat"], r[key])
        c.require(k not in data, "Duplicate saved answer")
        data[k] = {
            field: r.get(field) for field in ["result", "n_valid", "gold", "cluster"]
        }
    return data


def base(u, old, roster, raw, ids, groups, common, view="registered", strict=False):
    conditions = sorted({x for members in groups.values() for x in members})
    settings = [(k, r) for k in conditions for r in range(3)]
    labels = u["labels"]
    gold = np.array(
        [[l in roster[i]["gold"]["hate_type"] for l in labels] for i in ids], bool
    )
    c.require(
        all(
            roster[i]["gold"]["hate"] is True
            and roster[i]["gold"]["hate_type"] is not None
            for i in ids
        ),
        "Incorrect fine gold eligibility",
    )
    pred = np.zeros((len(settings), len(ids), len(labels)), bool)
    gate = np.zeros(pred.shape[:2], bool)
    valid = gate.copy()
    votes = np.zeros(gate.shape, np.int8)
    for j, (cond, rep) in enumerate(settings):
        for n, i in enumerate(ids):
            r = raw[cond, rep, i]
            ans = r["result"]
            votes[j, n] = r["n_valid"]
            valid[j, n] = ans is not None
            c.require(valid[j, n] == (votes[j, n] > 0), "Null/vote mismatch")
            if r.get("gold") is not None:
                c.require(r["gold"] == roster[i]["gold"], "Raw gold mismatch")
            if r.get("cluster") is not None and roster[i].get("cluster") is not None:
                c.require(r["cluster"] == roster[i]["cluster"], "Raw cluster mismatch")
            if ans is not None:
                c.require(set(ans["hate_type"]) <= set(labels), "Unknown type label")
                pred[j, n] = [l in ans["hate_type"] for l in labels]
                gate[j, n] = ans["hate"]
    if strict:
        valid &= votes == 3
    mask = np.array([i in set(common) for i in ids])
    h.equal(mask, valid.all(0), "Original common mask")
    groupidx = {
        name: [j for j, k in enumerate(settings) if k[0] in members]
        for name, members in groups.items()
    }
    return dict(
        id=u["id"] + ("" if view == "registered" else "/" + view),
        unit_id=u["id"],
        study=u["study"],
        endpoint=u["endpoint"],
        view=view,
        old=old,
        ids=ids,
        gold=gold,
        pred=pred,
        valid=valid,
        gate=gate,
        nvalid=votes,
        mask=mask,
        settings=settings,
        groups=groupidx,
        labels=labels,
        macro_labels=u["macro_labels"],
        macro_indices=[labels.index(l) for l in u["macro_labels"]],
        clusters=[str(roster[i].get("cluster", i)) for i in ids],
        columns={name: {name: 1} for name in groups},
        old_columns={},
        old_groups={},
        old_settings={},
        seed=old.get("bootstrap", {}).get("seed", old.get("seed", 0)),
        sampling="item" if u["study"] in ["feedback", "relevance"] else "cluster",
        registered_main=view == "registered",
    )


def contrasts(d, old):
    study = d["study"]
    for name, v in old["contrasts"].items():
        if "_minus_" in name:
            left, right = name.split("_minus_")
            w = {left: 1, right: -1}
        elif name == "text_by_label_overlap_interaction":
            w = dict(
                relevant_full=1, relevant_labels=-1, disjoint_full=-1, disjoint_labels=1
            )
        elif name in [
            "interaction",
            "explanation_by_application_interaction",
            "explanation_by_selectivity_interaction",
        ]:
            w = dict(
                specific_selective=1,
                plain_selective=-1,
                specific_always=-1,
                plain_always=1,
            )
        else:
            raise ValueError("Unknown original contrast " + name)
        d["columns"][name] = w
        z = dict(
            point=v.get("difference", v.get("mean_difference")),
            ci95=v.get("ci95", v.get("macro_f1_ci95")),
            missing_bounds=v.get(
                "full_roster_missing_bounds",
                v.get(
                    "full_roster_failure_difference_bounds",
                    v.get(
                        "full_roster_missing_prediction_difference_bounds",
                        v.get("full_roster_missing_difference_bounds"),
                    ),
                ),
            ),
        )
        if v.get("ci9875") is not None:

            z.update(adjusted_interval=v["ci9875"], adjusted_confidence=0.9875)
        d["old_columns"][name] = z
    for name, v in old["groups"].items():
        d["old_groups"][name] = (
            v["metrics"]["macro_f1"] if study == "feedback" else v["macro_f1"]
        )
        d["old_columns"][name] = {"point": d["old_groups"][name]}
    if study == "feedback":
        for cond, v in old["conditions"].items():
            for rep, x in enumerate(v["common_valid_runs"]):
                d["old_settings"][cond, rep] = {"macro": x["macro_f1"]}
    else:
        for k, v in old.get("settings", {}).items():
            cond, rep = k.rsplit("_rep", 1)
            x = v["common_valid"]
            entry = {"macro": x["macro_f1"]}
            counts = x.get("label_counts", x.get("counts_tp_fp_fn"))
            if counts is not None:
                entry["counts"] = counts
            d["old_settings"][cond, int(rep)] = entry
    return d


def feedback():
    root = B / "sq3_full_kb_feedback_20260928_v1"
    roster = {r["id"]: r for r in c.lines(root / "feasibility_v1/roster.jsonl")}
    science = c.read(root / "scientific_plan_v1.json")
    selection = c.read(root / "baseline_review_v1/selection.json")
    done = c.read(root / "updates_review_v1/complete.json")
    raw = {}
    for name, sha in done["outputs"].items():
        c.require(c.sha(c.ROOT / name) == sha, "Changed assembled feedback answer")
        raw.update(raw_rows(c.ROOT / name, key="id"))
    for rep in range(3):
        raw.update(
            raw_rows(
                root / "baseline_review_v1/assembled" / f"reference_rep{rep}.jsonl",
                key="id",
                baseline=True,
            )
        )
    for u in units("feedback"):
        old = original(u)
        ep = u["endpoint"].split("_b")[0]
        ids = selection["sentinel_ids"] if ep == "sentinel" else science["rosters"][ep]
        groups = {k: v["conditions"] for k, v in old["groups"].items()}
        yield contrasts(
            base(u, old, roster, raw, ids, groups, old["common_valid_ids"]), old
        )


def relevance():
    root = B / "sq3_full_kb_relevance_20260930_v1"
    roster = {
        r["id"]: r
        for r in c.lines(
            B / "sq3_full_kb_feedback_20260928_v1/feasibility_v1/roster.jsonl"
        )
    }
    science = c.read(root / "scientific_plan_v1.json")
    raw = raw_rows(root / "review_v1/predictions.jsonl", key="id")
    for u in units("relevance"):
        old = original(u)
        ids = science["rosters"][
            "evaluation" if u["endpoint"] == "all_transfer" else "primary_full_cover"
        ]
        groups = {
            k: (
                [k]
                if k == "reference"
                else [f"{k}_s{s}" for s in [202609281, 202609282, 202609283]]
            )
            for k in old["groups"]
        }
        yield contrasts(
            base(u, old, roster, raw, ids, groups, old["common_valid_ids"]), old
        )


def explanations():
    root = B / "sq3_full_kb_explanations_20261001_v1"
    roster = {r["id"]: r for r in c.lines(root / "roster.jsonl")}
    raw = raw_rows(
        B / "sq3_combined_results_20261004_v1/review/predictions.jsonl",
        key="id",
        study="explanations",
    )
    ids = sorted(
        i
        for i, r in roster.items()
        if r["gold"]["hate"] is True and r["gold"]["hate_type"] is not None
    )
    for u in units("explanations"):
        old = original(u)
        groups = {
            k: (
                [k]
                if k == "reference"
                else [f"{k}_s{s}" for s in [202609281, 202609282, 202609283]]
            )
            for k in old["groups"]
        }
        yield contrasts(
            base(u, old, roster, raw, ids, groups, old["common_valid_ids"]), old
        )


def balanced():
    for study, root, out in [
        (
            "balanced",
            "sq3_balanced_corrections_20261006_v1",
            "sq3_balanced_results_20261010_v1",
        ),
        (
            "ordinary_insertion",
            "sq3_five_slot_20261010_v2",
            "sq3_five_slot_results_20261012_v1",
        ),
    ]:
        u = units(study)[0]
        roster = {r["key"]: r for r in c.lines(B / root / "roster.jsonl")}
        raw = raw_rows(B / out / "review/predictions.jsonl")
        primary = original(u)
        parent = (c.ROOT / u["source"]).parent
        for file, view in [
            ("all_policies.json", "registered"),
            ("all_policies_strict.json", "strict"),
            ("representatives.json", "representatives"),
        ]:
            old = primary if view == "registered" else c.read(parent / file)
            d = contrasts(
                base(
                    u,
                    old,
                    roster,
                    raw,
                    old["original_ids"],
                    old["condition_groups"],
                    old["common_ids"],
                    view=view,
                    strict=view == "strict",
                ),
                old,
            )
            if view == "strict":

                d["original_bound_valid"] = d["nvalid"] > 0
                d["provenance"] = (
                    "Strict common-mask scores; historical own-valid bounds used all nonnull outputs. Both that original bound and fresh strict-valid bounds are retained separately."
                )
            yield d


def reserved():
    u = units("reserved")[0]
    allold = original(u)
    root = (c.ROOT / u["source"]).parent
    specs = c.read(root / "started.json")["metric_specifications"]
    details = c.read(root / "paired_run_details.json")
    roster = {
        r["key"]: r for r in c.lines(B / "sq3_reserved_slot_20261012_v1/roster.jsonl")
    }
    raw = {}
    for r in c.lines(root / "quality_outcomes.jsonl"):
        truth = roster[r["key"]]["gold"]
        if truth["hate"] is not True or truth["hate_type"] is None:
            continue
        key = r["condition"], r["repeat"], r["key"]
        c.require(key not in raw, "Duplicate reserved row")
        ans = (
            None
            if r["n_valid"] == 0
            else {"hate": r["predicted_hate"], "hate_type": r["predicted_types"]}
        )
        if ans is not None:
            c.require(
                r["gold_hate"] == truth["hate"]
                and set(r["gold_types"]) == set(truth["hate_type"]),
                "Reserved raw gold mismatch",
            )
        raw[key] = dict(result=ans, n_valid=r["n_valid"], gold=truth)
    for spec in specs:
        if not spec["fine"]:
            continue
        name = spec["name"]
        old = allold[name]
        ids = [
            i
            for i, r in roster.items()
            if i not in spec["excluded"]
            and r["gold"]["hate"] is True
            and r["gold"]["hate_type"] is not None
        ]
        groups = {
            "treatment": spec["spec"]["treatment_conditions"],
            "control": spec["spec"]["control_conditions"],
        }
        d = base(u, old, roster, raw, ids, groups, old["common_ids"], view=name)
        d["columns"]["treatment-control"] = {"treatment": 1, "control": -1}
        v = old["metrics"]["type_macro_f1_fixed7"]
        d["old_columns"] = {
            "treatment": {"point": v["treatment"]},
            "control": {"point": v["control"]},
            "treatment-control": {"point": v["difference"], "ci95": v["interval95"]},
        }
        for arm in details[name]["arms"]:
            for run in arm:
                d["old_settings"][run["condition"], run["repeat"]] = {
                    "macro": run["metrics"]["type_macro_f1_fixed7"],
                    "counts": [
                        [run["per_label"][l][k] for k in ["tp", "fp", "fn"]]
                        for l in p.TYPE
                    ],
                }
        d["old_draw_path"] = c.rel(root / "draws" / (name.replace("/", "__") + ".npy"))
        d["old_draw_columns"] = [("treatment-control", 0, 0, 0)]
        yield d


def modern(study):
    us = units(study)
    records = original(us[0])
    root = (c.ROOT / us[0]["source"]).parent
    if study == "rounds":
        records = records["records"]
    schema = c.read(root / "feature_schema.json")
    meta = list(c.lines(root / "metadata.jsonl"))
    pred = h.read_arrays(root / "prediction_codes.npz")
    features = h.read_arrays(root / "features.npz")
    for u in us:
        c.require(
            u["source_sha256"] == us[0]["source_sha256"],
            "Modern registry source conflict",
        )
        old = records[u["endpoint"]]
        head = u["dimension"]
        labels = u["labels"]
        eligible = old["eligible_indices"]
        common = old["common_indices"] if study == "transfer" else old["indices"]
        ids = [meta[i]["key"] for i in eligible]
        gold = np.array(
            [
                [
                    l
                    in meta[i][
                        (
                            "gold_hate_types"
                            if head == "hate_type"
                            else "gold_target_groups"
                        )
                    ]
                    for l in labels
                ]
                for i in eligible
            ],
            bool,
        )
        c.require(
            all(meta[i]["gold_hate"] is True for i in eligible), "Nonhateful fine row"
        )
        shape = pred[head].shape[1:]
        code = pred[head][eligible].reshape(len(eligible), -1).T
        vote = pred["votes"][eligible].reshape(len(eligible), -1).T
        gate = pred["gates"][eligible].reshape(len(eligible), -1).T > 0
        valid = code >= 0
        bit = (code[..., None] & (1 << np.arange(len(labels)))) != 0
        bit[~valid] = False
        feature = features[head][common].sum(0)
        h.equal(feature, old["counts"], study + " original feature totals")
        commonmask = np.array([i in set(common) for i in eligible])
        macroindices = old["macro_indices"]
        oldsetting = np.asarray(old["run_scores"])[..., 0].reshape(-1)
        positions = [
            schema["features"][head].index(l + ":" + k)
            for l in labels
            for k in ["tp", "fp", "fn"]
        ]
        oldcount = feature[..., positions].reshape(-1, len(labels), 3)
        h.equal(
            h.encode(gold, bit, valid)[:, commonmask].sum(1),
            oldcount,
            study + " raw codes versus saved counts",
        )
        if study == "transfer":
            settings = [
                (k, r)
                for k in [
                    "reference",
                    "pool_2026100441",
                    "pool_2026100442",
                    "pool_2026100443",
                ]
                for r in range(3)
            ]
            groups = {
                k: [j for j, v in enumerate(settings) if v[0] == k]
                for k in dict(settings)
            }
            groups["feedback"] = list(range(3, 12))
            columns = {k: {k: 1} for k in groups}
            columns["feedback-reference"] = {"feedback": 1, "reference": -1}
            oldcols = {}
            ref = old["metrics"]["macro_f1"]
            mic = old["metrics"]["micro_f1"]
            for name, key in [
                ("reference", "reference"),
                ("feedback", "treatment"),
                ("feedback-reference", "difference"),
            ]:
                oldcols[name] = dict(point=ref[key], micro=mic[key])
                if key == "difference":
                    oldcols[name].update(
                        ci95=ref["interval95"], micro_ci95=mic["interval95"]
                    )
            for n, name in enumerate(
                ["pool_2026100441", "pool_2026100442", "pool_2026100443"]
            ):
                oldcols[name] = dict(
                    point=ref["pool_values"][n], micro=mic["pool_values"][n]
                )
            bound = old["missing_bounds"]["metrics"]
            for name, key in [
                ("reference", "reference"),
                ("feedback", "treatment"),
                ("feedback-reference", "opposing_difference"),
            ]:
                oldcols[name].update(
                    missing_bounds=bound["macro_f1"][key],
                    micro_missing_bounds=bound["micro_f1"][key],
                )
            olddraw = [
                (name, j, m, m) for j, name in enumerate(columns) for m in [0, 1]
            ]
            seed = old["bootstrap"]["seed"]
            ix = list(range(len(settings)))
        else:
            allsettings = [
                (rd, k, r)
                for rd in range(4)
                for k in schema["conditions"]
                for r in range(3)
            ]
            rounds = sorted(
                {term[0] for terms in old["columns"].values() for term in terms}
            )
            ix = [i for i, k in enumerate(allsettings) if k[0] in rounds]
            settings = [allsettings[i] for i in ix]
            groups = {
                f"r{rd+1}_{policy}": [
                    j
                    for j, (r, k, rep) in enumerate(settings)
                    if r == rd and (k == policy or k.startswith(policy + "_s"))
                ]
                for rd in rounds
                for policy in schema["policies"]
            }
            columns = {
                name: {
                    f"r{rd+1}_{policy}": coef
                    for rd, policy, coef in old["columns"][name]
                }
                for name in old["column_order"]
            }
            oldcols = {
                name: dict(
                    point=old["estimates"][j][0],
                    micro=old["estimates"][j][1],
                    ci95=old["intervals"][j][0] if "intervals" in old else None,
                    micro_ci95=old["intervals"][j][1] if "intervals" in old else None,
                )
                for j, name in enumerate(old["column_order"])
            }
            if old.get("missing_bounds"):
                for name, entry in oldcols.items():
                    key = name[3:] if name.startswith("r4_") else None
                    if (
                        key is not None
                        and key in old["missing_bounds"]["metrics"]["macro_f1"]
                    ):
                        entry.update(
                            missing_bounds=old["missing_bounds"]["metrics"]["macro_f1"][
                                key
                            ],
                            micro_missing_bounds=old["missing_bounds"]["metrics"][
                                "micro_f1"
                            ][key],
                        )
            olddraw = [
                (name, j, m, m)
                for j, name in enumerate(old["column_order"])
                for m in [0, 1]
            ]
            seed = old.get("seed", 0)
        available = valid[ix].copy()
        if old["view"].startswith("strict"):
            available &= vote[ix] == 3
        d = dict(
            id=u["id"],
            unit_id=u["id"],
            study=study,
            endpoint=u["endpoint"],
            view=old["view"],
            ids=ids,
            gold=gold,
            pred=bit[ix],
            valid=available,
            gate=gate[ix],
            nvalid=vote[ix],
            mask=commonmask,
            settings=settings,
            groups=groups,
            columns=columns,
            labels=labels,
            macro_labels=[labels[i] for i in macroindices],
            macro_indices=macroindices,
            clusters=[meta[i]["inference_cluster"] for i in eligible],
            seed=seed,
            sampling="cluster",
            old_columns=oldcols,
            old_settings={
                settings[j]: dict(macro=oldsetting[i], counts=oldcount[i])
                for j, i in enumerate(ix)
            },
            old_draw_path=old.get("draw_file"),
            old_draw_columns=olddraw,
            descriptive=study == "rounds" and "intervals" not in old,
        )
        yield d


def transfer():
    yield from modern("transfer")


def rounds():
    yield from modern("rounds")


def target(r, name):
    if name == "dataset":
        return r["gold"]["hate"], r["gold"]["hate_type"]
    j = r["judgments"][name + "_project"]
    gate = j["binary_hate"]
    return gate, (
        None
        if gate is None
        else (
            []
            if gate is False
            else j["types"] if j["type_state"] == "assigned" else None
        )
    )


def human():
    u = units("human")[0]
    old = original(u)
    root = (c.ROOT / u["source"]).parent
    cases = c.read(c.ROOT / u["gold_and_cases_path"])
    outputs = c.read(c.ROOT / u["prediction_path"])
    raw = {
        (r["condition"], r["repeat"], r["key"]): dict(
            result=r["prediction"], n_valid=r["n_valid"]
        )
        for r in outputs
    }
    c.require(len(raw) == 216, "Human output count")
    tr = [r for r in cases if r["transfer_eligible"]]
    clusters = {}
    for r in sorted(tr, key=lambda x: x["packet_id"]):
        clusters.setdefault(r["cluster"], r)
    cohorts = dict(
        other_all=tr,
        other_changed_messages=[r for r in tr if r["selected_correction"] != "R059"],
        other_identical_messages=[r for r in tr if r["selected_correction"] == "R059"],
        both_reviewers_resolved=[
            r for r in tr if all(target(r, k)[0] is not None for k in ["A", "B"])
        ],
        exclude_retrieval_replay=[r for r in tr if not r["retrieval_replay_flag"]],
        exclude_training_overlap=[r for r in tr if not r["touches_training"]],
        one_per_overlap_cluster=list(clusters.values()),
        all_three_votes_valid=tr,
    )
    for cohort, rows in cohorts.items():
        for name in ["A", "B", "dataset"]:
            saved = old["transfer"][cohort][name]["labels"]
            eligible = [
                r
                for r in rows
                if target(r, name)[0] is True and target(r, name)[1] is not None
            ]
            roster = {
                r["key"]: {
                    "gold": {"hate": True, "hate_type": target(r, name)[1]},
                    "cluster": r["cluster"],
                }
                for r in eligible
            }
            ids = list(roster)
            c.require(
                len(ids) == saved["annotation_eligible_texts"], "Human annotation mask"
            )
            groups = {k: [k] for k in saved["arms"]}
            d = base(
                u,
                {},
                roster,
                raw,
                ids,
                groups,
                ids,
                view=cohort + "/" + name,
                strict=cohort == "all_three_votes_valid",
            )
            d["descriptive"] = True
            for arm, v in saved["arms"].items():
                d["old_groups"][arm] = v["mean_macro_f1"]
                d["old_columns"][arm] = {"point": v["mean_macro_f1"]}
                for rep, r in enumerate(v["per_repeat"]):
                    d["old_settings"][arm, rep] = dict(
                        macro=r["macro_f1_fixed_seven"],
                        counts=[
                            [r["labels"][l][k] for k in ["tp", "fp", "fn"]]
                            for l in p.TYPE
                        ],
                    )
            for a, b in [
                ("human_feedback5", "dataset_feedback5"),
                ("human_feedback5", "reference5"),
                ("dataset_feedback5", "reference5"),
            ]:
                d["columns"][a + "-" + b] = {a: 1, b: -1}
            d["selection_notes"] = dict(
                cohort=cohort,
                target=name,
                total_cohort=len(rows),
                eligible_aliases=[r["packet_id"] for r in eligible],
                primary_complete_outcome=old["transfer"][cohort][name]["complete"][
                    "arms"
                ],
                interpretation="Selected descriptive sample. Original annotation target and eligible mask; no adjudication, no causal or population claim.",
            )
            yield d


def extras():
    c.read(p.BASE / "coverage_addendum.json")
    root = B / "sq3_additions_results_20261006_v1"
    roster = {
        r["key"]: r
        for r in c.lines(B / "sq3_additions_20261004_v1/human_meaning_v2/roster.jsonl")
    }
    raw = raw_rows(root / "review/predictions.jsonl", study="human_meaning")
    u = dict(
        id="dataset_meanings/hate_type",
        study="dataset_meanings",
        endpoint="hate_type",
        labels=p.TYPE,
        macro_labels=p.TYPE,
    )
    for filename, view in [
        ("primary", "primary"),
        ("all_three_valid", "strict"),
        ("representatives", "representatives"),
        ("all_policy_hateful", "all_policies"),
    ]:
        old = c.read(root / "human_meaning" / (filename + ".json"))
        d = contrasts(
            base(
                u,
                old,
                roster,
                raw,
                old["original_ids"],
                old["condition_groups"],
                old["common_ids"],
                view=view,
                strict=view == "strict",
            ),
            old,
        )
        if view == "strict":
            d["original_bound_valid"] = d["nvalid"] > 0
        d["provenance"] = (
            "Coverage addendum: dataset-provided meanings, distinct from assistant explanations and the two-person study. Nominal95 supporting micro, no enlarged primary family."
        )
        yield d
    root = B / "sq3_full_kb_feedback_20260928_v1"
    roster = {r["id"]: r for r in c.lines(root / "feasibility_v1/roster.jsonl")}
    old = c.read(root / "results_v1/candidate_diagnostics.json")
    science = c.read(root / "scientific_plan_v1.json")
    pools = {
        r["condition"]: r
        for r in c.read(root / "update_payloads_v1/pools.json")["pools"]
    }
    raw = {}
    done = c.read(root / "updates_review_v1/complete.json")
    for name, digest in done["outputs"].items():
        c.require(c.sha(c.ROOT / name) == digest, "Uptake saved run changed")
        raw.update(raw_rows(c.ROOT / name, key="id"))
    for rep in range(3):
        raw.update(
            raw_rows(
                root / "baseline_review_v1/assembled" / f"reference_rep{rep}.jsonl",
                key="id",
                baseline=True,
            )
        )
    u = dict(
        id="source_uptake/hate_type",
        study="source_uptake",
        endpoint="hate_type",
        labels=p.TYPE,
        macro_labels=p.TYPE,
    )
    for record in old:
        cond = record["condition"]
        pool = pools[cond]
        ids = (
            pool["ids"]
            if record["role"] == "selected"
            else sorted(
                set(science["rosters"]["eligible_candidates"]) - set(pool["ids"])
            )
        )
        common = sorted(set(ids) - set(record["excluded_ids"]))
        d = base(
            u,
            {},
            roster,
            raw,
            ids,
            {"reference": ["reference"], "update": [cond]},
            common,
            view=cond + "/" + record["role"],
        )
        d["descriptive"] = True
        d["columns"]["update-reference"] = {"update": 1, "reference": -1}
        for arm, runs in record["runs"].items():
            name = "reference" if arm == "reference" else cond
            for rep, r in enumerate(runs):
                d["old_settings"][name, rep] = dict(
                    macro=r["macro_f1"],
                    counts=[
                        [r["per_label"][l][k] for k in ["tp", "fp", "fn"]]
                        for l in p.TYPE
                    ],
                )
            d["old_columns"][arm] = {
                "point": record["summary"][arm]["macro_f1"]["mean"]
            }
        d["selection_notes"] = {
            k: record[k] for k in ["arm", "role", "pool_size", "seed", "interpretation"]
        }
        yield d
