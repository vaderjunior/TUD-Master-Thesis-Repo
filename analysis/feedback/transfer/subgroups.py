import os

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import sys
import unittest
import numpy as np
from analysis.feedback.transfer import archive as a
from analysis.feedback.transfer import effects_metrics as k

OUT = a.ROOT / "experiments/astra/sq3_transfer_part3_20261023_v1"
P2 = a.ROOT / "experiments/astra/sq3_transfer_part2_20261023_v1"
SOURCES = ["hatexplain", "implicit_hate", "mhs"]
FUNCTIONS = {
    "derog_neg_emote_h": "Negative emotions",
    "derog_neg_attrib_h": "Negative attributes",
    "derog_dehum_h": "Dehumanisation",
    "derog_impl_h": "Implicit derogation",
    "threat_dir_h": "Direct threats",
    "threat_norm_h": "Normative threats",
    "slur_h": "Hateful slurs",
    "slur_homonym_nh": "Non-hateful slur homonyms",
    "slur_reclaimed_nh": "Reclaimed slurs",
    "profanity_h": "Hateful profanity",
    "profanity_nh": "Non-hateful profanity",
    "ref_subs_clause_h": "Reference in a later clause",
    "ref_subs_sent_h": "Reference in a later sentence",
    "negate_pos_h": "Negated positive statements",
    "negate_neg_nh": "Negated hateful statements",
    "phrase_question_h": "Hate phrased as a question",
    "phrase_opinion_h": "Hate phrased as an opinion",
    "ident_neutral_nh": "Neutral identity mentions",
    "ident_pos_nh": "Positive identity mentions",
    "counter_quote_nh": "Counter-speech quoting hate",
    "counter_ref_nh": "Counter-speech referring to hate",
    "target_obj_nh": "Abuse of objects",
    "target_indiv_nh": "Abuse of individuals",
    "target_group_nh": "Abuse of unprotected groups",
    "spell_char_swap_h": "Swapped characters",
    "spell_char_del_h": "Deleted characters",
    "spell_space_add_h": "Added spaces",
    "spell_space_del_h": "Removed spaces",
    "spell_leet_h": "Leetspeak substitutions",
}
HEADS = ("hate_types", "target_groups", "severity", "legal")


def save(name, value):
    path = OUT / name
    a.need(path.resolve().is_relative_to(OUT), "Write outside Part 3")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(
            value if isinstance(value, bytes) else a.encoded(value, pretty=True)
        )
    return a.rel(path)


def seed(name):
    return int.from_bytes(
        hashlib.sha256(("A16-part3-support-v1|" + name).encode()).digest()[:8], "little"
    )


def annotation(row):
    raw = json.loads(row["raw"])
    out = {
        name: row[name]
        for name in (
            "key",
            "id",
            "population",
            "source",
            "inference_cluster",
            "functional_cluster",
            "text_cluster",
            "overlap_cluster",
        )
    }
    out["gold_hate"] = row["gate"]
    out["fine_gold"] = {name: row.get(name) for name in HEADS}
    source = row["source"]
    if source == "hatecheck":
        fn = raw["functionality"]
        a.need(
            fn in FUNCTIONS and row["gate"] == fn.endswith("_h"),
            "Unknown function or wrong polarity",
        )
        ident = raw["target_ident"]
        if ident is None or (isinstance(ident, float) and math.isnan(ident)):
            ident = "not_specified"
        a.need(
            isinstance(ident, str) and ident != "nan", "Unexpected identity encoding"
        )
        out.update(
            functionality=fn,
            template_id=raw["templ_id"],
            case_id=raw["case_id"],
            ref_case_id=raw["ref_case_id"],
            target_identity=ident,
            block=raw["block"],
            direction=raw["direction"],
        )
    elif source == "hatexplain":
        votes = raw["votes"]
        counter = Counter(votes)
        majority, count = counter.most_common(1)[0]
        a.need(
            len(votes) == 3
            and count >= 2
            and row["gate"] == (majority == "hatespeech"),
            "HateXplain mapping differs",
        )
        out.update(
            majority_label=majority,
            annotation_agreement="unanimous" if count == 3 else "split",
            annotator_votes=votes,
            original_split=raw["official_split"],
        )
    elif source == "implicit_hate":
        a.need(
            raw["stage1_class"]
            in (
                "not_hate",
                "explicit_hate",
                "implicit_hate",
                "implicit_hate (stg2-only)",
            ),
            "Unknown stage class",
        )
        a.need(
            row["gate"] == (raw["stage1_class"] != "not_hate"),
            "Implicit Hate mapping differs",
        )
        out["stage1_class"] = raw["stage1_class"]
    elif source == "mhs":
        score = raw["score"]
        a.need(
            row["gate"] == (score > 0.5),
            "MHS rounded raw score and frozen gate disagree",
        )
        expected = (
            ("low" if score <= 1 else "medium" if score <= 1.63 else "high")
            if row["gate"]
            else None
        )
        a.need(row["severity"] == expected, "MHS frozen severity band differs")
        out.update(score=score, annotation_row_count=raw["n_annotators"])
    else:
        raise AssertionError("Unknown source " + source)
    return out


def load():
    meta = []
    errors = np.full((15758, 4, 3), -1, dtype=np.int8)
    votes = np.zeros_like(errors)
    for i, (roster, matrix) in enumerate(
        zip(a.rows(a.OUT / "roster.jsonl"), a.rows(a.OUT / "matrix.jsonl"), strict=True)
    ):
        m = annotation(roster)
        a.need(
            i == matrix["input_index"]
            and m["key"] == matrix["key"]
            and m["gold_hate"] == matrix["gold_hate"],
            "Roster join differs",
        )
        for field in (
            "source",
            "population",
            "inference_cluster",
            "functional_cluster",
            "text_cluster",
        ):
            a.need(m[field] == matrix[field], "Metadata differs: " + field)
        m.update(
            input_index=i,
            stage=matrix["stage"],
            execution_period=matrix["execution_period"],
        )
        meta.append(m)
        for ci, condition in enumerate(matrix["predictions"]):
            for ri, pred in enumerate(condition):
                votes[i, ci, ri] = pred["n_valid"]
                if pred["result"] is not None:
                    errors[i, ci, ri] = int(pred["result"]["hate"] != m["gold_hate"])
        a.need(
            matrix["masks"]["common_valid"] == bool(np.all(errors[i] >= 0)),
            "Common mask differs",
        )
    a.need(
        len(meta) == 15758 and len({r["key"] for r in meta}) == 15758,
        "Roster shape differs",
    )
    a.need(np.array_equal(errors < 0, votes == 0), "Missingness differs")
    return meta, errors, votes


def subset(meta, error, votes, members, truth, view, exclusions):
    strict, representative, exclude = k.VIEWS[view]
    eligible = [
        int(i)
        for i in members
        if meta[i]["gold_hate"] == truth
        and (not exclude or meta[i]["key"] not in exclusions)
    ]
    if representative:
        groups = defaultdict(list)
        for i in eligible:
            groups[meta[i]["inference_cluster"]].append(i)
        keep = {
            min(
                ids,
                key=lambda i: hashlib.sha256(
                    (k.REP_PREFIX + meta[i]["key"]).encode()
                ).hexdigest(),
            )
            for ids in groups.values()
        }
        eligible = [i for i in eligible if i in keep]
    populations = {meta[i]["population"] for i in members}
    a.need(len(populations) == 1, "Mixed populations in subgroup")
    return k.cohort(
        meta,
        error,
        votes,
        next(iter(populations)),
        truth,
        strict=strict,
        eligible_override=eligible,
    )


def group_definitions(meta):
    groups = {}

    def add(name, indices, truths, views, boots, family):
        a.need(len(indices) > 0 and name not in groups, "Empty or duplicate group")
        groups[name] = dict(
            indices=indices,
            truths=truths,
            views=views,
            bootstrap_views=boots,
            family=family,
        )

    for source in SOURCES:
        ids = [i for i, r in enumerate(meta) if r["source"] == source]
        add(
            "source/" + source,
            ids,
            [False, True],
            list(k.VIEWS),
            list(k.VIEWS),
            "planned_source",
        )
    for fn in FUNCTIONS:
        ids = [i for i, r in enumerate(meta) if r.get("functionality") == fn]
        add(
            "function/" + fn,
            ids,
            [False, True],
            list(k.VIEWS),
            ["main", "strict", "replay_excluded"],
            "planned_function",
        )
    for source, field in [
        ("hatexplain", "majority_label"),
        ("implicit_hate", "stage1_class"),
        ("mhs", "severity"),
    ]:
        ids = [i for i, r in enumerate(meta) if r["source"] == source]
        strata = defaultdict(list)
        for i in ids:
            r = meta[i]
            value = (
                r["majority_label"] + "_" + r["annotation_agreement"]
                if source == "hatexplain"
                else (
                    r[field]
                    if source == "implicit_hate"
                    else r["fine_gold"]["severity"]
                )
            )
            if value is not None:
                strata[str(value)].append(i)
        for value, members in sorted(strata.items()):
            truth = meta[members[0]]["gold_hate"]
            a.need(
                all(meta[i]["gold_hate"] == truth for i in members),
                "Subtype crosses polarity",
            )
            add(
                "subtype/" + source + "/" + value,
                members,
                [truth],
                ["main", "strict"],
                ["main"],
                "exploratory_annotation_subtype",
            )
    for field in ("block", "target_identity", "direction"):
        strata = defaultdict(list)
        for i, r in enumerate(meta):
            if r["source"] == "hatecheck":
                strata[r[field]].append(i)
        for value, members in sorted(strata.items()):
            add(
                "hatecheck_" + field + "/" + value,
                members,
                [False, True],
                ["main", "strict"],
                [],
                "exploratory_metadata_points",
            )
    return groups


def prepare():
    a.need(
        not OUT.exists(),
        "Preserve earlier Part 3 output; create an explicit new version",
    )
    bindings = {}
    for checkpoint in (a.OUT / "complete.json", P2 / "complete.json"):
        receipt = a.read(checkpoint)
        for name, expected in {
            **receipt["input_bindings"],
            **receipt["artifacts"],
        }.items():
            a.need(
                a.sha(a.ROOT / name) == expected, "Changed completed evidence: " + name
            )
            bindings[name] = expected
        bindings[a.rel(checkpoint)] = a.sha(checkpoint)
    for name in [
        "src/hsrag/data/hatexplain.py",
        "src/hsrag/data/implicit_hate.py",
        "src/hsrag/data/mhs.py",
        "scripts/make_hatecheck_subset.py",
        "data/mappings/mhs_thresholds.md",
        "analysis/classification/prepare.py",
        "Thesis_Log/astra_result/astra_sq3_transfer_review_plan_20261023_v1.md",
    ]:
        bindings[name] = a.sha(a.ROOT / name)
    from tests.feedback.transfer import test_subgroups as tests

    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(tests)
    )
    a.need(result.wasSuccessful(), "Part 3 fixtures failed: " + log.getvalue())
    for path in (Path(__file__), Path(tests.__file__), Path(k.__file__)):
        bindings[a.rel(path)] = a.sha(path)
    contract = dict(
        status="part3_conventions_recorded_before_subgroup_scoring",
        created_utc=a.utc(),
        bindings=bindings,
        planned_scope="All three natural sources, all 29 HateCheck functions, all fixed pools/repeats, template/cluster structure and known fine-label supports.",
        functions=FUNCTIONS,
        views=k.VIEWS,
        bootstrap=dict(
            draws=10000,
            generator="PCG64",
            confidence=0.95,
            seed_prefix="A16-part3-support-v1|",
            seed_conversion="First eight SHA256 bytes, little endian",
            unit="Sorted frozen inference cluster IDs within the endpoint/group; preserve all repeats and pools.",
            source_views="Six views, reusing the six independently verified Part 2 main-source draw sets.",
            function_views="Main, strict and replay-excluded; other three masks are point diagnostics.",
            subtype_views="Main only; strict points also retained.",
            weighting="Item-weighted within resampled clusters; equal repeats and then equal fixed pools, reference once.",
        ),
        representatives="Supporting one-per-cluster views select minimum SHA256(A16-representative| + key) within each subgroup/polarity, after replay exclusions and BEFORE checking validity. Function representatives are therefore local to each function, not a partition of the global representative sample.",
        exploratory_subtypes="HateXplain original majority label crossed with unanimity; all four Implicit Hate stage classes; MHS existing low/medium/high hateful severity bands. No new outcome-selected score thresholds.",
        extra_metadata="HateCheck original block, mentioned identity and direction: point descriptions only, not causal coverage or demographic fairness tests.",
        templates="All original templates and frozen connected inference clusters get main/strict point tables. Function-specific leave-one-cluster-out points use a fixed common mask. They are not confidence intervals.",
        linked_cases="Every original ref_case_id edge is retained. On paired common-valid endpoints, report both-cases-correct across conditions/repeats, separated by child function and whether gold polarity changes. These are dependent diagnostic pairs, not independent new texts or a causal minimal-pair experiment.",
        inference_limits="Nominal supporting 95% intervals, no new confirmatory family and no significance-based function selection. A difference of subgroup point estimates is not a tested subgroup interaction. Few clusters and degenerate all-correct bootstraps limit generalisation. Never infer independent functions from their separate intervals.",
        fine_gold="Count known, unknown, known-empty and per-label positive gold on full/common/strict cohorts. No fine-output accuracy in this part.",
        new_Qwen_calls=0,
        new_fits=0,
        new_embeddings=0,
        environment=dict(
            python=sys.version, numpy=np.__version__, platform=platform.platform()
        ),
    )
    save("analysis_contract.json", contract)
    save(
        "focused_tests.json",
        dict(passed=True, tests_run=result.testsRun, log=log.getvalue()),
    )
    return contract


def count_support(meta, error, votes, groups):
    out = {}
    for name, members in groups.items():
        views = {}
        for view in ("full", "common", "strict"):
            ids = [
                i
                for i in members
                if view == "full"
                or (
                    np.all(error[i] >= 0)
                    and (view != "strict" or np.all(votes[i] == 3))
                )
            ]
            heads = {}
            for head in HEADS:
                known = [
                    i
                    for i in ids
                    if meta[i]["fine_gold"][head] is not None
                    and (head == "legal" or meta[i]["gold_hate"])
                ]
                positives = Counter()
                for i in known:
                    value = meta[i]["fine_gold"][head]
                    positives.update([value] if isinstance(value, str) else value)
                heads[head] = dict(
                    eligible_known=len(known),
                    unknown_or_ineligible=len(ids) - len(known),
                    known_empty=sum(meta[i]["fine_gold"][head] == [] for i in known),
                    positive_support=dict(sorted(positives.items())),
                )
            views[view] = dict(
                n=len(ids),
                gold_hateful=sum(meta[i]["gold_hate"] for i in ids),
                heads=heads,
            )
        out[name] = views
    return out


def structure(meta):
    hc = [r for r in meta if r["source"] == "hatecheck"]
    units = {}
    for field in ("template_id", "functional_cluster", "inference_cluster"):
        groups = defaultdict(list)
        for r in hc:
            groups[str(r[field])].append(r)
        units[field] = dict(
            count=len(groups),
            size_histogram=dict(
                sorted(Counter(len(v) for v in groups.values()).items())
            ),
            largest_size=max(map(len, groups.values())),
            crosses_function=sum(
                len({r["functionality"] for r in v}) > 1 for v in groups.values()
            ),
            crosses_gold=sum(
                len({r["gold_hate"] for r in v}) > 1 for v in groups.values()
            ),
            members={
                label: dict(
                    keys=[r["key"] for r in rows],
                    functions=sorted({r["functionality"] for r in rows}),
                    gold_counts=dict(Counter(str(r["gold_hate"]) for r in rows)),
                )
                for label, rows in sorted(groups.items())
            },
        )
    natural = {}
    for source in SOURCES:
        rows = [r for r in meta if r["source"] == source]
        natural[source] = dict(
            n=len(rows),
            gold_counts=dict(Counter(str(r["gold_hate"]) for r in rows)),
            clusters=len({r["inference_cluster"] for r in rows}),
        )
        if source == "hatexplain":
            natural[source]["original_splits"] = dict(
                Counter(r["original_split"] for r in rows)
            )
        if source == "mhs":
            natural[source]["annotation_row_count_histogram"] = dict(
                sorted(Counter(r["annotation_row_count"] for r in rows).items())
            )
            natural[source]["score_by_gold"] = {
                str(t): np.quantile(
                    [r["score"] for r in rows if r["gold_hate"] == t],
                    [0, 0.25, 0.5, 0.75, 1],
                ).tolist()
                for t in (False, True)
            }
    return dict(
        hatecheck=units,
        natural=natural,
        caveats="HateCheck original templates have one function, but connected components may cross functions and polarity. Raw MHS n_annotators counts annotation rows, not proven unique people. Project TEST does not equal the original HateXplain official test split.",
    )


def calculate():
    a.offline()
    contract = prepare()
    meta, errors, votes = load()
    science = a.read(a.STUDY / "scientific_plan_v1.json")
    exclusions = set(science["mandatory_replay_keys"])
    conditions = science["conditions"]
    definitions = group_definitions(meta)
    prior = a.read(P2 / "rates.json")
    save("metadata.jsonl", b"".join(a.encoded(r) for r in meta))
    save("group_definitions.json", definitions)
    save("structure.json", structure(meta))
    support_groups = {
        s: [i for i, r in enumerate(meta) if r["source"] == s]
        for s in [*SOURCES, "hatecheck"]
    }
    support_groups.update(
        {
            "function/" + fn: d["indices"]
            for name, d in definitions.items()
            if name.startswith("function/")
            for fn in [name.split("/")[1]]
        }
    )
    save("known_label_support.json", count_support(meta, errors, votes, support_groups))
    rates = {}
    descriptions = {}
    contributions = {}
    omissions = {}
    tables = {}
    new_draw_sets = reused_draw_sets = 0
    main_denominators = {
        f"{pop}_{ep}": prior[f"main/{pop}_{ep}"]["common_n"]
        for pop in ("en_test", "en_hatecheck_eval")
        for ep in ("FPR", "FNR")
    }
    for name, spec in definitions.items():
        members = spec["indices"]
        pop = meta[members[0]]["population"]
        for truth in spec["truths"]:
            ep = "FNR" if truth else "FPR"
            for view in spec["views"]:
                ident = name + "/" + view + "/" + ep
                c = subset(meta, errors, votes, members, truth, view, exclusions)
                record = dict(
                    id=ident,
                    group=name,
                    family=spec["family"],
                    endpoint=ep,
                    population=pop,
                    view=view,
                    **c["point"],
                    eligible_indices=c["eligible"].tolist(),
                    common_indices=c["common"].tolist(),
                    cluster_count=len(
                        {meta[i]["inference_cluster"] for i in c["common"]}
                    ),
                    template_count=(
                        len({meta[i]["template_id"] for i in c["common"]})
                        if pop == "en_hatecheck_eval"
                        else None
                    ),
                    status=(
                        "defined"
                        if len(c["common"])
                        else (
                            "undefined_no_eligible_gold"
                            if not len(c["eligible"])
                            else "undefined_no_common_outputs"
                        )
                    ),
                )
                if view in spec["bootstrap_views"] and len(c["common"]):
                    if name.startswith("source/") and view == "main":
                        key = name + "/en_test_" + ep
                        reuse = prior[key]
                        a.need(
                            c["point"]["difference"] == reuse["difference"]
                            and c["common"].tolist() == reuse["common_indices"],
                            "Source reuse differs",
                        )
                        record.update(
                            bootstrap=reuse["bootstrap"],
                            interval95=reuse["interval95"],
                            pool_difference_intervals95=reuse[
                                "pool_difference_intervals95"
                            ],
                            draw_file=reuse["draw_file"],
                            reused_from=dict(
                                path=a.rel(P2 / "rates.json"),
                                id=key,
                                proof=a.rel(P2 / "independent_check.json"),
                            ),
                        )
                        reused_draw_sets += 1
                    else:
                        draws, info = k.bootstrap(meta, errors, c, seed(ident))
                        draw_path = (
                            OUT
                            / "draws"
                            / (hashlib.sha256(ident.encode()).hexdigest()[:24] + ".npy")
                        )
                        draw_path.parent.mkdir(parents=True, exist_ok=True)
                        with draw_path.open("xb") as f:
                            np.save(f, draws, allow_pickle=False)
                        intervals = k.intervals(draws)
                        record.update(
                            bootstrap=info,
                            interval95=intervals["interval95"],
                            pool_difference_intervals95=intervals[
                                "pool_difference_intervals95"
                            ],
                            draw_file=a.rel(draw_path),
                        )
                        new_draw_sets += 1
                    record["interval_role"] = (
                        "Supporting nominal 95%; no correction for the collection of subgroup views."
                    )
                    record["degenerate_difference_draws"] = (
                        record["interval95"][0] == record["interval95"][1]
                    )
                rates[ident] = record
                if view == "main" and len(c["common"]):
                    descriptions[ident], _ = k.description(
                        meta, errors, votes, c, conditions
                    )
                    numerator = (
                        record["treatment_error_count"]
                        - 3 * record["reference_error_count"]
                    )
                    contributions[ident] = dict(
                        difference_ninths=numerator,
                        endpoint_denominator=main_denominators[pop + "_" + ep],
                        contribution_to_population_rate=numerator
                        / (9 * main_denominators[pop + "_" + ep]),
                    )
                    labels, counts, sizes, _ = k.grouped(meta, errors, c["common"])
                    total_counts = counts.sum(axis=0)
                    total_n = int(sizes.sum())
                    omitted = []
                    for label, counts_i, n_i in zip(labels, counts, sizes):
                        remaining = total_n - int(n_i)
                        sums = total_counts - counts_i
                        delta = (
                            (int(sums[1:].sum()) - 3 * int(sums[0])) / (9 * remaining)
                            if remaining
                            else None
                        )
                        omitted.append(
                            dict(
                                cluster=label,
                                removed_n=int(n_i),
                                remaining_n=remaining,
                                difference=delta,
                            )
                        )
                    omissions[ident] = omitted
        if name.startswith(("source/", "function/", "subtype/")):
            print("Reviewed " + name, flush=True)
    for field in ("template_id", "inference_cluster"):
        units = defaultdict(list)
        for i, r in enumerate(meta):
            if r["source"] == "hatecheck":
                units[str(r[field])].append(i)
        table = []
        for unit, members in sorted(units.items()):
            for truth in (False, True):
                if not any(meta[i]["gold_hate"] == truth for i in members):
                    continue
                for view in ("main", "strict"):
                    c = subset(meta, errors, votes, members, truth, view, exclusions)
                    table.append(
                        dict(
                            unit=unit,
                            field=field,
                            gold_hate=truth,
                            view=view,
                            functions=sorted(
                                {meta[i]["functionality"] for i in c["eligible"]}
                            ),
                            eligible_indices=c["eligible"].tolist(),
                            common_indices=c["common"].tolist(),
                            **c["point"],
                        )
                    )
        tables[field] = table
    by_case = {
        r["case_id"]: i for i, r in enumerate(meta) if r["source"] == "hatecheck"
    }
    pairs = []
    linked = defaultdict(list)
    for i, r in enumerate(meta):
        if r["source"] != "hatecheck" or r["ref_case_id"] is None:
            continue
        j = by_case[r["ref_case_id"]]
        a.need(
            r["inference_cluster"] == meta[j]["inference_cluster"],
            "Linked cases separated across inference units",
        )
        pair = dict(
            child=i,
            parent=j,
            child_key=r["key"],
            parent_key=meta[j]["key"],
            function=r["functionality"],
            parent_function=meta[j]["functionality"],
            gold_relation=(
                "opposite" if r["gold_hate"] != meta[j]["gold_hate"] else "same"
            ),
            cluster=r["inference_cluster"],
        )
        pairs.append(pair)
        linked[(pair["function"], pair["gold_relation"])].append(pair)
    pair_summaries = []
    for (fn, relation), members in sorted(linked.items()):
        for view in ("main", "strict"):
            eligible = [
                p
                for p in members
                if np.all(errors[[p["child"], p["parent"]]] >= 0)
                and (view != "strict" or np.all(votes[[p["child"], p["parent"]]] == 3))
            ]
            counts = np.zeros((4, 3), dtype=np.int64)
            for p in eligible:
                counts += (errors[p["child"]] == 0) & (errors[p["parent"]] == 0)
            n = len(eligible)
            ref = int(counts[0].sum()) / (3 * n) if n else None
            treat = int(counts[1:].sum()) / (9 * n) if n else None
            pair_summaries.append(
                dict(
                    function=fn,
                    gold_relation=relation,
                    view=view,
                    eligible_pairs=len(members),
                    common_pairs=n,
                    clusters=len({p["cluster"] for p in eligible}),
                    condition_repeat_both_correct=counts.tolist(),
                    reference_both_correct=ref,
                    treatment_both_correct=treat,
                    difference=treat - ref if n else None,
                )
            )
    save("rates.json", rates)
    save("descriptions.json", descriptions)
    save("contributions.json", contributions)
    save("leave_one_cluster_out.json", omissions)
    save("template_cluster_points.json", tables)
    save("linked_reference_pairs.json", pairs)
    save("linked_reference_summary.json", pair_summaries)
    artifacts = {a.rel(p): a.sha(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
    save(
        "calculated.json",
        dict(
            status="part3_calculated_independent_verification_pending",
            bindings=contract["bindings"],
            artifacts=artifacts,
            new_draw_sets=new_draw_sets,
            reused_draw_sets=reused_draw_sets,
            rate_views=len(rates),
            template_cluster_points=sum(map(len, tables.values())),
            linked_pairs=len(pairs),
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
            completed_utc=a.utc(),
        ),
    )
    print(
        json.dumps(
            dict(
                new_draw_sets=new_draw_sets,
                reused_draw_sets=reused_draw_sets,
                rate_views=len(rates),
                linked_pairs=len(pairs),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    calculate()
