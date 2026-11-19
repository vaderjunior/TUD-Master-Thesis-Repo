import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
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
from analysis.feedback.transfer import quality_metrics as k

OUT = a.ROOT / "experiments/astra/sq3_transfer_part4_20261023_v1"
P3 = a.ROOT / "experiments/astra/sq3_transfer_part3_20261023_v1"
HEADS = ["binary", "hate_type", "target_group", "severity", "known_answer"]
POPS = ["en_test", "en_hatecheck_eval"]
FIELD = {
    "hate_type": "hate_types",
    "target_group": "target_groups",
    "severity": "severity",
    "legal": "legal",
}


def native(x):
    if isinstance(x, np.ndarray):
        return native(x.tolist())
    if isinstance(x, np.generic):
        return native(x.item())
    if isinstance(x, float) and (not math.isfinite(x)):
        return None
    if isinstance(x, dict):
        return {str(key): native(v) for key, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [native(v) for v in x]
    return x


def save(name, value):
    path = OUT / name
    a.need(path.resolve().is_relative_to(OUT), "Write left new analysis")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(
            value if isinstance(value, bytes) else a.encoded(native(value), pretty=True)
        )
    return a.rel(path)


def save_array(name, values):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        np.save(f, values, allow_pickle=False)
    return a.rel(path)


def prepare():
    a.need(not OUT.exists(), "Preserve earlier Part 4 evidence")
    done = a.read(P3 / "complete.json")
    a.need(done["status"] == "transfer_part3_complete_reviewed", "Part 3 not ready")
    bindings = {
        **done["input_bindings"],
        **done["artifacts"],
        **done["analysis_sources"],
        a.rel(P3 / "complete.json"): a.sha(P3 / "complete.json"),
    }
    for name, digest in bindings.items():
        a.need(a.sha(a.ROOT / name) == digest, "Changed completed evidence " + name)
    from tests.feedback.transfer import test_quality as tests

    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(tests)
    )
    a.need(result.wasSuccessful(), "Quality fixtures failed: " + log.getvalue())
    for p in [
        Path(__file__),
        Path(k.__file__),
        Path(tests.__file__),
        a.ROOT / "analysis/common/scoring.py",
        a.ROOT / "analysis/feedback/reserved/support.py",
        a.ROOT
        / "experiments/astra/later_evaluation_20261008_v1/scoring_contract_v1.json",
    ]:
        bindings[a.rel(p)] = a.sha(p)
    science = a.read(a.STUDY / "scientific_plan_v1.json")
    labels = {
        pop: {
            head: (
                k.LABELS[head]
                if head == "binary"
                else (
                    []
                    if head == "known_answer"
                    else [
                        lab
                        for lab in k.LABELS[head]
                        if lab
                        in science["fixed_support"][pop][FIELD[head]]["support_ge_10"]
                    ]
                )
            )
            for head in [*HEADS, "legal"]
        }
        for pop in POPS
    }
    contract = dict(
        status="part4_conventions_recorded_before_quality_scoring",
        created_utc=a.utc(),
        bindings=bindings,
        labels=k.LABELS,
        population_macro_labels=labels,
        full_gold_support=science["fixed_support"],
        minimum_full_gold_support=10,
        heads=HEADS,
        explicit_unscorable_head="Legal has no known gold in either population; HateCheck has no type or severity gold.",
        fixed_labels="Population full-roster label support >=10 is fixed across all masks and resamples, including source/function views. Source-local support>=10 F1 is a separately labelled descriptive sensitivity. Micro F1 and exact sets retain all taxonomy labels, including absent/rare labels.",
        zero_denominators="Per-label precision/recall/F1 use zero for zero denominator. Empty cohorts are undefined. Balanced accuracy requires both gold classes. Missing severity categories are valid-output errors, not low severity; ordinal errors require an issued severity category and retain coverage.",
        averaging="Score each of the twelve actual runs separately; average three repeats per condition and then the three pools equally. Reference has three outputs, counted once. Never compute F1 from pooled run counts. Difference is an algebraic mean of matched-repeat score differences, not a controlled random-seed intervention.",
        masks=k.VIEWS,
        representatives="Minimum SHA256(A16-representative| + key) within eligible head/group, gold polarity and inference cluster, after replay exclusions and before validity. Binary follows the Part 2 endpoint-specific representative convention.",
        missingness="Common-valid across all four conditions/repeats; own-valid runs and opposing whole-eligible bounds accompany contrasts. Strict requires all three votes for every output. Null results never become negative predictions.",
        bootstrap=dict(
            draws=10000,
            confidence=0.95,
            generator="PCG64",
            seed_prefix="A16-part4-quality-v1|",
            seed_conversion="First eight SHA256 bytes little endian",
            unit="Whole frozen inference clusters, all twelve values paired",
            columns=k.COLS,
            main_views="All six masks for every supported population/head; source-stratified natural and item-level HateCheck main sensitivities.",
            subgroup_views="Main source/head and gold-coverage-profile intervals. Source strict, all HateCheck target functions and all-twelve-gate-positive fine-head views are descriptive points.",
            uncertainty_role="Supporting 95%, not a new multiplicity-adjusted discovery family. Conditional on saved repeats and fixed source pools. Undefined metric draws remain explicitly counted; no unsupported confidence interval is manufactured.",
        ),
        gold_profile="Joint exact agreement on gate plus every available eligible fine field. No invented missing labels. Profiles separate benign/hateful gate-only, type, target, and target+severity. No input has all three type/target/severity gold. Never call this validation of every schema field.",
        multilabel_diagnostics="Per-label TP/FP/FN and precision/recall/F1, exact versus containment, extra-only/missing-only/both, cardinality distributions, known-empty sets, and gate-conditioned transitions.",
        severity_diagnostics="Gold low/medium/high versus predicted none/low/medium/high; coverage, exact class, over/under rank and conditional absolute/signed rank error. Missing rank has no numeric ordinal distance.",
        conditional_diagnostics="All twelve gates positive is outcome-selected descriptive evidence. Also preserve matched-repeat/pool gate-transition partitions and exact fixes/regressions. Reused references in pair diagnostics are not additional actual outputs or causal mediation.",
        omissions="For each global mask/head: fixed-mask and recomputed-mask leave-one-pool points, and fixed-mask leave-one-batch points. No omission changes primary conclusions or scope.",
        outcomes_used_to_choose_conventions=False,
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
    return (contract, science)


def load():
    meta = []
    data = {h: np.zeros((15758, 4, 3, len(k.names(h))), dtype=np.int16) for h in HEADS}
    gates = np.full((15758, 4, 3), -1, dtype=np.int8)
    votes = np.zeros_like(gates)
    codes = {
        h: np.full((15758, 4, 3), -1 if h != "severity" else -2, dtype=np.int16)
        for h in ("hate_type", "target_group", "severity")
    }
    prior = list(a.rows(P3 / "metadata.jsonl"))
    for i, row in enumerate(a.rows(a.OUT / "matrix.jsonl")):
        a.need(
            row["input_index"] == i and row["key"] == prior[i]["key"],
            "Matrix identity changed",
        )
        m = {
            key: value
            for key, value in row.items()
            if key not in ("predictions", "masks")
        }
        m["functionality"] = prior[i].get("functionality")
        fields = [
            head
            for head in ("hate_type", "target_group", "severity", "legal")
            if k.eligible(m, head)
        ]
        m["gold_profile"] = ("hateful_" if m["gold_hate"] else "benign_") + (
            "+".join(fields) if fields else "gate_only"
        )
        meta.append(m)
        for c, condition in enumerate(row["predictions"]):
            for rep, pred in enumerate(condition):
                value = pred["result"]
                votes[i, c, rep] = pred["n_valid"]
                if value is not None:
                    a.need(type(value["hate"]) is bool, "Nonboolean gate")
                    if not value["hate"]:
                        a.need(
                            value["hate_type"] == []
                            and value["target_group"] == []
                            and (value["severity"] is None),
                            "Ungated fine output",
                        )
                    gates[i, c, rep] = value["hate"]
                    for head in ("hate_type", "target_group"):
                        a.need(
                            len(value[head]) == len(set(value[head]))
                            and set(value[head]) <= set(k.LABELS[head]),
                            "Unknown/duplicate predicted label",
                        )
                        codes[head][i, c, rep] = sum(
                            (1 << k.LABELS[head].index(lab) for lab in value[head])
                        )
                    codes["severity"][i, c, rep] = (
                        -1
                        if value["severity"] is None
                        else k.LABELS["severity"].index(value["severity"])
                    )
                for head in HEADS:
                    data[head][i, c, rep] = k.features(m, value, head)
        a.need(
            row["masks"]["common_valid"] == bool(np.all(gates[i] >= 0)),
            "Common mask changed",
        )
    a.need(
        len(meta) == 15758 and np.array_equal(gates < 0, votes == 0),
        "Shape/null validation",
    )
    save("metadata.jsonl", b"".join((a.encoded(m) for m in meta)))
    with (OUT / "features.npz").open("xb") as f:
        np.savez_compressed(f, **data)
    with (OUT / "prediction_codes.npz").open("xb") as f:
        np.savez_compressed(f, gates=gates, votes=votes, **codes)
    save(
        "feature_schema.json",
        dict(
            features={head: k.names(head) for head in HEADS},
            metrics={head: k.metric_names(head) for head in HEADS},
            prediction_codes="Multilabel bit positions follow contract labels; -1 is invalid. Severity -2 invalid, -1 no category, 0/1/2 low/medium/high. Gates -1 invalid, 0 benign, 1 hateful.",
        ),
    )
    return (meta, data, gates, votes, codes)


def summary(counts, head, indices):
    scores = k.score(counts, head, indices)
    averages = k.summaries(scores)
    return dict(
        counts=counts,
        run_scores=scores,
        condition_means=averages[:4],
        metrics={
            metric: dict(
                reference=averages[0, j],
                treatment=averages[4, j],
                difference=averages[5, j],
                pool_values=averages[1:4, j],
                pool_differences=(scores[1:, :, j] - scores[:1, :, j]).mean(axis=1),
            )
            for j, metric in enumerate(k.metric_names(head))
        },
    )


def calculate():
    contract, science = prepare()
    meta, data, gates, votes, codes = load()
    valid = gates >= 0
    exclusions = set(science["mandatory_replay_keys"])
    results = {}
    omissions = {}
    descriptors = {}
    draw_count = 0

    def evaluate(
        name,
        pop,
        head,
        members,
        view="main",
        mode="cluster",
        boot=False,
        kind="global",
        reuse=None,
    ):
        nonlocal draw_count
        macro_labels = contract["population_macro_labels"][pop][head]
        mi = [k.LABELS[head].index(lab) for lab in macro_labels]
        if head not in HEADS:
            results[name] = dict(
                id=name,
                population=pop,
                head=head,
                group_kind=kind,
                view=view,
                status="unscorable_no_known_gold",
                eligible_n=0,
                common_n=0,
                metrics={},
            )
            return results[name]
        eligible, common = k.selected(
            meta, members, head, view, exclusions, valid, votes
        )
        info = dict(
            id=name,
            population=pop,
            head=head,
            group_kind=kind,
            view=view,
            mode=mode,
            population_macro_labels=macro_labels,
            macro_indices=mi,
            eligible_indices=eligible,
            common_indices=common,
            eligible_n=len(eligible),
            common_n=len(common),
            excluded_by_availability=len(eligible) - len(common),
            cluster_count=len({meta[i]["inference_cluster"] for i in common}),
            gold_profile_counts=dict(
                Counter((meta[i]["gold_profile"] for i in common))
            ),
            status=(
                "defined"
                if len(common)
                else (
                    "unscorable_no_known_gold"
                    if not len(eligible)
                    else "undefined_no_common_outputs"
                )
            ),
        )
        if not len(eligible):
            info["metrics"] = {}
            results[name] = native(info)
            return info
        counts = data[head][common].sum(axis=0, dtype=np.int64)
        info.update(summary(counts, head, mi))
        own = valid[eligible] & (votes[eligible] == 3 if k.VIEWS[view][0] else True)
        info["own_valid_descriptive"] = summary(
            (data[head][eligible] * own[..., None]).sum(axis=0, dtype=np.int64),
            head,
            mi,
        )
        info["missing_bounds"] = k.missing_bounds(
            meta, data[head], eligible, valid, votes, head, mi, k.VIEWS[view][0]
        )
        if head in ("hate_type", "target_group", "severity"):
            full_eligible = [i for i in members if k.eligible(meta[i], head)]
            full_support = {
                lab: sum(
                    (
                        (
                            meta[i][k.GOLD[head]] == lab
                            if head == "severity"
                            else lab in meta[i][k.GOLD[head]]
                        )
                        for i in full_eligible
                    )
                )
                for lab in k.LABELS[head]
            }
            local = [
                j for j, lab in enumerate(k.LABELS[head]) if full_support[lab] >= 10
            ]
            info.update(
                group_full_gold_support=full_support,
                group_supported_macro_labels=[k.LABELS[head][j] for j in local],
                group_supported_macro_f1_descriptive=k.summaries(
                    k.score(counts, head, local)
                )[:, 0],
            )
        if boot and len(common):
            if reuse is not None:
                old = results[reuse]
                a.need(
                    old["common_indices"] == common.tolist()
                    and old["macro_indices"] == mi,
                    "Reused quality cohort differs",
                )
                info["bootstrap"] = old["bootstrap"]
                info["draw_file"] = old["draw_file"]
                info["reuse_of"] = reuse
                for metric in info["metrics"]:
                    for field in (
                        "interval95",
                        "pool_difference_intervals95",
                        "finite_bootstrap_draws",
                    ):
                        info["metrics"][metric][field] = old["metrics"][metric][field]
            else:
                draws, record = k.bootstrap(
                    meta, data[head], common, eligible, head, mi, k.seed(name), mode
                )
                info["bootstrap"] = record
                info["draw_file"] = save_array(
                    "draws/" + hashlib.sha256(name.encode()).hexdigest()[:24] + ".npy",
                    draws,
                )
                for j, metric in enumerate(k.metric_names(head)):
                    finite = np.all(np.isfinite(draws[:, :, j]), axis=1)
                    info["metrics"][metric].update(
                        finite_bootstrap_draws=int(finite.sum()),
                        interval95=(
                            np.quantile(draws[:, 5, j], [0.025, 0.975]).tolist()
                            if finite.all()
                            else None
                        ),
                        pool_difference_intervals95=(
                            np.quantile(
                                draws[:, 1:4, j] - draws[:, :1, j],
                                [0.025, 0.975],
                                axis=0,
                            ).T.tolist()
                            if finite.all()
                            else None
                        ),
                    )
                draw_count += 1
        results[name] = native(info)
        return results[name]

    for pop in POPS:
        members = [i for i, m in enumerate(meta) if m["population"] == pop]
        for head in [*HEADS, "legal"]:
            for view in k.VIEWS:
                name = f"population/{pop}/{head}/{view}"
                record = evaluate(name, pop, head, members, view, boot=True)
                if not record["common_n"]:
                    continue
                common = np.asarray(record["common_indices"], dtype=np.int64)
                mi = record["macro_indices"]
                total = data[head][common].sum(axis=0, dtype=np.int64)
                out = []
                for drop in (1, 2, 3):
                    arms = tuple((c for c in (1, 2, 3) if c != drop))
                    for fixed in (True, False):
                        keep = (
                            common
                            if fixed
                            else k.selected(
                                meta,
                                members,
                                head,
                                view,
                                exclusions,
                                valid,
                                votes,
                                arms,
                            )[1]
                        )
                        count = (
                            total
                            if fixed
                            else data[head][keep].sum(axis=0, dtype=np.int64)
                        )
                        runs = k.score(count, head, mi)
                        delta = (runs[list(arms)] - runs[:1]).mean(axis=(0, 1))
                        out.append(
                            dict(
                                kind=(
                                    "pool_fixed_mask"
                                    if fixed
                                    else "pool_recomputed_mask"
                                ),
                                omitted=drop,
                                common_n=len(keep),
                                metrics=dict(zip(k.metric_names(head), delta)),
                            )
                        )
                for batch in sorted({meta[i]["stage"] for i in common}):
                    drop = [i for i in common if meta[i]["stage"] == batch]
                    remaining = len(common) - len(drop)
                    count = total - data[head][drop].sum(axis=0, dtype=np.int64)
                    delta = k.summaries(k.score(count, head, mi))[5]
                    out.append(
                        dict(
                            kind="execution_batch",
                            omitted=batch,
                            common_n=remaining,
                            metrics=dict(zip(k.metric_names(head), delta)),
                        )
                    )
                omissions[name] = native(out)
            if results[f"population/{pop}/{head}/main"]["common_n"]:
                mode = "source_stratified" if pop == "en_test" else "item"
                evaluate(
                    f"{mode}/{pop}/{head}/main",
                    pop,
                    head,
                    members,
                    mode=mode,
                    boot=True,
                    kind="alternative_resampling",
                )
            print("Global quality: " + pop + "/" + head, flush=True)
    for source in ("hatexplain", "implicit_hate", "mhs"):
        members = [i for i, m in enumerate(meta) if m["source"] == source]
        for head in HEADS:
            for view in ("main", "strict"):
                reuse = (
                    f"population/en_test/{head}/main"
                    if view == "main"
                    and (
                        source == "implicit_hate"
                        and head == "hate_type"
                        or (source == "mhs" and head == "severity")
                    )
                    else None
                )
                evaluate(
                    f"source/{source}/{head}/{view}",
                    "en_test",
                    head,
                    members,
                    view,
                    boot=view == "main",
                    kind="source",
                    reuse=reuse,
                )
    for pop in POPS:
        profiles = sorted({m["gold_profile"] for m in meta if m["population"] == pop})
        for profile in profiles:
            members = [
                i
                for i, m in enumerate(meta)
                if m["population"] == pop and m["gold_profile"] == profile
            ]
            for view in ("main", "strict"):
                evaluate(
                    f"profile/{pop}/{profile}/{view}",
                    pop,
                    "known_answer",
                    members,
                    view,
                    boot=view == "main",
                    kind="gold_coverage_profile",
                )
    for fn in sorted(
        {m["functionality"] for m in meta if m["functionality"] is not None}
    ):
        members = [i for i, m in enumerate(meta) if m["functionality"] == fn]
        for view in ("main", "strict"):
            evaluate(
                f"function/{fn}/target_group/{view}",
                "en_hatecheck_eval",
                "target_group",
                members,
                view,
                kind="function",
            )
    for pop in POPS:
        for head in ("hate_type", "target_group", "severity"):
            members = [
                i
                for i, m in enumerate(meta)
                if m["population"] == pop and np.all(gates[i] == 1)
            ]
            if members:
                evaluate(
                    f"all_gate_positive/{pop}/{head}/main",
                    pop,
                    head,
                    members,
                    kind="outcome_selected_all_twelve_hateful",
                )
    save("quality_results.json", results)
    save("omissions.json", omissions)
    artifacts = {a.rel(p): a.sha(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
    save(
        "calculated.json",
        dict(
            status="part4_calculated_independent_verification_pending",
            created_utc=a.utc(),
            bindings=contract["bindings"],
            artifacts=artifacts,
            quality_views=len(results),
            new_bootstrap_sets=draw_count,
            reused_bootstrap_sets=sum(("reuse_of" in r for r in results.values())),
            omission_records=sum(map(len, omissions.values())),
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        json.dumps(
            dict(
                quality_views=len(results),
                new_bootstrap_sets=draw_count,
                omission_records=sum(map(len, omissions.values())),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    with a.offline():
        calculate()
