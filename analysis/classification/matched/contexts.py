from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from analysis.classification.matched import metrics as q
import numpy as np
from analysis.classification import validate_prompts as pc
from analysis.classification import validate_preparation as oc
from analysis.classification import validate_coverage as cc

m = q.m


def check_prompts():
    path = m.BASE / "prompt_checks.json"
    if path.exists():
        saved = m.a.read(path)
        m.c.verify(saved["sources"])
        return saved["checks"]
    with m.runtime.offline():
        checks = {"original": pc.captured_check(oc), "coverage": pc.captured_check(cc)}
        m.save(
            path,
            dict(
                checks=checks,
                sources=m.bindings(
                    [Path(pc.__file__), Path(oc.__file__), Path(cc.__file__)]
                ),
            ),
        )
        return checks


def run():
    with m.runtime.offline():
        groups, outputs, meta = m.load()
        gold = {r["id"]: r for rows in groups.values() for r in rows}
        phases = {r["id"]: phase for phase, rows in groups.items() for r in rows}
        prompts = {}
        for path in [
            m.c.PREP / "matched_prompts.jsonl",
            m.c.PREP / "coverage_matched_prompts.jsonl",
        ]:
            for row in m.runtime.stream_rows(path):
                prompts[row["key"], row["condition"]] = row
        m.a.require(len(prompts) == 12233, "Prompt matrix mismatch")
        checks = check_prompts()
        kb = {
            r["id"]: r
            for r in m.runtime.stream_rows(m.c.CORE / "kb/full/records.jsonl")
            if r["kind"] == "example"
        }
        aggregate = {}
        details = []
        mutations = defaultdict(Counter)
        unchanged = []
        for (key, cond), p in sorted(prompts.items()):
            phase = phases[key]
            group = aggregate.setdefault(
                phase + "/" + cond,
                dict(
                    prompts=0,
                    sources=Counter(),
                    annotations=Counter(),
                    labels=defaultdict(Counter),
                    slot_count_histograms=defaultdict(Counter),
                    unique_text_ids=set(),
                    unique_annotation_ids=set(),
                    chars=[],
                ),
            )
            group["prompts"] += 1
            group["chars"].append(sum(len(x["text"]) for x in p["messages"]))
            text_shown = cond != "retrieved_labels_only"
            labels_shown = cond != "retrieved_text_only"
            group["unique_text_ids"].update(p["example_text_ids"] if text_shown else [])
            group["unique_annotation_ids"].update(
                p["annotation_source_ids"] if labels_shown else []
            )
            fields = Counter()
            labels = defaultdict(Counter)
            sources = Counter()
            for tid, lid in zip(p["example_text_ids"], p["annotation_source_ids"]):
                if text_shown:
                    sources[kb[tid]["source"]] += 1
                if not labels_shown:
                    continue
                annotation = kb[lid]["meta"]
                for dim, field in q.s.original.GOLD_FIELDS.items():
                    value = annotation.get(field)
                    if value is None:
                        continue
                    fields[dim] += 1
                    labels[dim].update(
                        ["hate" if value else "not_hate"]
                        if dim == "hate"
                        else [value] if dim == "severity" else value
                    )
                fields["auxiliary_legal_flags"] += bool(
                    annotation.get("astra_aux_stgb")
                )
            group["sources"].update(sources)
            group["annotations"].update(fields)
            for dim in m.DIMENSIONS + ["auxiliary_legal_flags"]:
                group["slot_count_histograms"][dim][str(fields[dim])] += 1
            for dim, labs in labels.items():
                group["labels"][dim].update(labs)
            for rep in range(3):
                m.a.require(
                    meta[key, cond, rep]["messages_sha256"] == p["messages_sha256"],
                    "Delivered prompt SHA differs",
                )
            detail = dict(
                key=key,
                phase=phase,
                condition=cond,
                example_text_ids=p["example_text_ids"],
                annotation_source_ids=p["annotation_source_ids"],
                text_shown=text_shown,
                labels_shown=labels_shown,
                annotated_slots=dict(fields),
                label_exposures=dict(labels),
                text_sources=dict(sources),
                changed_annotation_slots=p.get("changed_annotation_slots"),
                messages_sha256=p["messages_sha256"],
            )
            if cond.startswith("coverage_matched_"):
                ref = prompts[key, "retrieved_complete"]
                m.a.require(
                    not set(p["example_text_ids"]) & set(ref["example_text_ids"]),
                    "Coverage control reuses retrieved example",
                )
                detail["replaced_text_ids"] = ref["example_text_ids"]
            if cond.startswith("permuted_"):
                changed = p["changed_annotation_slots"]
                mutations[phase][str(changed)] += 1
                if changed == 0:
                    m.a.require(
                        p["messages"] == prompts[key, "retrieved_complete"]["messages"],
                        "Unchanged annotation block has changed messages",
                    )
                    unchanged.append((key, cond))
            details.append(detail)
        m.a.require(len(unchanged) == 434, "Identical permutation count differs")
        for group in aggregate.values():
            group["unique_text_ids"] = len(group["unique_text_ids"])
            group["unique_annotation_ids"] = len(group["unique_annotation_ids"])
            chars = group.pop("chars")
            group["message_chars"] = {
                "mean": float(np.mean(chars)),
                "min": min(chars),
                "max": max(chars),
            }
        m.jsonlines(m.BASE / "prompt_delivery.jsonl", details)

        strata = {}
        pairwise = {}
        case_cards = []
        import yaml

        taxonomy = yaml.safe_load(
            (m.ROOT / "config/taxonomy.yaml").read_text(encoding="utf8")
        )
        for phase, rr in groups.items():
            dim = q.s.MATCHED_DIMS[phase]
            ep = q.s.endpoint(rr, dim, taxonomy)
            keys = {r["id"] for r in rr}
            runs = {
                k: q.s.contributions(rr, {i: v[i] for i in keys}, ep)
                for k, v in outputs.items()
            }
            ids = next(iter(runs.values()))["ids"]
            left = q.m.names(runs, "retrieved_complete")
            strata[phase] = {}
            for cond in ["permuted_0", "permuted_1", "permuted_2"]:
                right = q.m.names(runs, cond)
                st = {}
                for state in ["unchanged", "changed"]:
                    mask = np.array(
                        [
                            (prompts[key, cond]["changed_annotation_slots"] == 0)
                            == (state == "unchanged")
                            for key in ids
                        ]
                    )
                    st[state] = q.point(
                        [q.masked(runs[k], mask) for k in left],
                        [q.masked(runs[k], mask) for k in right],
                    )
                strata[phase][cond] = st
            for family in m.PAIRS[:2]:
                right = m.names(runs, family)
                eligible = sorted(
                    k
                    for k in keys
                    if all(outputs[v][k] is not None for v in left + right)
                )
                transitions = Counter()
                differences = {}
                by_source = defaultdict(Counter)
                for key in eligible:
                    l = [q.exact_known(gold[key], outputs[v][key]) for v in left]
                    r = [q.exact_known(gold[key], outputs[v][key]) for v in right]
                    differences[key] = float(np.mean(l) - np.mean(r))
                    for lv in l:
                        for rv in r:
                            kind = (
                                "both_correct"
                                if lv and rv
                                else (
                                    "both_wrong"
                                    if not lv and not rv
                                    else (
                                        "reference_fix"
                                        if lv
                                        else "reference_regression"
                                    )
                                )
                            )
                            transitions[kind] += 1
                            by_source[gold[key]["source"]][kind] += 1
                pairwise[phase + "/" + family] = {
                    "common_inputs": len(eligible),
                    "setting_pairs_per_input": len(left) * len(right),
                    "counts": dict(transitions),
                    "by_source": dict(by_source),
                    "item_directions": dict(
                        Counter(
                            (
                                "reference_better"
                                if v > 0
                                else "reference_worse" if v < 0 else "same_mean"
                            )
                            for v in differences.values()
                        )
                    ),
                }
                for direction in ["improvement", "regression"]:
                    selected = (
                        next((key for key in eligible if differences[key] > 0), None)
                        if direction == "improvement"
                        else next(
                            (key for key in eligible if differences[key] < 0), None
                        )
                    )
                    case_cards.append(
                        case(
                            selected,
                            phase,
                            family,
                            direction,
                            left + right,
                            gold,
                            outputs,
                            meta,
                            prompts,
                            kb,
                            differences.get(selected),
                        )
                    )
            same = sorted((key, cond) for key, cond in unchanged if key in keys)
            selected = next(
                (
                    (key, cond)
                    for key, cond in same
                    if any(
                        outputs["retrieved_complete", rep][key] is not None
                        and outputs[cond, rep][key] is not None
                        and outputs["retrieved_complete", rep][key]
                        != outputs[cond, rep][key]
                        for rep in range(3)
                    )
                ),
                None,
            )
            key, cond = selected if selected else (None, None)
            case_cards.append(
                case(
                    key,
                    phase,
                    cond,
                    "identical_prompt_disagreement",
                    left + (m.names(runs, cond) if cond else []),
                    gold,
                    outputs,
                    meta,
                    prompts,
                    kb,
                    None,
                )
            )
        resource = {}
        for record in meta.values():
            label = record["phase"] + "/" + record["condition"]
            group = resource.setdefault(
                label,
                dict(
                    outputs=0,
                    invalid=0,
                    partial_votes=0,
                    repairs=0,
                    parse_failures=0,
                    request_seconds=0.0,
                    backoff_seconds=0.0,
                    usage=Counter(),
                    repair_usage=Counter(),
                    accounting=Counter(),
                    http_classes=Counter(),
                ),
            )
            group["outputs"] += 1
            group["invalid"] += record["result"] is None
            group["partial_votes"] += 0 < record["n_valid"] < 3
            for k in [
                "repairs",
                "parse_failures",
                "request_seconds",
                "backoff_seconds",
            ]:
                group[k] += record[k]
            for k in ["usage", "repair_usage", "accounting"]:
                group[k].update(record[k])
            group["http_classes"].update(record["response_classes"])
        execution = m.a.read(m.BASE / "execution_review.json")
        chronological = []
        for stage in execution["stages"]:
            for attempt in stage["attempts"]:
                start = datetime.fromisoformat(attempt["start"])
                end = datetime.fromisoformat(attempt["end"])
                chronological.append(
                    dict(
                        stage=stage["id"],
                        start=attempt["start"],
                        end=attempt["end"],
                        generation_hours=(end - start).total_seconds() / 3600.0,
                        outputs=stage["counts"]["predictions"],
                    )
                )
        start = min(datetime.fromisoformat(v["start"]) for v in chronological)
        end = max(datetime.fromisoformat(v["end"]) for v in chronological)
        complete = datetime.fromisoformat(execution["final_file_mtime_utc"])
        timing = dict(
            first_generation_utc=start.isoformat(),
            last_generation_utc=end.isoformat(),
            generation_span_hours=(end - start).total_seconds() / 3600,
            complete_file_proxy_hours=(complete - start).total_seconds() / 3600,
            total_batch_generation_hours=sum(
                v["generation_hours"] for v in chronological
            ),
            batches=chronological,
            notes="Batch start/end include model check and local preparation; file mtime is a proxy for final completion. Request times overlap and are not GPU compute hours.",
        )
        m.save(
            m.BASE / "context_review.json",
            dict(
                independent_checks=checks,
                aggregate=aggregate,
                mutation_histograms=dict(mutations),
                unchanged_message_pairs=len(unchanged),
                mutation_strata=strata,
                known_exact_transitions=pairwise,
                scope="Descriptive controls; selection variants are fixed, not extra independent repeats.",
            ),
        )
        m.save(
            m.BASE / "cases.json",
            dict(
                selection="Lexicographically first positive/negative mean known-field exact difference per phase/primary family; first identical-message disagreement per phase. All relevant repeat/variant outcomes included.",
                cases=case_cards,
            ),
        )
        m.save(
            m.BASE / "resources.json",
            dict(
                by_population_condition=resource,
                timing=timing,
                observed_totals=execution["observed_usage"],
                counts=execution["counts"],
                accounting=execution["accounting"],
            ),
        )
        m.save(
            m.BASE / "context_receipt.json",
            dict(
                outputs=m.bindings(
                    [
                        m.BASE / p
                        for p in [
                            "prompt_delivery.jsonl",
                            "context_review.json",
                            "cases.json",
                            "resources.json",
                        ]
                    ]
                ),
                sources=m.bindings(
                    [
                        Path(__file__),
                        Path(pc.__file__),
                        Path(oc.__file__),
                        Path(cc.__file__),
                    ]
                ),
                Qwen_calls=0,
                training_fits=0,
            ),
        )
        print("Prompt delivery, cases and resources reviewed.", flush=True)


def case(
    key, phase, family, direction, settings, gold, outputs, metadata, prompts, kb, delta
):
    result = dict(
        key=key,
        phase=phase,
        family=family,
        direction=direction,
        mean_known_exact_difference=delta,
    )
    if key is None:
        return result | {"reason": "No eligible case"}
    result.update(
        text=gold[key]["text"],
        source=gold[key]["source"],
        known_gold=q.known_values(gold[key]),
        outcomes=[
            dict(
                condition=k[0],
                repeat=k[1],
                prediction=outputs[k][key],
                known_exact=q.exact_known(gold[key], outputs[k][key]),
                n_valid=metadata[key, *k]["n_valid"],
                result_path=metadata[key, *k]["result_path"],
                result_sha256=metadata[key, *k]["result_sha256"],
            )
            for k in settings
        ],
        contexts={
            cond: [
                dict(
                    slot=i + 1,
                    text_id=tid,
                    annotation_id=lid,
                    text=kb[tid]["text"],
                    annotations=kb[lid]["meta"],
                )
                for i, (tid, lid) in enumerate(
                    zip(
                        prompts[key, cond]["example_text_ids"],
                        prompts[key, cond]["annotation_source_ids"],
                    )
                )
            ]
            for cond in sorted({k[0] for k in settings})
        },
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-only", action="store_true")
    args = parser.parse_args()
    if args.prompts_only:
        check_prompts()
    else:
        run()
