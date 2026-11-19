import os

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    CUDA_VISIBLE_DEVICES="",
)
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
from types import FunctionType, SimpleNamespace
import yaml
from analysis.classification.broad import archive as original_reader
from analysis.common import result_reader as scoring
from analysis.common import replay as gold
from analysis.common import prompt as prompt

r = original_reader.r
a, c = (r.a, r.c)
ROOT = c.ROOT
BASE = ROOT / "experiments/astra/sq2_broad_reader_review_20261104_v1"
PLAN = ROOT / "Thesis_Log/astra_result/astra_sq2_broad_analysis_plan_20261104_v1.md"
CONDITIONS = ("rag_full", "rag_small", "zero_shot")
PHASE_COUNTS = {
    "larger_dev": 2506,
    "en_test": 12041,
    "de_test": 3175,
    "de_legal_test": 177,
    "en_hatecheck_eval": 3728,
}
FIELDS = ("hate", "target_group", "hate_type", "severity", "legal")
REPLAY = c.BASE / "replay_review_20261009_v1/sensitivity_plan.json"
ENCODERS = c.BASE / "encoders/assembled/complete.json"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def save(path, value):
    path = Path(path).resolve()
    require(
        path.is_relative_to(BASE.resolve()), "Derived output outside review directory"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(c.encoded(value))


def rows(path):
    return r.original.stream_rows(path)


def bindings(paths):
    return {r.reference(path): c.sha(path) for path in paths}


def clone(fn, **changes):
    out = FunctionType(
        fn.__code__,
        dict(fn.__globals__, **changes),
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )
    out.__kwdefaults__ = dict(fn.__kwdefaults__ or {})
    return out


def prepare():
    save(
        BASE / "analysis_contract.json",
        {
            "status": "part1_scope_before_matrix_consumption",
            "created_utc": c.utc(),
            "source_bindings": bindings(
                [
                    PLAN,
                    r.PLAN,
                    r.SCHEDULE,
                    r.MANIFEST,
                    REPLAY,
                    c.BASE / "scoring_contract_v1.json",
                    ENCODERS,
                ]
            ),
            "scope": "Read and validate saved outputs; establish gold masks and comparisons. No F1, error scoring, bootstrap or outcome case selection.",
            "matrix": {
                "inputs": 21627,
                "conditions": list(CONDITIONS),
                "outer_repeats": [0, 1, 2],
                "inner_votes": 3,
                "predictions": 194643,
                "audited_batches": 52,
            },
            "reader": "Unchanged D-aware production reader, with an isolated observation hook. Every consumed item receipt and result is checked against the audited chain.",
            "raw_review": "First two entries in frozen schedule order per population/condition/valid-vote count; also first two per population/condition with each repair, tie or gate-normalization flag. Reparse saved strings only. Independently aggregate every saved vote set.",
            "cache": "Exclusive-create local derived files, frozen schedule order, retained nulls, source paths/hashes. No write to D evidence.",
            "scoring_contract": scoring.contract(),
            "primary": [list(x) for x in sorted(scoring.PRIMARY_BROAD)],
            "reference_policy": "Only original aligned English full/KB RoBERTa and German GBERT, three fitted seeds. Later equal-budget GPU results are a different study.",
            "fine_exact_policy": "Original fine-set exactness remains available; separately label complete-answer exactness requiring the correct hate gate on gold-hate fine endpoints, including known-empty fine sets.",
            "sensitivity_policy": "Retain all primary rows. Common 27-key replay exclusion on every compared system; separately exclude the one-space German row only for the declared German hate sensitivity. Freeze macro labels before either exclusion.",
            "prohibited": [
                "network",
                "model loading",
                "inference",
                "training",
                "retrieval rerun",
                "new Qwen calls",
                "source mutation",
                "performance rankings",
            ],
        },
    )


def check_decision(value, taxonomy, *, allow_duplicates=False):
    if value is None:
        return
    require(
        set(FIELDS) <= value.keys() and type(value["hate"]) is bool,
        "Malformed decision",
    )
    for field in ("target_group", "hate_type", "legal"):
        vals = value[field]
        require(
            isinstance(vals, list)
            and (allow_duplicates or len(vals) == len(set(vals))),
            "Malformed multilabel field",
        )
        require(
            set(vals) <= set(taxonomy["dimensions"][field]["labels"]),
            "Unknown predicted label",
        )
    require(
        value["severity"] is None
        or value["severity"] in taxonomy["dimensions"]["severity"]["labels"],
        "Unknown severity",
    )
    if not value["hate"]:
        require(
            value["target_group"] == value["hate_type"] == []
            and value["severity"] is None,
            "Contradictory delivered hate gate",
        )


def delivered_aggregate(votes):

    result = gold.independent_aggregate(votes)
    if result is None:
        return None
    valid = [v for v in votes if v is not None]
    for field in ("target_group", "hate_type", "legal"):
        counts = Counter((label for vote in valid for label in vote[field]))
        result[field] = sorted(
            (label for label, count in counts.items() if 2 * count > len(valid))
        )
    if not result["hate"]:
        result.update(target_group=[], hate_type=[], severity=None)
    return result


def duplicate_fields(votes):
    return [
        {
            "vote": index,
            "field": field,
            "repeated_labels": {
                label: count
                for label, count in Counter(vote[field]).items()
                if count > 1
            },
        }
        for index, vote in enumerate(votes)
        if vote is not None
        for field in ("target_group", "hate_type", "legal")
        if len(vote[field]) != len(set(vote[field]))
    ]


def check_votes(value, taxonomy):
    votes = value["run_predictions"]
    require(
        value["n_runs"] == len(votes) == len(value["raw_runs"]) == 3,
        "Three inner vote slots required",
    )
    for vote in votes:
        check_decision(vote, taxonomy, allow_duplicates=True)
    check_decision(value["result"], taxonomy)
    valid = [v for v in votes if v is not None]
    require(value["n_valid"] == len(valid), "Valid-vote count differs")
    require(
        (value["result"] is None) == (len(valid) == 0),
        "Null output cannot be a negative decision",
    )
    require(
        delivered_aggregate(votes) == gold.decision(value["result"]),
        "Independent vote aggregation differs",
    )
    tie = bool(valid) and sum((v["hate"] for v in valid)) * 2 == len(valid)
    require(value["uncertain"] == tie, "Gate tie stamp differs")
    if valid:
        chosen = next(
            (v for v in valid if v["hate"] == value["result"]["hate"]), valid[0]
        )
        require(
            value["result"]["reasoning"] == chosen["reasoning"],
            "Aggregate rationale origin differs",
        )


def replay_strings(value, parser):
    texts = 0
    for raw_run, saved_vote in zip(value["raw_runs"], value["run_predictions"]):
        require(1 <= len(raw_run) <= 3, "Unexpected repair chain length")
        last = None
        for index, raw in enumerate(raw_run):
            parsed, _, error = parser.parse_response(raw)
            require((parsed is None) == bool(error), "Parser error state differs")
            require(
                parsed is None or index == len(raw_run) - 1,
                "A valid answer was followed by repair",
            )
            last = parsed.model_dump() if parsed is not None else None
            texts += 1
        require(
            gold.decision(last) == gold.decision(saved_vote),
            "Saved vote differs from raw text",
        )
        require(
            last is None or last["reasoning"] == saved_vote["reasoning"],
            "Saved vote rationale differs",
        )
    return texts


def assemble():
    contract = c.read(BASE / "analysis_contract.json")
    r.verify(contract["source_bindings"])
    taxonomy = yaml.safe_load(
        (ROOT / "config/taxonomy.yaml").read_text(encoding="utf8")
    )
    groups, _ = scoring.scientific_rosters("broad")
    roster = {row["key"]: row for rr in groups.values() for row in rr}
    require(
        {k: len(v) for k, v in groups.items()} == PHASE_COUNTS and len(roster) == 21627,
        "Scientific population identity differs",
    )
    schedule = c.read(r.SCHEDULE)
    jobs = {
        (j["key"], j["condition"], j["repeat"]): (j, stage["id"])
        for stage in schedule["stages"]
        for j in stage["jobs"]
    }
    expected = {
        (key, cond, rep) for key in roster for cond in CONDITIONS for rep in range(3)
    }
    require(
        set(jobs) == expected
        and sum((len(s["jobs"]) for s in schedule["stages"])) == len(expected),
        "Scheduled cross-product differs",
    )
    before = bindings(
        [
            Path(__file__),
            Path(original_reader.__file__),
            Path(gold.__file__),
            Path(scoring.__file__),
            BASE / "analysis_contract.json",
            r.BASE / "storage_complete.json",
            r.BASE / "live/complete.json",
            r.BASE / "concurrency_20261029_v1/live_complete.json",
        ]
    )
    before.update(bindings([BASE / "reader_correction.json"]))
    save(BASE / "preservation_before.json", before)
    counters, by_cell = (Counter(), defaultdict(Counter))
    selected, sample_counts, raw_texts = ([], Counter(), 0)
    duplicates = []
    stages, consumed = ([], 0)
    parser = r.e.isolated.adapter(ROOT / "config/taxonomy.yaml").bundle.parser
    dest = BASE / "predictions.jsonl"
    start = time.perf_counter()
    with dest.open("x", encoding="utf8", newline="\n") as handle:

        def observed_read(path):
            nonlocal consumed, raw_texts
            value = a.read(path)
            if Path(path).name != "result.json":
                return value
            ident = value["identity"]
            condition = ident["condition"].removeprefix("broad::")
            key = ident["key"]
            job, stage = jobs[key, condition, ident["repeat"]]
            require(
                ident
                == {
                    "condition": "broad::" + condition,
                    "key": key,
                    "repeat": job["repeat"],
                    "experiment_sha256": r.original.PLANS["broad"],
                },
                "Unexpected saved identity",
            )
            require(
                value["active_model"] == value["requested_model"] == a.MODEL,
                "Model alias differs",
            )
            check_votes(value, taxonomy)
            repeated = duplicate_fields(value["run_predictions"])
            phase = roster[key]["phase"]
            flags = ["valid_" + str(value["n_valid"])]
            flags += [
                name
                for name, active in [
                    ("repair", value["repairs"]),
                    ("tie", value["uncertain"]),
                    ("gate_normalization", value["gate_normalised"]),
                ]
                if active
            ]
            reasons = [
                name for name in flags if sample_counts[phase, condition, name] < 2
            ]
            if repeated:
                reasons.append("repeated_label_occurrences")
                unique_vote_result = gold.independent_aggregate(
                    value["run_predictions"]
                )
                duplicates.append(
                    {
                        "key": key,
                        "condition": condition,
                        "repeat": job["repeat"],
                        "phase": phase,
                        "repeated_fields": repeated,
                        "result_path": r.reference(path),
                        "delivered_result": gold.decision(value["result"]),
                        "one_vote_per_label_diagnostic": unique_vote_result,
                        "changed_delivered_fields": [
                            field
                            for field in FIELDS
                            if unique_vote_result[field] != value["result"][field]
                        ],
                    }
                )
            if reasons:
                for name in reasons:
                    sample_counts[phase, condition, name] += 1
                text_count = replay_strings(value, parser)
                raw_texts += text_count
                selected.append(
                    {
                        "key": key,
                        "condition": condition,
                        "repeat": job["repeat"],
                        "phase": phase,
                        "reasons": reasons,
                        "saved_strings_checked": text_count,
                        "result_path": r.reference(path),
                    }
                )
            counts = {
                "predictions": 1,
                "nulls": int(value["result"] is None),
                "gate_ties": int(value["uncertain"]),
                "valid_votes_" + str(value["n_valid"]): 1,
            }
            counts.update(
                {
                    name: value[name]
                    for name in [
                        "repairs",
                        "parse_failures",
                        "gate_normalised",
                        "normalisations",
                    ]
                }
            )
            counters.update(counts)
            by_cell[phase, condition, job["repeat"]].update(counts)
            row = {
                "key": key,
                "phase": phase,
                "source": roster[key]["source"],
                "condition": condition,
                "repeat": job["repeat"],
                "result": value["result"],
                "votes": [gold.decision(v) for v in value["run_predictions"]],
                "n_valid": value["n_valid"],
                "uncertain": value["uncertain"],
                "messages_sha256": value["messages_sha256"],
                "stage": stage,
                "result_path": r.reference(path),
            }
            row.update(
                {
                    name: value[name]
                    for name in [
                        "repairs",
                        "parse_failures",
                        "gate_normalised",
                        "normalisations",
                    ]
                }
            )
            handle.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
                + "\n"
            )
            consumed += 1
            return value

        def consume(b, schedule, root, plan_sha, **kw):

            def verify_stage(stage, store, digest, mode):
                result = b.verify_review(stage, store, digest, mode)
                stages.append(stage["id"])
                print(
                    f"Reading batch {len(stages)}/52; {consumed}/194643 saved predictions checked",
                    flush=True,
                )
                return result

            proxy = SimpleNamespace(**dict(vars(b), verify_review=verify_stage))
            adapter = SimpleNamespace(**dict(vars(a), read=observed_read))
            return clone(original_reader.read_matrix, a=adapter)(
                proxy, schedule, root, plan_sha, **kw
            )

        with r.original.offline():
            print(
                "Checking the frozen D-aware production reader and completion chain...",
                flush=True,
            )
            output, provenance = clone(
                original_reader.load_predictions, read_matrix=consume
            )()
    require(
        consumed == sum((len(v) for v in output.values())) == 194643,
        "Cache length differs",
    )
    require(
        set(output) == {(cond, rep) for cond in CONDITIONS for rep in range(3)},
        "Run identity differs",
    )
    require(
        all((set(value) == set(roster) for value in output.values())),
        "Run input roster differs",
    )
    require(
        dict(counters) == c.read(r.BASE / "live/complete.json")["counts"],
        "Reader totals differ from original audits",
    )
    del output
    save(BASE / "item_bindings.json", provenance.pop("result_bindings"))
    save(
        BASE / "reader_review.json",
        {
            "status": "all_saved_predictions_consumed_and_vote_aggregates_checked",
            "counts": dict(counters),
            "population_condition_repeat_counts": [
                dict(phase=k[0], condition=k[1], repeat=k[2], counts=dict(v))
                for k, v in sorted(by_cell.items())
            ],
            "original_audit_batches_reused": stages,
            "full_raw_HTTP_reaudit": False,
            "raw_string_checks": selected,
            "raw_string_count": raw_texts,
            "duplicate_inner_label_cases": duplicates,
            "duplicate_label_policy": "Preserve the actual frozen list-occurrence voter. One-vote-per-label results are diagnostics only, not replacements. No effectiveness scoring here.",
            "provenance": provenance,
            "elapsed_reader_seconds": time.perf_counter() - start,
            "new_Qwen_calls": 0,
            "effectiveness_scoring": False,
        },
    )
    r.verify(before)
    save(
        BASE / "assembly.json",
        {
            "status": "verified_D_reader_cache_complete",
            "outputs": bindings(
                [
                    dest,
                    BASE / "item_bindings.json",
                    BASE / "reader_review.json",
                    BASE / "preservation_before.json",
                ]
            ),
            "sources": before,
            "new_Qwen_calls": 0,
            "model_loads": 0,
            "training_fits": 0,
        },
    )
    print("Cache complete:", dict(counters), flush=True)


def invalid_ids(ids, runs):
    require(bool(runs), "No compared runs")
    require(all((set(ids) <= run.keys() for run in runs)), "Missing compared input")
    return [key for key in ids if any((run[key] is None for run in runs))]


def eligible_value(row, dim):
    return scoring.original.truth(row, dim)


def comparison_setup():
    assembly = c.read(BASE / "assembly.json")
    r.verify(assembly["outputs"])
    groups, dimensions = scoring.scientific_rosters("broad")
    roster = {row["key"]: row for rr in groups.values() for row in rr}
    taxonomy = yaml.safe_load(
        (ROOT / "config/taxonomy.yaml").read_text(encoding="utf8")
    )
    predictions = defaultdict(dict)
    for row in rows(BASE / "predictions.jsonl"):
        ident = (row["condition"], row["repeat"])
        require(row["key"] not in predictions[ident], "Duplicate cached prediction")
        predictions[ident][row["key"]] = row["result"]
    encoder = c.read(ENCODERS)
    require(
        encoder["status"] == "all_logits_verified_and_gated_predictions_assembled"
        and encoder["supporting_only"],
        "Encoder completion differs",
    )
    c.verify(encoder["inputs"])
    c.verify(encoder["source_sha256"])
    c.verify(encoder["outputs"])
    encoder_info = []
    for name, digest in sorted(encoder["outputs"].items()):
        count = 0
        expected_ident = None
        for row in rows(ROOT / name):
            ident = (row["condition"], row["seed"])
            if expected_ident is None:
                expected_ident = ident
            require(
                ident == expected_ident
                and ident[0] in encoder["available_dimensions"]
                and (ident[1] in range(3)),
                "Encoder seed/condition differs",
            )
            require(
                row["available_dimensions"]
                == encoder["available_dimensions"][ident[0]],
                "Unsupported encoder head",
            )
            require(
                row["key"] not in predictions[ident] and row["key"] in roster,
                "Duplicate/unknown encoder input",
            )
            check_decision(row["result"], taxonomy)
            require(row["result"] is not None, "Invalid fixed encoder output")
            predictions[ident][row["key"]] = row["result"]
            count += 1
        lang = "de" if expected_ident[0] == "encoder_gbert" else "en"
        require(
            set(predictions[expected_ident])
            == {key for key, row in roster.items() if row["lang"] == lang},
            "Encoder language roster differs",
        )
        encoder_info.append(
            {
                "condition": expected_ident[0],
                "seed": expected_ident[1],
                "rows": count,
                "available_dimensions": encoder["available_dimensions"][
                    expected_ident[0]
                ],
                "path": name,
                "sha256": digest,
            }
        )
    exclusion = c.read(REPLAY)["exclusion_roster"]
    replay_keys = {row["key"] for row in exclusion}
    require(
        len(exclusion) == len(replay_keys) == 27 and replay_keys <= roster.keys(),
        "Replay roster differs",
    )
    require(
        all((roster[row["key"]]["phase"] == row["population"] for row in exclusion)),
        "Replay population differs",
    )
    require(
        roster[c.EMPTY_KEY]["text"] == " "
        and roster[c.EMPTY_KEY]["phase"] == "de_test",
        "Flagged text is not exactly one ASCII space",
    )
    endpoint_records, comparisons, population_info = ({}, [], [])
    for phase, rr in groups.items():
        memberships = defaultdict(set)
        for row in rr:
            require(
                row["text_cluster"]
                == hashlib.sha256(gold.normalized(row["text"]).encode()).hexdigest(),
                "Text-cluster definition differs",
            )
            cluster = (
                row["functional_cluster"]
                if phase == "en_hatecheck_eval"
                else row["text_cluster"]
            )
            require(cluster is not None, "Missing cluster")
            memberships[cluster].add(row["source"])
        population_info.append(
            {
                "population": phase,
                "inputs": len(rr),
                "sources": dict(Counter((row["source"] for row in rr))),
                "text_clusters": len({row["text_cluster"] for row in rr}),
                "primary_clusters": len(memberships),
                "cluster_strata": dict(
                    Counter(("|".join(sorted(v)) for v in memberships.values()))
                ),
                "cross_source_clusters": sum(
                    (len(v) > 1 for v in memberships.values())
                ),
                "replay_exclusion_inputs": sum(
                    (row["key"] in replay_keys for row in rr)
                ),
            }
        )
        for dim in dimensions[phase]:
            ep = scoring.endpoint(rr, dim, taxonomy)
            ids = ep["scorable_ids"]
            field = scoring.original.GOLD_FIELDS[dim]
            available = list(CONDITIONS) + [
                cond
                for cond, dims in sorted(encoder["available_dimensions"].items())
                if dim in dims
                and set((row["key"] for row in rr)) <= predictions[cond, 0].keys()
            ]
            own = {
                f"{cond}/{rep}": invalid_ids(ids, [predictions[cond, rep]])
                for cond in available
                for rep in range(3)
            }
            both = [row for row in rr if eligible_value(row, dim) is not None]
            endpoint_records[phase + "/" + dim] = dict(
                ep,
                phase=phase,
                available_conditions=available,
                annotation_missing=sum((row.get(field) is None for row in rr)),
                annotation_observed=sum((row.get(field) is not None for row in rr)),
                annotated_but_gate_ineligible=sum(
                    (
                        row.get(field) is not None and eligible_value(row, dim) is None
                        for row in rr
                    )
                ),
                observed_empty_eligible=sum(
                    (eligible_value(row, dim) == [] for row in both)
                ),
                own_invalid_ids=own,
                three_Qwen_conditions_common_invalid_ids=invalid_ids(
                    ids,
                    [predictions[cond, rep] for cond in CONDITIONS for rep in range(3)],
                ),
                row_clusters=[
                    (
                        roster[key]["functional_cluster"]
                        if phase == "en_hatecheck_eval"
                        else roster[key]["text_cluster"]
                    )
                    for key in ids
                ],
                row_sources=[roster[key]["source"] for key in ids],
            )
            for right in ["rag_small", "zero_shot"] + [
                name for name in available if name.startswith("encoder_")
            ]:
                runs = [
                    predictions[cond, rep]
                    for cond in ["rag_full", right]
                    for rep in range(3)
                ]
                invalid = invalid_ids(ids, runs)
                replay = [key for key in ids if key in replay_keys]
                primary = right == "rag_small" and (phase, dim) in scoring.PRIMARY_BROAD
                comparisons.append(
                    {
                        "population": phase,
                        "dimension": dim,
                        "left": "rag_full",
                        "right": right,
                        "role": "primary" if primary else "supporting",
                        "family_size": 4 if primary else None,
                        "family_confidence": 0.9875 if primary else None,
                        "eligible_rows": len(ids),
                        "common_valid_rows": len(ids) - len(invalid),
                        "common_invalid_ids": invalid,
                        "replay_exclusion_eligible_ids": replay,
                        "replay_common_valid_rows": sum(
                            (key not in set(invalid) | replay_keys for key in ids)
                        ),
                        "whitespace_exclusion_eligible_ids": (
                            [c.EMPTY_KEY]
                            if phase == "de_test" and dim == "hate"
                            else []
                        ),
                        "macro_labels_frozen": ep["macro_labels"],
                    }
                )
    require(
        sum((row["role"] == "primary" for row in comparisons)) == 4,
        "Primary family changed",
    )
    require(
        next(
            (row for row in population_info if row["population"] == "en_hatecheck_eval")
        )["primary_clusters"]
        == 570,
        "HateCheck component count changed",
    )
    save(BASE / "endpoint_setup.json", endpoint_records)
    save(BASE / "comparison_map.json", comparisons)
    save(BASE / "population_setup.json", population_info)
    save(
        BASE / "encoder_references.json",
        {
            "completion": r.reference(ENCODERS),
            "completion_sha256": c.sha(ENCODERS),
            "runs": encoder_info,
            "relation_to_GPU": "Historical fixed encoder references only; equal-budget GPU study is separate.",
        },
    )
    prompt_metadata = inspect_prompts(roster)
    save(BASE / "prompt_setup.json", prompt_metadata)
    save(
        BASE / "setup_complete.json",
        {
            "status": "reader_and_comparison_setup_complete_without_effectiveness_scores",
            "sources": bindings(
                [
                    Path(__file__),
                    Path(scoring.__file__),
                    Path(prompt.__file__),
                    c.PREP / "broad_roster.jsonl",
                    ROOT / "config/taxonomy.yaml",
                    ENCODERS,
                    REPLAY,
                    BASE / "assembly.json",
                ]
            ),
            "outputs": bindings(
                [
                    BASE / name
                    for name in [
                        "endpoint_setup.json",
                        "comparison_map.json",
                        "population_setup.json",
                        "encoder_references.json",
                        "prompt_setup.json",
                    ]
                ]
            ),
            "new_Qwen_calls": 0,
            "new_inference": 0,
            "performance_scores": 0,
            "bootstrap_draws": 0,
        },
    )
    print(
        "Comparison setup complete:",
        len(endpoint_records),
        "endpoints,",
        len(comparisons),
        "planned contrasts",
        flush=True,
    )


def inspect_prompts(roster):
    catalog = {}
    summary = []
    shared_system = None
    empty = dict(definitions=[], guidelines=[], example_groups=[])
    for cond, name in [
        ("rag_full", "full/prompts.jsonl"),
        ("rag_small", "small/prompts.jsonl"),
        ("zero_shot", "zero_shot.jsonl"),
    ]:
        path = c.BASE / "retrieval_v3" / name
        count = 0
        channel_counts = defaultdict(Counter)
        for row in rows(path):
            key = row["key"]
            require(
                key in roster
                and row["condition"] == cond
                and ((key, cond) not in catalog),
                "Prompt roster differs",
            )
            source = roster[key]
            require(
                row["population"] == source["phase"]
                and row["lang"] == source["lang"]
                and (
                    str(row["item_id"])
                    == str(source["id"]).removeprefix(source["lang"] + "/")
                ),
                "Prompt source identity differs",
            )
            require(
                c.signature(row["messages"]) == row["messages_sha256"],
                "Saved prompt digest differs",
            )
            require(
                [x["role"] for x in row["messages"]] == ["system", "user"],
                "Unexpected prompt roles",
            )
            system = row["messages"][0]["text"]
            if shared_system is None:
                shared_system = system
            require(system == shared_system, "Shared system instructions differ")
            context = empty if cond == "zero_shot" else row["context"]
            require(
                row["messages"]
                == prompt.render_messages(
                    source["text"], source["lang"], context, "corrected"
                ),
                "Saved prompt does not render from its exact saved input/context",
            )
            if cond != "zero_shot":
                require(
                    c.signature(context) == row["context_sha256"]
                    and prompt.hit_ids(context) == row["retrieved"],
                    "Saved context binding differs",
                )
            for channel in ["definitions", "guidelines"]:
                channel_counts[channel][len(context[channel])] += 1
            channel_counts["examples"][
                sum((len(items) for _, items in context["example_groups"]))
            ] += 1
            catalog[key, cond] = {
                "messages_sha256": row["messages_sha256"],
                "population": row["population"],
            }
            count += 1
        require(count == 21627, "Prompt count differs")
        summary.append(
            {
                "condition": cond,
                "path": r.reference(path),
                "sha256": c.sha(path),
                "rows": count,
                "context_channel_count_histograms": {
                    k: dict(v) for k, v in channel_counts.items()
                },
            }
        )
        print("Checked exact saved prompt construction:", cond, flush=True)
    schema_sha = c.sha(ROOT / "config/taxonomy.yaml")
    schedule = c.read(r.SCHEDULE)
    for stage in schedule["stages"]:
        require(
            stage["items"] == len(stage["jobs"])
            and stage["ordinary_calls"] == 3 * stage["items"] <= 12000,
            "Batch size differs",
        )
        for job in stage["jobs"]:
            cat = catalog[job["key"], job["condition"]]
            require(
                all((job[k] == cat[k] for k in ["population", "messages_sha256"]))
                and job["schema_path"] == "config/taxonomy.yaml"
                and (job["schema_sha256"] == schema_sha),
                "Scheduled prompt/schema differs",
            )
    return {
        "conditions": summary,
        "shared_system_instructions": shared_system,
        "shared_system_sha256": hashlib.sha256(shared_system.encode()).hexdigest(),
        "zero_shot": "Same system instructions and schema; user message contains only exact input and language. No retrieved definitions, guidelines or examples.",
        "scope": "Construction/identity checks only. Context relevance, overlap, outcomes and resource analysis remain later parts.",
        "new_retrieval_queries": 0,
        "new_embeddings": 0,
    }


def load():

    seal = c.read(BASE / "part1_complete.json")
    require(
        seal["status"] == "part1_reader_and_comparison_setup_independently_verified",
        "Part 1 not sealed",
    )
    r.verify(seal["outputs"])
    assembly = c.read(BASE / "assembly.json")
    r.verify(assembly["outputs"])
    setup = c.read(BASE / "setup_complete.json")
    r.verify(setup["outputs"])
    runs = defaultdict(dict)
    for row in rows(BASE / "predictions.jsonl"):
        runs[row["condition"], row["repeat"]][row["key"]] = row["result"]
    refs = c.read(BASE / "encoder_references.json")
    for entry in refs["runs"]:
        require(
            c.sha(ROOT / entry["path"]) == entry["sha256"], "Encoder reference changed"
        )
        for row in rows(ROOT / entry["path"]):
            runs[row["condition"], row["seed"]][row["key"]] = row["result"]
    groups, dims = scoring.scientific_rosters("broad")
    return (
        groups,
        dims,
        dict(runs),
        c.read(BASE / "endpoint_setup.json"),
        c.read(BASE / "comparison_map.json"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["prepare", "assemble", "setup"], required=True
    )
    args = parser.parse_args()
    with r.original.offline():
        {"prepare": prepare, "assemble": assemble, "setup": comparison_setup}[
            args.mode
        ]()
