"""Policy labels."""

import os

os.environ.update(
    CUDA_VISIBLE_DEVICES="",
    PYTHONDONTWRITEBYTECODE="1",
    OMP_NUM_THREADS="2",
    OPENBLAS_NUM_THREADS="2",
)
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
from analysis.knowledge.policy import common as c
from analysis.knowledge.policy import reader as reader
from analysis.knowledge.policy import metrics as metric

BASE = c.ROOT / "experiments/astra/complete_policy_part4_20261115_v1"
PART3 = reader.BASE
CONDITIONS = c.CONDITIONS
CONTRASTS = ((2, 0), (1, 0), (2, 1))
CASE_SEED = "complete-policy-part4-cases-20261115-v1"
NAMES = {
    "en_hate": "English hate",
    "en_hate_type": "English hate types",
    "de_hate": "German hate",
    "de_legal": "German legal labels",
    "en_target_group": "English targets",
    "en_severity": "English severity",
}


def save(path, value):
    c.require(c.normal(path).is_relative_to(BASE), "Derived write outside Part 4")
    c.save(path, value)


def rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as f:
        for v in values:
            f.write(json.dumps(v, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def freeze():
    target = BASE / "analysis_contract.json"
    c.require(not target.exists(), "Part 4 setup exists; preserve it")
    done = c.read(PART3 / "part3_complete.json")
    c.require(
        done["status"] == "PART_3_COMPLETE_PERFORMANCE_ERRORS_AND_CONSISTENCY_VERIFIED",
        "Part 3 incomplete",
    )
    for name in ("artifacts", "reports", "scripts"):
        c.verify(done[name])
    p3root = c.ROOT / c.read(PART3 / "analysis_complete.json")["run"]
    inputs = [
        PART3 / "part3_complete.json",
        PART3 / "reader_complete.json",
        p3root / "complete.json",
        c.SETUP / "policy_inventory.json",
        c.SETUP / "context_design_rows.jsonl",
        c.SETUP / "prompt_bindings.jsonl",
        c.SETUP / "endpoints.json",
        c.SETUP / "roster.jsonl",
        c.SETUP / "supporting_length_groups.json",
        c.BASE / "prompts.jsonl",
        c.BASE / "prompt_index.json",
        c.PLAN,
        c.ORIGINAL_PROMPTS,
        reader.LIVE / "complete.json",
        c.ROOT / "Thesis_Log/astra_result/complete_policy_work_plan_20261114_v2.md",
        c.ROOT / "Thesis_Log/astra_result/complete_policy_part1_protocol_20261114_v1.md",
    ]
    save(
        target,
        dict(
            status="frozen_part4_descriptive_analysis_rules",
            created_utc=c.a.utc(),
            inputs=c.bindings(inputs),
            part3_root=c.ref(p3root),
            thesis_before=reader.snapshot_thesis(),
            main_mask="Unchanged Part 3 common-nine endpoint cohorts. Full eligible roster also used for exposure counts.",
            scope="Actual initial policy content/order/length, sources, labels, descriptive exposure associations, paired cases. No new model requests or thesis edits.",
            inference="Part 3 main intervals remain authoritative. No new subgroup significance tests. Exposure groups are not randomized and are not isolated rule effects.",
            groups=[
                "source",
                "A hate/dimension definition present",
                "A gold-label definition coverage none/some/all/empty gold",
                "A legal guideline count 0/1/2",
                "each original guideline present/absent",
                "frozen A-token quartiles",
                "original retrieved-order reversal in complete blocks",
            ],
            label_views="Fixed label universe; per-label TP/FP/FN changes and gold-positive recall split by original matching-definition presence.",
            case_selection=dict(
                seed=CASE_SEED,
                main="C versus A: each endpoint/source; separate gold classes for hate and empty/nonempty legal; one fix and one regression per cell.",
                extra="B-A and C-B: one fix and one regression for each of types, severity and legal, across that endpoint.",
                persistent="One identical stable-wrong A/C case per endpoint.",
                priority="Prefer strict stable transitions (wrong in every right repeat and correct in every left repeat, or reverse). If none, largest signed change in number of correct repeats. Break ties with SHA256(seed|key). Empty cells remain empty.",
                rationale="No reading query text or rationale to rank candidates; all selected examples are illustrative of their outcome stratum, not representative population evidence.",
            ),
            preservation="No writes to raw evidence, prior analyses, scientific source files, original thesis or writing copy. Concurrent writer changes logged separately.",
        ),
    )
    (BASE / "session_before.md").write_bytes((c.ROOT / "Thesis_Log/astra_session.md").read_bytes())
    print("Part 4 rules and source snapshot saved.", flush=True)


def load_inputs():
    contract = c.read(BASE / "analysis_contract.json")
    c.verify(contract["inputs"])
    root = c.ROOT / contract["part3_root"]
    roster = {v["key"]: v for v in c.rows(c.SETUP / "roster.jsonl")}
    endpoints = c.read(c.SETUP / "endpoints.json")
    result = {}
    for endpoint in endpoints:
        with np.load(root / endpoint / "encoded.npz", allow_pickle=False) as f:
            result[endpoint] = {k: f[k].copy() for k in f.files}
    return (contract, root, roster, endpoints, result)


def definition_line(record):
    return (
        "- "
        + record["dimension"]
        + (" / " + record["label"] if record.get("label") else "")
        + ": "
        + " ".join(record["text"].split())
    )


def sections(user):
    first = "## Label definitions\n"
    second = "\n\n## Annotation guidelines\n"
    third = "\n\n## Labelled examples\n"
    c.require(
        user.startswith(first) and user.count(second) == 1 and (user.count(third) == 1),
        "Policy section boundaries differ",
    )
    definitions, rest = user[len(first) :].split(second, 1)
    guidelines, suffix = rest.split(third, 1)
    return (definitions.splitlines(), guidelines.splitlines(), "## Labelled examples\n" + suffix)


def audit_payload(entry):
    body = c.a.body(entry["messages"])
    digest = c.signature(body)
    path = reader.LIVE / "payloads" / f"{digest}.json"
    c.require(c.sha(path) == digest, "Saved initial request body differs: " + entry["key"])
    return dict(
        key=entry["key"],
        condition=entry["condition"],
        messages_sha256=entry["messages_sha256"],
        payload_sha256=digest,
    )


def contexts():
    contract, root, roster, endpoints, arrays = load_inputs()
    inventory = c.read(c.SETUP / "policy_inventory.json")
    records = {v["record"]["id"]: v["record"] for v in inventory}
    ids = {
        kind: [v["record"]["id"] for v in inventory if v["record"]["kind"] == kind]
        for kind in ("definition", "guideline")
    }
    for value in inventory:
        c.require(
            c.signature(value["record"]) == value["record_sha256"], "Inventory record changed"
        )
        c.require(
            hashlib.sha256(value["record"]["text"].encode()).hexdigest() == value["text_sha256"],
            "Policy text changed",
        )
    definition_lookup = {definition_line(records[k]): k for k in ids["definition"]}
    guideline_lookup = {"- " + records[k]["text"]: k for k in ids["guideline"]}
    c.require(
        len(definition_lookup) == 21 and len(guideline_lookup) == 19, "Ambiguous policy mapping"
    )
    bindings = {v["key"]: v for v in c.rows(c.SETUP / "prompt_bindings.jsonl")}
    designed = {v["key"]: v for v in c.rows(c.SETUP / "context_design_rows.jsonl")}
    old_context = {}
    old_suffix = {}
    old_system = {}
    for line, old in enumerate(c.rows(c.ORIGINAL_PROMPTS), 1):
        key = old["key"]
        if key not in roster:
            continue
        c.require(
            line == bindings[key]["source_prompt_line"]
            and c.signature(old["context"]) == bindings[key]["original_context_sha256"]
            and (c.signature(old["messages"]) == bindings[key]["original_messages_sha256"]),
            "Original context binding differs",
        )
        old_context[key] = old["context"]
        old_system[key] = old["messages"][0]
        old_suffix[key] = sections(old["messages"][1]["text"])[2]
    c.require(set(old_context) == set(roster), "Missing original context")
    decoded = {k: {} for k in roster}
    payload_bindings = []
    pending = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for row in c.rows(c.BASE / "prompts.jsonl"):
            key, cond = (row["key"], row["condition"])
            ci = CONDITIONS.index(cond)
            c.require(
                c.signature(row["messages"])
                == row["messages_sha256"]
                == bindings[key]["message_hashes"][cond],
                "Saved prompt hash differs",
            )
            dl, gl, suffix = sections(row["messages"][1]["text"])
            c.require(
                all((x in definition_lookup for x in dl))
                and all((x in guideline_lookup for x in gl)),
                "Unrecognized policy text",
            )
            defs = [definition_lookup[x] for x in dl]
            guides = [guideline_lookup[x] for x in gl]
            wanted_defs = designed[key]["original_definition_ids"] if ci == 0 else ids["definition"]
            wanted_guides = ids["guideline"] if ci == 2 else designed[key]["original_guideline_ids"]
            c.require(
                defs == wanted_defs and guides == wanted_guides,
                "Actual policy identifiers/order differ",
            )
            c.require(
                row["messages"][0] == old_system[key] and suffix == old_suffix[key],
                "System/examples/query changed",
            )
            marker = "## Text to classify (language: " + roster[key]["lang"] + ")\n"
            c.require(
                suffix.count(marker) == 1 and suffix.split(marker, 1)[1] == roster[key]["text"],
                "Query differs from roster",
            )
            c.require(
                row["reference_input_tokens"]
                == bindings[key]["counts"][cond]
                == designed[key]["input_tokens"][cond],
                "Token binding differs",
            )
            decoded[key][cond] = dict(
                definition_ids=defs,
                guideline_ids=guides,
                tokens=row["reference_input_tokens"],
                policy_characters=len(row["messages"][1]["text"]) - len(suffix),
                examples_and_query_sha256=hashlib.sha256(suffix.encode()).hexdigest(),
                messages_sha256=row["messages_sha256"],
            )
            pending.append(row)
            if len(pending) == 192:
                payload_bindings.extend(pool.map(audit_payload, pending))
                pending = []
                if len(payload_bindings) % 1920 == 0:
                    print(
                        "Actual initial request bodies checked:",
                        len(payload_bindings),
                        "/20718",
                        flush=True,
                    )
        if pending:
            payload_bindings.extend(pool.map(audit_payload, pending))
    c.require(
        len(payload_bindings) == 20718
        and all((set(v) == set(CONDITIONS) for v in decoded.values())),
        "Incomplete prompt inspection",
    )
    features = []
    for key in sorted(roster):
        a = decoded[key][CONDITIONS[0]]
        d = a["definition_ids"]
        g = a["guideline_ids"]
        ex = [record for _, group in old_context[key]["example_groups"] for record in group]
        c.require(
            [x["id"] for x in ex] == designed[key]["original_example_ids"] and len(ex) == 5,
            "Examples were replaced",
        )
        def_reversed = [x for x in ids["definition"] if x in d] != d
        guide_reversed = [x for x in ids["guideline"] if x in g] != g
        features.append(
            dict(
                key=key,
                source=roster[key]["source"],
                population=roster[key]["phase"],
                definitions_A=d,
                guidelines_A=g,
                definition_dimensions_A=[records[x]["dimension"] for x in d],
                legal_guidelines_A=sum((records[x]["dimension"] == "legal" for x in g)),
                general_guidelines_A=sum((records[x]["dimension"] == "hate" for x in g)),
                definitions_added=[x for x in ids["definition"] if x not in d],
                guidelines_added=[x for x in ids["guideline"] if x not in g],
                definition_relative_order_reversed=def_reversed,
                guideline_relative_order_reversed=guide_reversed,
                example_ids=[x["id"] for x in ex],
                example_sources=dict(Counter((x["meta"].get("source") for x in ex))),
                example_gate_counts=dict(Counter((str(x["meta"].get("gate")) for x in ex))),
                settings=decoded[key],
            )
        )
    rows(BASE / "context_features.jsonl", features)
    rows(BASE / "initial_payload_bindings.jsonl", payload_bindings)
    rows(
        BASE / "selected_original_contexts.jsonl",
        [dict(key=k, context=v) for k, v in sorted(old_context.items())],
    )
    summaries = {}
    lookup = {x["key"]: x for x in features}
    populations = {
        **{name: meta["keys"] for name, meta in endpoints.items()},
        **{
            phase: [k for k, v in roster.items() if v["phase"] == phase]
            for phase in sorted({v["phase"] for v in roster.values()})
        },
        "all_unique": list(roster),
    }
    for name, keys in populations.items():
        settings = {}
        for cond in CONDITIONS:
            tokens = [lookup[k]["settings"][cond]["tokens"] for k in keys]
            settings[cond] = dict(
                mean_tokens=statistics.mean(tokens),
                median_tokens=statistics.median(tokens),
                min_tokens=min(tokens),
                max_tokens=max(tokens),
            )
        summaries[name] = dict(
            n=len(keys),
            tokens=settings,
            definitions_A=dict(Counter((x for k in keys for x in lookup[k]["definitions_A"]))),
            guidelines_A=dict(Counter((x for k in keys for x in lookup[k]["guidelines_A"]))),
            definition_dimensions_A=dict(
                Counter((x for k in keys for x in lookup[k]["definition_dimensions_A"]))
            ),
            legal_guidelines_A=dict(Counter((str(lookup[k]["legal_guidelines_A"]) for k in keys))),
            any_dimension_definition_A={
                field: sum((field in lookup[k]["definition_dimensions_A"] for k in keys))
                for field in ("hate", "hate_type", "target_group", "severity", "legal")
            },
            definition_order_reversed=sum(
                (lookup[k]["definition_relative_order_reversed"] for k in keys)
            ),
            guideline_order_reversed=sum(
                (lookup[k]["guideline_relative_order_reversed"] for k in keys)
            ),
        )
    save(
        BASE / "context_summary.json",
        dict(
            populations=summaries,
            policy_ids=ids,
            counts=dict(
                inputs=6906,
                prompts=20718,
                initial_payload_files=20718,
                all_example_groups_unchanged=True,
                examples_replaced=0,
                definitions_added_per_B_or_C=19,
                guidelines_added_per_C=17,
            ),
            scope="All saved unique ordinary request bodies, exact policy record mapping, original suffix/system and Part 3 completed-output bindings. Repair replay was already verified in Part 3.",
        ),
    )
    save(
        BASE / "context_complete.json",
        dict(
            status="ALL_INITIAL_CONTEXTS_AND_PAYLOADS_VERIFIED",
            artifacts=c.bindings(
                [
                    BASE / n
                    for n in (
                        "context_features.jsonl",
                        "initial_payload_bindings.jsonl",
                        "selected_original_contexts.jsonl",
                        "context_summary.json",
                    )
                ]
            ),
            script_sha256=c.sha(Path(__file__)),
        ),
    )
    print("All initial policy bodies, examples and queries verified.", flush=True)


def safe_ratio(a, b):
    return None if not b else float(a / b)


def subset(meta, arr, mask):
    mask = np.asarray(mask, dtype=bool) & arr["common"]
    n = int(mask.sum())
    if not n:
        return dict(n=0)
    y = arr["gold"][mask]
    p = arr["predicted"][:, :, mask]
    correct = arr["correct"][:, :, mask]
    counts = metric.contributions(y, p).sum(axis=2)
    mac, mic, label = metric.f1(counts)
    answer = dict(
        n=n,
        gold_support=y.sum(axis=0),
        macro_f1=mac.mean(axis=1),
        macro_repeats=mac,
        micro_f1=mic.mean(axis=1),
        exact=correct.mean(axis=(1, 2)),
        per_label_f1=label.mean(axis=1),
        counts=counts,
        gate_closed=(~arr["gates"][:, :, mask]).mean(axis=(1, 2)),
        labels_with_no_gold=[
            lab for lab, count in zip(meta["all_labels"], y.sum(axis=0)) if count == 0
        ],
        small_descriptive_group=n < 30,
    )
    if meta["dimension"] == "hate":
        yy = y[:, meta["all_labels"].index("hate")]
        pp = p[..., meta["all_labels"].index("hate")]
        conf = {
            name: values.sum(axis=2)
            for name, values in [
                ("tp", pp & yy),
                ("tn", ~pp & ~yy),
                ("fp", pp & ~yy),
                ("fn", ~pp & yy),
            ]
        }
        answer.update(
            positive_n=int(yy.sum()),
            negative_n=int((~yy).sum()),
            confusion_repeats=conf,
            false_positive_rate=[safe_ratio(x, (~yy).sum()) for x in conf["fp"].mean(axis=1)],
            false_negative_rate=[safe_ratio(x, yy.sum()) for x in conf["fn"].mean(axis=1)],
        )
    if meta["dimension"] in ("hate_type", "target_group", "legal"):
        union = (p | y).sum(axis=-1)
        intersection = (p & y).sum(axis=-1)
        jaccard = np.where(union == 0, 1, metric.divide(intersection, union))
        answer.update(
            jaccard=jaccard.mean(axis=(1, 2)),
            extra_labels_per_text=(p & ~y).sum(axis=-1).mean(axis=(1, 2)),
            missing_labels_per_text=(~p & y).sum(axis=-1).mean(axis=(1, 2)),
        )
    transitions = []
    for left, right in CONTRASTS:
        delta = correct[left].sum(axis=0) - correct[right].sum(axis=0)
        stable_fix = (correct[left] & ~correct[right]).all(axis=0)
        stable_regression = (~correct[left] & correct[right]).all(axis=0)
        transitions.append(
            dict(
                left=CONDITIONS[left],
                right=CONDITIONS[right],
                fixes_repeats=(correct[left] & ~correct[right]).sum(axis=1),
                regressions_repeats=(~correct[left] & correct[right]).sum(axis=1),
                stable_fixes=int(stable_fix.sum()),
                stable_regressions=int(stable_regression.sum()),
                more_correct_repeat_texts=int((delta > 0).sum()),
                fewer_correct_repeat_texts=int((delta < 0).sum()),
            )
        )
    answer["transitions"] = transitions
    return metric.serial(answer)


def definition_coverage(meta, roster, feature, key):
    field = meta["dimension"]
    present = set(feature[key]["definitions_A"])
    if field == "hate":
        wanted = {"def-hate"}
    else:
        gold = roster[key][
            {"hate_type": "hate_types", "target_group": "target_groups"}.get(field, field)
        ]
        wanted = {f"def-{field}-{label}" for label in ([gold] if field == "severity" else gold)}
    if not wanted:
        return "empty_gold"
    n = len(present & wanted)
    return "all" if n == len(wanted) else "some" if n else "none"


def case_state(arr, ci, i):
    return arr["states"][ci, :, i].tolist()


def choose_case(keys, arr, positions, left, right, direction):
    candidates = []
    stable = []
    for key in keys:
        i = positions[key]
        if not arr["common"][i]:
            continue
        lv = arr["correct"][left, :, i]
        rv = arr["correct"][right, :, i]
        delta = int(lv.sum()) - int(rv.sum())
        strength = delta if direction == "fix" else -delta
        if strength <= 0:
            continue
        candidates.append((key, strength))
        if (lv & ~rv).all() if direction == "fix" else (~lv & rv).all():
            stable.append((key, strength))
    if not candidates:
        return dict(key=None, candidates=0, stable_candidates=0, selection="empty")
    pool = stable or candidates
    best = max((n for _, n in pool))
    tied = [k for k, n in pool if n == best]
    chosen = min(
        tied, key=lambda k: (hashlib.sha256((CASE_SEED + "|" + k).encode()).hexdigest(), k)
    )
    return dict(
        key=chosen,
        candidates=len(candidates),
        stable_candidates=len(stable),
        selection="all_three_stable_transition" if stable else "largest_net_correct_repeat_change",
        net_strength=best,
    )


def analyze():
    contract, root, roster, endpoints, arrays = load_inputs()
    c.verify(c.read(BASE / "context_complete.json")["artifacts"])
    features = {v["key"]: v for v in c.rows(BASE / "context_features.jsonl")}
    policy = [v["record"] for v in c.read(c.SETUP / "policy_inventory.json")]
    lengths = c.read(c.SETUP / "supporting_length_groups.json")
    output = {}
    label_rows = []
    cases = []
    case_catalogue = []
    for endpoint, meta in endpoints.items():
        arr = arrays[endpoint]
        keys = meta["keys"]
        positions = {k: i for i, k in enumerate(keys)}
        c.require(arr["keys"].tolist() == keys, "Part 3 key order differs")
        sources = sorted({roster[k]["source"] for k in keys})
        group_masks = {}
        for source in sources:
            group_masks["source/" + source] = np.array(
                [roster[k]["source"] == source for k in keys]
            )
        for group in ("none", "some", "all", "empty_gold"):
            group_masks["gold_definition_coverage/" + group] = np.array(
                [definition_coverage(meta, roster, features, k) == group for k in keys]
            )
        for value in range(3):
            group_masks["legal_guidelines_A/" + str(value)] = np.array(
                [features[k]["legal_guidelines_A"] == value for k in keys]
            )
        for flag in (False, True):
            for name in ("definition_relative_order_reversed", "guideline_relative_order_reversed"):
                group_masks[name + "/" + str(flag)] = np.array(
                    [features[k][name] == flag for k in keys]
                )
            group_masks["dimension_definition_A/" + str(flag)] = np.array(
                [
                    (meta["dimension"] in features[k]["definition_dimensions_A"]) == flag
                    for k in keys
                ]
            )
        for group, members in lengths[endpoint]["keys"].items():
            member_set = set(members)
            group_masks["frozen_A_token_quartile/" + group] = np.array(
                [k in member_set for k in keys]
            )
        for entry in policy:
            if entry["kind"] != "guideline":
                continue
            identifier = entry["id"]
            for flag in (False, True):
                group_masks["original_guideline/" + identifier + "/" + str(flag)] = np.array(
                    [(identifier in features[k]["guidelines_A"]) == flag for k in keys]
                )
        groups = {name: subset(meta, arr, mask) for name, mask in group_masks.items()}
        whole = subset(meta, arr, np.ones(len(keys), bool))
        old = c.read(root / endpoint / "results.json")["point"]
        c.require(
            np.allclose(whole["macro_f1"], old["macro_f1"])
            and np.allclose(whole["exact"], old["exact"]),
            "Part 3 headline does not reproduce",
        )
        for source in sources:
            old_source = c.read(root / endpoint / "results.json")["sources"][source]
            c.require(
                np.allclose(groups["source/" + source]["macro_f1"], old_source["macro_f1"]),
                "Part 3 source macro differs",
            )
        for j, label in enumerate(meta["all_labels"]):
            yy = arr["gold"][:, j]
            pp = arr["predicted"][:, :, :, j]
            expected = (
                "def-hate" if meta["dimension"] == "hate" else f"def-{meta['dimension']}-{label}"
            )
            present = np.array([expected in features[k]["definitions_A"] for k in keys])
            for truth in (False, True):
                for exposed in (False, True):
                    mask = arr["common"] & (yy == truth) & (present == exposed)
                    n = int(mask.sum())
                    if not n:
                        continue
                    label_rows.append(
                        dict(
                            endpoint=endpoint,
                            label=label,
                            gold_positive=truth,
                            definition_A_present=exposed,
                            n=n,
                            predicted_positive_rate=pp[:, :, mask].mean(axis=(1, 2)).tolist(),
                            predicted_positive_counts=pp[:, :, mask].sum(axis=2).tolist(),
                            small_group=n < 30,
                            meaning="recall" if truth else "false_positive_rate",
                            causal_warning="Original exposure follows retrieval and is confounded with the text and the other supplied records.",
                        )
                    )
        for i, key in enumerate(keys):
            if not arr["common"][i]:
                continue
            decisions = {
                CONDITIONS[ci]: dict(
                    correct=arr["correct"][ci, :, i].tolist(), states=case_state(arr, ci, i)
                )
                for ci in range(3)
            }
            for left, right in CONTRASTS:
                a = arr["correct"][left, :, i]
                b = arr["correct"][right, :, i]
                case_catalogue.append(
                    dict(
                        endpoint=endpoint,
                        key=key,
                        source=roster[key]["source"],
                        left=CONDITIONS[left],
                        right=CONDITIONS[right],
                        correct_repeat_delta=int(a.sum()) - int(b.sum()),
                        stable_fix=bool((a & ~b).all()),
                        stable_regression=bool((~a & b).all()),
                        changed_state_pairs=int(
                            (arr["states"][left, :, i] != arr["states"][right, :, i]).sum()
                        ),
                        correct=decisions,
                        original_gold_definition_coverage=definition_coverage(
                            meta, roster, features, key
                        ),
                    )
                )
        for source in sources:
            source_keys = [k for k in keys if roster[k]["source"] == source]
            if meta["dimension"] == "hate":
                strata = {
                    str(truth): [k for k in source_keys if roster[k]["gate"] == truth]
                    for truth in (False, True)
                }
            elif meta["dimension"] == "legal":
                strata = {
                    name: [k for k in source_keys if bool(roster[k]["legal"]) == truth]
                    for name, truth in [("empty", False), ("nonempty", True)]
                }
            else:
                strata = {"all": source_keys}
            for stratum, kk in strata.items():
                for direction in ("fix", "regression"):
                    cases.append(
                        dict(
                            endpoint=endpoint,
                            source=source,
                            stratum=stratum,
                            left=CONDITIONS[2],
                            right=CONDITIONS[0],
                            direction=direction,
                            **choose_case(kk, arr, positions, 2, 0, direction),
                        )
                    )
        if endpoint in ("en_hate_type", "en_severity", "de_legal"):
            for left, right in ((1, 0), (2, 1)):
                for direction in ("fix", "regression"):
                    cases.append(
                        dict(
                            endpoint=endpoint,
                            source="all",
                            stratum="middle_condition",
                            left=CONDITIONS[left],
                            right=CONDITIONS[right],
                            direction=direction,
                            **choose_case(keys, arr, positions, left, right, direction),
                        )
                    )
        persistent = [
            k
            for i, k in enumerate(keys)
            if arr["common"][i]
            and (not arr["correct"][0, :, i].any())
            and (not arr["correct"][2, :, i].any())
            and (len(set(arr["states"][0, :, i].tolist() + arr["states"][2, :, i].tolist())) == 1)
        ]
        key = (
            min(
                persistent,
                key=lambda k: (hashlib.sha256((CASE_SEED + "|" + k).encode()).hexdigest(), k),
            )
            if persistent
            else None
        )
        cases.append(
            dict(
                endpoint=endpoint,
                source="all",
                stratum="persistent_identical_wrong",
                left=CONDITIONS[2],
                right=CONDITIONS[0],
                direction="persistent_wrong",
                key=key,
                candidates=len(persistent),
                selection="same_wrong_state_all_six_A_C_repeats",
            )
        )
        extras = {}
        if meta["dimension"] == "severity":
            mask = arr["common"]
            gold = arr["gold"][mask].argmax(axis=1)
            p = arr["predicted"][:, :, mask]
            pred = np.where(p.any(axis=-1), p.argmax(axis=-1), 3)
            confusion = np.zeros((3, 3, 3, 4), int)
            for ci in range(3):
                for rep in range(3):
                    for gg, ppp in zip(gold, pred[ci, rep]):
                        confusion[ci, rep, gg, ppp] += 1
            extras["severity_gold_by_prediction"] = dict(
                gold=["low", "medium", "high"],
                predicted=["low", "medium", "high", "absent"],
                counts_repeats=confusion.tolist(),
            )
        output[endpoint] = dict(
            meta={k: v for k, v in meta.items() if k != "keys"},
            whole=whole,
            groups=groups,
            persistent_identical_wrong_count=len(persistent),
            **extras,
        )
        print("Descriptive groups and case strata complete:", endpoint, flush=True)
    save(BASE / "groups_and_sources.json", output)
    save(BASE / "label_definition_exposure.json", label_rows)
    save(BASE / "case_selection.json", cases)
    rows(BASE / "all_paired_cases.jsonl", case_catalogue)
    save(
        BASE / "descriptive_complete.json",
        dict(
            status="descriptive_analysis_complete_no_new_significance_tests",
            artifacts=c.bindings(
                [
                    BASE / name
                    for name in (
                        "groups_and_sources.json",
                        "label_definition_exposure.json",
                        "case_selection.json",
                        "all_paired_cases.jsonl",
                    )
                ]
            ),
            case_slots=len(cases),
            selected_slots=sum((x["key"] is not None for x in cases)),
            unique_case_texts=len({x["key"] for x in cases if x["key"]}),
            script_sha256=c.sha(Path(__file__)),
        ),
    )
    print("Saved all paired cases and deterministic review selection.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "contexts", "analyze"])
    args = parser.parse_args()
    with c.offline():
        globals()[args.mode]()
