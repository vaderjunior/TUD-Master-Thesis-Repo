import os

for _k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_k] = "2"
from collections import Counter
import argparse
import csv
import hashlib
import json
import socket
import unicodedata
from unittest.mock import patch
import numpy as np
import pandas as pd
from analysis.common import feedback_inputs as c
from analysis.feedback.summary.validate_closing_feasibility import blocks
from analysis.common.prompt import render_messages, corrected_gold

PATHS = {
    "factorial": c.BASE / "factorial",
    "human_meaning": c.BASE / "human_meaning_v2",
    "benign_selection": c.BASE / "benign_selection",
}


def factorial(root, roster, prompts):
    old = {
        (p["item_id"], p["condition"]): p for p in c.rows(c.OLD_TAX / "prompts.jsonl")
    }
    expected = {
        (p["id"], p["label"])
        for p in c.read(c.REVIEW / "rendered_block_audit.json")[
            "both_changed_pair_inventory"
        ]
    }
    c.require(
        {(r["id"], r["target_label"]) for r in roster.values()} == expected,
        "Changed pair sampling",
    )
    checks = 0
    endpoints = 0
    for p in prompts:
        r = roster[p["key"]]
        iid = r["id"]
        label = r["target_label"]
        cond = p["condition"]
        lo = old[iid, "rag_schema_" + label]
        hi = old[iid, "rag_full"]
        lb = blocks(lo)
        hb = blocks(hi)
        chosen_e = hb if cond[1] == "1" else lb
        chosen_d = hb if cond[3] == "1" else lb
        user = (
            "\n\n".join(
                [chosen_d["definitions"], lb["guidelines"], chosen_e["examples"]]
            )
            + "\n\n## Text to classify (language: en)\n"
            + r["text"]
        )
        c.require(
            p["messages"] == [hi["messages"][0], {"role": "user", "text": user}],
            "Independent crossed-block reconstruction failed",
        )
        c.require(
            p["context"]["guidelines"] == lo["context"]["guidelines"],
            "Guideline metadata changed",
        )
        c.require(
            len(p["context"]["definitions"]) == 2
            and len(p["context"]["guidelines"]) == 2
            and (sum((len(rs) for _, rs in p["context"]["example_groups"])) == 5),
            "Slot budget drift",
        )
        if cond in ("E0D0", "E1D1"):
            c.require(
                p["messages"] == (hi if cond == "E1D1" else lo)["messages"],
                "Endpoint drift",
            )
            endpoints += 1
        c.require(
            r["target_positive"] == (label in (r["gold"]["hate_type"] or [])),
            "Target support drift",
        )
        checks += 1
    return {
        "independently_reconstructed_prompts": checks,
        "unchanged_endpoint_messages": endpoints,
        "original_texts": len({r["id"] for r in roster.values()}),
        "pairs": len(roster),
        "hateful_pairs": sum(
            (r["population"] == "hateful_types" for r in roster.values())
        ),
        "target_positive_pairs": sum((r["target_positive"] for r in roster.values())),
    }


def human(root, roster, prompts):
    old = {
        (p["item_id"], p["condition"]): p for p in c.rows(c.OLD_EX / "prompts.jsonl")
    }
    cards = {r["source_id"]: r for r in c.rows(root / "source_review_cards.jsonl")}
    with (c.ROOT / "data/raw/implicit_hate/implicit_hate_v1_stg3_posts.tsv").open(
        encoding="utf8", newline=""
    ) as stream:
        raw = list(csv.DictReader(stream, delimiter="\t"))
    excluded = {"implicit_hate-15757", "implicit_hate-19841"}
    choices = {
        (r["query_id"], r["pool_seed"]): r for r in c.rows(c.OLD_EX / "choices.jsonl")
    }
    expected_roster = {r["key"]: r for r in c.rows(c.OLD_EX / "roster.jsonl")}
    c.require(set(roster) == set(expected_roster), "Changed human-study roster")
    for sid, card in cards.items():
        raw_matches = [
            {"target": r["target"], "implied_statement": r["implied_statement"]}
            for r in raw
            if r["post"] == card["source_text"] and r["implied_statement"].strip()
        ]
        c.require(
            raw_matches == card["raw_annotations"], "Human annotation join changed"
        )
        c.require(
            card["eligible"] == (bool(raw_matches) and sid not in excluded),
            "Unexpected source quality rule",
        )
        if card["eligible"]:
            annotation = raw_matches[0]
            expected = (
                "Annotator-supplied target: "
                + json.dumps(annotation["target"], ensure_ascii=False)
                + ". Annotator interpretation of the source: "
                + json.dumps(annotation["implied_statement"], ensure_ascii=False)
                + "."
            )
            c.require(
                expected == card["human_note"], "Human string was edited or invented"
            )
        else:
            c.require(
                not card["human_note"] and (not card["generic_note"]),
                "Fallback has extra information",
            )
    unchanged = 0
    added = Counter()
    tok = c.tokenizer()
    for p in prompts:
        r = roster[p["key"]]
        original_r = expected_roster[p["key"]]
        c.require(
            all((r[k] == original_r[k] for k in original_r)),
            "Changed input/gold/cluster",
        )
        if p["condition"] == "reference":
            original = old[r["id"], "reference"]
            card = None
            policy = "reference"
        else:
            policy, seed_string = p["condition"].split("_s")
            seed = int(seed_string)
            original = old[r["id"], f"plain_always_s{seed}"]
            sid = choices[r["id"], seed]["source_id"]
            card = cards[sid]
            c.require(
                p["source_id"] == sid
                and sid != r["id"]
                and (card["source_text"] != r["text"]),
                "Query annotation/source leak",
            )
            feedback = original["context"]["example_groups"][-1][1]
            c.require(
                len(feedback) == 1
                and feedback[0]["id"] == "sq3fb-" + sid
                and (feedback[0]["text"] == card["source_text"]),
                "Source not bound to delivered example",
            )
        c.require(
            p["context"] == original["context"],
            "Human policy changed retrieval context",
        )
        if policy in ("reference", "plain") or not card["eligible"]:
            c.require(
                p["messages"] == original["messages"],
                "Unchanged/fallback prompt differs",
            )
            unchanged += 1
        else:
            suffix = "\n\n## Text to classify (language: en)\n" + r["text"]
            note = card[policy + "_note"]
            prefix = "  Source interpretation note (an annotation, not a fact about a group or the new input): "
            expected_user = (
                original["messages"][1]["text"][: -len(suffix)]
                + "\n"
                + prefix
                + note
                + suffix
            )
            c.require(
                p["messages"]
                == [original["messages"][0], {"role": "user", "text": expected_user}],
                "Source note placement differs",
            )
            added[policy] += 1
            c.require(
                abs(card["generic_minus_human_tokens"]) <= 5,
                "Control length difference exceeds reviewed bound",
            )
    c.require(
        len(cards) == 58
        and sum((r["eligible"] for r in cards.values())) == 39
        and (unchanged == 3180)
        and (added == {"human": 1200, "generic": 1200}),
        "Human matrix counts differ",
    )
    return {
        "source_exact_joins": 58,
        "eligible_source_cards": 39,
        "raw_available_cards": 41,
        "plain_fallback_reference_matches": unchanged,
        "annotated_prompts_by_policy": dict(added),
        "generic_token_difference_bound": 5,
    }


def benign(root, roster, prompts):
    core = c.rows(c.CORE / "dev_inputs.jsonl")
    protected = {r["item_id"] for r in core if r["lang"] == "en"}
    ev = {r["id"] for r in c.rows(c.OLD_TAX / "roster.jsonl")}
    ids = {r["id"] for r in roster.values()}
    c.require(not ids & (protected | ev), "Source/evaluation ID leak")
    c.require(
        len(ids) == 1024
        and Counter((r["gold"]["hate"] for r in roster.values()))
        == {True: 512, False: 512},
        "Source polarity counts differ",
    )
    df = pd.read_parquet(c.ROOT / "data/processed/en_dev.parquet")
    dev = {r["id"]: r for r in df.to_dict("records")}
    ordered = c.rows(root / "roster.jsonl")
    qv = np.load(root / "query_vectors.npy", allow_pickle=False)
    prov = c.rows(root / "query_vector_sources.jsonl")
    arrays = {}
    for i, r in enumerate(ordered):
        raw = dev[r["id"]]
        c.require(
            raw["text"] == r["text"]
            and raw["source"] == "implicit_hate"
            and (raw["gate"] == r["gold"]["hate"]),
            "Candidate origin drift",
        )
        pr = prov[i]
        c.require(
            pr["id"] == r["id"]
            and pr["text_sha256"] == hashlib.sha256(r["text"].encode()).hexdigest(),
            "Vector/text provenance drift",
        )
        if pr["file"] not in arrays:
            with np.load(c.ROOT / pr["file"], allow_pickle=False) as data:
                arrays[pr["file"]] = data["vectors"].copy()
        c.require(
            np.array_equal(qv[i], arrays[pr["file"]][pr["row"]]),
            "Changed vector payload",
        )
    records = c.rows(c.CORE / "kb/full/records.jsonl")
    known = {r["id"]: r for r in records}
    fv = np.load(
        c.CORE / "retrieval_v1/full_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    index = [
        i for i, r in enumerate(records) if r["kind"] == "example" and r["lang"] == "en"
    ]
    train = np.asarray(fv[index], dtype=np.float64)
    train /= np.sqrt((train * train).sum(axis=1))[:, None]
    query = np.asarray(qv, dtype=np.float64)
    query /= np.sqrt((query * query).sum(axis=1))[:, None]
    maximum = -1.0
    for start in range(0, len(query), 32):
        value = float(
            np.max(
                np.einsum("id,jd->ij", query[start : start + 32], train, optimize=True)
            )
        )
        maximum = max(maximum, value)
        c.require(value < 0.9495, "Retained source is too close to existing TRAIN")
    prefix = lambda t: " ".join(
        unicodedata.normalize("NFKC", t).casefold().split()
    ).lstrip(": ")
    protected_text = {prefix(r["text"]) for r in core if r["lang"] == "en"}
    train_text = {prefix(records[i]["text"]) for i in index}
    c.require(
        all((prefix(r["text"]) not in protected_text | train_text for r in ordered)),
        "Normalized text leak",
    )
    all_roster = c.rows(root / "candidate_universe.jsonl")
    all_ids = {r["id"] for r in all_roster}
    details = c.read(root / "scope_details.json")
    c.require(
        ids <= set(details["eligible_before_cap"])
        and ids.isdisjoint(details["excluded"]),
        "Scope membership drift",
    )
    for group in details["components"]:
        if set(group) & ids:
            c.require(
                not set(group) & (protected | set(details["candidate_TRAIN_exposed"]))
                and len(set(group) & ids) == 1,
                "Transitive split leak",
            )
    first = c.read(root / "retrieval_first_pass.json")
    cold = c.read(root / "cold_replay.json")
    c.require(
        first["prompts_sha256"] == c.sha(root / "prompts.jsonl")
        and cold["compared"] == 1024
        and (not cold["differences"]),
        "Cold prompt reproducibility failed",
    )
    for p in prompts:
        r = roster[p["key"]]
        ctx = p["context"]
        c.require(
            p["messages"] == render_messages(r["text"], "en", ctx, "corrected"),
            "Baseline renderer mismatch",
        )
        c.require(
            len(ctx["definitions"]) == 2
            and len(ctx["guidelines"]) == 2
            and (sum((len(rs) for _, rs in ctx["example_groups"])) == 5),
            "Baseline budgets changed",
        )
        for hit in (
            ctx["definitions"]
            + ctx["guidelines"]
            + [hit for _, group in ctx["example_groups"] for hit in group]
        ):
            source = known[hit["id"]]
            c.require(hit["text"] == source["text"], "Unknown retrieved text")
            for field, value in source.get("meta", {}).items():
                c.require(
                    hit["meta"].get(field) == value, "Retrieved annotation differs"
                )
        c.require(
            not any(
                (
                    hit["id"] in ids
                    for _, group in ctx["example_groups"]
                    for hit in group
                )
            ),
            "Selection source already in KB",
        )
    c.verify(c.read(root / "cache_bindings.json"))
    return {
        "candidate_sources": 1024,
        "hateful": 512,
        "benign": 512,
        "exact_cached_vectors": 1024,
        "retained_source_max_cosine_to_English_TRAIN_float64": maximum,
        "cold_prompt_reconstructions": 1024,
        "query_gold_used_for_retrieval": False,
        "correction_sources_inside_evaluation": 0,
    }


def main(study):
    root = PATHS[study]
    prepared = c.read(root / "prepared.json")
    c.verify(prepared["outputs"])
    c.verify(c.read(root / "preparation_plan.json")["inputs"])
    c.preserve_originals()
    roster = {r["key"]: r for r in c.rows(root / "roster.jsonl")}
    prompts = c.rows(root / "prompts.jsonl")
    science = c.read(root / "scientific_plan_v1.json")
    conditions = science["conditions"]
    c.require(len(roster) == len(c.rows(root / "roster.jsonl")), "Duplicate roster")
    c.require(
        len(prompts) == len(roster) * len(conditions)
        and {(p["key"], p["condition"]) for p in prompts}
        == {(k, cond) for k in roster for cond in conditions},
        "Missing/repeated matrix",
    )
    tok = c.tokenizer()
    for p in prompts:
        r = roster[p["key"]]
        c.require(
            p["scientific_plan_sha256"] == c.sha(root / "scientific_plan_v1.json")
            and p["messages_sha256"] == c.sig(p["messages"])
            and (p["context_sha256"] == c.sig(p["context"])),
            "Prompt binding differs",
        )
        c.require(
            p["text"] == r["text"] and p["item_id"] == r["id"],
            "Input attribution differs",
        )
        count = len(
            tok.encode(c.local_chat(p["messages"]), add_special_tokens=False).ids
        )
        c.require(
            count == p["reference_input_tokens"] and count <= 8192,
            "Independent token count differs",
        )
    detail = {
        "factorial": factorial,
        "human_meaning": human,
        "benign_selection": benign,
    }[study](root, roster, prompts)
    c.verify(prepared["outputs"])
    c.preserve_originals()
    report = {
        "status": "passed",
        "study": study,
        "checker_sha256": c.sha(__file__),
        "prepared_sha256": c.sha(root / "prepared.json"),
        "matrix_prompts": len(prompts),
        "ordinary_calls": len(prompts) * 9,
        "checks": detail,
        "new_provider_calls": 0,
        "source_and_output_hashes_verified": len(prepared["outputs"])
        + len(c.read(root / "preparation_plan.json")["inputs"]),
    }
    c.write(root / "independent_verification.json", report)
    c.write(
        root / "complete.json",
        {
            "status": "offline_preparation_independently_verified",
            "prepared_sha256": c.sha(root / "prepared.json"),
            "verification": c.rel(root / "independent_verification.json"),
            "verification_sha256": c.sha(root / "independent_verification.json"),
        },
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("study", choices=list(PATHS))
    args = p.parse_args()
    with patch.object(
        socket.socket, "connect", side_effect=AssertionError("Offline verification")
    ):
        main(args.study)
