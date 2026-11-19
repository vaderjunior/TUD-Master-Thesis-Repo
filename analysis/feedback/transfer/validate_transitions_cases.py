from itertools import combinations
from pathlib import Path
import hashlib
import numpy as np
from analysis.feedback.transfer import transitions as p

a = p.a
OUT = p.OUT


def main():
    a.need(not (OUT / "cases_verified.json").exists(), "Preserve case verification")
    done = a.read(OUT / "cases_complete.json")
    verified = a.read(OUT / "verified.json")
    choice = a.read(OUT / "case_selection.json")
    index = a.read(OUT / "case_index.json")
    for path, digest in done["artifacts"].items():
        a.need(a.sha(a.ROOT / path) == digest, "Changed case artifact")
    a.need(
        verified["calculated_sha256"] == a.sha(OUT / "calculated.json"),
        "Checked numerical checkpoint",
    )
    with np.load(OUT / "arrays.npz", allow_pickle=False) as f:
        S = f["scores"]
        C = f["codes"]
        M = f["messages"]
    matrix = list(a.rows(a.OUT / "matrix.jsonl"))
    meta = list(a.rows(p.P4 / "metadata.jsonl"))
    roster = {r["key"]: r for r in a.rows(a.OUT / "roster.jsonl")}
    heads = a.read(OUT / "array_schema.json")["heads"]
    strict = [i for i, r in enumerate(matrix) if r["masks"]["strict_all_votes"]]
    for name, q in choice.items():
        if name.startswith("part5_context/"):
            continue
        parts = name.split("/")
        eligible = []
        for i in strict:
            r = meta[i]
            ref = int(S[i, 0, :, 0].sum())
            pool = S[i, 1:, :, 0].sum(axis=1).tolist()
            total = sum(pool)
            flags = {
                "consistent_fix": ref == 0 and total == 9,
                "consistent_regression": ref == 3 and total == 0,
                "persistently_wrong": ref == 0 and total == 0,
            }
            ok = False
            if parts[0] == "binary":
                ok = (
                    r["population"] == parts[1]
                    and int(r["gold_hate"]) == int(parts[2][-1])
                    and flags[parts[3]]
                )
            elif parts[0] == "source":
                ok = r["source"] == parts[1] and flags[parts[2]]
            elif parts[0] == "variation" and r["population"] == parts[1]:
                kind = parts[2]
                if kind == "opposing_pool_directions":
                    ok = min(pool) < ref < max(pool)
                elif kind == "equal_average_with_changes":
                    ok = total == 3 * ref and any(
                        S[i, 0, t, 0] != S[i, c, t, 0]
                        for c in (1, 2, 3)
                        for t in range(3)
                    )
                elif kind == "gate_repeat_variation":
                    ok = any(min(C[i, c, :, 0]) != max(C[i, c, :, 0]) for c in range(4))
                else:
                    ok = any(
                        M[i, c] == M[i, d]
                        and any(
                            tuple(C[i, c, t]) != tuple(C[i, d, t]) for t in range(3)
                        )
                        for c, d in combinations(range(1, 4), 2)
                    )
            elif parts[0] == "fine" and r["population"] == parts[1]:
                h = heads.index(parts[2])
                refh = S[i, 0, :, h].tolist()
                if min(refh) < 0:
                    continue
                if parts[3] == "one_pool_consistent_fix":
                    ok = sum(refh) == 0 and any(
                        sum(S[i, c, :, h]) == 3 for c in (1, 2, 3)
                    )
                elif parts[3] == "one_pool_consistent_regression":
                    ok = sum(refh) == 3 and any(
                        sum(S[i, c, :, h]) == 0 for c in (1, 2, 3)
                    )
                else:
                    ok = ref == 0 and any(
                        pool[c - 1] == 3
                        and sum(S[i, c, :, h]) == 0
                        and sum(S[i, c, :, 1]) == 0
                        for c in (1, 2, 3)
                    )
            if ok:
                eligible.append(i)
        a.need(
            eligible == q["candidate_indices"]
            and len(eligible) == q["candidate_count"],
            "Full case eligibility " + name,
        )
        rank = lambda i: hashlib.sha256(
            ("A16-part6-cases-v1|" + name + "|" + meta[i]["key"]).encode()
        ).hexdigest()
        chosen = min(eligible, key=rank) if eligible else None
        a.need(
            chosen == q["input_index"]
            and (chosen is None or rank(chosen) == q["selection_hash"]),
            "Mechanical minimum case " + name,
        )
    prior = a.read(p.P5 / "context_examples.json")
    for name, c in prior.items():
        q = choice["part5_context/" + name]
        a.need(
            q["key"] == c["query_context"]["key"]
            and q["selection_hash"] == c["selection_hash"],
            "Earlier context case retained",
        )
    cases = {q["key"]: a.read(OUT / q["json"]) for q in index}
    a.need(len(cases) == 43 and len(choice) == 46, "Case inventory")
    countprompt = countpred = 0
    for r in a.rows(a.STUDY / "prompts.jsonl"):
        if r["key"] in cases:
            saved = cases[r["key"]]["prompts"][r["condition"]]
            a.need(saved == r, "Full original case prompt")
            countprompt += 1
    for path in sorted((a.OUT / "assembly").glob("transfer_*.jsonl")):
        for r in a.rows(path):
            if r["key"] in cases:
                saved = cases[r["key"]]["actual_records"][
                    r["condition"] + "/" + str(r["repeat"])
                ]
                a.need(
                    saved == {key: r[key] for key in saved},
                    "Full original case output/votes/receipt",
                )
                countpred += 1
    catalog = a.read(p.P5 / "source_catalog.json")
    for r in a.rows(p.P5 / "query_contexts.jsonl"):
        if r["key"] in cases:
            a.need(
                cases[r["key"]]["query_context"] == r, "Original query-specific context"
            )
    deliv = {key: [] for key in cases}
    for r in a.rows(p.P5 / "delivery.jsonl"):
        if r["key"] in cases:
            deliv[r["key"]].append(r)
    for key, c in cases.items():
        i = c["input_index"]
        a.need(
            c["roster"] == roster[key]
            and c["metadata"] == meta[i]
            and c["predictions"] == matrix[i]["predictions"]
            and c["masks"] == matrix[i]["masks"],
            "Original case text/gold/results",
        )
        a.need(
            c["delivered_corrections"] == deliv[key]
            and c["source_records"]
            == {r["source_id"]: catalog[r["source_id"]] for r in deliv[key]},
            "Original correction records",
        )
        a.need(
            c["selected_slots"]
            == [name for name, q in choice.items() if q["key"] == key],
            "Case slot membership",
        )
    a.need(
        countprompt == 172 and countpred == 516,
        "Complete selected context/repeat evidence",
    )
    p.save(
        "cases_verified.json",
        dict(
            status="part6_case_choice_original_context_votes_and_gold_verified",
            checked_utc=a.utc(),
            verifier_sha256=a.sha(Path(__file__)),
            cases_complete_sha256=a.sha(OUT / "cases_complete.json"),
            numerical_verification_sha256=a.sha(OUT / "verified.json"),
            candidate_slots=38,
            prior_context_slots=8,
            empty_slots=2,
            unique_cases=43,
            original_prompts=countprompt,
            original_predictions=countpred,
            original_vote_slots=countpred * 3,
            selection_uses_fixed_hash=True,
            all_empty_slots_retained=True,
            all_original_evidence_unchanged=True,
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        "All 46 slots, 43 cases, 172 original prompts and 516 saved predictions/vote sets independently verified."
    )


if __name__ == "__main__":
    with a.offline():
        main()
