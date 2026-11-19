import math
from pathlib import Path
import numpy as np
from analysis.feedback.human import feedback_common as c
from analysis.common import prompt as renderer
from analysis.common import feedback_inputs as tokens


def check():
    if (c.BASE / "complete.json").exists():
        done = c.read(c.BASE / "complete.json")
        c.verify(c.read(c.BASE / "preparation_plan.json")["inputs"])
        c.verify(c.read(c.BASE / "prepared.json")["outputs"])
        c.require(
            done["prepared_sha256"] == c.sha(c.BASE / "prepared.json")
            and done["verification_sha256"] == c.sha(c.ROOT / done["verification"]),
            "Independent receipt changed",
        )
        c.require(
            c.read(c.ROOT / done["verification"])["checker_sha256"] == c.sha(__file__),
            "Checker changed after verification",
        )
        return c.read(c.ROOT / done["verification"])
    guard = c.active_preservation()
    c.verify(c.read(c.BASE / "preparation_plan.json")["inputs"])
    c.verify(c.read(c.BASE / "prepared.json")["outputs"])
    roster = list(c.rows(c.BASE / "roster.jsonl"))
    by_alias = {r["packet_id"]: r for r in roster}
    decoded = {}
    for name, who in (("Reviewer_A_Part_2", "A"), ("Reviewer_B_Part_1", "B")):
        inspection = c.read(c.HUMAN / (name + "_inspection.json"))
        form, export = (
            c.read(inspection["form_path"]),
            c.read(inspection["export_path"]),
        )
        blocks = {x["uuid"]: x for x in form["blocks"]}
        questions = {x["id"]: x for x in export["questions"]}
        c.require(len(export["submissions"]) == 1, "Unexpected duplicate human return")
        answers = {}
        for response in export["submissions"][0]["responses"]:
            title = questions[response["questionId"]]["title"]
            answer = response["answer"]
            c.require(title not in answers, "Repeated question answer")
            answers[title] = (
                [blocks[uid]["payload"]["text"] for uid in answer]
                if isinstance(answer, list)
                else answer
            )
        for alias, r in by_alias.items():
            gate = answers[alias + " · Is this hateful?"]
            c.require(len(gate) == 1, "Missing single human gate")
            hate = True if gate == ["Yes"] else False if gate == ["No"] else None
            types = (
                sorted(
                    answers.get(
                        alias + " · Hate types (select all justified types)", []
                    )
                    or []
                )
                if hate is True
                else []
            )
            state = (
                ("assigned" if types else "unassigned")
                if hate is True
                else "not_applicable"
            )
            j = r["judgments"][who + "_project"]
            c.require(
                j["gate"] == gate[0]
                and j["binary_hate"] == hate
                and (j["types"] == types)
                and (j["type_state"] == state),
                "Raw project answer differs from joined target",
            )
            decoded[alias, who] = (
                hate,
                types if state == "assigned" else [] if hate is False else None,
            )
    expected = {}
    for alias, r in by_alias.items():
        a, b = (decoded[alias, "A"], decoded[alias, "B"])
        old = r["model_repeat_0"]
        c.require(
            c.sha(c.ROOT / r["model_source"]) == r["model_source_sha256"],
            "Historical prediction changed",
        )
        if a[0] is None or a[0] != b[0]:
            continue
        t = a[1] if a[1] is not None and a[1] == b[1] else None
        if a[0] != old["hate"] or (t is not None and set(t) != set(old["hate_type"])):
            expected[alias] = (a[0], t)
    c.require(
        tuple(sorted(expected)) == c.SOURCE_ALIASES,
        "Independent correction selection disagrees",
    )
    correction_rows = c.read(c.BASE / "corrections.json")
    for x in correction_rows:
        gate, types = expected[x["packet_id"]]
        c.require(
            x["human"]["meta"]
            == {
                "gate": gate,
                "hate_types": types,
                "target_groups": None,
                "severity": None,
                "legal": None,
            },
            "Human feedback content differs",
        )
        dataset = by_alias[x["packet_id"]]["gold"]
        c.require(
            x["dataset"]["meta"]
            == {
                "gate": dataset["hate"],
                "hate_types": (
                    sorted(dataset["hate_type"]) if types is not None else None
                ),
                "target_groups": None,
                "severity": None,
                "legal": None,
            },
            "Matched annotation availability differs",
        )
    by_id = {r["id"]: r for r in roster}
    dev = list(c.rows(c.CORE / "dev_inputs.jsonl"))
    indices = {r["key"]: i for i, r in enumerate(dev)}
    vv = np.load(
        c.CORE / "retrieval_v1/query_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    vectors = {
        r["packet_id"]: [float(v) for v in vv[indices[r["key"]]]] for r in roster
    }
    norms = {k: math.sqrt(math.fsum((x * x for x in v))) for k, v in vectors.items()}
    choices = {r["key"]: r for r in c.rows(c.BASE / "choices.jsonl")}
    maxdiff = 0.0
    for r in roster:
        alias = r["packet_id"]
        v = vectors[alias]
        scores = {
            s: math.fsum((x * y for x, y in zip(v, vectors[s])))
            / (norms[alias] * norms[s])
            for s in c.SOURCE_ALIASES
        }
        order = sorted(scores, key=lambda s: (-scores[s], s))
        actual = choices[r["key"]]
        c.require(
            order == [v["alias"] for v in actual["ranking"]]
            and actual["selected_alias"] == order[0] == r["selected_correction"],
            "Independent cosine rank differs",
        )
        for v in actual["ranking"]:
            maxdiff = max(maxdiff, abs(v["cosine"] - scores[v["alias"]]))
    c.require(maxdiff < 1e-12, "Independent cosine values differ")
    component = {n: d for d in c.read(c.SCOPE / "components.json") for n in d["nodes"]}
    source_clusters = {
        component["development|en/" + by_alias[a]["id"]]["cluster"] for a in expected
    }
    for r in roster:
        comp = component["development|" + r["key"]]
        c.require(
            r["cluster"] == comp["cluster"]
            and r["transfer_eligible"]
            == (
                r["packet_id"] not in expected
                and comp["cluster"] not in source_clusters
            ),
            "Source/transfer leakage mask differs",
        )
        c.require(
            r["touches_training"]
            == ("component_touches_TRAIN" in comp["exclusion_reasons"]),
            "TRAIN overlap flag differs",
        )
    for r in roster:
        if r["transfer_eligible"]:
            c.require(
                all(
                    (
                        r["id"] != by_alias[s]["id"]
                        and c.norm(r["text"]) != c.norm(by_alias[s]["text"])
                        for s in expected
                    )
                ),
                "Direct source duplicate in transfer",
            )
    base = {}
    for p in c.rows(c.A15 / "prompts.jsonl"):
        if p["condition"] == "reference5" and p["item_id"] in by_id:
            base[p["key"]] = p
    cat = {(p["key"], p["condition"]): p for p in c.rows(c.BASE / "prompts.jsonl")}
    c.require(
        len(cat) == 72
        and set(cat) == {(r["key"], a) for r in roster for a in c.CONDITIONS},
        "Prompt matrix differs",
    )
    tok = tokens.tokenizer()
    correction_map = {x["packet_id"]: x for x in correction_rows}
    identical = 0
    for r in roster:
        original = base[r["key"]]
        a, b = (cat[r["key"], arm] for arm in ("dataset_feedback5", "human_feedback5"))
        c.require(
            (
                a["messages"] == b["messages"]
                if r["selected_correction"] == "R059"
                else a["messages"] != b["messages"]
            ),
            "Matched-arm identity differs",
        )
        identical += a["messages"] == b["messages"]
        for arm in c.CONDITIONS:
            p = cat[r["key"], arm]
            ctx = p["context"]
            c.require(
                ctx["definitions"] == original["context"]["definitions"]
                and ctx["guidelines"] == original["context"]["guidelines"],
                "Abstract knowledge changed",
            )
            group = ctx["example_groups"]
            old = original["context"]["example_groups"]
            c.require(
                len(group) == 1
                and group[0][0] == "Labelled examples"
                and (len(group[0][1]) == 5),
                "Five-slot policy differs",
            )
            if arm == "reference5":
                c.require(
                    ctx == original["context"]
                    and p["messages"] == original["messages"],
                    "Reference not exact",
                )
            else:
                correction = correction_map[r["selected_correction"]][
                    "human" if arm == "human_feedback5" else "dataset"
                ]
                c.require(
                    group[0][1][:4] == old[0][1][:4] and group[0][1][4] == correction,
                    "Source/label/position intervention differs",
                )
                c.require(
                    all(
                        (
                            c.norm(x["text"]) != c.norm(correction["text"])
                            for x in old[0][1][:4]
                        )
                    ),
                    "Correction duplicates ordinary neighbour",
                )
            c.require(
                renderer.render_messages(r["text"], "en", ctx, "corrected")
                == p["messages"],
                "Prompt not reconstructed",
            )
            c.require(
                p["context_sha256"] == c.sig(ctx)
                and p["messages_sha256"] == c.sig(p["messages"]),
                "Prompt identity differs",
            )
            c.require(
                p["reference_input_tokens"] == tokens.token_count(tok, p["messages"]),
                "Token count differs",
            )
    c.require(identical == 15, "Identical-message cases changed")
    c.require(c.active_preservation() == guard, "Current queue changed")
    result = {
        "status": "passed",
        "created_utc": c.utc(),
        "checker_sha256": c.sha(__file__),
        "project_cards_redecoded": 48,
        "independent_sources": sorted(expected),
        "prompts_reconstructed": len(cat),
        "independent_cosine_max_difference": maxdiff,
        "source_disjoint_other_texts": sum((r["transfer_eligible"] for r in roster)),
        "identical_feedback_prompt_pairs": identical,
        "active_queue_preservation": guard,
        "actual_API_calls": 0,
        "new_human_judgments": 0,
    }
    path = c.BASE / "independent_verification.json"
    c.save(path, result)
    c.save(
        c.BASE / "complete.json",
        {
            "status": "offline_preparation_independently_verified",
            "prepared_sha256": c.sha(c.BASE / "prepared.json"),
            "verification": c.rel(path),
            "verification_sha256": c.sha(path),
            "new_model_calls": 0,
        },
    )
    return result


if __name__ == "__main__":
    with c.offline():
        print(check())
