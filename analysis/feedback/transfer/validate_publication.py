from pathlib import Path
import json
import re
import numpy as np
from analysis.feedback.transfer import archive as a

OUT = a.ROOT / "experiments/astra/sq3_transfer_part6_20261023_v1"
STATES = ("unchanged_error", "fix", "regression", "unchanged_correct")


def pct(x):
    return "undefined" if x is None else f"{100*x:.2f}%"


def pp(x):
    return "undefined" if x is None else f"{100*x:+.2f}"


def section(text, start, end):
    return text.split(start, 1)[1].split(end, 1)[0]


def tablerows(name):
    lines = (OUT / "tables" / name).read_text(encoding="utf8").splitlines()
    return [
        [c.strip() for c in line.strip("|").split("|")]
        for line in lines
        if line.startswith("| ")
    ][1:]


def checkrows(name, expected):
    actual = tablerows(name)
    a.need(
        actual == [[str(v) for v in row] for row in expected],
        "Published table mismatch " + name,
    )
    return len(actual)


def main():
    dest = OUT / "publication/final_check.json"
    a.need(not dest.exists(), "Preserve prior publication check")
    complete = a.read(OUT / "complete.json")
    bound = {}
    for group in (
        "input_bindings",
        "analysis_sources",
        "artifacts",
        "navigation_at_publication",
    ):
        for name, digest in complete[group].items():
            a.need(
                name not in bound or bound[name] == digest,
                "Conflicting binding " + name,
            )
            bound[name] = digest
    for name, digest in bound.items():
        a.need(a.sha(a.ROOT / name) == digest, "Changed bound evidence " + name)
    a.need(
        complete["completed_transfer_parts"] == [1, 2, 3, 4, 5, 6]
        and complete["remaining_transfer_parts"] == [7, 8, 9, 10]
        and not complete["whole_study_closed"],
        "Only Part 6 closes",
    )
    a.need(
        all(
            complete[n] == 0
            for n in (
                "new_Qwen_calls",
                "new_fits",
                "new_embeddings",
                "new_bootstrap_sets",
            )
        ),
        "Offline scope",
    )
    a.need(
        complete["experiment_inventory"] == dict(complete=3, open=6, retained_rows=16),
        "Inventory preserved",
    )
    verified = a.read(OUT / "verified.json")
    cases = a.read(OUT / "cases_verified.json")
    tests = a.read(OUT / "focused_tests.json")
    a.need(tests["passed"] and tests["tests_run"] == 14, "Fourteen focused tests")
    a.need(
        verified["calculated_sha256"] == a.sha(OUT / "calculated.json")
        and cases["numerical_verification_sha256"] == a.sha(OUT / "verified.json"),
        "Independent checks current",
    )
    a.need(
        cases["unique_cases"] == 43
        and cases["empty_slots"] == 2
        and cases["original_prompts"] == 172
        and cases["original_predictions"] == 516
        and cases["original_vote_slots"] == 1548,
        "Case evidence coverage",
    )
    links = 0
    for path in [
        a.ROOT / complete["report"],
        a.ROOT / complete["roadmap"],
        OUT / "case_index.md",
        *sorted((OUT / "cases").glob("*.md")),
        *sorted((OUT / "tables").glob("*.md")),
    ]:
        text = path.read_text(encoding="utf8")
        a.need("\ufffd" not in text, "Invalid replacement character " + path.name)
        prose = re.sub(r"```.*?```", "", text, flags=re.S)
        for link in re.findall(r"\]\(([^)]+)\)", prose):
            if link.startswith(("https:", "http:", "#")):
                continue
            target = (path.parent / link.strip("<>").split("#", 1)[0]).resolve()

            a.need(
                target.exists() or target == dest.resolve(),
                "Broken local link " + str(target),
            )
            links += 1
    for name in ("astra_session", "astra_review"):
        old = (OUT / f"publication/{name}_before.md").read_bytes()
        now = (a.ROOT / f"Thesis_Log/{name}.md").read_bytes()
        at = old.index(b"\n") + 1
        added = len(now) - len(old)
        a.need(
            added > 0 and old[:at] == now[:at] and old[at:] == now[at + added :],
            "Prior navigation bytes changed",
        )
        entry = now[at : at + added]
        a.need(
            entry.count(b"ASTRA-TRANSFER-PART6-20261023-v1") == 1 and b"v81" in entry,
            "Current entry",
        )
    old = (OUT / "publication/previous_roadmap_v80.md").read_bytes()
    a.need(
        old
        == (
            a.ROOT
            / "Thesis_Log/astra_result/astra_remaining_work_before_writing_20261023_v80.md"
        ).read_bytes(),
        "Old roadmap altered",
    )
    new = (a.ROOT / complete["roadmap"]).read_bytes()
    reserved = lambda b: b.split(
        b"## Ten-part deep review of the completed reserved-correction study", 1
    )[1].split(b"## Other useful offline tasks", 1)[0]
    a.need(reserved(old) == reserved(new), "Earlier reserved review byte preservation")
    road = new.decode("utf8")
    experiments = section(
        road, "## The nine experimental work items", "## Broader-transfer review"
    )
    a.need(
        experiments.count("- [x]") == 3 and experiments.count("- [ ]") == 6,
        "Nine experiment checkboxes",
    )
    a.need(
        road.count("- [x] **Transfer Part") == 6
        and road.count("- [ ] **Transfer Part") == 4,
        "Ten transfer review parts",
    )
    a.need(
        sum(
            line.startswith("| ")
            for line in section(
                road,
                "## Full accepted inventory retained",
                "## Run order and preservation",
            ).splitlines()
        )
        == 17,
        "Sixteen inventory rows",
    )
    s = a.read(OUT / "transition_summary.json")
    r = a.read(OUT / "repeat_variation.json")
    shared = a.read(OUT / "same_message_variation.json")
    edits = a.read(OUT / "fine_label_edits.json")
    flat = a.read(OUT / "tables/transition_metrics.json")
    a.need(
        len(flat) == len(s) == 160 and len({q["id"] for q in flat}) == 160,
        "Flat transition coverage",
    )
    for row in flat:
        q = s[row["id"]]
        pat = q["text_patterns"]
        expected = dict(
            id=row["id"],
            **{
                n: q[n]
                for n in (
                    "head",
                    "texts",
                    "clusters",
                    "dependent_pairs",
                    "counts",
                    "reference_rate",
                    "feedback_rate",
                    "exact_difference",
                    "error_difference",
                )
            },
            directions=pat["direction"],
            stability=pat["stability"],
            any_fix_texts=pat["any_aligned_fix"],
            any_regression_texts=pat["any_aligned_regression"],
            offsetting_texts=pat["offsetting_aligned_changes"],
            opposing_pool_texts=pat["opposing_pool_directions"],
        )
        a.need(row == expected, "Flat transition export " + row["id"])
    counts = {}
    rows = []
    patterns = []
    align = []
    for name, q in s.items():
        c = q["counts"]
        pat = q["text_patterns"]
        d = pat["direction"]
        st = pat["stability"]
        al = q["repeat_alignments"]
        ac = al["counts"]
        cross = al["all_cross_repeat_counts"]
        rows.append(
            [
                name,
                q["texts"],
                q["dependent_pairs"],
                c["fix"],
                c["regression"],
                c["unchanged_correct"],
                c["unchanged_error"],
                pp(q["exact_difference"]),
            ]
        )
        patterns.append(
            [
                name,
                q["texts"],
                d["improved"],
                d["regressed"],
                d["equal_average"],
                st["consistent_fix"],
                st["consistent_regression"],
                st["persistently_correct"],
                st["persistently_wrong"],
                st["variable"],
                pat["offsetting_aligned_changes"],
                pat["opposing_pool_directions"],
            ]
        )
        align.append(
            [
                name,
                min(x[1] for x in ac),
                max(x[1] for x in ac),
                min(x[2] for x in ac),
                max(x[2] for x in ac),
                cross[1],
                cross[2],
                al["all_cross_repeat_pairs"],
                pp(q["exact_difference"]),
            ]
        )
        a.need(
            sum(c.values()) == q["texts"] * 9 == q["dependent_pairs"],
            "Dependent-pair denominator",
        )
        a.need(
            all(v[1] - v[2] == c["fix"] - c["regression"] for v in ac),
            "Repeat-alignment net invariant",
        )
    counts["all_transitions"] = checkrows("all_transitions.md", rows)
    counts["per_text_patterns"] = checkrows("per_text_patterns.md", patterns)
    counts["repeat_alignment"] = checkrows("repeat_alignment.md", align)
    rows = []
    fields = []
    for name, q in r.items():
        n = q["texts"]
        nv = q["three_vote_outputs"]
        h = q["aggregate_repeat_distinct_histogram"]
        v = q["three_vote_distinct_histogram"]
        rows.append(
            [
                name,
                n,
                pct(h[0][0] / n),
                pct(h[0][5] / n),
                nv,
                pct(v[0][0] / nv) if nv else "undefined",
                pct(v[0][5] / nv) if nv else "undefined",
                q["aggregate_not_observed_three_votes"],
                q["gate_ties"],
            ]
        )
        for j, field in enumerate(q["fields"]):
            fields.append(
                [
                    name,
                    field,
                    n,
                    *[h[z][j] for z in range(3)],
                    q["aggregate_repeat_disagree_pairs"][j],
                    q["aggregate_repeat_available_pairs"],
                    nv,
                    *[v[z][j] for z in range(3)],
                ]
            )
    counts["repeat_stability"] = checkrows("repeat_stability.md", rows)
    counts["field_variation"] = checkrows("field_variation.md", fields)
    rows = []
    for name, q in shared.items():
        n = q["aligned_output_pairs"]
        rows.append(
            [
                name,
                q["query_pool_pairs"],
                q["unique_texts"],
                n,
                q["aligned_disagreements"][0],
                pct(q["aligned_disagreements"][0] / n),
                q["aligned_disagreements"][5],
                pct(q["aligned_disagreements"][5] / n),
            ]
        )
    counts["same_messages"] = checkrows("same_message_outputs.md", rows)
    rows = []
    for name, partitions in edits.items():
        for part, q in partitions.items():
            for label, c in q["labels"].items():
                rows.append(
                    [
                        name,
                        part,
                        label,
                        q["pairs"],
                        c["correct_added"],
                        c["correct_removed"],
                        c["extra_added"],
                        c["extra_removed"],
                    ]
                )
    counts["fine_label_changes"] = checkrows("fine_label_changes.md", rows)
    rows = []
    for pop, title in [("en_test", "English TEST"), ("en_hatecheck_eval", "HateCheck")]:
        mat = s[f"population/{pop}/known_answer/main"][
            "binary_vs_known_exact_transition_matrix"
        ]
        for b in range(4):
            for j in range(4):
                rows.append([title, STATES[b], STATES[j], mat[b][j]])
    counts["binary_and_known_answer"] = checkrows("binary_and_known_answer.md", rows)
    a.need(
        counts
        == dict(
            all_transitions=160,
            per_text_patterns=160,
            repeat_alignment=160,
            repeat_stability=144,
            field_variation=864,
            same_messages=8,
            fine_label_changes=96,
            binary_and_known_answer=32,
        ),
        "Complete readable table coverage",
    )

    plotted = a.read(OUT / "figures_v2/plot_values.json")
    a.need(
        set(plotted) == {"text_outcomes", "repeat_stability", "case_repeat_matrix"},
        "Current three figures",
    )
    for row in plotted["text_outcomes"]:
        q = s[row["id"]]
        a.need(
            row
            == dict(
                id=row["id"],
                texts=q["texts"],
                direction=q["text_patterns"]["direction"],
            ),
            "Outcome plot value",
        )
    for row in plotted["repeat_stability"]:
        q = r[row["id"]]
        h = q["aggregate_repeat_distinct_histogram"]
        a.need(
            row
            == dict(
                id=row["id"], texts=q["texts"], stable_gate=h[0][0], stable_full=h[0][5]
            ),
            "Repeat plot value",
        )
    with np.load(OUT / "arrays.npz", allow_pickle=False) as f:
        scores = f["scores"]
    heads = a.read(OUT / "array_schema.json")["heads"]
    for row in plotted["case_repeat_matrix"]:
        case = a.read(OUT / "cases" / (row["case_id"] + ".json"))
        vals = (
            scores[case["input_index"], ..., heads.index(row["head"])]
            .reshape(12)
            .tolist()
        )
        a.need(
            row["key"] == case["key"] and row["values"] == vals and min(vals) >= 0,
            "Case plot value",
        )
    a.need(
        [
            len(plotted[k])
            for k in ("text_outcomes", "repeat_stability", "case_repeat_matrix")
        ]
        == [4, 8, 12],
        "Plot observation coverage",
    )
    visual = a.read(OUT / "figures_v2/visual_review.json")
    a.need(
        visual["status"] == "three_figures_visually_reviewed"
        and len(visual["images"]) == 3
        and visual["rendered_sha256"] == a.sha(OUT / "figures_v2/rendered.json"),
        "Figure visual review receipt",
    )

    case_claims = {
        4: {"binary": (3, [0, 0, 0])},
        5: {"binary": (0, [3, 3, 3])},
        14: {"binary": (0, [3, 3, 3]), "target_group": (0, [3, 3, 3])},
        15: {"binary": (0, [0, 0, 0])},
        17: {
            "binary": (0, [3, 3, 3]),
            "target_group": (0, [0, 0, 0]),
            "known_answer": (0, [0, 0, 0]),
        },
        21: {"binary": (2, [3, 3, 0])},
        23: {"binary": (0, [2, 3, 0]), "hate_type": (0, [2, 3, 0])},
        26: {"binary": (3, [3, 0, 3]), "hate_type": (3, [0, 0, 0])},
        33: {"binary": (2, [3, 2, 1])},
        35: {
            "binary": (3, [3, 3, 3]),
            "target_group": (3, [3, 3, 3]),
            "severity": (0, [3, 3, 3]),
        },
        40: {"binary": (3, [3, 3, 3]), "severity": (3, [0, 0, 0])},
    }
    for n, claims in case_claims.items():
        case = a.read(OUT / "cases" / f"case_{n:02}.json")
        for head, (ref, pools) in claims.items():
            profile = case["profiles"][head]
            a.need(
                (profile["reference_correct"], profile["pool_correct"]) == (ref, pools),
                "Narrative case count " + str(n) + " " + head,
            )
    en = s["population/en_test/binary/main"]
    joint = s["population/en_test/known_answer/main"]
    miss = s["endpoint/en_test_FNR/main"]
    a.need(
        en["text_patterns"]["direction"]
        == dict(improved=929, regressed=1470, equal_average=9485),
        "Main text claim",
    )
    a.need(
        joint["binary_vs_known_exact_transition_matrix"][1][0] == 845
        and miss["counts"]["fix"] == 1156
        and pct(845 / 1156) == "73.10%",
        "Missed-hate fix qualification",
    )
    report = (a.ROOT / complete["report"]).read_text(encoding="utf8")
    for phrase in (
        "nine **dependent** comparisons",
        "Repeat indices are not matched random seeds",
        "Equal average can include",
        "It is not all-schema validation",
        "Full agreement includes unannotated fields and is not accuracy.",
        "Do not subtract their disagreement rates",
        "English TEST was previously exposed",
        "Two slots are empty and remain empty",
        "no invented intervals",
        "Parts 7–10",
        "zero Qwen calls, fits, embeddings or new bootstrap sets",
    ):
        a.need(phrase in report, "Missing report qualification " + phrase)
    result = dict(
        status="part6_publication_preservation_tables_figures_cases_and_claims_pass",
        checked_utc=a.utc(),
        complete_sha256=a.sha(OUT / "complete.json"),
        script_sha256=a.sha(Path(__file__)),
        publication_check_amendment_sha256=a.sha(
            OUT / "publication/check_amendment_v2.json"
        ),
        checked_bound_files=len(bound),
        checked_local_links=links,
        checked_readable_rows=counts,
        flat_transition_rows=160,
        plotted_observations=24,
        plotted_case_cells=144,
        figures_visually_reviewed=3,
        checked_narrative_case_claims=sum(len(q) for q in case_claims.values()),
        unique_case_cards=43,
        prior_navigation_byte_preserved=True,
        earlier_reserved_review_byte_preserved=True,
        completed_transfer_parts=[1, 2, 3, 4, 5, 6],
        remaining_transfer_parts=[7, 8, 9, 10],
        whole_study_closed=False,
        new_Qwen_calls=0,
        new_fits=0,
        new_embeddings=0,
        new_bootstrap_sets=0,
    )
    with dest.open("xb") as f:
        f.write(a.encoded(result, pretty=True))
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    with a.offline():
        main()
