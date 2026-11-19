from analysis.classification.matched import common as m
from collections import Counter, defaultdict
from pathlib import Path


def run():
    with m.runtime.offline():
        groups, outputs, meta = m.load()
        phase = {r["id"]: p for p, rows in groups.items() for r in rows}
        totals = {
            k: Counter() for k in ["finish_reasons", "reported_models", "http_statuses"]
        }
        batches = []
        audit_bindings = {}
        root = m.e.queue("matched").BASE / "live/matched/stages"
        for stage in sorted(root.iterdir()):
            if not stage.is_dir():
                continue
            review = m.a.read(stage / "review.json")
            path = stage / review["report"]
            report = m.a.read(path)
            m.a.require(
                review["artifacts"][review["report"]] == m.c.sha(path),
                "Audit report changed",
            )
            for k in totals:
                totals[k].update(report[k])
            batches.append(
                dict(
                    stage=stage.name,
                    attempts=report["attempt_count"],
                    availability_checks=report["model_availability_checks"],
                    audit_API_calls=report["actual_audit_API_calls"],
                )
            )
            audit_bindings.update(m.bindings([path, stage / "review.json"]))
        m.a.require(
            dict(totals["finish_reasons"]) == {"stop": 118018},
            "Unexpected termination reason",
        )
        m.a.require(
            dict(totals["reported_models"]) == {"Qwen/Qwen3-VL-32B-Instruct": 118018},
            "Unexpected reported model",
        )
        same = defaultdict(Counter)
        for row in m.runtime.stream_rows(m.c.PREP / "matched_prompts.jsonl"):
            if (
                not row["condition"].startswith("permuted_")
                or row["changed_annotation_slots"] != 0
            ):
                continue
            group = same[phase[row["key"]]]
            group["prompt_pairs"] += 1
            for rep in range(3):
                l = outputs["retrieved_complete", rep][row["key"]]
                r = outputs[row["condition"], rep][row["key"]]
                group["paired_repeat_comparisons"] += 1
                if l is None or r is None:
                    group["unavailable"] += 1
                    continue
                group["valid_pairs"] += 1
                group["complete_prediction_disagreements"] += l != r
                dim = m.scoring.MATCHED_DIMS[phase[row["key"]]]
                group["focal_endpoint_disagreements"] += l[dim] != r[dim]
        m.a.require(
            sum(v["prompt_pairs"] for v in same.values()) == 434,
            "Identical-prompt roster differs",
        )
        m.save(
            m.BASE / "audit_and_identical_prompts.json",
            dict(
                audit_metadata={k: dict(v) for k, v in totals.items()},
                batches=batches,
                identical_message_comparisons=dict(same),
                scope="Same-numbered outer repeats are illustrative paired comparisons, not shared random seeds or independent experimental units. Identical prompt differences are generation variation, not effects of label shuffling.",
                inputs=audit_bindings,
                sources=m.bindings([Path(__file__)]),
            ),
        )
        old_path = (
            m.ROOT
            / "experiments/astra/sq3_iterative_review_20261026_v1/part8/partial_programme_ledger.json"
        )
        old = m.a.read(old_path)
        resources = m.a.read(m.BASE / "resources.json")
        account = resources["accounting"]
        new = dict(
            programme="SQ2 matched example information",
            ordinary_slots=account["ordinary_logical_calls"],
            repair_slots=account["repair_logical_calls"],
            retry_or_recovered_failure_attempts=account["retry_attempts"],
            attempt_slots=account["post_attempts"],
            successful_responses=account["http_success_with_text"],
            reported_tokens=resources["observed_totals"]["total_tokens"],
            source=m.runtime.rel(m.BASE / "resources.json"),
            source_sha256=m.c.sha(m.BASE / "resources.json"),
        )
        fields = list(old["totals"])
        sums = {k: sum(x[k] for x in old["programmes"]) for k in fields}
        m.a.require(sums == old["totals"], "Prior programme ledger does not reconcile")
        programmes = old["programmes"] + [new]
        total = {k: sum(x[k] for x in programmes) for k in fields}
        m.save(
            m.BASE / "partial_programme_ledger.json",
            dict(
                status="fourteen_programme_partial_ledger",
                programmes=programmes,
                totals=total,
                previous_ledger=m.runtime.rel(old_path),
                previous_ledger_sha256=m.c.sha(old_path),
                inherited_thirteen_programme_objects_unchanged=programmes[:-1]
                == old["programmes"],
                scope="Partial remote programme ledger; not whole-thesis or SQ3-only usage, bill, energy, GPU time or author effort. Existing eight administrative unknown outcomes remain unknown. Two earlier SQ3 programmes and capacity diagnostics retain their separately reported boundaries. No local classifier, benchmark or current server-training time is folded into remote tokens.",
                actual_Qwen_calls_during_this_review=0,
                training_fits_during_this_review=0,
            ),
        )
        print("Audit metadata:", {k: dict(v) for k, v in totals.items()}, flush=True)
        print("Partial ledger:", total, flush=True)
        print("Identical messages:", dict(same), flush=True)


if __name__ == "__main__":
    run()
