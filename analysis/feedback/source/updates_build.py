from src.hsrag.paths import PROJECT_ROOT
import ast
import hashlib
import json
from pathlib import Path

ROOT = PROJECT_ROOT
BASE = ROOT / "experiments/astra/sq3_full_kb_feedback_20260928_v1/update_api_v1"


def auditor_source(original):
    result = original.replace(
        "astra_sq3_feedback_api_v1", "astra_sq3_feedback_updates_api_v1"
    )
    result = result.replace(
        "astra_sq3_feedback_stage_check_v1", "astra_sq3_feedback_updates_stage_check_v1"
    )
    result = result.replace(
        "SQ3 full-KB baseline Qwen stage", "SQ3 full-KB feedback-update Qwen stage"
    )
    result = result.replace(
        "sq3_feedback_baseline_{stage_id}_audit_",
        "sq3_feedback_updates_{stage_id}_audit_",
    )
    result = result.replace(
        "baseline_verified_selection_review_pending", "completed_updates_stage_verified"
    )
    result = result.replace(
        "offline_checked_ready_for_user_stage_01",
        "offline_checked_ready_for_user_updates",
    )
    start = result.index("    endpoint_progress={}")
    end = result.index("    payload_keys=", start)
    result = (
        result[:start]
        + "    endpoint_progress=b.endpoint_progress(schedule,stage_index)\n\n"
        + result[end:]
    )
    return result


def main():
    names = [
        "analysis/feedback/source/run.py",
        "analysis/feedback/source/audit.py",
        "tests/feedback/source/test_audit.py",
    ]
    sources = {n: (ROOT / n).read_text(encoding="utf8") for n in names}
    hashes = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}
    api_path = ROOT / "analysis/feedback/source/updates.py"
    api = api_path.read_text(encoding="utf8")
    marker = "# INHERITED_STAGE_ENGINE"
    assert api.count(marker) == 1, "Preserve already built updates"
    old = sources[names[0]]
    engine = old[old.index("def identity(") : old.index("def run_stage(")].strip()
    api_new = api.replace(marker, engine)
    audit = auditor_source(sources[names[1]])
    tests = sources[names[2]].replace(
        "astra_sq3_feedback_stage_check_v1", "astra_sq3_feedback_updates_stage_check_v1"
    )
    tests = tests.replace(
        "self.assertEqual(c.BASE.name, 'baseline_api_v1')",
        "self.assertEqual(c.BASE.name, 'update_api_v1')",
    )
    tests = tests.replace(
        "astra_sq3_feedback_api_build_v1", "astra_sq3_feedback_updates_build_v1"
    )
    tests = tests.replace(
        "old = (c.ROOT/'analysis/knowledge/sampled_banks/validate_stage.py')",
        "old = (c.ROOT/'analysis/feedback/source/audit.py')",
    )
    outputs = {
        "analysis/feedback/source/update_audit.py": audit,
        "tests/feedback/source/test_update_audit.py": tests,
    }
    for name, source in outputs.items():
        assert not (ROOT / name).exists(), "Preserve " + name
        ast.parse(source)
    ast.parse(api_new)
    dest = BASE / "development/source_build_receipt_v1.json"
    assert not dest.exists(), "Preserve build receipt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with (dest.parent / "api_before_engine_insertion.py.txt").open(
        "x", encoding="utf8"
    ) as stream:
        stream.write(api)
    api_path.write_text(api_new, encoding="utf8", newline="\n")
    for name, source in outputs.items():
        with (ROOT / name).open("x", encoding="utf8", newline="\n") as stream:
            stream.write(source)
    for name, h in hashes.items():
        assert (
            hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == h
        ), "Original baseline source changed"
    with dest.open("x", encoding="utf8") as stream:
        json.dump(
            {
                "sources_unchanged": hashes,
                "engine_inheritance": "Exact identity-through-previous_review source from frozen baseline.",
                "auditor_changes": "Study/output/status/readiness bindings; endpoint progress counts completed predictions as well as text IDs because every update batch reuses all text IDs.",
                "initial_generated_sources": {
                    n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest()
                    for n in [*outputs, "analysis/feedback/source/updates.py"]
                },
            },
            stream,
            indent=2,
        )
    print("Prepared isolated update stage engine/auditor; baseline sources unchanged.")


if __name__ == "__main__":
    main()
