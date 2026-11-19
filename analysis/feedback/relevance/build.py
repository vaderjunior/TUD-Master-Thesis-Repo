from src.hsrag.paths import PROJECT_ROOT
import ast
import hashlib
import json
from pathlib import Path

ROOT = PROJECT_ROOT
BASE = ROOT / "experiments/astra/sq3_full_kb_relevance_20260930_v1/api_execution_v1"


def function_source(source, name):
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return "\n".join(source.splitlines()[node.lineno - 1 : node.end_lineno])


def engine_source(old):
    middle = old[old.index("def identity(") : old.index("def run_stage(")].strip()
    middle = middle.replace(
        "Stop here; Astra reviews saved results before the next named stage.",
        "Automatic offline audit follows; advance only after verified success.",
    )
    middle = middle.replace(
        "Tell Astra to inspect. No next stage started.",
        "Automatic offline audit follows before any next batch.",
    )
    return (
        middle
        + "\n\n\n"
        + function_source(old, "run_sequence")
        + "\n\n\n"
        + function_source(old, "endpoint_progress")
    )


def auditor_source(old):
    return (
        old.replace("astra_sq3_feedback_updates_api_v1", "astra_sq3_relevance_api_v1")
        .replace(
            "astra_sq3_feedback_updates_stage_check_v1",
            "astra_sq3_relevance_stage_check_v1",
        )
        .replace(
            "SQ3 full-KB feedback-update Qwen stage",
            "SQ3 full-KB correction-relevance Qwen stage",
        )
        .replace(
            "sq3_feedback_updates_{stage_id}_audit_", "sq3_relevance_{stage_id}_audit_"
        )
        .replace(
            "completed_updates_stage_verified", "completed_relevance_stage_verified"
        )
        .replace(
            "offline_checked_ready_for_user_updates",
            "offline_checked_ready_for_user_relevance",
        )
    )


def main():
    names = [
        "analysis/feedback/source/updates.py",
        "analysis/feedback/source/update_audit.py",
        "tests/feedback/source/test_update_audit.py",
    ]
    old = {n: (ROOT / n).read_text(encoding="utf8") for n in names}
    hashes = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}
    api_path = ROOT / "analysis/feedback/relevance/run.py"
    template = api_path.read_text(encoding="utf8")
    marker = "# INHERITED_RELEVANCE_ENGINE"
    assert template.count(marker) == 1, "Preserve generated API source"
    api = template.replace(marker, engine_source(old[names[0]]))
    audit = auditor_source(old[names[1]])
    tests = old[names[2]].replace(
        "astra_sq3_feedback_updates_stage_check_v1",
        "astra_sq3_relevance_stage_check_v1",
    )
    begin = tests.index("    def test_baseline_study_binding(")
    end = tests.index("    def setUp(", begin)
    tests = tests[:begin] + """    def test_study_binding(self):
        self.assertEqual(c.BASE.name, 'api_execution_v1')
        self.assertEqual(c.b.STUDY.name, 'sq3_full_kb_relevance_20260930_v1')
        self.assertEqual(c.a.digest(c.b.SCIENCE), c.b.STUDY_PLAN_SHA)
        science=c.a.read(c.b.SCIENCE)
        self.assertEqual(len(science['rosters']['evaluation']),350)
        self.assertEqual(len(science['rosters']['primary_full_cover']),343)

    def test_auditor_inheritance_is_exact_declared_transform(self):
        from .astra_sq3_relevance_build_v1 import auditor_source
        old=(c.ROOT/'analysis/feedback/source/update_audit.py').read_text(encoding='utf8')
        current=(c.ROOT/'analysis/feedback/relevance/audit.py').read_text(encoding='utf8')
        self.assertEqual(current,auditor_source(old))

""" + tests[end:]
    outputs = {
        "analysis/feedback/relevance/audit.py": audit,
        "tests/feedback/relevance/test_audit.py": tests,
    }
    for name, text in outputs.items():
        assert not (ROOT / name).exists(), "Preserve " + name
        ast.parse(text)
    ast.parse(api)
    dest = BASE / "development/source_build_receipt_v1.json"
    assert not dest.exists(), "Preserve build receipt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with (dest.parent / "api_before_engine_insertion.py.txt").open(
        "x", encoding="utf8"
    ) as f:
        f.write(template)
    api_path.write_text(api, encoding="utf8", newline="\n")
    for name, text in outputs.items():
        with (ROOT / name).open("x", encoding="utf8", newline="\n") as f:
            f.write(text)
    assert all(
        hashlib.sha256((ROOT / n).read_bytes()).hexdigest() == sha
        for n, sha in hashes.items()
    ), "Parent source changed"
    receipt = {
        "sources_unchanged": hashes,
        "engine": "Same identity, persistence, request, resume and batch sequence logic; only completion wording updated.",
        "auditor": "Only study/module, output and readiness/status bindings changed; operational checks unchanged.",
        "outputs": {
            n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest()
            for n in [*outputs, str(api_path.relative_to(ROOT)).replace("\\", "/")]
        },
    }
    with dest.open("x", encoding="utf8") as f:
        json.dump(receipt, f, indent=2)
    print(
        "Prepared inherited relevance execution/audit sources; original sources unchanged."
    )


if __name__ == "__main__":
    main()
