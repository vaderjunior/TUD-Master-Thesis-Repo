from pathlib import Path
from analysis.feedback.rounds import prepare_quality_earlier as original

a = original.a


def main():
    contract = a.read(a.OUT / "analysis_contract.json")
    for path, digest in {
        **contract["input_bindings"],
        **contract["source_hashes"],
    }.items():
        a.need(a.sha(a.ROOT / path) == digest, "Frozen input changed")
    a.need(
        set(p.name for p in a.OUT.iterdir())
        == {"analysis_contract.json", "tests.json"},
        "Unexpected prior outputs; preserve and review",
    )
    a.need(a.read(a.OUT / "tests.json")["passed"], "Original fixtures did not pass")
    source = Path(original.__file__).read_text(encoding="utf8")
    before = "raw[{'gold_hate_types':'hate_types','gold_target_groups':'target_groups','gold_severity':'severity','gold_legal':'legal'}[field]]"
    after = "raw.get({'gold_hate_types':'hate_types','gold_target_groups':'target_groups','gold_severity':'severity','gold_legal':'legal'}[field])"
    a.need(source.count(before) == 1, "Optional-field patch is not unique")
    amended = source.replace(before, after)
    path = a.OUT / "assembly_amendment_v2.json"
    a.write(
        path,
        dict(
            status="optional_raw_gold_read_repaired_before_scoring",
            created_utc=a.pure.utc(),
            reason="v1 stopped on the first original roster row because legal is absent rather than explicitly null. The verified reader already treats absent optional fine gold as unknown via get().",
            change="Only the raw-roster comparison uses get() for optional fields. Features, labels, eligibility, masks, metrics, uncertainty and planned cases remain unchanged.",
            previous_contract_sha256=a.sha(a.OUT / "analysis_contract.json"),
            previous_tests_sha256=a.sha(a.OUT / "tests.json"),
            failed_source_sha256=a.sha(Path(original.__file__)),
            recovery_source_sha256=a.sha(Path(__file__)),
            completed_predictions_recomputed=False,
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
            earlier_evidence_changed=False,
        ),
    )

    def reuse_freeze(matrix):
        a.need(len(matrix) == 2077, "Matrix size changed")
        result = dict(contract)
        result["input_bindings"] = {
            **contract["input_bindings"],
            a.rel(path): a.sha(path),
        }
        result["source_hashes"] = {
            **contract["source_hashes"],
            a.rel(Path(__file__)): a.sha(Path(__file__)),
        }
        return result

    namespace = dict(vars(original), __name__="offline_part6_repaired_builder")
    exec(
        compile(
            amended, str(Path(original.__file__)) + "[optional-field repair]", "exec"
        ),
        namespace,
    )
    namespace["freeze"] = reuse_freeze
    namespace["main"]()


if __name__ == "__main__":
    with a.pure.offline():
        main()
