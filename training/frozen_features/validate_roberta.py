from pathlib import Path
from unittest.mock import patch
from training.frozen_features import roberta as r
from analysis.common import study as c
from analysis.common.api import save_new


def json_info(value):
    if isinstance(value, dict):
        return {key: json_info(v) for key, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return [json_info(v) for v in sorted(value, key=repr)]
    if isinstance(value, (tuple, list)):
        return [json_info(v) for v in value]
    return value


def main():
    r.validate()
    plan = r.BASE / "preflight_execution_amendment_v2.json"
    if not plan.exists():
        save_new(
            plan,
            {
                "created_utc": c.utc(),
                "status": "metadata_only_preflight_amendment",
                "original_plan_sha256": c.sha(r.BASE / "plan.json"),
                "source_sha256": {
                    c.rel(Path(__file__)): c.sha(Path(__file__)),
                    "tests/training/frozen_features/test_validate_roberta.py": c.sha(
                        c.ROOT
                        / "tests/training/frozen_features/test_validate_roberta.py"
                    ),
                },
                "reason": "Transformers returned sets in loading_info; v1 raised TypeError while serializing the preflight receipt after numerical checks and probe arrays. No completed preflight receipt was written.",
                "change": "Convert metadata sets to deterministically ordered JSON lists. Same source, cached weights, tokenizer, FP32 pooling, probes, masks, tolerance and output arrays. Recompute/check rather than fabricate a successful preflight.",
            },
        )
    amendment = c.read(plan)
    c.verify(amendment["source_sha256"])
    c.require(
        amendment["original_plan_sha256"] == c.sha(r.BASE / "plan.json"),
        "Scientific plan changed",
    )
    original = r.load

    def load(device, config):
        tok, model, info = original(device, config)
        return (tok, model, json_info(info))

    with patch.object(r, "load", load):
        r.preflight()
    save_new(
        r.BASE / "preflight_amendment_complete_v2.json",
        {
            "status": "metadata_amended_preflight_complete",
            "amendment_sha256": c.sha(plan),
            "preflight_sha256": c.sha(r.BASE / "preflight.json"),
        },
    )


if __name__ == "__main__":
    with c.offline():
        main()
