from collections import Counter
import io
import json
import socket
import threading
import time
import unittest
from unittest.mock import patch
from analysis.common import batch as b

a = b.a
ROOT = b.ROOT
BASE = b.BASE
OUT = (
    ROOT
    / "Thesis_Log/astra_result/evidence/core_api_execution_preparation_20260915_v1.json"
)


def forbidden(*args, **kwargs):
    raise AssertionError("No network in offline execution review")


def main():
    import pandas as pd
    import yaml

    a.require(
        not OUT.exists() and (not (BASE / "offline_check_complete.json").exists()),
        "Preserve completed review",
    )
    plan = b.validate()
    schedule = a.read(BASE / "schedule.json")
    scoring = a.read(BASE / "scoring_plan.json")
    cat = b.catalog()
    dev = b.rows(b.CORE / "dev_inputs.jsonl")
    jobs = [j for st in schedule["stages"] for j in st["jobs"]]
    tuples = [(j["key"], j["condition"], j["repeat"]) for j in jobs]
    a.require(len(jobs) == len(set(tuples)) == 32274, "Wrong unique job count")
    a.require(
        set(tuples) == {(key, c, r) for key, c in cat for r in range(3)},
        "Catalogue coverage differs",
    )
    membership = {r["key"]: r["scopes"][0]["subset"] for r in dev}
    locations = {}
    counts = Counter()
    for stage in schedule["stages"]:
        a.require(
            stage["ordinary_responses"] == len(stage["jobs"]) * 3 <= 8100,
            "Stage response budget differs",
        )
        for j in stage["jobs"]:
            a.require(j["subset"] == membership[j["key"]], "Wrong job endpoint")
            locations.setdefault(j["key"], set()).add(stage["id"])
            counts[j["subset"]] += 1
    a.require(
        all((len(v) == 1 for v in locations.values())),
        "Input comparison split across stages",
    )
    a.require(
        sum((x["ordinary_responses"] for x in schedule["stages"])) == 96822,
        "Whole-programme budget differs",
    )
    taxonomy = yaml.safe_load(
        (ROOT / "config/taxonomy.yaml").read_text(encoding="utf8")
    )
    fields = {
        "hate": "gate",
        "legal": "legal",
        "hate_type": "hate_types",
        "target_group": "target_groups",
        "severity": "severity",
    }
    endpoint_checks = {}
    for subset, endpoint in scoring["endpoints"].items():
        df = pd.read_parquet(ROOT / f"data/processed/{subset}.parquet")
        dim = endpoint["dimension"]
        ids = []
        support = Counter()
        for row in df.itertuples(index=False):
            gate = row.gate
            if gate is not None and pd.isna(gate):
                gate = None
            value = getattr(row, fields[dim], None)
            if hasattr(value, "tolist"):
                value = value.tolist()
            if isinstance(value, float) and pd.isna(value):
                value = None
            if dim not in ("hate", "legal") and gate is not True:
                continue
            if value is None:
                continue
            ids.append(str(row.id))
            labels = (
                ["hate" if value else "not_hate"]
                if dim == "hate"
                else [value] if dim == "severity" else value
            )
            support.update(set(labels))
        a.require(
            ids == endpoint["scorable_ids"], "Independent annotation mask differs"
        )
        fixed = (
            endpoint["all_labels"]
            if dim == "hate"
            else [x for x in taxonomy["dimensions"][dim]["labels"] if support[x] >= 10]
        )
        a.require(
            fixed == endpoint["macro_labels"]
            and {l: support[l] for l in endpoint["all_labels"]}
            == endpoint["gold_support"],
            "Independent fixed-label support differs",
        )
        endpoint_checks[subset] = {
            "rows": len(df),
            "scorable": len(ids),
            "macro_labels": fixed,
            "scheduled_predictions": counts[subset],
        }
    attempt = a.storage_path(BASE / f"offline_check/attempt_{time.time_ns()}")
    pilot = schedule["pilot_jobs"]
    counter = [0]
    mutex = threading.Lock()
    fixture = json.dumps(
        {
            "reasoning": "OFFLINE FIXTURE, not a model response",
            "hate": False,
            "target_group": [],
            "hate_type": [],
            "severity": None,
            "legal": ["insult_defamation"],
        }
    )

    def client():

        def send(payload):
            with mutex:
                counter[0] += 1
            return a.WireReply(
                200,
                json.dumps(
                    {
                        "model": a.MODEL,
                        "choices": [
                            {"finish_reason": "stop", "message": {"content": fixture}}
                        ],
                        "mock_only": True,
                    }
                ).encode(),
                {},
            )

        return send

    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ):
        suite = unittest.defaultTestLoader.loadTestsFromNames(
            ["tests.common.test_scoring", "tests.common.test_batch"]
        )
        output = io.StringIO()
        tests = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
        a.save_new(
            attempt / "tests.json",
            {
                "passed": tests.wasSuccessful(),
                "tests": tests.testsRun,
                "output": output.getvalue(),
            },
        )
        a.require(tests.wasSuccessful(), "Execution/scoring tests failed")
        first = b.run_jobs(
            pilot,
            cat,
            attempt / "mock",
            attempt / "first",
            client_factory=client,
            mode="mock",
        )
        before = counter[0]
        second = b.run_jobs(
            pilot,
            cat,
            attempt / "mock",
            attempt / "restart",
            client_factory=forbidden,
            mode="mock",
        )
        a.require(
            counter[0] == before == 45
            and first["saved_items"] == second["saved_items"] == 15
            and (second["submitted"] == 0),
            "Batch resume changed request count",
        )
    a.require(
        not (BASE / "live").exists(), "Offline checker unexpectedly created live output"
    )
    inputs = {
        **plan["input_hashes"],
        **plan["source_hashes"],
        **plan["generated_hashes"],
        b.rel(b.PLAN): a.digest(b.PLAN),
    }
    outputs = {
        b.rel(p): a.digest(p)
        for p in attempt.rglob("*")
        if p.is_file() and p.name != "item.lock"
    }
    for name, h in inputs.items():
        a.require(a.digest(ROOT / name) == h, "Input changed during offline check")
    report = {
        "created_utc": a.utc(),
        "status": "offline_batch_checked_live_pilot_pending",
        "plan_sha256": a.digest(b.PLAN),
        "programme": plan["programme"],
        "stages": [
            {k: v for k, v in x.items() if k != "jobs"} for x in schedule["stages"]
        ],
        "endpoints": endpoint_checks,
        "tests": tests.testsRun,
        "mock_items": 15,
        "mock_POSTs": 45,
        "new_POSTs_on_restart": 0,
        "actual_API_calls": 0,
        "real_gold_outcome_scoring": False,
        "verified_input_hashes": inputs,
        "verified_output_hashes": outputs,
        "notes": [
            "Gold masks and label-support sets were checked, but no actual prediction was scored.",
            "One initial fixture exposed ~3e-17 roundoff for identical 3-versus-9 setting comparisons; algebraically equivalent mean pairwise subtraction now gives exact zero for identical settings.",
            "Every input comparison stays inside one stage. Pilot items are real planned identities in a separate mock root; these mock outputs must never be reused as live results.",
        ],
    }
    a.save_new(OUT, report)
    a.save_new(
        BASE / "offline_check_complete.json",
        {
            "status": report["status"],
            "plan_sha256": a.digest(b.PLAN),
            "evidence": b.rel(OUT),
            "evidence_sha256": a.digest(OUT),
        },
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "tests": tests.testsRun,
                "programme": plan["programme"],
                "first_stage": report["stages"][0],
                "mock_items": 15,
                "mock_POSTs": 45,
                "resumed_POSTs": 0,
                "evidence_sha256": a.digest(OUT),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
