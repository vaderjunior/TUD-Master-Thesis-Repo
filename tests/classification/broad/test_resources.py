import copy
from pathlib import Path
import unittest
from analysis.classification.broad import resources as x


def fixture(repair=False, retry=False, missing=False):
    meta = dict(
        key="test/key",
        phase="en_test",
        source="test",
        condition="rag_full",
        repeat=0,
        stage="test",
        n_valid=0 if missing else 3,
        repairs=int(repair),
        parse_failures=int(missing) * 3,
        gate_normalised=0,
        normalisations=0,
        uncertain=False,
        result_path="test/result.json",
        messages_sha256="prompt",
    )
    value = dict(
        identity=dict(key=meta["key"], repeat=0, condition="broad::rag_full"),
        messages_sha256="prompt",
        n_valid=meta["n_valid"],
        latencies=[2.0, 2.0, 2.0],
        raw_runs=[["abc"], ["abc"], ["abc"]],
        events=[],
        repairs=int(repair),
        requested_model="test-model",
        active_model="test-model",
        result=None if missing else {"reasoning": "ä"},
    )
    for vote in range(3):
        if vote == 0 and retry:
            value["events"].append(
                dict(
                    vote=0,
                    turn=0,
                    attempt=0,
                    kind="ordinary",
                    action="retry",
                    response_class="ReadTimeout",
                    elapsed_s=120.0,
                    backoff_elapsed_s=0.0,
                    model_response_metadata=None,
                )
            )
            value["latencies"][0] += 122.0
        value["events"].append(
            dict(
                vote=vote,
                turn=0,
                attempt=int(vote == 0 and retry),
                kind="ordinary",
                action="ok",
                response_class="success",
                elapsed_s=2.0,
                backoff_elapsed_s=2.0 if vote == 0 and retry else 0.0,
                model_response_metadata=dict(
                    reported_model="test-model",
                    finish_reason="stop",
                    usage=dict(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                ),
            )
        )
        if vote == 0 and repair:
            value["events"].append(
                dict(
                    vote=0,
                    turn=1,
                    attempt=0,
                    kind="repair",
                    action="ok",
                    response_class="success",
                    elapsed_s=3.0,
                    backoff_elapsed_s=0.0,
                    model_response_metadata=dict(
                        reported_model="test-model",
                        finish_reason="stop",
                        usage=dict(
                            prompt_tokens=30, completion_tokens=6, total_tokens=36
                        ),
                    ),
                )
            )
            value["raw_runs"][0].append("ä")
            value["latencies"][0] += 3.0
    value["accounting"] = dict(
        ordinary_logical_calls=3,
        repair_logical_calls=int(repair),
        post_attempts=3 + int(repair) + int(retry),
        retry_attempts=int(retry),
        retryable_failures=int(retry),
        http_success_with_text=3 + int(repair),
        reported_model_present=3 + int(repair),
    )
    return value, meta


class ResourceTests(unittest.TestCase):
    def get(self, **kwargs):
        return x.extract_record(*fixture(**kwargs), 7)

    def rejects(self, change):
        a, b = fixture()
        change(a, b)
        with self.assertRaises((ValueError, KeyError)):
            x.extract_record(a, b, 0)

    def test_plain(self):
        a, ev, _ = self.get()
        self.assertEqual(
            a["usage"], dict(prompt_tokens=30, completion_tokens=15, total_tokens=45)
        )
        self.assertEqual(len(ev), 3)

    def test_zero_repairs_retained(self):
        self.assertEqual(
            self.get()[0]["repair_usage"],
            dict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    def test_repair_is_not_retry(self):
        a, _, _ = self.get(repair=True)
        self.assertEqual(
            (
                a["accounting"]["repair_logical_calls"],
                a["accounting"]["retry_attempts"],
            ),
            (1, 0),
        )

    def test_retry_has_no_invented_tokens(self):
        a, _, _ = self.get(retry=True)
        self.assertEqual(a["usage"]["total_tokens"], 45)
        self.assertEqual(a["request_seconds"], 126)
        self.assertEqual(a["backoff_seconds"], 2)

    def test_repair_extended_prompt(self):
        a, _, _ = self.get(repair=True)
        self.assertEqual(a["repair_usage"]["prompt_tokens"], 30)
        self.assertEqual(a["ordinary_usage"]["prompt_tokens"], 30)

    def test_both(self):
        self.assertEqual(
            self.get(repair=True, retry=True)[0]["accounting"]["post_attempts"], 5
        )

    def test_null_keeps_cost(self):
        a, _, _ = self.get(missing=True)
        self.assertEqual(a["usage"]["total_tokens"], 45)
        self.assertIsNone(a["delivered_rationale_chars"])

    def test_unicode_bytes(self):
        a, _, _ = self.get(repair=True)
        self.assertEqual(
            (a["delivered_rationale_chars"], a["delivered_rationale_utf8_bytes"]),
            (1, 2),
        )
        self.assertEqual(a["success_raw_utf8_bytes"], 11)

    def test_no_token_character_equivalence(self):
        self.assertNotEqual(
            self.get()[0]["usage"]["completion_tokens"],
            self.get()[0]["success_raw_chars"],
        )

    def test_missing_usage_rejected(self):
        self.rejects(
            lambda a, b: a["events"][0]["model_response_metadata"].update(usage=None)
        )

    def test_negative_token_rejected(self):
        self.rejects(
            lambda a, b: a["events"][0]["model_response_metadata"]["usage"].update(
                prompt_tokens=-1
            )
        )

    def test_bad_token_sum(self):
        self.rejects(
            lambda a, b: a["events"][0]["model_response_metadata"]["usage"].update(
                total_tokens=16
            )
        )

    def test_bool_token_rejected(self):
        self.rejects(
            lambda a, b: a["events"][0]["model_response_metadata"]["usage"].update(
                completion_tokens=True
            )
        )

    def test_bad_time(self):
        self.rejects(lambda a, b: a["events"][0].update(elapsed_s=float("nan")))

    def test_negative_wait(self):
        self.rejects(lambda a, b: a["events"][0].update(backoff_elapsed_s=-1))

    def test_backoff_first_attempt(self):
        self.rejects(lambda a, b: a["events"][0].update(backoff_elapsed_s=2))

    def test_duplicate_attempt(self):
        self.rejects(lambda a, b: a["events"].insert(0, copy.deepcopy(a["events"][0])))

    def test_attempt_gap(self):
        self.rejects(lambda a, b: a["events"][0].update(attempt=2))

    def test_turn_kind(self):
        self.rejects(lambda a, b: a["events"][0].update(kind="repair"))

    def test_identity(self):
        self.rejects(lambda a, b: a["identity"].update(key="other"))

    def test_metadata_repeat(self):
        self.rejects(lambda a, b: b.update(repeat=2))

    def test_model(self):
        self.rejects(
            lambda a, b: a["events"][0]["model_response_metadata"].update(
                reported_model="other"
            )
        )

    def test_accounting(self):
        self.rejects(lambda a, b: a["accounting"].update(post_attempts=30))

    def test_latency_boundary(self):
        self.rejects(lambda a, b: a.update(latencies=[1.0, 2.0, 2.0]))

    def test_utc_offset(self):
        self.assertEqual(
            x.seconds("2026-10-30T12:00:00+01:00", "2026-10-30T12:00:00+00:00"), 3600
        )

    def test_reversed_clock(self):
        with self.assertRaises(ValueError):
            x.seconds("2026-10-30T12:00:00+00:00", "2026-10-30T11:00:00+00:00")

    def test_fractional_time(self):
        self.assertAlmostEqual(
            x.seconds("2026-10-30T12:00:00.1+00:00", "2026-10-30T12:00:00.3+00:00"), 0.2
        )


if __name__ == "__main__":
    with x.r.original.offline():
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(ResourceTests)
        )
        x.require(result.wasSuccessful(), "Scientific accounting tests failed")
        x.save(
            x.BASE / "tests.json",
            dict(
                status="passed",
                tests=result.testsRun,
                sources=x.bindings([Path(__file__), Path(x.__file__)]),
                new_Qwen_calls=0,
            ),
        )
