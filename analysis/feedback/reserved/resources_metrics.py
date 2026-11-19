from collections import Counter
from datetime import datetime, timezone
import math


def need(ok, message):
    if not ok:
        raise ValueError(message)


def timestamp(value):
    x = datetime.fromisoformat(value)
    need(x.tzinfo is not None and x.utcoffset() is not None, "Naive timestamp")
    return x.astimezone(timezone.utc)


def seconds(start, end):
    value = (timestamp(end) - timestamp(start)).total_seconds()
    need(value >= 0, "Negative chronological interval")
    return value


def quantile(values, q):
    need(bool(values) and 0 <= q <= 1, "Invalid quantile")
    s = sorted(values)
    index = (len(s) - 1) * q
    lo = math.floor(index)
    return s[lo] + (s[math.ceil(index)] - s[lo]) * (index - lo)


def accounting_check(a):
    need(all(type(v) is int and v >= 0 for v in a.values()), "Bad integer accounting")
    need(
        a["post_attempts"]
        == a["ordinary_logical_calls"]
        + a["repair_logical_calls"]
        + a["retry_attempts"],
        "Attempt slots do not reconcile",
    )
    need(
        a["http_success_with_text"]
        == a["ordinary_logical_calls"] + a["repair_logical_calls"],
        "Not all logical calls have a successful response",
    )
    need(a["retryable_failures"] == a["retry_attempts"], "Retry accounting differs")


def usage_check(u):
    need(
        all(
            type(u.get(k)) is int and u[k] >= 0
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        ),
        "Invalid usage",
    )
    need(
        u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"],
        "Usage differs",
    )


class Group:
    def __init__(self):
        self.accounting = Counter()
        self.usage = Counter()
        self.ordinary = Counter()
        self.repair = Counter()
        self.durations = []
        self.backoffs = []
        self.request_sums = []
        self.counts = Counter()

    def add(self, row):
        a = row["accounting"]
        accounting_check(a)
        need(
            a["ordinary_logical_calls"] == 3,
            "Prediction must retain all three ordinary slots",
        )
        for u in (row["observed_usage"], row["ordinary_usage"]):
            usage_check(u)
        r = {k: row["repair_usage"].get(k, 0) for k in row["observed_usage"]}
        usage_check(r)
        need(
            all(row["observed_usage"][k] == row["ordinary_usage"][k] + r[k] for k in r),
            "Ordinary/repair usage not disjoint",
        )
        ds = row["known_request_durations_s"]
        n = row["administrative_unknown_requests"]
        need(
            all(type(d) in (int, float) and math.isfinite(d) and d >= 0 for d in ds),
            "Invalid known request duration",
        )
        need(
            type(n) is int and n >= 0 and len(ds) + n == a["post_attempts"],
            "Unknown sentinels incorrectly counted as known durations",
        )
        need(
            len(ds) == row["known_request_duration_count"]
            and math.isclose(
                math.fsum(ds), row["known_request_seconds_sum"], abs_tol=1e-8
            ),
            "Request duration count/sum differs",
        )
        wait = row["retry_backoff_seconds_sum"]
        need(math.isfinite(wait) and wait >= 0, "Invalid retry wait")
        self.accounting.update(a)
        self.usage.update(row["observed_usage"])
        self.ordinary.update(row["ordinary_usage"])
        self.repair.update(r)
        self.durations.extend(ds)
        self.backoffs.append(wait)
        self.request_sums.append(math.fsum(ds))
        self.counts.update(
            predictions=1,
            nulls=int(row["result"] is None),
            administrative_unknown_requests=n,
            parse_failures=row["parse_failures"],
            repairs=row["repairs"],
        )
        self.counts["valid_votes_" + str(row["n_valid"])] += 1

    def finish(self):
        a, u, n = self.accounting, self.usage, self.counts["predictions"]
        accounting_check(a)
        usage_check(u)
        duration = math.fsum(self.durations)
        need(
            math.isclose(duration, math.fsum(self.request_sums), abs_tol=1e-7),
            "Duration aggregation differs",
        )
        need(
            len(self.durations) + self.counts["administrative_unknown_requests"]
            == a["post_attempts"],
            "Missing duration slots",
        )
        return {
            "counts": dict(self.counts),
            "accounting": dict(a),
            "observed_usage": dict(u),
            "ordinary_usage": dict(self.ordinary),
            "repair_usage": dict(self.repair),
            "known_request_duration_count": len(self.durations),
            "known_request_seconds_sum": duration,
            "known_request_seconds_mean": duration / len(self.durations),
            "known_request_seconds_min": min(self.durations),
            "known_request_seconds_max": max(self.durations),
            "known_request_seconds_p50": quantile(self.durations, 0.5),
            "known_request_seconds_p95": quantile(self.durations, 0.95),
            "known_request_seconds_p99": quantile(self.durations, 0.99),
            "retry_backoff_seconds_sum": math.fsum(self.backoffs),
            "per_planned_prediction": {
                "denominator": n,
                "observed_tokens": u["total_tokens"] / n,
                "prompt_tokens": u["prompt_tokens"] / n,
                "completion_tokens": u["completion_tokens"] / n,
                "repair_calls": a["repair_logical_calls"] / n,
                "attempt_slots": a["post_attempts"] / n,
                "known_request_seconds": duration / n,
            },
            "repair_calls_per_ordinary_call": a["repair_logical_calls"]
            / a["ordinary_logical_calls"],
            "extra_attempts_per_logical_call": a["retry_attempts"]
            / (a["ordinary_logical_calls"] + a["repair_logical_calls"]),
        }


def attempt_partition(attempts, final_end):

    ordered = sorted(attempts, key=lambda r: timestamp(r["start"]))
    need(len({r["start"] for r in ordered}) == len(ordered), "Duplicate start marker")
    pieces = []
    for i, r in enumerate(ordered):
        stop = ordered[i + 1]["start"] if i + 1 < len(ordered) else final_end
        seconds(r["start"], stop)
        end = r.get("end")
        check = r.get("model_check_start")
        if end is not None:
            seconds(r["start"], end)
            seconds(end, stop)
        if check is not None:
            seconds(r["start"], check)
            seconds(check, end or stop)
            pieces.append(
                {"kind": "start_to_model_check", "start": r["start"], "end": check}
            )
            pieces.append(
                {
                    "kind": (
                        "model_check_to_recorded_end"
                        if end
                        else "unallocated_interrupted_window"
                    ),
                    "start": check,
                    "end": end or stop,
                }
            )
        else:
            pieces.append(
                {
                    "kind": (
                        "recorded_attempt_without_model_check"
                        if end
                        else "unallocated_interrupted_window"
                    ),
                    "start": r["start"],
                    "end": end or stop,
                }
            )
        if end is not None and timestamp(end) < timestamp(stop):
            pieces.append(
                {"kind": "explicit_between_attempt_gap", "start": end, "end": stop}
            )
    totals = Counter()
    for p in pieces:
        p["seconds"] = seconds(p["start"], p["end"])
        totals[p["kind"]] += p["seconds"]
    need(
        math.isclose(
            sum(totals.values()), seconds(ordered[0]["start"], final_end), abs_tol=1e-6
        ),
        "Timing partition differs",
    )
    return {"pieces": pieces, "seconds_by_kind": dict(totals)}
