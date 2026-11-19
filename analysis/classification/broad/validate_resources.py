from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from analysis.classification.broad import resource_summary as analysis

x = analysis.x
checks = 0
max_difference = 0.0


def eq(a, b, label=""):
    global checks, max_difference
    a = x.prior.safe(a)
    b = x.prior.safe(b)
    if isinstance(a, dict):
        x.require(
            isinstance(b, dict) and a.keys() == b.keys(), "Different keys " + label
        )
        for k in a:
            eq(a[k], b[k], label + "/" + str(k))
    elif isinstance(a, list):
        x.require(
            isinstance(b, list) and len(a) == len(b), "Different lengths " + label
        )
        for i, (u, v) in enumerate(zip(a, b)):
            eq(u, v, label + "/" + str(i))
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        x.require(
            isinstance(b, (int, float)) and not isinstance(b, bool),
            "Different numeric type " + label,
        )
        difference = abs(a - b)
        max_difference = max(max_difference, difference)
        checks += 1
        x.require(
            np.isclose(a, b, rtol=2e-12, atol=1e-7),
            "Different number " + label + ": " + str(a) + " / " + str(b),
        )
    else:
        checks += 1
        x.require(a == b, "Different value " + label)


def stats(values):
    a = sorted(float(v) for v in values if np.isfinite(v))
    n = len(a)
    if not n:
        return dict(
            n=0,
            sum=0.0,
            mean=None,
            min=None,
            p50=None,
            p90=None,
            p95=None,
            p99=None,
            max=None,
        )

    def q(p):
        i = (n - 1) * p
        lo = int(i)
        hi = min(lo + 1, n - 1)
        return a[lo] + (a[hi] - a[lo]) * (i - lo)

    return dict(
        n=n,
        sum=sum(a),
        mean=sum(a) / n,
        min=a[0],
        p50=q(0.5),
        p90=q(0.9),
        p95=q(0.95),
        p99=q(0.99),
        max=a[-1],
    )


def group_keys(row):
    return [
        "all",
        "condition/" + row["condition"],
        "phase/" + row["phase"] + "/" + row["condition"],
        "repeat/" + row["phase"] + "/" + row["condition"] + "/" + str(row["repeat"]),
        "source/" + row["source"] + "/" + row["condition"],
        "stage/" + row["stage"],
        "validity/" + row["condition"] + "/" + str(row["n_valid"]),
    ]


def run():
    extraction = x.c.read(x.BASE / "extraction_complete.json")
    x.r.verify(extraction["sources"])
    x.r.verify(extraction["outputs"])
    summary = x.c.read(x.BASE / "resource_summary.json")
    x.r.verify(summary["sources"])
    context = x.c.read(x.BASE / "preparation_and_storage.json")
    x.r.verify(context["sources"])
    with np.load(x.BASE / "events.npz", allow_pickle=False) as z:
        e = z["values"]
        eq(z["columns"].tolist(), list(x.EVENT_COLUMNS))
    with np.load(x.BASE / "record_metrics.npz", allow_pickle=False) as z:
        a = z["values"]
        eq(z["columns"].tolist(), list(analysis.METRICS))
    index = x.c.read(x.BASE / "record_index.json")
    identifiers = index["records"]
    eq(index["columns"], list(analysis.METRICS))
    eq(len(identifiers), 194643)
    eq(len(e), 641429)
    n = len(a)
    ri = e[:, 0].astype(int)
    x.require(
        np.all(np.diff(ri) >= 0) and ri.min() == 0 and ri.max() == n - 1,
        "Event order/index differs",
    )
    ok = e[:, 4] == 1
    repair = e[:, 2] > 0

    def total(column, mask=None):
        return np.bincount(
            ri,
            weights=e[:, column] if mask is None else e[:, column] * mask,
            minlength=n,
        )

    recon = np.column_stack(
        [
            total(7),
            total(8),
            total(9),
            total(7, repair),
            total(8, repair),
            total(9, repair),
            np.bincount(ri, minlength=n),
            np.bincount(ri, weights=ok & repair, minlength=n),
            np.bincount(ri, weights=e[:, 3] > 0, minlength=n),
            total(5),
            total(6),
            a[:, 11],
            total(10),
            total(11),
            a[:, 14],
            a[:, 15],
            total(5, ~repair),
            total(5, repair),
            total(5, ok),
            total(5, ~ok),
        ]
    )
    eq(recon, a, "Independent event reduction")
    groups = defaultdict(list)
    classes = defaultdict(Counter)
    lookup = {}
    raw_paths = {}
    stage_edges = defaultdict(list)
    for i, (row, cache) in enumerate(
        zip(
            x.reader.rows(x.BASE / "records.jsonl"),
            x.reader.rows(x.reader.BASE / "predictions.jsonl"),
            strict=True,
        )
    ):
        eq(row["index"], i)
        eq({k: row[k] for k in identifiers[i]}, identifiers[i])
        eq(
            {
                k: row[k]
                for k in (
                    "key",
                    "phase",
                    "source",
                    "condition",
                    "repeat",
                    "stage",
                    "n_valid",
                    "repairs",
                    "parse_failures",
                    "uncertain",
                    "result_path",
                )
            },
            {
                k: cache[k]
                for k in (
                    "key",
                    "phase",
                    "source",
                    "condition",
                    "repeat",
                    "stage",
                    "n_valid",
                    "repairs",
                    "parse_failures",
                    "uncertain",
                    "result_path",
                )
            },
        )
        eq([row["usage"][k] for k in x.FIELDS], a[i, :3])
        eq([row["repair_usage"][k] for k in x.FIELDS], a[i, 3:6])
        eq(row["request_seconds"], a[i, 9])
        eq(row["backoff_seconds"], a[i, 10])
        eq(sum(row["vote_seconds"]), a[i, 9] + a[i, 10])
        eq(row["delivered_rationale_chars"], x.prior.safe(a[i, 14]))
        eq(row["delivered_rationale_utf8_bytes"], x.prior.safe(a[i, 15]))
        eq(
            row["usage"]["prompt_tokens"] + row["usage"]["completion_tokens"],
            row["usage"]["total_tokens"],
        )
        eq(row["accounting"]["ordinary_logical_calls"], 3)
        eq(row["accounting"]["post_attempts"], int(a[i, 6]))
        eq(row["accounting"]["repair_logical_calls"], int(a[i, 7]))
        eq(row["accounting"]["retry_attempts"], int(a[i, 8]))
        lookup[row["key"], row["condition"], row["repeat"]] = i
        raw_paths[row["result_path"]] = (i, row["result_sha256"])
        stage_edges[row["stage"]].append(row["result_path"])
        for g in group_keys(row):
            groups[g].append(i)
            classes[g].update(row["response_classes"])
    eq(i + 1, n)
    for group in summary["groups"]:
        name = group["group"]
        ii = groups[name]
        m = np.zeros(n, bool)
        m[ii] = True
        vv = a[ii]
        ev = e[m[ri]]
        success = ev[:, 4] == 1
        rep = ev[:, 2] > 0
        eq(group["predictions"], len(ii))
        eq(group["valid_predictions"], int((vv[:, 11] > 0).sum()))
        eq(group["null_predictions"], int((vv[:, 11] == 0).sum()))
        eq(group["partial_predictions"], int(((vv[:, 11] > 0) & (vv[:, 11] < 3)).sum()))
        eq(group["repaired_predictions"], int((vv[:, 7] > 0).sum()))
        eq(group["response_classes"], dict(classes[name]))
        for j, k in enumerate(analysis.METRICS):
            if k != "n_valid":
                eq(group["metrics"][k], stats(vv[:, j]), name + "/" + k)
        masks = {
            "all": np.ones(len(ev), bool),
            "ordinary": ~rep,
            "repair": rep,
            "successful": success,
            "failed": ~success,
            "ordinary_successful": success & ~rep,
            "repair_successful": success & rep,
        }
        for label, mask in masks.items():
            z = ev[mask]
            ss = z[:, 4] == 1
            want = dict(
                attempts=len(z),
                successes=int(ss.sum()),
                request_seconds=stats(z[:, 5]),
                backoff_seconds=stats(z[:, 6]),
            )
            for k, j in [
                ("prompt_tokens", 7),
                ("completion_tokens", 8),
                ("total_tokens", 9),
                ("raw_chars", 10),
                ("raw_utf8_bytes", 11),
            ]:
                want[k] = stats(z[ss, j])
            eq(group["event_slices"][label], want, name + "/" + label)
        eq(
            group["mean_tokens_per_valid_output_with_all_attempt_cost"],
            (
                sum(vv[:, 2]) / group["valid_predictions"]
                if group["valid_predictions"]
                else None
            ),
        )
        eq(group["repair_token_share"], sum(vv[:, 5]) / sum(vv[:, 2]))
        eq(group["retry_attempt_share"], sum(vv[:, 8]) / sum(vv[:, 6]))
    eq(set(groups), {v["group"] for v in summary["groups"]})

    selected = {
        a["result_path"]
        for a in x.c.read(x.reader.BASE / "reader_review.json")["raw_string_checks"]
    }
    selected |= {p for values in stage_edges.values() for p in (values[0], values[-1])}
    sample_sources = {}
    for path in sorted(selected):
        i, digest = raw_paths[path]
        data = Path(path).read_bytes()
        eq(hashlib.sha256(data).hexdigest(), digest)
        sample_sources[path] = digest
        v = json.loads(data)
        expected = []
        chars = Counter()
        for event in v["events"]:
            u = (event.get("model_response_metadata") or {}).get("usage") or {}
            is_ok = event["action"] == "ok"
            text = v["raw_runs"][event["vote"]][event["turn"]] if is_ok else ""
            expected.append(
                [
                    i,
                    event["vote"],
                    event["turn"],
                    event["attempt"],
                    int(is_ok),
                    event["elapsed_s"],
                    event["backoff_elapsed_s"],
                    u.get("prompt_tokens", 0),
                    u.get("completion_tokens", 0),
                    u.get("total_tokens", 0),
                    len(text),
                    len(text.encode("utf8")),
                ]
            )
        lo, hi = np.searchsorted(ri, [i, i + 1])
        eq(e[lo:hi], expected, "Original sample event")
        eq(a[i, 14], len(v["result"]["reasoning"]) if v["result"] is not None else None)
        eq(
            a[i, 15],
            (
                len(v["result"]["reasoning"].encode("utf8"))
                if v["result"] is not None
                else None
            ),
        )
    inventory = x.c.read(x.BASE / "execution_inventory.json")
    x.r.verify(inventory["source_bindings"])
    whole = {}
    for stage in inventory["stages"]:
        root = x.r.BASE / "live/broad/stages" / stage["stage"]
        review = x.c.read(root / "review.json")
        report = x.c.read(root / review["report"])
        complete = x.c.read(root / "complete.json")
        x.r.verify(
            {
                x.r.reference(x.r.BASE / "live/broad" / k): v
                for k, v in report["attempt_hashes"].items()
            }
        )
        final_attempt = Path(complete["final_attempt"])
        eq(x.c.sha(final_attempt / "summary.json"), complete["summary_sha256"])
        eq(x.c.read(final_attempt / "summary.json"), complete["summary"])
        for attempt in stage["attempts"]:
            folder = Path(attempt["path"])
            start = x.c.read(folder / "start.json")
            end = x.c.read(folder / "summary.json")
            check = x.c.read(folder / "model_check_response.json")
            eq(attempt["start"], start["started_utc"])
            eq(attempt["end"], end["finished_utc"])
            eq(attempt["peak_workers"], end["peak_submitted"])
            eq(attempt["saved_items"], end["saved_items"])
            eq(attempt["model_check_seconds"], check["elapsed_s"])
            eq(check["status"], 200)
    for key in ("counts", "accounting", "usage"):
        count = Counter()
        for stage in inventory["stages"]:
            count.update(stage[key])
        eq(dict(count), inventory["totals"][key])
        whole[key] = dict(count)
    final = x.c.read(x.r.BASE / "live/complete.json")
    eq(whole["counts"], final["counts"])
    eq(whole["accounting"], final["accounting"])
    eq(whole["usage"], final["observed_usage"])
    for row in inventory["stages"]:
        ids = groups["stage/" + row["stage"]]
        vv = a[ids]
        eq(int(vv[:, 0].sum()), row["usage"]["prompt_tokens"])
        eq(int(vv[:, 1].sum()), row["usage"]["completion_tokens"])
        eq(float(vv[:, 9:11].sum()), row["vote_seconds"])

    def diff(u, v):
        return (datetime.fromisoformat(v) - datetime.fromisoformat(u)).total_seconds()

    t = summary["timing"]
    tt = t["batches"]
    eq(
        diff(tt[0]["start_utc"], tt[-1]["audit_utc"]),
        t["first_to_last_stage_audit_seconds"],
    )
    eq(
        diff(tt[0]["start_utc"], t["final_completion_mtime_utc"]),
        t["total_mtime_proxy_seconds"],
    )
    eq(
        sum(
            t[k]
            for k in (
                "generation_attempt_seconds",
                "between_attempt_seconds",
                "post_summary_commit_seconds",
                "complete_to_audit_seconds",
                "audit_to_next_start_seconds",
            )
        ),
        t["first_to_last_stage_audit_seconds"],
    )
    for j, (row, saved) in enumerate(zip(tt, inventory["stages"], strict=True)):
        eq(
            row["generation_attempt_seconds"],
            sum(diff(k["start"], k["end"]) for k in saved["attempts"]),
        )
        eq(
            row["post_summary_commit_seconds"],
            diff(row["summary_utc"], row["complete_utc"]),
        )
        eq(
            row["complete_to_audit_seconds"],
            diff(row["complete_utc"], row["audit_utc"]),
        )
        eq(
            row["audit_to_next_start_seconds"],
            diff(row["audit_utc"], tt[j + 1]["start_utc"]) if j + 1 < len(tt) else None,
        )
        eq(
            row["completed_predictions_per_attempt_second"],
            row["predictions"] / row["generation_attempt_seconds"],
        )
        eq(
            row["reported_tokens_per_attempt_second"],
            row["tokens"] / row["generation_attempt_seconds"],
        )
        eq(
            row["summed_vote_seconds_per_attempt_second"],
            saved["vote_seconds"] / row["generation_attempt_seconds"],
        )
    for row in summary["quality_cost"]:
        old, terms = x.prior.endpoint(row["endpoint"])
        excluded = set(old["three_condition_common"]["excluded_ids"])
        keys = [k for k in next(iter(terms.values()))["ids"] if k not in excluded]
        ids = [lookup[k, row["condition"], rep] for k in keys for rep in range(3)]
        vv = a[ids]
        eq(len(keys), row["n_common_inputs"])
        eq(len(ids), row["predictions"])
        p = old["three_condition_common"]["per_condition"][row["condition"]]
        eq(row["mean_macro_f1"], p["mean_macro_f1"])
        eq(row["repeat_macro_f1"], p["repeat_macro_f1"])
        for key, j in [
            ("mean_reported_tokens_per_prediction", 2),
            ("mean_prompt_tokens_per_prediction", 0),
            ("mean_completion_tokens_per_prediction", 1),
            ("mean_request_seconds_per_prediction", 9),
            ("mean_backoff_seconds_per_prediction", 10),
            ("mean_repairs_per_prediction", 7),
        ]:
            eq(row[key], sum(vv[:, j]) / len(vv))
    for row in context["files"]:
        eq((x.ROOT / row["path"]).stat().st_size, row["bytes"])
        eq(x.c.sha(x.ROOT / row["path"]), row["sha256"])
    for row in context["retrieval_passes"]:
        old = x.c.read(x.c.BASE / "retrieval_v3" / row["pass_name"] / "complete.json")
        eq(row["seconds"], old["seconds"])
        eq(row["reused_rows"], old["checkpoint_resume"]["reused_rows"])
        eq(row["complete_pass_timer"], row["reused_rows"] == 0)
    descriptions = defaultdict(lambda: defaultdict(list))
    for row in x.reader.rows(
        x.ROOT
        / "experiments/astra/sq2_broad_context_review_20261104_v1/prompt_descriptors.jsonl"
    ):
        for key in (row["condition"], row["phase"] + "/" + row["condition"]):
            for metric in ("message_chars", "message_utf8_bytes", "context_chars"):
                descriptions[key][metric].append(row[metric])
    for row in context["prompt_sizes"]:
        for metric in ("message_chars", "message_utf8_bytes", "context_chars"):
            eq(row[metric], stats(descriptions[row["group"]][metric]))
    x.save(
        x.BASE / "independent_verification.json",
        dict(
            status="resource_events_records_summaries_timeline_preparation_and_quality_links_passed",
            checked_values=checks,
            maximum_absolute_float_difference=max_difference,
            tolerance=dict(relative=2e-12, absolute=1e-7),
            predictions=n,
            events=len(e),
            groups=len(groups),
            raw_records_independently_checked=len(selected),
            source_hashes=sample_sources,
            sources=x.bindings(
                [
                    Path(__file__),
                    x.BASE / "extraction_complete.json",
                    x.BASE / "resource_summary.json",
                    x.BASE / "record_index.json",
                    x.BASE / "record_metrics.npz",
                    x.BASE / "preparation_and_storage.json",
                ]
            ),
            new_Qwen_calls=0,
        ),
    )
    print(
        "Independent resource verification passed:",
        checks,
        "values;",
        len(selected),
        "original result samples.",
        flush=True,
    )


if __name__ == "__main__":
    with x.r.original.offline(), x.prior.threadpool_limits(limits=2):
        run()
