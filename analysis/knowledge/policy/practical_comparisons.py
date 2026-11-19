import numpy as np
from analysis.knowledge.policy import resources as p

c = p.c


def cost_direction(delta_fp, delta_fn):
    tol = 1e-10
    if abs(delta_fp) < tol:
        delta_fp = 0.0
    if abs(delta_fn) < tol:
        delta_fn = 0.0
    root = None if delta_fn == 0 else -delta_fp / delta_fn
    if delta_fp <= 0 and delta_fn <= 0:
        rule = "Equal if both are zero; otherwise fewer weighted errors for all strictly positive error costs."
    elif delta_fp >= 0 and delta_fn >= 0:
        rule = "Equal if both are zero; otherwise more weighted errors for all strictly positive error costs."
    elif delta_fn > 0:
        rule = f"Fewer weighted errors only when cost_FN/cost_FP < {root:.8g}; equal at the boundary."
    else:
        rule = f"Fewer weighted errors only when cost_FN/cost_FP > {root:.8g}; equal at the boundary."
    return dict(
        delta_FP=delta_fp,
        delta_FN=delta_fn,
        break_even_FN_over_FP=root,
        rule=rule,
        definition="Delta observed mean error cost = cost_FP * delta_FP + cost_FN * delta_FN. This is an illustration, not an adopted application loss.",
    )


def run():
    resource = c.read(p.BASE / "resources.json")
    timing = c.read(p.BASE / "timing.json")
    budget = c.read(c.SETUP / "budget.json")
    endpoint = c.read(c.SETUP / "endpoints.json")
    pairs = []
    quality = []
    utilities = []
    groups = resource["groups"]
    for left, right in ((1, 0), (2, 0), (2, 1)):
        a, b = (groups["condition/" + p.NAMES[i]] for i in (left, right))
        ua, ub = (v["usage"]["all_success"] for v in (a, b))
        fields = {
            name: dict(
                left=ua[name]["sum"],
                right=ub[name]["sum"],
                difference=ua[name]["sum"] - ub[name]["sum"],
                ratio=ua[name]["sum"] / ub[name]["sum"],
            )
            for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        repair_input_saving = (
            b["usage"]["repair"]["prompt_tokens"]["sum"]
            - a["usage"]["repair"]["prompt_tokens"]["sum"]
        )
        ordinary_input_increase = (
            a["usage"]["ordinary"]["prompt_tokens"]["sum"]
            - b["usage"]["ordinary"]["prompt_tokens"]["sum"]
        )
        pairs.append(
            dict(
                left=p.NAMES[left],
                right=p.NAMES[right],
                token_comparison=fields,
                fewer_repair_requests=b["repair_logical_calls"]
                - a["repair_logical_calls"],
                repair_reduction_fraction=(
                    b["repair_logical_calls"] - a["repair_logical_calls"]
                )
                / b["repair_logical_calls"],
                fewer_physical_attempts=b["physical_POST_attempts"]
                - a["physical_POST_attempts"],
                extra_valid_answers=a["valid_final"] - b["valid_final"],
                ordinary_input_increase=ordinary_input_increase,
                repair_input_saving=repair_input_saving,
                fraction_extra_ordinary_input_offset_by_fewer_repair_input_tokens=repair_input_saving
                / ordinary_input_increase,
                request_chain_mean_ratio=a["prediction_request_chain_s"]["mean"]
                / b["prediction_request_chain_s"]["mean"],
            )
        )
    p.save("condition_tradeoffs.json", pairs)
    for e in endpoint:
        point = c.read(p.RUN / e / "results.json")["point"]
        strict = c.read(p.P5 / "sensitivities" / e / "all_27_inner_valid/results.json")[
            "point"
        ]
        for i, cond in enumerate(p.NAMES):
            costs = resource["endpoint_common_costs"][e][cond]
            entry = dict(
                endpoint=e,
                condition=cond,
                n=point["n"],
                macro_f1=point["macro_f1"][i],
                micro_f1=point.get("micro_f1", [None] * 3)[i],
                exact=point["exact"][i],
                pairwise_agreement=point["repeatability"]["pairwise_agreement"][i],
                stable_correct=point["repeatability"]["stable_correct"][i],
                exact_answers_per_1000=1000 * point["exact"][i],
                prompt_tokens_per_voted_answer=costs["per_voted_answer"][
                    "prompt_tokens"
                ],
                total_tokens_per_voted_answer=costs["per_voted_answer"]["total_tokens"],
                request_chain_mean_s=costs["prediction_request_chain_s"]["mean"],
                successful_requests_per_1000_voted_answers=1000
                * costs["per_voted_answer"]["successful_requests"],
                strict_inner_macro=strict["macro_f1"][i],
            )
            quality.append(entry)
    p.save(
        "quality_cost_matrix.json",
        dict(
            rows=quality,
            caution="Copied verified quality values, matched common populations. One voted output contains all five fields. Costs and correct counts across overlapping endpoints must not be added; these are not an application utility or a new significance analysis.",
        ),
    )
    source = c.read(p.P4 / "groups_and_sources.json")
    for e in ("en_hate", "de_hate"):
        point = c.read(p.RUN / e / "results.json")["point"]
        selected = {"whole": point} | {
            name: r
            for name, r in source[e]["groups"].items()
            if name.startswith("source/")
        }
        for group, value in selected.items():
            if "confusion_mean" in value:
                fp = np.array(value["confusion_mean"]["fp"])
                fn = np.array(value["confusion_mean"]["fn"])
            else:
                fp = np.array(value["confusion_repeats"]["fp"]).mean(axis=1)
                fn = np.array(value["confusion_repeats"]["fn"]).mean(axis=1)
            for left, right in ((1, 0), (2, 0), (2, 1)):
                utilities.append(
                    dict(
                        endpoint=e,
                        group=group,
                        n=value["n"],
                        left=p.NAMES[left],
                        right=p.NAMES[right],
                        **cost_direction(
                            float(fp[left] - fp[right]), float(fn[left] - fn[right])
                        ),
                    )
                )
    p.save(
        "illustrative_error_costs.json",
        dict(
            rows=utilities,
            caution="Conditional arithmetic on observed mean FP/FN counts. No selected real-world cost ratio, no confidence interval for the threshold, no change to macro-primary design and no recommendation to deploy from these thresholds.",
        ),
    )
    wall = timing["wall"]
    rate = groups["all"]["successful_responses"] / wall["batch_attempt_s"]
    durations = wall["invocation_s"] / 3600
    p.save(
        "planning_reconciliation.json",
        dict(
            actual_invocation_h=durations,
            planned_range_h=[8, 16],
            reserved_h=18,
            within_planning_range=8 <= durations <= 16,
            planned_ordinary_input_tokens=sum(
                (
                    v["ordinary_input_tokens"]
                    for v in budget["condition_input_tokens"].values()
                )
            ),
            observed_ordinary_input_tokens=groups["all"]["usage"]["ordinary"][
                "prompt_tokens"
            ]["sum"],
            current_prompt_budget=dict(
                local_input_guard=8192,
                output_allowance=4096,
                maximum_initial_reference=max(
                    (v["max"] for v in budget["condition_input_tokens"].values())
                ),
                remaining_input_headroom_at_maximum_initial=8192
                - max((v["max"] for v in budget["condition_input_tokens"].values())),
            ),
            workload="6906 inputs x 3 conditions x 3 outer repeats x 3 inner responses. Production-style per-answer normalizations retain three inner responses; they remove the evaluation-only multiplication by outer repeats.",
            unmeasured=[
                "Actual GPU utilization/compute/energy",
                "Peak client RAM/CPU",
                "Allocated disk blocks",
                "Server cache-hit/discount details",
                "Standalone homogeneous-condition throughput",
                "Real-time retrieval latency avoided",
                "A billed monetary price",
            ],
            caution="Headroom is under this frozen local request guard, not a validated limit for a larger future policy. No linear runtime or per-token money estimate.",
        ),
    )
    print(
        "Matched quality/cost matrix and explicitly illustrative trade-offs saved.",
        flush=True,
    )


if __name__ == "__main__":
    with c.offline():
        run()
