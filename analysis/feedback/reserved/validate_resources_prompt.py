from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from analysis.feedback.reserved import resources as a


def main():
    io, need = a.io, a.need
    original = io.STUDY / "prompts.jsonl"
    assembly = io.read(a.ASSEMBLY / "complete.json")
    need(
        io.sha(original) == assembly["input_bindings"][io.rel(original)],
        "Frozen prompts changed",
    )
    stat = io.read(a.OUT / "statistics_complete.json")
    io.verify(a.ROOT, stat["bindings"])
    io.verify(a.ROOT, stat["artifacts"])
    meta = {
        (r["key"], r["condition"]): r for r in io.rows(a.P4 / "prompt_metadata.jsonl")
    }
    old = io.read(a.OUT / "local_prompt_costs.json")
    groups, keys, by_condition = defaultdict(list), set(), defaultdict(dict)
    for p in io.rows(original):
        key = p["key"], p["condition"]
        need(key not in keys, "Duplicate original prompt")
        keys.add(key)
        m = meta[key]

        token = p["reference_input_tokens"]
        need(
            type(token) is int
            and token == m["reference_input_tokens"]
            and p["messages_sha256"] == m["messages_sha256"],
            "Original/metadata disagreement",
        )
        by_condition[p["condition"]][p["key"]] = token
        for label in (
            "all",
            "condition/" + p["condition"],
            "mode/" + p["condition"].split("__")[0],
            "policy/" + a.policy(p["condition"]),
        ):
            groups[label].append(token)
    need(len(keys) == len(meta) == 49950, "Prompt coverage differs")
    for (key, condition), m in meta.items():
        need(
            by_condition[condition][key] - by_condition["reference5"][key]
            == m["token_difference_vs_reference"],
            "Independent original-token contrast differs",
        )
    corrected = deepcopy(old)
    changed_fields = []
    resource = io.read(a.OUT / "resource_groups.json")
    for label, tokens in groups.items():
        z = corrected["groups"][label]
        replacement = {
            "local_token_sum": sum(tokens),
            "local_token_mean": sum(tokens) / len(tokens),
            "local_token_min": min(tokens),
            "local_token_max": max(tokens),
            "local_tokens_times_nine_ordinary_slots": 9 * sum(tokens),
            "observed_minus_local_times_nine": resource[label]["ordinary_usage"][
                "prompt_tokens"
            ]
            - 9 * sum(tokens),
        }
        for field, value in replacement.items():
            if value != z[field]:
                changed_fields.append(
                    {"group": label, "field": field, "before": z[field], "after": value}
                )
            z[field] = value
        need(
            z["observed_minus_local_times_nine"] == 0,
            "Provider ordinary prompt usage differs from nine original local counts",
        )
    need(
        corrected["groups"]["all"]["local_token_min"] == 622
        and corrected["groups"]["all"]["local_token_max"] == 1208,
        "Frozen token bounds differ",
    )
    corrected["qualification"] += (
        " Corrected v2 reads reference_input_tokens as the entire count of each individual prompt; "
        "the historical field name does not mean reference5 for every condition. "
        "All ordinary reported prompt totals equal nine times the frozen local counts in this run; "
        "repair usage remains additional. This equality is observed here, not a universal pricing rule."
    )
    corrected["supersedes"] = io.rel(a.OUT / "local_prompt_costs.json")
    a.write(a.OUT / "local_prompt_costs_v2.json", corrected)
    check = {
        "status": "part8_all_original_prompt_token_counts_checked_and_corrected",
        "created_utc": io.utc(),
        "original_prompts_checked": 49950,
        "raw_original_contrast_checks": 49950,
        "resource_groups_with_exact_usage_match": len(groups),
        "changed_derived_fields": changed_fields,
        "reason": "Unpublished Part 8 v2 local summary added the contrast delta to a field that "
        "already contains the full prompt count. Original prompts, prior analyses, "
        "provider usage, request accounting and timing are unchanged.",
        "old_artifacts_preserved": True,
        "new_model_calls": 0,
        "bindings": {
            io.rel(p): io.sha(p)
            for p in [
                original,
                a.P4 / "prompt_metadata.jsonl",
                a.OUT / "local_prompt_costs.json",
                a.OUT / "local_prompt_costs_v2.json",
                a.OUT / "statistics_complete.json",
                Path(__file__),
            ]
        },
    }
    a.write(a.OUT / "prompt_token_check.json", check)
    prior = a.OUT / "independent_resource_check.json"
    review = io.read(prior)
    review["status"] = (
        "part8_resources_and_original_prompt_tokens_independently_reconciled"
    )
    review["created_utc"] = io.utc()
    review["local_token_claims_superseded_by"] = io.rel(
        a.OUT / "prompt_token_check.json"
    )
    review["local_token_source"] = io.rel(a.OUT / "local_prompt_costs_v2.json")
    review["original_frozen_prompts_checked"] = 49950
    review["bindings"].update(
        {
            io.rel(p): io.sha(p)
            for p in [
                prior,
                a.OUT / "prompt_token_check.json",
                a.OUT / "local_prompt_costs_v2.json",
            ]
        }
    )
    a.write(a.OUT / "independent_resource_check_v2.json", review)
    print(
        "All 49,950 original prompt counts and deltas checked; corrected v2 totals "
        "exactly match ordinary reported prompt usage. Other resource statistics unchanged."
    )


if __name__ == "__main__":
    with a.io.offline():
        main()
