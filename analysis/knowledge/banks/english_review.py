from collections import Counter, defaultdict
from datetime import datetime, timezone
import re
import numpy as np
from analysis.knowledge.banks import english_results as r

OUT = (
    r.ROOT
    / "Thesis_Log/astra_result/evidence/english_kb_review_supplement_20260914_v1.json"
)
TABLE = r.ROOT / "Thesis_Log/astra_result/astra_english_kb_review_20260914_v1.md"


def main():
    assert not OUT.exists() and (not TABLE.exists()), "Preserve existing evidence"
    base = r.read(r.OUT)
    plan = r.read(r.BASE / "plan.json")
    train = r.lines(r.BASE / "train_texts.jsonl")
    held = r.lines(r.BASE / "heldout_texts.jsonl")
    rows = r.lines(r.BASE / "nearest_train.jsonl")
    paths = [
        r.OUT,
        r.BASE / "plan.json",
        r.BASE / "complete.json",
        r.BASE / "train_texts.jsonl",
        r.BASE / "heldout_texts.jsonl",
        r.BASE / "nearest_train.jsonl",
        r.OLD / "batch_complete.json",
        r.OLD / "plan.json",
        r.OLD / "retrieval_complete.json",
        r.OLD / "retrieval_small.json",
        r.OLD / "retrieval_all.json",
        r.ROOT / "analysis/knowledge/banks/build.py",
        r.ROOT / "analysis/knowledge/banks/english_results.py",
        r.ROOT / "analysis/knowledge/banks/english_review.py",
    ]
    receipt = r.read(r.OLD / "batch_complete.json")
    oldplan = r.read(r.OLD / "plan.json")
    assert receipt["plan_sha256"] == r.digest(r.OLD / "plan.json")
    assert (
        r.read(r.OLD / "retrieval_complete.json")["plan_sha256"]
        == receipt["plan_sha256"]
    )
    for p, h in receipt["result_hashes"].items():
        assert r.digest(r.ROOT / p) == h
    paths += [r.ROOT / p for p in receipt["result_hashes"]]
    hashes = {r.rel(p): r.digest(p) for p in paths}
    delivery = []
    for p in receipt["result_hashes"]:
        records = r.lines(r.ROOT / p)
        headers = [x["_manifest"] for x in records if "_manifest" in x]
        assert len(headers) == 1
        mf = headers[0]
        pool = "small" if "_small_" in p else "all"
        assert mf["subset"] in ("en_dev_eval_main", "en_dev_eval_sq3_types")
        contexts = r.read(r.OLD / f"retrieval_{pool}.json")
        actual = [x for x in records if "_manifest" not in x]
        assert len({x["item_id"] for x in actual}) == len(actual)
        expected = {
            key.split("/", 1)[1]
            for key in contexts
            if key.startswith(mf["subset"] + "/")
        }
        assert {x["item_id"] for x in actual} == expected
        for x in actual:
            assert x["retrieved"] == contexts[f"{mf['subset']}/{x['item_id']}"]
        delivery.append(
            {
                "file": p,
                "pool": pool,
                "subset": mf["subset"],
                "rows": len(actual),
                "all_delivered_contexts_match": True,
            }
        )
    assert len(delivery) == 12 and sum((x["rows"] for x in delivery)) == 4596
    a, ap, _ = r.embeddings(plan, train, "train")
    b, bp, _ = r.embeddings(plan, held, "heldout")
    boundary = [i for i, x in enumerate(rows) if abs(x["max_cosine"] - 0.95) <= 0.0005]
    checks = []
    scores = a[boundary].astype(np.float64) @ b.astype(np.float64).T
    for j, i in enumerate(boundary):
        arg = int(scores[j].argmax())
        mx = float(scores[j, arg])
        saved = rows[i]
        assert (
            arg == saved["heldout_text_index"]
            and (mx >= 0.95) == saved["threshold_excluded"]
        )
        checks.append(
            {
                "train_id": saved["train_id"],
                "saved_fp32": saved["max_cosine"],
                "float64_max": mx,
                "distance_to_095": abs(mx - 0.95),
                "decision_unchanged": True,
            }
        )

    def body(text):
        s = r.unicodedata.normalize(
            "NFKC", re.sub("(?<![\\w@])@[A-Za-z0-9_]+", "@user", text)
        ).casefold()
        return re.sub("\\s+", " ", s).strip(" \t\r\n\"'`«»“”‘’.,:;!?-–—…")

    bybody = defaultdict(list)
    for i, h in enumerate(held):
        bybody[body(h["text"])].append(i)
    candidates = []
    for t, x in zip(train, rows):
        if not x["proposed_excluded"] and body(t["text"]) in bybody:
            candidates.append(
                {
                    "train_id": t["id"],
                    "train_text": t["text"],
                    "max_cosine": x["max_cosine"],
                    "matches": [held[i] for i in bybody[body(t["text"])]],
                    "assessment": "Identical short body after handle normalization; target identity differs. Candidate for consistent conservative body-match exclusion, not proven duplicate provenance.",
                }
            )
    assert {x["train_id"] for x in candidates} == {
        x["train_id"] for x in base["retained_username_candidates"]
    }
    ordered = sorted(base["retained_review_band"], key=lambda x: -x["max_cosine"])
    assert len(ordered) == 83
    shared = {
        4: "Same three-word text; internal punctuation differs.",
        6: "Same headline; leading separator and trailing via differ.",
        10: "Same Netflix headline; quotes/spacing/link wrapper differ.",
        12: "Same headline/question; more/via link wrapper differs.",
        13: "Same short claim with people/ppl abbreviation, emoji and URL.",
        14: "Same short slogan; mention, contraction and small grammar differences.",
        15: "Same short slogan; ok/okay, apostrophe and punctuation differences.",
        16: "Same London-mayor headline; link wrapper/hashtag differs.",
        17: "Shared quoted body with truncation versus appended attribution/comment.",
        18: "Same short insult; dialect spelling expands to you are.",
        21: "Same London-mayor headline; link wrapper differs.",
        22: "Long shared copypasta; spelling, truncation and continuation differ.",
        23: "Same headline; retweet, quotes and link wrapper differ.",
        24: "Same text; retweet prefix differs.",
        26: "Same short statement; trailing via differs.",
        27: "Long shared body and numbered claims; opening sentence paraphrased.",
        29: "Shared repeated/lyric-like passage; different prefix and suffix.",
        30: "Same headline; separator and via differ.",
        31: "Very long identical body; appended emoji/contact request differs.",
        34: "Same publication/headline; trailing link wrapper differs.",
        35: "Very long identical body; closing sentence/emoji differs.",
        37: "Same long headline; brief framing prefix differs.",
        38: "Same headline; retweet/quotes/via wrapper differs.",
        44: "Same headline; hashtags and word spacing differ.",
        46: "Same distinctive rhetorical question paraphrased; shared template, not proven provenance.",
        48: "Same headline/body; retweet prefix differs.",
        49: "Same headline; separator/link wrapper differs.",
        50: "Same short statement; trailing URL marker differs.",
        51: "Same long body; inserted punctuation within words differs.",
        53: "Shared headline; abbreviated words, number spelling and framing comment differ.",
        56: "Same distinctive long statement; truncation differs.",
        59: "Same headline; short added for-no-reason clause/link wrapper differs.",
        60: "Same long statement; retweet and truncation differ.",
        61: "Same quoted headline; retweet/via/quotes differ.",
        62: "Shared slogan and hashtags; order and added hashtag differ.",
        65: "Same headline/quote; appended reaction differs.",
        66: "Same distinctive headline; inline hashtags and extra hashtag differ.",
        67: "Same headline; more/via link wrapper differs.",
        68: "Same headline; retweet and video/link wrapper differ.",
        69: "Same body; only trailing #tcot differs. Delivered in old full type prompts.",
        70: "Same headline; trailing source/link/hashtag wrapper differs.",
        71: "Same headline; punctuation and via spelling/hashtag differ.",
        74: "Same headline; joined words and trailing via differ.",
        75: "Same headline; retweet/separator prefix differs.",
        76: "Same headline; read-more link wrapper differs.",
        77: "Same headline; retweet prefix differs.",
        78: "Same short insult; contraction/colloquial spelling differs.",
        79: "Long shared copypasta; spelling substitutions and minor wording differ.",
        80: "Same distinctive body; extra ellipsis/hashtag and separator differ.",
    }
    altered = {
        1: "Adds you, changing direct address versus a group-level statement; short generic construction.",
        11: "Different sexual imperative/preposition; semantic similarity alone is insufficient for duplicate status.",
        19: "Common short evaluation with an added intensifier; referent/provenance unknown.",
        32: "Different short sexual/profane imperatives; wording and possible meaning differ.",
        40: "Targets change from ISIS/Mohammed to Islam/Allah; shared construction but substantive distinction.",
        55: "Same press-briefing template but different dates; potentially separate events.",
        58: "Shared violent idiom but kick/want-to-kick and framing differ; no duplicate provenance.",
        72: "Common short insult with different descriptive wording; no distinctive shared body.",
        83: "First-person versus third-person claim with different mention counts; referent changes.",
    }
    review = []
    for i, x in enumerate(ordered, 1):
        category = (
            "shared_text_or_template"
            if i in shared
            else (
                "meaning_or_referent_changed"
                if i in altered
                else "short_similar_wording"
            )
        )
        reason = shared.get(
            i,
            altered.get(
                i,
                "Short closely related wording; emoji, intensifier, mention, number or inflection differs. Text resemblance is real but shared-source provenance is not established.",
            ),
        )
        review.append(
            {
                **x,
                "review_order": i,
                "assessment": category,
                "reason": reason,
                "policy_state": "Reviewed candidate only; final additive exclusion policy not yet frozen.",
            }
        )
    assert len({x["train_id"] for x in review}) == 83
    assert not {x["train_id"] for x in candidates} & {x["train_id"] for x in review}
    aliases = {x["train_id"] for x in candidates}
    hi = {ref["id"]: h for h in held for ref in h["references"]}
    alias_delivery = {}
    for pool in ("small", "all"):
        found = []
        for key, ctx in r.read(r.OLD / f"retrieval_{pool}.json").items():
            subset, eid = key.split("/", 1)
            for rid in (x.removeprefix("ex-") for x in ctx["Labelled examples"]):
                if rid in aliases:
                    t = next((x for x in candidates if x["train_id"] == rid))
                    found.append(
                        dict(
                            subset=subset,
                            query_id=eid,
                            train_id=rid,
                            same_normalized_body_as_query=body(t["train_text"])
                            == body(hi[eid]["text"]),
                        )
                    )
        alias_delivery[pool] = found
    assert all((r.digest(r.ROOT / p) == h for p, h in base["input_hashes"].items()))
    assert all((r.digest(r.ROOT / p) == h for p, h in hashes.items()))
    result = dict(
        id="ASTRA-ENGLISH-OVERLAP-REVIEW-SUPPLEMENT-20260914-v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_results_sha256=r.digest(r.OUT),
        all_12_historical_runs_contexts_verified=delivery,
        historical_context_rows=4596,
        float64_boundary_checks=checks,
        manual_review=review,
        manual_category_counts=dict(Counter((x["assessment"] for x in review))),
        username_body_candidates=candidates,
        username_candidate_historical_delivery=alias_delivery,
        input_hashes=hashes,
        decision="Numerical audit and text review complete. Preserve 320-ID automatic proposal. Before final KB build, freeze a separate additive policy addressing shared-body/template and username candidates; do not silently lower the threshold or remove every semantically similar sentence. Apply consistently to small/full. No final cleaned English count is declared here.",
        limitations="Manual text review is not gold-label adjudication or proof of common provenance. Below-.94 shared-body families may remain; need declared text-wrapper/family screening during policy preparation. Previous type comparison has two directly delivered near-duplicate-body cases plus one related-text case; no score inflation is estimated. No test gold or new model calls used.",
    )
    with OUT.open("xb") as f:
        f.write(r.js(result))
    md = [
        "# English overlap audit: verified results and text review",
        "",
        "Completed 2026-09-14. Original experiment and classifier results are preserved. This is an overlap review, not an accuracy experiment.",
        "",
        "Evidence: `evidence/english_kb_results_20260914_v1.json` and `evidence/english_kb_review_supplement_20260914_v1.json`.",
        "",
        "## Verified result",
        "",
        "| Item | Count |",
        "|---|---:|",
        "| TRAIN / distinct held-out texts | 56,190 / 27,810 |",
        "| Cached chunks, all valid | 657 |",
        "| Pair similarities independently recomputed | 1,562,643,900 |",
        "| Rows at cosine >=0.95 / >=0.98 | 311 / 87 |",
        "| Previous exclusions carried forward | 81 |",
        "| Additional proposed exclusions / total | 239 / 320 |",
        "| Provisional full / small size | 55,870 / 147 |",
        "| Retained nearest pairs in [0.94,0.95), all read | 83 |",
        "| Additional username-body candidates | 6 |",
        "",
        "All nearest indices and .94/.95/.98 decisions reproduce. Maximum score difference is 4.77e-7; all ten .95-boundary rows also pass full-heldout float64 checks. The user run took 15.10 minutes and peak allocated GPU memory was 2.17 GiB. No rerun is needed.",
        "",
        "## Historical result implications",
        "",
        "Both old pools contain no >=0.95 matches to the two actual English sensitivity dev endpoints. The broader full dev/test rosters identify additional candidates. The newly flagged small example `implicit_hate-8484` matches full test, not the selected main/type dev endpoints. Availability in a pool is different from delivery to the matched query.",
        "",
        "Every delivered context was verified against all 4,596 saved item results in the 12 old runs. No retrieved example reaches .94 for its own main-gate query in either pool. In full-pool type contexts, `implicit_hate-14915 -> implicit_hate-14916` (0.949475) and `implicit_hate-3030 -> implicit_hate-3029` (0.941891) share near-duplicate headline bodies. `implicit_hate-2055 -> implicit_hate-4242` (0.944148) is a related construction with different targets. All occur in the three full repeats. Preserve the type scores as historical sensitivity evidence with this caveat; do not estimate score inflation or transfer the numbers to the final cleaned pool.",
        "",
        "## Review and next step",
        "",
        "55,870 is a threshold-only provisional count. Six additional handle-normalized short bodies and many below-threshold shared headlines/templates need an explicit additive policy. Similar short insults or changed targets/dates need separate consideration. This review does not silently exclude them or claim exhaustive semantic independence. No final KB has been built.",
        "",
        "The JSON supplement preserves both texts and references for every reviewed item. The table below records the judgment for each retained nearest pair.",
        "",
        "| TRAIN ID | Cosine | Assessment | Reason |",
        "|---|---:|---|---|",
    ]
    for x in review:
        md.append(
            f"| {x['train_id']} | {x['max_cosine']:.6f} | {x['assessment']} | {x['reason']} |"
        )
    md += [
        "",
        "Review categories: " + str(result["manual_category_counts"]) + ".",
        "",
        "Next bounded stage: declare and apply the extra text-family/body policy to both pools; verify any effect on the small control and encoder data alignment; then prepare the combined KB, isolated API logging fix and core comparison protocol. Priority 3 and LaTeX remain parked.",
        "",
    ]
    with TABLE.open("x", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(
        r.js(
            {
                "output": r.rel(OUT),
                "sha256": r.digest(OUT),
                "table": r.rel(TABLE),
                "table_sha256": r.digest(TABLE),
                "manual_counts": result["manual_category_counts"],
                "username_delivery": alias_delivery,
                "min_boundary_distance": min((x["distance_to_095"] for x in checks)),
                "historical_rows": 4596,
            }
        ).decode(),
        flush=True,
    )


if __name__ == "__main__":
    main()
