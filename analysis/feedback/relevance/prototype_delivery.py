import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from analysis.feedback.relevance.prototype_coverage import PROCESSED
from analysis.feedback.relevance.prototype_coverage import local_gold_sets
from src.hsrag.prompt import PromptContext, build_prompt, prompt_version
from src.hsrag.retrieve import Retriever

SUBSET = "en_dev_eval_sq3_types"
GOLD_COL = "hate_types"
ROLE = "held_out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--chroma", required=True)
    ap.add_argument(
        "--k-feedback",
        type=int,
        default=1,
        help="0 is the deliberate-break mode: the assert MUST "
        "fire, which is how the guard is verified",
    )
    ap.add_argument("--role", default=ROLE)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg_all = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    cfg = {**cfg_all["retrieval"], "k_examples_feedback": args.k_feedback}

    records = [
        json.loads(l)
        for l in Path(args.records).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    n_fb = sum(1 for r in records if r["kind"] == "feedback")
    print(f"\n{'=' * 74}\n{args.records}: {len(records)} records, " f"{n_fb} feedback")
    print(f"  chroma          {args.chroma}")
    print(f"  k_examples      {cfg['k_examples']}")
    print(
        f"  k_feedback      {args.k_feedback}"
        + ("   <- deliberate break" if args.k_feedback == 0 else "")
    )
    print(f"  prompt_version  {prompt_version()}")

    retriever = Retriever(
        chroma_path=Path(args.chroma),
        records_path=Path(args.records),
        model_name=cfg_all["kb"]["embedding_model"],
        cfg=cfg,
    )
    kb_version = retriever.col.metadata.get("kb_version")
    print(f"  kb_version      {kb_version}")

    df = pd.read_parquet(PROCESSED / f"{SUBSET}.parquet")
    items = df[df["sq3_role"] == args.role].reset_index(drop=True)
    if args.limit:
        items = items.head(args.limit)
    gold = local_gold_sets(items, GOLD_COL)
    print(f"  probing         {len(items)} {args.role} items\n")

    got, used, label_hit, ex_counts = 0, Counter(), 0, Counter()
    first_prompt = None
    for row in items.itertuples(index=False):
        res = retriever.retrieve(str(row.text), str(row.lang))
        ex_counts[len(res.examples)] += 1
        fb = getattr(res, "feedback", [])
        if not fb:
            continue
        got += 1
        for h in fb:
            used[h.id] += 1
            if set(h.meta.get("hate_types") or []) & gold[str(row.id)]:
                label_hit += 1
                break
        if first_prompt is None:
            ctx = PromptContext.from_retrieval(res)
            first_prompt = (
                str(row.id),
                build_prompt(str(row.text), str(row.lang), ctx),
            )

    print(
        f"  coverage    {got}/{len(items)} items received a correction "
        f"({got / len(items):.3f})"
    )
    print(
        f"  label hit   {label_hit}/{len(items)} carried the item's gold "
        f"label ({label_hit / len(items):.3f})"
    )

    fb_labels = [
        set(r["meta"].get("hate_types") or [])
        for r in records
        if r["kind"] == "feedback"
    ]
    if fb_labels:
        chance = sum(
            sum(1 for fl in fb_labels if fl & gold[str(row.id)]) / len(fb_labels)
            for row in items.itertuples(index=False)
        ) / len(items)
        print(
            f"  chance      {chance:.3f} if the correction were drawn at "
            f"random from the {len(fb_labels)} available"
        )
    print(
        f"  regular example slots per item: {dict(sorted(ex_counts.items()))}"
        f"  (feedback must ADD a slot, not take one)"
    )

    if used:
        top_id, top_n = used.most_common(1)[0]
        print(
            f"\n  spread      {len(used)} of {n_fb} corrections were "
            f"retrieved at least once"
        )
        print(
            f"              most-used {top_id} for {top_n} items "
            f"({top_n / max(got, 1):.3f} of covered items)"
        )
        for rec_id, n in used.most_common(5):
            print(f"                {rec_id:24} {n:4}")
        if top_n / max(got, 1) > 0.5:
            print(
                f"\n  WARNING one correction serves over half the covered "
                f"items. The round is closer to a one-record experiment "
                f"than a {n_fb}-record one."
            )

    if first_prompt:
        item_id, msgs = first_prompt
        user = msgs[1]["text"]
        block = [
            s
            for s in user.split("\n\n")
            if s.startswith("## Additional labelled examples")
        ]
        print(f"\n  rendered block for item {item_id}:")
        print(
            "    "
            + (
                "\n    ".join(block[0].splitlines()[:6])
                if block
                else "NOT PRESENT IN THE PROMPT TEXT"
            )
        )
        assert block, (
            "a correction was retrieved but does not appear in the rendered "
            "prompt. Retrieval and rendering are separate hops and this is "
            "the one that fails silently."
        )

    print()
    if got == 0:
        print(f"{'=' * 74}\nFAIL: no held-out item received a correction.")
        if args.k_feedback == 0:
            print(
                "  This is the expected failure at --k-feedback 0. The " "guard works."
            )
        else:
            print(
                "  Records were written but never retrieved. Check that "
                "they are in the index (kind='feedback', matching lang) "
                "and that k_examples_feedback reached the Retriever cfg."
            )
        print(f"{'=' * 74}\n")
        sys.exit(1)

    assert got == len(items), (
        f"only {got} of {len(items)} items received a correction. Retrieval "
        f"is unconditional at k>=1, so a partial coverage means records are "
        f"missing from the index, not ranking poorly."
    )
    print(
        f"{'=' * 74}\nPASS: corrections reach the prompt for every "
        f"{args.role} item.\n{'=' * 74}\n"
    )


if __name__ == "__main__":
    main()
