import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

from src.hsrag.classify import ItemResult, text_hash
from src.hsrag.client import make_client
from src.hsrag.parse import run_once
from src.hsrag.prompt import (
    PromptContext,
    build_prompt,
    prompt_version,
    taxonomy_version,
)
from src.hsrag.retrieve import Hit, Retriever

from src.hsrag.vote import aggregate

RESULTS = Path("experiments/results")
PROCESSED = Path("data/processed")
SUBSET = "en_hatecheck_eval"
BLOCK = "guideline_covered"


MAPPING = {
    "counter_quote_nh": (
        "guide-counter-speech",
        "guide-reclaimed-slurs",
        "guide-quoted-slurs",
    ),
    "counter_ref_nh": (
        "guide-counter-speech",
        "guide-reclaimed-slurs",
        "guide-quoted-slurs",
    ),
    "negate_neg_nh": (
        "guide-negation-and-support",
        "guide-quoted-slurs",
        "guide-reclaimed-slurs",
    ),
    "profanity_nh": (
        "guide-profanity-without-target",
        "guide-quoted-slurs",
        "guide-reclaimed-slurs",
    ),
    "slur_reclaimed_nh": (
        "guide-reclaimed-slurs",
        "guide-negation-and-support",
        "guide-quoted-slurs",
    ),
    "target_indiv_nh": (
        "guide-group-vs-individual",
        "guide-quoted-slurs",
        "guide-reclaimed-slurs",
    ),
}


def kb_guidelines() -> dict:

    out = {}
    for line in Path("kb/records.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["kind"] == "guideline":
            out[r["id"]] = r
    return out


def as_hit(rec: dict) -> Hit:

    return Hit(
        id=rec["id"],
        kind=rec["kind"],
        text=rec["text"],
        meta={
            "kind": rec["kind"],
            "lang": rec["lang"],
            "source": rec["source"],
            "dimension": rec.get("dimension"),
            "label": rec.get("label"),
        },
        score=0.0,
        via="oracle",
    )


def pick(functionality: str, condition: str, guides: dict) -> list:

    if condition == "N":
        return []
    intended, b_rule, filler = MAPPING[functionality]
    first = intended if condition == "A" else b_rule
    assert first != filler, (
        f"{functionality}: slot 1 and filler are the same record ({first}); "
        f"A and B would then differ in zero slots"
    )
    return [guides[first], guides[filler]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["A", "B", "N"], required=True)
    ap.add_argument(
        "--rep",
        type=int,
        default=1,
        help="replicate index. The injected records are chosen "
        "DETERMINISTICALLY from the item's functionality, so "
        "they are identical across replicates and only "
        "decoding varies.",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="MockClient, no API calls. Renders the first item's "
        "full prompt for inspection.",
    )
    ap.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="skip the confirmation prompt; for queued runs",
    )
    args = ap.parse_args()

    cfg_all = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    api, ccfg, kb = cfg_all["api"], cfg_all["classify"], cfg_all["kb"]
    n_votes = ccfg["n_votes_dev"]
    pinned = "Qwen/Qwen3-VL-32B-Instruct"
    provider = "mock" if args.dry_run else "peasec"

    df = pd.read_parquet(PROCESSED / f"{SUBSET}.parquet")
    meta = {str(r.id): json.loads(r.raw) for r in df.itertuples(index=False)}
    items = [r for r in df.itertuples(index=False) if meta[str(r.id)]["block"] == BLOCK]
    assert len(items) == 693, f"expected 693 guideline_covered items, got {len(items)}"

    guides = kb_guidelines()
    for f, (i, b, fl) in MAPPING.items():
        for gid in (i, b, fl):
            assert gid in guides, f"{gid} not in kb/records.jsonl"

    retriever = Retriever(
        chroma_path=Path(kb["chroma_path"]),
        records_path=Path(kb["records_path"]),
        model_name=kb["embedding_model"],
        cfg={**cfg_all["retrieval"], "k_guidelines": 0},
    )
    kb_version = retriever.col.metadata.get("kb_version")

    tag = "mock" if args.dry_run else "live"
    suffix = "" if args.rep == 1 else f"_rep{args.rep}"
    out_path = RESULTS / f"guideoracle_{args.condition}{suffix}_{tag}.jsonl"

    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["item_id"])
        print(f"resuming: {len(done)} items already complete")

    todo = [r for r in items if str(r.id) not in done]
    if args.dry_run:
        todo = todo[:8]

    by_func = {}
    for r in items:
        by_func.setdefault(meta[str(r.id)]["functionality"], 0)
        by_func[meta[str(r.id)]["functionality"]] += 1

    print(f"\n{'=' * 74}\nGUIDELINE ORACLE, condition {args.condition}")
    label = {
        "A": "A = intended rule + filler (oracle upper bound)",
        "B": "B = different decisive-false rule + same filler",
        "N": "N = no guidelines at all, k_guidelines = 0",
    }
    print(f"  {label[args.condition]}")
    print(f"  subset {SUBSET}, block {BLOCK}, {len(items)} items")
    for f in sorted(by_func, key=lambda k: -by_func[k]):
        i, b, fl = MAPPING[f]
        shown = {"A": (i, fl), "B": (b, fl), "N": ("-", "-")}[args.condition]
        print(f"    {f:20s} {by_func[f]:4d}   {shown[0]:32s} {shown[1]}")
    print(f"  kb {kb_version}   prompt {prompt_version()}   model {pinned}")
    print(f"  T {api['temperature']}   n_votes {n_votes}   " f"workers {args.workers}")
    print(f"  {len(todo)} items x {n_votes} votes = {len(todo) * n_votes} calls")
    if not args.dry_run:
        est = len(todo) * n_votes * 1.2 / 60
        print(f"  ~{est:.0f} min at 1.2 s/call measured")
        if not args.yes and input("proceed? [y/N] ").strip().lower() != "y":
            return

    lock = threading.Lock()
    local = threading.local()

    def client():
        if not hasattr(local, "c"):
            if provider == "mock":
                local.c = make_client("mock")
            else:
                p = api[provider]
                local.c = make_client(
                    provider,
                    [pinned],
                    temperature=api["temperature"],
                    timeout=api["timeout_seconds"],
                    max_tokens=api.get("max_tokens"),
                    allow_fallback=False,
                    url_env=p["url_env"],
                    token_env=p["token_env"],
                )
        return local.c

    def work(row):
        iid, text, lang = str(row.id), str(row.text), str(row.lang)
        func = meta[iid]["functionality"]
        injected = pick(func, args.condition, guides)

        with lock:
            res = retriever.retrieve(text, lang)
        assert not res.guidelines, "k_guidelines=0 did not empty the bucket"
        res.guidelines = [as_hit(g) for g in injected]
        ctx = PromptContext.from_retrieval(res)
        assert len(ctx.guidelines) == len(injected)
        messages = build_prompt(text, lang, ctx)

        out = ItemResult(
            item_id=iid,
            text_hash=text_hash(text),
            lang=lang,
            arm="rag",
            prompt_version=prompt_version(),
            taxonomy_version=taxonomy_version(),
            kb_version=kb_version,
            retrieval_config=dict(retriever.cfg),
            retrieved={
                "definitions": [h.id for h in ctx.definitions],
                "guidelines": [h.id for h in ctx.guidelines],
                **{name: [h.id for h in hits] for name, hits in ctx.example_groups},
                "injected": [g["id"] for g in injected],
                "functionality": func,
                "condition": args.condition,
            },
            n_runs=n_votes,
            timestamp=time.time(),
            workers=args.workers,
            temperature=api["temperature"],
        )

        if args.dry_run and iid == str(todo[0].id):
            print(
                f"\n{'-' * 74}\nFIRST PROMPT, condition {args.condition}, "
                f"{func}\n{'-' * 74}"
            )
            print(messages[1]["text"])
            print(f"{'-' * 74}\n")

        runs = []
        for _ in range(n_votes):
            r = run_once(client(), messages, max_repairs=ccfg["max_repair_retries"])
            runs.append(r)
            out.raw_runs.append(r.raw_outputs)
            out.latencies.append(round(r.latency_s, 2))
            out.repairs += r.repairs
            out.normalisations += r.normalisations
            out.gate_normalised += int(r.gate_normalised)
            out.parse_failures += int(r.parse_failure)
            out.run_predictions.append(
                r.result.model_dump() if r.result is not None else None
            )

        if not args.dry_run and client().active_model != pinned:
            raise RuntimeError(f"expected {pinned}, got {client().active_model}")
        out.active_model = client().active_model

        v = aggregate(runs)
        out.result = v.result.model_dump() if v.result is not None else None
        out.uncertain, out.n_valid, out.agreement = (
            v.uncertain,
            v.n_valid,
            v.agreement,
        )
        return out

    t0, ok, failed = time.time(), 0, 0
    with open(out_path, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(work, todo):
                with lock:
                    f.write(r.to_json() + "\n")
                    f.flush()
                    ok += 1
                    if ok % 50 == 0 or ok == len(todo):
                        el = time.time() - t0
                        print(
                            f"  {ok}/{len(todo)}  {el / 60:.1f} min "
                            f"elapsed, ~{el / ok * (len(todo) - ok) / 60:.1f}"
                            f" min left"
                        )

    print(f"\n  wrote {out_path}  ({ok} records, {failed} failed)")
    print(f"{'=' * 74}\n")


if __name__ == "__main__":
    main()
