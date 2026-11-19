import argparse
import json
import time
from collections import Counter
from pathlib import Path
from src.hsrag.paths import project_path

import pandas as pd
import yaml
import threading
from concurrent.futures import ThreadPoolExecutor

from src.hsrag.metrics import score_all
from scripts.score_predictions import report
from src.hsrag.classify import Arms, classify
from src.hsrag.client import make_client
from src.hsrag.retrieve import Retriever

RESULTS = project_path("experiments/results")
ARMS = ["zero_shot", "few_shot", "rag"]
PROCESSED = project_path("data/processed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        help="take run settings from a manifest; " "overrides the CLI defaults below",
    )
    ap.add_argument("--subset", default="en_dev_eval_main")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="skip the confirmation prompt; for queued runs",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel API calls; each worker gets its own client",
    )
    ap.add_argument("--dry-run", action="store_true", help="MockClient, no API")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg_all = yaml.safe_load(
        project_path("config/config.yaml").read_text(encoding="utf-8")
    )
    kb, ccfg, api = cfg_all["kb"], cfg_all["classify"], cfg_all["api"]

    mf = None
    if args.manifest:
        from src.hsrag.manifest import load as load_manifest, resolve

        mf = load_manifest(args.manifest)

        _stem = Path(args.manifest).stem
        assert mf.name == _stem, (
            f"manifest name/filename mismatch: the file is {_stem}.yaml but "
            f"`name:` says {mf.name!r}. Results would go to "
            f"{mf.name}_live.jsonl and an existing run of that name would be "
            f"RESUMED rather than replicated. Fix the `name:` field."
        )
        args.subset, args.arms, args.workers = mf.subset, mf.arms, mf.workers

        args.n = (
            mf.limit
            if mf.limit
            else len(pd.read_parquet(PROCESSED / f"{mf.subset}.parquet"))
        )
        ccfg = {
            **ccfg,
            "n_votes_dev": mf.n_votes,
            "max_repair_retries": mf.max_repair_retries,
            "pinned_model": mf.pinned_model or ccfg["pinned_model"],
        }

        api = {
            **api,
            "temperature": (
                mf.temperature if mf.temperature is not None else api["temperature"]
            ),
        }

    df = pd.read_parquet(
        project_path("data/processed") / f"{args.subset}.parquet"
    ).head(args.n)
    tag = "mock" if args.dry_run else "live"

    stem = f"{mf.name}_{tag}" if mf else f"slice1_{args.subset}_{tag}"
    out_path = Path(args.out or RESULTS / f"{stem}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if "_manifest" in r:
                    continue
                done.add((r["item_id"], r["arm"]))
        print(f"resuming: {len(done)} (item, arm) pairs already complete")

    if not args.score_only:

        chroma_dir = (
            Path(mf.chroma_path) if (mf and mf.chroma_path) else Path(kb["chroma_path"])
        )
        records_file = (
            Path(mf.records_path)
            if (mf and mf.records_path)
            else Path(kb["records_path"])
        )
        retriever = Retriever(
            chroma_path=chroma_dir,
            records_path=records_file,
            model_name=kb["embedding_model"],
            cfg={
                **cfg_all["retrieval"],
                **(dict(mf.retrieval) if (mf and mf.retrieval) else {}),
            },
        )
        arms = Arms(
            retriever=retriever,
            records_path=records_file,
            k_examples=cfg_all["retrieval"]["k_examples"],
            seed=ccfg["fewshot_seed"],
        )
        kb_version = retriever.col.metadata.get("kb_version")
        if mf and mf.chroma_path:
            print(f"  KB OVERRIDE: {chroma_dir} -> kb_version {kb_version}")

        if mf is not None and not out_path.exists():

            out_path.write_text(
                resolve(mf, retriever).header() + "\n", encoding="utf-8"
            )

        n_votes = ccfg["n_votes_dev"]

        todo = [
            (row, arm)
            for row in df.itertuples(index=False)
            for arm in args.arms
            if (str(row.id), arm) not in done
        ]
        calls = len(todo) * n_votes

        provider = "mock" if args.dry_run else (mf.provider if mf else ccfg["provider"])
        if provider == "mock":
            wall_s = 0.01
        elif provider == "peasec":

            wall_s = 8.0 if args.workers == 1 else 1.2
        else:
            wall_s = 30.0 if args.workers == 1 else 17.0
        est = calls * wall_s / 60
        print(f"\n{len(todo)} item-arm pairs x {n_votes} votes = {calls} calls")
        print(
            f"  provider {provider}, {args.workers} worker(s), "
            f"~{wall_s:g} s/call measured -> ~{est:.0f} min"
        )

        if not args.dry_run and not args.yes:
            if input("proceed? [y/N] ").strip().lower() != "y":
                return

        local = threading.local()

        def get_client():
            if not hasattr(local, "client"):
                if provider == "mock":
                    local.client = make_client("mock")
                else:
                    p = api[provider]
                    local.client = make_client(
                        provider,
                        [ccfg["pinned_model"]],
                        temperature=api["temperature"],
                        timeout=api["timeout_seconds"],
                        max_tokens=api.get("max_tokens"),
                        allow_fallback=False,
                        url_env=p["url_env"],
                        token_env=p["token_env"],
                    )
            return local.client

        def work(job):
            row, arm = job
            try:
                return classify(
                    item_id=str(row.id),
                    text=row.text,
                    lang=row.lang,
                    arm=arm,
                    client=get_client(),
                    arms=arms,
                    n_votes=n_votes,
                    max_repairs=ccfg["max_repair_retries"],
                    pinned_model=None if args.dry_run else ccfg["pinned_model"],
                    kb_version=kb_version,
                    workers=args.workers,
                    temperature=api["temperature"],
                )
            except RuntimeError as e:

                return (row.id, arm, str(e))

        write_lock = threading.Lock()
        state = {"ok": 0, "failed": 0, "consecutive": 0}
        t0 = time.time()

        with open(out_path, "a", encoding="utf-8") as f:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for res in ex.map(work, todo):
                    with write_lock:
                        if isinstance(res, tuple):
                            state["failed"] += 1
                            state["consecutive"] += 1
                            print(
                                f"  FAILED {res[0]} [{res[1]}] "
                                f"({state['consecutive']} in a row): "
                                f"{res[2][:120]}"
                            )
                        else:
                            state["consecutive"] = 0
                            state["ok"] += 1
                            f.write(res.to_json() + "\n")
                            f.flush()
                        n = state["ok"] + state["failed"]
                        if n % 10 == 0 or n == len(todo):
                            el = time.time() - t0
                            rate = el / max(state["ok"], 1) / n_votes * args.workers
                            print(
                                f"  {n}/{len(todo)}  {el / 60:.1f} min elapsed, "
                                f"~{el / n * (len(todo) - n) / 60:.1f} min left "
                                f"({rate:.1f} s/call, {state['failed']} failed)"
                            )

        if state["failed"]:
            print(
                f"\n{state['failed']} pairs failed and were not written. "
                f"Rerun the same command to retry only those."
            )

    results = [
        r
        for r in (
            json.loads(l)
            for l in out_path.read_text(encoding="utf-8").splitlines()
            if l.strip()
        )
        if "_manifest" not in r
    ]
    scores = score_all(df, results, gate_mapping=(mf.gate_mapping if mf else "both"))
    report(scores)

    json_path = out_path.with_suffix(".metrics.json")
    json_path.write_text(json.dumps(scores, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out_path}\nwrote {json_path}")

    if not args.dry_run:
        models = Counter(r["active_model"] for r in results)
        print(f"models that answered: {dict(models)}")
        assert set(models) == {ccfg["pinned_model"]}, "unattributable results"


if __name__ == "__main__":
    main()
