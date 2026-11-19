"Predict labels with saved classifier heads."

import argparse
import json
import time
from pathlib import Path
import hashlib
from src.hsrag.paths import PROJECT_ROOT
from training.encoder import taxonomy_labels

import numpy as np
import pandas as pd
import yaml

DATA = PROJECT_ROOT / "data/encoder"
MODELS = PROJECT_ROOT / "models/encoder"
RESULTS = PROJECT_ROOT / "results/encoder_predictions"
PROCESSED = PROJECT_ROOT / "data/processed"


ARM_SUBSETS = {
    "full": [
        "en_dev_eval_main",
        "en_dev_eval_targets",
        "en_dev_eval_types",
        "en_dev_eval_sq3_types",
        "en_dev_eval_severity",
        "en_test_eval",
        "en_test_eval_targets",
        "en_test_eval_types",
        "en_test_eval_severity",
        "en_hatecheck_eval",
    ],
    "kb": [
        "en_dev_eval_main",
        "en_dev_eval_targets",
        "en_dev_eval_types",
        "en_dev_eval_sq3_types",
        "en_dev_eval_severity",
        "en_test_eval",
        "en_test_eval_targets",
        "en_test_eval_types",
        "en_test_eval_severity",
        "en_hatecheck_eval",
    ],
    "de": ["de_dev_eval", "de_legal_dev_eval", "de_test_eval", "de_legal_test_eval"],
}
ARM_HEADS = {
    "full": ["hate", "target_group", "hate_type", "severity"],
    "kb": ["hate", "target_group", "hate_type", "severity"],
    "de": ["hate", "legal"],
}
MULTILABEL = {"target_group", "hate_type", "legal"}


EQUIVALENT_TAXONOMIES = {
    frozenset({"taxonomy-072387cd", "taxonomy-02dabf77"}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARM_HEADS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--tag",
        default=None,
        help="predict with a tagged arm, e.g. --tag nogrievance. "
        "Heads the tagged arm does not have are BORROWED "
        "from the untagged arm at the same seed.",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="first 8 items of the first subset, print the "
        "assembled Result, write nothing",
    )
    ap.add_argument("--models-dir", type=Path, default=MODELS)
    ap.add_argument("--processed-dir", type=Path, default=PROCESSED)
    ap.add_argument("--output-dir", type=Path, default=RESULTS)
    ap.add_argument(
        "--taxonomy", type=Path, default=PROJECT_ROOT / "config/taxonomy.yaml"
    )
    args = ap.parse_args()

    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BertForSequenceClassification,
        BertTokenizerFast,
    )
    from src.hsrag.classify import ItemResult, code_version, text_hash
    from src.hsrag.prompt import taxonomy_version
    from src.hsrag.schema import Result

    arm, seed = args.arm, args.seed
    slug = f"{arm}_{args.tag}" if args.tag else arm
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"\n{'=' * 78}\nENCODER PREDICT  arm={slug} seed={seed}  "
        f"device={dev}\n{'=' * 78}"
    )

    heads, missing, borrowed = {}, [], []
    for head in ARM_HEADS[arm]:
        d = args.models_dir / slug / head / f"seed{seed}"
        if not (d / "meta.json").exists() and args.tag:
            d = args.models_dir / arm / head / f"seed{seed}"
            if (d / "meta.json").exists():
                borrowed.append(head)
        if not (d / "meta.json").exists():
            missing.append(head)
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))

        now = "taxonomy-" + hashlib.sha256(args.taxonomy.read_bytes()).hexdigest()[:8]
        if meta["taxonomy_version"] != now:
            if frozenset({meta["taxonomy_version"], now}) in EQUIVALENT_TAXONOMIES:
                print(
                    f"  {head:14} taxonomy {meta['taxonomy_version']} != "
                    f"{now}: KNOWN EQUIVALENT PAIR, continuing. The label "
                    f"order check below still applies."
                )
            else:
                raise SystemExit(
                    f"{head}: checkpoint trained under "
                    f"{meta['taxonomy_version']}, taxonomy is now {now}. "
                    f"Refusing to predict - the label alignment is stale."
                )

        excl = meta.get("labels_excluded_from_output") or []
        want = [l for l in taxonomy_labels(head, args.taxonomy) if l not in excl]
        if meta["labels"] != want:
            raise SystemExit(
                f"{head}: label ORDER differs from taxonomy.yaml minus "
                f"{excl}. checkpoint {meta['labels']}, expected {want}"
            )
        if meta["lang"] == "de":
            tok = BertTokenizerFast.from_pretrained(str(d), local_files_only=True)
            mdl = BertForSequenceClassification.from_pretrained(
                str(d), local_files_only=True
            )
        else:
            tok = AutoTokenizer.from_pretrained(str(d), local_files_only=True)
            mdl = AutoModelForSequenceClassification.from_pretrained(
                str(d), local_files_only=True
            )
        heads[head] = {
            "tok": tok,
            "model": mdl.to(dev).eval(),
            "meta": meta,
            "labels": meta["labels"],
        }
        tagmark = (
            "  <- BORROWED from untagged"
            if head in borrowed
            else (
                f"  [{len(meta['labels'])} labels"
                + (
                    f", excl {meta['labels_excluded_from_output']}"
                    if meta.get("labels_excluded_from_output")
                    else ""
                )
                + "]"
            )
        )
        print(
            f"  {head:14} {meta['base_model']:22} "
            f"n_train {meta['n_train']:>6}  thr {meta['threshold']}  "
            f"val_macro {meta['val_macro_f1']:.4f}{tagmark}"
        )
    if not heads:
        raise SystemExit("no trained heads for this arm and seed")
    if missing:
        print(
            f"  MISSING HEADS: {missing} - written as schema defaults and "
            f"recorded in encoder_meta.heads"
        )

    kb_version = None
    if arm == "kb":
        from src.hsrag.kb import load_collection

        cfg = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
        col = load_collection(
            Path(cfg["kb"]["chroma_path"]), cfg["kb"]["embedding_model"]
        )
        kb_version = col.metadata.get("kb_version")
        print(f"  kb_version     {kb_version}  (Arm B trains on the KB)")

    @torch.no_grad()
    def infer(head: str, texts: list) -> np.ndarray:
        h = heads[head]
        out = []
        for i in range(0, len(texts), args.batch_size):
            enc = h["tok"](
                texts[i : i + args.batch_size],
                truncation=True,
                max_length=h["meta"]["max_seq_length"],
                padding=True,
                return_tensors="pt",
            ).to(dev)
            logit = h["model"](**enc).logits
            p = (
                torch.sigmoid(logit)
                if head in MULTILABEL
                else torch.softmax(logit, dim=-1)
            )
            out.append(p.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1))

    out_path = args.output_dir / f"encoder_{slug}_seed{seed}_live.jsonl"
    written, t0 = 0, time.time()
    lines = []

    for subset in ARM_SUBSETS[arm]:
        p = args.processed_dir / f"{subset}.parquet"
        if not p.exists():
            print(f"\n  {subset}: MISSING, skipped")
            continue
        df = pd.read_parquet(p)
        if args.dry_run:
            df = df.head(8)
        texts = [str(t) for t in df["text"]]
        ids = [str(i) for i in df["id"]]
        langs = [str(l) for l in df["lang"]]

        t_sub = time.time()
        probs = {h: infer(h, texts) for h in heads}
        infer_s = round(time.time() - t_sub, 2)
        print(f"\n  {subset:24} {len(df):>5} items  {infer_s:>6.2f}s")

        for i, item_id in enumerate(ids):
            fields, raw, agree = {}, {}, {}
            for head, h in heads.items():
                labels, thr = h["labels"], h["meta"]["threshold"]
                pv = probs[head][i]
                raw[head] = {l: round(float(x), 4) for l, x in zip(labels, pv)}
                if head == "hate":
                    fields["hate"] = bool(int(pv.argmax()))
                    agree["hate"] = float(pv.max())
                elif head == "severity":
                    fields["severity"] = labels[int(pv.argmax())]
                    agree["severity"] = float(pv.max())
                else:
                    fields[head] = [l for l, x in zip(labels, pv) if x >= thr]

                    agree[head] = float(np.mean(np.abs(pv - thr)))

            res = Result(
                reasoning=f"encoder {arm} seed{seed}",
                hate=fields.get("hate", False),
                target_group=fields.get("target_group", []),
                hate_type=fields.get("hate_type", []),
                severity=fields.get("severity"),
                legal=fields.get("legal", []),
            )
            r = ItemResult(
                item_id=item_id,
                text_hash=text_hash(texts[i]),
                lang=langs[i],
                arm=f"encoder_{slug}",
                timestamp=time.time(),
                workers=1,
                temperature=None,
                active_model="+".join(
                    f"{h}:{heads[h]['meta']['base_model']}" for h in heads
                ),
                prompt_version=None,
                taxonomy_version=taxonomy_version(),
                code_version=code_version(),
                kb_version=kb_version,
                retrieval_config=None,
                retrieved={},
                result=res.model_dump(),
                uncertain=False,
                n_valid=1,
                n_runs=1,
                agreement=agree,
                parse_failures=0,
                repairs=0,
                normalisations=0,
                gate_normalised=int(res.gate_normalised),
                encoder_meta={
                    "arm": slug,
                    "seed": seed,
                    "subset": subset,
                    "tag": args.tag,
                    "heads_borrowed": borrowed,
                    "labels_excluded": {
                        h: heads[h]["meta"].get("labels_excluded_from_output") or []
                        for h in heads
                    },
                    "heads": {
                        h: {
                            "base_model": heads[h]["meta"]["base_model"],
                            "threshold": heads[h]["meta"]["threshold"],
                            "n_train": heads[h]["meta"]["n_train"],
                            "train_seconds": heads[h]["meta"]["train_seconds"],
                            "train_script_sha": heads[h]["meta"].get(
                                "train_script_sha"
                            ),
                        }
                        for h in heads
                    },
                    "heads_missing": missing,
                    "infer_seconds_subset": infer_s,
                },
                raw_runs=[],
                run_predictions=[raw],
                latencies=[],
            )
            lines.append(r.to_json())
            written += 1

        if args.dry_run:
            print(
                f"\n  first assembled Result:\n    "
                + json.dumps(json.loads(lines[0])["result"], ensure_ascii=False)[:400]
            )
            print(
                f"    raw: "
                + json.dumps(
                    json.loads(lines[0])["run_predictions"][0], ensure_ascii=False
                )[:400]
            )
            break

    total = round(time.time() - t0, 1)
    print(f"\n{'-' * 78}")

    n_norm = sum(json.loads(l)["gate_normalised"] for l in lines)
    print(
        f"  gate_normalised on {n_norm}/{written} items "
        f"({n_norm / max(written, 1):.1%}) = the gate head's negative rate. "
        f"NOT head disagreement (7.10.5)."
    )
    if args.dry_run:
        print(f"\n  DRY RUN - nothing written.\n{'=' * 78}\n")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {out_path}: {written} records in {total}s")
    print(
        f"\n  Next: python -m scripts.check_encoder_reachability "
        f"--arm {arm} --seed {seed}"
    )
    print(
        f"  Hops 6, 7 and 8 should turn ok. Then --break-head to watch the "
        f"guard fail for the first time.\n{'=' * 78}\n"
    )


if __name__ == "__main__":
    main()
