"Train one classifier head."

import argparse
import hashlib
import json
import time
from pathlib import Path
from src.hsrag.paths import PROJECT_ROOT

import numpy as np
import pandas as pd
import yaml

DATA = PROJECT_ROOT / "data/encoder"
MODELS = PROJECT_ROOT / "models/encoder"
TAXONOMY = PROJECT_ROOT / "config/taxonomy.yaml"

BASE_MODEL = {"en": "roberta-base", "de": "deepset/gbert-base"}
GOLD_COL = {
    "hate": "gate",
    "target_group": "target_groups",
    "hate_type": "hate_types",
    "severity": "severity",
    "legal": "legal",
}


def clean(val):
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if hasattr(val, "tolist"):
        return val.tolist()
    return val


def taxonomy_labels(dim: str, taxonomy_path=None) -> list:
    tax = yaml.safe_load(Path(taxonomy_path or TAXONOMY).read_text(encoding="utf-8"))
    d = tax["dimensions"][dim]
    if d["type"] == "binary":
        return ["false", "true"]
    labels = d["labels"]
    return list(labels) if isinstance(labels, dict) else list(labels)


def encode_targets(values, kind: str, labels: list):
    "Gold column -> model targets, in taxonomy label ORDER."
    idx = {l: i for i, l in enumerate(labels)}
    if kind == "binary":
        return np.array([int(bool(v)) for v in values], dtype=np.int64)
    if kind == "ordinal":
        return np.array([idx[str(v)] for v in values], dtype=np.int64)
    Y = np.zeros((len(values), len(labels)), dtype=np.float32)
    for i, v in enumerate(values):
        for l in clean(v) or []:

            if str(l) not in idx:
                raise ValueError(
                    f"label {l!r} is not in the taxonomy's {labels}. The "
                    f"taxonomy changed after make_encoder_data ran; rebuild "
                    f"the head's data."
                )
            Y[i, idx[str(l)]] = 1.0
    return Y


def macro_f1(y_true, y_pred, multilabel: bool, n_labels: int) -> float:

    f1s = []
    for k in range(n_labels):
        if multilabel:
            t, p = y_true[:, k] > 0.5, y_pred[:, k] > 0.5
        else:
            t, p = y_true == k, y_pred == k
        tp = int((t & p).sum())
        fp = int((~t & p).sum())
        fn = int((t & ~p).sum())
        if tp + fn == 0:
            continue
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn)
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["full", "de", "kb"])
    ap.add_argument("--head", required=True, choices=sorted(GOLD_COL))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seq-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="default 5 (20 for arm kb, which has no usable " "validation split)",
    )
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument(
        "--tag",
        default=None,
        help="read data/encoder/{arm}_{tag}/ and write "
        "models/encoder/{arm}_{tag}/. For 7.10: "
        "--tag nogrievance",
    )
    ap.add_argument(
        "--bf16",
        action="store_true",
        help="the 3070 Ti is Ampere so bf16 works and is more "
        "stable than fp16; fp16 is the default because the "
        "build guide specifies it",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a two-step training check without exporting a model. "
        "Exercises tokenizer, binarizer, collator and loss "
        "before any GPU time is committed.",
    )
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--output-dir", type=Path, default=MODELS)
    ap.add_argument("--taxonomy", type=Path, default=TAXONOMY)
    ap.add_argument("--model", help="Model name or original local checkpoint path")
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BertConfig,
        BertForSequenceClassification,
        BertTokenizerFast,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )
    from src.hsrag.classify import code_version
    from src.hsrag.prompt import taxonomy_version

    head, arm = args.head, args.arm

    slug = f"{arm}_{args.tag}" if args.tag else arm
    d = args.data_dir / slug
    meta_in = json.loads((d / f"{head}_meta.json").read_text(encoding="utf-8"))
    lang, kind = meta_in["lang"], meta_in["kind"]

    excl = meta_in.get("labels_excluded_from_output") or []
    labels = [l for l in taxonomy_labels(head, args.taxonomy) if l not in excl]
    assert labels == meta_in["labels"], (
        f"label order changed since make_encoder_data ran: data says "
        f"{meta_in['labels']}, taxonomy minus {excl} says {labels}. "
        f"Rebuild the head."
    )
    multilabel = kind == "multilabel"
    base = args.model or BASE_MODEL[lang]
    epochs = args.epochs if args.epochs else (20 if arm == "kb" else 5)

    print(f"\n{'=' * 78}\nTRAIN  arm={arm} head={head} seed={args.seed}")
    print(f"  base model      {base}")
    print(f"  kind            {kind}, {len(labels)} labels in taxonomy order")
    print(f"  labels          {labels}")
    print(f"  cuda            {torch.cuda.is_available()}")
    print(f"{'=' * 78}")

    if lang == "de":
        ConfigCls = BertConfig
        ModelCls = BertForSequenceClassification
        tok = BertTokenizerFast.from_pretrained(
            base, local_files_only=args.local_files_only
        )
    else:
        ConfigCls = AutoConfig
        ModelCls = AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(
            base, local_files_only=args.local_files_only
        )
    tok_class = type(tok).__name__
    print(f"  config/model    {ConfigCls.__name__} / {ModelCls.__name__}")
    print(f"  tokenizer       {tok_class}")

    gold_col = GOLD_COL[head]
    frames = {}
    for split in ("train", "val"):
        df = pd.read_parquet(d / f"{head}_{split}.parquet")
        texts = [str(t) for t in df["text"]]
        enc = tok(texts, truncation=True, max_length=args.max_seq_length, padding=False)
        y = encode_targets(list(df[gold_col]), kind, labels)
        n_trunc = sum(1 for ids in enc["input_ids"] if len(ids) >= args.max_seq_length)
        print(
            f"  {split:5} {len(df):>6} rows, {n_trunc} truncated at "
            f"{args.max_seq_length} ({n_trunc / max(len(df), 1):.2%})"
        )
        frames[split] = (enc, y, n_trunc, len(df))

    class DS(torch.utils.data.Dataset):
        def __init__(self, enc, y):
            self.enc, self.y = enc, y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            out = {k: self.enc[k][i] for k in ("input_ids", "attention_mask")}
            out["labels"] = self.y[i]
            return out

    from transformers import DataCollatorWithPadding

    collate = DataCollatorWithPadding(tok)

    cfg = ConfigCls.from_pretrained(
        base,
        local_files_only=args.local_files_only,
        num_labels=len(labels),
        problem_type=(
            "multi_label_classification"
            if multilabel
            else "single_label_classification"
        ),
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = ModelCls.from_pretrained(
        base, config=cfg, local_files_only=args.local_files_only
    )

    def compute_metrics(p):
        logits = p.predictions
        y = p.label_ids
        if multilabel:
            pred = (1 / (1 + np.exp(-logits)) >= args.threshold).astype(float)
        else:
            pred = logits.argmax(axis=-1)
        return {"macro_f1": macro_f1(y, pred, multilabel, len(labels))}

    out_dir = args.output_dir / slug / head / f"seed{args.seed}"

    early = arm != "kb"
    targs = TrainingArguments(
        output_dir=str(out_dir / "_hf"),
        seed=args.seed,
        data_seed=args.seed,
        num_train_epochs=2 if args.dry_run else epochs,
        max_steps=2 if args.dry_run else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=not args.bf16,
        bf16=args.bf16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=early,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        report_to=[],
        disable_tqdm=False,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=DS(*frames["train"][:2]),
        eval_dataset=DS(*frames["val"][:2]),
        data_collator=collate,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if early else [],
    )

    t0 = time.time()
    trainer.train()
    train_seconds = round(time.time() - t0, 1)
    ev = trainer.evaluate()
    print(f"\n  val macro_f1    {ev.get('eval_macro_f1'):.4f}")
    print(f"  train_seconds   {train_seconds}")

    if args.dry_run:
        print(
            f"\n  Two-step check complete; no final model exported. The tokenizer, "
            f"binarizer, collator, loss and metric all ran.\n{'=' * 78}\n"
        )
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    meta = {
        "head": head,
        "dimension": head,
        "kind": kind,
        "labels": labels,
        "taxonomy_version": "taxonomy-"
        + hashlib.sha256(args.taxonomy.read_bytes()).hexdigest()[:8],
        "code_version": code_version(),
        "seed": args.seed,
        "base_model": base,
        "tokenizer_class": tok_class,
        "tokenizer_is_fast": bool(getattr(tok, "is_fast", False)),
        "config_class": ConfigCls.__name__,
        "model_class": ModelCls.__name__,
        "transformers_version": __import__("transformers").__version__,
        "train_script_sha": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:8],
        "warmup_ratio": targs.warmup_ratio,
        "weight_decay": targs.weight_decay,
        "precision": "bf16" if targs.bf16 else ("fp16" if targs.fp16 else "fp32"),
        "max_seq_length": args.max_seq_length,
        "truncated_train": frames["train"][2],
        "truncated_val": frames["val"][2],
        "threshold": args.threshold,
        "n_train": frames["train"][3],
        "n_val": frames["val"][3],
        "leakage_excluded": meta_in["leakage_excluded"],
        "leakage_threshold": meta_in["leakage_threshold"],
        "train_seconds": train_seconds,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "early_stopping": early,
        "val_macro_f1": float(ev.get("eval_macro_f1", 0.0)),
        "class_weighting": False,
        "resampling": False,
        "arm": arm,
        "lang": lang,
        "tag": args.tag,
        "labels_excluded_from_output": excl,
        "n_dropped_excluded_label": meta_in.get("n_dropped_excluded_label"),
        "n_stripped_excluded_label": meta_in.get("n_stripped_excluded_label"),
        "support_train_plus_val": meta_in["support_train_plus_val"],
        "labels_below_min_support": meta_in["labels_below_min_support"],
        "note": (
            "severity is trained as plain 3-class; ordinality is not "
            "exploited, unlike the LLM's ordinal-median vote"
            if head == "severity"
            else ""
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  wrote           {out_dir}")


if __name__ == "__main__":
    main()
