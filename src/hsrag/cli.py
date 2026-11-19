"""Commands for classification, preparation and scoring."""

import argparse
import importlib
import json
import sys
from pathlib import Path

import yaml

from src.hsrag.paths import project_path

PREPARATION = {
    "splits": "scripts.make_splits",
    "legal": "scripts.make_botox_splits",
    "subsets": "scripts.make_eval_subsets",
    "knowledge": "scripts.build_kb_records",
    "hatecheck": "scripts.make_hatecheck_subset",
    "encoder": "scripts.make_encoder_data",
}


def _inputs(args):
    if args.text is not None:
        if args.lang is None:
            raise ValueError("Use --lang en or --lang de with --text.")
        return [{"id": args.id or "input", "text": args.text, "lang": args.lang}]
    rows = []
    for line_number, line in enumerate(
        args.input.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise ValueError(f"Input line {line_number} needs a text string.")
        if row.get("lang") not in ("en", "de"):
            raise ValueError(f"Input line {line_number} needs lang=en or lang=de.")
        rows.append(
            {
                "id": str(row.get("id", line_number)),
                "text": row["text"],
                "lang": row["lang"],
            }
        )
    if not rows:
        raise ValueError("The input file contains no records.")
    return rows


def classify_command(args):
    from src.hsrag.classify import Arms, classify
    from src.hsrag.client import make_client

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    settings, api, knowledge = config["classify"], config["api"], config["kb"]
    provider = args.provider or settings["provider"]
    votes = args.votes if args.votes is not None else settings["n_votes_dev"]
    if votes < 1:
        raise ValueError("--votes must be positive.")
    rows = _inputs(args)
    if args.output and args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    records = args.records or project_path(knowledge["records_path"])
    index = args.index or project_path(knowledge["chroma_path"])
    retriever = None
    if args.arm != "zero_shot":
        if not records.is_file():
            raise FileNotFoundError(f"Knowledge records not found: {records}")
        if args.arm == "rag":
            if not index.is_dir():
                raise FileNotFoundError(f"Retrieval index not found: {index}")
            from src.hsrag.retrieve import Retriever

            retriever = Retriever(
                index, records, knowledge["embedding_model"], config["retrieval"]
            )
    arms = Arms(
        retriever=retriever,
        records_path=records if args.arm == "few_shot" else None,
        k_examples=config["retrieval"]["k_examples"],
        seed=settings["fewshot_seed"],
    )
    if args.arm == "few_shot" and any(
        not arms.context("few_shot", r["text"], r["lang"]).example_groups for r in rows
    ):
        raise ValueError(
            "The records contain no eligible examples for an input language."
        )
    if provider == "mock":
        client, model = make_client("mock"), None
    else:
        provider_config = api[provider]
        model = args.model or (
            settings["pinned_model"]
            if provider == settings["provider"]
            else provider_config["default_model"]
        )
        client = make_client(
            provider,
            [model],
            temperature=api["temperature"],
            timeout=api["timeout_seconds"],
            max_tokens=api.get("max_tokens"),
            allow_fallback=False,
            url_env=provider_config["url_env"],
            token_env=provider_config["token_env"],
        )
    output = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output = args.output.open("x", encoding="utf-8", newline="\n")
    try:
        for row in rows:
            prediction = classify(
                row["id"],
                row["text"],
                row["lang"],
                args.arm,
                client,
                arms,
                n_votes=votes,
                max_repairs=settings["max_repair_retries"],
                pinned_model=model,
                temperature=api["temperature"],
                kb_version=(
                    retriever.col.metadata.get("kb_version") if retriever else None
                ),
            )
            if output:
                output.write(prediction.to_json() + "\n")
                output.flush()
            value = (
                json.loads(prediction.to_json()) if args.details else prediction.result
            )
            print(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    indent=2 if args.text is not None else None,
                )
            )
    finally:
        if output:
            output.close()
    return 0


def parser():
    ap = argparse.ArgumentParser(
        prog="hsrag", description="Hate speech classification and data preparation."
    )
    commands = ap.add_subparsers(dest="command", required=True)
    classify = commands.add_parser("classify", help="classify a text or a JSONL file")
    inputs = classify.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text")
    inputs.add_argument(
        "--input", type=Path, help="JSONL records with text, lang and optional id"
    )
    classify.add_argument("--lang", choices=("en", "de"))
    classify.add_argument("--id")
    classify.add_argument("--provider", choices=("peasec", "tudagpt", "mock"))
    classify.add_argument("--model")
    classify.add_argument(
        "--arm", choices=("zero_shot", "few_shot", "rag"), default="zero_shot"
    )
    classify.add_argument("--votes", type=int)
    classify.add_argument("--records", type=Path)
    classify.add_argument("--index", type=Path)
    classify.add_argument(
        "--config", type=Path, default=project_path("config/config.yaml")
    )
    classify.add_argument(
        "--output", type=Path, help="save full prediction records to a new JSONL file"
    )
    classify.add_argument(
        "--details",
        action="store_true",
        help="also print response and provenance fields",
    )
    prepare = commands.add_parser(
        "prepare", help="prepare corpus splits, subsets or knowledge records"
    )
    prepare.add_argument("action", choices=tuple(PREPARATION))
    prepare.add_argument("arguments", nargs=argparse.REMAINDER)
    commands.add_parser(
        "index", help="build a retrieval index; use hsrag index --help for options"
    )
    commands.add_parser(
        "score", help="score saved predictions; use hsrag score --help for options"
    )
    return ap


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv and argv[0] == "index":
            return importlib.import_module("scripts.build_kb").main(argv[1:]) or 0
        if argv and argv[0] == "score":
            return (
                importlib.import_module("scripts.score_predictions").main(argv[1:]) or 0
            )
        if len(argv) >= 2 and argv[0] == "prepare" and argv[1] in PREPARATION:
            return importlib.import_module(PREPARATION[argv[1]]).main(argv[2:]) or 0
        args = parser().parse_args(argv)
        return classify_command(args)
    except ModuleNotFoundError as exc:
        extra = (
            "retrieval"
            if exc.name in {"chromadb", "sentence_transformers", "rank_bm25", "torch"}
            else "data"
        )
        print(
            f"Missing dependency {exc.name}. Install optional dependencies with pip install -e '.[{extra}]'.",
            file=sys.stderr,
        )
        return 2
    except (OSError, ValueError, KeyError, RuntimeError, AssertionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
