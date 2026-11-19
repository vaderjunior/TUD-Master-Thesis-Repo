"""Inspect the records retrieved for one query."""

import argparse
from pathlib import Path
import yaml
from src.hsrag.paths import project_path


def show(text: str, lang: str, retriever):
    result = retriever.retrieve(text, lang=lang)
    print(f"\n{'=' * 72}\nQUERY ({lang}): {text}\n{'=' * 72}")
    for bucket_name, hits in [
        ("DEFINITIONS", result.definitions),
        ("GUIDELINES", result.guidelines),
        ("EXAMPLES", result.examples),
    ]:
        print(f"\n  --- {bucket_name} ({len(hits)}) ---")
        for i, h in enumerate(hits, 1):
            if h.kind == "example":
                tag = f"gate={h.meta.get('gate')} tg={h.meta.get('target_groups')}"
                if h.meta.get("illustrative_only"):
                    tag += f" [LEGAL §{h.meta.get('stgb')}]"
            else:
                tag = f"{h.meta.get('dimension') or ''}/{h.meta.get('label') or ''}"
            print(f"  [{i}] {h.score:.3f} ({h.via}) {h.id}")
            if h.meta.get("_ranks"):
                print(f"      ranks: {h.meta['_ranks']}")
            print(f"      {tag}")
            print(f"      {h.text[:88]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text")
    parser.add_argument("--lang", default="en", choices=["en", "de"])
    parser.add_argument("--strategy", choices=["dense", "bm25", "hybrid"])
    parser.add_argument(
        "--config", type=Path, default=project_path("config/config.yaml")
    )
    parser.add_argument("--records", type=Path)
    parser.add_argument("--index", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    knowledge = config["kb"]
    records = args.records or project_path(knowledge["records_path"])
    index = args.index or project_path(knowledge["chroma_path"])
    if not records.is_file():
        raise FileNotFoundError(records)
    if not index.is_dir():
        raise FileNotFoundError(index)
    settings = config["retrieval"]
    if args.strategy:
        settings = {**settings, "strategy": args.strategy}
    from src.hsrag.retrieve import Retriever

    retriever = Retriever(index, records, knowledge["embedding_model"], settings)
    show(args.text, args.lang, retriever)


if __name__ == "__main__":
    main()
