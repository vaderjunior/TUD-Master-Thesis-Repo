from pathlib import Path
import json
import time
from collections import Counter
from analysis.common import retrieval as r
from analysis.common import prompt as prompt


def main():
    import chromadb

    plan = r.validate()
    records, vectors, dev, q = r.data("full")
    known = {x["id"]: x for x in records}
    col = chromadb.PersistentClient(path=str(r.BASE / "full/chroma")).get_collection(
        "hate_kb"
    )
    retriever = r.make_retriever(
        col,
        r.CORE / "kb/full/records.jsonl",
        plan["retrieval"],
        r.CachedEncoder([x["text"] for x in dev], q),
    )
    saved = {x["key"]: x for x in r.rows(r.BASE / "full/prompts.jsonl")}
    selected = [x for x in dev if x["key"] == "de/detox-1389930895067467778"]
    output = []
    for x in selected:
        old = saved[x["key"]]
        for repeat in range(3):
            ctx = prompt.from_retrieval(retriever.retrieve(x["text"], x["lang"]))
            r.validate_context(ctx, known, x["lang"])
            messages = prompt.render_messages(x["text"], x["lang"], ctx, "corrected")
            output.append(
                {
                    "key": x["key"],
                    "repeat": repeat,
                    "old_context": old["context"],
                    "new_context": ctx,
                    "old_messages_sha256": old["messages_sha256"],
                    "new_messages_sha256": r.signature(messages),
                    "same_context": ctx == old["context"],
                    "same_messages": messages == old["messages"],
                    "old_ids": old["retrieved"],
                    "new_ids": prompt.hit_ids(ctx),
                }
            )
    path = r.BASE / f"reopen_probe_{time.time_ns()}.json"
    r.put(
        path,
        {
            "plan_sha256": r.sha(r.PLAN),
            "source_sha256": r.sha(Path(__file__)),
            "results": output,
        },
    )
    print(
        json.dumps(
            {
                "evidence": r.rel(path),
                "results": [
                    {
                        k: v
                        for k, v in x.items()
                        if k not in ("old_context", "new_context")
                    }
                    for x in output
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
