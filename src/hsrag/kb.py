"""Build and read a Chroma index from labelled knowledge records."""

import hashlib
import json
from pathlib import Path
from src.hsrag.paths import project_path

COLLECTION_NAME = "hate_kb"


def _list_meta_keys() -> tuple[str, ...]:
    import yaml
    tax = yaml.safe_load(
        project_path("config/taxonomy.yaml").read_text(encoding="utf-8"))
    keys = []
    for dim, spec in tax["dimensions"].items():
        if spec.get("type") != "multilabel":
            continue

        keys += [dim, f"{dim}s"]
    return tuple(dict.fromkeys(keys))

LIST_META_KEYS = _list_meta_keys()


def _kb_hash(records_path: Path) -> str:
    return hashlib.sha256(records_path.read_bytes()).hexdigest()[:16]

def unsentinel(value):
    """Restore missing values and label lists from scalar index metadata."""
    if value == "__none__" or value == "":
        return None
    if isinstance(value, str) and value.startswith("["):

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    return value


def unsentinel_meta(meta: dict) -> dict:
    """Apply unsentinel to every value in a Chroma metadata dict."""
    return {k: unsentinel(v) for k, v in meta.items()}

def _flatten_meta(record: dict) -> dict:
    """Encode missing values and label lists for the index."""
    meta = {
        "kind": record["kind"],
        "lang": record["lang"],
        "source": record["source"],

        "dimension": record["dimension"] or "",
        "label": record["label"] or "",
    }

    for k, v in record.get("meta", {}).items():
        if v is None:
            meta[k] = "__none__"
        elif k in LIST_META_KEYS or isinstance(v, (list, tuple)):

            meta[k] = json.dumps(list(v))
        else:
            meta[k] = v

    return meta


def build(records_path: Path, chroma_path: Path, model_name: str) -> None:
    """Rebuild the Chroma collection from knowledge records."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"Loaded {len(records)} records from {records_path}")

    if not records:
        raise ValueError("Knowledge records are empty")

    model = SentenceTransformer(model_name)
    texts = [r["text"] for r in records]
    print(f"Encoding {len(texts)} texts with {model_name}...")
    embeddings = model.encode(
        texts, normalize_embeddings=True, batch_size=16, show_progress_bar=True
    ).tolist()

    client = chromadb.PersistentClient(path=str(chroma_path))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    col = client.create_collection(
        COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": model_name,
            "kb_version": _kb_hash(records_path),
        },
    )

    batch_size = min(len(records), getattr(client, "get_max_batch_size", lambda: len(records))())
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        col.add(
            ids=[r["id"] for r in batch],
            embeddings=embeddings[start:start + batch_size],
            documents=texts[start:start + batch_size],
            metadatas=[_flatten_meta(r) for r in batch],
        )

    print(f"Ingested {col.count()} records.")
    print(f"kb_version: {_kb_hash(records_path)}")


def load_collection(chroma_path: Path, model_name: str):
    """Check that the collection uses the configured embedding model."""
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_path))
    col = client.get_collection(COLLECTION_NAME)

    ingested_model = col.metadata.get("embedding_model")
    if ingested_model != model_name:
        raise RuntimeError(
            f"Model mismatch: collection was built with '{ingested_model}', "
            f"but you're querying with '{model_name}'. Rebuild or fix config."
        )

    return col
