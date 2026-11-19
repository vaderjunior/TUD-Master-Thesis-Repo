import hashlib
import json


def signature(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def retain(catalog, example):
    identity = example["id"]
    text = example["text"]
    meta = example["meta"]
    digest = signature(meta)
    if identity not in catalog:
        catalog[identity] = {"id": identity, "text": text, "metadata_variants": {}}
    if catalog[identity]["text"] != text:
        raise ValueError("Example identity has inconsistent text: " + identity)
    variants = catalog[identity]["metadata_variants"]
    if digest in variants and variants[digest] != meta:
        raise ValueError("Metadata digest collision")
    variants[digest] = meta
    return digest
