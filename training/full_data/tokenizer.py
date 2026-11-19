from pathlib import Path
from functools import lru_cache
import hashlib
import json
import os

ROOT = Path(os.environ.get("THESIS_INPUT_DIR", "/inputs")).resolve()
REFERENCE = ROOT / "tokenizer_reference.json"
REFERENCE_SHA = "374dc8a4fa0c77de6197bf97170e2dfcdd7d10258621d27b9f755d2b33c3bec9"
SEMANTIC_SHA = "5a92bf5835b0573619b18d451cb7c37e3451dd14f4cff668e9af4bebc2f60342"
PROBES = [
    "This is a simple sentence.",
    " leading  spaces\nnext\tline",
    "Äpfel, Grüße und Flüchtlinge.",
    "hello🙂 world",
    "<mask> [CLS] <s> [UNK]",
    "",
    "x" * 600,
    "word " * 180,
    "https://example.org/a?q=1",
    "don't—“quoted”",
]


def require(ok, message):
    if not ok:
        raise RuntimeError("Tokenizer guard: " + message)


def fingerprint(obj):
    return hashlib.sha256(
        json.dumps(
            obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def semantics(backend):
    d = json.loads(backend.to_str())
    d.pop("padding", None)
    d.pop("truncation", None)
    return d


@lru_cache(maxsize=1)
def reference():
    from tokenizers import Tokenizer

    require(
        hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == REFERENCE_SHA,
        "reference file changed",
    )
    ref = Tokenizer.from_file(str(REFERENCE))
    require(fingerprint(semantics(ref)) == SEMANTIC_SHA, "reference semantics changed")
    return ref


def reference_encoding(texts, padding=False):
    ref = reference()
    ref.enable_truncation(max_length=128, direction="right")
    ref.no_padding()
    if padding:
        ref.enable_padding(
            direction="right",
            length=128 if padding == "max_length" else None,
            pad_id=1,
            pad_type_id=0,
            pad_token="<pad>",
        )
    encoded = ref.encode_batch(texts)
    return {
        "input_ids": [x.ids for x in encoded],
        "attention_mask": [x.attention_mask for x in encoded],
    }


def check_encoding(enc, texts, padding=False):
    expected = reference_encoding(texts, padding)
    for key in ("input_ids", "attention_mask"):
        actual = enc[key].tolist() if hasattr(enc[key], "tolist") else enc[key]
        require(
            actual == expected[key],
            f"{key} differs from frozen reference on actual inputs",
        )
    return enc


def encode(tok, texts, padding=False, return_tensors=None):
    enc = tok(
        texts,
        truncation=True,
        max_length=128,
        padding=padding,
        return_tensors=return_tensors,
    )
    return check_encoding(enc, texts, padding)


def check(tok):
    d = semantics(tok.backend_tokenizer)
    require(d["model"]["type"] == "BPE", "expected BPE model")
    require(len(d["model"].get("merges", [])) == 50000, "expected 50,000 BPE merges")
    require(
        (
            d.get("pre_tokenizer", {}).get("type") == "ByteLevel"
            if d.get("pre_tokenizer")
            else False
        ),
        "missing ByteLevel pre-tokenizer",
    )
    require(
        fingerprint(d) == SEMANTIC_SHA,
        "vocabulary/merges/normalizer/special-token semantics differ",
    )
    require(
        len(tok) == 50265
        and tok.bos_token_id == 0
        and (tok.pad_token_id == 1)
        and (tok.eos_token_id == 2),
        "vocabulary or special IDs differ",
    )
    require(
        tok.padding_side == tok.truncation_side == "right",
        "padding/truncation side changed",
    )
    for padding in (False, True, "max_length"):
        encode(tok, PROBES, padding=padding)
    return {
        "status": "pass",
        "tokenizer_class": type(tok).__name__,
        "reference": REFERENCE.relative_to(ROOT).as_posix(),
        "reference_sha256": REFERENCE_SHA,
        "semantic_sha256": SEMANTIC_SHA,
        "merges": 50000,
        "vocab_size": 50265,
        "neutral_probe_count": len(PROBES),
        "padding_modes": ["none", "longest", "max_length_128"],
    }


def load(path):
    from transformers import RobertaTokenizer

    tok = RobertaTokenizer.from_pretrained(str(path), local_files_only=True)
    check(tok)
    return tok
