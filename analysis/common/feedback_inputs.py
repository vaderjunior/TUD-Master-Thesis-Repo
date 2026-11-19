"Saved common feedback inputs."

from src.hsrag.paths import PROJECT_ROOT
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = PROJECT_ROOT
BASE = ROOT / "experiments/astra/sq3_additions_20261004_v1"
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
OLD_EX = ROOT / "experiments/astra/sq3_full_kb_explanations_20261001_v1"
OLD_TAX = ROOT / "experiments/astra/sq3_full_kb_taxonomy_20261001_v1"
REVIEW = ROOT / "experiments/astra/sq3_closing_review_20261004_v1"
OLD_FEEDBACK = ROOT / "experiments/astra/sq3_full_kb_feedback_20260928_v1"
TOK = ROOT / "experiments/astra/qwen_tokenizer_assets_20261001_v1"
LABELS = [
    "explicit",
    "incitement",
    "inferiority",
    "irony",
    "stereotyping",
    "threatening",
    "grievance",
]
SEEDS = [202609281, 202609282, 202609283]


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def rows(path):
    with Path(path).open(encoding="utf8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sig(value):
    return hashlib.sha256(
        (
            json.dumps(
                value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
            )
            + "\n"
        ).encode()
    ).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    require(
        path.resolve().is_relative_to(BASE.resolve()),
        "Write outside isolated additions",
    )
    content = (
        value
        if isinstance(value, str)
        else json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    )
    data = content.encode("utf8")
    if path.exists():
        require(path.read_bytes() == data, "Preserve existing artifact: " + rel(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def jsonlines(path, values):
    write(
        path,
        "".join(
            (
                json.dumps(v, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
                for v in values
            )
        ),
    )


def verify(bindings):
    for name, expected in bindings.items():
        require(sha(ROOT / name) == expected, "Bound file changed: " + name)


def bindings(paths):
    return {rel(p): sha(p) for p in paths}


def source_order(seed, value):
    return hashlib.sha256((str(seed) + "|" + str(value)).encode()).hexdigest()


def local_chat(messages):
    require(
        [m["role"] for m in messages] == ["system", "user"], "Unexpected conversation"
    )
    return (
        "".join(
            (
                "<|im_start|>" + m["role"] + "\n" + m["text"] + "<|im_end|>\n"
                for m in messages
            )
        )
        + "<|im_start|>assistant\n"
    )


def tokenizer():
    from tokenizers import Tokenizer

    receipt = read(TOK / "complete.json")
    for name, value in receipt["files"].items():
        require(sha(TOK / name) == value["sha256"], "Tokenizer changed")
    result = Tokenizer.from_file(str(TOK / "tokenizer.json"))
    result.no_padding()
    result.no_truncation()
    return result


def token_count(tok, messages):
    result = len(tok.encode(local_chat(messages), add_special_tokens=False).ids)
    require(0 < result <= 8192, "Prompt exceeds declared local token guard")
    return result


def inputs_for_study(paths):
    return bindings(
        [
            *paths,
            Path(__file__),
            ROOT / "config/taxonomy.yaml",
            ROOT / "prompts/classify_v1.txt",
            ROOT / "src/hsrag/prompt.py",
            ROOT / "analysis/common/prompt.py",
            TOK / "complete.json",
        ]
    )


def preserve_originals():
    original = read(REVIEW / "plan.json")["frozen_execution_bindings"]
    verify(original)
    return original
