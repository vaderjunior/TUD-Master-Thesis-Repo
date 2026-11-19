from src.hsrag.paths import PROJECT_ROOT
from pathlib import Path
import hashlib
import json
import socket
from unittest.mock import patch

ROOT = PROJECT_ROOT
BASE = ROOT / "experiments/astra/sq3_full_kb_taxonomy_20261001_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    amendment = json.loads(
        (BASE / "checker_amendment_v1.json").read_text(encoding="utf8")
    )
    assert amendment["old_checker_sha256"] == sha(
        ROOT / "analysis/feedback/restoration/validate_earlier.py"
    )
    assert amendment["new_checker_sha256"] == sha(Path(__file__))
    for name, expected in amendment["unchanged_science_and_scope"].items():
        assert sha(BASE / name) == expected
    source = (ROOT / "analysis/feedback/restoration/validate_earlier.py").read_text(
        encoding="utf8"
    )
    old = "chat=environment.from_string((TOK/'chat_template.jinja').read_text(encoding='utf8'))"
    new = (
        "chat=environment.from_string(read(TOK/'chat_template.json')['chat_template'])"
    )
    assert source.count(old) == 1
    source = source.replace(old, new)
    marker = (
        "    with (BASE/'complete.json').open('x',encoding='utf8',newline='\\n') as f:"
    )
    extra = """    result['checker_amendment_sha256']=sha(BASE/'checker_amendment_v1.json')
    result['superseded_checker_sha256']=sha(ROOT/'analysis/feedback/restoration/validate_earlier.py')
    queue=BASE/'preparation_queue_v1/plan.json'
    assert queue.exists(), 'Frozen offline execution queue required'
    check_files(ROOT,read(queue)['inputs'])
    result['offline_queue_plan_sha256']=sha(queue)
"""
    assert source.count(marker) == 1
    source = source.replace(marker, extra + marker)
    env = {
        "__name__": "_astra_taxonomy_independent_checker_v2",
        "__file__": __file__,
        "__package__": "scratch.analysis",
    }
    exec(
        compile(
            source,
            str(ROOT / "analysis/feedback/restoration/validate_earlier.py"),
            "exec",
        ),
        env,
    )
    with patch.object(
        socket.socket,
        "connect",
        side_effect=RuntimeError("Network forbidden in checker"),
    ):
        env["main"]()


if __name__ == "__main__":
    main()
