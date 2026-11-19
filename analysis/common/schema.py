"""Common schema."""

import ast
from functools import lru_cache
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from analysis.common import api as a
from analysis.common import prompt_archive as taxonomy

ROOT = a.ROOT
LOCK = RLock()


def rel(path):
    return Path(str(path).removeprefix("\\\\?\\")).resolve().relative_to(ROOT).as_posix()


class ReplaceImports(ast.NodeTransformer):

    def __init__(self):
        self.seen = []

    def visit_ImportFrom(self, node):
        if node.module in ("src.hsrag.parse", "src.hsrag.vote"):
            a.require(
                all((n.asname is None for n in node.names)), "Unexpected aliased engine import"
            )
            self.seen.append((node.module, tuple((n.name for n in node.names))))
            return None
        return node


@lru_cache(maxsize=16)
def _adapter(path, taxonomy_sha, core_sha, adapter_sha, bundle_sha):
    a.require(a.digest(path) == taxonomy_sha, "Taxonomy file changed")
    bundle = taxonomy.bundle(path)
    engine = ROOT / "analysis/common/api.py"
    a.require(a.digest(engine) == core_sha, "Durable engine changed during binding")
    source = ast.parse(engine.read_text(encoding="utf8"))
    functions = [
        n
        for n in source.body
        if isinstance(n, ast.FunctionDef) and n.name in ("execute_item", "audit_item")
    ]
    a.require(len(functions) == 2, "Expected two durable entry points")
    replacer = ReplaceImports()
    module = replacer.visit(ast.Module(body=functions, type_ignores=[]))
    a.require(
        replacer.seen
        == [
            ("src.hsrag.parse", ("run_once",)),
            ("src.hsrag.vote", ("aggregate",)),
            ("src.hsrag.parse", ("parse_response", "REPAIR_TEMPLATE", "RunResult")),
            ("src.hsrag.vote", ("aggregate",)),
        ],
        "Engine import transformation changed",
    )
    implementation = a.implementation()
    implementation.update(
        {
            rel(path): taxonomy_sha,
            rel(__file__): adapter_sha,
            "analysis/common/prompt_archive.py": bundle_sha,
            "analysis/common/prompt.py": a.digest(ROOT / "analysis/common/prompt.py"),
        }
    )
    namespace = dict(vars(a))
    namespace.update(
        run_once=bundle.parser.run_once,
        parse_response=bundle.parser.parse_response,
        REPAIR_TEMPLATE=bundle.parser.REPAIR_TEMPLATE,
        RunResult=bundle.parser.RunResult,
        aggregate=bundle.voter.aggregate,
        implementation=lambda: dict(implementation),
    )
    exec(
        compile(ast.fix_missing_locations(module), str(engine) + "[isolated taxonomy]", "exec"),
        namespace,
    )
    return SimpleNamespace(
        execute_item=namespace["execute_item"],
        audit_item=namespace["audit_item"],
        bundle=bundle,
        implementation=lambda: dict(implementation),
        taxonomy_path=path,
        taxonomy_sha256=taxonomy_sha,
        import_substitutions=replacer.seen,
    )


def adapter(path):
    path = Path(path).resolve()
    with LOCK:
        return _adapter(
            path,
            a.digest(path),
            a.digest(ROOT / "analysis/common/api.py"),
            a.digest(__file__),
            a.digest(ROOT / "analysis/common/prompt_archive.py"),
        )
