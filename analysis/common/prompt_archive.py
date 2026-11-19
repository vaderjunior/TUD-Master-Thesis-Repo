"""Common prompt archive."""

from src.hsrag.paths import PROJECT_ROOT
import ast
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
import sys
from threading import RLock
from types import ModuleType, SimpleNamespace
from analysis.common.prompt import corrected_gold

ROOT = PROJECT_ROOT
_LOCK = RLock()


def _module(filename, name, inject=None, remove=(), paths=None):
    path = ROOT / filename
    tree = ast.parse(path.read_text(encoding="utf8"))
    seen_imports, seen_paths = ([], [])
    body = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in remove:
            seen_imports.append(node.module)
            continue
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in (paths or {}):
                node.value = ast.Call(
                    func=ast.Name(id="Path", ctx=ast.Load()),
                    args=[ast.Constant(str(paths[target.id]))],
                    keywords=[],
                )
                seen_paths.append(target.id)
        body.append(node)

    class DeferredImports(ast.NodeTransformer):

        def visit_ImportFrom(self, node):
            if node.module in remove:
                seen_imports.append(node.module)
                return None
            return node

        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            node.body = node.body or [ast.Pass()]
            return node

    module = DeferredImports().visit(ast.Module(body=body, type_ignores=[]))
    body = module.body
    assert sorted(seen_imports) == sorted(remove), (filename, seen_imports)
    assert sorted(seen_paths) == sorted(paths or {}), (filename, seen_paths)
    tree.body = body
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__dict__.update(inject or {})
    assert name not in sys.modules, "Isolated module name already in use"
    sys.modules[name] = module
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)
    return module


@dataclass(frozen=True)
class Bundle:
    taxonomy_sha256: str
    schema: object
    parser: object
    voter: object
    prompt: object

    def render(self, text, lang, context):
        hit = lambda row: SimpleNamespace(**row)
        ctx = self.prompt.PromptContext(
            definitions=[hit(r) for r in context["definitions"]],
            guidelines=[hit(r) for r in context["guidelines"]],
            example_groups=[
                (title, [hit(r) for r in rows])
                for title, rows in context["example_groups"]
            ],
        )
        return self.prompt.build_prompt(text, lang, ctx)


@lru_cache(maxsize=16)
def _cached(path, digest):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    prefix = (
        "_feedback_taxonomy_"
        + hashlib.sha256((str(path) + digest).encode()).hexdigest()
    )
    schema = _module(
        "src/hsrag/schema.py", prefix + "_schema", paths={"TAXONOMY": path}
    )
    parser = _module(
        "src/hsrag/parse.py",
        prefix + "_parse",
        inject={"Result": schema.Result, "normalise_raw": schema.normalise_raw},
        remove=("src.hsrag.schema",),
    )
    voter = _module(
        "src/hsrag/vote.py",
        prefix + "_vote",
        inject={"Result": schema.Result, "SEVERITY_ORDER": schema.SEVERITY_ORDER},
        remove=("src.hsrag.schema",),
    )
    prompt = _module(
        "src/hsrag/prompt.py",
        prefix + "_prompt",
        inject={"Hit": SimpleNamespace},
        remove=("src.hsrag.retrieve",),
        paths={"TAXONOMY": path, "TEMPLATE": ROOT / "prompts/classify_v1.txt"},
    )
    prompt._render_gold = corrected_gold
    return Bundle(digest, schema, parser, voter, prompt)


def bundle(path):
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with _LOCK:
        return _cached(path, digest)


def retrieval_module():
    """Original ranking code with the two heavyweight providers left unbound."""
    with _LOCK:
        return _retrieval_module()


@lru_cache(maxsize=1)
def _retrieval_module():
    path = ROOT / "src/hsrag/kb.py"
    tree = ast.parse(path.read_text(encoding="utf8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("unsentinel", "unsentinel_meta")
    ]
    assert len(functions) == 2
    import json

    env = {"json": json}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), env)

    def unavailable(*args, **kwargs):
        raise RuntimeError(
            "Only the explicit copied collection and cached vectors are allowed"
        )

    return _module(
        "src/hsrag/retrieve.py",
        "_feedback_retrieval",
        inject={
            "SentenceTransformer": unavailable,
            "load_collection": unavailable,
            "unsentinel_meta": env["unsentinel_meta"],
        },
        remove=("sentence_transformers", "src.hsrag.kb"),
    )
