"""Policy engine."""

import ast
from functools import lru_cache
from pathlib import Path
from types import FunctionType, SimpleNamespace
from analysis.knowledge.policy import common as c
from analysis.common import schema as isolated


class PausedBeforeRequest(c.a.CallStopped):
    """Nothing was sent and no new request intent was created."""


@lru_cache(maxsize=1)
def guarded_complete():
    source = c.ROOT / "analysis/common/api.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DurableVoteClient")
    )
    method = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "complete"))

    class InsertGuard(ast.NodeTransformer):
        count = 0

        def visit_Assign(self, node):
            if any(
                (isinstance(t, ast.Name) and t.id == "intent" for t in node.targets)
            ) and isinstance(node.value, ast.Dict):
                self.count += 1
                return [
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr="before_request",
                                ctx=ast.Load(),
                            ),
                            args=[ast.Name(id="payload", ctx=ast.Load())],
                            keywords=[],
                        )
                    ),
                    node,
                ]
            return node

    transform = InsertGuard()
    changed = transform.visit(method)
    c.require(transform.count == 1, "Durable engine guard insertion point changed")
    namespace = dict(vars(c.a))
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[changed], type_ignores=[])),
            str(source) + "[local guard]",
            "exec",
        ),
        namespace,
    )
    return namespace["complete"]


def adapter(before_request):
    original = isolated.adapter(c.ROOT / "config/taxonomy.yaml")

    class Client(c.a.DurableVoteClient):
        complete = guarded_complete()

        def before_request(self, payload):
            before_request(payload)

    implementation = original.implementation() | {
        c.ref(Path(__file__)): c.sha(__file__),
        "analysis/knowledge/policy/common.py": c.sha(
            c.ROOT / "analysis/knowledge/policy/common.py"
        ),
        "analysis/knowledge/policy/run.py": c.sha(c.ROOT / "analysis/knowledge/policy/run.py"),
    }
    namespace = dict(original.execute_item.__globals__)
    namespace.update(DurableVoteClient=Client, implementation=lambda: dict(implementation))
    functions = {
        name: FunctionType(
            getattr(original, name).__code__,
            namespace,
            name,
            getattr(original, name).__defaults__,
            getattr(original, name).__closure__,
        )
        for name in ("execute_item", "audit_item")
    }
    namespace.update(functions)
    return SimpleNamespace(
        **functions,
        bundle=original.bundle,
        implementation=lambda: dict(implementation),
        original=original,
        guard_insertions=1
    )
