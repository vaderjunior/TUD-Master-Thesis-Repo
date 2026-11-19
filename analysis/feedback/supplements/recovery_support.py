import ast
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from types import FunctionType, SimpleNamespace
from analysis.common import api as a
from analysis.common import schema as isolated

BASE = a.ROOT / "experiments/astra/sq3_additions_20261004_v1/execution"
HERE = BASE / "recovery_20261005_v1"
MANIFEST = HERE / "manifest.json"
READY = HERE / "ready.json"
ACTIVE = None


def rel(path):
    return isolated.rel(path)


def activate(manifest, manifest_sha):
    global ACTIVE
    ACTIVE = {**manifest, "manifest_sha256": manifest_sha}
    _adapter.cache_clear()


def context():
    a.require(ACTIVE is not None, "Recovery manifest must be validated before use")
    return ACTIVE


def verify_files(bindings):
    for name, sha in bindings.items():
        a.require(
            a.digest(a.ROOT / name) == sha,
            "Preserved recovery artifact changed: " + name,
        )


def validate():
    m = a.read(MANIFEST)
    ready = a.read(READY)
    a.require(
        m["status"] == "reviewed_connection_outage_amendment",
        "Unexpected recovery scope",
    )
    a.require(
        ready["status"] == "offline_verified_ready_for_user"
        and ready["manifest_sha256"] == a.digest(MANIFEST),
        "Recovery is not ready",
    )
    a.require(
        a.digest(BASE / "plan.json") == m["original_plan_sha256"],
        "Original plan changed",
    )
    verify_files(m["source_hashes"])
    verify_files(m["preserved_hashes"])
    a.require(
        a.digest(HERE / "tests.json") == ready["tests_sha256"]
        and a.read(HERE / "tests.json")["passed"],
        "Recovery tests changed",
    )
    a.require(
        a.digest(HERE / "rehearsal.json") == ready["rehearsal_sha256"]
        and a.read(HERE / "rehearsal.json")["passed"],
        "Recovery rehearsal changed",
    )
    a.require(
        len(m["calls"]) == 8 and m["extra_attempts_per_call"] == 5,
        "Unexpected amendment size",
    )
    activate(m, a.digest(MANIFEST))
    return m


def scope(root, job, vote, turn):
    ctx = context()
    store = a.storage_path(root).resolve()
    item = a.storage_path(job).resolve()
    for entry in ctx["calls"]:
        if (
            store == a.storage_path(a.ROOT / entry["store"]).resolve()
            and item == a.storage_path(a.ROOT / entry["item"]).resolve()
            and (vote == entry["vote"])
            and (turn == entry["turn"])
        ):
            verify_files(entry["prior_hashes"])
            return entry
    return None


def reviewed_attempts(root, job, vote, turn):
    return range(10 if scope(root, job, vote, turn) else 5)


def reviewed_backoff(attempt):
    a.require(type(attempt) is int and 1 <= attempt <= 9, "Unexpected retry ordinal")
    if attempt == 5:
        return 0
    return a.CONTRACT["backoff_seconds"][attempt - 1 if attempt < 5 else attempt - 6]


def partial_outcome(root, job, intent, outcome):
    action, reason, _ = a.interpret(outcome)
    entry = scope(root, job, intent["vote"], intent["turn"])
    limit = 9 if entry else 4
    a.require(
        type(intent["attempt"]) is int and 0 <= intent["attempt"] <= limit,
        "Unapproved retry attempt",
    )
    if action == "stop" or (action == "retry" and intent["attempt"] == limit):
        raise a.CallStopped("Preserved terminal request: " + reason)


def verify_stops(root, target):
    allowed = context()["reviewed_stop_files"]
    for path in target.glob("stop_*.json"):
        name = rel(path)
        a.require(
            name in allowed and a.digest(path) == allowed[name],
            "New or changed execution stop requires review",
        )
    for name, sha in allowed.items():
        path = a.storage_path(a.ROOT / name)
        if path.parent == a.storage_path(target):
            a.require(
                path.is_file() and a.digest(path) == sha,
                "Reviewed stop history missing",
            )


def verify_attempt_summary(attempt, summary):
    if not summary["errors"] and (not summary["stopped"]):
        return
    path = attempt / "summary.json"
    name = rel(path)
    allowed = context()["reviewed_failed_summaries"]
    a.require(
        name in allowed and a.digest(path) == allowed[name],
        "Unreviewed failed execution attempt",
    )


def audit_history(root, target):
    return {
        Path(name).relative_to(Path(rel(root))).as_posix(): sha
        for name, sha in context()["reviewed_stop_files"].items()
        if a.storage_path(a.ROOT / name).parent == a.storage_path(target)
    }


def provenance():
    ctx = context()
    return {
        "manifest": rel(MANIFEST),
        "manifest_sha256": ctx["manifest_sha256"],
        "supplemental_source_hashes": ctx["source_hashes"],
        "scope": "Only eight frozen outage calls may use attempts 5..9. All old attempts remain counted.",
        "timing": "Recorded request durations exclude pause/restart downtime; remote work on transport errors is unknown.",
        "original_stamp_qualification": "Original stamps identify the unchanged base; this manifest identifies the additional execution and audit rules.",
    }


def verify_review(stage, root, plan_sha, mode, original):
    record = original(stage, root, plan_sha, mode)
    if stage["id"] == context()["stage"]:
        report = a.read(
            a.storage_path(root) / "stages" / stage["id"] / record["report"]
        )
        a.require(
            report["reviewed_transport_amendment"] == provenance(),
            "Recovery audit provenance differs",
        )
        verify_stops(root, a.storage_path(root) / "stages" / stage["id"])
    return record


class RetryTransform(ast.NodeTransformer):

    def __init__(self, client):
        self.client = client
        self.ranges = 0
        self.backoffs = 0

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "range"
            and (len(node.args) == 1)
            and isinstance(node.args[0], ast.Constant)
            and (node.args[0].value == 5)
        ):
            self.ranges += 1
            args = (
                "self.root, self.job, self.vote, self.turn"
                if self.client
                else "root, job, vote, turn"
            )
            return ast.copy_location(
                ast.parse("reviewed_attempts(" + args + ")", mode="eval").body, node
            )
        return node

    def visit_Subscript(self, node):
        node = self.generic_visit(node)
        if ast.dump(node, include_attributes=False) == ast.dump(
            ast.parse("CONTRACT['backoff_seconds'][attempt-1]", mode="eval").body,
            include_attributes=False,
        ):
            self.backoffs += 1
            return ast.copy_location(
                ast.parse("reviewed_backoff(attempt)", mode="eval").body, node
            )
        return node


@lru_cache(maxsize=4)
def _adapter(path, schema_sha, manifest_sha):
    original = isolated.adapter(path)
    a.require(
        original.taxonomy_sha256 == schema_sha
        and context()["manifest_sha256"] == manifest_sha,
        "Recovery adapter attribution changed",
    )
    tree = ast.parse((a.ROOT / "analysis/common/api.py").read_text(encoding="utf8"))
    client_node = deepcopy(
        next(
            (
                n
                for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == "DurableVoteClient"
            )
        )
    )
    transform = RetryTransform(True)
    client_node = transform.visit(client_node)
    a.require(
        (transform.ranges, transform.backoffs) == (1, 1),
        "Unexpected durable-client transformation",
    )
    client_ns = dict(vars(a))
    client_ns.update(
        reviewed_attempts=reviewed_attempts, reviewed_backoff=reviewed_backoff
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[client_node], type_ignores=[])),
            __file__ + "[reviewed client]",
            "exec",
        ),
        client_ns,
    )
    ns = dict(original.execute_item.__globals__)
    ns.update(
        DurableVoteClient=client_ns["DurableVoteClient"],
        reviewed_attempts=reviewed_attempts,
    )
    original_save = ns["save_new"]

    def save_with_provenance(path, value):
        p = a.storage_path(path)
        if p.name == "complete.json":
            entries = [
                e
                for e in context()["calls"]
                if p.parent == a.storage_path(a.ROOT / e["item"])
            ]
            if entries:
                value = {**value, "reviewed_transport_amendment_sha256": manifest_sha}
        original_save(path, value)

    ns["save_new"] = save_with_provenance
    execute = FunctionType(
        original.execute_item.__code__,
        ns,
        original.execute_item.__name__,
        original.execute_item.__defaults__,
        original.execute_item.__closure__,
    )
    audit_node = deepcopy(
        next(
            (
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "audit_item"
            )
        )
    )
    imports = isolated.ReplaceImports()
    audit_node = imports.visit(audit_node)
    a.require(
        imports.seen
        == [
            ("src.hsrag.parse", ("parse_response", "REPAIR_TEMPLATE", "RunResult")),
            ("src.hsrag.vote", ("aggregate",)),
        ],
        "Unexpected audit import substitutions",
    )
    transform = RetryTransform(False)
    audit_node = transform.visit(audit_node)
    a.require(
        (transform.ranges, transform.backoffs) == (1, 0),
        "Unexpected audit retry transformation",
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[audit_node], type_ignores=[])),
            __file__ + "[reviewed audit]",
            "exec",
        ),
        ns,
    )
    semantic_audit = ns["audit_item"]

    def audit_item(root, identity, messages, execution_mode="mock"):
        result = semantic_audit(root, identity, messages, execution_mode=execution_mode)
        item = a.storage_path(root) / "items" / a.signature(identity)
        if any(
            (item == a.storage_path(a.ROOT / e["item"]) for e in context()["calls"])
        ):
            a.require(
                a.read(item / "complete.json")["reviewed_transport_amendment_sha256"]
                == manifest_sha,
                "Missing amended completion provenance",
            )
        return result

    return SimpleNamespace(
        execute_item=execute,
        audit_item=audit_item,
        bundle=original.bundle,
        implementation=original.implementation,
        taxonomy_path=original.taxonomy_path,
        taxonomy_sha256=original.taxonomy_sha256,
    )


def adapter(path):
    return _adapter(Path(path).resolve(), a.digest(path), context()["manifest_sha256"])
