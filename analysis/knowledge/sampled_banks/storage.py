"""Sampled banks storage."""

import json
from pathlib import Path
import sqlite3
import struct
import numpy as np
from analysis.knowledge.sampled_banks import retrieval as p

r = p.r
HEADER_FORMAT = "<iQQQQQQiIQQQdQ"
EMPTY_HEADER = (1, 0, 100, 0, 4236, 4228, 132, -1, 4294967295, 16, 32, 16, 0.36067376022224085, 100)


def compact_header(raw):
    r.require(len(raw) == struct.calcsize(HEADER_FORMAT) == 100, "Unexpected HNSW header size")
    values = struct.unpack(HEADER_FORMAT, raw)
    r.require(values == EMPTY_HEADER, "Only the exact empty default 1024D header qualifies")
    return {
        "version": values[0],
        "capacity": values[2],
        "persisted_elements": values[3],
        "bytes_per_element": values[4],
        "maximum_level": values[7],
    }


def changed_file_policy(changed, graph_name, count):
    permitted = {
        "chroma.sqlite3",
        str(Path(graph_name) / "data_level0.bin"),
        str(Path(graph_name) / "length.bin"),
    }
    r.require(
        count == 340 and set(changed) <= permitted,
        "Unexpected changed graph file or noncompact pool",
    )


def lock_table_policy(before, after):
    r.require(
        all((type(i) is int and state == 1 for i, state in before + after)),
        "Unexpected lock bookkeeping",
    )
    r.require(
        len(before) == len(set(before))
        and len(after) == len(set(after))
        and (set(before) <= set(after))
        and (len(after) <= len(before) + 1),
        "Unexpected acquire_write mutation",
    )


def logical_tables(directory):
    with sqlite3.connect((directory / "chroma.sqlite3").as_uri() + "?mode=ro", uri=True) as conn:
        schema = conn.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"
        ).fetchall()
        tables = {}
        for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ):
            quoted = '"' + name.replace('"', '""') + '"'
            tables[name] = sorted(conn.execute("SELECT * FROM " + quoted).fetchall(), key=repr)
    return (schema, tables)


def empty_snapshot(directory, done, records):
    r.require(
        done["pool_id"] in ("p00", "p01", "p02") and len(records) == 340,
        "Empty-graph exception is compact-only",
    )
    graphs = list(directory.glob("*/header.bin"))
    r.require(len(graphs) == 1, "Ambiguous graph directory")
    graph = graphs[0].parent
    header = compact_header(graphs[0].read_bytes())
    expected = {
        "header.bin": 100,
        "data_level0.bin": 423600,
        "length.bin": 400,
        "link_lists.bin": 0,
    }
    r.require(
        {x.name: x.stat().st_size for x in graph.iterdir()} == expected,
        "Unexpected empty graph files or metadata",
    )
    stored = np.load(p.ROOT / done["indexed_vectors"], allow_pickle=False)
    offset = {rec["id"]: i for i, rec in enumerate(records)}
    with sqlite3.connect((directory / "chroma.sqlite3").as_uri() + "?mode=ro", uri=True) as conn:
        schema = json.loads(conn.execute("SELECT schema_str FROM collections").fetchone()[0])
        config = schema["keys"]["#embedding"]["float_list"]["vector_index"]["config"]
        r.require(
            config["space"] == "cosine"
            and config["hnsw"]["sync_threshold"] == 1000
            and (config["hnsw"]["batch_size"] == 100),
            "Persistence defaults changed",
        )
        vector_segments = conn.execute("SELECT id FROM segments WHERE scope='VECTOR'").fetchall()
        r.require(vector_segments == [(graph.name,)], "Graph ownership changed")
        r.require(
            conn.execute(
                "SELECT COUNT(*) FROM max_seq_id WHERE segment_id=?", (graph.name,)
            ).fetchone()[0]
            == 0,
            "Empty header has a persisted vector sequence",
        )
        queue = conn.execute(
            "SELECT seq_id,operation,id,vector,encoding,metadata FROM embeddings_queue ORDER BY seq_id"
        ).fetchall()
        r.require(
            len(queue) == 340
            and [x[0] for x in queue] == list(range(1, 341))
            and ([x[2] for x in queue] == [rec["id"] for rec in records]),
            "Compact WAL does not contain exact full insertion order",
        )
        maximum = 0.0
        for seq, operation, rid, vector, encoding, metadata in queue:
            r.require(
                operation == 0 and encoding == "FLOAT32" and (len(vector) == 4096),
                "WAL vector shape/operation",
            )
            vv = np.frombuffer(vector, dtype="<f4")
            err = float(np.max(np.abs(vv - stored[offset[rid]])))
            r.require(
                np.isfinite(vv).all() and err < 2e-07,
                "WAL vector differs from exported stored vector",
            )
            maximum = max(maximum, err)
            r.require(
                json.loads(metadata)["chroma:document"] == records[offset[rid]]["text"],
                "WAL document mismatch",
            )
    return {
        "graph_directory": graph.name,
        "header": header,
        "sync_threshold": 1000,
        "complete_ordered_WAL_records": len(queue),
        "WAL_vector_max_error": maximum,
        "index_metadata_present": False,
        "persisted_vector_sequence_present": False,
    }


def amended_graph_check(done, records, source, directory, changed):
    graph_changes = [name for name in changed if name != "chroma.sqlite3"]
    if done["pool_id"] not in ("p00", "p01", "p02"):
        r.require(not graph_changes, "Noncompact persisted graph changed")
        return {"policy": "all_persisted_graph_files_exact", "graph_files_changed": []}
    parent = r.read(p.ROOT / done["source_receipt"])
    old = empty_snapshot(source, parent, records)
    new = empty_snapshot(directory, done, records)
    r.require(old == new, "Compact persistence state differs")
    changed_file_policy(changed, old["graph_directory"], len(records))
    old_schema, old_tables = logical_tables(source)
    new_schema, new_tables = logical_tables(directory)
    r.require(
        old_schema == new_schema and old_tables.keys() == new_tables.keys(),
        "Copied SQLite schema changed",
    )
    changed_tables = []
    for key in old_tables:
        if old_tables[key] == new_tables[key]:
            continue
        r.require(key == "acquire_write", "Real compact SQLite content changed: " + key)
        lock_table_policy(old_tables[key], new_tables[key])
        changed_tables.append(key)
    r.require(
        r.sha(p.ROOT / parent["indexed_vectors"]) == r.sha(p.ROOT / done["indexed_vectors"]),
        "Compact exported vectors changed",
    )
    r.require(
        p.tree_hashes(source) == parent["index_hashes"]
        and p.tree_hashes(directory) == done["index_hashes"],
        "Read-only persistence inspection changed files",
    )
    return {
        "policy": "empty_graph_capacity_bytes_qualified_complete_WAL_verified",
        "state": new,
        "graph_files_changed": graph_changes,
        "logical_tables_changed": changed_tables,
        "all_other_SQL_rows_exact": True,
        "exported_vectors_exact": True,
        "limitation": "Reopening reconstructs an in-memory graph from the WAL. This is not proof of identical in-memory graph or exact ANN candidates; retain all first-pass prompts and actual replay differences.",
    }
