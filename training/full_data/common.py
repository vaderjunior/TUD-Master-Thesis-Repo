import hashlib
import io
import json
import os
from pathlib import Path
import uuid
from datetime import datetime, timezone
import numpy as np


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def ready(value):
    if isinstance(value, dict):
        return {str(k): ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [ready(v) for v in value]
    if isinstance(value, set):
        return sorted(ready(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def encoded(value):
    return (
        json.dumps(
            ready(value), sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False
        )
        + "\n"
    ).encode("utf8")


def signature(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def commit_file(temporary, path):

    if os.name == "nt":
        os.rename(temporary, path)
    else:
        os.link(temporary, path)
        Path(temporary).unlink()


def publish(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == payload, "A saved file differs: " + str(path))
        return
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".writing")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        commit_file(temporary, path)
    except FileExistsError:
        require(path.read_bytes() == payload, "Conflicting concurrent publication.")
        temporary.unlink()


def save(path, value):
    publish(path, encoded(value))


def array(path, value):
    buf = io.BytesIO()
    np.save(buf, value, allow_pickle=False)
    publish(path, buf.getvalue())


def arrays(path, values):
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            require(
                set(stored.files) == set(values)
                and all(np.array_equal(stored[k], v) for k, v in values.items()),
                "Saved arrays differ.",
            )
    else:
        buf = io.BytesIO()
        np.savez(buf, **values)
        publish(path, buf.getvalue())


from training.full_data.plan import HEADS
from training.full_data.plan import CV_SEEDS
from training.full_data.plan import FINAL_SEEDS
from training.full_data.plan import THRESHOLDS
from training.full_data.plan import FAMILIES
from training.full_data.plan import config_map
from training.full_data.plan import protocol


def class_weights(*args, **kwargs):
    from training.full_data.selection import class_weights as f

    return f(*args, **kwargs)


def probabilities(*args, **kwargs):
    from training.full_data.selection import probabilities as f

    return f(*args, **kwargs)


def decisions(*args, **kwargs):
    from training.full_data.selection import decisions as f

    return f(*args, **kwargs)
