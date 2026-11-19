import os

for key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import argparse
import json
import sys

sys.dont_write_bytecode = True
from pathlib import Path


def main():
    from training.full_data import plan as p
    from training.full_data import runtime as r
    from training.full_data import common as c

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode")
    sub.add_parser("inspect")
    execute = sub.add_parser("run")
    execute.add_argument("--stage", choices=["search", "final", "all"], default="all")
    execute.add_argument("--storage-review", type=Path, required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("assignment", type=Path)
    release = sub.add_parser("release")
    release.add_argument("pool", choices=p.NAMES)
    release.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.mode in (None, "inspect"):
        identity = r.validate_package()
        print(
            json.dumps(
                dict(
                    package_sha256=identity,
                    plan=r.settings(),
                    training_started=False,
                    Qwen_calls=0,
                ),
                indent=2,
            )
        )
    elif args.mode == "worker":
        from training.full_data.controller import worker

        worker(args.assignment)
    elif args.mode == "run":
        from training.full_data.controller import run

        run(args.stage, args.storage_review)
    else:
        from training.full_data.archive import release

        r.configure()
        r.validate_package()
        c.require(
            args.receipt.resolve().is_relative_to(Path("/inputs").resolve()),
            "Use a private uploaded backup receipt.",
        )
        with r.lock(Path("/results/worker.lock")):
            release(args.pool, args.receipt)


if __name__ == "__main__":
    main()
