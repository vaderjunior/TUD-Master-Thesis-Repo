import argparse
import os
from pathlib import Path
import socket
from unittest.mock import patch


def forbidden(*args, **kwargs):
    raise RuntimeError("Offline encoder entry: network forbidden")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prepare", "preflight", "validate", "run"])
    args = p.parse_args()
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
    )
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ):
        import pyarrow.dataset
        import pyarrow.parquet
        import pandas
        from analysis.common import study as c
        from training.original import inference as e

        stamp = {
            "entry": c.rel(Path(__file__)),
            "entry_sha256": c.sha(Path(__file__)),
            "inference_source_sha256": c.sha(Path(e.__file__)),
            "policy": "Arrow dataset/parquet and pandas loaded before Torch; frozen model/inference functions unchanged.",
        }
        c.save(e.BASE / "runtime_entry.json", stamp)
        if args.mode == "prepare":
            e.prepare()
        elif args.mode == "validate":
            e.validate()
            print("Fixed checkpoint inputs validated; no inference.")
        elif args.mode == "preflight":
            e.preflight()
            c.save(
                e.BASE / "runtime_preflight.json",
                stamp
                | {
                    "plan_sha256": c.sha(e.PLAN),
                    "preflight_sha256": c.sha(e.BASE / "preflight.json"),
                },
            )
        else:
            check = c.read(e.BASE / "runtime_preflight.json")
            c.require(
                check
                == stamp
                | {
                    "plan_sha256": c.sha(e.PLAN),
                    "preflight_sha256": c.sha(e.BASE / "preflight.json"),
                },
                "Preflight/runtime binding changed",
            )
            e.run()
            c.save(
                e.BASE / "runtime_complete.json",
                stamp
                | {
                    "plan_sha256": c.sha(e.PLAN),
                    "complete_sha256": c.sha(e.BASE / "complete.json"),
                    "runtime_preflight_sha256": c.sha(
                        e.BASE / "runtime_preflight.json"
                    ),
                },
            )


if __name__ == "__main__":
    main()
