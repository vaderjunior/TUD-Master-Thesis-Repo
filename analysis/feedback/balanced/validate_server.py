import json
import time
from analysis.common import vote_api as b


def main():
    target = b.BASE / "recovery_20261006_v1" / ("health_" + str(time.time_ns()))
    try:
        b.availability(target, b.core.live_client())
    except (b.a.IntegrityError, b.a.CallStopped) as exc:
        response = target / "model_check_response.json"
        result = {
            "status": "not_available",
            "error_type": type(exc).__name__,
            "classification_calls": 0,
            "evidence": str(target),
        }
        if response.exists():
            result["http_status"] = b.a.read(response)["status"]
        b.a.save_new(target / "summary.json", result)
        print(
            "Qwen availability check did not pass. No classification was sent. Try later; do not change the model or delete saved results.",
            flush=True,
        )
        print(json.dumps(result, indent=2), flush=True)
        raise SystemExit(3) from None
    result = {
        "status": "pinned_model_listed",
        "model": b.a.MODEL,
        "classification_calls": 0,
        "evidence": str(target),
    }
    b.a.save_new(target / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
