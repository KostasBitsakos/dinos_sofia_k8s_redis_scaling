#!/usr/bin/env python3
"""Detached rollback runner. Started by gate.py in a NEW SESSION so that tools/launch_run.py's SIGTERM/SIGKILL of the
experiment's process group cannot kill it. Runs rollback.sh (which holds the rollback lock) with its own timeout and records the
outcome in $QRL_RUN_DIR/rollback_status.json and in the fixed file code/calibration/.last_rollback.json.

RUNS ONLY VIA tools/launch_run.py WITH AN APPROVED real SPEC (started by gate.py).
Usage: rollback_runner.py [--timeout 420] [--reason TEXT]
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def last_path():
    return os.environ.get("QRL_LAST_ROLLBACK") or os.path.join(HERE, ".last_rollback.json")


def write_status(rec):
    paths = [last_path()]
    if os.environ.get("QRL_RUN_DIR"):
        paths.append(os.path.join(os.environ["QRL_RUN_DIR"], "rollback_status.json"))
    for p in paths:
        try:
            tmp = p + f".tmp{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(rec, f, indent=2)
            os.replace(tmp, p)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=420)
    ap.add_argument("--reason", default="")
    a = ap.parse_args()
    rec = {"status": "running", "pid": os.getpid(), "start": time.time(), "end": None, "exit_code": None,
           "reason": a.reason, "namespace_absent_proof": None, "output_tail": ""}
    write_status(rec)
    try:
        p = subprocess.run(["bash", os.path.join(HERE, "rollback.sh")], capture_output=True, text=True, timeout=a.timeout,
                           cwd=HERE)
        out = p.stdout + p.stderr
        rec.update(exit_code=p.returncode, status="done" if p.returncode == 0 else "failed", output_tail=out[-600:])
        # the proof is the API's own NotFound answer, echoed by rollback.sh on success
        rec["namespace_absent_proof"] = "NotFound" in p.stdout and p.returncode == 0
    except subprocess.TimeoutExpired as e:
        rec.update(exit_code=-1, status="failed", output_tail=f"timeout after {a.timeout}s", namespace_absent_proof=False)
    except BaseException as e:  # noqa: BLE001
        rec.update(exit_code=-2, status="failed", output_tail=repr(e)[:300], namespace_absent_proof=False)
    rec["end"] = time.time()
    write_status(rec)
    sys.exit(0 if rec["status"] == "done" else 1)


if __name__ == "__main__":
    main()
