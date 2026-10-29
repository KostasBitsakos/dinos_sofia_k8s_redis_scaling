#!/usr/bin/env python3
"""Report the last detached rollback (code/calibration/.last_rollback.json, or --file). Read-only.
Exit 0 only if the last rollback finished successfully with the namespace-absent proof; 1 if failed, running, stale or absent."""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=os.environ.get("QRL_LAST_ROLLBACK") or os.path.join(HERE, ".last_rollback.json"))
    a = ap.parse_args()
    if not os.path.exists(a.file):
        print("no rollback recorded")
        sys.exit(1)
    r = json.load(open(a.file))
    state = r["status"]
    if state == "running":
        try:
            os.kill(r["pid"], 0)
        except OSError:
            state = "stale (runner died without finishing)"
    dur = (r["end"] or time.time()) - r["start"]
    print(f"rollback {state}: exit={r['exit_code']} duration={dur:.0f}s proof={r['namespace_absent_proof']} reason={r.get('reason')!r}")
    print(r.get("output_tail", "").strip())
    sys.exit(0 if (r["status"] == "done" and r["namespace_absent_proof"]) else 1)


if __name__ == "__main__":
    main()
