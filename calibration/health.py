#!/usr/bin/env python3
"""Read-only cluster health check (kubectl get, redis-cli cluster info/nodes). Exit 0 healthy, 2 unhealthy.
Usage: health.py [--pods N] [--wait SECONDS]"""
import argparse
import json
import sys
import time

import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pods", type=int)
    ap.add_argument("--wait", type=float, default=0)
    ap.add_argument("--no-fail", action="store_true", help="always exit 0 (report only)")
    a = ap.parse_args()
    common.check_context()
    t0 = time.time()
    while True:
        h = common.health(a.pods)
        if h["ok"] or time.time() - t0 >= a.wait:
            break
        time.sleep(3)
    print(json.dumps(h))
    sys.exit(0 if h["ok"] or a.no_fail else 2)


if __name__ == "__main__":
    main()
