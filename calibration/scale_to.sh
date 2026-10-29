#!/usr/bin/env bash
# RUNS ONLY VIA tools/launch_run.py WITH AN APPROVED real SPEC.
# Mutates the Redis cluster in namespace qrl-calib (context kubernetes-admin@kubernetes). Usage: scale_to.sh N
set -euo pipefail
cd "$(dirname "$0")"
exec ../../.venv/bin/python scale_to.py --target "$1" "${@:2}"
