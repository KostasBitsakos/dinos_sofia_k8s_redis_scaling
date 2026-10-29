#!/usr/bin/env bash
# RUNS ONLY VIA tools/launch_run.py WITH AN APPROVED real SPEC (or by the researcher, as the spec's rollback command).
# Rollback = delete namespace qrl-calib (Deployment, pods, service, all test data). The Deployment in namespace default
# is never touched. Every kubectl call names the context explicitly. Delete errors are NOT suppressed; absence is proven
# only by a `get namespace` that fails with NotFound specifically. Any other error, a timeout, or an existing namespace
# exits non-zero.
set -uo pipefail
# One rollback at a time: a detached rollback started by gate.py (survives the experiment's SIGTERM/SIGKILL) holds this lock;
# a later rollback (the final spec entry) waits here for it to finish, then re-checks (idempotent).
LOCK="${QRL_ROLLBACK_LOCK:-$(dirname "$0")/.rollback.lock}"
exec 9>"$LOCK"
flock 9 || { echo "ROLLBACK FAILED: cannot take the rollback lock"; exit 1; }
CTX="kubernetes-admin@kubernetes"
NS="qrl-calib"
K="kubectl --context $CTX"
[ "$($K config current-context 2>/dev/null)" = "$CTX" ] || { echo "ROLLBACK FAILED: wrong or unreadable kubectl context"; exit 1; }

# 1. delete (an absent namespace is fine: --ignore-not-found; every other error is fatal)
if ! timeout 330 $K delete namespace "$NS" --ignore-not-found=true --wait=true --timeout=300s; then
  echo "ROLLBACK FAILED: delete namespace $NS returned an error or timed out"; exit 1
fi
# 2. prove absence with a successful API round trip that says NotFound
out="$(timeout 30 $K get namespace "$NS" 2>&1)"; rc=$?
if [ $rc -eq 0 ]; then
  echo "ROLLBACK FAILED: namespace $NS still exists"; exit 1
elif echo "$out" | grep -q "NotFound"; then
  echo "namespace $NS is gone (API said NotFound)"; exit 0
else
  echo "ROLLBACK FAILED: cannot prove absence (get namespace failed with something other than NotFound): $out"; exit 1
fi
