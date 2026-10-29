#!/usr/bin/env python3
"""Entry gate: `gate.py [--health] [--allow-failed] [--no-rollback] -- CMD [ARGS...]`, runs CMD only if allowed.

RUNS ONLY VIA tools/launch_run.py WITH AN APPROVED real SPEC (it wraps every entry of the calibration specs).
tools/launch_run.py (checked 2026-10-16) starts `--all` entries in index order WITHOUT looking at the status of earlier entries:
a failed entry does not stop the next one, and nothing triggers the rollback. So every entry starts here:
  1. the previous entry of the same request (run dir <QRL_LAB_DIR>/.lab/runs/*-<request>-e<N-1>-a*) must have FINISHED
     (launcher.json exists) and succeeded (status done, exit code 0). Otherwise: run rollback.sh (idempotent: deletes namespace
     qrl-calib) and exit 1, so every later entry fails the same way. `--allow-failed` (the rollback entry itself) only requires the
     previous entry to have finished; `--no-rollback` refuses without rolling back.
  2. with --health: common.health() must be OK (3+ pods Ready, 16384 slots covered, cluster_state ok on every node).
Entry 0 has no predecessor.
"""
import argparse
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_ID = re.compile(r"^\d{8}-\d{6}-(R-[\d-]+?)-e(\d+)-a(\d+)$")


def previous_status(run_id, runs_root):
    """-> None (no predecessor) | ("running", None) | ("done"|"failed"|..., exit_code)."""
    m = RUN_ID.match(run_id)
    if not m:
        raise SystemExit(f"gate: cannot parse QRL_RUN_ID {run_id!r}")
    rid, entry = m.group(1), int(m.group(2))
    if entry == 0:
        return None
    dirs = sorted(glob.glob(os.path.join(runs_root, f"*-{rid}-e{entry - 1}-a*")))
    if not dirs:
        return ("missing", None)
    if os.path.exists(os.path.join(dirs[-1], "cleanup_failed")):   # its process group could not be emptied
        return ("cleanup_failed", None)
    j = os.path.join(dirs[-1], "launcher.json")      # latest attempt of the previous entry
    if not os.path.exists(j):
        return ("running", None)
    d = json.load(open(j))
    return (d.get("status"), d.get("exit_code"))


def decide(prev, allow_failed):
    """-> (ok, reason)."""
    if prev is None:
        return True, "first entry"
    status, code = prev
    if status in ("running", "missing"):
        return False, f"previous entry is {status}"
    if allow_failed:
        return True, f"previous entry ended as {status}; this entry is the rollback"
    if status == "done" and code == 0:
        return True, "previous entry succeeded"
    return False, f"previous entry ended as {status} (exit {code})"


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


_REAP = {}


def reap_group(pgid, leader, grace=10.0):
    """Terminate the WHOLE process group of the experiment child (the leader may already be gone while grandchildren live on):
    SIGTERM the group, poll for emptiness (os.killpg(pgid, 0)) for `grace` s, SIGKILL the group, poll again, reap the leader.
    Returns only when no member of the group exists (or after a final 10 s poll). Idempotent: a repeated call returns the
    first result (signal handlers never call it; the main flow does, once)."""
    if pgid in _REAP:
        return _REAP[pgid]
    _REAP[pgid] = _reap_group(pgid, leader, grace)
    return _REAP[pgid]


def _reap_group(pgid, leader, grace):
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for stage, limit in (("term", grace), ("kill", 10.0)):
        if stage == "kill":
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        t_end = time.time() + limit
        while time.time() < t_end:
            leader.poll()          # reap the leader: a zombie leader keeps the group visible to killpg(pgid, 0)
            if not group_alive(pgid):
                leader.wait()
                return True
            time.sleep(0.1)
    leader.wait()
    return not group_alive(pgid)


def start_rollback(reason, wait=True, timeout=450):
    """Start rollback_runner.py in a NEW SESSION: launch_run.py signals the whole process group of the experiment (SIGTERM, then
    SIGKILL after 30 s) and must not be able to kill a rollback that takes up to 300 s. With wait=True the gate polls the status
    file until the runner finishes (or `timeout`); with wait=False (SIGTERM path) it returns at once."""
    log = open(os.path.join(os.environ.get("QRL_RUN_DIR") or HERE, "rollback_runner.log"), "ab")
    p = subprocess.Popen([sys.executable, os.path.join(HERE, "rollback_runner.py"), "--reason", reason],
                         start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=log, cwd=HERE)
    if not wait:
        return None
    try:
        return p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return None   # the runner keeps going on its own


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--health", action="store_true")
    ap.add_argument("--allow-failed", action="store_true", help="this IS the rollback entry: run after a failure, never roll back itself")
    ap.add_argument("--no-rollback", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        raise SystemExit("gate: no command")
    run_id = os.environ.get("QRL_RUN_ID", "")
    run_dir = os.environ.get("QRL_RUN_DIR", "")
    if not run_id or not run_dir:
        raise SystemExit("gate: QRL_RUN_ID / QRL_RUN_DIR missing: runs only via tools/launch_run.py")
    do_rollback = not (a.no_rollback or a.allow_failed)
    child = None
    state = {"terminated": 0.0, "rb": False, "pgid": None, "spawned": False}

    def on_term(signum, frame):
        # Installed FIRST (before the status check and the health query). It ONLY sets a flag, starts the detached rollback once
        # (not in the rollback entry) and forwards SIGTERM to the experiment's process group. All cleanup is in one place
        # (reap_group in the main flow), so a signal can neither skip nor duplicate it.
        state["terminated"] = state["terminated"] or time.time()
        if do_rollback and not state["rb"]:
            state["rb"] = True
            start_rollback(f"signal {signum}", wait=False)
        if state["pgid"] is not None:
            try:
                os.killpg(state["pgid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
        elif not state["spawned"]:
            # no experiment process exists yet (status check / health query): nothing to clean, leave now
            print("gate: signal before the entry started; rollback started" if do_rollback else "gate: signal", flush=True)
            os._exit(143)
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    def refuse(msg):
        if do_rollback:
            print("gate: running rollback.sh (detached session)", flush=True)
            start_rollback(msg)
        sys.exit(f"gate: refusing: {msg}")

    ok, why = decide(previous_status(run_id, os.path.dirname(run_dir.rstrip("/"))), a.allow_failed)
    print(f"gate: {why}", flush=True)
    if not ok:
        refuse(why)
    if a.health:
        try:  # ANY exception while querying health refuses and rolls back; it never bypasses the gate
            sys.path.insert(0, HERE)
            import common
            h = common.health()
        except BaseException as e:  # noqa: BLE001
            refuse(f"health query raised {e!r}")
        print("gate: health", json.dumps(h), flush=True)
        if not h["ok"]:
            refuse(f"cluster not healthy: {h['reasons']}")
    if a.allow_failed:
        # the rollback entry: run it in a detached session too (a stop/timeout of this entry must not kill it); its rollback.sh waits
        # for any in-flight detached rollback on the shared lock before acting
        rc = start_rollback("rollback entry")
        sys.exit(1 if rc is None else rc)
    # run the entry as a CHILD (never exec) in its OWN session/process group (pgid = child.pid, the leader), so the group can be
    # killed while the detached rollback (another session) is untouched
    pgid = None
    sigs = {signal.SIGTERM, signal.SIGINT}
    try:
        state["spawned"] = True              # from here on a signal must not take the fast path
        signal.pthread_sigmask(signal.SIG_BLOCK, sigs)       # no signal between Popen and pgid being recorded
        try:
            child = subprocess.Popen(cmd, start_new_session=True,
                                     preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_UNBLOCK, sigs))
            pgid = state["pgid"] = child.pid
        finally:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, sigs)   # a pending signal is delivered now, with pgid known
        rc = None
        while rc is None:
            try:
                rc = child.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                if state["terminated"] and time.time() - state["terminated"] > 10:   # leader ignores SIGTERM: kill the group
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    rc = child.wait()
    except BaseException as e:  # noqa: BLE001
        print(f"gate: exception while running the entry: {e!r}", flush=True)
        rc = 1
    finally:
        cleaned = True
        if pgid is not None:   # never exit while ANY member of the experiment's group exists, leader alive or not
            cleaned = reap_group(pgid, child)
    if not cleaned:
        # members survived SIGKILL and polling: mark the run (the next entry's gate refuses), start the detached rollback, fail
        print("gate: ERROR: experiment process group could not be emptied; cleanup_failed", flush=True)
        try:
            with open(os.path.join(run_dir, "cleanup_failed"), "w") as f:
                f.write(f"pgid {pgid}\n")
        except OSError:
            pass
        if do_rollback and not state["rb"]:
            state["rb"] = True
            start_rollback("cleanup_failed", wait=False)
        sys.exit(70 if rc in (0, None) else (rc if rc > 0 else 128 - rc))
    if state["terminated"]:
        sys.exit(143 if rc == 0 else (rc if rc > 0 else 128 - rc))      # rollback already running detached; do not wait
    if rc != 0 and do_rollback:
        print(f"gate: entry failed (exit {rc}); running rollback.sh (detached session)", flush=True)
        start_rollback(f"entry exit {rc}")
    sys.exit(0 if rc == 0 else (rc if rc > 0 else 128 - rc))


if __name__ == "__main__":
    main()
