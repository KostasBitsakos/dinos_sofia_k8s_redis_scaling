"""Shared helpers for the Sim v2 calibration pilot (cluster access, events log, health check).

Read-only helpers (health, node listing) are safe anywhere. Mutating helpers (scale/add-node/rebalance)
live in scale_to.py and run ONLY VIA tools/launch_run.py WITH AN APPROVED real SPEC.
"""
import datetime
import json
import os
import re
import subprocess
import time

CONTEXT = "kubernetes-admin@kubernetes"
DEPLOY = "redis-autoscale-cluster"
PORT = 6379
# Isolated calibration namespace (decision R-20261016-018): a copy of the researcher's Deployment, stateless, in qrl-calib.
# Their `default` deployment is never touched. Rollback = delete the namespace.
NAMESPACE = "qrl-calib"  # hard-coded: this pilot may only ever touch this namespace
if os.environ.get("QRL_NAMESPACE", NAMESPACE) != NAMESPACE:
    raise SystemExit(f"refusing: QRL_NAMESPACE={os.environ['QRL_NAMESPACE']!r}; only {NAMESPACE!r} is allowed")
KUBECTL = ["kubectl", "--context", CONTEXT, "--namespace", NAMESPACE]  # EVERY namespaced kubectl call uses this prefix
WORKLOAD = f"deployment/{DEPLOY}"
LABEL = "redis=redis-cluster-pods-lbl"

_IPV4 = re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b")
_allow_cache = {"t": 0.0, "ips": frozenset()}


def allowed_ips(force=False):
    """IPs of the pods in namespace qrl-calib (cached 2 s). Any Redis endpoint outside this set is refused."""
    if force or time.time() - _allow_cache["t"] > 2.0:
        out = subprocess.run([*KUBECTL, "get", "pods", "-o", "jsonpath={.items[*].status.podIP}"], capture_output=True,
                             text=True, timeout=30)
        if out.returncode != 0:
            raise RuntimeError(f"cannot list pods of {NAMESPACE} for the endpoint allowlist: {out.stderr.strip()[:200]}")
        _allow_cache.update(t=time.time(), ips=frozenset(out.stdout.split()))
    return _allow_cache["ips"]


def check_endpoints(cmd):
    """redis-cli commands may only address qrl-calib pod IPs."""
    if not cmd or os.path.basename(cmd[0]) != "redis-cli":
        return
    for a in cmd[1:]:
        for ip in _IPV4.findall(a):
            if ip not in allowed_ips() and ip not in allowed_ips(force=True):
                raise RuntimeError(f"refusing redis-cli endpoint {ip}: not a pod of namespace {NAMESPACE}")


_KUBECTL_RO = {"get", "top", "describe", "logs", "version", "config"}
_REDIS_RO_TOP = {"ping", "dbsize", "info", "--scan"}
_REDIS_RO_CLUSTER = {"nodes", "info", "slots", "myid"}


def is_readonly(cmd):
    """True for kubectl get/top/describe/logs/version/config and redis-cli ping/dbsize/info/--scan/cluster nodes|info|slots.
    `sh` runs ONLY these. Everything that mutates must go through scale_to.tracked (cancellable, killed on a trip)."""
    exe = os.path.basename(cmd[0])
    toks, i = [], 1
    if exe == "kubectl":
        while i < len(cmd):
            if cmd[i] in ("--context", "--namespace", "-n"):
                i += 2
                continue
            if cmd[i].startswith("-"):
                i += 1
                continue
            return cmd[i] in _KUBECTL_RO
        return False
    if exe == "redis-cli":
        while i < len(cmd):
            if cmd[i] in ("-h", "-p"):
                i += 2
                continue
            toks.append(cmd[i].lower())
            i += 1
        if not toks:
            return False
        if toks[0] in _REDIS_RO_TOP:
            return True
        return toks[0] == "cluster" and len(toks) > 1 and toks[1] in _REDIS_RO_CLUSTER
    return True  # other executables (python, bash) are not kubectl/redis-cli


def write_trip(why, **kw):
    """Record a trip: all scaling stops (require_no_trip) until recover_after_trip succeeds."""
    with open(trip_marker(), "w") as f:
        f.write(why)
    event("trip", why=why, **kw)


def sh(cmd, timeout=60, check=False):
    """Run a READ-ONLY command list, return (rc, stdout, stderr). Never uses a shell. Mutating kubectl/redis-cli commands
    are refused here: they must run through scale_to.tracked."""
    if not is_readonly(cmd):
        raise RuntimeError(f"mutating command refused outside the tracked runner: {' '.join(cmd[:6])}")
    check_endpoints(cmd)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} -> rc={p.returncode}: {p.stderr.strip()[:300]}")
    return p.returncode, p.stdout, p.stderr


def kubectl(*args, timeout=60, check=True):
    return sh([*KUBECTL, *args], timeout=timeout, check=check)[1]


def check_context():
    ctx = sh(["kubectl", "config", "current-context"])[1].strip()
    if ctx != CONTEXT:
        raise SystemExit(f"refusing: kubectl context is {ctx!r}, expected {CONTEXT!r}")


def redis_pods():
    """[{name, ip, ready, created}] of the Redis worker pods (not the coordinator)."""
    out = json.loads(kubectl("get", "pods", "-l", LABEL, "-o", "json"))
    pods = []
    for it in out["items"]:
        if it["metadata"].get("deletionTimestamp"):
            continue
        st = it.get("status", {})
        rc = [c for c in st.get("conditions", []) if c.get("type") == "Ready" and c.get("status") == "True"]
        ready = bool(rc)
        since = None
        if rc and rc[0].get("lastTransitionTime"):
            try:
                since = datetime.datetime.strptime(rc[0]["lastTransitionTime"], "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=datetime.timezone.utc).timestamp()
            except ValueError:
                since = None
        pods.append({"name": it["metadata"]["name"], "ip": st.get("podIP"), "ready": ready, "ready_since": since,
                     "uid": it["metadata"]["uid"], "created": it["metadata"]["creationTimestamp"]})
    return sorted(pods, key=lambda p: p["created"])


READY_GRACE_S = 15.0   # a pod's memory is required only this long after it became Ready (cgroup / metrics lag behind Ready)


def watchable(pods, state, now=None):
    """Pods whose memory the watch must be able to read: Ready for at least READY_GRACE_S. Not-Ready pods (starting,
    terminating) and pods inside the grace window are skipped; an unreadable Ready pod after grace still trips."""
    now = time.time() if now is None else now
    seen, out = state.setdefault("ready_seen", {}), []
    for p in pods:
        if not p.get("ready", True):
            seen.pop(p["name"], None)
            continue
        t0 = p.get("ready_since") or seen.setdefault(p["name"], now)
        if now - t0 >= READY_GRACE_S:
            out.append(p)
    return out


def rcli(ip, *args, timeout=30):
    return sh(["redis-cli", "-h", ip, "-p", str(PORT), *args], timeout=timeout)


def fill_myself(nodes, ip):
    """A fresh, unjoined Redis node reports its own `myself` line with an EMPTY host (":6379@16379") until it learns its address
    (seen on the first real run, 2026-10-16). That line is the queried pod itself: substitute the IP we queried (which is an
    allowlisted pod IP, since rcli refuses anything else). Any OTHER empty or unparsable address is left as is."""
    for nd in nodes:
        host, _, port = nd["addr"].rpartition(":")
        if "myself" in nd["flags"] and host == "" and port:
            nd["addr"] = f"{ip}:{port}"
    return nodes


def cluster_nodes(ip):
    """Parse CLUSTER NODES from one node -> list of dicts (id, addr, flags, slots=[(lo,hi)])."""
    rc, out, _ = rcli(ip, "cluster", "nodes", timeout=10)
    if rc != 0:
        return []
    return fill_myself(parse_cluster_nodes(out), ip)


def parse_cluster_nodes(text):
    nodes = []
    for line in text.strip().splitlines():
        f = line.split()
        if len(f) < 8:
            continue
        slots, open_slots = [], []
        for s in f[8:]:
            if s.startswith("["):
                open_slots.append(s)  # migrating/importing marker, e.g. [5500->-id] or [5500-<-id]
                continue
            lo, _, hi = s.partition("-")
            slots.append((int(lo), int(hi or lo)))
        nodes.append({"id": f[0], "addr": f[1].split("@")[0], "flags": f[2].split(","), "link": f[7],
                      "slots": slots, "open": open_slots})
    return nodes


def cluster_nodes_strict(ip):
    """CLUSTER NODES from one pod; a failed or EMPTY read raises (never treated as a clean view)."""
    rc, out, err = rcli(ip, "cluster", "nodes", timeout=10)
    if rc != 0 or not out.strip():
        raise RuntimeError(f"CLUSTER NODES on {ip} failed or was empty (rc={rc}): {err.strip()[:120]}")
    nodes = fill_myself(parse_cluster_nodes(out), ip)
    if not nodes:
        raise RuntimeError(f"CLUSTER NODES on {ip} had no parsable node")
    return nodes


def slots_covered(nodes):
    return sum(hi - lo + 1 for n in nodes if "master" in n["flags"] for lo, hi in n["slots"])


def health(expected_pods=None):
    """Read-only. Returns dict with ok flag and reasons. Healthy = every pod Ready, every pod a connected master,
    16384 slots covered, cluster_state ok on every node, no failed nodes."""
    reasons = []
    pods = redis_pods()
    if expected_pods is not None and len(pods) != expected_pods:
        reasons.append(f"pods={len(pods)} expected {expected_pods}")
    for p in pods:
        if not p["ready"]:
            reasons.append(f"{p['name']} not Ready")
    states = {}
    cov = None
    for p in pods:
        if not p["ip"]:
            continue
        rc, out, _ = rcli(p["ip"], "cluster", "info", timeout=10)
        m = re.search(r"cluster_state:(\w+)", out)
        states[p["ip"]] = m.group(1) if m else "unreachable"
    if pods and pods[0]["ip"]:
        nodes = cluster_nodes(pods[0]["ip"])
        cov = slots_covered(nodes)
        if cov != 16384:
            reasons.append(f"slots covered {cov}/16384")
        bad = [n["addr"] for n in nodes if any(x in n["flags"] for x in ("fail", "fail?", "noaddr", "handshake"))]
        if bad:
            reasons.append(f"failed nodes {bad}")
        if len(nodes) != len(pods):
            reasons.append(f"cluster knows {len(nodes)} nodes, {len(pods)} pods")
    if any(s != "ok" for s in states.values()):
        reasons.append(f"cluster_state {states}")
    return {"ok": not reasons, "reasons": reasons, "pods": len(pods), "slots": cov}


def run_dir():
    d = os.environ.get("QRL_RUN_DIR") or "."
    os.makedirs(d, exist_ok=True)
    return d


def event(name, **kw):
    """Append {"t": wall-clock seconds, "event": name, ...} to $QRL_RUN_DIR/events.jsonl and echo it."""
    rec = {"t": time.time(), "event": name, **kw}
    with open(os.path.join(run_dir(), "events.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
    return rec


def write_progress(step, total):
    with open(os.path.join(run_dir(), "progress.json"), "w") as f:
        json.dump({"step": step, "total": total}, f)


def metric(step, **kw):
    with open(os.path.join(run_dir(), "metrics.jsonl"), "a") as f:
        f.write(json.dumps({"step": step, **kw}) + "\n")


# ---------------------------------------------------------------- safety: memory, persistence, host telemetry
import glob
from concurrent.futures import ThreadPoolExecutor

CG_MEM = "/sys/fs/cgroup/memory/system.slice/containerd.service"
CG_CPU = "/sys/fs/cgroup/cpuacct/system.slice/containerd.service"
MEM_ABORT_FRAC = 0.95   # of the cgroup limit (16 MiB), on the WORKING SET (usage - inactive_file, as kubelet does)
MEM_START_FRAC = 0.85   # refuse to start a load if any pod is above this already (2026-10-16 idle: 0.50-0.82)
AOF_GROWTH_ABORT = 20_000_000  # bytes of AOF growth within one run
AOF_ABS_ABORT = 100_000_000
# Generator budget (measured stage A: 38 us of host CPU per op, one worker invalid above 0.85 core): offered ops/s one worker can drive validly.
GEN_US_PER_OP, GEN_MAX_CORE = 38.0, 0.85
GEN_OPS_PER_WORKER = GEN_MAX_CORE / (GEN_US_PER_OP * 1e-6)


def gen_budget(workers):
    """Highest offered rate (ops/s) at which the generator is still valid by the fit.py rules. Levels above it are never run."""
    return workers * GEN_OPS_PER_WORKER


POD_CPU_LIMIT = 0.2     # cores per pod (limit 200m)


def _read(path):
    with open(path) as f:
        return f.read()


def trip_marker():
    return os.path.join(run_dir(), "MEM_TRIPPED")


def require_no_trip():
    """Scaling gate: after a memory trip no scaling is allowed until recover_after_trip() has succeeded."""
    if os.path.exists(trip_marker()):
        raise RuntimeError("memory trip recorded in this run: scaling is stopped, recovery required")


def pod_cgroup(uid, root=CG_MEM):
    """Container cgroup dir of a pod (the one with a real memory limit, not the pause sandbox), or None."""
    pat = os.path.join(root, f"kubepods-*-pod{uid.replace('-', '_')}.slice:cri-containerd:*")
    for d in glob.glob(pat):
        try:
            if int(_read(os.path.join(d, "memory.limit_in_bytes"))) < (1 << 50):
                return d
        except (OSError, ValueError):
            continue
    return None


def pod_memory(pod, cg_root=CG_MEM):
    """(working_set_bytes, limit_bytes, source) from the pod's cgroup (read-only), else kubectl top (coarse, lagging).
    Raises RuntimeError if neither works: callers fail closed."""
    d = pod_cgroup(pod["uid"], cg_root) if pod.get("uid") else None
    if d:
        usage = int(_read(os.path.join(d, "memory.usage_in_bytes")))
        st = dict(line.split() for line in _read(os.path.join(d, "memory.stat")).splitlines() if len(line.split()) == 2)
        # page cache (mapped binary, AOF) is mostly reclaimable: judge the working set, as the kubelet does
        return usage - int(st.get("inactive_file", 0)), int(_read(os.path.join(d, "memory.limit_in_bytes"))), "cgroup"
    rc, out, _ = sh([*KUBECTL, "top", "pod", pod["name"], "--no-headers"], timeout=20)
    m = re.search(r"(\d+)Mi\s*$", out.strip())
    if rc == 0 and m:
        return int(m.group(1)) * 1048576, 16 * 1048576, "kubectl-top"
    raise RuntimeError(f"cannot read memory of {pod['name']} (no cgroup, no kubectl top)")


def pod_cpu_ns(pod, mem_root=CG_MEM, cpu_root=CG_CPU):
    """Cumulative CPU ns of the pod's container. The cpuacct hierarchy is looked up by the container directory NAME found
    in the memory hierarchy (the cpuacct dir has no memory.limit_in_bytes, so the memory helper cannot search it)."""
    d = pod_cgroup(pod["uid"], mem_root) if pod.get("uid") else None
    if not d:
        return None
    try:
        return int(_read(os.path.join(cpu_root, os.path.basename(d), "cpuacct.usage")))
    except OSError:
        return None


def pod_cpu_stat(pod, mem_root=CG_MEM, cpu_root=CG_CPU):
    """CFS limits and throttling counters of the pod's container cgroup (cgroup v1, read-only):
    {quota_us, period_us, nr_periods, nr_throttled, throttled_ns} or None (unreadable: never a silent zero)."""
    d = pod_cgroup(pod["uid"], mem_root) if pod.get("uid") else None
    if not d:
        return None
    base = os.path.join(cpu_root, os.path.basename(d))
    try:
        st = dict(line.split() for line in _read(os.path.join(base, "cpu.stat")).splitlines() if len(line.split()) == 2)
        out = {"quota_us": int(_read(os.path.join(base, "cpu.cfs_quota_us"))), "period_us": int(_read(os.path.join(base, "cpu.cfs_period_us"))),
               "nr_periods": int(st["nr_periods"]), "nr_throttled": int(st["nr_throttled"])}
        out["throttled_ns"] = int(st["throttled_time"]) if "throttled_time" in st else int(st["throttled_usec"]) * 1000
        return out
    except (OSError, KeyError, ValueError):
        return None


def redis_info(ip, section, timeout=2.0):
    """INFO <section> over a raw socket; raises on any error or on an empty/malformed reply (no silent zeros)."""
    import socket
    s = socket.create_connection((ip, PORT), timeout=timeout)
    try:
        s.sendall(b"INFO " + section.encode() + b"\r\n")
        buf = b""
        while b"\r\n" not in buf:
            chunk = s.recv(4096)
            if not chunk:
                raise RuntimeError(f"{ip}: connection closed during INFO {section}")
            buf += chunk
        if not buf.startswith(b"$"):
            raise RuntimeError(f"{ip}: INFO {section} replied {buf[:60]!r}")
        n = int(buf[1:buf.index(b"\r\n")])
        body = buf[buf.index(b"\r\n") + 2:]
        while len(body) < n:
            chunk = s.recv(4096)
            if not chunk:
                raise RuntimeError(f"{ip}: short INFO {section}")
            body += chunk
        return body[:n].decode()
    finally:
        s.close()


def served_calls(ip):
    """Total client GET + SET calls processed by the pod (INFO commandstats), or None when unreadable. Migration uses
    RESTORE / MIGRATE, cluster traffic uses other commands, so an increase means a client request was served."""
    try:
        out = redis_info(ip, "commandstats")
    except Exception:
        return None
    return sum(int(m.group(1)) for m in re.finditer(r"cmdstat_(?:get|set):calls=(\d+)", out))


def persistence_state(ip):
    out = redis_info(ip, "persistence")
    vals = {}
    for k in ("aof_enabled", "aof_rewrite_in_progress", "rdb_bgsave_in_progress"):
        m = re.search(rf"{k}:(\d+)", out)
        if m is None:
            raise RuntimeError(f"{ip}: INFO persistence has no {k}")
        vals[k] = int(m.group(1))
    m = re.search(r"aof_current_size:(\d+)", out)  # only present while AOF is on (the qrl-calib pods run with it off)
    if m is None and vals["aof_enabled"]:
        raise RuntimeError(f"{ip}: AOF is on but INFO persistence has no aof_current_size")
    vals["aof_current_size"] = int(m.group(1)) if m else 0
    return {"aof_enabled": vals["aof_enabled"], "aof_size": vals["aof_current_size"],
            "aof_rewrite": vals["aof_rewrite_in_progress"], "bgsave": vals["rdb_bgsave_in_progress"]}


def check_pod(p):
    """One pod, run in its own thread: memory fraction + persistence flags. Raises on anything unreadable."""
    u, lim, src = pod_memory(p)
    ps = persistence_state(p["ip"]) if p.get("ip") else {"aof_size": 0, "aof_rewrite": 0, "bgsave": 0}
    return {"pod": p["name"], "frac": u / lim, "source": src, **ps}


def check_all(pods, pool, timeout=4.0):
    """All pods CONCURRENTLY (a 30-pod poll must not take 30 sequential round trips). Raises if any check fails or is late."""
    futs = [pool.submit(check_pod, p) for p in pods]
    return [f.result(timeout=timeout) for f in futs]


def memory_snapshot():
    with ThreadPoolExecutor(max_workers=16) as pool:
        res = check_all(redis_pods(), pool)
    return {"worst_frac": max(r["frac"] for r in res), "aof_total": sum(r["aof_size"] for r in res),
            "persist_active": sum(r["aof_rewrite"] + r["bgsave"] for r in res),
            "source": sorted({r["source"] for r in res})}


def require_safe_to_load():
    """Start gate: memory readable and below MEM_START_FRAC on every pod."""
    snap = memory_snapshot()
    if snap["worst_frac"] > MEM_START_FRAC:
        raise SystemExit(f"refusing to load: a pod is at {snap['worst_frac']:.0%} of its memory limit")
    event("mem_start_check", **snap)
    return snap


def require_healthy(expected_pods=None):
    """Gate for every entry that follows a possibly failed one: refuse unless health confirms recovery."""
    h = health(expected_pods)
    if not h["ok"]:
        raise SystemExit(f"refusing to start: cluster not healthy ({h['reasons']}); run rollback.sh and health.py first")
    return h


def evaluate(results, aof0):
    """Pure decision on one concurrent sweep over all pods -> (trip_reason or None, snapshot dict). Unit-tested."""
    worst = max(r["frac"] for r in results)
    aof = sum(r["aof_size"] for r in results)
    active = [r["pod"] for r in results if r["aof_rewrite"] or r["bgsave"]]
    snap = {"worst_frac": worst, "aof_total": aof, "persist_active": active}
    if worst > MEM_ABORT_FRAC:
        return "memory", snap
    if active and worst > MEM_START_FRAC:
        return "persist-spike", snap  # fork / rewrite while already close to the limit: COW would OOM-kill the pod
    if aof0 is not None and (aof - aof0 > AOF_GROWTH_ABORT or aof > AOF_ABS_ABORT):
        return "aof-growth", snap
    return None, snap


class MemWatch:
    """Background thread, per-pod checks run concurrently every `every` s. On a trip (memory > 95 % of the limit, fork or AOF
    rewrite near the limit, AOF growth, or ANY unreadable/late check = fail closed) it writes MEM_TRIPPED to the run dir
    (which stops all scaling: require_no_trip), terminates `proc` (the load generator) and calls `on_trip`."""

    def __init__(self, every=1.0, on_trip=None):
        import threading
        self.every, self.proc, self.on_trip = every, None, on_trip
        self.stop, self.tripped, self.peak, self.aof0 = threading.Event(), False, 0.0, None
        self.expect_pods = False  # set True once pods must exist (a load level or scaling step is running)
        self.pool = ThreadPoolExecutor(max_workers=16)
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _trip(self, why, **kw):
        if self.tripped:
            return
        self.tripped = True
        try:
            with open(trip_marker(), "w") as f:
                f.write(why)
            event("memory_abort", why=why, **kw)
        except Exception:
            pass
        try:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
            if self.on_trip:
                self.on_trip()
        except Exception:
            pass

    def _step(self, state):
        """One concurrent sweep. A pod removed since the cached list must not trip: on failure refresh the pod list and
        retry once; only a failure with a fresh list is a trip."""
        if time.time() - state["t_pods"] > 3 or not state["pods"]:
            state["pods"], state["t_pods"] = redis_pods(), time.time()
        if not state["pods"]:
            return []            # empty list: "no pods to watch" (the caller decides whether that is a trip)
        try:
            results = check_all(watchable(state["pods"], state), self.pool)
        except Exception:
            state["pods"], state["t_pods"] = redis_pods(), time.time()   # raises -> caller trips (cannot even list pods)
            if not state["pods"]:
                return []
            results = check_all(watchable(state["pods"], state), self.pool)
        return results

    def _run(self):
        """Never dies silently: any exception in a cycle (listing, checking, evaluating) records a trip."""
        state, self.n, self.active_prev = {"pods": [], "t_pods": 0.0}, 0, False
        while not self.stop.wait(self.every):
            try:
                self._cycle(state)
            except BaseException as e:  # noqa: BLE001 - the watcher must report, not vanish
                self._trip("watcher-error", error=repr(e)[:200])

    def _cycle(self, state):
        try:
            results = self._step(state)
        except Exception as e:  # unreadable or late = unsafe
            self._trip("unreadable", error=str(e)[:200])
            return
        if not results:
            if self.expect_pods and not state["pods"]:   # pods exist but none Ready past grace: nothing to read yet, not a trip
                self._trip("pods-vanished", note="empty pod list while a level / scaling step is running")
            return
        why, snap = evaluate(results, self.aof0)
        if self.aof0 is None:
            self.aof0 = snap["aof_total"]
        self.peak = max(self.peak, snap["worst_frac"])
        if bool(snap["persist_active"]) != self.active_prev:
            self.active_prev = bool(snap["persist_active"])
            event("persist_activity", active=snap["persist_active"], worst_frac=snap["worst_frac"])
        self.n += 1
        if self.n % 15 == 0:
            event("mem_telemetry", **snap)
        if why:
            self._trip(why, **snap)

    def close(self):
        self.stop.set()
        self.pool.shutdown(wait=False)


def recover_after_trip(timeout=1800):
    """After a trip: run the idempotent reconcile (no further scaling by the experiment). Returns its exit code."""
    here = os.path.dirname(os.path.abspath(__file__))
    marker = trip_marker()
    saved = open(marker).read() if os.path.exists(marker) else None
    if saved is not None:
        os.remove(marker)  # reconcile itself must be allowed to scale; the experiment is not
    import sys
    rc = subprocess.run([sys.executable, os.path.join(here, "reconcile.py"), "--target", "3"], timeout=timeout).returncode
    event("recovery", rc=rc, trip=saved)
    if rc != 0 and saved is not None:
        open(marker, "w").write(saved)
    return rc


def _proc_children(pid):
    kids = set()
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                f = _read(f"/proc/{d}/stat")
                kids.add((int(d), int(f[f.rindex(")") + 2:].split()[1])))
            except (OSError, ValueError):
                pass
    tree, stack = set(), [pid]
    while stack:
        p = stack.pop()
        tree.add(p)
        stack += [c for c, pp in kids if pp == p and c not in tree]
    return tree


def _cpu_ticks(pid):
    x = _read(f"/proc/{pid}/stat")
    x = x[x.rindex(")") + 2:].split()
    return int(x[11]) + int(x[12])


class HostWatch:
    """Writes $QRL_RUN_DIR/telemetry.jsonl every `every` s: host busy cores, generator cores in total AND the busiest single
    generator process (`gen_worker_max_cores`; a CPU-bound worker invalidates a level even if the average is low), Redis pod
    cores (cgroup cpuacct), the number of pods and host cpus. fit.py uses it to reject levels."""

    def __init__(self, every=2.0):
        import threading
        self.every, self.gen_pid, self.stop = every, None, threading.Event()
        self.ncpu = os.cpu_count() or 1
        self.hz = os.sysconf("SC_CLK_TCK")
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _host(self):
        v = list(map(int, _read("/proc/stat").splitlines()[0].split()[1:]))
        return sum(v) - v[3] - v[4], time.time()  # busy ticks = total - idle - iowait

    def _gen(self):
        out = {}
        if self.gen_pid:
            for p in _proc_children(self.gen_pid):
                try:
                    out[p] = _cpu_ticks(p)
                except (OSError, ValueError, IndexError):
                    pass
        return out

    def _pods(self):
        tot, n = 0, 0
        for p in redis_pods():
            v = pod_cpu_ns(p)
            if v is None:
                return None, n
            tot, n = tot + v, n + 1
        return tot, n

    def _throttle(self):
        """Summed cumulative throttling counters over all pods (None if any pod is unreadable) + the CFS quota / period of the first pod."""
        tot = {"nr_periods": 0, "nr_throttled": 0, "throttled_ns": 0}
        first = None
        for p in redis_pods():
            v = pod_cpu_stat(p)
            if v is None:
                return None
            first = first or v
            for k in tot:
                tot[k] += v[k]
        return (tot, first) if first else None

    def _run(self):
        h0, g0 = self._host(), self._gen()
        try:
            p0, _ = self._pods()
        except Exception:
            p0 = None
        try:
            th0 = self._throttle()
        except Exception:
            th0 = None
        while not self.stop.wait(self.every):
            try:
                h1, g1 = self._host(), self._gen()
                try:
                    p1, npods = self._pods()
                except Exception:
                    p1, npods = None, 0
                try:
                    th1 = self._throttle()
                except Exception:
                    th1 = None
                dt = h1[1] - h0[1]
                per = [max(0, g1[p] - g0.get(p, g1[p])) / self.hz / dt for p in g1]
                rec = {"t": h1[1], "ncpu": self.ncpu, "host_busy_cores": (h1[0] - h0[0]) / self.hz / dt,
                       "gen_cores": sum(per), "gen_worker_max_cores": max(per, default=0.0), "n_pods": npods,
                       "pods_cores": (p1 - p0) / 1e9 / dt if (p1 is not None and p0 is not None) else None}
                if th0 and th1 and all(th1[0][k] >= th0[0][k] for k in th0[0]):   # a pod change resets sums: skip that interval
                    rec.update({"cfs_quota_us": th1[1]["quota_us"], "cfs_period_us": th1[1]["period_us"],
                                "d_nr_periods": th1[0]["nr_periods"] - th0[0]["nr_periods"],
                                "d_nr_throttled": th1[0]["nr_throttled"] - th0[0]["nr_throttled"],
                                "d_throttled_s": (th1[0]["throttled_ns"] - th0[0]["throttled_ns"]) / 1e9})
                with open(os.path.join(run_dir(), "telemetry.jsonl"), "a") as f:
                    f.write(json.dumps(rec) + "\n")
                h0, g0, p0, th0 = h1, g1, p1, th1
            except Exception:
                continue

    def close(self):
        self.stop.set()


class Children:
    """Track child processes; terminate and reap them in `finally` (also on SIGTERM from the launcher)."""

    def __init__(self):
        self.procs = []

    def popen(self, cmd, **kw):
        p = subprocess.Popen(cmd, **kw)
        self.procs.append(p)
        return p

    def reap(self, grace=10.0):
        import signal
        for p in self.procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        t_end = time.time() + grace
        for p in self.procs:
            try:
                p.wait(max(0.1, t_end - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        self.procs = []


def install_sigterm_as_exception():
    import signal

    def h(signum, frame):
        raise KeyboardInterrupt("SIGTERM")
    signal.signal(signal.SIGTERM, h)
