#!/usr/bin/env python3
"""Open-loop Redis Cluster load generator (no third-party packages: asyncio + raw RESP).

Arrivals are Poisson at a target rate, scheduled independent of responses (no coordinated omission): latency is
completion time minus the INTENDED send time. Client-side latency, completed, errors, timeouts and client drops
are recorded in 1 s windows to $QRL_RUN_DIR/metrics.jsonl (PLAN section 7: {"step": n, numeric metrics}) and
$QRL_RUN_DIR/progress.json. Seeded: --seed fixes keys, op mix and arrival times (per worker: seed*1000+worker).
Host tools found 2026-10-16: redis-cli and redis-benchmark 7.x on the host; redis-py is NOT installed, hence raw RESP.
Run from the host (pod IPs are routable from it); runs on the same 40-core node as the pods, so keep
workers <= 12 and read generator_lag_ms: points with lag > 50 ms are generator-limited, not Redis-limited.

Metrics fields: warm=1 first --warmup s of each level, warm=2 / level=-1 drain windows after the schedule (ignore both).
Schedule: --schedule "rate:seconds[,rate:seconds...]"  (ops/s per level, all workers together).
Read-only?  NO: it SETs keys 'cal:<id>' (--prefill and the SET share of the mix). Keys are removed by cleanup_keys.py.
"""
import argparse
import asyncio
import json
import math
import multiprocessing as mp
import queue
import os
import random
import subprocess
import sys
import time
from collections import deque

NB = 400
BASE = 1.05
LAT0 = 1e-5  # histogram: bucket i covers [LAT0*BASE**i, LAT0*BASE**(i+1)) seconds


def crc16(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def key_slot(key: bytes) -> int:
    s = key.find(b"{")
    if s >= 0:
        e = key.find(b"}", s + 1)
        if e > s + 1:
            key = key[s + 1:e]
    return crc16(key) % 16384


def bucket(lat_s):
    if lat_s <= LAT0:
        return 0
    return min(NB - 1, int(math.log(lat_s / LAT0) / math.log(BASE)))


def hist_quantile(h, q):
    n = sum(h)
    if n == 0:
        return float("nan")
    target, acc = q * n, 0
    for i, c in enumerate(h):
        acc += c
        if acc >= target:
            return LAT0 * BASE ** (i + 0.5)  # bucket midpoint (geometric): +-2.5 % precision
    return LAT0 * BASE ** NB


def enc(*parts):
    out = [b"*%d\r\n" % len(parts)]
    for p in parts:
        out.append(b"$%d\r\n%s\r\n" % (len(p), p))
    return b"".join(out)


def parse_reply(buf, pos):
    """-> (kind, value, newpos) or None if incomplete. kind in '+', '-', ':', '$'."""
    if pos >= len(buf):
        return None
    end = buf.find(b"\r\n", pos)
    if end < 0:
        return None
    t = chr(buf[pos])
    line = bytes(buf[pos + 1:end])
    if t in "+-:":
        return t, line, end + 2
    if t == "$":
        n = int(line)
        if n < 0:
            return t, None, end + 2
        if len(buf) < end + 2 + n + 2:
            return None
        return t, bytes(buf[end + 2:end + 2 + n]), end + 2 + n + 2
    raise ValueError(f"unexpected reply type {t!r}")


CONTEXT = "kubernetes-admin@kubernetes"
NAMESPACE = "qrl-calib"  # hard-coded: the generator may only talk to pods of this namespace
LABEL = "redis=redis-cluster-pods-lbl"
if os.environ.get("QRL_NAMESPACE", NAMESPACE) != NAMESPACE:
    raise SystemExit(f"refusing: QRL_NAMESPACE must be {NAMESPACE!r}")


def allowed_ips():
    """IPs of the Redis pods in namespace qrl-calib (explicit context and namespace). Anything else is refused."""
    out = subprocess.run(["kubectl", "--context", CONTEXT, "--namespace", NAMESPACE, "get", "pods", "-l", LABEL, "-o",
                          "jsonpath={.items[*].status.podIP}"], capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise SystemExit(f"cannot list pods of {NAMESPACE}: {out.stderr.strip()[:200]}")
    return set(out.stdout.split())


def check_slotmap(m, allowed):
    bad = sorted({addr.rsplit(":", 1)[0] for addr in m.values()} - set(allowed))
    if bad:
        raise RuntimeError(f"cluster view lists endpoints outside {NAMESPACE}: {bad}")
    return m


def fetch_slotmap(seed_ip, port=6379):
    """{slot: 'ip:port'} from CLUSTER NODES via redis-cli (masters only, skipping failed nodes)."""
    out = subprocess.run(["redis-cli", "-h", seed_ip, "-p", str(port), "cluster", "nodes"], capture_output=True,
                         text=True, timeout=10).stdout
    m = {}
    for line in out.strip().splitlines():
        f = line.split()
        if len(f) < 8 or "master" not in f[2] or "fail" in f[2]:
            continue
        addr = f[1].split("@")[0]
        if "myself" in f[2] and addr.startswith(":"):   # fresh node: own address not yet known, it is the node we asked
            addr = seed_ip + addr
        for s in f[8:]:
            if s.startswith("["):
                continue
            lo, _, hi = s.partition("-")
            for sl in range(int(lo), int(hi or lo) + 1):
                m[sl] = addr
    return m


def seed_ip_from_kubectl():
    ips = sorted(allowed_ips())
    if not ips:
        raise SystemExit("no Redis pods found")
    return ips[0]


class Conn:
    __slots__ = ("addr", "outq", "pending", "buf", "wake", "dead", "reader", "writer", "tasks")

    def __init__(self, addr):
        self.addr, self.outq, self.pending, self.buf = addr, deque(), deque(), bytearray()
        self.wake, self.dead, self.reader, self.writer, self.tasks = asyncio.Event(), False, None, None, []


E_T, E_CMD, E_RETRY, E_DONE, E_SLOT, E_MARK, E_ASK = range(7)


class Worker:
    """Scheduling is decoupled from transport: the arrival loop only draws times, picks keys and appends entries to
    in-memory queues (no awaits except its own sleep). Per-connection sender tasks connect, write and drain.
    Every entry [t_intended, cmd, retries, done, slot, is_asking_marker, ask] is resolved exactly once as ok, err,
    timeout or drop (`resolve`), so offered == ok + err + timeout + drop after final settlement."""

    def __init__(self, wid, a, t0, schedule, slotmap, q, stop=None):
        self.wid, self.a, self.t0, self.schedule, self.slotmap, self.q = wid, a, t0, schedule, dict(slotmap), q
        self.stop = stop
        self.allowed = set(getattr(a, "allowed", None) or ())
        self.rng = random.Random(a.seed * 1000 + wid)
        self.conns, self.bad_until = {}, {}
        self.win = {}
        self.flushed = -1
        self.live = 0  # real (non-marker) entries not yet resolved
        self.done_gen = False
        self.refresh_needed = False
        self.total = sum(d for _, d in schedule)
        self.keys = [b"cal:%08d" % i + b"x" * max(0, a.key_size - 12) for i in range(a.keys)]
        self.slots = [key_slot(k) for k in self.keys]
        self.value = b"v" * a.value_size
        self.get_cmds = [enc(b"GET", k) for k in self.keys]
        self.set_cmds = [enc(b"SET", k, self.value) for k in self.keys]

    # ---- accounting
    def w(self, t):
        i = max(int((t - self.t0) / self.a.window), self.flushed + 1)  # late events go to the first open window
        d = self.win.get(i)
        if d is None:
            d = self.win[i] = {"offered": 0, "ok": 0, "err": 0, "timeout": 0, "drop": 0, "redirect": 0,
                               "hist": [0] * NB, "lag": 0.0, "late": 0, "refused": 0}
        return d

    def resolve(self, e, kind, now):
        if e[E_MARK] or e[E_DONE]:
            return
        e[E_DONE] = True
        self.live -= 1
        d = self.w(now)
        d[kind] += 1
        if kind == "ok":
            d["hist"][bucket(now - e[E_T])] += 1

    # ---- transport
    def get_conn(self, addr, now):
        c = self.conns.get(addr)
        if c is not None and not c.dead:
            return c
        if self.bad_until.get(addr, 0) > now:
            return None
        c = Conn(addr)
        self.conns[addr] = c
        c.tasks.append(asyncio.ensure_future(self.sender(c)))
        return c

    def kill_conn(self, c, now=None):
        now = time.monotonic() if now is None else now
        if c.dead:
            return
        c.dead = True
        self.bad_until[c.addr] = now + 0.5
        if self.conns.get(c.addr) is c:
            del self.conns[c.addr]
        for q in (c.outq, c.pending):
            while q:
                self.resolve(q.popleft(), "err", now)
        if c.writer is not None:
            c.writer.close()
        for t in c.tasks:
            if t is not asyncio.current_task():
                t.cancel()

    async def sender(self, c):
        host, port = c.addr.rsplit(":", 1)
        try:
            c.reader, c.writer = await asyncio.wait_for(asyncio.open_connection(host, int(port)), self.a.connect_timeout)
        except (OSError, asyncio.TimeoutError):
            self.kill_conn(c)
            return
        c.tasks.append(asyncio.ensure_future(self.reader_loop(c)))
        try:
            while not c.dead:
                if not c.outq:
                    c.wake.clear()
                    await c.wake.wait()
                    continue
                chunks, n = [], 0
                while c.outq and n < 500:
                    e = c.outq.popleft()
                    n += 1
                    if e[E_DONE]:
                        continue  # timed out while queued
                    if e[E_ASK]:
                        c.pending.append([e[E_T], b"", 0, True, e[E_SLOT], True, False])
                        chunks.append(enc(b"ASKING"))
                        e[E_ASK] = False
                    c.pending.append(e)
                    chunks.append(e[E_CMD])
                if chunks:
                    c.writer.write(b"".join(chunks))
                    await c.writer.drain()
        except (OSError, asyncio.CancelledError):
            pass
        finally:
            self.kill_conn(c)

    async def reader_loop(self, c):
        try:
            while not c.dead:
                data = await c.reader.read(1 << 18)
                if not data:
                    break
                c.buf += data
                pos, now = 0, time.monotonic()
                while True:
                    r = parse_reply(c.buf, pos)
                    if r is None:
                        break
                    kind, val, pos = r
                    self.on_reply(c, kind, val, now)
                del c.buf[:pos]
        except (OSError, asyncio.CancelledError):
            pass
        finally:
            self.kill_conn(c)

    def on_reply(self, c, kind, val, now):
        if not c.pending:
            return
        e = c.pending.popleft()
        if e[E_MARK]:  # ASKING reply
            return
        if e[E_DONE]:  # already settled as a timeout. Only a SUCCESSFUL reply is late server work; an error or MOVED/ASK is not
            if kind != "-":
                self.w(now)["late"] += 1
            return
        if now - e[E_T] > self.a.timeout:  # expired before this reply: a timeout; only a SUCCESS is late server work
            if kind != "-":
                self.w(now)["late"] += 1
            self.resolve(e, "timeout", now)
            return
        if kind == "-" and (val.startswith(b"MOVED") or val.startswith(b"ASK")) and e[E_RETRY] < 2:
            parts = val.split()
            self.w(now)["redirect"] += 1
            target = parts[2].decode()
            if target.rsplit(":", 1)[0] not in self.allowed:  # redirect outside qrl-calib: refuse, never connect or remap
                self.w(now)["refused"] += 1
                self.refresh_needed = True
                self.resolve(e, "err", now)
                return
            if val.startswith(b"MOVED"):
                self.slotmap[int(parts[1])] = target
                self.refresh_needed = True
            e[E_RETRY] += 1
            e[E_ASK] = val.startswith(b"ASK")
            tc = self.get_conn(target, now)
            if tc is None:
                self.resolve(e, "err", now)
            else:
                tc.outq.append(e)  # same entry, same live slot: not a new arrival
                tc.wake.set()
            return
        if kind == "-":
            self.resolve(e, "err", now)
        else:
            self.resolve(e, "ok", now)

    def sweep_timeouts(self, now):
        for c in list(self.conns.values()):
            for q in (c.outq, c.pending):
                for e in q:
                    if e[E_DONE]:
                        continue
                    if now - e[E_T] > self.a.timeout:
                        self.resolve(e, "timeout", now)
                    else:
                        break
            while c.outq and c.outq[0][E_DONE]:
                c.outq.popleft()

    async def sweeper(self):
        while not self.done_gen or self.live > 0:
            await asyncio.sleep(0.5)
            now = time.monotonic()
            self.sweep_timeouts(now)
            self.flush(now, final=False)

    async def refresher(self):
        while not self.done_gen:
            await asyncio.sleep(2.0 if not self.refresh_needed else 0.5)
            if self.wid == 0 or self.refresh_needed:
                try:
                    loop = asyncio.get_event_loop()
                    allowed = await loop.run_in_executor(None, allowed_ips)
                    m = await loop.run_in_executor(None, fetch_slotmap, self.a.seed_ip)
                    if len(m) == 16384:
                        self.slotmap = check_slotmap(m, allowed)
                        self.allowed = set(allowed)
                except Exception:
                    pass
                self.refresh_needed = False

    def flush(self, now, final):
        cur = int((now - self.t0) / self.a.window)
        upto = max(self.win) if final and self.win else cur - 2
        for i in sorted(k for k in self.win if k <= upto):
            self.q.put((self.wid, i, self.win.pop(i)))
            self.flushed = max(self.flushed, i)

    def level_rate(self, t):
        x = t - self.t0
        for r, d in self.schedule:
            if x < d:
                return r
            x -= d
        return 0.0

    def arrive(self, t, now, r):
        rng, a = self.rng, self.a
        k = rng.randrange(a.keys)
        is_get = rng.random() < a.get_ratio
        d = self.w(t)
        d["offered"] += 1
        d["lag"] = max(d["lag"], now - t)
        e = [t, (self.get_cmds if is_get else self.set_cmds)[k], 0, False, self.slots[k], False, False]
        self.live += 1
        addr = self.slotmap.get(self.slots[k])
        c = self.get_conn(addr, now) if addr else None
        if c is None:
            self.resolve(e, "err", now)
        elif self.live > a.max_outstanding:
            self.resolve(e, "drop", now)
        else:
            c.outq.append(e)
            c.wake.set()

    async def run(self):
        a = self.a
        asyncio.ensure_future(self.sweeper())
        asyncio.ensure_future(self.refresher())
        t = self.t0
        await asyncio.sleep(max(0, self.t0 - time.monotonic()))
        end = self.t0 + self.total
        while t < end:
            now = time.monotonic()
            if self.stop is not None and self.stop.is_set():
                break
            batch = 0
            while t <= now and t < end and batch < 5000:
                r = self.level_rate(t) / a.workers
                if r <= 0:
                    t += 0.001
                    continue
                self.arrive(t, now, r)
                t += self.rng.expovariate(r)
                batch += 1
            await asyncio.sleep(min(max(min(t, end) - time.monotonic(), 0.0), 0.002))
        self.done_gen = True
        # final settlement: wait for replies up to timeout, then settle everything left as a timeout
        dl = time.monotonic() + a.timeout + 1
        while self.live > 0 and time.monotonic() < dl:
            await asyncio.sleep(0.1)
            if self.stop is not None and self.stop.is_set() and time.monotonic() > dl - a.timeout + 2:
                break
        now = time.monotonic()
        for c in list(self.conns.values()):
            for q in (c.outq, c.pending):
                for e in q:
                    self.resolve(e, "timeout", now)
        self.flush(now, final=True)
        self.q.put((self.wid, None, {"live_left": self.live}))


def worker_main(wid, a, t0, schedule, slotmap, q, stop):
    import signal
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    asyncio.run(Worker(wid, a, t0, schedule, slotmap, q, stop).run())


def prefill(a, slotmap):
    """Sequential pipelined SETs of all keys through raw sockets (closed loop, not part of the measurement)."""
    import socket
    value = b"v" * a.value_size
    by_node = {}
    for i in range(a.keys):
        k = b"cal:%08d" % i + b"x" * max(0, a.key_size - 12)
        by_node.setdefault(slotmap[key_slot(k)], []).append(enc(b"SET", k, value))
    for addr, cmds in by_node.items():
        host, port = addr.rsplit(":", 1)
        s = socket.create_connection((host, int(port)), timeout=10)
        for i in range(0, len(cmds), 200):
            chunk = cmds[i:i + 200]
            s.sendall(b"".join(chunk))
            got, buf = 0, bytearray()
            while got < len(chunk):
                buf += s.recv(65536)
                pos = 0
                while (r := parse_reply(buf, pos)) is not None:
                    if r[0] == "-":
                        raise SystemExit(f"prefill error from {addr}: {r[1]!r}")
                    got, pos = got + 1, r[2]
                del buf[:pos]
        s.close()
    print(f"prefilled {a.keys} keys on {len(by_node)} nodes", flush=True)


def parse_schedule(s):
    out = []
    for part in s.split(","):
        r, d = part.split(":")
        out.append((float(r), float(d)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--seed", type=int, default=int(os.environ.get("QRL_RUN_SEED") or 0))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keys", type=int, default=2000, help="dataset size (keys); memory limit per pod is 16Mi")
    ap.add_argument("--key-size", type=int, default=16)
    ap.add_argument("--value-size", type=int, default=64)
    ap.add_argument("--get-ratio", type=float, default=0.95)
    ap.add_argument("--timeout", type=float, default=10.0, help="client timeout (s); late answers count as timeouts")
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--connect-timeout", type=float, default=3.0)
    ap.add_argument("--max-outstanding", type=int, default=5000, help="per worker; beyond it requests are dropped")
    ap.add_argument("--seed-ip", help="IP of any cluster node (default: first Redis pod via kubectl get)")
    ap.add_argument("--prefill", action="store_true")
    ap.add_argument("--phase", type=float, default=0, help="numeric tag copied into every metrics line")
    ap.add_argument("--level-base", type=int, default=0, help="level index offset for this invocation")
    ap.add_argument("--pods", type=float, default=float("nan"), help="numeric tag: pods during this schedule")
    ap.add_argument("--warmup", type=float, default=10.0, help="first seconds of every level flagged warm=1")
    a = ap.parse_args(argv)
    allowed = allowed_ips()
    a.seed_ip = a.seed_ip or seed_ip_from_kubectl()
    if a.seed_ip not in allowed:
        raise SystemExit(f"refusing seed endpoint {a.seed_ip}: not a pod of namespace {NAMESPACE}")
    a.allowed = allowed
    schedule = parse_schedule(a.schedule)
    slotmap = fetch_slotmap(a.seed_ip)
    if len(slotmap) != 16384:
        raise SystemExit(f"slots covered {len(slotmap)}/16384: refusing to generate load")
    check_slotmap(slotmap, allowed)
    if a.prefill:
        prefill(a, slotmap)
    rd = os.environ.get("QRL_RUN_DIR", ".")
    os.makedirs(rd, exist_ok=True)
    mpath = os.path.join(rd, "metrics.jsonl")
    step0 = sum(1 for _ in open(mpath)) if os.path.exists(mpath) else 0
    total = sum(d for _, d in schedule)
    q = mp.Queue()
    stop = mp.Event()
    import signal
    signal.signal(signal.SIGTERM, lambda *_: stop.set())  # graceful: stop generating, settle, flush, exit 0
    t0 = time.monotonic() + 1.5
    wall0 = time.time() + 1.5
    ps = [mp.Process(target=worker_main, args=(i, a, t0, schedule, slotmap, q, stop)) for i in range(a.workers)]
    for p in ps:
        p.start()
    agg, finished, nw = {}, 0, a.workers
    done_through = -1
    out = open(mpath, "a")

    def emit(i, d):
        t_rel = i * a.window
        x, lvl, within = t_rel, 0, 0.0
        for j, (r, dur) in enumerate(schedule):
            if x < dur:
                lvl, within = j, x
                break
            x -= dur
        else:
            lvl, within = len(schedule) - 1, schedule[-1][1]
        h = d["hist"]
        tail = t_rel >= total  # drain windows after the schedule ended: late completions only
        rec = {"step": step0 + i, "t": wall0 + t_rel, "phase": a.phase, "level": -1 if tail else a.level_base + lvl,
               "target_rate": schedule[lvl][0], "warm": 2 if tail else (1 if within < a.warmup else 0), "pods": a.pods,
               "offered_ops": d["offered"] / a.window, "completed_ops": d["ok"] / a.window, "late_ops": d["late"] / a.window,
               "served_ops": (d["ok"] + d["late"]) / a.window,
               "error_ops": d["err"] / a.window, "timeout_ops": d["timeout"] / a.window,
               "drop_ops": d["drop"] / a.window, "redirects": d["redirect"] / a.window, "refused_ops": d["refused"] / a.window,
               "failed_ops": (d["err"] + d["timeout"] + d["drop"]) / a.window,
               "p50_ms": 1e3 * hist_quantile(h, 0.50), "p95_ms": 1e3 * hist_quantile(h, 0.95),
               "p99_ms": 1e3 * hist_quantile(h, 0.99), "generator_lag_ms": 1e3 * d["lag"], "workers": a.workers,
               "hist": {str(j): n for j, n in enumerate(h) if n}}  # sparse; fit.py merges these, never averages quantiles
        out.write(json.dumps(rec) + "\n")
        out.flush()
        with open(os.path.join(rd, "progress.json"), "w") as f:
            json.dump({"step": min(i + 1, int(total / a.window)), "total": int(total / a.window)}, f)

    got = {}
    ok = False
    try:
        while finished < nw:
            try:
                wid, i, d = q.get(timeout=1.0)
            except queue.Empty:
                if not any(p.is_alive() for p in ps):  # every worker is gone and the queue is drained: stop waiting
                    try:
                        wid, i, d = q.get(timeout=0.5)
                    except queue.Empty:
                        break
                else:
                    continue
            if i is None:
                finished += 1
                continue
            m = agg.setdefault(i, {"offered": 0, "ok": 0, "err": 0, "timeout": 0, "drop": 0, "redirect": 0,
                                   "hist": [0] * NB, "lag": 0.0, "late": 0, "refused": 0})
            for k in ("offered", "ok", "err", "timeout", "drop", "redirect", "late", "refused"):
                m[k] += d[k]
            m["lag"] = max(m["lag"], d["lag"])
            m["hist"] = [x + y for x, y in zip(m["hist"], d["hist"])]
            got[i] = got.get(i, 0) + 1
            while got.get(done_through + 1, 0) >= nw:
                done_through += 1
                emit(done_through, agg.pop(done_through))
        for i in sorted(agg):  # windows some worker never reported (late completions): emit what we have
            emit(i, agg[i])
        ok = True
    finally:
        if not ok:
            stop.set()  # abnormal exit (exception, KeyboardInterrupt): tell workers to settle and leave
        for p in ps:
            p.join(timeout=a.timeout + 15)
            if p.is_alive():
                p.terminate()
                p.join(5)
                if p.is_alive():
                    p.kill()
                    p.join()
        out.close()
        q.close()
    bad = [i for i, p in enumerate(ps) if p.exitcode != 0]
    if bad:
        raise SystemExit(f"load generator workers {bad} failed (exit codes {[ps[i].exitcode for i in bad]}); metrics incomplete")


if __name__ == "__main__":
    main()
