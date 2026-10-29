"""Loadgen tests against an in-process fake RESP server (no cluster, no kubectl)."""
import asyncio
import pytest
import json
import threading

import loadgen


def test_crc16_and_slot_vectors():
    assert loadgen.crc16(b"123456789") == 0x31C3  # standard XMODEM check value
    assert loadgen.key_slot(b"foo") == 12182      # documented Redis Cluster slot of "foo"
    assert loadgen.key_slot(b"{foo}bar") == 12182  # hash tag


def test_parse_reply():
    assert loadgen.parse_reply(bytearray(b"+OK\r\n"), 0) == ("+", b"OK", 5)
    assert loadgen.parse_reply(bytearray(b"$3\r\nabc\r\n"), 0) == ("$", b"abc", 9)
    assert loadgen.parse_reply(bytearray(b"$3\r\nab"), 0) is None
    assert loadgen.parse_reply(bytearray(b"$-1\r\n"), 0)[1] is None


def test_hist_quantile():
    h = [0] * loadgen.NB
    for _ in range(100):
        h[loadgen.bucket(0.002)] += 1
    assert abs(loadgen.hist_quantile(h, 0.5) - 0.002) / 0.002 < 0.06


def start_fake(delay=0.0, moved_to=None):
    state = {}

    async def handle(r, w):
        buf = b""
        try:
            while True:
                data = await r.read(65536)
                if not data:
                    break
                buf += data
                while buf.startswith(b"*"):
                    # parse one command
                    try:
                        n = int(buf[1:buf.index(b"\r\n")])
                        pos = buf.index(b"\r\n") + 2
                        parts = []
                        for _ in range(n):
                            ln = int(buf[pos + 1:buf.index(b"\r\n", pos)])
                            s = buf.index(b"\r\n", pos) + 2
                            parts.append(buf[s:s + ln])
                            pos = s + ln + 2
                    except ValueError:
                        break
                    buf = buf[pos:]
                    if delay:
                        await asyncio.sleep(delay)
                    cmd = parts[0].upper()
                    if cmd == b"SET":
                        w.write(b"+OK\r\n")
                    elif cmd == b"GET":
                        w.write(b"$1\r\nv\r\n")
                    else:
                        w.write(b"+OK\r\n")
        except ConnectionError:
            pass

    ready = threading.Event()

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        srv = loop.run_until_complete(asyncio.start_server(handle, "127.0.0.1", 0))
        state["port"] = srv.sockets[0].getsockname()[1]
        ready.set()
        loop.run_forever()

    threading.Thread(target=run, daemon=True).start()
    ready.wait()
    return state["port"]


def test_open_loop_rate_and_accounting(tmp_path, monkeypatch):
    port = start_fake()
    addr = f"127.0.0.1:{port}"
    monkeypatch.setattr(loadgen, "allowed_ips", lambda: {"127.0.0.1"})
    monkeypatch.setattr(loadgen, "fetch_slotmap", lambda ip, port=6379: {s: addr for s in range(16384)})
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path))
    loadgen.main(["--schedule", "500:3,1500:3", "--seed", "1", "--workers", "2", "--seed-ip", "127.0.0.1", "--keys", "200",
                  "--timeout", "2", "--warmup", "1", "--pods", "3"])
    rows = [json.loads(line) for line in open(tmp_path / "metrics.jsonl")]
    assert [r["step"] for r in rows] == list(range(len(rows)))
    assert len(rows) >= 6
    lvl0 = [r for r in rows if r["level"] == 0 and r["warm"] == 0]
    lvl1 = [r for r in rows if r["level"] == 1 and r["warm"] == 0]
    m0 = sum(r["offered_ops"] for r in lvl0) / len(lvl0)
    m1 = sum(r["offered_ops"] for r in lvl1) / len(lvl1)
    assert 350 < m0 < 650 and 1100 < m1 < 1900  # Poisson, 2 workers
    done = sum(r["completed_ops"] for r in rows)
    off = sum(r["offered_ops"] for r in rows)
    assert done / off > 0.97
    assert all(r["failed_ops"] == 0 for r in rows)
    assert all(r["p95_ms"] < 200 for r in rows if r["completed_ops"] > 0)
    assert json.load(open(tmp_path / "progress.json"))["total"] == 6


def test_timeouts_counted_when_server_slow(tmp_path, monkeypatch):
    port = start_fake(delay=0.05)  # 20 ops/s per connection service rate (serial)
    addr = f"127.0.0.1:{port}"
    monkeypatch.setattr(loadgen, "allowed_ips", lambda: {"127.0.0.1"})
    monkeypatch.setattr(loadgen, "fetch_slotmap", lambda ip, port=6379: {s: addr for s in range(16384)})
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path))
    loadgen.main(["--schedule", "200:3", "--seed", "2", "--workers", "1", "--seed-ip", "127.0.0.1", "--keys", "50",
                  "--timeout", "0.5", "--warmup", "0"])
    rows = [json.loads(line) for line in open(tmp_path / "metrics.jsonl")]
    assert sum(r["timeout_ops"] for r in rows) > 0
    assert sum(r["completed_ops"] for r in rows) < 0.5 * sum(r["offered_ops"] for r in rows)
    # late replies are server-side work: served = ok + late, and late > 0 here
    assert sum(r["late_ops"] for r in rows) > 0
    assert all(abs(r["served_ops"] - r["completed_ops"] - r["late_ops"]) < 1e-9 for r in rows)


# ---------------------------------------------------------------- ASK accounting, decoupling, conservation
import argparse
import queue
import time


def start_server(reply):
    """reply(parts, state) -> bytes. Returns (port, state)."""
    state = {"cmds": []}

    async def handle(r, w):
        buf = b""
        try:
            while True:
                data = await r.read(65536)
                if not data:
                    break
                buf += data
                while buf.startswith(b"*"):
                    try:
                        n = int(buf[1:buf.index(b"\r\n")])
                        pos = buf.index(b"\r\n") + 2
                        parts = []
                        for _ in range(n):
                            ln = int(buf[pos + 1:buf.index(b"\r\n", pos)])
                            s = buf.index(b"\r\n", pos) + 2
                            parts.append(buf[s:s + ln])
                            pos = s + ln + 2
                    except ValueError:
                        break
                    buf = buf[pos:]
                    state["cmds"].append(parts[0].upper())
                    out = reply(parts, state)
                    if out is not None:
                        w.write(out)
        except ConnectionError:
            pass

    ready = threading.Event()

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        srv = loop.run_until_complete(asyncio.start_server(handle, "127.0.0.1", 0))
        state["port"] = srv.sockets[0].getsockname()[1]
        ready.set()
        loop.run_forever()

    threading.Thread(target=run, daemon=True).start()
    ready.wait()
    return state["port"], state


def make_worker(slotmap, **kw):
    a = argparse.Namespace(seed=1, keys=50, key_size=16, value_size=8, get_ratio=1.0, timeout=2.0, window=1.0,
                           max_outstanding=100000, workers=1, connect_timeout=1.0, seed_ip="127.0.0.1", allowed={"127.0.0.1"})
    for k, v in kw.items():
        setattr(a, k, v)
    return loadgen.Worker(0, a, time.monotonic(), [(100.0, 1.0)], slotmap, queue.Queue())


def totals(w):
    t = {"offered": 0, "ok": 0, "err": 0, "timeout": 0, "drop": 0}
    while not w.q.empty():
        _, i, d = w.q.get()
        if i is not None:
            for k in t:
                t[k] += d[k]
    for d in w.win.values():
        for k in t:
            t[k] += d[k]
    return t


def test_ask_redirect_counts_each_request_once():
    """Regression: an ASK made the ASKING marker decrement the outstanding counter, so it went negative."""
    pb, sb = start_server(lambda parts, s: b"+OK\r\n" if parts[0].upper() == b"ASKING" else b"$1\r\nv\r\n")
    pa, sa = start_server(lambda parts, s: b"-ASK 100 127.0.0.1:%d\r\n" % pb)

    async def go():
        w = make_worker({s: f"127.0.0.1:{pa}" for s in range(16384)})
        lows = []
        for i in range(30):
            w.arrive(time.monotonic(), time.monotonic(), 100.0)
        for _ in range(100):
            lows.append(w.live)
            if w.live == 0:
                break
            await asyncio.sleep(0.05)
        w.done_gen = True
        w.flush(time.monotonic(), final=True)
        return w, lows

    w, lows = asyncio.run(go())
    t = totals(w)
    assert min(lows) >= 0 and w.live == 0
    assert t["offered"] == 30 and t["ok"] == 30 and t["err"] == t["timeout"] == t["drop"] == 0
    assert sb["cmds"].count(b"ASKING") == 30 and sb["cmds"].count(b"GET") == 30


def test_arrivals_not_delayed_by_slow_connect_and_conserved(monkeypatch):
    real = asyncio.open_connection

    async def slow(*a, **k):
        await asyncio.sleep(1.5)
        return await real(*a, **k)
    monkeypatch.setattr(asyncio, "open_connection", slow)
    pa, sa = start_server(lambda parts, s: b"$1\r\nv\r\n")

    async def go():
        w = make_worker({s: f"127.0.0.1:{pa}" for s in range(16384)}, connect_timeout=5.0)
        lags = []
        t0 = time.monotonic()
        t = t0
        while time.monotonic() - t0 < 1.0:  # the arrival loop body, with no await except its own sleep
            now = time.monotonic()
            while t <= now:
                w.arrive(t, now, 100.0)
                lags.append(now - t)
                t += 0.005
            await asyncio.sleep(0.002)
        n = len(lags)
        w.done_gen = True
        for _ in range(100):
            if w.live == 0:
                break
            await asyncio.sleep(0.1)
        w.flush(time.monotonic(), final=True)
        return w, n, max(lags)

    w, n, lag = asyncio.run(go())
    t = totals(w)
    assert lag < 0.05                       # the connect (1.5 s) never delays scheduling
    assert t["offered"] == n and t["ok"] == n  # everything queued during the connect was sent afterwards


def test_conservation_with_refused_dropped_and_timed_out(tmp_path, monkeypatch):
    """offered == ok + err + timeout + drop over all windows, with refused connections, a server that never answers
    and a tiny outstanding cap."""
    pa, sa = start_server(lambda parts, s: None)  # never answers: timeouts
    pc, sc = start_server(lambda parts, s: b"-ERR boom\r\n")
    closed = 1  # port 1: connection refused
    slots = {s: (f"127.0.0.1:{pa}" if s < 5000 else f"127.0.0.1:{pc}" if s < 10000 else f"127.0.0.1:{closed}")
             for s in range(16384)}
    monkeypatch.setattr(loadgen, "allowed_ips", lambda: {"127.0.0.1"})
    monkeypatch.setattr(loadgen, "fetch_slotmap", lambda ip, port=6379: slots)
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path))
    loadgen.main(["--schedule", "400:3", "--seed", "3", "--workers", "2", "--seed-ip", "127.0.0.1", "--keys", "300",
                  "--timeout", "1", "--warmup", "0", "--max-outstanding", "60"])
    rows = [json.loads(line) for line in open(tmp_path / "metrics.jsonl")]
    keys = ("completed_ops", "error_ops", "timeout_ops", "drop_ops")
    off = sum(r["offered_ops"] for r in rows)
    out = sum(r[k] for r in rows for k in keys)
    assert off > 800 and abs(off - out) < 1e-6
    assert sum(r["timeout_ops"] for r in rows) > 0 and sum(r["error_ops"] for r in rows) > 0
    assert sum(r["drop_ops"] for r in rows) > 0
    assert all("hist" in r for r in rows)


def test_stop_event_ends_generation_early_and_settles():
    pa, sa = start_server(lambda parts, s: b"$1\r\nv\r\n")
    import multiprocessing as mp
    stop = mp.Event()

    async def go():
        a = make_worker({s: f"127.0.0.1:{pa}" for s in range(16384)}).a
        w = loadgen.Worker(0, a, time.monotonic() + 0.2, [(200.0, 30.0)], {s: f"127.0.0.1:{pa}" for s in range(16384)},
                           queue.Queue(), stop)
        asyncio.get_event_loop().call_later(1.2, stop.set)
        t0 = time.monotonic()
        await w.run()
        return w, time.monotonic() - t0

    w, el = asyncio.run(go())
    t = totals(w)
    assert el < 6 and w.live == 0 and t["offered"] == t["ok"] > 100


def test_late_success_counts_as_served_but_errors_and_moved_do_not(tmp_path):
    """served = ok + late SUCCESS. A reply after the timeout that is an error or MOVED/ASK is not server throughput."""
    async def run(reply):
        port, _ = start_server(reply)
        w = make_worker({s: f"127.0.0.1:{port}" for s in range(16384)}, timeout=0.2)
        for _ in range(10):
            w.arrive(time.monotonic(), time.monotonic(), 100.0)
        await asyncio.sleep(1.0)
        w.done_gen = True
        w.sweep_timeouts(time.monotonic())
        w.flush(time.monotonic(), final=True)
        return w, totals_with_late(w)

    def totals_with_late(w):
        t = {"ok": 0, "late": 0, "timeout": 0, "err": 0}
        while not w.q.empty():
            _, i, d = w.q.get()
            if i is not None:
                for k in t:
                    t[k] += d[k]
        return t

    import threading as th

    def slow(kind):
        def reply(parts, state):
            # answer after 0.5 s (> timeout 0.2 s) from a helper thread-free trick: block this handler's loop briefly
            time.sleep(0.01)
            return kind
        return reply

    # server answers late: emulate by a delay in the handler via a sync sleep per command (serial, 0.05 s each)
    for kind, expect_late in ((b"$1\r\nv\r\n", True), (b"-ERR boom\r\n", False), (b"-MOVED 1 10.9.9.9:6379\r\n", False)):
        def reply(parts, state, kind=kind):
            time.sleep(0.05)
            return kind
        w, t = asyncio.run(run(reply))
        assert t["timeout"] > 0
        assert (t["late"] > 0) == expect_late, (kind, t)
        assert t["late"] <= t["timeout"]       # a late success is a subset of the timeouts: each request counted once


def test_redirect_outside_namespace_is_refused_not_followed():
    refused_port, srefused = start_server(lambda parts, s: b"$1\r\nv\r\n")
    pa, sa = start_server(lambda parts, s: b"-MOVED 1 10.99.99.99:6379\r\n")

    async def go():
        w = make_worker({s: f"127.0.0.1:{pa}" for s in range(16384)})
        for _ in range(5):
            w.arrive(time.monotonic(), time.monotonic(), 100.0)
        for _ in range(60):
            if w.live == 0:
                break
            await asyncio.sleep(0.05)
        w.done_gen = True
        w.flush(time.monotonic(), final=True)
        return w

    w = asyncio.run(go())
    t = totals(w)
    assert t["err"] == 5 and w.live == 0 and not any("10.99.99.99" in a for a in w.conns)


def test_slotmap_and_seed_outside_allowlist_are_refused(monkeypatch):
    with pytest.raises(RuntimeError, match="outside"):
        loadgen.check_slotmap({0: "10.1.1.1:6379", 1: "10.2.2.2:6379"}, {"10.1.1.1"})
    assert loadgen.check_slotmap({0: "10.1.1.1:6379"}, {"10.1.1.1"})
    monkeypatch.setattr(loadgen, "allowed_ips", lambda: {"10.1.1.1"})
    with pytest.raises(SystemExit, match="refusing seed"):
        loadgen.main(["--schedule", "10:1", "--seed-ip", "10.9.9.9"])


def test_dead_worker_does_not_hang_main_and_fails_loudly(tmp_path, monkeypatch):
    port, _ = start_server(lambda parts, s: b"$1\r\nv\r\n")
    addr = f"127.0.0.1:{port}"
    monkeypatch.setattr(loadgen, "allowed_ips", lambda: {"127.0.0.1"})
    monkeypatch.setattr(loadgen, "fetch_slotmap", lambda ip, port=6379: {s: addr for s in range(16384)})
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path))

    def crash(*a, **k):
        raise RuntimeError("worker crashed")
    monkeypatch.setattr(loadgen, "worker_main", crash)
    t0 = time.time()
    with pytest.raises(SystemExit, match="workers"):
        loadgen.main(["--schedule", "100:2", "--workers", "2", "--seed-ip", "127.0.0.1", "--timeout", "1"])
    assert time.time() - t0 < 20
