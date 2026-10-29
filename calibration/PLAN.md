# Sim v2 calibration pilot: design (PREPARED, NOT RUN)

**Backend (decision R-20261016-018, 2026-10-16):** an isolated, stateless copy of the researcher's Deployment in namespace
`qrl-calib` (`k8s/qrl-calib.yaml`: same image, 150m/8Mi requests, 200m/16Mi limits, nodeSelector, labels; AOF and RDB off via
args; runs `redis-server /redis.conf` directly, so no coordinator). The `default` deployment is never touched. **The researcher
accepted data loss in the calibration namespace.** Scale-down is a plain replica decrement, as in 2025; whichever pod
Kubernetes removes is recorded, not judged. Rollback = delete the namespace. No StatefulSet.

Council C-20261016-01 (D2). Parameters to calibrate (`sim_v2/params.py`): mu (ops/s per pod), theta (C = mu c^theta),
latency-vs-rho curve, startup delay d, rebalance loss delta and duration k, client timeout.
Nothing here has been run on the cluster. Every real entry needs its own approval through `tools/launch_run.py`.

## 1. Read-only inspection (2026-10-16, kubectl get/describe/top, redis-cli INFO/CLUSTER INFO)
- Context `kubernetes-admin@kubernetes`, namespace `default`, API server v1.28.7 up. One node `kbitsak-very-large`
  (control-plane, label `role=redis-workers`), allocatable cpu 40, memory 201181272Ki, pods 110.
  Requests now 2650m (6 %), limits 2600m. `kubectl top`: node 1398m / 9.6 GiB; redis pods 2-3m CPU, 7-12Mi.
- Deployment `redis-autoscale-cluster`: 3/3, image `capitanodocker/redis-autoscale-cluster:0.1`
  (sha256:1d51a336...), command `/bin/bash /entrypoint.sh`, Redis 7.2.4 in **cluster mode**, AOF on, `save 3600 1 300 100 60 10000`.
  Requests 150m / 8Mi, limits 200m / 16Mi. `cluster-require-full-coverage yes`. No HPA exists.
- `redis-cluster-coordinator` (image `capitanodocker/redis-cluster-controller:0.1`, plain Redis + a Go program) waits
  for "all the seed nodes to register". Its source is downloaded at image build time; **its behaviour on new
  registrations is unknown** (risk R3).
- **FINDING: the cluster is not formed.** Every pod: `cluster_state:fail`, 0 slots, knows only itself, DBSIZE 0. The
  pod logs show "Connection refused" when registering with the coordinator. The starting state "3 pods, slots covered,
  healthy" therefore does not exist; entry 01 forms it (`redis-cli --cluster create`, 3 masters, no replicas).
  In the new namespace the cluster is formed from scratch; rollback deletes the namespace, so Q1 is moot.
- Host tools: `redis-cli` and `redis-benchmark` 7.x at `/usr/bin` (host, not the pod image; the pod image's contents
  were not inspected, `exec` is forbidden). `redis-server` is not on the host. `redis-py` is not in `.venv` (numpy,
  pytest are). The load generator therefore uses raw RESP over asyncio and installs nothing. `redis-benchmark` is not
  used: it has no open-loop rate control.
- Pod cgroups are readable read-only from the host (cgroup v1, `/sys/fs/cgroup/memory/system.slice/containerd.service/kubepods-*-pod<uid>.slice:cri-containerd:*`): memory usage, memory.stat, cpuacct. Idle 2026-10-16: working set (usage - inactive_file) 0.50-0.82 of the 16Mi limit; raw usage_in_bytes reaches 0.99 because of page cache (mapped binary), so the safety check uses the working set.
- Pod IPs (10.244.x.x) are reachable from the host. The load generator runs on the **same node** as the pods.
- Unrelated: `redis-vector-0` is Pending (PVC unbound), not touched.
- The old DInos load (`metrics-new/stress.py:30`) did `SET key value` on ONE key, i.e. one slot, one pod: capacity then
  does not scale with pods at all. The pilot uses uniform keys; this choice fixes theta and is a researcher decision (Q2).

## 2. Max feasible c
**c = 30 fits**: 30 pods x 150m = 4.5 cpu requests (27 more pods = 4.05) against 40 allocatable (2.65 used), memory
requests 240Mi, 110 pod slots. Scheduler-feasible; the hard cap stays 30 (`scale_to.py`). Whether a measurement is
trustworthy at 30 is a different matter: 30 pods x 200m = 6 cores at saturation plus the generator on the same host.
`generator_lag_ms > 50` flags generator-limited levels; fit.py drops them and says so.

## 3. Workload (fixed for all c)
Open-loop Poisson arrivals at a target rate (latency = completion minus intended send time, no coordinated omission).
Scheduling is decoupled from transport: the arrival loop only appends to in-memory queues; per-connection tasks connect,
write and drain. Every arrival is settled exactly once (ok, error, timeout or client drop); `offered = ok + err + timeout + drop`
is tested. Mix 95 % GET / 5 % SET, 2000 uniform keys `cal:<id>` (16 B key, 64 B value, ~0.3 MB), cluster-aware routing with
MOVED/ASK follow (max 2). Client timeout 10 s. Per 1 s window: offered, completed, errors, timeouts, drops, redirects,
a sparse log-histogram (5 % buckets), generator lag. First 10 s of each level `warm=1`.

## 4. Safety gates (fail closed)
- Isolation: the namespace is a hard-coded constant (`QRL_NAMESPACE` other than qrl-calib aborts at import); every kubectl call carries
  `--context kubernetes-admin@kubernetes --namespace qrl-calib` (static test over all scripts and specs); every redis-cli endpoint and
  every endpoint in the cluster view or in a MOVED/ASK redirect must be a pod IP listed by `kubectl get pods -n qrl-calib`, otherwise it
  is refused (counted as `refused_ops`).
- Cancellation: every mutating step (kubectl scale, add-node, create, fix, forget, rebalance) checks for a memory trip first, and the
  redis-cli commands run as tracked children that are killed on a trip, on the deadline or on SIGTERM. The memory watch refreshes its pod
  list before declaring a pod unreadable (a pod removed by scale-down does not trip it). Worker joins in the generator are exception-safe
  (dead worker: no hang, non-zero exit).
- No untracked mutation: `common.sh` runs only read-only kubectl/redis-cli (guard `is_readonly`; it refuses scale, apply, delete,
  `--cluster ...`, flushall, `cluster forget/reset/meet`); every mutation (replica change, create, add-node, fix, rebalance, forget, flushall,
  reset) runs through `scale_to.tracked` / `CHILDREN.popen`, with a trip check before it starts. A static AST test enforces this.
  Nothing issues `cluster meet` (redis-cli add-node does, only towards an allowlisted pod IP).
- Topology allowlist: `CLUSTER NODES` on EVERY allowlisted pod (not one anchor) before and after every create / add-node / fix / rebalance;
  a node IP outside the qrl-calib pod IPs, an unreadable pod or an empty read records a trip (`foreign-node`, `topology-unreadable`) and
  aborts. Before CREATE every participating pod must see only itself with no slots (`pre-create-view`).
- Watcher: never dies silently; any exception records a trip; an empty pod list is "no pods to watch" unless pods are expected (a level
  or scaling step is running), where it records `pods-vanished`.
- Entry gating (tools/launch_run.py checked 2026-10-16): `start --all` starts the next entries in index order **without looking at the
  status of earlier ones**, and nothing runs a rollback. Therefore every entry is wrapped in `gate.py`: the previous entry of the same
  request must have finished with status done and exit 0, otherwise `rollback.sh` runs and the entry refuses (every later entry fails the same
  way). The entry itself runs as a CHILD of the gate (no exec): on a non-zero exit or an exception `rollback.sh` runs immediately and the gate
  exits with the child's failure code; SIGTERM is forwarded to the child; an exception in the health query refuses and rolls back. `run_c`
  and `run_transition` fail the level and the entry on any non-zero load-generator exit (a memory trip goes through recovery instead); sweep and transition entries also require `health` OK; the final rollback entry uses `--allow-failed`. Tested with fake run dirs.
- Throughput: `served` = ok + late SUCCESS replies only (errors and MOVED/ASK are not served; a late success is a subset of the timeouts).
- Rollback proof: `rollback.sh` fails on any delete error and accepts only a `get namespace` that says NotFound.
- Memory: `MemWatch` checks every pod CONCURRENTLY each second (thread pool; cgroup working set = usage - inactive_file, plus
  `INFO persistence` over a socket whose reply is validated: missing fields or an unreadable pod are a trip). Refuses to start above
  85 % of 16Mi, trips above 95 %, trips when a BGSAVE/AOF rewrite runs while a pod is above 85 % (fork copy-on-write), and on AOF growth
  (n/a with AOF off). A trip writes `MEM_TRIPPED` into the run dir: **all scaling stops** (`require_no_trip` in every scale step), the
  generator and any rebalance child are terminated, and `recover_after_trip` runs `reconcile.py` (back to 3 pods, covered, only qrl-calib).
- Host CPU telemetry (`telemetry.jsonl`, 2 s): host busy cores, generator cores in total and **per process** (busiest worker), pod
  cores from cpuacct (looked up by the container directory name; the earlier memory-only helper lookup returned nothing), pod count.
- Every entry starts with a health gate; child processes are terminated and reaped in `finally` and on SIGTERM; the generator on
  SIGTERM stops arrivals, settles, flushes and exits 0.

## 5. Steps (stage A = `calibration_spec.json`, 16 entries, in order; each entry is `gate.py [flags] -- <command>`)
| # | entry | what | mutates |
|---|---|---|---|
| 00 | create | `kubectl apply -f k8s/qrl-calib.yaml` (namespace, Deployment 3 replicas, Service) | namespace qrl-calib |
| 01 | wait | `rollout status` | no |
| 02 | bootstrap | `scale_to.py --bootstrap 3` (`redis-cli --cluster create`, 3 masters, no replicas) | cluster |
| 03-14 | sweeps c = 1, 2, 4, 8, 16, 30, **30, 16, 8, 4, 2, 1** | `run_c.py --c N`: health gate -> scale -> memory gate -> prefill -> saturation probe (500 ops/s x c, x1.6 per 12 s until completed < 0.9 offered, p99 > 1 s, generator lag > 50 ms or 300k) -> sweep at 0.3 ... 1.5 x probed capacity (8 levels, 40 s) -> health. Two sweeps per c. | scale, keys |
| 15 | rollback | `kubectl delete namespace qrl-calib` | namespace |

Stage B (`transitions_spec.template.json`, not part of the default approval: stage A is already about 3 h, and stage B needs
C_est(hi) from stage A, so it is a second approval): creates the namespace again, then for (lo, hi) = (2,3), (4,5), (8,9), (16,17)
runs `run_transition.py` under saturating load 1.15 x C(hi), 3 repetitions: pre-window, scale up (blocks until healthy), settle, scale
down by plain decrement (blocks until covered, rebalanced and healthy), settle, SIGTERM. The lifecycle follows the scaling events.

Scaling (`scale_to.py`): up = replicas +1, wait Ready + PING, `--cluster add-node`, `--cluster rebalance --cluster-use-empty-masters`,
then `ensure_covered`. Down = replicas -1 only; then record the removed pod, its slots and keys (`pod_removed`), `cluster forget` of
the dead node on all survivors, `redis-cli --cluster fix` repeatedly until 16384 slots are covered and `cluster_state:ok`
(events `fix_start`, `covered`), rebalance, health. No abort on a "wrong victim". Lost cal:* keys are re-created by the next prefill.

Detached rollback: `tools/launch_run.py` sends SIGTERM to the experiment's whole process group on stop or wall-clock expiry and SIGKILL
after 30 s, which would kill a rollback that may take up to 300 s. So the gate starts `rollback_runner.py` in a NEW SESSION (own timeout 420 s)
whenever it rolls back; on SIGTERM it forwards the signal, starts the detached rollback and exits within seconds without waiting. The outcome
(status, start, end, exit code, namespace-absent proof) goes to `$QRL_RUN_DIR/rollback_status.json` and `code/calibration/.last_rollback.json`;
`rollback_status.py` reports it (exit 0 only for done + proof). `rollback.sh` takes a lock file first, so the final rollback entry (also run
detached by the gate) waits for any in-flight rollback and then re-checks. Tested with SIGTERM, then SIGKILL of the group 1 s later.
Single cleanup path: the signal handler only sets a flag, starts the detached rollback once and forwards SIGTERM to the experiment's group; the
main flow alone calls `reap_group` (idempotent), which confirms the group is empty (`killpg(pgid, 0)` -> ProcessLookupError) before the gate
exits, also when the leader has already exited. If members survive SIGKILL the gate exits non-zero, writes `cleanup_failed` into the run dir
(the next entry's gate refuses) and starts the detached rollback.

## 6. Definitions (aligned with `sim_v2/params.py`: C = mu r^theta (1 - delta 1[rebalancing]), pending pod serves after d steps)
- **Throughput** = server-side `served_ops` = ok + late replies (answers after the client timeout still cost the server work).
- **C(c)**: mean served of the two highest saturated levels of a sweep, accepted only with a demonstrated bracket: >= 1 valid level
  with served/offered >= 0.97, client drops <= 0.1 % and timeouts <= 1 %; and >= 2 valid levels with served/offered < 0.95 whose
  pods are busy (pod CPU >= 80 % of the 200m limits; a plateau from client drops with idle pods does not count) and agree within 10 %.
  Needs >= 2 sweeps from different runs within 15 %; median used. mu, theta by OLS (theta needs >= 3 pod counts).
- **Invalid levels** (excluded and listed): generator lag > 50 ms, error share > 1 %, the busiest single generator worker > 85 % of a
  core, host busy > 60 % of cpus, or no CPU telemetry.
- **Latency**: merged histograms per level; timeouts right-censored; a quantile in the censored mass is +inf and unused.
  Fit p95 = a + b/(1 - rho); the sim has a = 0, b = 1e3 ln20 c/C.
- **d** = scale_cmd -> first slot owned by the new pod, in 15 s steps (also: pod Ready time; scale_cmd -> rebalance end).
- **delta, k** (capacity-relative, saturating load, only windows with valid telemetry; an interval with < 80 % valid windows, an
  unsaturated interval, or non-busy reference pods is censored and excluded, never read as zero loss):
  up: delta = 1 - mean served in [first_slot, rebalance_end] / C_hi(measured after); k = ceil(duration/15 s).
  down (plain decrement): delta = 1 - mean served in [scale_cmd, t_restored] / C_lo(measured after), t_restored = first time served
  >= 90 % of C_lo for 3 windows; k = ceil((t_restored - scale_cmd)/15 s). **Recovery time** (scale_cmd -> covered, and -> restored),
  the pods removed, and slots and keys lost are reported. This is the real behaviour of the researcher's setup, and what Sim v2 should
  model for scale-down (the sim currently drops a pod instantly and loses delta for k steps; the measured down window is the data for
  that lump).
- `loss_pre` (capacity loss between scale_cmd and first_slot) decides `rebalance_trigger`. **timeout**: client policy, only the tail is reported.

## 7. Duration and limits (estimates, not measured)
Stage A about 3 h: 12 sweeps of 8-23 min, bootstrap, namespace create/delete. Stage B about 1.5 h. `limits.wall_clock_hours` is per
entry: 1.0. <= 30 pods, 8 generator processes. Stop at any time from the dashboard, then rollback.

## 8. Rollback
`rollback.sh` = `kubectl delete namespace qrl-calib --wait`, fails loudly if the namespace survives; idempotent. `reconcile.py` (3 pods,
covered, no open slots or stale members; aborts on non-cal:* keys; tested on a fake cluster) is only the in-run recovery after a memory trip.

## 9. Risks and open questions
Risks: R1 plain scale-down can remove a master and uncover slots; the cluster is down until `fix` finishes (that is the measurement, and
a client's view of it is in the latency/failure data). R2 OOM at 16Mi: working set idles at 50-82 % in the old pods; the 85 % start gate may
refuse to run (new pods have persistence off, expected lower). R3 the image is pinned by digest (`capitanodocker/redis-autoscale-cluster@sha256:1d51a336a275bdf8e9acbb6e53b7bea8e5da3202d13c3299876abdd87e9829d0` = imageID of the running pods in `default`, read-only `kubectl get pod`, 2026-10-16; tag 0.1; IfNotPresent, so the node's cached image is used and no pull should happen; if the node lacks a digest reference the pods show ImagePullBackOff and entry 01 fails). R3b `/redis.conf` is assumed to exist in the image (Dockerfile copies it;
no exec to verify), otherwise pods crash-loop and entry 01 fails. R4 generator co-located with the pods; host telemetry decides validity;
c=30 may stay unmeasured. R5 single node: d and recovery times are lower bounds. R6 `k8s/qrl-calib.yaml` not validated by kubectl (apply is forbidden for
me, even as dry-run). R7 untested on a real cluster: generator ceiling, `cluster fix` behaviour, timings.
Questions: Q2 uniform keys, or the DInos single-hot-key workload (capacity then does not scale)? Q4 SLO reading (2 x p95 sojourn at rho = 0.5)?
(Q1 and the persistence question Q3 are settled: stateless namespace, persistence off, R-20261016-018.)

## 10. Fit
`python fit.py RUN_DIR...` writes `fit_result.json`, `params_diff.json`, `params_diff.txt`; never edits `params.py`.
Tests: `../../.venv/bin/python -m pytest -q tests` (synthetic data, fake RESP servers, fake cluster, mocked kubectl).
