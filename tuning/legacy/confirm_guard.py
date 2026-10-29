"""Execution-level guard for the one-time final confirmation split test10 (11000-11019).

Base review of ef0e770 (.lab/tuning/cycles/c18_base_review.md, finding 1): the test10 refusal lived only in the entry
generator (make_c18.py); run_legacy.py and orig_baselines.py accepted test10 directly. Every process that would
evaluate on test10 now calls require() before any work and is refused unless ALL of these hold:

  - a manifest is named by env QRL_CONFIRM_MANIFEST or the CLI flag --confirm-manifest (the sandbox launcher strips
    QRL_* variables, so frozen entries carry the flag; if both are given they must name the same file);
  - the manifest was frozen by `make_c18.py --confirm`: its sha256 is a key of the ledger
    <qrl-lab>/.lab/tuning/confirm_ledger.json with status 'frozen' or 'in_use';
  - it lists the locked arms, their architectures, the checkpoint grid and the locked checkpoint rule (LOCKED_RULE),
    >= MIN_SEEDS distinct training seeds, disjoint from every training seed of every other *_entries.json in the lab
    directory (all tuning cycles, c18* included), and the sha256 of its entries file, which must still match;
  - this exact command line (script + every argument) is one entry of that entries file, and an agent entry's
    --seed is one of the manifest's training seeds;
  - single use: no other manifest of the ledger was ever started ('in_use'/'closed'), and this entry was not started
    before (the start is recorded under the manifest's 'started' map under a file lock; a crashed entry is not
    re-run with the same manifest).
"""
import contextlib
import datetime as dt
import fcntl
import glob
import hashlib
import json
import os

CONFIRM_SPLIT = "test10"
ENV = "QRL_CONFIRM_MANIFEST"
KIND = "c18_confirm_manifest"
MIN_SEEDS = 5
LOCKED_RULE = ("across-seed (locked): per target and arm, the checkpoint with the largest mean VAL return over the "
               "training seeds (ties: earlier checkpoint), sum_c10.py; weights exported per seed as <run>.locked.pt")
QRL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class ConfirmError(SystemExit):
    pass


def refuse(msg):
    raise ConfirmError(f"refusing test10: {msg}")


def lab_dir():
    return os.environ.get("QRL_CONFIRM_LAB_DIR") or os.path.join(QRL_ROOT, ".lab", "tuning")


def ledger_path():
    return os.environ.get("QRL_CONFIRM_LEDGER") or os.path.join(lab_dir(), "confirm_ledger.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 16), b""):
            h.update(b)
    return h.hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def entry_seed(cmd):
    """--seed of an agent entry command, None for other entries (baselines)."""
    if "--agent" not in cmd or "--seed" not in cmd:
        return None
    return int(cmd[cmd.index("--seed") + 1])


def tuning_seeds(lab=None, exclude=()):
    """Training seeds of every agent entry in <lab>/*_entries.json except the files in `exclude`."""
    ex = {os.path.realpath(x) for x in exclude}
    out = set()
    for f in sorted(glob.glob(os.path.join(lab or lab_dir(), "*_entries.json"))):
        if os.path.realpath(f) in ex:
            continue
        try:
            ents = json.load(open(f))
        except ValueError:
            continue
        for e in ents if isinstance(ents, list) else ():
            s = entry_seed(e.get("cmd", [])) if isinstance(e, dict) else None
            if s is not None:
                out.add(s)
    return out


@contextlib.contextmanager
def ledger_lock(path=None):
    """Exclusive lock around a read-modify-write of the ledger; yields (ledger dict, save function)."""
    path = path or ledger_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            led = json.load(open(path)) if os.path.exists(path) else {"manifests": {}}

            def save():
                tmp = f"{path}.tmp{os.getpid()}"
                with open(tmp, "w") as f:
                    json.dump(led, f, indent=1)
                os.replace(tmp, path)
            yield led, save
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def validate_manifest(m, lab=None):
    """Content checks shared by make_c18.py --confirm (before freezing) and require() (before running)."""
    if not isinstance(m, dict) or m.get("kind") != KIND:
        refuse(f"manifest kind is not {KIND}")
    if m.get("report") != CONFIRM_SPLIT:
        refuse("manifest report split is not test10")
    lk = m.get("locked") or {}
    if not lk.get("arms") or not isinstance(lk.get("architectures"), dict) or not lk.get("checkpoints"):
        refuse("manifest does not list the locked arms / architectures / checkpoints")
    if lk.get("checkpoint_rule") != LOCKED_RULE:
        refuse("manifest checkpoint rule is not the across-seed locked rule")
    seeds = m.get("training_seeds") or []
    if len(seeds) < MIN_SEEDS or len(set(seeds)) != len(seeds) or not all(isinstance(s, int) for s in seeds):
        refuse(f"manifest needs >= {MIN_SEEDS} distinct integer training seeds, got {seeds}")
    ent = m.get("entries")
    if not ent or not os.path.exists(ent):
        refuse(f"manifest entries file {ent!r} not found")
    if sha256_file(ent) != m.get("entries_sha256"):
        refuse("entries file changed after the manifest was frozen (sha256 mismatch)")
    clash = sorted(set(seeds) & tuning_seeds(lab, exclude=(ent,)))
    if clash:
        refuse(f"training seeds {clash} were used by tuning entries")
    ents = json.load(open(ent))
    bad = sorted({entry_seed(e["cmd"]) for e in ents} - {None} - set(seeds))
    if bad:
        refuse(f"entries use seeds {bad} that the manifest does not list")
    return ents


def manifest_path(flag_path=None):
    env = os.environ.get(ENV)
    if env and flag_path and os.path.realpath(env) != os.path.realpath(flag_path):
        refuse(f"{ENV} and --confirm-manifest name different files")
    return env or flag_path


def require(splits, script, argv, flag_path=None, seed=None):
    """Call before any work. No-op unless test10 is among `splits`; else checks the frozen manifest (see module doc)
    and records this entry as started in the ledger. Returns the ledger note (or None)."""
    if CONFIRM_SPLIT not in tuple(splits):
        if flag_path:
            refuse("--confirm-manifest given but test10 is not evaluated")
        return None
    path = manifest_path(flag_path)
    if not path:
        refuse(f"test10 (11000-11019) is the one-time final confirmation split: set {ENV} (or --confirm-manifest) "
               "to a manifest frozen by make_c18.py --confirm")
    if not os.path.exists(path):
        refuse(f"manifest {path} not found")
    sha = sha256_file(path)
    m = json.load(open(path))
    ents = validate_manifest(m)
    argv = list(argv)
    match = [e for e in ents if os.path.basename(e["cmd"][1]) == os.path.basename(script) and e["cmd"][2:] == argv]
    if len(match) != 1:
        refuse("this command line is not exactly one entry of the manifest's frozen entries file")
    e = match[0]
    if entry_seed(e["cmd"]) is not None and (seed is None or int(seed) not in m["training_seeds"]):
        refuse(f"seed {seed} is not a manifest training seed")
    with ledger_lock() as (led, save):
        man = led.setdefault("manifests", {})
        rec = man.get(sha)
        if rec is None:
            refuse(f"manifest sha256 {sha[:12]} is not recorded in {ledger_path()} (not frozen by make_c18 --confirm)")
        other = [k for k, r in man.items() if k != sha and r.get("status") in ("in_use", "closed")]
        if other:
            refuse(f"test10 was already used by manifest {other[0][:12]} (single use)")
        if rec.get("status") not in ("frozen", "in_use"):
            refuse(f"manifest status is {rec.get('status')!r}")
        if e["name"] in rec.setdefault("started", {}):
            refuse(f"entry {e['name']} already started under this manifest at {rec['started'][e['name']]} (no reuse)")
        rec["started"][e["name"]] = now()
        rec["status"] = "in_use"
        save()
    return dict(manifest=os.path.abspath(path), sha256=sha, entry=e["name"])


def freeze(manifest_file, lab=None):
    """make_c18.py --confirm: validate and record the manifest's sha256 in the ledger (status 'frozen'). Refused if
    test10 was already used; earlier frozen-but-unused manifests become 'superseded'."""
    m = json.load(open(manifest_file))
    validate_manifest(m, lab)
    sha = sha256_file(manifest_file)
    with ledger_lock() as (led, save):
        man = led.setdefault("manifests", {})
        used = [k for k, r in man.items() if r.get("status") in ("in_use", "closed")]
        if used:
            refuse(f"test10 was already used by manifest {used[0][:12]} (single use); no new confirmation manifest")
        for k, r in man.items():
            if r.get("status") == "frozen" and k != sha:
                r["status"], r["superseded_utc"] = "superseded", now()
        man[sha] = dict(path=os.path.abspath(manifest_file), tag=m.get("tag"), entries=m["entries"],
                        entries_sha256=m["entries_sha256"], training_seeds=m["training_seeds"], frozen_utc=now(),
                        status="frozen", started={})
        save()
    return sha
