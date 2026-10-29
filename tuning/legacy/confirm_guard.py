"""Execution-level guard for the one-time final confirmation splits test10 (11000-11019) and test11 (12000-12019).

Cycle 22: test10 was used once (manifest 7f9face1, c21) and stays closed; test11 is a second held-out confirmation
split wired the same way. Every rule below applies PER confirmation split: a manifest records its split in 'report'
(the ledger record in 'split'; ledger records without it predate test11 and are test10), a manifest authorises only
runs of its own split, a process may evaluate at most one confirmation split, and once any manifest of a split was
started ('in_use'/'closed') no new manifest for that split may be frozen or used.

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

CONFIRM_SPLITS = ("test10", "test11")
CONFIRM_SPLIT = CONFIRM_SPLITS[0]          # backward compatibility (pre-c22 callers)
LEGACY_SPLIT = "test10"                    # ledger records without 'split' (e.g. 7f9face1) are test10
SPLIT_SEEDS_TXT = {"test10": "11000-11019", "test11": "12000-12019"}
_GRANTS = {}                               # split -> note of require() in THIS process, consumed by authorise()
ENV = "QRL_CONFIRM_MANIFEST"
KIND = "c18_confirm_manifest"
MIN_SEEDS = 5
LOCKED_RULE = ("across-seed (locked): per target and arm, the checkpoint with the largest mean VAL return over the "
               "training seeds (ties: earlier checkpoint), sum_c10.py; weights exported per seed as <run>.locked.pt")
QRL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class ConfirmError(SystemExit):
    pass


def refuse(msg, split=CONFIRM_SPLIT):
    raise ConfirmError(f"refusing {split}: {msg}")


def record_split(rec):
    """Confirmation split of a ledger record (records frozen before c22 carry no 'split' -> test10)."""
    return rec.get("split") or LEGACY_SPLIT


def used_by(man, split, exclude=None):
    """Ledger keys of manifests of `split` that were ever started."""
    return [k for k, r in man.items() if k != exclude and record_split(r) == split
            and r.get("status") in ("in_use", "closed")]


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


def validate_manifest(m, lab=None, split=None):
    """Content checks shared by make_c18.py --confirm (before freezing) and require() (before running). `split`: the
    confirmation split the caller evaluates (None = any of CONFIRM_SPLITS; the manifest's own split is used)."""
    sp = split or (m.get("report") if isinstance(m, dict) and m.get("report") in CONFIRM_SPLITS else CONFIRM_SPLIT)
    if not isinstance(m, dict) or m.get("kind") != KIND:
        refuse(f"manifest kind is not {KIND}", sp)
    if m.get("report") not in CONFIRM_SPLITS:
        refuse(f"manifest report split {m.get('report')!r} is not one of {CONFIRM_SPLITS}", sp)
    if split is not None and m.get("report") != split:
        refuse(f"manifest is for {m.get('report')}, not {split}", sp)
    lk = m.get("locked") or {}
    if not lk.get("arms") or not isinstance(lk.get("architectures"), dict) or not lk.get("checkpoints"):
        refuse("manifest does not list the locked arms / architectures / checkpoints", sp)
    if lk.get("checkpoint_rule") != LOCKED_RULE:
        refuse("manifest checkpoint rule is not the across-seed locked rule", sp)
    seeds = m.get("training_seeds") or []
    if len(seeds) < MIN_SEEDS or len(set(seeds)) != len(seeds) or not all(isinstance(s, int) for s in seeds):
        refuse(f"manifest needs >= {MIN_SEEDS} distinct integer training seeds, got {seeds}", sp)
    ent = m.get("entries")
    if not ent or not os.path.exists(ent):
        refuse(f"manifest entries file {ent!r} not found", sp)
    if sha256_file(ent) != m.get("entries_sha256"):
        refuse("entries file changed after the manifest was frozen (sha256 mismatch)", sp)
    clash = sorted(set(seeds) & tuning_seeds(lab, exclude=(ent,)))
    if clash:
        refuse(f"training seeds {clash} were used by tuning entries", sp)
    ents = json.load(open(ent))
    bad = sorted({entry_seed(e["cmd"]) for e in ents} - {None} - set(seeds))
    if bad:
        refuse(f"entries use seeds {bad} that the manifest does not list", sp)
    return ents


def manifest_path(flag_path=None, split=CONFIRM_SPLIT):
    env = os.environ.get(ENV)
    if env and flag_path and os.path.realpath(env) != os.path.realpath(flag_path):
        refuse(f"{ENV} and --confirm-manifest name different files", split)
    return env or flag_path


def require(splits, script, argv, flag_path=None, seed=None):
    """Call before any work. No-op unless a confirmation split (CONFIRM_SPLITS) is among `splits`; else checks the
    frozen manifest of that split (see module doc) and records this entry as started in the ledger. Returns the
    ledger note (or None)."""
    conf = [x for x in CONFIRM_SPLITS if x in tuple(splits)]
    if not conf:
        if flag_path:
            refuse("--confirm-manifest given but no confirmation split (test10/test11) is evaluated")
        return None
    if len(conf) > 1:
        refuse(f"one confirmation split per process, got {conf}", conf[0])
    sp = conf[0]
    path = manifest_path(flag_path, sp)
    if not path:
        refuse(f"{sp} ({SPLIT_SEEDS_TXT[sp]}) is a one-time final confirmation split: set {ENV} (or "
               f"--confirm-manifest) to a manifest frozen by make_c18.py --confirm for {sp}", sp)
    if not os.path.exists(path):
        refuse(f"manifest {path} not found", sp)
    sha = sha256_file(path)
    m = json.load(open(path))
    ents = validate_manifest(m, split=sp)
    argv = list(argv)
    match = [e for e in ents if os.path.basename(e["cmd"][1]) == os.path.basename(script) and e["cmd"][2:] == argv]
    if len(match) != 1:
        refuse("this command line is not exactly one entry of the manifest's frozen entries file", sp)
    e = match[0]
    if entry_seed(e["cmd"]) is not None and (seed is None or int(seed) not in m["training_seeds"]):
        refuse(f"seed {seed} is not a manifest training seed", sp)
    with ledger_lock() as (led, save):
        man = led.setdefault("manifests", {})
        rec = man.get(sha)
        if rec is None:
            refuse(f"manifest sha256 {sha[:12]} is not recorded in {ledger_path()} (not frozen by make_c18 --confirm)",
                   sp)
        if record_split(rec) != sp:
            refuse(f"ledger records manifest {sha[:12]} for {record_split(rec)}, not {sp}", sp)
        other = used_by(man, sp, exclude=sha)
        if other:
            refuse(f"{sp} was already used by manifest {other[0][:12]} (single use)", sp)
        if rec.get("status") not in ("frozen", "in_use"):
            refuse(f"manifest status is {rec.get('status')!r}", sp)
        if e["name"] in rec.setdefault("started", {}):
            refuse(f"entry {e['name']} already started under this manifest at {rec['started'][e['name']]} (no reuse)",
                   sp)
        rec["started"][e["name"]] = now()
        rec["status"] = "in_use"
        save()
    note = dict(manifest=os.path.abspath(path), sha256=sha, entry=e["name"], split=sp)
    _GRANTS[sp] = note
    return dict(note)


def authorise(splits, who):
    """Evaluator boundary (c5_protocol.run_c5, run_legacy.run, orig/coupled baselines): call where split names become
    eval seeds. No-op for non-confirmation splits. A confirmation split needs a grant issued by require() earlier in
    this process (i.e. a CLI entry with a frozen, unused manifest of that split, recorded as started in the ledger);
    each grant authorises ONE evaluator call. Direct Python calls without it are refused (Codex review of a191edb)."""
    conf = [x for x in CONFIRM_SPLITS if x in tuple(splits)]
    if not conf:
        return None
    if len(conf) > 1:
        refuse(f"{who}: one confirmation split per process, got {conf}", conf[0])
    sp = conf[0]
    note = _GRANTS.pop(sp, None)
    if note is None:
        refuse(f"{who}: evaluating {sp} needs a grant from confirm_guard.require() in this process (a frozen, unused "
               f"{sp} manifest via --confirm-manifest / {ENV}); direct calls are refused", sp)
    with ledger_lock() as (led, _save):
        rec = led.get("manifests", {}).get(note["sha256"]) or {}
    if record_split(rec) != sp or rec.get("status") != "in_use" or note["entry"] not in rec.get("started", {}):
        refuse(f"{who}: ledger does not record entry {note['entry']} as started under manifest "
               f"{note['sha256'][:12]} for {sp}", sp)
    return note


def freeze(manifest_file, lab=None):
    """make_c18.py --confirm: validate and record the manifest's sha256 and confirmation split (its 'report') in the
    ledger (status 'frozen'). Refused if that split was already used; earlier frozen-but-unused manifests of the
    same split become 'superseded' (other splits are untouched)."""
    m = json.load(open(manifest_file))
    validate_manifest(m, lab)
    sp = m["report"]
    sha = sha256_file(manifest_file)
    with ledger_lock() as (led, save):
        man = led.setdefault("manifests", {})
        used = used_by(man, sp)
        if used:
            refuse(f"{sp} was already used by manifest {used[0][:12]} (single use); no new confirmation manifest", sp)
        for k, r in man.items():
            if r.get("status") == "frozen" and k != sha and record_split(r) == sp:
                r["status"], r["superseded_utc"] = "superseded", now()
        man[sha] = dict(path=os.path.abspath(manifest_file), tag=m.get("tag"), split=sp, entries=m["entries"],
                        entries_sha256=m["entries_sha256"], training_seeds=m["training_seeds"], frozen_utc=now(),
                        status="frozen", started={})
        save()
    return sha
