"""Freeze enforcement for experiments/dinos_simv2 (R-20261017-001, Codex review of c5beef7..99b1ab4).

A spec carries `frozen_sha256` (path relative to the spec's directory -> sha256) and `frozen_git_commit`.
`verify_freeze(spec_path)` runs BEFORE any training and raises FreezeError if
  * a listed file is missing, untracked, modified in the working tree/index, or its sha256 differs;
  * the recorded commit is not an ancestor of (or equal to) HEAD, or a listed file differs between it and HEAD;
`check_loaded_modules(manifest_paths)` (called once the agents are built) raises if the process has imported a
source file under the code worktree that is not in the manifest (so the manifest cannot silently miss a dependency).
`FROZEN` is the list make_spec.py hashes. Dependency versions are recorded by `versions()`.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
PYTORCH = "../../sim/timeseries/Pytorch"
FROZEN = [
    "adapter.py", "agents.py", "freeze.py", "train_eval.py", "gate_rev1.json", "PREREG_rev1.md",
    "explore_w0.json", "PREREG_explore_w0.md", "analyze_explore_w0.py", "select_hpa_dev.py", "hpa_selection_dev.json",
    "../../sim_v2/env.py", "../../sim_v2/params.py", "../../sim_v2/reward_dinos.json", "../../sim_v2/oracle.py", "../../sim_v2/policies.py",
    "../../sim_v2/workloads.py", "../../sim_v2/sensitivity.py", "../../sim_v2/sensitivity_v21.json",
    PYTORCH + "/lstm_transfer/agentHuber.py", PYTORCH + "/lstm_transfer/networks.py",
    PYTORCH + "/lstm_transfer/modelCPU_5.pth",
    PYTORCH + "/dense/ddqnagent.py", PYTORCH + "/dense/network.py", PYTORCH + "/dense/environment.py",
]


class FreezeError(RuntimeError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(cwd, *args):
    r = subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def git_head(cwd=HERE):
    rc, out = _git(cwd, "rev-parse", "HEAD")
    if rc:
        raise FreezeError("not a git checkout (cannot read HEAD)")
    return out


def dirty_paths(abs_paths, cwd=HERE):
    """Listed paths that are untracked or differ from HEAD in the index/working tree."""
    bad = []
    for p in abs_paths:
        rc, _ = _git(cwd, "ls-files", "--error-unmatch", p)
        if rc:
            bad.append(p + " (untracked)")
    rc, out = _git(cwd, "status", "--porcelain", "--", *abs_paths)
    bad += [line[3:] + " (modified)" for line in out.splitlines()]
    return bad


def make_manifest(base=HERE, files=None):
    return {os.path.normpath(r): sha256_file(os.path.join(base, r)) for r in (files or FROZEN)}


def verify_freeze(spec_path, required=None):
    """Returns a report dict; raises FreezeError on any mismatch. `spec_path` is the approved spec JSON."""
    spec_path = os.path.abspath(spec_path)
    base = os.path.dirname(spec_path)
    with open(spec_path) as f:
        spec = json.load(f)
    man = spec.get("frozen_sha256")
    if not man:
        raise FreezeError(f"{spec_path} has no frozen_sha256")
    missing_from_manifest = sorted(set(os.path.normpath(r) for r in (FROZEN if required is None else required)) - set(man))
    if missing_from_manifest:
        raise FreezeError(f"manifest lacks required files: {missing_from_manifest}")
    paths = {}
    for rel, want in man.items():
        p = os.path.normpath(os.path.join(base, rel))
        if not os.path.isfile(p):
            raise FreezeError(f"frozen file missing: {rel}")
        got = sha256_file(p)
        if got != want:
            raise FreezeError(f"sha256 mismatch for {rel}: expected {want[:12]}..., found {got[:12]}...")
        paths[rel] = p
    dirty = dirty_paths(list(paths.values()), base)
    if dirty:
        raise FreezeError(f"frozen paths not clean in git: {dirty}")
    head = git_head(base)
    commit = spec.get("frozen_git_commit")
    if not commit:
        raise FreezeError(f"{spec_path} has no frozen_git_commit")
    rc, _ = _git(base, "merge-base", "--is-ancestor", commit, "HEAD")
    if rc:
        raise FreezeError(f"frozen commit {commit[:10]} is not an ancestor of HEAD {head[:10]}")
    rc, out = _git(base, "diff", "--name-only", commit, "HEAD", "--", *paths.values())
    if rc or out:
        raise FreezeError(f"frozen files changed between {commit[:10]} and HEAD: {out}")
    return dict(spec=spec_path, git_head=head, frozen_git_commit=commit, frozen_sha256=man, versions=versions())


def check_loaded_modules(manifest, base=HERE):
    """Every source file imported from the code worktree (except tests and the stdlib/venv) must be in the manifest."""
    allowed = {os.path.normpath(os.path.join(base, r)) for r in manifest}
    seen = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and os.path.isabs(f):
            seen.add(os.path.normpath(f))
    try:
        import agents
        seen |= set(agents.LOADED_FILES)
    except ImportError:
        pass
    extra = sorted(f for f in seen if f.startswith(CODE_ROOT + os.sep) and f.endswith(".py")
                   and os.sep + "tests" + os.sep not in f and f not in allowed
                   and os.sep + ".venv" + os.sep not in f)
    if extra:
        raise FreezeError(f"imported source files missing from the frozen manifest: {extra}")


def versions():
    import numpy
    out = dict(python=platform.python_version(), numpy=numpy.__version__, platform=platform.platform())
    try:
        import torch
        out["torch"] = torch.__version__
    except ImportError:
        out["torch"] = None
    try:
        import pandas
        out["pandas"] = pandas.__version__
    except ImportError:
        out["pandas"] = None
    return out
