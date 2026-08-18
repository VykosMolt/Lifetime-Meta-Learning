"""O1 isolation guard (contract §13 and §22).

The Foundation Learner and the O1 calibration programme share exactly three
things on the accelerator: the checkpoint, the runtime, and the accelerator
itself.  Nothing else.  In particular FL never reads O1 calibration outcomes,
transport results, difficulty selection, or analysis, and never writes anywhere
under an O1 root.

Every campaign file access goes through :class:`IsolationGuard`:

* the path is ``os.path.realpath``-resolved FIRST (a symlink into an O1 root is
  the obvious attack, and only the resolved path can be judged);
* any access under a forbidden root is refused with :class:`O1IsolationError`;
* the two shared READ-ONLY inputs (``/artifacts/ouro_rltt_local`` and
  ``/artifacts/TOKENIZER_BINDING.json``) are readable and NOT writable — a write
  attempt there is refused too, because mutation of a shared input would leak
  FL state into the O1 programme.

The forbidden-root list has two sources, both applied:

1. the FROZEN defaults of contract §22, which are also the
   ``isolation.o1_forbidden_roots`` list of
   ``FOUNDATION_LEARNER_V0_MANIFEST.template.json`` (the two are asserted equal
   by a unit test, so the manifest cannot drift away from the code); and
2. roots DISCOVERED at supervisor start from the O1 transfer manifests when
   those manifests are present on the instance.

Discovery only ever ADDS roots.  A missing or unreadable O1 manifest can
therefore never widen FL's reach; it merely leaves the frozen list in force.
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterable, Sequence

__all__ = [
    "O1IsolationError",
    "FROZEN_FORBIDDEN_ROOTS",
    "SHARED_READONLY_INPUTS",
    "MODE_READ",
    "MODE_WRITE",
    "IsolationGuard",
    "default_guard",
    "set_default_guard",
    "guard_path",
    "discover_roots_from_o1_manifest",
]


class O1IsolationError(RuntimeError):
    """A path under an O1 root (or a write to a shared input) was refused."""


#: Contract §22 frozen defaults.  Mirrored in the campaign manifest template.
FROZEN_FORBIDDEN_ROOTS: tuple[str, ...] = (
    "/outputs",
    "/opt/o1_b200",
    "/artifacts/AXIS_PACKAGE_V2",
    "/artifacts/COHORTS",
    "/artifacts/calibration_seed_matrix.json",
    "/artifacts/policies",
    "/workspace/o1_calibration",
)

#: Contract §22: allowed shared inputs, READ-ONLY.
SHARED_READONLY_INPUTS: tuple[str, ...] = (
    "/artifacts/ouro_rltt_local",
    "/artifacts/TOKENIZER_BINDING.json",
)

MODE_READ = "read"
MODE_WRITE = "write"
_MODES = (MODE_READ, MODE_WRITE)


def _norm(path: str) -> str:
    """Realpath-resolved absolute path (works for non-existent paths too)."""
    return os.path.realpath(os.path.abspath(os.path.expanduser(str(path))))


def _is_under(path: str, root: str) -> bool:
    """True when ``path`` is ``root`` itself or lies beneath it."""
    if path == root:
        return True
    return path.startswith(root.rstrip(os.sep) + os.sep)


def discover_roots_from_o1_manifest(manifest_path: str) -> list[str]:
    """Extract artifact roots from an O1 transfer manifest.

    Accepts the O1 ``o1b200.transfer_manifest.v1`` shape
    (``{"artifacts": {name: {"path": ...}}}``) and the plain
    ``{"roots": [...]}`` / list-of-strings shapes, because the FL package must
    not depend on the internals of a sealed package it may not modify.  Only
    PATH STRINGS are read; no scientific content is parsed.
    """
    with open(manifest_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    found: list[str] = []

    def _add(value: Any) -> None:
        if isinstance(value, str) and value.strip():
            found.append(value.strip())

    if isinstance(payload, list):
        for entry in payload:
            _add(entry)
    elif isinstance(payload, dict):
        artifacts = payload.get("artifacts")
        if isinstance(artifacts, dict):
            for spec in artifacts.values():
                if isinstance(spec, dict):
                    _add(spec.get("path"))
                else:
                    _add(spec)
        for key in ("roots", "o1_roots", "result_roots", "forbidden_roots"):
            value = payload.get(key)
            if isinstance(value, list):
                for entry in value:
                    _add(entry)
            else:
                _add(value)
    # A directory entry contributes itself; a file entry contributes itself as
    # well (containment is checked with an exact-or-beneath rule).
    return sorted({_norm(p) for p in found})


class IsolationGuard:
    """Realpath guard for every campaign file access.

    ``guard(path, mode)`` returns the resolved path when the access is allowed
    and raises :class:`O1IsolationError` otherwise.  The guard is intentionally
    dumb about *why* a path is FL-owned: it refuses O1 roots and refuses writes
    to shared read-only inputs, and permits everything else, so that a campaign
    module cannot accidentally acquire read access to O1 material by naming it
    differently.
    """

    def __init__(
        self,
        forbidden_roots: Sequence[str] = FROZEN_FORBIDDEN_ROOTS,
        shared_readonly: Sequence[str] = SHARED_READONLY_INPUTS,
        *,
        label: str = "FL_CAMPAIGN",
    ):
        self.label = str(label)
        self._frozen_roots = tuple(str(r) for r in forbidden_roots)
        self._roots: list[str] = sorted({_norm(r) for r in forbidden_roots})
        self._shared: list[str] = sorted({_norm(r) for r in shared_readonly})
        self._discovered: list[str] = []
        self.refusals: list[dict[str, str]] = []

    # ---------------- root management ----------------

    @property
    def forbidden_roots(self) -> tuple[str, ...]:
        return tuple(self._roots)

    @property
    def discovered_roots(self) -> tuple[str, ...]:
        return tuple(self._discovered)

    @property
    def shared_readonly(self) -> tuple[str, ...]:
        return tuple(self._shared)

    def add_forbidden_roots(self, roots: Iterable[str]) -> list[str]:
        """Add roots (discovery only ever widens the refusal set)."""
        added = []
        for root in roots:
            resolved = _norm(root)
            if resolved not in self._roots:
                self._roots.append(resolved)
                added.append(resolved)
        self._roots.sort()
        return added

    def discover_from_o1_manifests(self, manifest_paths: Iterable[str]) -> dict:
        """Bind O1 result roots from the O1 transfer manifests, when present.

        A manifest that is absent or unreadable is RECORDED and skipped; the
        frozen list stays in force.  Nothing here reads scientific content.
        """
        report = {"manifests": [], "added_roots": []}
        for path in manifest_paths:
            entry: dict[str, Any] = {"path": str(path)}
            try:
                roots = discover_roots_from_o1_manifest(path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                entry["status"] = "UNREADABLE"
                entry["error"] = repr(exc)
                report["manifests"].append(entry)
                continue
            added = self.add_forbidden_roots(roots)
            self._discovered.extend(r for r in added if r not in self._discovered)
            entry["status"] = "READ"
            entry["roots_in_manifest"] = len(roots)
            entry["roots_added"] = added
            report["manifests"].append(entry)
            report["added_roots"].extend(added)
        self._discovered.sort()
        report["added_roots"] = sorted(set(report["added_roots"]))
        return report

    # ---------------- the guard ----------------

    def _refuse(self, path: str, resolved: str, mode: str, reason: str) -> None:
        record = {"path": str(path), "resolved": resolved, "mode": mode,
                  "reason": reason}
        self.refusals.append(record)
        raise O1IsolationError(
            f"[{self.label}] REFUSED {mode} of {path!r} (resolved {resolved!r}): "
            f"{reason}")

    def guard(self, path: str, mode: str = MODE_READ) -> str:
        """Resolve ``path`` and refuse any O1-isolation violation."""
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        resolved = _norm(path)
        for root in self._roots:
            if _is_under(resolved, root):
                self._refuse(path, resolved, mode,
                             f"path lies under the O1-forbidden root {root!r}; "
                             "FL never reads or writes O1 material")
        if mode == MODE_WRITE:
            for shared in self._shared:
                if _is_under(resolved, shared):
                    self._refuse(path, resolved, mode,
                                 f"{shared!r} is a shared READ-ONLY input; "
                                 "mutation of shared inputs is forbidden")
        return resolved

    # ---------------- guarded file operations ----------------

    def open(self, path: str, mode: str = "r", **kwargs):
        access = MODE_WRITE if any(c in mode for c in "wxa+") else MODE_READ
        resolved = self.guard(path, access)
        return open(resolved, mode, **kwargs)

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        with self.open(path, "r", encoding=encoding) as fh:
            return fh.read()

    def read_bytes(self, path: str) -> bytes:
        with self.open(path, "rb") as fh:
            return fh.read()

    def read_json(self, path: str) -> Any:
        return json.loads(self.read_text(path))

    def makedirs(self, path: str, exist_ok: bool = True) -> str:
        resolved = self.guard(path, MODE_WRITE)
        os.makedirs(resolved, exist_ok=exist_ok)
        return resolved

    def write_text(self, path: str, text: str, *, encoding: str = "utf-8") -> str:
        """Atomic guarded write (tmp + fsync + rename + directory fsync)."""
        resolved = self.guard(path, MODE_WRITE)
        parent = os.path.dirname(resolved) or "."
        os.makedirs(parent, exist_ok=True)
        tmp = resolved + ".tmp"
        with open(tmp, "w", encoding=encoding) as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, resolved)
        fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return resolved

    def write_json(self, path: str, obj: Any) -> str:
        return self.write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")

    def append_line(self, path: str, line: str) -> str:
        """Append one fsync'd line (journals and the sealed opening ledger)."""
        resolved = self.guard(path, MODE_WRITE)
        os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
        with open(resolved, "a", encoding="utf-8") as fh:
            fh.write(line.rstrip("\n") + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return resolved

    def listdir(self, path: str) -> list[str]:
        return sorted(os.listdir(self.guard(path, MODE_READ)))

    def walk(self, path: str):
        """Guarded ``os.walk``.

        Every visited directory, every subdirectory ENTRY and every file entry
        is guarded, because ``os.walk`` does not follow symlinks: a symlinked
        subtree pointing into an O1 root would otherwise be listed (and then
        hashed or archived by a caller) without ever being resolved.
        """
        root = self.guard(path, MODE_READ)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            filenames.sort()
            self.guard(dirpath, MODE_READ)
            for name in list(dirnames) + list(filenames):
                self.guard(os.path.join(dirpath, name), MODE_READ)
            yield dirpath, dirnames, filenames

    def to_dict(self) -> dict:
        return {
            "schema": "flb200.isolation_guard.v1",
            "label": self.label,
            "frozen_forbidden_roots": list(self._frozen_roots),
            "forbidden_roots": list(self._roots),
            "discovered_roots": list(self._discovered),
            "shared_readonly_inputs": list(self._shared),
            "refusals": list(self.refusals),
        }


_DEFAULT_GUARD = IsolationGuard()


def default_guard() -> IsolationGuard:
    """The process-wide guard used when a caller supplies none."""
    return _DEFAULT_GUARD


def set_default_guard(guard: IsolationGuard) -> IsolationGuard:
    """Install a guard (the supervisor does this once, after discovery)."""
    global _DEFAULT_GUARD
    if not isinstance(guard, IsolationGuard):
        raise TypeError("default guard must be an IsolationGuard")
    _DEFAULT_GUARD = guard
    return _DEFAULT_GUARD


def guard_path(path: str, mode: str = MODE_READ,
               guard: IsolationGuard | None = None) -> str:
    """Module-level convenience over the default (or supplied) guard."""
    return (guard or _DEFAULT_GUARD).guard(path, mode)
