"""Object-storage adapter boundary + resumable, noninteractive transfer.

No storage is selected or paid for in this task.  The boundary is strict:

  * LocalArtifactStore — filesystem/local-network staging (also the dress-
    rehearsal store);
  * MockArtifactStore — scriptable failures for tests.

A future S3-compatible store implements the same three methods.  Transfers
are chunked with bounded retry, write to a temporary destination, verify the
SHA-256 (or tree hash), atomically rename, clean up partials, and time out.
The future Pod refuses model loading until every artifact hash passes
(deploy/verify_artifacts.py consumes the transfer manifest).
"""
from __future__ import annotations

import abc
import hashlib
import os
import shutil
import time

CHUNK = 8 << 20
MAX_RETRIES = 5
TIMEOUT_SECONDS = 3600


class TransferFailure(RuntimeError):
    pass


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ArtifactStore(abc.ABC):
    @abc.abstractmethod
    def open_range(self, source: str, offset: int):
        """Return a readable stream positioned at offset."""

    @abc.abstractmethod
    def size(self, source: str) -> int:
        """Total byte size of the source object."""

    def fetch(self, source: str, dest_dir: str, *,
              expected_sha256: str | None = None,
              clock=time.monotonic, sleep=time.sleep) -> dict:
        """Resumable chunked fetch: tmp file + retry + hash + atomic rename."""
        os.makedirs(dest_dir, exist_ok=True)
        name = os.path.basename(source.rstrip("/"))
        final = os.path.join(dest_dir, name)
        tmp = final + ".partial"
        total = self.size(source)
        deadline = clock() + TIMEOUT_SECONDS
        attempt = 0
        offset = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        while True:
            if clock() > deadline:
                self._cleanup(tmp)
                raise TransferFailure(f"transfer timeout for {name}")
            try:
                with self.open_range(source, offset) as src, \
                        open(tmp, "ab") as out:
                    while offset < total:
                        chunk = src.read(min(CHUNK, total - offset))
                        if not chunk:
                            raise TransferFailure("short read")
                        out.write(chunk)
                        offset += len(chunk)
                    out.flush()
                    os.fsync(out.fileno())
                break
            except (OSError, TransferFailure):
                attempt += 1
                if attempt > MAX_RETRIES:
                    self._cleanup(tmp)
                    raise TransferFailure(
                        f"transfer of {name} failed after {MAX_RETRIES} "
                        f"retries at offset {offset}")
                sleep(min(30.0, 2.0 ** attempt))
                offset = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        got = sha256_file(tmp)
        if expected_sha256 is not None and got != expected_sha256:
            self._cleanup(tmp)
            raise TransferFailure(
                f"{name}: hash {got[:16]} != expected "
                f"{expected_sha256[:16]}; partial removed")
        os.replace(tmp, final)
        return {"path": final, "bytes": total, "sha256": got,
                "retries": attempt}

    @staticmethod
    def _cleanup(tmp: str) -> None:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass


class LocalArtifactStore(ArtifactStore):
    def open_range(self, source: str, offset: int):
        fh = open(source, "rb")
        fh.seek(offset)
        return fh

    def size(self, source: str) -> int:
        return os.path.getsize(source)


class MockArtifactStore(ArtifactStore):
    """Scriptable: fail N times, corrupt bytes, or truncate mid-stream."""

    def __init__(self, files: dict, *, fail_reads: int = 0,
                 corrupt: bool = False, truncate_at: int | None = None):
        self.files = {k: bytes(v) for k, v in files.items()}
        self.fail_reads = fail_reads
        self.corrupt = corrupt
        self.truncate_at = truncate_at

    def size(self, source: str) -> int:
        return len(self.files[source])

    def open_range(self, source: str, offset: int):
        import io
        if self.fail_reads > 0:
            self.fail_reads -= 1
            raise OSError("injected transient read failure")
        data = self.files[source]
        if self.corrupt:
            data = b"X" + data[1:]
        if self.truncate_at is not None and self.truncate_at > offset:
            data = data[: self.truncate_at]
        return io.BytesIO(data[offset:])
