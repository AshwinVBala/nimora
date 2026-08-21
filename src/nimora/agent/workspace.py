from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


class WorkspaceBoundary:
    """A path boundary for trusted local tools, not an OS security sandbox."""

    def __init__(self, root: str | Path, max_file_bytes: int = 1_000_000) -> None:
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError(f"workspace is not a directory: {self.root}")
        self.max_file_bytes = max_file_bytes

    def resolve(self, value: str, *, must_exist: bool = True) -> Path:
        relative = PurePosixPath(value or ".")
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe workspace path: {value!r}")
        candidate = (self.root / Path(*relative.parts)).resolve(strict=must_exist)
        try:
            candidate.relative_to(self.root)
        except ValueError as error:
            raise ValueError(f"path escapes workspace: {value!r}") from error
        return candidate

    def list_files(self, path: str = ".", limit: int = 500) -> dict[str, Any]:
        directory = self.resolve(path)
        if not directory.is_dir():
            raise ValueError(f"not a directory: {path}")
        entries: list[dict[str, Any]] = []
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            if self._is_sensitive(child):
                continue
            entries.append(
                {
                    "name": child.name,
                    "path": child.relative_to(self.root).as_posix(),
                    "type": "directory" if child.is_dir() else "file",
                }
            )
            if len(entries) >= limit:
                break
        return {"entries": entries, "truncated": len(entries) >= limit}

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        target = self.resolve(path)
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"not a regular file: {path}")
        raw = self._bounded_read(target)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"file is not UTF-8 text: {path}") from error
        lines = text.splitlines(keepends=True)
        start = max(1, start_line)
        stop = len(lines) if end_line is None else min(len(lines), end_line)
        content = "".join(lines[start - 1 : stop])
        return {
            "path": target.relative_to(self.root).as_posix(),
            "content": content,
            "start_line": start,
            "end_line": stop,
            "line_count": len(lines),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    def search(
        self,
        query: str,
        path: str = ".",
        regex: bool = False,
        limit: int = 200,
        max_scanned_files: int = 10_000,
    ) -> dict[str, Any]:
        root = self.resolve(path)
        pattern = re.compile(query if regex else re.escape(query))
        matches: list[dict[str, Any]] = []
        scanned_files = 0
        candidates = [root] if root.is_file() else root.rglob("*")
        for candidate in candidates:
            if len(matches) >= limit or scanned_files >= max_scanned_files:
                break
            if self._is_sensitive(candidate) or candidate.is_symlink() or not candidate.is_file():
                continue
            try:
                scanned_files += 1
                raw = self._bounded_read(candidate)
                text = raw.decode("utf-8")
            except (ValueError, UnicodeDecodeError, OSError):
                continue
            for line_number, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    matches.append(
                        {
                            "path": candidate.relative_to(self.root).as_posix(),
                            "line": line_number,
                            "text": line[:500],
                        }
                    )
                    if len(matches) >= limit:
                        break
        return {
            "matches": matches,
            "scanned_files": scanned_files,
            "truncated": len(matches) >= limit or scanned_files >= max_scanned_files,
        }

    def write_file(
        self,
        path: str,
        content: str,
        expected_sha256: str | None,
    ) -> dict[str, Any]:
        target = self.resolve(path, must_exist=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.is_symlink() or not target.is_file():
                raise ValueError(f"not a writable regular file: {path}")
            current = hashlib.sha256(self._bounded_read(target)).hexdigest()
            if expected_sha256 is None:
                raise ValueError("expected_sha256 is required when replacing a file")
            if current != expected_sha256:
                raise ValueError("file changed since it was read")
            mode = target.stat().st_mode & 0o777
        else:
            if expected_sha256 is not None:
                raise ValueError("expected_sha256 must be null when creating a file")
            mode = 0o644
        encoded = content.encode("utf-8")
        if len(encoded) > self.max_file_bytes:
            raise ValueError("new file exceeds max_file_bytes")
        descriptor, temporary_name = tempfile.mkstemp(prefix=".nimora-", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, mode)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        return {
            "path": target.relative_to(self.root).as_posix(),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "bytes": len(encoded),
        }

    def replace_text(
        self,
        path: str,
        old_text: str,
        new_text: str,
        expected_sha256: str,
        expected_occurrences: int = 1,
    ) -> dict[str, Any]:
        if not old_text:
            raise ValueError("old_text cannot be empty")
        if expected_occurrences < 1:
            raise ValueError("expected_occurrences must be positive")
        target = self.resolve(path)
        raw = self._bounded_read(target)
        current = hashlib.sha256(raw).hexdigest()
        if current != expected_sha256:
            raise ValueError("file changed since it was read")
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"file is not UTF-8 text: {path}") from error
        occurrences = content.count(old_text)
        if occurrences != expected_occurrences:
            raise ValueError(
                f"expected {expected_occurrences} matches for old_text, found {occurrences}"
            )
        replacement = content.replace(old_text, new_text, expected_occurrences)
        return self.write_file(path, replacement, expected_sha256)

    def run_command(
        self,
        argv: list[str],
        cwd: str = ".",
        timeout_seconds: int = 300,
        output_limit: int = 30_000,
    ) -> dict[str, Any]:
        directory = self.resolve(cwd)
        environment = {
            key: os.environ[key]
            for key in ("HOME", "LANG", "LC_ALL", "PATH", "TMPDIR")
            if key in os.environ
        }
        completed = subprocess.run(
            argv,
            cwd=directory,
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            env=environment,
            timeout=timeout_seconds,
        )
        stdout = completed.stdout[:output_limit]
        stderr = completed.stderr[:output_limit]
        return {
            "argv": argv,
            "cwd": directory.relative_to(self.root).as_posix() or ".",
            "exit_code": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "truncated": (
                len(completed.stdout) > output_limit
                or len(completed.stderr) > output_limit
            ),
        }

    def _bounded_read(self, path: Path) -> bytes:
        if self._is_sensitive(path):
            raise ValueError(f"refusing to read protected file: {path}")
        with path.open("rb") as handle:
            raw = handle.read(self.max_file_bytes + 1)
        if len(raw) > self.max_file_bytes:
            raise ValueError(f"file exceeds max_file_bytes: {path}")
        return raw

    def is_sensitive_path(self, value: str) -> bool:
        try:
            return self._is_sensitive(self.resolve(value, must_exist=False))
        except ValueError:
            return True

    def _is_sensitive(self, path: Path) -> bool:
        try:
            relative = path.relative_to(self.root)
        except ValueError:
            return True
        if ".git" in relative.parts:
            return True
        name = relative.name.lower()
        return name == ".env" or name.startswith(".env.") or path.suffix.lower() in {
            ".key",
            ".pem",
        }
