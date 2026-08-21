from __future__ import annotations

import os
import re
import subprocess
from typing import Any

from nimora.agent.workspace import WorkspaceBoundary


class GitWorkspace:
    def __init__(self, workspace: WorkspaceBoundary, timeout_seconds: int = 300) -> None:
        self.workspace = workspace
        self.timeout_seconds = timeout_seconds
        top_level = self._run("rev-parse", "--show-toplevel").strip()
        if os.path.realpath(top_level) != str(workspace.root):
            raise ValueError("workspace must be the Git repository root")

    def revision(self) -> dict[str, str]:
        return {
            "sha": self._run("rev-parse", "HEAD").strip(),
            "branch": self._run("branch", "--show-current").strip(),
        }

    def status(self) -> dict[str, Any]:
        output = self._run("status", "--porcelain=v1", "--branch")
        return {"porcelain": output, "clean": not any(
            line and not line.startswith("##") for line in output.splitlines()
        )}

    def diff(self, staged: bool = False, path: str | None = None) -> dict[str, Any]:
        arguments = ["diff"]
        if staged:
            arguments.append("--cached")
        if path is not None:
            self.workspace.resolve(path, must_exist=False)
            if self.workspace.is_sensitive_path(path):
                raise ValueError(f"refusing to diff protected path: {path}")
            paths = [path]
        else:
            names = [*arguments, "--name-only", "-z"]
            raw_names = self._run(*names).split("\0")
            paths = [
                value
                for value in raw_names
                if value and not self.workspace.is_sensitive_path(value)
            ]
        if not paths:
            return {"diff": "", "truncated": False}
        arguments.extend(["--", *paths])
        output = self._run(*arguments)
        limit = 100_000
        return {"diff": output[:limit], "truncated": len(output) > limit}

    def create_branch(self, name: str, expected_head: str) -> dict[str, str]:
        current = self.revision()["sha"]
        if current != expected_head:
            raise ValueError(f"HEAD changed: expected {expected_head}, found {current}")
        self._run("check-ref-format", "--branch", name)
        self._run("switch", "-c", name)
        return self.revision()

    def commit(
        self,
        message: str,
        paths: list[str],
        expected_head: str,
    ) -> dict[str, str]:
        if not message.strip():
            raise ValueError("commit message cannot be empty")
        if not paths:
            raise ValueError("commit requires at least one explicit path")
        current = self.revision()["sha"]
        if current != expected_head:
            raise ValueError(f"HEAD changed: expected {expected_head}, found {current}")
        cached = self._run_result("diff", "--cached", "--quiet")
        if cached.returncode not in {0, 1}:
            raise ValueError(cached.stderr.strip() or "could not inspect staged changes")
        if cached.returncode == 1:
            raise ValueError("refusing to commit while unrelated staged changes exist")
        for path in paths:
            self.workspace.resolve(path, must_exist=False)
        self._run("add", "--", *paths)
        self._run("commit", "-m", message, "--only", "--", *paths)
        return self.revision()

    def push(self, remote: str, branch: str, expected_head: str) -> dict[str, str]:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", remote):
            raise ValueError("remote name contains unsupported characters")
        current = self.revision()
        if current["sha"] != expected_head:
            raise ValueError(
                f"HEAD changed: expected {expected_head}, found {current['sha']}"
            )
        if current["branch"] != branch:
            raise ValueError(
                f"current branch is {current['branch']!r}, expected {branch!r}"
            )
        self._run("push", "--set-upstream", remote, branch)
        return current

    def _run(self, *arguments: str) -> str:
        completed = self._run_result(*arguments)
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            detail = re.sub(r"(https?://)[^/@\s]+@", r"\1<REDACTED>@", detail)
            raise ValueError(f"git {' '.join(arguments)} failed: {detail}")
        return completed.stdout

    def _run_result(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        environment = {
            key: os.environ[key]
            for key in ("HOME", "LANG", "LC_ALL", "PATH", "TMPDIR")
            if key in os.environ
        }
        return subprocess.run(
            ["git", "-C", str(self.workspace.root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            env=environment,
            timeout=self.timeout_seconds,
        )
