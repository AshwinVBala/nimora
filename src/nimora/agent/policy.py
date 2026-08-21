from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import yaml

from nimora.agent.types import Action, PolicyDenied


def _matches(path: str, pattern: str) -> bool:
    if fnmatch.fnmatch(path, pattern):
        return True
    return pattern.startswith("**/") and fnmatch.fnmatch(path, pattern[3:])


@dataclass(slots=True)
class RuntimePolicy:
    max_steps: int = 64
    max_context_chars: int = 120_000
    max_observation_chars: int = 30_000
    max_file_bytes: int = 1_000_000
    command_timeout_seconds: int = 300
    require_evidence_before_completion: bool = True
    require_passing_checks_for_approval: bool = True
    allow_regex_search: bool = False
    allow_write: bool = False
    allow_shell: bool = False
    allow_git_mutation: bool = False
    allow_git_push: bool = False
    allow_remote_read: bool = False
    allow_open_change: bool = False
    allow_approve_change: bool = False
    allow_merge_change: bool = False
    allowed_commands: list[str] = field(default_factory=list)
    allowed_write_globs: list[str] = field(default_factory=lambda: ["**"])
    protected_globs: list[str] = field(
        default_factory=lambda: [
            ".git",
            ".git/**",
            ".env*",
            "**/.env*",
            "**/*.key",
            "**/*.pem",
        ]
    )

    def validate(self) -> None:
        for name in (
            "max_steps",
            "max_context_chars",
            "max_observation_chars",
            "max_file_bytes",
            "command_timeout_seconds",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.allow_shell and not self.allowed_commands:
            raise ValueError("allow_shell requires at least one allowed command")

    @classmethod
    def load(cls, path: str) -> RuntimePolicy:
        with open(path, encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        if not isinstance(values, dict):
            raise ValueError("runtime policy must be a YAML object")
        policy = cls(**values)
        policy.validate()
        return policy

    def authorize(self, action: Action) -> None:
        name = action.name
        if name in {
            "workspace.list",
            "git.diff",
            "git.revision",
            "git.status",
        }:
            return
        if name in {"workspace.read", "workspace.search"}:
            self._authorize_read(action)
            return
        if name in {"workspace.replace", "workspace.write"}:
            self._authorize_write(action)
            return
        if name == "shell.run":
            self._authorize_shell(action)
            return
        if name in {"git.create_branch", "git.commit"}:
            self._require(self.allow_git_mutation, "Git mutation is disabled")
            return
        if name == "git.push":
            self._require(self.allow_git_push, "Git push is disabled")
            return
        if name in {"provider.get_change", "provider.get_checks", "provider.get_diff"}:
            self._require(self.allow_remote_read, "remote provider reads are disabled")
            return
        if name == "provider.open_change":
            self._require(self.allow_open_change, "opening remote changes is disabled")
            return
        if name == "provider.approve_change":
            self._require(self.allow_approve_change, "remote approval is disabled")
            return
        if name == "provider.merge_change":
            self._require(self.allow_merge_change, "remote merge is disabled")
            return
        raise PolicyDenied(f"Unknown or unauthorized tool: {name}")

    def _authorize_write(self, action: Action) -> None:
        self._require(self.allow_write, "workspace writes are disabled")
        path_value = action.arguments.get("path")
        if not isinstance(path_value, str):
            raise PolicyDenied(f"{action.name} requires a string path")
        path = PurePosixPath(path_value).as_posix()
        if any(_matches(path, pattern) for pattern in self.protected_globs):
            raise PolicyDenied(f"write targets a protected path: {path}")
        if not any(_matches(path, pattern) for pattern in self.allowed_write_globs):
            raise PolicyDenied(f"write path is not allowlisted: {path}")

    def _authorize_read(self, action: Action) -> None:
        path_value = action.arguments.get("path", ".")
        if not isinstance(path_value, str):
            raise PolicyDenied(f"{action.name} requires a string path")
        path = PurePosixPath(path_value).as_posix()
        if path != "." and any(
            _matches(path, pattern) for pattern in self.protected_globs
        ):
            raise PolicyDenied(f"read targets a protected path: {path}")
        if action.name == "workspace.search" and action.arguments.get("regex"):
            self._require(self.allow_regex_search, "regular-expression search is disabled")

    def _authorize_shell(self, action: Action) -> None:
        self._require(self.allow_shell, "shell execution is disabled")
        argv = action.arguments.get("argv")
        if not isinstance(argv, list) or not argv or not all(
            isinstance(value, str) for value in argv
        ):
            raise PolicyDenied("shell.run requires a non-empty string argv array")
        executable = PurePosixPath(argv[0]).name
        if executable != argv[0]:
            raise PolicyDenied("shell executable must be an allowlisted command name")
        if executable not in self.allowed_commands:
            raise PolicyDenied(f"command is not allowlisted: {executable}")

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise PolicyDenied(message)
