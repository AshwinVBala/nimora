from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from nimora.agent.git_workspace import GitWorkspace
from nimora.agent.loop import AgentRuntime
from nimora.agent.policy import RuntimePolicy
from nimora.agent.recording import TrajectoryRecorder
from nimora.agent.tools import build_local_tools
from nimora.agent.types import JsonObject, ModelBackend
from nimora.agent.workspace import WorkspaceBoundary


@dataclass(frozen=True, slots=True)
class EvalCheck:
    argv: list[str]
    cwd: str = "."


@dataclass(frozen=True, slots=True)
class EvalCase:
    case_id: str
    repository: Path
    revision: str
    task: str
    checks: list[EvalCheck]


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    identifiers: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            case_id = str(value.get("case_id", "")).strip()
            if not case_id or case_id in identifiers:
                raise ValueError(f"Invalid or duplicate case_id at {path}:{line_number}")
            revision = str(value.get("revision", ""))
            invalid_revision = len(revision) != 40 or any(
                character not in "0123456789abcdef" for character in revision
            )
            if invalid_revision:
                raise ValueError(f"revision must be a full Git SHA at {path}:{line_number}")
            checks = []
            for check in value.get("checks", []):
                argv = check.get("argv") if isinstance(check, dict) else None
                if not isinstance(argv, list) or not argv or not all(
                    isinstance(item, str) for item in argv
                ):
                    raise ValueError(f"invalid check argv at {path}:{line_number}")
                checks.append(EvalCheck(argv, str(check.get("cwd", "."))))
            if not checks:
                raise ValueError(f"eval case needs at least one check at {path}:{line_number}")
            task = str(value.get("task", "")).strip()
            repository = value.get("repository")
            if not task or not isinstance(repository, str) or not repository:
                raise ValueError(f"eval case needs repository and task at {path}:{line_number}")
            identifiers.add(case_id)
            cases.append(
                EvalCase(
                    case_id=case_id,
                    repository=Path(repository).expanduser().resolve(),
                    revision=revision,
                    task=task,
                    checks=checks,
                )
            )
    if not cases:
        raise ValueError("evaluation manifest is empty")
    return cases


class EvaluationHarness:
    def __init__(
        self,
        backend_factory: Callable[[EvalCase], ModelBackend],
        policy: RuntimePolicy,
        output: str | Path,
        trajectory_output: str | Path | None = None,
    ) -> None:
        self.backend_factory = backend_factory
        self.policy = policy
        self.output = Path(output)
        self.trajectory_output = trajectory_output

    def run(self, cases: list[EvalCase]) -> JsonObject:
        if self.output.exists():
            raise FileExistsError(f"refusing to overwrite eval output: {self.output}")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        passed = 0
        results = []
        with self.output.open("w", encoding="utf-8") as handle:
            for case in cases:
                result = self._run_case(case)
                passed += int(result["passed"])
                results.append(result)
                handle.write(json.dumps(result, sort_keys=True) + "\n")
                handle.flush()
        return {"cases": len(cases), "passed": passed, "pass_rate": passed / len(cases)}

    def _run_case(self, case: EvalCase) -> JsonObject:
        with tempfile.TemporaryDirectory(prefix="nimora-eval-") as temporary:
            workspace_path = Path(temporary) / "workspace"
            self._clone_case(case, workspace_path)
            workspace = WorkspaceBoundary(workspace_path, self.policy.max_file_bytes)
            git = GitWorkspace(workspace, self.policy.command_timeout_seconds)
            tools = build_local_tools(workspace, self.policy, git)
            runtime = AgentRuntime(
                self.backend_factory(case),
                tools,
                TrajectoryRecorder(self.trajectory_output),
            )
            run = runtime.run(case.task)
            checks = [self._run_check(workspace, check) for check in case.checks]
            diff = subprocess.run(
                ["git", "-C", str(workspace_path), "diff", "--binary", case.revision],
                check=False,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=self.policy.command_timeout_seconds,
            ).stdout
            untracked = subprocess.run(
                [
                    "git",
                    "-C",
                    str(workspace_path),
                    "ls-files",
                    "--others",
                    "--exclude-standard",
                    "-z",
                ],
                check=True,
                capture_output=True,
                timeout=self.policy.command_timeout_seconds,
            ).stdout.split(b"\0")
            fingerprint = hashlib.sha256(diff.encode("utf-8"))
            for raw_path in sorted(value for value in untracked if value):
                relative = raw_path.decode("utf-8")
                target = workspace.resolve(relative)
                fingerprint.update(raw_path + b"\0")
                with target.open("rb") as handle:
                    content = handle.read(self.policy.max_file_bytes + 1)
                if len(content) > self.policy.max_file_bytes:
                    fingerprint.update(b"<oversized-untracked-file>")
                else:
                    fingerprint.update(content)
            passed = run.status == "completed" and all(
                check["exit_code"] == 0 for check in checks
            )
            return {
                "case_id": case.case_id,
                "passed": passed,
                "runtime_status": run.status,
                "steps": run.steps,
                "checks": checks,
                "diff_sha256": fingerprint.hexdigest(),
                "diff_bytes": len(diff.encode("utf-8")),
                "untracked_files": len([value for value in untracked if value]),
            }

    def _clone_case(self, case: EvalCase, destination: Path) -> None:
        if not case.repository.is_dir():
            raise ValueError(f"eval repository does not exist: {case.repository}")
        subprocess.run(
            ["git", "clone", "--quiet", "--no-hardlinks", str(case.repository), str(destination)],
            check=True,
            capture_output=True,
            text=True,
            timeout=self.policy.command_timeout_seconds,
        )
        subprocess.run(
            ["git", "-C", str(destination), "checkout", "--quiet", case.revision],
            check=True,
            capture_output=True,
            text=True,
            timeout=self.policy.command_timeout_seconds,
        )

    def _run_check(self, workspace: WorkspaceBoundary, check: EvalCheck) -> JsonObject:
        result = workspace.run_command(
            check.argv,
            check.cwd,
            self.policy.command_timeout_seconds,
            self.policy.max_observation_chars,
        )
        return {"definition": asdict(check), **result}
