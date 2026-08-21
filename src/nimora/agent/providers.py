from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import yaml

from nimora.agent.tools import ToolRegistry, _object_schema
from nimora.agent.types import JsonObject, ToolSpec


@dataclass(frozen=True, slots=True)
class RemoteChange:
    number: int
    url: str
    state: str
    title: str
    head_sha: str
    base_branch: str


class GitProvider(Protocol):
    def get_change(self, number: int) -> RemoteChange: ...

    def get_diff(self, number: int) -> JsonObject: ...

    def get_checks(self, sha: str) -> JsonObject: ...

    def open_change(
        self, title: str, body: str, head_branch: str, base_branch: str
    ) -> RemoteChange: ...

    def approve_change(
        self, number: int, expected_sha: str, body: str
    ) -> JsonObject: ...

    def merge_change(
        self, number: int, expected_sha: str, method: str
    ) -> JsonObject: ...


class HttpTransport:
    def __init__(
        self,
        base_url: str,
        headers: dict[str, str],
        timeout_seconds: int = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Accept": "application/json", **headers}
        self.timeout_seconds = timeout_seconds

    def request(
        self,
        method: str,
        path: str,
        payload: JsonObject | None = None,
    ) -> Any:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = dict(self.headers)
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read(10_000_001)
                if len(raw) > 10_000_000:
                    raise RuntimeError("provider JSON response exceeded 10 MB")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as error:
            detail = error.read(4_096).decode("utf-8", errors="replace")
            raise RuntimeError(f"provider returned HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"provider request failed: {error.reason}") from error

    def request_text(self, path: str, accept: str = "text/plain") -> str:
        headers = {**self.headers, "Accept": accept}
        request = urllib.request.Request(
            self.base_url + path,
            headers=headers,
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read(500_001).decode("utf-8", errors="replace")
        except urllib.error.HTTPError as error:
            detail = error.read(4_096).decode("utf-8", errors="replace")
            raise RuntimeError(f"provider returned HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"provider request failed: {error.reason}") from error


class GitHubProvider:
    def __init__(
        self,
        repository: str,
        token: str,
        base_url: str = "https://api.github.com",
    ) -> None:
        self.repository = repository.strip("/")
        self.http = HttpTransport(
            base_url,
            {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2026-03-10",
            },
        )

    def get_change(self, number: int) -> RemoteChange:
        value = self.http.request("GET", f"/repos/{self.repository}/pulls/{number}")
        return RemoteChange(
            number=int(value["number"]),
            url=str(value["html_url"]),
            state=str(value["state"]),
            title=str(value["title"]),
            head_sha=str(value["head"]["sha"]),
            base_branch=str(value["base"]["ref"]),
        )

    def get_checks(self, sha: str) -> JsonObject:
        value = self.http.request(
            "GET", f"/repos/{self.repository}/commits/{sha}/check-runs?per_page=100"
        )
        checks = [
            {
                "name": item.get("name"),
                "status": item.get("status"),
                "conclusion": item.get("conclusion"),
                "url": item.get("html_url"),
            }
            for item in value.get("check_runs", [])
        ]
        return {"sha": sha, "checks": checks, "total": len(checks)}

    def get_diff(self, number: int) -> JsonObject:
        diff = self.http.request_text(
            f"/repos/{self.repository}/pulls/{number}",
            "application/vnd.github.v3.diff",
        )
        return {"number": number, "diff": diff[:500_000], "truncated": len(diff) > 500_000}

    def open_change(
        self, title: str, body: str, head_branch: str, base_branch: str
    ) -> RemoteChange:
        value = self.http.request(
            "POST",
            f"/repos/{self.repository}/pulls",
            {"title": title, "body": body, "head": head_branch, "base": base_branch},
        )
        return self.get_change(int(value["number"]))

    def approve_change(
        self, number: int, expected_sha: str, body: str
    ) -> JsonObject:
        change = self._require_revision(number, expected_sha)
        value = self.http.request(
            "POST",
            f"/repos/{self.repository}/pulls/{number}/reviews",
            {"body": body, "event": "APPROVE", "commit_id": expected_sha},
        )
        return {"change": asdict(change), "review_id": value.get("id"), "approved": True}

    def merge_change(
        self, number: int, expected_sha: str, method: str
    ) -> JsonObject:
        self._require_revision(number, expected_sha)
        return self.http.request(
            "PUT",
            f"/repos/{self.repository}/pulls/{number}/merge",
            {"sha": expected_sha, "merge_method": method},
        )

    def _require_revision(self, number: int, expected_sha: str) -> RemoteChange:
        change = self.get_change(number)
        if change.head_sha != expected_sha:
            raise ValueError(
                f"change revision moved: expected {expected_sha}, found {change.head_sha}"
            )
        return change


class GitLabProvider:
    def __init__(
        self,
        project: str,
        token: str,
        base_url: str = "https://gitlab.com/api/v4",
    ) -> None:
        self.project = urllib.parse.quote(project, safe="")
        self.http = HttpTransport(base_url, {"PRIVATE-TOKEN": token})

    def get_change(self, number: int) -> RemoteChange:
        value = self.http.request(
            "GET", f"/projects/{self.project}/merge_requests/{number}"
        )
        return RemoteChange(
            number=int(value["iid"]),
            url=str(value["web_url"]),
            state=str(value["state"]),
            title=str(value["title"]),
            head_sha=str(value["sha"]),
            base_branch=str(value["target_branch"]),
        )

    def get_checks(self, sha: str) -> JsonObject:
        value = self.http.request(
            "GET",
            f"/projects/{self.project}/repository/commits/{sha}/statuses?per_page=100",
        )
        checks = [
            {
                "name": item.get("name"),
                "status": item.get("status"),
                "url": item.get("target_url"),
            }
            for item in value
        ]
        return {"sha": sha, "checks": checks, "total": len(checks)}

    def get_diff(self, number: int) -> JsonObject:
        value = self.http.request(
            "GET", f"/projects/{self.project}/merge_requests/{number}/diffs?per_page=100"
        )
        rendered = "\n".join(
            f"diff --git a/{item.get('old_path')} b/{item.get('new_path')}\n{item.get('diff', '')}"
            for item in value
        )
        return {
            "number": number,
            "diff": rendered[:500_000],
            "truncated": len(rendered) > 500_000 or len(value) >= 100,
        }

    def open_change(
        self, title: str, body: str, head_branch: str, base_branch: str
    ) -> RemoteChange:
        value = self.http.request(
            "POST",
            f"/projects/{self.project}/merge_requests",
            {
                "title": title,
                "description": body,
                "source_branch": head_branch,
                "target_branch": base_branch,
            },
        )
        return self.get_change(int(value["iid"]))

    def approve_change(
        self, number: int, expected_sha: str, body: str
    ) -> JsonObject:
        del body
        change = self._require_revision(number, expected_sha)
        value = self.http.request(
            "POST",
            f"/projects/{self.project}/merge_requests/{number}/approve",
            {"sha": expected_sha},
        )
        return {"change": asdict(change), "approved": bool(value.get("approved", True))}

    def merge_change(
        self, number: int, expected_sha: str, method: str
    ) -> JsonObject:
        del method
        self._require_revision(number, expected_sha)
        return self.http.request(
            "PUT",
            f"/projects/{self.project}/merge_requests/{number}/merge",
            {"sha": expected_sha},
        )

    def _require_revision(self, number: int, expected_sha: str) -> RemoteChange:
        change = self.get_change(number)
        if change.head_sha != expected_sha:
            raise ValueError(
                f"change revision moved: expected {expected_sha}, found {change.head_sha}"
            )
        return change


def _api_v1_url(base_url: str) -> str:
    """Accept either an instance URL or an already-qualified v1 API URL."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/api/v1"):
        return normalized
    return normalized + "/api/v1"


class GiteaProvider(GitHubProvider):
    def __init__(self, repository: str, token: str, base_url: str) -> None:
        self.repository = repository.strip("/")
        self.http = HttpTransport(
            _api_v1_url(base_url),
            {"Authorization": f"token {token}"},
        )

    def get_checks(self, sha: str) -> JsonObject:
        value = self.http.request(
            "GET", f"/repos/{self.repository}/commits/{sha}/statuses"
        )
        checks = [
            {
                "name": item.get("context"),
                "status": item.get("status"),
                "url": item.get("target_url"),
            }
            for item in value
        ]
        return {"sha": sha, "checks": checks, "total": len(checks)}

    def get_diff(self, number: int) -> JsonObject:
        diff = self.http.request_text(f"/repos/{self.repository}/pulls/{number}.diff")
        return {"number": number, "diff": diff[:500_000], "truncated": len(diff) > 500_000}

    def approve_change(
        self, number: int, expected_sha: str, body: str
    ) -> JsonObject:
        change = self._require_revision(number, expected_sha)
        value = self.http.request(
            "POST",
            f"/repos/{self.repository}/pulls/{number}/reviews",
            {"body": body, "event": "APPROVED", "commit_id": expected_sha},
        )
        return {"change": asdict(change), "review_id": value.get("id"), "approved": True}

    def merge_change(
        self, number: int, expected_sha: str, method: str
    ) -> JsonObject:
        self._require_revision(number, expected_sha)
        return self.http.request(
            "POST",
            f"/repos/{self.repository}/pulls/{number}/merge",
            {"Do": method, "head_commit_id": expected_sha},
        )


class ForgejoProvider(GiteaProvider):
    """Forgejo provider using Forgejo's versioned REST API contract."""

    def __init__(self, repository: str, token: str, base_url: str) -> None:
        self.repository = repository.strip("/")
        self.http = HttpTransport(
            _api_v1_url(base_url),
            {"Authorization": f"token {token}"},
        )


def register_provider_tools(registry: ToolRegistry, provider: GitProvider) -> None:
    revisions: dict[int, str] = {}
    reviewed_diffs: set[tuple[int, str]] = set()
    checked_revisions: set[str] = set()
    passing_revisions: set[str] = set()

    def get_change(number: int) -> JsonObject:
        change = provider.get_change(number)
        revisions[number] = change.head_sha
        return asdict(change)

    def get_diff(number: int) -> JsonObject:
        change = provider.get_change(number)
        revisions[number] = change.head_sha
        value = provider.get_diff(number)
        reviewed_diffs.add((number, change.head_sha))
        return {"head_sha": change.head_sha, **value}

    def get_checks(sha: str) -> JsonObject:
        value = provider.get_checks(sha)
        checked_revisions.add(sha)
        checks = value.get("checks", [])
        all_passed = bool(checks) and all(_check_passed(item) for item in checks)
        value["all_passed"] = all_passed
        if all_passed:
            passing_revisions.add(sha)
        return value

    def require_review_evidence(number: int, sha: str) -> None:
        if revisions.get(number) != sha or (number, sha) not in reviewed_diffs:
            raise ValueError("diff evidence for this exact change revision is missing")
        if sha not in checked_revisions:
            raise ValueError("check evidence for this exact change revision is missing")
        if registry.policy.require_passing_checks_for_approval and sha not in passing_revisions:
            raise ValueError("required checks for this change revision are not passing")

    registry.register(
        ToolSpec(
            "provider.get_change",
            "Read a pull or merge request and its exact head revision.",
            _object_schema({"number": {"type": "integer", "minimum": 1}}, ["number"]),
        ),
        lambda values: get_change(int(values["number"])),
    )
    registry.register(
        ToolSpec(
            "provider.get_checks",
            "Read checks for an exact commit SHA.",
            _object_schema({"sha": {"type": "string"}}, ["sha"]),
        ),
        lambda values: get_checks(str(values["sha"])),
    )
    registry.register(
        ToolSpec(
            "provider.get_diff",
            "Read the diff and bind it to the change's current head SHA.",
            _object_schema({"number": {"type": "integer", "minimum": 1}}, ["number"]),
        ),
        lambda values: get_diff(int(values["number"])),
    )
    registry.register(
        ToolSpec(
            "provider.open_change",
            "Open a pull or merge request from an already-pushed branch.",
            _object_schema(
                {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "head_branch": {"type": "string"},
                    "base_branch": {"type": "string"},
                },
                ["title", "body", "head_branch", "base_branch"],
            ),
        ),
        lambda values: asdict(
            provider.open_change(
                str(values["title"]),
                str(values["body"]),
                str(values["head_branch"]),
                str(values["base_branch"]),
            )
        ),
    )
    registry.register(
        ToolSpec(
            "provider.approve_change",
            "Approve only if the remote head still equals expected_sha.",
            _object_schema(
                {
                    "number": {"type": "integer", "minimum": 1},
                    "expected_sha": {"type": "string"},
                    "body": {"type": "string"},
                },
                ["number", "expected_sha", "body"],
            ),
        ),
        lambda values: _approve_with_evidence(provider, require_review_evidence, values),
    )
    registry.register(
        ToolSpec(
            "provider.merge_change",
            "Merge only if the remote head still equals expected_sha.",
            _object_schema(
                {
                    "number": {"type": "integer", "minimum": 1},
                    "expected_sha": {"type": "string"},
                    "method": {
                        "type": "string",
                        "enum": ["merge", "squash", "rebase"],
                    },
                },
                ["number", "expected_sha", "method"],
            ),
        ),
        lambda values: _merge_with_evidence(provider, require_review_evidence, values),
    )


def _approve_with_evidence(
    provider: GitProvider,
    require_evidence,
    values: JsonObject,
) -> JsonObject:
    number = int(values["number"])
    sha = str(values["expected_sha"])
    require_evidence(number, sha)
    return provider.approve_change(number, sha, str(values["body"]))


def _check_passed(value: JsonObject) -> bool:
    conclusion = value.get("conclusion")
    if conclusion is not None:
        return value.get("status") == "completed" and conclusion in {"success", "skipped"}
    return value.get("status") in {"success", "skipped"}


def _merge_with_evidence(
    provider: GitProvider,
    require_evidence,
    values: JsonObject,
) -> JsonObject:
    number = int(values["number"])
    sha = str(values["expected_sha"])
    require_evidence(number, sha)
    return provider.merge_change(number, sha, str(values["method"]))


def load_provider(path: str) -> GitProvider:
    with open(path, encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ValueError("provider configuration must be a YAML object")
    kind = str(values.get("provider", "")).lower()
    repository = str(values.get("repository", "")).strip()
    token_environment = str(values.get("token_env", "")).strip()
    if not repository or not token_environment:
        raise ValueError("provider configuration needs repository and token_env")
    token = os.environ.get(token_environment)
    if not token:
        raise ValueError(
            f"provider token environment variable is unset: {token_environment}"
        )
    base_url = values.get("base_url")
    if kind == "github":
        return GitHubProvider(
            repository,
            token,
            str(base_url) if base_url else "https://api.github.com",
        )
    if kind == "gitlab":
        return GitLabProvider(
            repository,
            token,
            str(base_url) if base_url else "https://gitlab.com/api/v4",
        )
    if kind == "gitea":
        if not base_url:
            raise ValueError("Gitea provider configuration requires base_url")
        return GiteaProvider(repository, token, str(base_url))
    if kind == "forgejo":
        if not base_url:
            raise ValueError("Forgejo provider configuration requires base_url")
        return ForgejoProvider(repository, token, str(base_url))
    raise ValueError(f"unsupported provider: {kind!r}")
