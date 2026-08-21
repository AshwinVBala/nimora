from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from nimora.agent.git_workspace import GitWorkspace
from nimora.agent.policy import RuntimePolicy
from nimora.agent.types import Action, JsonObject, ToolResult, ToolSpec
from nimora.agent.workspace import WorkspaceBoundary


Handler = Callable[[JsonObject], Any]


def _validate_arguments(schema: JsonObject, value: Any, location: str = "arguments") -> None:
    expected = schema.get("type")
    types = expected if isinstance(expected, list) else [expected]
    matches = any(_matches_type(name, value) for name in types)
    if expected is not None and not matches:
        raise ValueError(f"{location} has the wrong type")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{location} is not an allowed value")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in value:
                raise ValueError(f"{location}.{required} is required")
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(properties))
            if unknown:
                raise ValueError(f"{location} has unknown fields: {unknown}")
        for key, item in value.items():
            if key in properties:
                _validate_arguments(properties[key], item, f"{location}.{key}")
    if isinstance(value, list):
        if len(value) < int(schema.get("minItems", 0)):
            raise ValueError(f"{location} has too few items")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                _validate_arguments(item_schema, item, f"{location}[{index}]")
    if isinstance(value, int) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{location} is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{location} is above maximum")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ValueError(f"{location} is shorter than allowed")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ValueError(f"{location} is longer than allowed")


def _matches_type(name: Any, value: Any) -> bool:
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    return False


def _object_schema(
    properties: JsonObject,
    required: list[str] | None = None,
    *,
    additional_properties: bool = False,
) -> JsonObject:
    value: JsonObject = {
        "type": "object",
        "properties": properties,
        "additionalProperties": additional_properties,
    }
    if required:
        value["required"] = required
    return value


class ToolRegistry:
    def __init__(self, policy: RuntimePolicy) -> None:
        self.policy = policy
        self._tools: dict[str, tuple[ToolSpec, Handler]] = {}

    def register(self, spec: ToolSpec, handler: Handler) -> None:
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool: {spec.name}")
        self._tools[spec.name] = (spec, handler)

    @property
    def specs(self) -> list[ToolSpec]:
        return [self._tools[name][0] for name in sorted(self._tools)]

    def execute(self, action: Action) -> ToolResult:
        try:
            registered = self._tools.get(action.name)
            if registered is None:
                raise ValueError(f"tool is not registered: {action.name}")
            _validate_arguments(registered[0].parameters, action.arguments)
            self.policy.authorize(action)
            output = registered[1](action.arguments)
            json.dumps(output)
            return ToolResult(ok=True, output=output)
        except Exception as error:
            return ToolResult(ok=False, error=f"{type(error).__name__}: {error}")


def build_local_tools(
    workspace: WorkspaceBoundary,
    policy: RuntimePolicy,
    git: GitWorkspace | None = None,
) -> ToolRegistry:
    registry = ToolRegistry(policy)
    registry.register(
        ToolSpec(
            "workspace.list",
            "List direct children of a workspace directory.",
            _object_schema(
                {
                    "path": {"type": "string", "default": "."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                }
            ),
        ),
        lambda values: workspace.list_files(
            str(values.get("path", ".")), int(values.get("limit", 500))
        ),
    )
    registry.register(
        ToolSpec(
            "workspace.read",
            "Read UTF-8 text with a content hash for concurrency-safe writes.",
            _object_schema(
                {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": ["integer", "null"], "minimum": 1},
                },
                ["path"],
            ),
        ),
        lambda values: workspace.read_file(
            str(values["path"]),
            int(values.get("start_line", 1)),
            int(values["end_line"]) if values.get("end_line") is not None else None,
        ),
    )
    registry.register(
        ToolSpec(
            "workspace.search",
            "Search UTF-8 workspace files using a literal string or regular expression.",
            _object_schema(
                {
                    "query": {"type": "string", "minLength": 1, "maxLength": 500},
                    "path": {"type": "string", "default": "."},
                    "regex": {"type": "boolean", "default": False},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                },
                ["query"],
            ),
        ),
        lambda values: workspace.search(
            str(values["query"]),
            str(values.get("path", ".")),
            bool(values.get("regex", False)),
            int(values.get("limit", 200)),
        ),
    )
    registry.register(
        ToolSpec(
            "workspace.write",
            "Atomically create or replace UTF-8 text. Existing files require their read hash.",
            _object_schema(
                {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "expected_sha256": {"type": ["string", "null"]},
                },
                ["path", "content", "expected_sha256"],
            ),
        ),
        lambda values: workspace.write_file(
            str(values["path"]),
            str(values["content"]),
            (
                str(values["expected_sha256"])
                if values.get("expected_sha256") is not None
                else None
            ),
        ),
    )
    registry.register(
        ToolSpec(
            "workspace.replace",
            "Replace an exact text span after verifying the file hash and match count.",
            _object_schema(
                {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                    "expected_sha256": {"type": "string"},
                    "expected_occurrences": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 1,
                    },
                },
                ["path", "old_text", "new_text", "expected_sha256"],
            ),
        ),
        lambda values: workspace.replace_text(
            str(values["path"]),
            str(values["old_text"]),
            str(values["new_text"]),
            str(values["expected_sha256"]),
            int(values.get("expected_occurrences", 1)),
        ),
    )
    registry.register(
        ToolSpec(
            "shell.run",
            "Run an allowlisted command without a shell inside a workspace directory.",
            _object_schema(
                {
                    "argv": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "cwd": {"type": "string", "default": "."},
                },
                ["argv"],
            ),
        ),
        lambda values: workspace.run_command(
            [str(value) for value in values["argv"]],
            str(values.get("cwd", ".")),
            policy.command_timeout_seconds,
            policy.max_observation_chars,
        ),
    )
    if git is not None:
        _register_git_tools(registry, git)
    return registry


def _register_git_tools(registry: ToolRegistry, git: GitWorkspace) -> None:
    empty = _object_schema({})
    registry.register(
        ToolSpec("git.revision", "Get HEAD SHA and branch.", empty),
        lambda _: git.revision(),
    )
    registry.register(
        ToolSpec("git.status", "Get porcelain Git status.", empty),
        lambda _: git.status(),
    )
    registry.register(
        ToolSpec(
            "git.diff",
            "Read a working-tree or staged diff.",
            _object_schema(
                {
                    "staged": {"type": "boolean", "default": False},
                    "path": {"type": ["string", "null"]},
                }
            ),
        ),
        lambda values: git.diff(
            bool(values.get("staged", False)),
            str(values["path"]) if values.get("path") is not None else None,
        ),
    )
    registry.register(
        ToolSpec(
            "git.create_branch",
            "Create and switch to a branch if HEAD still matches expected_head.",
            _object_schema(
                {
                    "name": {"type": "string"},
                    "expected_head": {"type": "string"},
                },
                ["name", "expected_head"],
            ),
        ),
        lambda values: git.create_branch(
            str(values["name"]), str(values["expected_head"])
        ),
    )
    registry.register(
        ToolSpec(
            "git.commit",
            "Commit only explicit paths if HEAD matches and no unrelated changes are staged.",
            _object_schema(
                {
                    "message": {"type": "string"},
                    "paths": {"type": "array", "items": {"type": "string"}},
                    "expected_head": {"type": "string"},
                },
                ["message", "paths", "expected_head"],
            ),
        ),
        lambda values: git.commit(
            str(values["message"]),
            [str(value) for value in values["paths"]],
            str(values["expected_head"]),
        ),
    )
    registry.register(
        ToolSpec(
            "git.push",
            "Push the current branch only if HEAD still matches expected_head.",
            _object_schema(
                {
                    "remote": {"type": "string"},
                    "branch": {"type": "string"},
                    "expected_head": {"type": "string"},
                },
                ["remote", "branch", "expected_head"],
            ),
        ),
        lambda values: git.push(
            str(values["remote"]),
            str(values["branch"]),
            str(values["expected_head"]),
        ),
    )
