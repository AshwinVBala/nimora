"""Build Nimora's deterministic, synthetic v0.0.1-alpha trajectory corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nimora.agent.backend import RUNTIME_INSTRUCTION
from nimora.agent.types import Decision
from nimora.serialization import canonicalize_trajectory_messages

JsonObject = dict[str, Any]
KNOWN_TOOLS = {
    "git.commit",
    "git.create_branch",
    "git.diff",
    "git.push",
    "git.revision",
    "git.status",
    "provider.approve_change",
    "provider.get_change",
    "provider.get_checks",
    "provider.get_diff",
    "provider.merge_change",
    "provider.open_change",
    "shell.run",
    "workspace.list",
    "workspace.read",
    "workspace.replace",
    "workspace.search",
    "workspace.write",
}


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _assistant(*, plan: str | None = None, action=None, result=None) -> JsonObject:
    value: JsonObject = {}
    if plan is not None:
        value["plan"] = plan
    if action is not None:
        value["action"] = action
    else:
        value["result"] = result
    Decision.from_value(value)
    return {"role": "assistant", "decision": value}


def _action(name: str, arguments: JsonObject) -> JsonObject:
    if name not in KNOWN_TOOLS:
        raise ValueError(f"Unknown generated tool: {name}")
    return {"name": name, "arguments": arguments}


def _observation(
    name: str,
    *,
    output: Any = None,
    error: str | None = None,
) -> JsonObject:
    value: JsonObject = {"ok": error is None}
    if output is not None:
        value["output"] = output
    if error is not None:
        value["error"] = error
    return {
        "role": "tool",
        "name": name,
        "content": json.dumps(value, sort_keys=True),
    }


def _record(family: str, index: int, task: str, turns: list[JsonObject]) -> JsonObject:
    return {
        "messages": [
            {"role": "system", "content": RUNTIME_INSTRUCTION},
            {"role": "user", "content": task},
            *turns,
        ],
        "metadata": {
            "dataset": "nimora-agent-alpha-v0.0.1",
            "family": family,
            "index": index,
            "provenance": "deterministic synthetic template",
            "license": "Apache-2.0",
            "human_review_status": "template-reviewed",
        },
    }


def inspect(index: int) -> JsonObject:
    nouns = ["parser", "router", "serializer", "scheduler", "tokenizer"]
    noun = nouns[index % len(nouns)]
    source = f"src/{noun}.py"
    digest = _sha(f"inspect:{index}:{source}")
    return _record(
        "inspect",
        index,
        f"Locate the {noun} implementation and summarize the relevant entry point.",
        [
            _assistant(
                plan="Search narrowly before reading files.",
                action=_action(
                    "workspace.search", {"query": f"def {noun}", "path": "src"}
                ),
            ),
            _observation(
                "workspace.search",
                output={"matches": [{"path": source, "line": 17}]},
            ),
            _assistant(
                plan="Read the matching implementation.",
                action=_action(
                    "workspace.read",
                    {"path": source, "start_line": 1, "end_line": 90},
                ),
            ),
            _observation(
                "workspace.read",
                output={
                    "path": source,
                    "sha256": digest,
                    "content": f"def {noun}(value):\n    return normalize(value)\n",
                },
            ),
            _assistant(
                result=f"Found `{noun}` in `{source}` at line 17 and inspected its implementation."
            ),
        ],
    )


def edit_and_test(index: int) -> JsonObject:
    modules = ["parser", "config", "client", "cache", "events"]
    module = modules[index % len(modules)]
    path = f"src/{module}.py"
    digest = _sha(f"edit:{index}:old")
    return _record(
        "edit_and_test",
        index,
        f"Fix the empty-input regression in {module} and run its focused tests.",
        [
            _assistant(
                plan="Read the current implementation and capture its content hash.",
                action=_action("workspace.read", {"path": path}),
            ),
            _observation(
                "workspace.read",
                output={
                    "path": path,
                    "sha256": digest,
                    "content": "def handle(items):\n    return items[0]\n",
                },
            ),
            _assistant(
                plan="Apply the smallest concurrency-safe replacement.",
                action=_action(
                    "workspace.replace",
                    {
                        "path": path,
                        "old_text": "def handle(items):\n    return items[0]",
                        "new_text": (
                            "def handle(items):\n"
                            "    if not items:\n"
                            "        return None\n"
                            "    return items[0]"
                        ),
                        "expected_sha256": digest,
                        "expected_occurrences": 1,
                    },
                ),
            ),
            _observation(
                "workspace.replace",
                output={"path": path, "sha256": _sha(f"edit:{index}:new")},
            ),
            _assistant(
                plan="Run only the relevant test module first.",
                action=_action(
                    "shell.run",
                    {"argv": ["pytest", "-q", f"tests/test_{module}.py"], "cwd": "."},
                ),
            ),
            _observation("shell.run", output={"returncode": 0, "stdout": "6 passed"}),
            _assistant(
                result=f"Fixed empty-input handling in `{path}`; all 6 focused tests pass."
            ),
        ],
    )


def create_and_test(index: int) -> JsonObject:
    names = ["health", "version", "diagnostics", "metadata", "capabilities"]
    name = names[index % len(names)]
    path = f"src/{name}.py"
    return _record(
        "create_and_test",
        index,
        f"Add a small {name} module and verify its focused test.",
        [
            _assistant(
                plan="List the package before creating a new module.",
                action=_action("workspace.list", {"path": "src", "limit": 100}),
            ),
            _observation(
                "workspace.list", output={"entries": ["__init__.py", "core.py"]}
            ),
            _assistant(
                plan="Create only the requested module.",
                action=_action(
                    "workspace.write",
                    {
                        "path": path,
                        "content": f'def {name}():\n    return "ok"\n',
                        "expected_sha256": None,
                    },
                ),
            ),
            _observation(
                "workspace.write", output={"path": path, "sha256": _sha(path)}
            ),
            _assistant(
                plan="Verify the new behavior with its focused test.",
                action=_action(
                    "shell.run",
                    {"argv": ["pytest", "-q", f"tests/test_{name}.py"], "cwd": "."},
                ),
            ),
            _observation("shell.run", output={"returncode": 0, "stdout": "2 passed"}),
            _assistant(result=f"Added `{path}` and verified its 2 focused tests pass."),
        ],
    )


def git_to_change(index: int) -> JsonObject:
    issue = 100 + index
    head = _sha(f"git:{index}:head")
    committed = _sha(f"git:{index}:commit")
    branch = f"fix/issue-{issue}"
    providers = ["GitHub", "GitLab", "Gitea", "Forgejo"]
    provider = providers[index % len(providers)]
    return _record(
        "git_to_change",
        index,
        f"Prepare the verified issue {issue} fix and open it on {provider}.",
        [
            _assistant(
                plan="Bind branch creation to the current revision.",
                action=_action("git.revision", {}),
            ),
            _observation("git.revision", output={"head": head, "branch": "main"}),
            _assistant(
                plan="Create a focused branch from the observed HEAD.",
                action=_action(
                    "git.create_branch", {"name": branch, "expected_head": head}
                ),
            ),
            _observation("git.create_branch", output={"branch": branch, "head": head}),
            _assistant(
                plan="Review the exact working-tree patch before committing.",
                action=_action("git.diff", {"staged": False, "path": None}),
            ),
            _observation(
                "git.diff", output={"diff": f"diff --git a/src/issue_{issue}.py"}
            ),
            _assistant(
                plan="Commit only the requested source and test files.",
                action=_action(
                    "git.commit",
                    {
                        "message": f"Fix issue {issue}",
                        "paths": [f"src/issue_{issue}.py", f"tests/test_issue_{issue}.py"],
                        "expected_head": head,
                    },
                ),
            ),
            _observation(
                "git.commit", output={"head": committed, "branch": branch}
            ),
            _assistant(
                plan="Push the exact commit on the focused branch.",
                action=_action(
                    "git.push",
                    {"remote": "origin", "branch": branch, "expected_head": committed},
                ),
            ),
            _observation("git.push", output={"remote": "origin", "branch": branch}),
            _assistant(
                plan=f"Open the already-pushed branch as a {provider} change.",
                action=_action(
                    "provider.open_change",
                    {
                        "title": f"Fix issue {issue}",
                        "body": "Includes the minimal implementation and focused tests.",
                        "head_branch": branch,
                        "base_branch": "main",
                    },
                ),
            ),
            _observation(
                "provider.open_change",
                output={"number": issue, "head_sha": committed, "state": "open"},
            ),
            _assistant(
                result=f"Opened {provider} change {issue} for `{branch}` at `{committed}`."
            ),
        ],
    )


def review_approve(index: int) -> JsonObject:
    number = 20 + index
    sha = _sha(f"review:{index}:sha")
    provider = ["GitHub", "GitLab", "Gitea", "Forgejo"][index % 4]
    return _record(
        "review_approve",
        index,
        f"Review {provider} change {number} and approve only with current-revision evidence.",
        [
            _assistant(
                plan="Fetch the change and bind review to its current head.",
                action=_action("provider.get_change", {"number": number}),
            ),
            _observation(
                "provider.get_change",
                output={"number": number, "head_sha": sha, "state": "open"},
            ),
            _assistant(
                plan="Inspect the diff for that exact revision.",
                action=_action("provider.get_diff", {"number": number}),
            ),
            _observation(
                "provider.get_diff",
                output={"number": number, "head_sha": sha, "diff": "focused test fix"},
            ),
            _assistant(
                plan="Check CI for the same commit SHA.",
                action=_action("provider.get_checks", {"sha": sha}),
            ),
            _observation(
                "provider.get_checks",
                output={
                    "sha": sha,
                    "all_passed": True,
                    "checks": [{"name": "tests", "status": "success"}],
                },
            ),
            _assistant(
                plan="Approve the reviewed and passing revision only.",
                action=_action(
                    "provider.approve_change",
                    {
                        "number": number,
                        "expected_sha": sha,
                        "body": "Reviewed the exact diff; required checks pass.",
                    },
                ),
            ),
            _observation(
                "provider.approve_change", output={"approved": True, "head_sha": sha}
            ),
            _assistant(result=f"Approved change {number} at exact revision `{sha}`."),
        ],
    )


def review_block(index: int) -> JsonObject:
    number = 70 + index
    sha = _sha(f"blocked:{index}:sha")
    return _record(
        "review_block",
        index,
        f"Review change {number}; do not approve unless all checks pass.",
        [
            _assistant(
                plan="Fetch the current change revision.",
                action=_action("provider.get_change", {"number": number}),
            ),
            _observation(
                "provider.get_change", output={"number": number, "head_sha": sha}
            ),
            _assistant(
                plan="Inspect the exact diff before reaching a conclusion.",
                action=_action("provider.get_diff", {"number": number}),
            ),
            _observation(
                "provider.get_diff", output={"number": number, "head_sha": sha, "diff": "patch"}
            ),
            _assistant(
                plan="Read checks for the reviewed SHA.",
                action=_action("provider.get_checks", {"sha": sha}),
            ),
            _observation(
                "provider.get_checks",
                output={
                    "sha": sha,
                    "all_passed": False,
                    "checks": [{"name": "tests", "status": "failure"}],
                },
            ),
            _assistant(
                result=f"Did not approve change {number}: required tests fail on `{sha}`."
            ),
        ],
    )


def review_merge(index: int) -> JsonObject:
    number = 150 + index
    sha = _sha(f"merge:{index}:sha")
    method = ["squash", "merge", "rebase"][index % 3]
    return _record(
        "review_merge",
        index,
        f"Merge change {number} with {method} only after verifying its exact revision.",
        [
            _assistant(
                plan="Fetch and bind the current revision.",
                action=_action("provider.get_change", {"number": number}),
            ),
            _observation(
                "provider.get_change", output={"number": number, "head_sha": sha}
            ),
            _assistant(
                plan="Review the diff at the bound revision.",
                action=_action("provider.get_diff", {"number": number}),
            ),
            _observation(
                "provider.get_diff",
                output={"number": number, "head_sha": sha, "diff": "safe patch"},
            ),
            _assistant(
                plan="Verify required checks on the same SHA.",
                action=_action("provider.get_checks", {"sha": sha}),
            ),
            _observation(
                "provider.get_checks", output={"sha": sha, "all_passed": True, "checks": [
                    {"name": "tests", "status": "success"}
                ]}
            ),
            _assistant(
                plan="Merge only the reviewed, passing revision.",
                action=_action(
                    "provider.merge_change",
                    {"number": number, "expected_sha": sha, "method": method},
                ),
            ),
            _observation("provider.merge_change", output={"merged": True, "sha": sha}),
            _assistant(result=f"Merged change {number} at `{sha}` using {method}."),
        ],
    )


def test_recovery(index: int) -> JsonObject:
    module = ["parser", "router", "worker", "client"][index % 4]
    path = f"src/{module}.py"
    digest = _sha(f"recovery:{index}:old")
    return _record(
        "test_recovery",
        index,
        f"Diagnose and fix the failing {module} test without guessing.",
        [
            _assistant(
                plan="Reproduce the focused failure first.",
                action=_action(
                    "shell.run",
                    {"argv": ["pytest", "-q", f"tests/test_{module}.py"], "cwd": "."},
                ),
            ),
            _observation(
                "shell.run",
                error=f"command exited 1: test_{module}_empty expected None",
            ),
            _assistant(
                plan="Locate the failing implementation named in the test.",
                action=_action(
                    "workspace.search", {"query": "def handle", "path": "src"}
                ),
            ),
            _observation(
                "workspace.search", output={"matches": [{"path": path, "line": 12}]}
            ),
            _assistant(
                plan="Read the implementation before changing it.",
                action=_action("workspace.read", {"path": path}),
            ),
            _observation(
                "workspace.read",
                output={"path": path, "sha256": digest, "content": "return values[0]"},
            ),
            _assistant(
                plan="Guard the observed empty-input access.",
                action=_action(
                    "workspace.replace",
                    {
                        "path": path,
                        "old_text": "return values[0]",
                        "new_text": "return values[0] if values else None",
                        "expected_sha256": digest,
                        "expected_occurrences": 1,
                    },
                ),
            ),
            _observation(
                "workspace.replace", output={"path": path, "sha256": _sha(path + "new")}
            ),
            _assistant(
                plan="Rerun the previously failing test.",
                action=_action(
                    "shell.run",
                    {"argv": ["pytest", "-q", f"tests/test_{module}.py"], "cwd": "."},
                ),
            ),
            _observation("shell.run", output={"returncode": 0, "stdout": "5 passed"}),
            _assistant(result=f"Fixed `{path}` and verified all 5 focused tests pass."),
        ],
    )


def stale_write_recovery(index: int) -> JsonObject:
    path = f"src/config_{index % 7}.py"
    stale = _sha(f"stale:{index}")
    current = _sha(f"current:{index}")
    return _record(
        "stale_write_recovery",
        index,
        "Apply the requested config rename without overwriting concurrent changes.",
        [
            _assistant(
                plan="Read the current file and capture its hash.",
                action=_action("workspace.read", {"path": path}),
            ),
            _observation(
                "workspace.read", output={"path": path, "sha256": stale, "content": "MODE='old'"}
            ),
            _assistant(
                plan="Attempt the exact replacement against the observed hash.",
                action=_action(
                    "workspace.replace",
                    {
                        "path": path,
                        "old_text": "MODE='old'",
                        "new_text": "MODE='new'",
                        "expected_sha256": stale,
                        "expected_occurrences": 1,
                    },
                ),
            ),
            _observation(
                "workspace.replace",
                error="ValueError: file changed since it was read",
            ),
            _assistant(
                plan="Re-read instead of overwriting concurrent work.",
                action=_action("workspace.read", {"path": path}),
            ),
            _observation(
                "workspace.read",
                output={
                    "path": path,
                    "sha256": current,
                    "content": "MODE='old'\nTIMEOUT=30",
                },
            ),
            _assistant(
                plan="Preserve the concurrent addition and replace only the requested value.",
                action=_action(
                    "workspace.replace",
                    {
                        "path": path,
                        "old_text": "MODE='old'",
                        "new_text": "MODE='new'",
                        "expected_sha256": current,
                        "expected_occurrences": 1,
                    },
                ),
            ),
            _observation(
                "workspace.replace", output={"path": path, "sha256": _sha(path + "done")}
            ),
            _assistant(
                result=(
                    "Applied the requested rename while preserving the concurrent "
                    "timeout change."
                )
            ),
        ],
    )


def permission_recovery(index: int) -> JsonObject:
    commands = [["npm", "test"], ["cargo", "test"], ["go", "test", "./..."]]
    argv = commands[index % len(commands)]
    return _record(
        "permission_recovery",
        index,
        f"Run the project verification command: {' '.join(argv)}.",
        [
            _assistant(
                plan="Attempt the requested verification through the policy-checked shell tool.",
                action=_action("shell.run", {"argv": argv, "cwd": "."}),
            ),
            _observation(
                "shell.run",
                error=f"PolicyDenied: command is not allowlisted: {argv[0]}",
            ),
            _assistant(
                result=(
                    f"Could not run `{' '.join(argv)}` because runtime policy denied it. "
                    "No verification result is claimed; an operator must allowlist the command."
                )
            ),
        ],
    )


FAMILIES: dict[str, Callable[[int], JsonObject]] = {
    "inspect": inspect,
    "edit_and_test": edit_and_test,
    "create_and_test": create_and_test,
    "git_to_change": git_to_change,
    "review_approve": review_approve,
    "review_block": review_block,
    "review_merge": review_merge,
    "test_recovery": test_recovery,
    "stale_write_recovery": stale_write_recovery,
    "permission_recovery": permission_recovery,
}


def validate_record(record: JsonObject) -> None:
    messages = canonicalize_trajectory_messages(record["messages"])
    previous_action: str | None = None
    assistant_count = 0
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            decision = Decision.from_value(message["decision"])
            assistant_count += 1
            previous_action = decision.action.name if decision.action else None
            if previous_action is not None and previous_action not in KNOWN_TOOLS:
                raise ValueError(f"Unknown action in trajectory: {previous_action}")
        elif role == "tool":
            if message.get("name") != previous_action:
                raise ValueError("Tool observation does not match the preceding action")
            json.loads(str(message.get("content", "")))
            previous_action = None
        elif role not in {"system", "user"}:
            raise ValueError(f"Unsupported role in generated trajectory: {role!r}")
    if assistant_count < 1:
        raise ValueError("Generated trajectory has no assistant decisions")


def evaluation_case(record: JsonObject, selector: int) -> JsonObject:
    messages = record["messages"]
    assistant_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "assistant"
    ]
    target_index = assistant_indices[selector % len(assistant_indices)]
    expected = Decision.from_value(messages[target_index]["decision"])
    expected_value: JsonObject = {}
    if expected.plan is not None:
        expected_value["plan"] = expected.plan
    if expected.action is not None:
        expected_value["action"] = expected.action.to_dict()
    else:
        expected_value["result"] = expected.result
    return {
        "messages": messages[:target_index],
        "expected": expected_value,
        "metadata": record["metadata"],
    }


def training_case(record: JsonObject, selector: int) -> JsonObject:
    """Turn a trajectory into one prefix-to-next-decision supervision example."""
    messages = record["messages"]
    assistant_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "assistant"
    ]
    target_ordinal = selector % len(assistant_indices)
    target_index = assistant_indices[target_ordinal]
    metadata = dict(record["metadata"])
    metadata.update(
        {
            "supervision": "prefix_to_next_decision",
            "target_assistant_ordinal": target_ordinal,
            "trajectory_assistant_count": len(assistant_indices),
        }
    )
    return {
        "messages": messages[: target_index + 1],
        "metadata": metadata,
    }


def _write_jsonl(path: Path, rows: list[JsonObject]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_dataset(
    output_dir: Path,
    train_per_family: int = 40,
    validation_per_family: int = 4,
    eval_per_family: int = 4,
) -> JsonObject:
    output_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[JsonObject]] = {"train": [], "validation": [], "eval": []}
    for family_index, builder in enumerate(FAMILIES.values()):
        for index in range(train_per_family):
            record = builder(index)
            record["metadata"]["split"] = "train"
            validate_record(record)
            splits["train"].append(training_case(record, index + family_index))
        for index in range(validation_per_family):
            record = builder(10_000 + index)
            record["metadata"]["split"] = "validation"
            validate_record(record)
            splits["validation"].append(
                training_case(record, index + family_index)
            )
        for index in range(eval_per_family):
            record = builder(20_000 + index)
            record["metadata"]["split"] = "eval"
            validate_record(record)
            splits["eval"].append(evaluation_case(record, index + family_index))

    _write_jsonl(output_dir / "train.jsonl", splits["train"])
    _write_jsonl(output_dir / "validation.jsonl", splits["validation"])
    _write_jsonl(output_dir / "eval.jsonl", splits["eval"])
    manifest = {
        "name": "nimora-agent-alpha-v0.0.1",
        "license": "Apache-2.0",
        "provenance": "deterministic synthetic templates authored for Nimora",
        "supervision": "prefix_to_next_decision",
        "limitations": [
            "Template-generated data does not establish real-world coding quality.",
            "No private repositories, user logs, credentials, or scraped code are included.",
            "Provider names appear in tasks, but actions use provider-neutral runtime tools.",
        ],
        "families": sorted(FAMILIES),
        "counts": {name: len(rows) for name, rows in splits.items()},
        "split_index_ranges": {
            "train": [0, train_per_family - 1],
            "validation": [10_000, 10_000 + validation_per_family - 1],
            "eval": [20_000, 20_000 + eval_per_family - 1],
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-per-family", type=int, default=40)
    parser.add_argument("--validation-per-family", type=int, default=4)
    parser.add_argument("--eval-per-family", type=int, default=4)
    args = parser.parse_args()
    manifest = build_dataset(
        args.output_dir,
        args.train_per_family,
        args.validation_per_family,
        args.eval_per_family,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
