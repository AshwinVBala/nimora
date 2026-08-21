from __future__ import annotations

import argparse
import json
from pathlib import Path

from nimora.agent.backend import OpenAICompatibleBackend
from nimora.agent.evals import EvaluationHarness, load_eval_cases
from nimora.agent.git_workspace import GitWorkspace
from nimora.agent.loop import AgentRuntime
from nimora.agent.policy import RuntimePolicy
from nimora.agent.providers import load_provider, register_provider_tools
from nimora.agent.recording import TrajectoryRecorder
from nimora.agent.tools import build_local_tools
from nimora.agent.workspace import WorkspaceBoundary


def add_agent_parsers(subparsers) -> None:
    agent = subparsers.add_parser("agent-run", help="Run the Nimora coding-agent loop")
    agent.add_argument("--workspace", required=True)
    agent.add_argument("--policy", required=True)
    agent.add_argument("--endpoint", required=True)
    agent.add_argument("--model", required=True)
    agent.add_argument("--api-key-env")
    task = agent.add_mutually_exclusive_group(required=True)
    task.add_argument("--task")
    task.add_argument("--task-file")
    agent.add_argument("--provider-config")
    agent.add_argument("--record")
    agent.set_defaults(handler=run_agent)

    evaluate = subparsers.add_parser("eval-run", help="Run isolated coding evaluations")
    evaluate.add_argument("--cases", required=True)
    evaluate.add_argument("--policy", required=True)
    evaluate.add_argument("--endpoint", required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--api-key-env")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--record")
    evaluate.set_defaults(handler=run_evaluations)


def run_agent(args: argparse.Namespace) -> None:
    policy = RuntimePolicy.load(args.policy)
    workspace = WorkspaceBoundary(args.workspace, policy.max_file_bytes)
    try:
        git = GitWorkspace(workspace, policy.command_timeout_seconds)
    except ValueError:
        git = None
    tools = build_local_tools(workspace, policy, git)
    if args.provider_config:
        register_provider_tools(tools, load_provider(args.provider_config))
    backend = OpenAICompatibleBackend.from_environment(
        args.endpoint,
        args.model,
        args.api_key_env,
        timeout_seconds=policy.command_timeout_seconds,
    )
    task = args.task
    if args.task_file:
        task = Path(args.task_file).read_text(encoding="utf-8")
    runtime = AgentRuntime(backend, tools, TrajectoryRecorder(args.record))
    result = runtime.run(str(task))
    print(
        json.dumps(
            {"status": result.status, "result": result.result, "steps": result.steps},
            indent=2,
        )
    )
    if result.status != "completed":
        raise SystemExit(2)


def run_evaluations(args: argparse.Namespace) -> None:
    policy = RuntimePolicy.load(args.policy)

    def backend_factory(_case):
        return OpenAICompatibleBackend.from_environment(
            args.endpoint,
            args.model,
            args.api_key_env,
            timeout_seconds=policy.command_timeout_seconds,
        )

    harness = EvaluationHarness(
        backend_factory,
        policy,
        args.output,
        trajectory_output=args.record,
    )
    summary = harness.run(load_eval_cases(args.cases))
    print(json.dumps(summary, indent=2))
