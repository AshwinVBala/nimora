from dataclasses import asdict

from nimora.agent.policy import RuntimePolicy
from nimora.agent.providers import RemoteChange, register_provider_tools
from nimora.agent.tools import ToolRegistry
from nimora.agent.types import Action


class FakeProvider:
    def __init__(self):
        self.change = RemoteChange(7, "https://example.test/7", "open", "Fix", "a" * 40, "main")
        self.approved = False

    def get_change(self, number):
        assert number == 7
        return self.change

    def get_diff(self, number):
        return {"number": number, "diff": "diff --git a/a b/a"}

    def get_checks(self, sha):
        return {"sha": sha, "checks": [{"name": "test", "status": "success"}]}

    def open_change(self, title, body, head_branch, base_branch):
        del title, body, head_branch, base_branch
        return self.change

    def approve_change(self, number, expected_sha, body):
        del body
        assert number == 7 and expected_sha == self.change.head_sha
        self.approved = True
        return {"approved": True, "change": asdict(self.change)}

    def merge_change(self, number, expected_sha, method):
        return {"merged": True, "number": number, "sha": expected_sha, "method": method}


def action(name, **arguments):
    return Action(name, arguments)


def test_approval_requires_diff_and_check_evidence_for_same_sha():
    provider = FakeProvider()
    policy = RuntimePolicy(allow_remote_read=True, allow_approve_change=True)
    registry = ToolRegistry(policy)
    register_provider_tools(registry, provider)
    denied = registry.execute(
        action(
            "provider.approve_change",
            number=7,
            expected_sha="a" * 40,
            body="Reviewed",
        )
    )
    assert not denied.ok
    assert not provider.approved

    assert registry.execute(action("provider.get_change", number=7)).ok
    assert registry.execute(action("provider.get_diff", number=7)).ok
    assert registry.execute(action("provider.get_checks", sha="a" * 40)).ok
    approved = registry.execute(
        action(
            "provider.approve_change",
            number=7,
            expected_sha="a" * 40,
            body="Reviewed exact revision",
        )
    )
    assert approved.ok
    assert provider.approved


def test_action_schema_rejects_unknown_fields():
    provider = FakeProvider()
    registry = ToolRegistry(RuntimePolicy(allow_remote_read=True))
    register_provider_tools(registry, provider)
    result = registry.execute(action("provider.get_change", number=7, surprise=True))
    assert not result.ok
    assert "unknown fields" in result.error
