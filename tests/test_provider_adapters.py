from nimora.agent.providers import ForgejoProvider, GiteaProvider, load_provider


class FakeHttp:
    def __init__(self):
        self.calls = []

    def request(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        if method == "GET" and path.endswith("/pulls/7"):
            return {
                "number": 7,
                "html_url": "https://forge.example/owner/repository/pulls/7",
                "state": "open",
                "title": "Fix parser",
                "head": {"sha": "a" * 40},
                "base": {"ref": "main"},
            }
        if path.endswith("/reviews"):
            return {"id": 42}
        if path.endswith("/merge"):
            return {"merged": True}
        raise AssertionError(f"unexpected request: {method} {path}")


def test_forgejo_accepts_instance_or_api_base_url():
    instance = ForgejoProvider("owner/repository", "secret", "https://forge.example/")
    api = ForgejoProvider(
        "owner/repository", "secret", "https://forge.example/api/v1/"
    )
    assert instance.http.base_url == "https://forge.example/api/v1"
    assert api.http.base_url == "https://forge.example/api/v1"


def test_load_provider_selects_first_class_forgejo_adapter(tmp_path, monkeypatch):
    config = tmp_path / "provider.yaml"
    config.write_text(
        "provider: forgejo\n"
        "repository: owner/repository\n"
        "base_url: https://forge.example\n"
        "token_env: TEST_FORGEJO_TOKEN\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_FORGEJO_TOKEN", "secret")
    provider = load_provider(str(config))
    assert type(provider) is ForgejoProvider
    assert provider.http.base_url == "https://forge.example/api/v1"


def test_forgejo_approval_and_merge_are_revision_bound():
    provider = ForgejoProvider("owner/repository", "secret", "https://forge.example")
    fake = FakeHttp()
    provider.http = fake

    approved = provider.approve_change(7, "a" * 40, "Reviewed exact revision")
    merged = provider.merge_change(7, "a" * 40, "squash")

    assert approved["approved"] is True
    assert merged["merged"] is True
    assert fake.calls[1] == (
        "POST",
        "/repos/owner/repository/pulls/7/reviews",
        {
            "body": "Reviewed exact revision",
            "event": "APPROVED",
            "commit_id": "a" * 40,
        },
    )
    assert fake.calls[3] == (
        "POST",
        "/repos/owner/repository/pulls/7/merge",
        {"Do": "squash", "head_commit_id": "a" * 40},
    )


def test_gitea_uses_approved_review_event_too():
    provider = GiteaProvider("owner/repository", "secret", "https://git.example")
    fake = FakeHttp()
    provider.http = fake
    provider.approve_change(7, "a" * 40, "Reviewed")
    assert fake.calls[1][2]["event"] == "APPROVED"
