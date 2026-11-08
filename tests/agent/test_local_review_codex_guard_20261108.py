"""Restore local Codex app-server background-review isolation without auth calls."""

from types import SimpleNamespace

from agent import background_review


def _fake_agent():
    return SimpleNamespace(
        provider="openai-codex", model="gpt-6-astra", api_mode="codex_app_server",
        api_key=None, base_url=None, max_tokens=1000, request_overrides={},
        _credential_pool=None, acp_command=None, acp_args=[],
        _current_main_runtime=lambda: {"api_mode": "codex_app_server"},
    )


def test_default_codex_app_server_background_review_skips_write_incapable_fork(monkeypatch):
    monkeypatch.setattr(background_review, "_background_review_task_config", lambda *_: {})
    runtime = background_review._resolve_review_runtime(_fake_agent())
    assert runtime["api_mode"] == "codex_responses"
    assert runtime["routed"] is False
    assert "skip_reason" in runtime


def test_explicit_other_provider_still_allowed_to_run(monkeypatch):
    from hermes_cli import runtime_provider
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", lambda **kwargs: {
        "provider": "custom", "model": "safe-local-guardian",
        "api_mode": "chat_completions", "base_url": "http://127.0.0.1:3000/v1",
        "api_key": "not-a-secret",
    })
    runtime = background_review._resolve_review_runtime(
        _fake_agent(), {"provider": "custom:local", "model": "safe-local-guardian"})
    assert runtime["routed"] is True
    assert "skip_reason" not in runtime
