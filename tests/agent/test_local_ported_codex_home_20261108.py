"""Regression test: HERMES_CODEX_APP_SERVER_HOME is passed to CodexAppServerSession."""
import os
from unittest.mock import MagicMock, patch
import pytest

def test_codex_app_server_home_env_passed_to_session(monkeypatch, tmp_path):
    from agent import codex_runtime

    custom_home = str(tmp_path / "custom_codex_home")
    monkeypatch.setenv("HERMES_CODEX_APP_SERVER_HOME", custom_home)

    captured_kwargs = {}

    class FakeSession:
        def __init__(self, *args, **kwargs):
            captured_kwargs.update(kwargs)
        def run_turn(self, *args, **kwargs):
            return MagicMock(final_text="ok", projected_messages=[], turn_id="t1", thread_id="th1")
        def close(self):
            pass

    with patch("agent.transports.codex_app_server_session.CodexAppServerSession", FakeSession):
        agent = MagicMock()
        agent._codex_session = None
        agent.model = "gpt-6-astra"
        agent.provider = "openai-codex"

        # Trigger session creation
        codex_runtime._ensure_codex_session(agent, messages=[])

    assert captured_kwargs.get("codex_home") == custom_home, f"Expected codex_home={custom_home}, got {captured_kwargs.get('codex_home')}"
