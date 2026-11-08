"""Regression: local 8b85b9eb portion ported to the new Codex projector."""
from agent.transports.codex_event_projector import CodexEventProjector


def _project(text, phase=None):
    item = {"type": "agentMessage", "text": text}
    if phase is not None:
        item["phase"] = phase
    return CodexEventProjector().project({"method": "item/completed", "params": {"item": item}})


def test_codex_commentary_is_history_but_not_final_text():
    result = _project("working", "commentary")
    assert result.messages[0]["content"] == "working"
    assert result.final_text is None


def test_codex_terminal_and_legacy_phases():
    assert _project("complete", "final_answer").final_text == "complete"
    assert _project("complete", "final").final_text == "complete"
    assert _project("legacy").final_text == "legacy"
    assert _project("unknown", "analysis").final_text is None
