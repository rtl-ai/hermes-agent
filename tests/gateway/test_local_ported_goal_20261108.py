"""Regression guard: partial or interrupted Hermes turns are not goal completions."""

import pytest

from gateway.run_goals import _nonterminal_turn_for_goal


@pytest.mark.parametrize("result", [
    {"final_response": "partial", "partial": True},
    {"final_response": "oops", "failed": True},
    {"final_response": "interrupted", "interrupted": True},
    {"final_response": "error", "error": "provider failed"},
    {"final_response": "cut off", "completed": False},
])
def test_nonterminal_turns_must_not_continue_goal(result):
    assert _nonterminal_turn_for_goal(result) is True


@pytest.mark.parametrize("result", [
    {"final_response": "done", "completed": True},
    {"final_response": "done"},
    "legacy completed answer",
])
def test_completed_turn_can_drive_goal(result):
    assert _nonterminal_turn_for_goal(result) is False
