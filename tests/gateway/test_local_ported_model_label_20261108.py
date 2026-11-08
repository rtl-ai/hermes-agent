"""Test model label suffix formatting in gateway responses."""
from unittest.mock import MagicMock
import pytest

def test_response_appends_active_model_label():
    from gateway.run_turn_persistence import _format_response_with_model_label

    agent_result = {"model": "qwen38-27b-uncensored-q3kp"}
    raw_response = "Here is the report."
    formatted = _format_response_with_model_label(raw_response, agent_result)
    assert formatted == "Here is the report.\n\n_[Model: qwen38-27b-uncensored-q3kp]_"

    # Idempotent if already has label
    idempotent = _format_response_with_model_label(formatted, agent_result)
    assert idempotent == formatted
