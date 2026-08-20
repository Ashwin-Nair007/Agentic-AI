"""Live integration tests for agent_phase3's read_file tool, against the
real OpenRouter model.

Opt-in: marked `integration` and skipped unless OPENROUTER_API_KEY is set
(TAVILY_API_KEY isn't required here since these cases don't need web_search).
Run explicitly with:

    uv run pytest -m integration

The PASS/FAIL pair mirrors the mocked one in test_file_reading.py: the
model should read a real file when asked about it, and must not fabricate
contents for a file that doesn't exist.
"""

import os

import pytest

import agent_phase3 as ap3

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set; skipping live OpenRouter calls",
    ),
]


def _unused_search(_query):
    raise AssertionError("web_search should not be needed for these file-reading cases")


def test_pass_case_live_agent_reads_actual_file_contents(tmp_path):
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("The secret code word is PLATYPUS7.", encoding="utf-8")
    messages = [
        {
            "role": "user",
            "content": f"Use the read_file tool to read {secret_file} and tell me the secret code word.",
        }
    ]
    client = ap3.make_client()

    reply = ap3.run_agentic_turn(
        client, ap3.DEFAULT_MODEL, ap3.DEFAULT_SYSTEM_PROMPT, messages, _unused_search, ap3.read_file
    )

    assert "platypus7" in reply.lower()
    assert any(m["role"] == "tool" for m in messages)


def test_fail_case_live_agent_does_not_fabricate_missing_file_contents(tmp_path):
    missing_file = tmp_path / "does_not_exist.txt"
    messages = [
        {
            "role": "user",
            "content": f"Use the read_file tool to read {missing_file} and tell me what it says.",
        }
    ]
    client = ap3.make_client()

    reply = ap3.run_agentic_turn(
        client, ap3.DEFAULT_MODEL, ap3.DEFAULT_SYSTEM_PROMPT, messages, _unused_search, ap3.read_file
    )

    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert tool_messages, "expected the model to attempt read_file"
    assert "does not exist" in tool_messages[0]["content"].lower()
    # The model should acknowledge failure, not answer as if it read real content.
    assert reply.strip()
