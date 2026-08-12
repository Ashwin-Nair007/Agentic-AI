"""Live integration tests against the real OpenRouter API.

These are opt-in: they cost tokens and are non-deterministic (real model
output), so they're marked `integration` and skipped unless OPENROUTER_API_KEY
is set. Run explicitly with:

    uv run pytest -m integration

Deselect them from a normal run with:

    uv run pytest -m "not integration"

They exercise the same PASS/FAIL pair as tests/test_state_management.py, but
against the real model instead of a fake one, so a state-management bug that
only shows up against real model behavior still gets caught.
"""

import os

import pytest

import agent_phase1 as ap

from dotenv import load_dotenv

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set; skipping live OpenRouter calls",
    ),
]


def test_pass_case_live_model_recalls_stated_name():
    """PASS: with 'My name is Athena.' earlier in the transcript, the real
    model answers 'What is my name?' by naming Athena."""
    client = ap.make_client()
    messages = [{"role": "user", "content": "My name is Athena. Just acknowledge that briefly."}]
    ack = ap.stream_reply(client, ap.DEFAULT_MODEL, ap.DEFAULT_SYSTEM_PROMPT, messages)
    messages.append({"role": "assistant", "content": ack})
    messages.append({"role": "user", "content": "What is my name?"})

    reply = ap.stream_reply(client, ap.DEFAULT_MODEL, ap.DEFAULT_SYSTEM_PROMPT, messages)

    assert "athena" in reply.lower()


def test_fail_case_live_model_cannot_recall_name_without_history():
    """FAIL (expected): asking 'What is my name?' as the only message, with
    no prior turn stating a name, gives the real model nothing to recall --
    it must not fabricate "Athena" out of nowhere."""
    client = ap.make_client()

    reply = ap.stream_reply(
        client,
        ap.DEFAULT_MODEL,
        ap.DEFAULT_SYSTEM_PROMPT,
        [{"role": "user", "content": "What is my name?"}],
    )

    assert "athena" not in reply.lower()
