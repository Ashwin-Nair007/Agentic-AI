"""Live integration tests for agent_phase2's web-search tool, against the
real OpenRouter model and the real Tavily API.

Opt-in: cost tokens/search quota and are non-deterministic, so they're
marked `integration` and skipped unless both OPENROUTER_API_KEY and
TAVILY_API_KEY are set. Run explicitly with:

    uv run pytest -m integration

The PASS/FAIL pair mirrors the mocked one in test_tool_calling.py, but
checks real model judgment: it should search for something it can't know
on its own, and should NOT waste a search call on something it already
knows.
"""

import os

import pytest

import agent_phase2 as ap2

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (os.environ.get("OPENROUTER_API_KEY") and os.environ.get("TAVILY_API_KEY")),
        reason="OPENROUTER_API_KEY and TAVILY_API_KEY are both required for live web-search tests",
    ),
]


def test_pass_case_live_agent_searches_for_current_information():
    client = ap2.make_client()
    messages = [
        {
            "role": "user",
            "content": "Use the web_search tool to check: what is the current version of Python listed on python.org?",
        }
    ]

    reply = ap2.run_agentic_turn(client, ap2.DEFAULT_MODEL, ap2.DEFAULT_SYSTEM_PROMPT, messages, ap2.tavily_search)

    assert reply.strip()
    assert any(m["role"] == "tool" for m in messages)


def test_fail_case_live_agent_skips_search_for_arithmetic():
    client = ap2.make_client()
    messages = [{"role": "user", "content": "What is 2 + 2? Just answer the number."}]

    reply = ap2.run_agentic_turn(client, ap2.DEFAULT_MODEL, ap2.DEFAULT_SYSTEM_PROMPT, messages, ap2.tavily_search)

    assert "4" in reply
    assert not any(m["role"] == "tool" for m in messages)
