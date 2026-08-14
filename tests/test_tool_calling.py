"""Mocked, deterministic tests for agent_phase2's web-search tool loop.

A FakeClient stands in for the OpenAI client so we can script exactly what
"the model" does across a multi-step tool-calling exchange, without any
network call to OpenRouter or Tavily.

The PASS/FAIL pair matches the task goal ("a web search tool the agent can
call if it needs to answer something factual"):
  PASS: asked something it doesn't know, the agent calls web_search, gets
        results, and answers grounded in them.
  FAIL (guarded against): the search backend itself breaks (e.g. missing
        TAVILY_API_KEY) -- that must be reported back to the model as a
        tool result, not crash the whole turn.
"""

import json

import pytest

import agent_phase2 as ap2


class FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.type = "function"
        self.function = FakeFunction(name, arguments)


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, message):
        self.message = message


class FakeResponse:
    def __init__(self, message):
        self.choices = [FakeChoice(message)]


class FakeCompletions:
    def __init__(self, reply_fn):
        self.reply_fn = reply_fn
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.reply_fn(kwargs["messages"])
        message = FakeMessage(content=result.get("content", ""), tool_calls=result.get("tool_calls"))
        return FakeResponse(message)


class FakeChat:
    def __init__(self, reply_fn):
        self.completions = FakeCompletions(reply_fn)


class FakeClient:
    def __init__(self, reply_fn):
        self.chat = FakeChat(reply_fn)


def fake_search(query):
    return [
        {
            "title": "Capital of Australia",
            "url": "https://example.com/canberra",
            "content": "Canberra is the capital city of Australia.",
        }
    ]


@pytest.fixture(autouse=True)
def isolated_sessions_dir(tmp_path, monkeypatch):
    import agent_phase1 as ap1

    monkeypatch.setattr(ap1, "SESSIONS_DIR", tmp_path / "sessions")


# ---- tool spec sanity -----------------------------------------------


def test_web_search_tool_spec_is_well_formed():
    fn = ap2.WEB_SEARCH_TOOL["function"]
    assert fn["name"] == "web_search"
    assert fn["parameters"]["required"] == ["query"]


def test_format_search_results_handles_empty_list():
    assert ap2.format_search_results([]) == "No results found."


def test_format_search_results_includes_title_url_and_snippet():
    formatted = ap2.format_search_results(fake_search("x"))
    assert "Capital of Australia" in formatted
    assert "https://example.com/canberra" in formatted
    assert "Canberra is the capital" in formatted


# ---- PASS/FAIL behavioral pair (mocked model) ------------------------


def searching_reply(messages):
    """A canned 'model' that always searches once, then answers from the tool result."""
    tool_messages = [m for m in messages if m["role"] == "tool"]
    if not tool_messages:
        call = FakeToolCall("call-1", "web_search", json.dumps({"query": "capital of Australia"}))
        return {"content": "", "tool_calls": [call]}
    return {"content": f"According to my search: {tool_messages[0]['content'].splitlines()[0]}"}


def test_pass_case_agent_calls_search_tool_and_answers_from_results():
    """PASS: a factual question the model can't answer on its own gets
    routed through web_search, and the final answer is grounded in the
    returned results."""
    client = FakeClient(searching_reply)
    messages = [{"role": "user", "content": "What is the capital of Australia?"}]

    reply = ap2.run_agentic_turn(client, "fake-model", ap2.DEFAULT_SYSTEM_PROMPT, messages, fake_search)

    assert "canberra" in reply.lower()
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert messages[1]["tool_calls"][0]["function"]["name"] == "web_search"
    assert messages[2]["tool_call_id"] == messages[1]["tool_calls"][0]["id"]


def test_no_search_needed_for_general_knowledge():
    """The model isn't forced to search -- if it answers directly with no
    tool_calls, the loop must not invent a search that wasn't requested."""

    def direct_reply(_messages):
        return {"content": "4", "tool_calls": None}

    client = FakeClient(direct_reply)
    messages = [{"role": "user", "content": "What is 2 + 2?"}]

    reply = ap2.run_agentic_turn(client, "fake-model", ap2.DEFAULT_SYSTEM_PROMPT, messages, fake_search)

    assert reply == "4"
    assert not any(m["role"] == "tool" for m in messages)
    assert len(client.chat.completions.calls) == 1


def test_fail_case_search_backend_failure_is_reported_not_crashed():
    """FAIL (the anti-pattern to guard against): the search backend itself
    breaks (e.g. missing TAVILY_API_KEY / network error). That must surface
    as a tool result the model can react to, not an unhandled crash."""

    def failing_search(_query):
        raise RuntimeError("TAVILY_API_KEY is not set.")

    def reply_after_failed_search(messages):
        if not any(m["role"] == "tool" for m in messages):
            call = FakeToolCall("call-1", "web_search", json.dumps({"query": "today's news"}))
            return {"content": "", "tool_calls": [call]}
        return {"content": "I couldn't search the web right now, so I can't confirm that."}

    client = FakeClient(reply_after_failed_search)
    messages = [{"role": "user", "content": "What happened in the news today?"}]

    reply = ap2.run_agentic_turn(client, "fake-model", ap2.DEFAULT_SYSTEM_PROMPT, messages, failing_search)

    assert "couldn't search" in reply.lower()
    tool_message = next(m for m in messages if m["role"] == "tool")
    assert "search failed" in tool_message["content"].lower()


def test_unknown_tool_call_is_reported_gracefully():
    def hallucinated_tool_reply(messages):
        if not any(m["role"] == "tool" for m in messages):
            call = FakeToolCall("call-1", "get_weather", "{}")
            return {"content": "", "tool_calls": [call]}
        return {"content": "Sorry, I can't check the weather."}

    client = FakeClient(hallucinated_tool_reply)
    messages = [{"role": "user", "content": "What's the weather?"}]

    ap2.run_agentic_turn(client, "fake-model", ap2.DEFAULT_SYSTEM_PROMPT, messages, fake_search)

    tool_message = next(m for m in messages if m["role"] == "tool")
    assert "unknown tool" in tool_message["content"].lower()


def test_gives_up_after_max_tool_iterations_instead_of_looping_forever():
    counter = {"n": 0}

    def always_wants_to_search(_messages):
        counter["n"] += 1
        call = FakeToolCall(f"call-{counter['n']}", "web_search", json.dumps({"query": "x"}))
        return {"content": "", "tool_calls": [call]}

    client = FakeClient(always_wants_to_search)
    messages = [{"role": "user", "content": "loop forever?"}]

    reply = ap2.run_agentic_turn(client, "fake-model", ap2.DEFAULT_SYSTEM_PROMPT, messages, fake_search)

    assert "couldn't find" in reply.lower()
    assert len(client.chat.completions.calls) == ap2.MAX_TOOL_ITERATIONS


# ---- end-to-end through run_chat, and persistence of tool messages ----


def test_run_chat_persists_full_tool_exchange_to_disk(monkeypatch):
    client = FakeClient(searching_reply)
    inputs = iter(["What is the capital of Australia?", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    ap2.run_chat(client, "fake-model", ap2.DEFAULT_SYSTEM_PROMPT, "search-session", fake_search)

    saved = ap2.load_session("search-session", ap2.DEFAULT_SYSTEM_PROMPT, "fake-model")
    roles = [m["role"] for m in saved["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert "canberra" in saved["messages"][-1]["content"].lower()


def test_run_chat_rolls_back_entire_failed_turn_including_tool_calls(monkeypatch):
    """If the model crashes mid tool-loop, none of that turn (user message,
    partial tool exchange) should be left dangling in persisted history."""
    call_count = {"n": 0}

    def first_turn_ok_second_turn_crashes(messages):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"content": "Nice to meet you, Athena."}
        if not any(m["role"] == "tool" for m in messages):
            call = FakeToolCall("call-1", "web_search", json.dumps({"query": "x"}))
            return {"content": "", "tool_calls": [call]}
        raise RuntimeError("500 Internal Server Error (simulated)")

    client = FakeClient(first_turn_ok_second_turn_crashes)
    inputs = iter(["My name is Athena.", "some factual question", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    ap2.run_chat(client, "fake-model", ap2.DEFAULT_SYSTEM_PROMPT, "crash-session", fake_search)

    saved = ap2.load_session("crash-session", ap2.DEFAULT_SYSTEM_PROMPT, "fake-model")
    assert saved["messages"][0] == {"role": "user", "content": "My name is Athena."}
    assert saved["messages"][1]["content"] == "Nice to meet you, Athena."
    assert len(saved["messages"]) == 2  # the crashed second turn left no trace
