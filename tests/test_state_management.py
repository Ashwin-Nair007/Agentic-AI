"""Mocked, deterministic tests for agent_phase1's statefulness.

These never hit the network: a FakeClient stands in for the OpenAI client so
we can assert on exactly what gets sent to / read back from "the model" and
on how conversation state is built, persisted, and reloaded.

The PASS/FAIL pair below is the concrete example from the task:
  PASS: [user: what is my name?] -> [agent: your name is "Athena"]
        because the earlier "My name is Athena." turn is present in history.
  FAIL: the same question with that turn missing from history -> the agent
        has no way to know the name and says so.
"""

import pytest

import agent_phase1 as ap


class FakeDelta:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.delta = FakeDelta(content)


class FakeChunk:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, reply_fn):
        self.reply_fn = reply_fn
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.reply_fn(kwargs["messages"])
        return [FakeChunk(reply)]


class FakeChat:
    def __init__(self, reply_fn):
        self.completions = FakeCompletions(reply_fn)


class FakeClient:
    def __init__(self, reply_fn):
        self.chat = FakeChat(reply_fn)


def stateful_reply(messages):
    """A canned 'model' that can only name the user if a prior turn told it."""
    for m in messages:
        if m["role"] == "user" and "my name is" in m["content"].lower():
            tail = m["content"].lower().split("my name is", 1)[1].strip()
            name = tail.rstrip(".!").split()[0]
            return f'Your name is "{name.capitalize()}".'
    return "I am an AI assistant, I don't have a name for you."


@pytest.fixture(autouse=True)
def isolated_sessions_dir(tmp_path, monkeypatch):
    """Redirect session files into a throwaway dir so tests never touch
    .chat_sessions/ in the real project and never interfere with each other."""
    monkeypatch.setattr(ap, "SESSIONS_DIR", tmp_path / "sessions")


# ---- session persistence ----------------------------------------------


def test_save_and_load_session_round_trip():
    data = {"model": "m", "system": "s", "messages": [{"role": "user", "content": "hi"}]}
    ap.save_session("alpha", data)

    loaded = ap.load_session("alpha", default_system="x", default_model="y")

    assert loaded == data


def test_load_session_missing_file_returns_defaults():
    loaded = ap.load_session("does-not-exist", default_system="sys", default_model="mod")

    assert loaded == {"model": "mod", "system": "sys", "messages": []}


def test_named_sessions_are_isolated():
    ap.save_session("alpha", {"model": "m", "system": "s", "messages": [{"role": "user", "content": "alpha msg"}]})
    ap.save_session("beta", {"model": "m", "system": "s", "messages": [{"role": "user", "content": "beta msg"}]})

    alpha = ap.load_session("alpha", "s", "m")
    beta = ap.load_session("beta", "s", "m")

    assert alpha["messages"][0]["content"] == "alpha msg"
    assert beta["messages"][0]["content"] == "beta msg"


def test_clear_session_removes_file():
    ap.save_session("gamma", {"model": "m", "system": "s", "messages": [{"role": "user", "content": "x"}]})
    assert ap.session_path("gamma").exists()

    ap.clear_session("gamma")

    assert not ap.session_path("gamma").exists()


def test_clear_session_missing_file_is_a_noop():
    ap.clear_session("never-existed")  # must not raise


# ---- request construction ----------------------------------------------


def test_build_request_messages_prepends_system_prompt():
    messages = [{"role": "user", "content": "hi"}]

    result = ap.build_request_messages("be nice", messages)

    assert result == [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}]


# ---- pass/fail behavioral cases (mocked model) --------------------------


def test_pass_case_agent_recalls_name_when_history_is_preserved():
    """PASS: [user: My name is Athena.] then [user: what is my name?]
    -> [agent: your name is "Athena"], because history carries the name."""
    client = FakeClient(stateful_reply)
    messages = [{"role": "user", "content": "My name is Athena."}]
    ack = ap.stream_reply(client, "fake-model", ap.DEFAULT_SYSTEM_PROMPT, messages)
    messages.append({"role": "assistant", "content": ack})

    messages.append({"role": "user", "content": "What is my name?"})
    reply = ap.stream_reply(client, "fake-model", ap.DEFAULT_SYSTEM_PROMPT, messages)

    assert "athena" in reply.lower()


def test_fail_case_agent_cannot_recall_name_when_history_is_dropped():
    """FAIL (the anti-pattern from the task): [user: what is my name?] with
    no prior "My name is ..." turn in the transcript -> the agent can't know
    it and falls back to a generic "I don't have a name for you" answer."""
    client = FakeClient(stateful_reply)
    broken_messages = [{"role": "user", "content": "What is my name?"}]  # history dropped

    reply = ap.stream_reply(client, "fake-model", ap.DEFAULT_SYSTEM_PROMPT, broken_messages)

    assert "don't have a name" in reply.lower()
    assert "athena" not in reply.lower()


# ---- end-to-end through run_chat (persistence makes the PASS case real) --


def test_run_chat_persists_full_turn_so_name_is_recallable_after_reload(monkeypatch):
    client = FakeClient(stateful_reply)
    inputs = iter(["My name is Athena.", "What is my name?", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    ap.run_chat(client, "fake-model", ap.DEFAULT_SYSTEM_PROMPT, "session-x")

    saved = ap.load_session("session-x", ap.DEFAULT_SYSTEM_PROMPT, "fake-model")
    assert saved["messages"][0] == {"role": "user", "content": "My name is Athena."}
    assert "athena" in saved["messages"][1]["content"].lower()
    assert saved["messages"][2] == {"role": "user", "content": "What is my name?"}
    assert "athena" in saved["messages"][3]["content"].lower()


def test_reset_command_clears_persisted_session(monkeypatch):
    ap.save_session("to-reset", {"model": "m", "system": "s", "messages": [{"role": "user", "content": "hi"}]})
    client = FakeClient(stateful_reply)
    inputs = iter(["/reset", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    ap.run_chat(client, "fake-model", ap.DEFAULT_SYSTEM_PROMPT, "to-reset")

    assert not ap.session_path("to-reset").exists()


def test_run_chat_survives_unexpected_provider_crash_without_losing_history(monkeypatch):
    """Regression test: a provider-side crash (e.g. OpenRouter/NVIDIA's free
    model returning an internal server error mid-stream) can surface as a raw,
    un-typed exception rather than one of the openai.* error classes. That
    must not kill the REPL or wipe out already-persisted history -- the failed
    turn should just roll back and the loop should keep going."""
    call_count = {"n": 0}

    def flaky_reply(messages):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("500 Internal Server Error (simulated provider crash)")
        return stateful_reply(messages)

    client = FakeClient(flaky_reply)
    inputs = iter(["My name is Athena.", "What is my name?", "What is my name?", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    ap.run_chat(client, "fake-model", ap.DEFAULT_SYSTEM_PROMPT, "crash-session")  # must not raise

    saved = ap.load_session("crash-session", ap.DEFAULT_SYSTEM_PROMPT, "fake-model")
    # Turn 1 persisted; the crashed 2nd turn left no trace; the retried 3rd
    # turn (also asking "What is my name?") succeeded and is persisted next.
    assert saved["messages"][0] == {"role": "user", "content": "My name is Athena."}
    assert "athena" in saved["messages"][1]["content"].lower()
    assert saved["messages"][2] == {"role": "user", "content": "What is my name?"}
    assert "athena" in saved["messages"][3]["content"].lower()
    assert len(saved["messages"]) == 4


def test_ctrl_c_during_streaming_exits_immediately(monkeypatch):
    """Regression test: pressing Ctrl+C while a reply is streaming (or the
    call is hung) must quit the program on the spot, like at the "You:"
    prompt -- not swallow the interrupt and loop back for more input. Only
    one input is queued; if run_chat looped back instead of returning, it
    would call input() again and raise StopIteration here."""

    def raising_reply(_messages):
        raise KeyboardInterrupt

    client = FakeClient(raising_reply)
    inputs = iter(["My name is Athena."])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    ap.run_chat(client, "fake-model", ap.DEFAULT_SYSTEM_PROMPT, "ctrlc-session")  # must return, not loop

    saved = ap.load_session("ctrlc-session", ap.DEFAULT_SYSTEM_PROMPT, "fake-model")
    assert saved["messages"] == []  # the interrupted turn was rolled back, nothing persisted


def test_resuming_a_session_reissues_prior_messages_to_the_model(monkeypatch):
    """A fresh process loading a saved session must still be able to answer
    the name question without the user restating it in the new run."""
    ap.save_session(
        "resume-me",
        {
            "model": "fake-model",
            "system": ap.DEFAULT_SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": "My name is Athena."},
                {"role": "assistant", "content": 'Your name is "Athena".'},
            ],
        },
    )
    client = FakeClient(stateful_reply)
    inputs = iter(["What is my name?", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    ap.run_chat(client, "fake-model", ap.DEFAULT_SYSTEM_PROMPT, "resume-me")

    sent_messages = client.chat.completions.calls[0]["messages"]
    assert any("my name is athena" in m["content"].lower() for m in sent_messages if m["role"] == "user")
