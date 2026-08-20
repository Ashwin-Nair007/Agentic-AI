"""Mocked, deterministic tests for agent_phase3's read_file tool and the
combined web_search + read_file tool loop.

A FakeClient scripts what "the model" does; read_file itself is exercised
against real temporary files on disk (there's no external service to fake
for local filesystem access, unlike Tavily in Phase 2).

The PASS/FAIL pair matches the task goal ("a read-only file system
interface... that allows the LLM to look at the contents of actual files"):
  PASS: asked about a specific file, the agent calls read_file, gets its
        real contents, and answers grounded in them.
  FAIL (guarded against): the file doesn't exist / isn't readable -- that
        must be reported back to the model as a tool result, not crash,
        and the agent must not fabricate contents for a file it couldn't read.
"""

import json

import pytest

import agent_phase3 as ap3


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


def unreachable_search(_query):
    raise AssertionError("web_search should not have been called for this scenario")


@pytest.fixture(autouse=True)
def isolated_sessions_dir(tmp_path_factory, monkeypatch):
    import agent_phase1 as ap1

    monkeypatch.setattr(ap1, "SESSIONS_DIR", tmp_path_factory.mktemp("sessions"))


# ---- read_file() against real temp files -----------------------------


def test_read_file_returns_full_contents(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hello from disk", encoding="utf-8")

    assert ap3.read_file(str(f)) == "hello from disk"


def test_read_file_truncates_large_files(tmp_path, monkeypatch):
    monkeypatch.setattr(ap3, "READ_FILE_MAX_CHARS", 10)
    f = tmp_path / "big.txt"
    f.write_text("0123456789ABCDEF", encoding="utf-8")

    result = ap3.read_file(str(f))

    assert result.startswith("0123456789")
    assert "truncated" in result.lower()


def test_read_file_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        ap3.read_file("/definitely/does/not/exist.txt")


def test_read_file_directory_raises(tmp_path):
    with pytest.raises(IsADirectoryError):
        ap3.read_file(str(tmp_path))


def test_read_file_binary_file_raises(tmp_path):
    f = tmp_path / "image.bin"
    f.write_bytes(bytes(range(256)))

    with pytest.raises(ValueError):
        ap3.read_file(str(f))


# ---- PASS/FAIL behavioral pair (mocked model) -------------------------


def reads_then_answers(expected_path):
    def reply_fn(messages):
        tool_messages = [m for m in messages if m["role"] == "tool"]
        if not tool_messages:
            call = FakeToolCall("call-1", "read_file", json.dumps({"path": expected_path}))
            return {"content": "", "tool_calls": [call]}
        return {"content": f"The file says: {tool_messages[0]['content']}"}

    return reply_fn


def test_pass_case_agent_reads_real_file_and_answers_from_contents(tmp_path):
    """PASS: asked about a specific file, the agent calls read_file, gets
    its real contents from disk, and answers grounded in them."""
    f = tmp_path / "secret_number.txt"
    f.write_text("42", encoding="utf-8")

    client = FakeClient(reads_then_answers(str(f)))
    messages = [{"role": "user", "content": f"What number is in {f}?"}]

    reply = ap3.run_agentic_turn(client, "fake-model", ap3.DEFAULT_SYSTEM_PROMPT, messages, unreachable_search, ap3.read_file)

    assert "42" in reply
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert messages[1]["tool_calls"][0]["function"]["name"] == "read_file"


def test_fail_case_missing_file_is_reported_not_fabricated(tmp_path):
    """FAIL (guarded against): the file doesn't exist. The tool must report
    that as a failure, and the agent must not invent contents for it."""
    missing = tmp_path / "does_not_exist.txt"
    client = FakeClient(reads_then_answers(str(missing)))
    messages = [{"role": "user", "content": f"What's in {missing}?"}]

    reply = ap3.run_agentic_turn(client, "fake-model", ap3.DEFAULT_SYSTEM_PROMPT, messages, unreachable_search, ap3.read_file)

    tool_message = next(m for m in messages if m["role"] == "tool")
    assert "file read failed" in tool_message["content"].lower()
    assert "does not exist" in tool_message["content"].lower()
    assert reply == f"The file says: {tool_message['content']}"


# ---- tool selection across both tools ----------------------------------


def test_no_tool_needed_for_general_knowledge():
    def direct_reply(_messages):
        return {"content": "4", "tool_calls": None}

    client = FakeClient(direct_reply)
    messages = [{"role": "user", "content": "What is 2 + 2?"}]

    reply = ap3.run_agentic_turn(client, "fake-model", ap3.DEFAULT_SYSTEM_PROMPT, messages, unreachable_search, ap3.read_file)

    assert reply == "4"
    assert not any(m["role"] == "tool" for m in messages)


def test_web_search_tool_still_works_alongside_read_file():
    def searching_reply(messages):
        tool_messages = [m for m in messages if m["role"] == "tool"]
        if not tool_messages:
            call = FakeToolCall("call-1", "web_search", json.dumps({"query": "capital of Australia"}))
            return {"content": "", "tool_calls": [call]}
        return {"content": f"According to search: {tool_messages[0]['content'].splitlines()[0]}"}

    def fake_search(_query):
        return [{"title": "Canberra", "url": "https://example.com", "content": "Canberra is the capital of Australia."}]

    client = FakeClient(searching_reply)
    messages = [{"role": "user", "content": "What is the capital of Australia?"}]

    reply = ap3.run_agentic_turn(client, "fake-model", ap3.DEFAULT_SYSTEM_PROMPT, messages, fake_search, ap3.read_file)

    assert "canberra" in reply.lower()
    assert messages[1]["tool_calls"][0]["function"]["name"] == "web_search"


def test_unknown_tool_call_is_reported_gracefully():
    def hallucinated_tool_reply(messages):
        if not any(m["role"] == "tool" for m in messages):
            call = FakeToolCall("call-1", "delete_file", json.dumps({"path": "x"}))
            return {"content": "", "tool_calls": [call]}
        return {"content": "I can't delete files."}

    client = FakeClient(hallucinated_tool_reply)
    messages = [{"role": "user", "content": "Delete my file."}]

    ap3.run_agentic_turn(client, "fake-model", ap3.DEFAULT_SYSTEM_PROMPT, messages, unreachable_search, ap3.read_file)

    tool_message = next(m for m in messages if m["role"] == "tool")
    assert "unknown tool" in tool_message["content"].lower()


def test_gives_up_after_max_tool_iterations_instead_of_looping_forever():
    counter = {"n": 0}

    def always_wants_to_read(_messages):
        counter["n"] += 1
        call = FakeToolCall(f"call-{counter['n']}", "read_file", json.dumps({"path": "whatever.txt"}))
        return {"content": "", "tool_calls": [call]}

    client = FakeClient(always_wants_to_read)
    messages = [{"role": "user", "content": "loop forever?"}]

    reply = ap3.run_agentic_turn(client, "fake-model", ap3.DEFAULT_SYSTEM_PROMPT, messages, unreachable_search, lambda p: "irrelevant")

    assert "couldn't finish" in reply.lower()
    assert len(client.chat.completions.calls) == ap3.MAX_TOOL_ITERATIONS


# ---- end-to-end through run_chat, and persistence of tool messages ----


def test_run_chat_persists_full_tool_exchange_to_disk(tmp_path, monkeypatch):
    f = tmp_path / "secret_number.txt"
    f.write_text("42", encoding="utf-8")
    client = FakeClient(reads_then_answers(str(f)))
    inputs = iter([f"What number is in {f}?", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    ap3.run_chat(client, "fake-model", ap3.DEFAULT_SYSTEM_PROMPT, "file-session", unreachable_search, ap3.read_file)

    saved = ap3.load_session("file-session", ap3.DEFAULT_SYSTEM_PROMPT, "fake-model")
    roles = [m["role"] for m in saved["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert "42" in saved["messages"][-1]["content"]


def test_run_chat_rolls_back_entire_failed_turn(tmp_path, monkeypatch):
    call_count = {"n": 0}

    def first_turn_ok_second_turn_crashes(messages):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"content": "Nice to meet you, Athena."}
        raise RuntimeError("500 Internal Server Error (simulated)")

    client = FakeClient(first_turn_ok_second_turn_crashes)
    inputs = iter(["My name is Athena.", "what's in some_file.txt?", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    ap3.run_chat(client, "fake-model", ap3.DEFAULT_SYSTEM_PROMPT, "crash-session", unreachable_search, ap3.read_file)

    saved = ap3.load_session("crash-session", ap3.DEFAULT_SYSTEM_PROMPT, "fake-model")
    assert saved["messages"] == [
        {"role": "user", "content": "My name is Athena."},
        {"role": "assistant", "content": "Nice to meet you, Athena."},
    ]
