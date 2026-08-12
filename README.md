# Agentic AI Project — Phase 1

A stateful CLI chat interface that talks to an LLM via
[OpenRouter](https://openrouter.ai)'s OpenAI-compatible API. Conversation
history is persisted to disk per named session, so a conversation survives
closing and reopening the CLI.

## Features

- Streams replies token-by-token to the terminal.
- Persists each session's message history to `.chat_sessions/<name>.json`,
  so `--session work` picks up right where it left off across restarts.
- `/reset` clears the current session's history; `/exit` or Ctrl+C quits
  immediately.
- Graceful error handling: authentication, rate-limit, connection, and API
  errors — plus any unexpected provider-side crash — roll back only the
  failed turn instead of crashing the whole program and losing the
  conversation.

## Setup

1. Install [uv](https://docs.astral.sh/uv/) if you don't already have it.
2. `uv sync`
3. Create a `.env` file in the project root containing:
   ```
   OPENROUTER_API_KEY=your-key-here
   ```

## Usage

```
uv run python agent_phase1.py [--model MODEL] [--system "system prompt"] [--session NAME]
```

- `--model` defaults to `nvidia/nemotron-3.5-lightning:free`.
- `--system` sets the system prompt (default: a generic helpful-assistant prompt).
- `--session` defaults to `default`. Each session name gets its own history
  file at `.chat_sessions/<name>.json` (gitignored — this is the agent's own
  memory, not tracked in version control).

## Testing

```
uv run pytest                  # mocked, deterministic, no network or token cost
uv run pytest -m integration   # live calls against the real OpenRouter API (needs OPENROUTER_API_KEY)
```

Tests cover session persistence (save/load/reset, isolated named sessions),
correct request construction, and a PASS/FAIL behavioral pair — the agent
must recall a name stated earlier in the conversation ("My name is Athena"
→ "What is my name?" → recalls "Athena") but must *not* fabricate an answer
when that context is missing. The same pair is also run live against the
real model in `test_live_behavior.py`.

## Project layout

- `agent_phase1.py` — the CLI chat implementation.
- `tests/test_state_management.py` — mocked unit tests for session
  persistence and stateful behavior.
- `tests/test_live_behavior.py` — live integration tests against the real
  model (opt-in, requires an API key in `.env`).
- `tests/conftest.py` — loads `.env` before test collection, so the live
  tests correctly detect `OPENROUTER_API_KEY` and run instead of skipping.
- `.chat_sessions/` — persisted conversation history per session (gitignored).
- `src/agentic_ai_project/` — unused scaffold left over from `uv init`, not
  part of Phase 1.

See [CLAUDE.md](CLAUDE.md) for architecture notes and a running log of the
decisions and fixes made so far.
