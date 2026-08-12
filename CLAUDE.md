# CLAUDE.md

Guidance for Claude Code (or anyone else) working in this repository.

## Project overview

Phase 1 of an agentic AI project: `agent_phase1.py` is a stateful CLI chat
interface that talks to an LLM through [OpenRouter](https://openrouter.ai)'s
OpenAI-compatible API. Conversation history persists to disk per named
session, so a session survives process restarts.

## Environment & setup

- Python >=3.12, dependency/venv management via `uv` (`uv.lock` is committed).
- Install deps: `uv sync`
- Secrets live in `.env` (gitignored, never commit it): requires
  `OPENROUTER_API_KEY`.
- Run: `uv run python agent_phase1.py [--model MODEL] [--system PROMPT] [--session NAME]`

## Architecture (`agent_phase1.py`)

- `make_client()` — builds an `openai.OpenAI` client pointed at OpenRouter's
  `base_url` instead of OpenAI's.
- `session_path` / `load_session` / `save_session` / `clear_session` — JSON
  persistence for a named session's message history, under
  `.chat_sessions/<name>.json`. **This is the agent's own conversation
  memory** — a separate thing from any Claude Code session transcript.
  It's gitignored on purpose (grows over time, may contain arbitrary
  personal content).
- `build_request_messages` — prepends the system prompt to the stored
  message list before sending it to the model.
- `stream_reply` — streams a completion, printing tokens as they arrive,
  and returns the full text.
- `run_chat` — the REPL: reads input, handles `/reset` and `/exit`, calls
  `stream_reply`, and appends+persists both turns to disk on success.

## Testing

- `uv run pytest` — mocked suite, fast/deterministic, no network or token
  cost. Runs by default.
- `uv run pytest -m integration` — live calls against the real OpenRouter
  API; needs `OPENROUTER_API_KEY`, costs tokens, and is skipped otherwise.
- `tests/test_state_management.py` — uses a `FakeClient` to cover session
  persistence, request construction, and the PASS/FAIL "what is my name?"
  behavioral example from the original task spec.
- `tests/test_live_behavior.py` — the same PASS/FAIL pair, but against the
  real model.

## Notable decisions & history

1. **Provider**: OpenRouter, not Anthropic or OpenAI directly — accessed via
   the `openai` SDK pointed at OpenRouter's `base_url`. Chosen explicitly
   over the initial anthropic-based draft.
2. **Default model**: `nvidia/nemotron-3.5-lightning:free` (OpenRouter's
   free tier). Known to be flaky / prone to internal-server errors under
   real use — override with `--model` if that becomes a problem.
3. **Persistence**: sessions are named and saved to disk
   (`--session <name>`, default `default`) rather than kept in-memory only,
   so a conversation survives closing and reopening the CLI.
4. **Fix — crash on provider errors**: an unhandled, non-`openai.*`-typed
   exception mid-stream (e.g. the free model's internal server errors
   surfacing as a raw exception rather than a typed `openai.APIStatusError`)
   used to crash the whole process and lose the in-memory conversation for
   that run. `run_chat` now has a catch-all `except Exception` that rolls
   back only the failed turn and keeps the REPL running; already-persisted
   turns on disk are unaffected.
5. **Fix — Ctrl+C didn't exit**: pressing Ctrl+C while a reply was
   streaming (or hung) used to just cancel that one turn and loop back to
   the prompt, silently requiring a second Ctrl+C to actually quit. A
   single Ctrl+C now always exits immediately, matching the behavior at the
   `You:` prompt.
6. **Fix — live tests always skipped even with a key set**: `load_dotenv()`
   in `agent_phase1.py` only runs inside `main()`, which pytest never calls,
   so `OPENROUTER_API_KEY` from `.env` never reached `os.environ` during a
   test run — `test_live_behavior.py`'s `skipif` always saw it as unset.
   Added `tests/conftest.py` to load `.env` at collection time, before any
   `skipif` is evaluated.

## Conventions

- No comments in code unless explaining a non-obvious "why".
- Tests were written test-first for behavioral changes: the PASS/FAIL
  "what is my name?" cases, and the two regression tests for the crash and
  Ctrl+C fixes above, were each written to fail against the old behavior
  before the fix landed.

## Not part of Phase 1

- `src/agentic_ai_project/` is the default `uv init` scaffold and is
  unrelated to `agent_phase1.py`.
- `practice/` is an empty scratch directory.
