# CLAUDE.md

Guidance for Claude Code (or anyone else) working in this repository.

## Project overview

An agentic AI project built up in phases:

- `agent_phase1.py` — a stateful CLI chat interface that talks to an LLM
  through [OpenRouter](https://openrouter.ai)'s OpenAI-compatible API.
  Conversation history persists to disk per named session, so a session
  survives process restarts.
- `agent_phase2.py` — the same chat loop, extended with a `web_search`
  tool the model can call via OpenRouter's function-calling API when it
  needs current or factual information it doesn't already know. Imports
  and reuses Phase 1's client/session helpers rather than duplicating them;
  Phase 1 itself is untouched.

## Environment & setup

- Python >=3.12, dependency/venv management via `uv` (`uv.lock` is committed).
- Install deps: `uv sync`
- Secrets live in `.env` (gitignored, never commit it): requires
  `OPENROUTER_API_KEY` (both phases) and `TAVILY_API_KEY` (Phase 2's search
  tool only — [tavily.com](https://tavily.com), free tier).
- Run Phase 1: `uv run python agent_phase1.py [--model MODEL] [--system PROMPT] [--session NAME]`
- Run Phase 2: `uv run python agent_phase2.py [--model MODEL] [--system PROMPT] [--session NAME]`
- **This OpenRouter account currently has no purchased credits** — every
  paid model 402s ("Insufficient credits"). Only `:free`-suffixed models
  work until credits are added. Verified empirically (see decision #7
  below) that the Phase 1 default, `nvidia/nemotron-3.5-lightning:free`,
  does support tool calling, so Phase 2 keeps the same default model.

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

## Architecture (`agent_phase2.py`)

Imports `make_client`, `load_session`, `save_session`, `clear_session`, and
`build_request_messages` from `agent_phase1` — only the REPL/turn logic is
reimplemented, because it genuinely differs (tool loop, multiple messages
per turn).

- `WEB_SEARCH_TOOL` — the OpenAI-style function-calling schema advertised
  to the model (`tools=[...]`, `tool_choice="auto"`).
- `tavily_search` — calls the real Tavily REST API (`httpx.post`), reads
  `TAVILY_API_KEY` from the environment.
- `format_search_results` — turns Tavily's result list into the plain-text
  blob fed back to the model as a tool result.
- `run_tool_call` — executes one tool call the model requested and returns
  the corresponding `{"role": "tool", "tool_call_id": ..., "content": ...}`
  message; catches unknown tool names and search failures and turns both
  into a reported tool result instead of raising.
- `run_agentic_turn` — the tool loop: calls the model (non-streaming, so
  `tool_calls` come back as structured JSON, not text deltas); if the
  model requests tool calls, executes each via `run_tool_call` and appends
  everything (the assistant's tool-call message + each tool result) to
  `messages` in place, then loops back to the model; once the model
  answers with no more tool calls, appends and returns that as the final
  answer. Bails out after `MAX_TOOL_ITERATIONS` (4) with a fallback
  message instead of looping forever.
- `run_chat` — same shape as Phase 1's, but calls `run_agentic_turn`
  instead of `stream_reply`, and rolls back to a checkpoint taken *before*
  the user's message on any failure — since a failed turn here can have
  appended several messages (tool calls + results), not just one.
- `search_fn` is threaded through as a parameter everywhere (`run_chat` →
  `run_agentic_turn` → `run_tool_call`) so tests can inject a fake search
  function and never hit the network, mirroring how Phase 1's tests inject
  a `FakeClient` instead of a real one.

## Testing

- `uv run pytest` — mocked suite, fast/deterministic, no network or token
  cost. Runs by default.
- `uv run pytest -m integration` — live calls against the real OpenRouter
  (and, for Phase 2, Tavily) APIs; needs the relevant API key(s), costs
  tokens/search quota, and is skipped otherwise.
- `tests/test_state_management.py` — uses a `FakeClient` to cover Phase 1's
  session persistence, request construction, and the PASS/FAIL "what is my
  name?" behavioral example from the original task spec.
- `tests/test_live_behavior.py` — the same PASS/FAIL pair, but against the
  real model.
- `tests/test_tool_calling.py` — uses a scriptable `FakeClient` (returns
  either a tool call or a final message depending on what's already in the
  transcript) to cover the tool loop: searches when it should, doesn't
  search for things it already knows, handles a broken search backend and
  an unknown/hallucinated tool name gracefully, stops after
  `MAX_TOOL_ITERATIONS`, and persists/rolls back full tool exchanges
  correctly through `run_chat`.
- `tests/test_live_tool_calling.py` — the same PASS/FAIL pair (searches for
  what it doesn't know, skips search for `2 + 2`) against the real model
  and real Tavily API; needs both `OPENROUTER_API_KEY` and
  `TAVILY_API_KEY`.

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
7. **Phase 2 search backend**: Tavily, chosen over DuckDuckGo (no key, but
   flakier/rawer results) and Serper (also viable) for its clean,
   LLM-oriented result format and free tier.
8. **Phase 2 tool-calling model**: before building anything, empirically
   tested several OpenRouter model slugs against this account with a
   `tools=[...]` request. Every paid model 402'd (no credits on the
   account); of the free models tried, only `nvidia/nemotron-3.5-lightning:free`
   both existed and correctly returned a structured `tool_calls` response.
   So Phase 2 keeps Phase 1's default model rather than switching to a paid
   one, contrary to the original plan to switch — the empirical result
   overrode the initial assumption that the free model wouldn't support
   tool calling.
9. **Phase 2 non-streaming tool-decision calls**: `run_agentic_turn` doesn't
   use `stream_reply` — tool-call detection needs the structured
   `tool_calls` field on the response message, which isn't reliably
   assembled from streamed text deltas without extra complexity. Only the
   final, no-more-tools-needed answer is printed (as a single block, not
   token-by-token like Phase 1).

## Conventions

- No comments in code unless explaining a non-obvious "why".
- Tests were written test-first for behavioral changes: the PASS/FAIL
  "what is my name?" cases, the PASS/FAIL search-tool cases, and the
  regression tests for the crash, Ctrl+C, and dotenv-in-tests fixes above,
  were each written to fail against the old/missing behavior before the
  fix or feature landed.

## Not part of either phase

- `src/agentic_ai_project/` is the default `uv init` scaffold and is
  unrelated to `agent_phase1.py`/`agent_phase2.py`.
- `practice/` is an empty scratch directory.
