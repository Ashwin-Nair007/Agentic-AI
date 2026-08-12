"""Phase 1: a stateful CLI chat interface that talks to an LLM via OpenRouter.

Run with:
    python agent_phase1.py [--model MODEL] [--system PROMPT] [--session NAME]

Requires OPENROUTER_API_KEY to be set in the environment (or in a .env file
in the project root). Conversation history is persisted to disk per session
name under .chat_sessions/, so `python agent_phase1.py --session work` picks
up right where that session left off, even across restarts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import openai
from dotenv import load_dotenv

DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning:free"
MAX_TOKENS = 8192
DEFAULT_SYSTEM_PROMPT = "You are a helpful, concise assistant."
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SESSIONS_DIR = Path(__file__).resolve().parent / ".chat_sessions"


def session_path(session: str) -> Path:
    return SESSIONS_DIR / f"{session}.json"


def load_session(session: str, default_system: str, default_model: str) -> dict:
    path = session_path(session)
    if not path.exists():
        return {"model": default_model, "system": default_system, "messages": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("model", default_model)
    data.setdefault("system", default_system)
    data.setdefault("messages", [])
    return data


def save_session(session: str, data: dict) -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    session_path(session).write_text(json.dumps(data, indent=2), encoding="utf-8")


def clear_session(session: str) -> None:
    path = session_path(session)
    if path.exists():
        path.unlink()


def build_request_messages(system: str, messages: list[dict]) -> list[dict]:
    return [{"role": "system", "content": system}, *messages]


def make_client(api_key: str | None = None) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
    )


def stream_reply(client, model: str, system: str, messages: list[dict]) -> str:
    """Stream the model's reply to stdout and return the full text."""
    chunks: list[str] = []
    stream = client.chat.completions.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=build_request_messages(system, messages),
        stream=True,
    )
    for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            chunks.append(delta)
    print()
    return "".join(chunks)


def run_chat(client, model: str, system: str, session: str) -> None:
    data = load_session(session, system, model)
    messages: list[dict] = data["messages"]
    system = data["system"]
    model = data["model"]

    print(f"OpenRouter CLI chat - Phase 1 (session: {session})")
    if messages:
        print(f"Resumed {len(messages)} prior message(s).")
    print("Commands: /reset (clear history), /exit (quit)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if not user_input:
            continue
        if user_input == "/exit":
            print("Goodbye.")
            return
        if user_input == "/reset":
            messages.clear()
            clear_session(session)
            print("Conversation history cleared.\n")
            continue

        messages.append({"role": "user", "content": user_input})
        print("Assistant: ", end="", flush=True)

        try:
            reply = stream_reply(client, model, system, messages)
        except KeyboardInterrupt:
            # Ctrl+C should always quit, same as at the "You:" prompt --
            # not silently cancel this turn and loop back, which made it
            # look like the program wouldn't exit.
            messages.pop()
            print("\nGoodbye.")
            return
        except openai.AuthenticationError:
            print(
                "\nAuthentication failed. Set OPENROUTER_API_KEY in .env.",
                file=sys.stderr,
            )
            messages.pop()
            continue
        except openai.RateLimitError:
            print("\nRate limited — please wait a moment and try again.")
            messages.pop()
            continue
        except openai.APIConnectionError:
            print("\nNetwork error — check your connection and try again.")
            messages.pop()
            continue
        except openai.APIStatusError as e:
            print(f"\nAPI error ({e.status_code}): {e.message}")
            messages.pop()
            continue
        except Exception as e:
            # Catches anything not covered above -- e.g. a provider-side
            # crash (like OpenRouter/NVIDIA's free-tier "internal server
            # error") that surfaces as a raw/unwrapped exception mid-stream
            # rather than a typed openai.* error. Without this, an unhandled
            # exception here would kill the process and lose the in-memory
            # conversation for this run.
            print(f"\nUnexpected error talking to the model: {e}")
            print("The previous turns are safe; try again or /reset.")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})
        save_session(session, {"model": model, "system": system, "messages": messages})
        print()


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Stateful CLI chat via OpenRouter.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"OpenRouter model slug (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to steer the assistant's behavior.",
    )
    parser.add_argument(
        "--session",
        default="default",
        help="Session name; conversation history persists to .chat_sessions/<name>.json (default: default)",
    )
    args = parser.parse_args()

    client = make_client()
    run_chat(client, args.model, args.system, args.session)


if __name__ == "__main__":
    main()
