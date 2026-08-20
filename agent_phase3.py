"""Phase 3: Phase 2's chat loop (web_search) plus a read_file tool, so the
model can look at the actual contents of a local file when the user refers
to one, instead of guessing.

Run with:
    python agent_phase3.py [--model MODEL] [--system PROMPT] [--session NAME]

Requires OPENROUTER_API_KEY (both phases) and TAVILY_API_KEY (web_search
only) in the environment or a .env file in the project root.

SECURITY NOTE: read_file is intentionally unsandboxed -- it can read any
file the OS user running this script can read, anywhere on disk (no
restriction to a project directory). Whatever it reads is sent to the
configured OpenRouter model as part of the conversation. Don't point this
at a session with access to files you don't want sent to a third-party API
(SSH keys, credentials, browser profiles, etc.).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import openai
from dotenv import load_dotenv

from agent_phase1 import (
    MAX_TOKENS,
    build_request_messages,
    clear_session,
    load_session,
    make_client,
    save_session,
)
from agent_phase2 import WEB_SEARCH_TOOL, format_search_results, tavily_search

DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning:free"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant with two tools: web_search for "
    "current events or facts you're not confident about, and read_file for "
    "looking at the actual contents of a local file the user refers to. "
    "Use each only when it's actually needed -- don't search or read files "
    "for things you already know. read_file genuinely reads files from the "
    "user's own local machine -- you do have this capability, it is not a "
    "sandboxed or hypothetical environment. If you choose not to repeat a "
    "file's contents back (e.g. because it contains offensive material), "
    "say that plainly as a content decision -- never claim you lack the "
    "ability to read local files, since that would be false."
)
MAX_TOOL_ITERATIONS = 4
READ_FILE_MAX_CHARS = 4000

READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read the contents of a text file from the local filesystem. "
            "Use this when the user refers to a specific file and you need "
            "to see what's actually in it, rather than guessing. Read-only "
            "-- this cannot modify, create, or delete files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative path to the file to read."}
            },
            "required": ["path"],
        },
    },
}

AGENT_TOOLS = [WEB_SEARCH_TOOL, READ_FILE_TOOL]


def read_file(path: str) -> str:
    if not path:
        raise ValueError("No path given.")
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{resolved} does not exist.")
    if resolved.is_dir():
        raise IsADirectoryError(f"{resolved} is a directory, not a file.")
    try:
        text = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"{resolved} doesn't look like a text file (couldn't decode as UTF-8).")

    if len(text) > READ_FILE_MAX_CHARS:
        return (
            text[:READ_FILE_MAX_CHARS]
            + f"\n\n[... truncated: file is {len(text)} characters, showing the first {READ_FILE_MAX_CHARS} ...]"
        )
    return text


def _serialize_tool_call(call) -> dict:
    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.function.name, "arguments": call.function.arguments},
    }


def run_tool_call(call, search_fn, read_file_fn) -> dict:
    name = call.function.name
    try:
        args = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}

    if name == "web_search":
        try:
            content = format_search_results(search_fn(args.get("query", "")))
        except Exception as e:
            content = f"Search failed: {e}"
    elif name == "read_file":
        try:
            content = read_file_fn(args.get("path", ""))
        except Exception as e:
            content = f"File read failed: {e}"
    else:
        content = f"Unknown tool requested: {name}"

    return {"role": "tool", "tool_call_id": call.id, "content": content}


def run_agentic_turn(client, model: str, system: str, messages: list[dict], search_fn, read_file_fn) -> str:
    """Let the model call web_search and/or read_file as needed, appending
    every intermediate tool exchange plus the final answer onto `messages`
    in place. Returns just the final answer text."""
    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.chat.completions.create(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=build_request_messages(system, messages),
            tools=AGENT_TOOLS,
            tool_choice="auto",
        )
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            reply = message.content or ""
            messages.append({"role": "assistant", "content": reply})
            print(reply)
            return reply

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [_serialize_tool_call(c) for c in tool_calls],
            }
        )
        for call in tool_calls:
            print(f"[{call.function.name}: {call.function.arguments}]")
            messages.append(run_tool_call(call, search_fn, read_file_fn))

    fallback = "I couldn't finish that after several tool calls — try rephrasing your question."
    messages.append({"role": "assistant", "content": fallback})
    print(fallback)
    return fallback


def run_chat(client, model: str, system: str, session: str, search_fn, read_file_fn) -> None:
    data = load_session(session, system, model)
    messages: list[dict] = data["messages"]
    system = data["system"]
    model = data["model"]

    print(f"OpenRouter CLI chat with web search + file reading - Phase 3 (session: {session})")
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

        checkpoint = len(messages)
        messages.append({"role": "user", "content": user_input})
        print("Assistant: ", end="", flush=True)

        try:
            run_agentic_turn(client, model, system, messages, search_fn, read_file_fn)
        except KeyboardInterrupt:
            del messages[checkpoint:]
            print("\nGoodbye.")
            return
        except openai.AuthenticationError:
            print(
                "\nAuthentication failed. Set OPENROUTER_API_KEY in .env.",
                file=sys.stderr,
            )
            del messages[checkpoint:]
            continue
        except openai.RateLimitError:
            print("\nRate limited — please wait a moment and try again.")
            del messages[checkpoint:]
            continue
        except openai.APIConnectionError:
            print("\nNetwork error — check your connection and try again.")
            del messages[checkpoint:]
            continue
        except openai.APIStatusError as e:
            print(f"\nAPI error ({e.status_code}): {e.message}")
            del messages[checkpoint:]
            continue
        except Exception as e:
            print(f"\nUnexpected error talking to the model: {e}")
            print("The previous turns are safe; try again or /reset.")
            del messages[checkpoint:]
            continue

        save_session(session, {"model": model, "system": system, "messages": messages})
        print()


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Stateful CLI chat via OpenRouter with web-search and read_file tools."
    )
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
    run_chat(client, args.model, args.system, args.session, search_fn=tavily_search, read_file_fn=read_file)


if __name__ == "__main__":
    main()
