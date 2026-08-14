"""Phase 2: agent_phase1's chat loop, plus a web_search tool the model can
call when it needs current or factual information it doesn't already know.

Run with:
    python agent_phase2.py [--model MODEL] [--system PROMPT] [--session NAME]

Requires OPENROUTER_API_KEY (same as Phase 1) and TAVILY_API_KEY (for the
search tool) in the environment or a .env file in the project root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import httpx
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

DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning:free"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. You have a web_search tool: use "
    "it for current events, facts you're not confident about, or anything "
    "time-sensitive. Don't use it for things you already know."
)
TAVILY_API_URL = "https://api.tavily.com/search"
SEARCH_RESULT_COUNT = 5
MAX_TOOL_ITERATIONS = 4

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the live web for current, factual, or recent information "
            "you don't already know or aren't sure about (news, prices, "
            "dates, scores, releases, etc.). Do not use it for general "
            "knowledge you already know confidently."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    },
}


def format_search_results(results: list[dict]) -> str:
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "(no title)")
        url = r.get("url", "")
        snippet = (r.get("content") or "").strip()
        lines.append(f"{i}. {title} — {url}\n   {snippet}")
    return "\n".join(lines)


def tavily_search(query: str, api_key: str | None = None, max_results: int = SEARCH_RESULT_COUNT) -> list[dict]:
    key = api_key or os.environ.get("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY is not set. Add it to .env to enable web search.")
    response = httpx.post(
        TAVILY_API_URL,
        json={"api_key": key, "query": query, "max_results": max_results},
        timeout=20.0,
    )
    response.raise_for_status()
    return response.json().get("results", [])


def _serialize_tool_call(call) -> dict:
    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.function.name, "arguments": call.function.arguments},
    }


def run_tool_call(call, search_fn) -> dict:
    if call.function.name != "web_search":
        content = f"Unknown tool requested: {call.function.name}"
    else:
        try:
            query = json.loads(call.function.arguments or "{}").get("query", "")
        except json.JSONDecodeError:
            query = ""
        try:
            content = format_search_results(search_fn(query))
        except Exception as e:
            content = f"Search failed: {e}"
    return {"role": "tool", "tool_call_id": call.id, "content": content}


def run_agentic_turn(client, model: str, system: str, messages: list[dict], search_fn) -> str:
    """Let the model call web_search as needed, appending every intermediate
    tool exchange plus the final answer onto `messages` in place. Returns
    just the final answer text."""
    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.chat.completions.create(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=build_request_messages(system, messages),
            tools=[WEB_SEARCH_TOOL],
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
            query = json.loads(call.function.arguments or "{}").get("query", "")
            print(f"[web_search: {query}]")
            messages.append(run_tool_call(call, search_fn))

    fallback = "I couldn't find a good answer after several searches — try rephrasing your question."
    messages.append({"role": "assistant", "content": fallback})
    print(fallback)
    return fallback


def run_chat(client, model: str, system: str, session: str, search_fn) -> None:
    data = load_session(session, system, model)
    messages: list[dict] = data["messages"]
    system = data["system"]
    model = data["model"]

    print(f"OpenRouter CLI chat with web search - Phase 2 (session: {session})")
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
            run_agentic_turn(client, model, system, messages, search_fn)
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

    parser = argparse.ArgumentParser(description="Stateful CLI chat via OpenRouter with a web-search tool.")
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
    run_chat(client, args.model, args.system, args.session, search_fn=tavily_search)


if __name__ == "__main__":
    main()
