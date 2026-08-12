"""Shared pytest setup.

agent_phase1.load_dotenv() only runs inside main(), which pytest never
calls -- so without this, OPENROUTER_API_KEY from .env never reaches
os.environ during a test run, and the live tests' skipif always sees it
as unset even when a real key is configured.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
