"""
In-memory rate limiter for LLM calls.
Resets on server restart — sufficient for a demo deployment.
"""

import threading
from collections import defaultdict

from app.config import MAX_REQUESTS_PER_SESSION, MAX_TOTAL_REQUESTS

_lock = threading.Lock()
_session_counts: dict[str, int] = defaultdict(int)
_global_count: int = 0

_DEMO_LIMIT_MSG = (
    "Sorry, this demo has reached its usage limit for now. "
    "Please try again later or contact the demo owner."
)
_SESSION_LIMIT_MSG = (
    "You've reached the request limit for this session. "
    "Start a new session to continue."
)


def check_rate_limit(session_id: str) -> str | None:
    """Return an error message string if a limit is hit, otherwise None."""
    global _global_count
    with _lock:
        if _global_count >= MAX_TOTAL_REQUESTS:
            return _DEMO_LIMIT_MSG
        if _session_counts[session_id] >= MAX_REQUESTS_PER_SESSION:
            return _SESSION_LIMIT_MSG
    return None


def record_llm_call(session_id: str) -> None:
    """Increment counters after an LLM call is made."""
    global _global_count
    with _lock:
        _session_counts[session_id] += 1
        _global_count += 1


def get_stats() -> dict:
    """Diagnostic helper — returns current counter state."""
    with _lock:
        return {
            "global_count": _global_count,
            "global_limit": MAX_TOTAL_REQUESTS,
            "session_counts": dict(_session_counts),
        }
