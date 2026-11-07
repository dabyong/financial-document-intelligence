"""Best-effort, app-instance-wide Gemini request guard for the Streamlit demo.

All Streamlit sessions sharing this SQLite database share one atomic limit.
Reservations are counted even if the API fails (conservative). This is NOT an
external distributed rate-limiting service: separate replicas or a reset cloud
filesystem will have separate histories. The Gemini provider quota remains
authoritative. Use an access code or disable live AI on a public demo.
"""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path


class RequestDenied(RuntimeError):
    """Friendly, non-sensitive denial message for the UI."""


def live_ai_enabled() -> bool:
    return os.getenv("P5_LIVE_AI_ENABLED", "false").strip().lower() == "true"


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        result = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def reserve_request(root: Path) -> tuple[int, int]:
    """Atomically reserve an API attempt. Returns (used_today, configured_limit).

    This rolling 24h budget is deliberately below the provider project quota;
    it is not the same as the provider's midnight-Pacific reset window.
    """
    daily_limit = _bounded_int("P5_APP_DAILY_LIMIT", 12, 1, 100)
    cooldown = _bounded_int("P5_APP_COOLDOWN_SECONDS", 60, 1, 3600)
    now = time.time()
    database = root / "data" / "runtime" / "ai_request_guard.sqlite3"
    database.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(database), timeout=10, isolation_level=None)
    try:
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("CREATE TABLE IF NOT EXISTS reservations (created_at REAL NOT NULL)")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM reservations WHERE created_at < ?", (now - 7 * 86400,))
        used = conn.execute(
            "SELECT COUNT(*) FROM reservations WHERE created_at > ?",
            (now - 86400,),
        ).fetchone()[0]
        latest = conn.execute("SELECT MAX(created_at) FROM reservations").fetchone()[0]
        if used >= daily_limit:
            conn.execute("ROLLBACK")
            raise RequestDenied(
                "The demo's shared AI request budget has been reached. "
                "The financial dashboard remains available."
            )
        if latest is not None and now - latest < cooldown:
            conn.execute("ROLLBACK")
            remaining = max(1, int(cooldown - (now - latest)) + 1)
            raise RequestDenied(
                f"A demo AI request was submitted recently. Retry in roughly {remaining} seconds."
            )
        conn.execute("INSERT INTO reservations (created_at) VALUES (?)", (now,))
        conn.execute("COMMIT")
        return used + 1, daily_limit
    except RequestDenied:
        raise
    except Exception:
        # Fail closed if lock/state is unavailable; never silently bypass limits.
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    finally:
        conn.close()
