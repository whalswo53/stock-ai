"""
Persistent SQLite store for 민재's investment decisions, rules, and patterns.
Provides the personalization context injected into every AI analysis.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional

from config.settings import STORAGE_DIR

DB_PATH = STORAGE_DIR / "memory.db"


class UserMemory:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Connection management ─────────────────────────────────────────────────

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS user_decisions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker      TEXT NOT NULL,
                    market      TEXT NOT NULL,
                    signal      TEXT NOT NULL,
                    action      TEXT NOT NULL,
                    reason      TEXT,
                    price       REAL,
                    sector      TEXT,
                    rsi         REAL,
                    macd_cross  TEXT,
                    outcome_pct REAL,
                    outcome_label TEXT,
                    context_json  TEXT,
                    created_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_rules (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_text    TEXT NOT NULL,
                    category     TEXT,
                    source       TEXT,
                    confidence   REAL DEFAULT 0.5,
                    win_rate     REAL,
                    sample_count INTEGER DEFAULT 0,
                    active       INTEGER DEFAULT 1,
                    created_at   TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_patterns (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    pattern_key  TEXT NOT NULL UNIQUE,
                    description  TEXT NOT NULL,
                    win_rate     REAL,
                    avg_return   REAL,
                    sample_count INTEGER DEFAULT 0,
                    updated_at   TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversation_history (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_decisions_ticker ON user_decisions(ticker);
                CREATE INDEX IF NOT EXISTS idx_decisions_created ON user_decisions(created_at);
                CREATE INDEX IF NOT EXISTS idx_conv_session ON conversation_history(session_id);
            """)

    # ── Decision recording ────────────────────────────────────────────────────

    def record_decision(
        self,
        ticker: str,
        market: str,
        signal: str,
        action: str,
        reason: str = "",
        price: Optional[float] = None,
        sector: str = "",
        rsi: Optional[float] = None,
        macd_cross: str = "",
        context: Optional[dict] = None,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO user_decisions
                    (ticker, market, signal, action, reason, price, sector,
                     rsi, macd_cross, context_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker, market, signal, action, reason, price, sector,
                    rsi, macd_cross,
                    json.dumps(context or {}, ensure_ascii=False),
                    datetime.now().isoformat(),
                ),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def update_outcome(
        self,
        decision_id: int,
        outcome_pct: float,
        outcome_label: str,
    ) -> None:
        """Call this after 1/4/12 weeks to record actual return."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE user_decisions SET outcome_pct=?, outcome_label=? WHERE id=?",
                (outcome_pct, outcome_label, decision_id),
            )

    # ── Decision queries ──────────────────────────────────────────────────────

    def get_recent_decisions(self, limit: int = 10) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM user_decisions
                ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_decisions_for_ticker(self, ticker: str, limit: int = 5) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM user_decisions
                WHERE ticker = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (ticker, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_decisions_with_outcomes(self) -> list[dict]:
        """Returns only decisions where outcome has been recorded."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM user_decisions
                WHERE outcome_pct IS NOT NULL
                ORDER BY created_at DESC
                """,
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Rules ─────────────────────────────────────────────────────────────────

    def add_rule(
        self,
        rule_text: str,
        category: str = "",
        source: str = "pattern",
        confidence: float = 0.5,
        win_rate: Optional[float] = None,
        sample_count: int = 0,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO user_rules
                    (rule_text, category, source, confidence,
                     win_rate, sample_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (rule_text, category, source, confidence,
                 win_rate, sample_count, datetime.now().isoformat()),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_active_rules(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM user_rules WHERE active=1 ORDER BY confidence DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def deactivate_rule(self, rule_id: int) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE user_rules SET active=0 WHERE id=?", (rule_id,))

    # ── Patterns ──────────────────────────────────────────────────────────────

    def upsert_pattern(
        self,
        pattern_type: str,
        pattern_key: str,
        description: str,
        win_rate: float,
        avg_return: float,
        sample_count: int,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO user_patterns
                    (pattern_type, pattern_key, description,
                     win_rate, avg_return, sample_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pattern_key) DO UPDATE SET
                    description  = excluded.description,
                    win_rate     = excluded.win_rate,
                    avg_return   = excluded.avg_return,
                    sample_count = excluded.sample_count,
                    updated_at   = excluded.updated_at
                """,
                (
                    pattern_type, pattern_key, description,
                    win_rate, avg_return, sample_count,
                    datetime.now().isoformat(),
                ),
            )

    def get_patterns(self, min_samples: int = 3) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM user_patterns
                WHERE sample_count >= ?
                ORDER BY win_rate DESC
                """,
                (min_samples,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Conversation history ──────────────────────────────────────────────────

    def save_message(self, session_id: str, role: str, content: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO conversation_history (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, datetime.now().isoformat()),
            )

    def get_recent_messages(
        self, session_id: str, limit: int = 20
    ) -> list[dict[str, str]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT role, content FROM conversation_history
                WHERE session_id = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    # ── Aggregated user context ───────────────────────────────────────────────

    def get_user_context(self, ticker: str = "") -> dict[str, Any]:
        """
        Returns a dict that ClaudeAnalyst injects into every analysis prompt.
        """
        return {
            "recent_decisions": self.get_recent_decisions(limit=5),
            "active_rules": self.get_active_rules(),
            "patterns": self.get_patterns(min_samples=3),
            "similar_decisions": (
                self.get_decisions_for_ticker(ticker, limit=3) if ticker else []
            ),
        }

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        with self._conn() as conn:
            n_decisions = conn.execute(
                "SELECT COUNT(*) FROM user_decisions"
            ).fetchone()[0]
            n_outcomes = conn.execute(
                "SELECT COUNT(*) FROM user_decisions WHERE outcome_pct IS NOT NULL"
            ).fetchone()[0]
            n_rules = conn.execute(
                "SELECT COUNT(*) FROM user_rules WHERE active=1"
            ).fetchone()[0]
            n_patterns = conn.execute(
                "SELECT COUNT(*) FROM user_patterns"
            ).fetchone()[0]
        return {
            "decisions": n_decisions,
            "outcomes_recorded": n_outcomes,
            "active_rules": n_rules,
            "patterns": n_patterns,
        }
