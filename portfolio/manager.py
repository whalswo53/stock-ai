"""
SQLite-backed portfolio holdings manager.
Schema mirrors user_memory.py patterns for consistency.
"""
from __future__ import annotations

import csv
import io
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional

from config.settings import STORAGE_DIR

DB_PATH = STORAGE_DIR / "portfolio.db"

GROUP_ACCUMULATING = "accumulating"   # 모으는 중
GROUP_HOLDING = "holding"             # 보유 중


class PortfolioManager:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Connection ────────────────────────────────────────────────────────────

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
                CREATE TABLE IF NOT EXISTS portfolio_holdings (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker      TEXT NOT NULL,
                    name        TEXT NOT NULL DEFAULT '',
                    quantity    REAL NOT NULL,
                    target_qty  REAL,
                    avg_cost    REAL NOT NULL,
                    buy_date    TEXT DEFAULT '',
                    group_type  TEXT NOT NULL DEFAULT 'holding',
                    sector      TEXT DEFAULT '',
                    notes       TEXT DEFAULT '',
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_holdings_ticker
                    ON portfolio_holdings(ticker);
                CREATE INDEX IF NOT EXISTS idx_holdings_group
                    ON portfolio_holdings(group_type);
            """)

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def add_holding(
        self,
        ticker: str,
        name: str = "",
        quantity: float = 0.0,
        avg_cost: float = 0.0,
        buy_date: str = "",
        group_type: str = GROUP_HOLDING,
        target_qty: Optional[float] = None,
        sector: str = "",
        notes: str = "",
    ) -> int:
        now = datetime.now().isoformat()
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO portfolio_holdings
                    (ticker, name, quantity, target_qty, avg_cost, buy_date,
                     group_type, sector, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ticker, name, quantity, target_qty, avg_cost, buy_date,
                 group_type, sector, notes, now, now),
            )
            return int(cur.lastrowid)  # type: ignore[arg-type]

    def update_holding(self, holding_id: int, **kwargs: Any) -> None:
        allowed = {
            "ticker", "name", "quantity", "target_qty", "avg_cost",
            "buy_date", "group_type", "sector", "notes",
        }
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return
        fields["updated_at"] = datetime.now().isoformat()
        sets = ", ".join(f"{k}=?" for k in fields)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE portfolio_holdings SET {sets} WHERE id=?",
                (*fields.values(), holding_id),
            )

    def delete_holding(self, holding_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM portfolio_holdings WHERE id=?", (holding_id,))

    def move_group(self, holding_id: int, new_group: str) -> None:
        self.update_holding(holding_id, group_type=new_group)

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_all(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM portfolio_holdings ORDER BY group_type, created_at"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_by_group(self, group_type: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM portfolio_holdings WHERE group_type=? ORDER BY created_at",
                (group_type,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_by_id(self, holding_id: int) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM portfolio_holdings WHERE id=?", (holding_id,)
            ).fetchone()
        return dict(row) if row else None

    def count(self) -> int:
        with self._conn() as conn:
            return int(conn.execute(
                "SELECT COUNT(*) FROM portfolio_holdings"
            ).fetchone()[0])

    # ── CSV import ────────────────────────────────────────────────────────────

    def import_csv(self, csv_content: str) -> tuple[int, list[str]]:
        """
        Imports holdings from a CSV string.
        Required columns: 종목코드 (or ticker), 수량 (or quantity), 평균매입가 (or avg_cost)
        Optional:  종목명, 매입일, 그룹, 목표수량, 섹터, 메모
        Returns (inserted_count, error_list).
        """
        reader = csv.DictReader(io.StringIO(csv_content))
        aliases: dict[str, list[str]] = {
            "ticker":     ["종목코드", "ticker", "Ticker", "종목"],
            "name":       ["종목명", "name", "Name"],
            "quantity":   ["수량", "quantity", "Quantity", "보유수량"],
            "avg_cost":   ["평균매입가", "avg_cost", "매입가", "평단가"],
            "buy_date":   ["매입일", "buy_date", "매입일자"],
            "group_type": ["그룹", "group_type", "group"],
            "target_qty": ["목표수량", "target_qty"],
            "sector":     ["섹터", "sector"],
            "notes":      ["메모", "notes"],
        }

        def _get(row: dict, key: str) -> Optional[str]:
            for alias in aliases[key]:
                if alias in row:
                    return str(row[alias]).strip()
            return None

        inserted, errors = 0, []
        for i, row in enumerate(reader, 1):
            try:
                ticker = (_get(row, "ticker") or "").upper()
                if not ticker:
                    errors.append(f"행 {i}: 종목코드 누락")
                    continue

                qty = float((_get(row, "quantity") or "0").replace(",", ""))
                avg_cost = float((_get(row, "avg_cost") or "0").replace(",", ""))

                raw_group = (_get(row, "group_type") or "holding").lower()
                group_type = (
                    GROUP_ACCUMULATING
                    if raw_group in ("모으는중", "모으는 중", "accumulating")
                    else GROUP_HOLDING
                )

                tgt_raw = _get(row, "target_qty")
                target_qty = float(tgt_raw.replace(",", "")) if tgt_raw else None

                self.add_holding(
                    ticker=ticker,
                    name=_get(row, "name") or "",
                    quantity=qty,
                    avg_cost=avg_cost,
                    buy_date=_get(row, "buy_date") or "",
                    group_type=group_type,
                    target_qty=target_qty,
                    sector=_get(row, "sector") or "",
                    notes=_get(row, "notes") or "",
                )
                inserted += 1
            except (ValueError, TypeError) as exc:
                errors.append(f"행 {i}: {exc}")

        return inserted, errors

    # ── CSV template ──────────────────────────────────────────────────────────

    @staticmethod
    def csv_template() -> str:
        header = "종목코드,종목명,수량,평균매입가,매입일,그룹,목표수량,메모"
        sample = "005930.KS,삼성전자,10,70000,2024-01-15,보유 중,,장기 보유"
        return f"{header}\n{sample}\n"
