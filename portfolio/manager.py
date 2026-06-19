"""
SQLite-backed portfolio holdings manager.

v2 스키마 변경:
  - buy_date 컬럼 제거
  - accum_period / accum_type / accum_value 추가 ("모으는 중" 적립 계획)
  - purchase_history 테이블 추가 (개별 매수 내역)
  - 기존 DB 자동 마이그레이션
"""
from __future__ import annotations

import csv
import io
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional

from config.settings import STORAGE_DIR

DB_PATH = STORAGE_DIR / "portfolio.db"

_usd_krw_cache: dict = {"rate": None, "ts": 0.0}


def get_usd_krw_rate() -> float:
    """USD/KRW 환율 조회 (1시간 캐시, 실패 시 1300 반환)."""
    if _usd_krw_cache["rate"] is not None and time.time() - _usd_krw_cache["ts"] < 3600:
        return _usd_krw_cache["rate"]
    try:
        import yfinance as yf
        ticker = yf.Ticker("KRW=X")
        hist = ticker.history(period="1d")
        if hist.empty:
            raise ValueError("빈 데이터")
        rate = float(hist["Close"].iloc[-1])
        _usd_krw_cache["rate"] = rate
        _usd_krw_cache["ts"] = time.time()
        return rate
    except Exception:
        return 1300.0

GROUP_ACCUMULATING = "accumulating"   # 모으는 중
GROUP_HOLDING      = "holding"        # 보유 중

# ── SQL 상수 ──────────────────────────────────────────────────────────────────

_DDL_HOLDINGS = """\
CREATE TABLE IF NOT EXISTS portfolio_holdings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT NOT NULL,
    name         TEXT NOT NULL DEFAULT '',
    quantity     REAL NOT NULL DEFAULT 0,
    target_qty   REAL,
    avg_cost     REAL NOT NULL DEFAULT 0,
    group_type   TEXT NOT NULL DEFAULT 'holding',
    accum_period TEXT NOT NULL DEFAULT '',
    accum_type   TEXT NOT NULL DEFAULT '',
    accum_value  REAL NOT NULL DEFAULT 0,
    sector       TEXT DEFAULT '',
    notes        TEXT DEFAULT '',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);\
"""

_DDL_HISTORY = """\
CREATE TABLE IF NOT EXISTS purchase_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    holding_id  INTEGER NOT NULL,
    buy_date    TEXT NOT NULL,
    quantity    REAL NOT NULL,
    price       REAL NOT NULL,
    created_at  TEXT NOT NULL
);\
"""

_DDL_INDEXES = """\
CREATE INDEX IF NOT EXISTS idx_holdings_ticker  ON portfolio_holdings(ticker);
CREATE INDEX IF NOT EXISTS idx_holdings_group   ON portfolio_holdings(group_type);
CREATE INDEX IF NOT EXISTS idx_history_holding  ON purchase_history(holding_id);\
"""

# buy_date 컬럼 제거 + accum 컬럼 추가 마이그레이션 SQL
_MIGRATION_V1_V2 = """\
CREATE TABLE portfolio_holdings_v2 (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT NOT NULL,
    name         TEXT NOT NULL DEFAULT '',
    quantity     REAL NOT NULL DEFAULT 0,
    target_qty   REAL,
    avg_cost     REAL NOT NULL DEFAULT 0,
    group_type   TEXT NOT NULL DEFAULT 'holding',
    accum_period TEXT NOT NULL DEFAULT '',
    accum_type   TEXT NOT NULL DEFAULT '',
    accum_value  REAL NOT NULL DEFAULT 0,
    sector       TEXT DEFAULT '',
    notes        TEXT DEFAULT '',
    created_at   TEXT NOT NULL DEFAULT '',
    updated_at   TEXT NOT NULL DEFAULT ''
);
INSERT INTO portfolio_holdings_v2
    (id, ticker, name, quantity, target_qty, avg_cost, group_type,
     accum_period, accum_type, accum_value, sector, notes, created_at, updated_at)
SELECT id, ticker, name, quantity, target_qty, avg_cost, group_type,
       '', '', 0, sector, notes, created_at, updated_at
FROM portfolio_holdings;
DROP TABLE portfolio_holdings;
ALTER TABLE portfolio_holdings_v2 RENAME TO portfolio_holdings;\
"""


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
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── DB 초기화 / 마이그레이션 ─────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn() as conn:
            tbl_exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='portfolio_holdings'"
            ).fetchone()

            if tbl_exists:
                cols = {row[1] for row in conn.execute("PRAGMA table_info(portfolio_holdings)")}

                if "buy_date" in cols:
                    # v1 → v2 마이그레이션 (buy_date 제거, accum 컬럼 추가)
                    conn.executescript(_MIGRATION_V1_V2)
                else:
                    # 이미 v2 이상: 누락된 accum 컬럼만 추가 (멱등)
                    for col, default in [
                        ("accum_period", "''"),
                        ("accum_type",   "''"),
                        ("accum_value",  "0"),
                    ]:
                        if col not in cols:
                            conn.execute(
                                f"ALTER TABLE portfolio_holdings "
                                f"ADD COLUMN {col} TEXT NOT NULL DEFAULT {default}"
                            )
            else:
                conn.execute(_DDL_HOLDINGS)

            # purchase_history + 인덱스 (항상 멱등)
            conn.execute(_DDL_HISTORY)
            conn.executescript(_DDL_INDEXES)

    # ── CRUD: 보유 종목 ───────────────────────────────────────────────────────

    def add_holding(
        self,
        ticker: str,
        name: str = "",
        quantity: float = 0.0,
        avg_cost: float = 0.0,
        group_type: str = GROUP_HOLDING,
        target_qty: Optional[float] = None,
        accum_period: str = "",
        accum_type: str = "",
        accum_value: float = 0.0,
        sector: str = "",
        notes: str = "",
    ) -> int:
        now = datetime.now().isoformat()
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO portfolio_holdings
                    (ticker, name, quantity, target_qty, avg_cost, group_type,
                     accum_period, accum_type, accum_value,
                     sector, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ticker, name, quantity, target_qty, avg_cost, group_type,
                 accum_period, accum_type, accum_value,
                 sector, notes, now, now),
            )
            return int(cur.lastrowid)  # type: ignore[arg-type]

    def update_holding(self, holding_id: int, **kwargs: Any) -> None:
        allowed = {
            "ticker", "name", "quantity", "target_qty", "avg_cost",
            "group_type", "accum_period", "accum_type", "accum_value",
            "sector", "notes",
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
            conn.execute("DELETE FROM purchase_history WHERE holding_id=?", (holding_id,))
            conn.execute("DELETE FROM portfolio_holdings WHERE id=?", (holding_id,))

    def move_group(self, holding_id: int, new_group: str) -> None:
        self.update_holding(holding_id, group_type=new_group)

    # ── CRUD: 매수 내역 ───────────────────────────────────────────────────────

    def add_purchase(
        self,
        holding_id: int,
        buy_date: str,
        quantity: float,
        price: float,
    ) -> int:
        now = datetime.now().isoformat()
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO purchase_history
                    (holding_id, buy_date, quantity, price, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (holding_id, buy_date, quantity, price, now),
            )
            return int(cur.lastrowid)  # type: ignore[arg-type]

    def get_purchases(self, holding_id: int) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM purchase_history WHERE holding_id=? ORDER BY buy_date, id",
                (holding_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_purchase(self, purchase_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM purchase_history WHERE id=?", (purchase_id,))

    def calc_accumulated(self, holding_id: int) -> tuple[float, float]:
        """
        매수 내역 + 초기 보유량을 합산해 (총수량, 평균매입가) 반환.
        - 매수 내역이 없으면 holding의 quantity / avg_cost 그대로 반환
        - 매수 내역이 있으면 초기 보유량(seed)과 가중 평균 계산
        """
        h = self.get_by_id(holding_id)
        if not h:
            return 0.0, 0.0

        seed_qty  = float(h["quantity"])
        seed_cost = float(h["avg_cost"])
        purchases = self.get_purchases(holding_id)

        if not purchases:
            return seed_qty, seed_cost

        ph_qty   = sum(float(p["quantity"]) for p in purchases)
        ph_cost  = sum(float(p["quantity"]) * float(p["price"]) for p in purchases)

        total_qty  = seed_qty + ph_qty
        total_cost = seed_qty * seed_cost + ph_cost
        avg_cost   = total_cost / total_qty if total_qty > 0 else 0.0
        return total_qty, avg_cost

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
        Required: 종목코드, 수량, 평균매입가
        Optional: 종목명, 그룹, 목표수량, 적립주기, 적립방식, 적립금액, 섹터, 메모
        """
        reader = csv.DictReader(io.StringIO(csv_content))
        aliases: dict[str, list[str]] = {
            "ticker":       ["종목코드", "ticker", "Ticker", "종목"],
            "name":         ["종목명", "name", "Name"],
            "quantity":     ["수량", "quantity", "Quantity", "보유수량"],
            "avg_cost":     ["평균매입가", "avg_cost", "매입가", "평단가"],
            "group_type":   ["그룹", "group_type", "group"],
            "target_qty":   ["목표수량", "target_qty"],
            "accum_period": ["적립주기", "accum_period"],
            "accum_type":   ["적립방식", "accum_type"],
            "accum_value":  ["적립금액", "accum_value"],
            "sector":       ["섹터", "sector"],
            "notes":        ["메모", "notes"],
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

                qty      = float((_get(row, "quantity") or "0").replace(",", ""))
                avg_cost = float((_get(row, "avg_cost") or "0").replace(",", ""))

                raw_group = (_get(row, "group_type") or "holding").lower()
                group_type = (
                    GROUP_ACCUMULATING
                    if raw_group in ("모으는중", "모으는 중", "accumulating")
                    else GROUP_HOLDING
                )

                tgt_raw    = _get(row, "target_qty")
                target_qty = float(tgt_raw.replace(",", "")) if tgt_raw else None

                acc_val_raw = _get(row, "accum_value")
                acc_value   = float(acc_val_raw.replace(",", "")) if acc_val_raw else 0.0

                self.add_holding(
                    ticker=ticker,
                    name=_get(row, "name") or "",
                    quantity=qty,
                    avg_cost=avg_cost,
                    group_type=group_type,
                    target_qty=target_qty,
                    accum_period=_get(row, "accum_period") or "",
                    accum_type=_get(row, "accum_type") or "",
                    accum_value=acc_value,
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
        header = "종목코드,종목명,수량,평균매입가,그룹,목표수량,적립주기,적립방식,적립금액,메모"
        sample1 = "005930.KS,삼성전자,10,70000,보유 중,,,,,장기 보유"
        sample2 = "NVDA,,0,0,모으는 중,20,weekly,amount,100000,매주 10만원 적립"
        return f"{header}\n{sample1}\n{sample2}\n"
