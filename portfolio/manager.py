"""
Portfolio holdings manager — Supabase (PostgreSQL) 백엔드.

v3 변경: SQLite → Supabase
  - 기존 SQLite 코드는 주석으로 보존 (롤백 대비)
  - 메서드 인터페이스 동일 유지
  - 테이블 DDL: portfolio/schema.sql 참고
"""
from __future__ import annotations

import calendar
import csv
import io
import os
import time
from datetime import date, datetime, timedelta
from typing import Any, Optional

from supabase import Client, create_client

# dotenv 로딩 보장 (config/settings.py 에서 load_dotenv() 호출)
from config.settings import STORAGE_DIR  # noqa: F401  (side-effect import)

# ── SQLite 레거시 (롤백 시 이 블록 복원) ──────────────────────────────────────
# import sqlite3
# from contextlib import contextmanager
# from pathlib import Path
# from typing import Generator
#
# DB_PATH = STORAGE_DIR / "portfolio.db"
#
# _DDL_HOLDINGS = """\
# CREATE TABLE IF NOT EXISTS portfolio_holdings (
#     id           INTEGER PRIMARY KEY AUTOINCREMENT,
#     ticker       TEXT NOT NULL,
#     name         TEXT NOT NULL DEFAULT '',
#     quantity     REAL NOT NULL DEFAULT 0,
#     target_qty   REAL,
#     avg_cost     REAL NOT NULL DEFAULT 0,
#     group_type   TEXT NOT NULL DEFAULT 'holding',
#     accum_period TEXT NOT NULL DEFAULT '',
#     accum_type   TEXT NOT NULL DEFAULT '',
#     accum_value  REAL NOT NULL DEFAULT 0,
#     sector       TEXT DEFAULT '',
#     notes        TEXT DEFAULT '',
#     created_at   TEXT NOT NULL,
#     updated_at   TEXT NOT NULL
# );\
# """
#
# _DDL_HISTORY = """\
# CREATE TABLE IF NOT EXISTS purchase_history (
#     id          INTEGER PRIMARY KEY AUTOINCREMENT,
#     holding_id  INTEGER NOT NULL,
#     buy_date    TEXT NOT NULL,
#     quantity    REAL NOT NULL,
#     price       REAL NOT NULL,
#     created_at  TEXT NOT NULL
# );\
# """
#
# _DDL_INDEXES = """\
# CREATE INDEX IF NOT EXISTS idx_holdings_ticker  ON portfolio_holdings(ticker);
# CREATE INDEX IF NOT EXISTS idx_holdings_group   ON portfolio_holdings(group_type);
# CREATE INDEX IF NOT EXISTS idx_history_holding  ON purchase_history(holding_id);\
# """
#
# _MIGRATION_V1_V2 = """\
# CREATE TABLE portfolio_holdings_v2 (
#     id           INTEGER PRIMARY KEY AUTOINCREMENT,
#     ticker       TEXT NOT NULL,
#     name         TEXT NOT NULL DEFAULT '',
#     quantity     REAL NOT NULL DEFAULT 0,
#     target_qty   REAL,
#     avg_cost     REAL NOT NULL DEFAULT 0,
#     group_type   TEXT NOT NULL DEFAULT 'holding',
#     accum_period TEXT NOT NULL DEFAULT '',
#     accum_type   TEXT NOT NULL DEFAULT '',
#     accum_value  REAL NOT NULL DEFAULT 0,
#     sector       TEXT DEFAULT '',
#     notes        TEXT DEFAULT '',
#     created_at   TEXT NOT NULL DEFAULT '',
#     updated_at   TEXT NOT NULL DEFAULT ''
# );
# INSERT INTO portfolio_holdings_v2
#     (id, ticker, name, quantity, target_qty, avg_cost, group_type,
#      accum_period, accum_type, accum_value, sector, notes, created_at, updated_at)
# SELECT id, ticker, name, quantity, target_qty, avg_cost, group_type,
#        '', '', 0, sector, notes, created_at, updated_at
# FROM portfolio_holdings;
# DROP TABLE portfolio_holdings;
# ALTER TABLE portfolio_holdings_v2 RENAME TO portfolio_holdings;\
# """
# ─────────────────────────────────────────────────────────────────────────────

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


def _accum_dates(start: date, end: date, period: str, ref_date: date) -> list[date]:
    """적립 주기에 맞는 날짜 목록 반환 (start 이상 end 이하).

    - daily  : 영업일(월~금)
    - weekly : ref_date 와 같은 요일
    - monthly: ref_date 와 같은 일 (월말 초과 시 말일로 클램프)
    """
    result: list[date] = []

    if period == "daily":
        d = start
        while d <= end:
            if d.weekday() < 5:
                result.append(d)
            d += timedelta(days=1)

    elif period == "weekly":
        target_wd  = ref_date.weekday()
        days_ahead = (target_wd - start.weekday()) % 7
        d = start + timedelta(days=days_ahead)
        while d <= end:
            result.append(d)
            d += timedelta(weeks=1)

    elif period == "monthly":
        target_day  = ref_date.day
        month_cursor = start.replace(day=1)
        while month_cursor <= end:
            last_day  = calendar.monthrange(month_cursor.year, month_cursor.month)[1]
            effective = min(target_day, last_day)
            candidate = month_cursor.replace(day=effective)
            if start <= candidate <= end:
                result.append(candidate)
            if month_cursor.month == 12:
                month_cursor = month_cursor.replace(year=month_cursor.year + 1, month=1)
            else:
                month_cursor = month_cursor.replace(month=month_cursor.month + 1)

    return result


def _make_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    # Streamlit Cloud: secrets.toml fallback
    if not url or not key:
        try:
            import streamlit as st
            url = url or st.secrets.get("SUPABASE_URL")
            key = key or st.secrets.get("SUPABASE_SERVICE_KEY")
        except Exception:
            pass

    if not url:
        raise RuntimeError(
            "SUPABASE_URL이 설정되지 않았습니다. "
            ".env 파일 또는 Streamlit secrets (secrets.toml)를 확인하세요."
        )
    if not key:
        raise RuntimeError(
            "SUPABASE_SERVICE_KEY가 설정되지 않았습니다. "
            ".env 파일 또는 Streamlit secrets (secrets.toml)를 확인하세요."
        )

    return create_client(url, key)


class PortfolioManager:
    def __init__(self) -> None:
        # SQLite 롤백 시: self.db_path = db_path; self.db_path.parent.mkdir(...); self._init_db()
        self._db: Client = _make_client()

    # ── SQLite 롤백 시: _conn / _init_db 메서드 복원 ──────────────────────────
    # @contextmanager
    # def _conn(self) -> Generator[sqlite3.Connection, None, None]:
    #     conn = sqlite3.connect(self.db_path)
    #     conn.row_factory = sqlite3.Row
    #     conn.execute("PRAGMA journal_mode=WAL")
    #     conn.execute("PRAGMA foreign_keys=ON")
    #     try:
    #         yield conn
    #         conn.commit()
    #     except Exception:
    #         conn.rollback()
    #         raise
    #     finally:
    #         conn.close()
    #
    # def _init_db(self) -> None:
    #     with self._conn() as conn:
    #         tbl_exists = conn.execute(
    #             "SELECT 1 FROM sqlite_master WHERE type='table' AND name='portfolio_holdings'"
    #         ).fetchone()
    #         if tbl_exists:
    #             cols = {row[1] for row in conn.execute("PRAGMA table_info(portfolio_holdings)")}
    #             if "buy_date" in cols:
    #                 conn.executescript(_MIGRATION_V1_V2)
    #             else:
    #                 for col, default in [
    #                     ("accum_period", "''"),
    #                     ("accum_type",   "''"),
    #                     ("accum_value",  "0"),
    #                 ]:
    #                     if col not in cols:
    #                         conn.execute(
    #                             f"ALTER TABLE portfolio_holdings "
    #                             f"ADD COLUMN {col} TEXT NOT NULL DEFAULT {default}"
    #                         )
    #         else:
    #             conn.execute(_DDL_HOLDINGS)
    #         conn.execute(_DDL_HISTORY)
    #         conn.executescript(_DDL_INDEXES)
    # ─────────────────────────────────────────────────────────────────────────

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
        accum_currency: str = "KRW",
        sector: str = "",
        notes: str = "",
    ) -> int:
        now = datetime.now().isoformat()
        res = (
            self._db.table("portfolio_holdings")
            .insert({
                "ticker": ticker,
                "name": name,
                "quantity": quantity,
                "target_qty": target_qty,
                "avg_cost": avg_cost,
                "group_type": group_type,
                "accum_period": accum_period,
                "accum_type": accum_type,
                "accum_value": accum_value,
                "accum_currency": accum_currency,
                "sector": sector,
                "notes": notes,
                "created_at": now,
                "updated_at": now,
            })
            .execute()
        )
        return int(res.data[0]["id"])

    def update_holding(self, holding_id: int, **kwargs: Any) -> None:
        allowed = {
            "ticker", "name", "quantity", "target_qty", "avg_cost",
            "group_type", "accum_period", "accum_type", "accum_value",
            "accum_currency", "sector", "notes",
        }
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return
        fields["updated_at"] = datetime.now().isoformat()
        self._db.table("portfolio_holdings").update(fields).eq("id", holding_id).execute()

    def delete_holding(self, holding_id: int) -> None:
        # purchase_history는 ON DELETE CASCADE로 자동 삭제되지만 명시적으로 처리
        self._db.table("purchase_history").delete().eq("holding_id", holding_id).execute()
        self._db.table("portfolio_holdings").delete().eq("id", holding_id).execute()

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
        res = (
            self._db.table("purchase_history")
            .insert({
                "holding_id": holding_id,
                "buy_date": buy_date,
                "quantity": quantity,
                "price": price,
                "created_at": now,
            })
            .execute()
        )
        return int(res.data[0]["id"])

    def get_purchases(self, holding_id: int) -> list[dict]:
        res = (
            self._db.table("purchase_history")
            .select("*")
            .eq("holding_id", holding_id)
            .order("buy_date")
            .order("id")
            .execute()
        )
        return res.data

    def delete_purchase(self, purchase_id: int) -> None:
        self._db.table("purchase_history").delete().eq("id", purchase_id).execute()

    def calc_accumulated(self, holding_id: int) -> tuple[float, float]:
        """
        매수 내역 + 초기 보유량을 합산해 (총수량, 평균매입가) 반환.
        매수 내역이 없으면 holding의 quantity / avg_cost 그대로 반환.
        """
        h = self.get_by_id(holding_id)
        if not h:
            return 0.0, 0.0

        seed_qty  = float(h["quantity"])
        seed_cost = float(h["avg_cost"])
        purchases = self.get_purchases(holding_id)

        if not purchases:
            return seed_qty, seed_cost

        ph_qty  = sum(float(p["quantity"]) for p in purchases)
        ph_cost = sum(float(p["quantity"]) * float(p["price"]) for p in purchases)

        total_qty  = seed_qty + ph_qty
        total_cost = seed_qty * seed_cost + ph_cost
        avg_cost   = total_cost / total_qty if total_qty > 0 else 0.0
        return total_qty, avg_cost

    def auto_record_purchases(
        self,
        usd_krw: float = 1300.0,
    ) -> list[tuple[str, str, int]]:
        """모으는 중 종목의 자동 적립 매수 내역 생성.

        마지막 매수일(또는 등록일)부터 오늘까지 적립 주기에 맞는 날짜에
        yfinance 종가 기준으로 purchase_history 레코드를 자동 생성.
        해당 날짜에 이미 내역이 있으면 건너뜀(멱등).

        통화 처리:
          - accum_currency=KRW + 미국 종목 → 금액을 usd_krw 로 나눠 USD 환산 후 수량 계산
          - accum_currency=USD              → 금액 그대로 USD 로 수량 계산

        Returns:
            [(종목명, 티커, 추가건수), ...]  1건 이상 추가된 종목만 포함
        """
        import pandas as pd
        import yfinance as yf

        results: list[tuple[str, str, int]] = []

        for h in self.get_by_group(GROUP_ACCUMULATING):
            accum_period = (h.get("accum_period") or "").strip()
            accum_type   = (h.get("accum_type")   or "").strip()
            accum_value  = float(h.get("accum_value") or 0)

            if not (accum_period and accum_type and accum_value > 0):
                continue

            hid      = int(h["id"])
            ticker   = h["ticker"]
            name     = h.get("name") or ticker
            currency = h.get("accum_currency") or "KRW"
            is_us    = not ticker.upper().endswith((".KS", ".KQ"))

            ref_date = date.fromisoformat(h["created_at"][:10])

            purchases      = self.get_purchases(hid)
            existing_dates = {p["buy_date"] for p in purchases}

            if purchases:
                last = date.fromisoformat(max(p["buy_date"] for p in purchases))
                start = last + timedelta(days=1)
            else:
                start = ref_date

            today = date.today()
            if start > today:
                continue

            target_dates = _accum_dates(start, today, accum_period, ref_date)
            new_dates    = [d for d in target_dates if str(d) not in existing_dates]
            if not new_dates:
                continue

            # ── 가격 데이터 다운로드 ─────────────────────────────────────────
            try:
                raw = yf.download(
                    ticker,
                    start=str(start),
                    end=str(today + timedelta(days=1)),
                    auto_adjust=True,
                    progress=False,
                )
                if raw.empty:
                    continue
                close_s = (
                    raw["Close"].iloc[:, 0]
                    if isinstance(raw.columns, pd.MultiIndex)
                    else raw["Close"]
                )
                close_dict: dict[str, float] = {
                    idx.strftime("%Y-%m-%d"): float(val)
                    for idx, val in close_s.items()
                    if not pd.isna(val)
                }
            except Exception:
                continue

            # ── 날짜별 매수 추가 ─────────────────────────────────────────────
            n_added = 0
            for d in new_dates:
                price = close_dict.get(str(d))
                if not price or price <= 0:
                    continue  # 해당일 거래 없음(휴장 등) → 건너뜀

                if accum_type == "amount":
                    amount_usd = (accum_value / usd_krw) if (currency == "KRW" and is_us) else accum_value
                    qty = amount_usd / price
                else:
                    qty = accum_value

                if qty <= 0:
                    continue

                try:
                    self.add_purchase(
                        holding_id=hid,
                        buy_date=str(d),
                        quantity=round(qty, 6),
                        price=price,
                    )
                    n_added += 1
                except Exception:
                    pass

            if n_added > 0:
                results.append((name, ticker, n_added))

        return results

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_all(self) -> list[dict]:
        res = (
            self._db.table("portfolio_holdings")
            .select("*")
            .order("group_type")
            .order("created_at")
            .execute()
        )
        return res.data

    def get_by_group(self, group_type: str) -> list[dict]:
        res = (
            self._db.table("portfolio_holdings")
            .select("*")
            .eq("group_type", group_type)
            .order("created_at")
            .execute()
        )
        return res.data

    def get_by_id(self, holding_id: int) -> Optional[dict]:
        res = (
            self._db.table("portfolio_holdings")
            .select("*")
            .eq("id", holding_id)
            .execute()
        )
        return res.data[0] if res.data else None

    def count(self) -> int:
        try:
            from postgrest.exceptions import APIError
        except ImportError:
            APIError = Exception  # type: ignore[misc,assignment]
        try:
            res = self._db.table("portfolio_holdings").select("id", count="exact").execute()
            return res.count or 0
        except APIError as exc:
            raise RuntimeError(
                f"portfolio_holdings 테이블 조회 실패.\n"
                f"원인: {exc}\n"
                f"확인사항: ① schema.sql 실행 여부  "
                f"② SUPABASE_URL/SERVICE_KEY 정확성  "
                f"③ Supabase 프로젝트 활성 상태"
            ) from exc

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
    def csv_template() -> bytes:
        header  = "종목코드,종목명,수량,평균매입가,그룹,목표수량,적립주기,적립방식,적립금액,메모"
        sample1 = "005930.KS,삼성전자,10,70000,보유 중,,,,,장기 보유"
        sample2 = "NVDA,,0,0,모으는 중,20,weekly,amount,100000,매주 10만원 적립"
        return f"{header}\n{sample1}\n{sample2}\n".encode("utf-8-sig")
