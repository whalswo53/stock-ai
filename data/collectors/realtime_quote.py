"""
Real-time / delayed quote lookup for dashboard price badges.

US tickers (bare symbols, e.g. AAPL):
  - FINNHUB_API_KEY 설정 시: Finnhub REST /quote → 실시간 시세
    (무료 티어, 60 req/min, 브로커 계좌 불필요)
  - 키가 없으면: yfinance 마지막 체결 시각(regularMarketTime) 기준으로
    지연 시간을 계산해 "지연 N분" 배지용 정보를 반환

KR tickers (.KS/.KQ): 시세 자체는 yfinance 일봉을 그대로 사용하고
배지를 표시하지 않으므로 이 모듈을 호출하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import requests

from config.settings import FINNHUB_API_KEY

_KST = ZoneInfo("Asia/Seoul")

# 마지막 체결이 이보다 오래됐으면 "지연 N분" 대신 "장 마감" 문구로 표시
_MARKET_CLOSED_MIN = 90.0


@dataclass
class QuoteInfo:
    price: Optional[float]           # 마지막 체결가 (없으면 None)
    last_trade_at: Optional[datetime]  # UTC
    delay_min: Optional[float]       # 현재 시각 - 마지막 체결 시각 (분)
    source: str                      # "finnhub" | "yfinance"
    is_realtime: bool


def is_us_ticker(ticker: str) -> bool:
    """접미사 없는 심볼만 미국 종목으로 취급 (지수 ^, 환율 =X 제외)."""
    t = ticker.strip().upper()
    return bool(t) and "." not in t and "^" not in t and "=" not in t


def get_quote(ticker: str) -> Optional[QuoteInfo]:
    """미국 종목의 현재 시세 + 지연 정보. 실패 시 None."""
    if not is_us_ticker(ticker):
        return None
    if FINNHUB_API_KEY:
        q = _finnhub_quote(ticker)
        if q is not None:
            return q
    return _yfinance_quote(ticker)


def badge_text(q: Optional[QuoteInfo]) -> str:
    """현재가 metric 아래 캡션으로 쓸 배지 문자열. 정보 없으면 ''."""
    if q is None:
        return ""
    if q.is_realtime:
        ts = f" · {q.last_trade_at.astimezone(_KST):%H:%M:%S} KST 체결" if q.last_trade_at else ""
        return f"🟢 **실시간 시세** (Finnhub{ts})"
    if q.delay_min is None:
        return "🕐 시세 출처: yfinance (지연 시간 확인 불가)"
    if q.delay_min <= _MARKET_CLOSED_MIN:
        return f"🕐 **지연 약 {q.delay_min:.0f}분** (yfinance) — 실시간 전환은 .env에 FINNHUB_API_KEY 설정"
    when = f"{q.last_trade_at.astimezone(_KST):%m/%d %H:%M} KST" if q.last_trade_at else "알 수 없음"
    return f"🌙 장 마감 — 마지막 체결 {when} (yfinance)"


# ── Sources ───────────────────────────────────────────────────────────────────

def _finnhub_quote(ticker: str) -> Optional[QuoteInfo]:
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": ticker.upper(), "token": FINNHUB_API_KEY},
            timeout=5,
        )
        r.raise_for_status()
        data = r.json()
        price = data.get("c")           # current price
        epoch = data.get("t")           # last trade epoch (s)
        if not price or price <= 0:
            return None
        last_at = (
            datetime.fromtimestamp(epoch, tz=timezone.utc) if epoch else None
        )
        delay = (
            max(0.0, (datetime.now(timezone.utc) - last_at).total_seconds() / 60.0)
            if last_at else None
        )
        return QuoteInfo(
            price=float(price),
            last_trade_at=last_at,
            delay_min=delay,
            source="finnhub",
            is_realtime=True,
        )
    except Exception:
        return None


def _yfinance_quote(ticker: str) -> Optional[QuoteInfo]:
    import yfinance as yf

    try:
        info = yf.Ticker(ticker).info
        price = info.get("regularMarketPrice")
        epoch = info.get("regularMarketTime")
        last_at = (
            datetime.fromtimestamp(epoch, tz=timezone.utc) if epoch else None
        )
        delay = (
            max(0.0, (datetime.now(timezone.utc) - last_at).total_seconds() / 60.0)
            if last_at else None
        )
        return QuoteInfo(
            price=float(price) if price else None,
            last_trade_at=last_at,
            delay_min=delay,
            source="yfinance",
            is_realtime=False,
        )
    except Exception:
        return None
