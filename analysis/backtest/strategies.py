"""
사전 정의된 백테스팅 전략 모음.
각 전략은 OHLCV DataFrame → 신호 시리즈(1/0/-1) 를 반환한다.
지표 파라미터는 전략별로 독립적으로 설정된다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


# ── 전략 메타데이터 ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StrategyConfig:
    id:          str
    name:        str
    description: str
    params:      dict   # 지표 파라미터 (문서용)


STRATEGIES: dict[str, StrategyConfig] = {
    "rsi_reversal": StrategyConfig(
        id="rsi_reversal",
        name="A. RSI 역추세",
        description="RSI(14) < 30 매수 / RSI > 70 매도",
        params={"RSI 기간": 14, "과매도": 30, "과매수": 70},
    ),
    "macd_cross": StrategyConfig(
        id="macd_cross",
        name="B. MACD 크로스",
        description="MACD(12/26/9) 골든크로스 매수 / 데드크로스 매도",
        params={"fast": 12, "slow": 26, "signal": 9},
    ),
    "bb_reversal": StrategyConfig(
        id="bb_reversal",
        name="C. 볼린저밴드",
        description="BB(20) 하단 터치 매수 / 상단 터치 매도",
        params={"기간": 20, "표준편차": 2},
    ),
    "ma_cross": StrategyConfig(
        id="ma_cross",
        name="D. MA 크로스",
        description="MA5 > MA20 매수 / MA5 < MA20 매도",
        params={"단기 MA": 5, "장기 MA": 20},
    ),
    "scalping": StrategyConfig(
        id="scalping",
        name="E. 단타 전략",
        description="RSI(9)<40 또는 BB(10) 하단 매수 / RSI>60 또는 BB 상단 매도",
        params={"RSI 기간": 9, "RSI 기준": "40/60", "MACD": "5/13/5", "BB": 10},
    ),
}


# ── 지표 계산 헬퍼 (순수 함수) ────────────────────────────────────────────────

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int, slow: int, sig: int) -> tuple[pd.Series, pd.Series]:
    ema_f    = close.ewm(span=fast, adjust=False).mean()
    ema_s    = close.ewm(span=slow, adjust=False).mean()
    macd     = ema_f - ema_s
    macd_sig = macd.ewm(span=sig, adjust=False).mean()
    return macd, macd_sig


def _bb(close: pd.Series, period: int, n_std: float = 2.0) -> tuple[pd.Series, pd.Series]:
    mid   = close.rolling(period, min_periods=period).mean()
    sigma = close.rolling(period, min_periods=period).std()
    return mid + n_std * sigma, mid - n_std * sigma


# ── 전략별 신호 생성 ──────────────────────────────────────────────────────────

def _sig_rsi_reversal(df: pd.DataFrame) -> pd.Series:
    rsi = _rsi(df["Close"], 14)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[rsi < 30] = 1
    sig[rsi > 70] = -1
    return sig


def _sig_macd_cross(df: pd.DataFrame) -> pd.Series:
    macd, macd_sig = _macd(df["Close"], 12, 26, 9)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[macd > macd_sig] = 1
    sig[macd < macd_sig] = -1
    return sig


def _sig_bb_reversal(df: pd.DataFrame) -> pd.Series:
    close = df["Close"]
    upper, lower = _bb(close, 20)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[close < lower] = 1
    sig[close > upper] = -1
    return sig


def _sig_ma_cross(df: pd.DataFrame) -> pd.Series:
    close = df["Close"]
    ma5   = _sma(close, 5)
    ma20  = _sma(close, 20)
    sig   = pd.Series(0, index=df.index, dtype=int)
    sig[ma5 > ma20] = 1
    sig[ma5 < ma20] = -1
    return sig


def _sig_scalping(df: pd.DataFrame) -> pd.Series:
    close = df["Close"]
    rsi9           = _rsi(close, 9)
    macd, macd_sig = _macd(close, 5, 13, 5)
    upper, lower   = _bb(close, 10)

    buy_cond  = (rsi9 < 40) | (close <= lower)
    sell_cond = (rsi9 > 60) | (close >= upper) | (macd < macd_sig)

    sig = pd.Series(0, index=df.index, dtype=int)
    sig[buy_cond]              = 1
    sig[sell_cond]             = -1   # 매도가 매수보다 우선
    sig[buy_cond & sell_cond]  = 0    # 신호 충돌 시 관망
    return sig


# ── 공개 인터페이스 ───────────────────────────────────────────────────────────

_SIGNAL_FNS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "rsi_reversal": _sig_rsi_reversal,
    "macd_cross":   _sig_macd_cross,
    "bb_reversal":  _sig_bb_reversal,
    "ma_cross":     _sig_ma_cross,
    "scalping":     _sig_scalping,
}


def generate_signals(df: pd.DataFrame, strategy_id: str) -> pd.Series:
    """
    전략 ID에 따라 신호 시리즈를 반환한다.
    반환값: pd.Series with values 1(매수), -1(매도), 0(유지)
    """
    fn = _SIGNAL_FNS.get(strategy_id)
    if fn is None:
        raise ValueError(f"알 수 없는 전략 ID: {strategy_id!r}")
    return fn(df).fillna(0).astype(int)
