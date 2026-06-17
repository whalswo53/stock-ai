"""
Single-stock mean reversion analysis.

Does NOT require a pair.  Computes:
  - Rolling Z-score on the price series
  - ADF test for stationarity (is the series mean-reverting at all?)
  - Ornstein-Uhlenbeck half-life estimate
  - BUY / SELL / CLOSE / WAIT signal from Z-score thresholds
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller

from data.collectors.price_collector import PriceCollector


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class MeanReversionResult:
    ticker: str
    dates: pd.DatetimeIndex
    price: pd.Series
    zscore: pd.Series
    zscore_window: int
    adf_pvalue: float
    is_mean_reverting: bool    # adf_pvalue < 0.05
    half_life_days: float      # OU half-life; inf if not mean-reverting
    zscore_latest: float
    signal: str                # BUY / SELL / CLOSE / WAIT
    label: str


# ── Main class ────────────────────────────────────────────────────────────────

class MeanReversionAnalyzer:
    """
    Single-ticker mean reversion.

    Note on validity:
      Most equity prices are NOT stationary — they drift.  The ADF test will
      often fail to reject the unit-root null.  When is_mean_reverting=False
      the Z-score signal is still computed but should be treated as unreliable.
      The half-life gives an intuition for how useful the signal is in practice.
    """

    def __init__(
        self,
        period: str = "1y",
        zscore_window: int = 30,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ) -> None:
        self._collector = PriceCollector()
        self.period = period
        self.zscore_window = zscore_window
        self.entry_z = entry_z
        self.exit_z = exit_z

    # ── Public API ────────────────────────────────────────────────────────

    def fetch_price(self, ticker: str) -> pd.Series:
        df = self._collector.fetch(ticker, period=self.period)
        if df.empty:
            raise ValueError(f"'{ticker}' 가격 데이터를 가져올 수 없습니다.")
        price = df["Close"].rename(ticker).dropna()
        if len(price) < 60:
            raise ValueError(
                f"데이터가 {len(price)}일뿐입니다 — 최소 60일이 필요합니다."
            )
        return price

    def analyze(self, price: pd.Series) -> MeanReversionResult:
        # Rolling Z-score
        roll_mean = price.rolling(self.zscore_window, min_periods=self.zscore_window).mean()
        roll_std  = price.rolling(self.zscore_window, min_periods=self.zscore_window).std()
        zscore = (price - roll_mean) / roll_std.replace(0, np.nan)

        # ADF test on price level
        adf_stat, adf_pvalue, *_ = adfuller(price.values, autolag="AIC")
        is_mr = bool(adf_pvalue < 0.05)

        # OU half-life via lag-1 regression: Δy = α + β·y_{t-1} + ε
        y = price.values
        delta_y = np.diff(y)
        lag_y   = y[:-1]
        x = add_constant(lag_y)
        try:
            model = OLS(delta_y, x).fit()
            beta = float(model.params[1])
            half_life = -np.log(2) / beta if beta < 0 else float("inf")
        except Exception:
            half_life = float("inf")

        # Current signal
        z_valid = zscore.dropna()
        if z_valid.empty:
            z_now = float("nan")
            signal, label = "WAIT", "Z-score 계산 불가"
        else:
            z_now = float(z_valid.iloc[-1])
            signal, label = self._classify(z_now, str(price.name))

        return MeanReversionResult(
            ticker=str(price.name),
            dates=price.index,
            price=price,
            zscore=zscore,
            zscore_window=self.zscore_window,
            adf_pvalue=float(adf_pvalue),
            is_mean_reverting=is_mr,
            half_life_days=round(half_life, 1) if np.isfinite(half_life) else float("inf"),
            zscore_latest=z_now,
            signal=signal,
            label=label,
        )

    def run(self, ticker: str) -> MeanReversionResult:
        price = self.fetch_price(ticker)
        return self.analyze(price)

    # ── Signal classifier ─────────────────────────────────────────────────

    def _classify(self, z: float, ticker: str) -> tuple[str, str]:
        if z > self.entry_z:
            return "SELL", f"과매수 (Z={z:.2f}) — {ticker} SELL"
        if z < -self.entry_z:
            return "BUY",  f"과매도 (Z={z:.2f}) — {ticker} BUY"
        if abs(z) < self.exit_z:
            return "CLOSE", f"중립 복귀 (Z={z:.2f}) — 포지션 청산"
        return "WAIT", f"관망 (Z={z:.2f}) — 진입 조건 미충족"
