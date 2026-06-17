"""
Statistical arbitrage via pairs trading.

Workflow:
  1. Fetch price history for two tickers.
  2. Run Engle-Granger cointegration test.
  3. Compute OLS hedge ratio → spread → rolling Z-score.
  4. Emit trade signals from Z-score thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import coint

from data.collectors.price_collector import PriceCollector


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class CointegrationResult:
    ticker_a: str
    ticker_b: str
    pvalue: float
    is_cointegrated: bool          # pvalue < 0.05
    test_stat: float
    critical_values: dict[str, float] = field(default_factory=dict)


@dataclass
class SpreadResult:
    dates: pd.DatetimeIndex
    price_a: pd.Series
    price_b: pd.Series
    hedge_ratio: float             # OLS beta: A ≈ β·B + ε
    spread: pd.Series              # price_a − hedge_ratio × price_b
    zscore: pd.Series              # rolling Z-score of spread
    zscore_window: int
    rsquared: float = 0.0          # OLS R² (goodness-of-fit)


@dataclass
class PairSignal:
    zscore_latest: float
    signal_a: str                  # BUY / SELL / CLOSE / WAIT
    signal_b: str
    label: str                     # human-readable summary


# ── Main class ────────────────────────────────────────────────────────────────

class PairsTrading:
    """
    End-to-end pairs trading analysis for two tickers.

    Thresholds (configurable):
        entry_z  = 2.0  — open position when |Z| crosses this
        exit_z   = 0.5  — close position when |Z| falls below this
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

    def fetch_prices(self, ticker_a: str, ticker_b: str) -> tuple[pd.Series, pd.Series]:
        """
        Returns aligned Close price series for both tickers.
        Raises ValueError if either ticker returns no data.
        """
        df_a = self._collector.fetch(ticker_a, period=self.period)
        df_b = self._collector.fetch(ticker_b, period=self.period)

        if df_a.empty:
            raise ValueError(f"'{ticker_a}' 가격 데이터를 가져올 수 없습니다.")
        if df_b.empty:
            raise ValueError(f"'{ticker_b}' 가격 데이터를 가져올 수 없습니다.")

        price_a = df_a["Close"].rename(ticker_a)
        price_b = df_b["Close"].rename(ticker_b)

        # Inner-join on dates so both series are aligned
        combined = pd.concat([price_a, price_b], axis=1).dropna()
        if len(combined) < 60:
            raise ValueError(
                f"공통 거래일이 {len(combined)}일뿐입니다 — 최소 60일이 필요합니다."
            )

        return combined[ticker_a], combined[ticker_b]

    def test_cointegration(
        self, price_a: pd.Series, price_b: pd.Series
    ) -> CointegrationResult:
        """Engle-Granger cointegration test (statsmodels.tsa.stattools.coint)."""
        stat, pvalue, crit = coint(price_a.values, price_b.values)
        return CointegrationResult(
            ticker_a=str(price_a.name),
            ticker_b=str(price_b.name),
            pvalue=float(pvalue),
            is_cointegrated=pvalue < 0.05,
            test_stat=float(stat),
            critical_values={
                "1%": float(crit[0]),
                "5%": float(crit[1]),
                "10%": float(crit[2]),
            },
        )

    def compute_spread(
        self, price_a: pd.Series, price_b: pd.Series
    ) -> SpreadResult:
        """
        OLS: price_a ~ β·price_b + intercept
        spread = price_a − β·price_b
        zscore  = (spread − rolling_mean) / rolling_std
        """
        x = add_constant(price_b.values)
        model = OLS(price_a.values, x).fit()
        hedge_ratio = float(model.params[1])

        spread = price_a - hedge_ratio * price_b

        roll_mean = spread.rolling(self.zscore_window, min_periods=self.zscore_window).mean()
        roll_std  = spread.rolling(self.zscore_window, min_periods=self.zscore_window).std()
        zscore = (spread - roll_mean) / roll_std.replace(0, np.nan)

        return SpreadResult(
            dates=price_a.index,
            price_a=price_a,
            price_b=price_b,
            hedge_ratio=hedge_ratio,
            spread=spread,
            zscore=zscore,
            zscore_window=self.zscore_window,
            rsquared=float(model.rsquared),
        )

    def generate_signal(self, spread_result: SpreadResult) -> PairSignal:
        """Derives the current trade signal from the latest Z-score."""
        z = spread_result.zscore.dropna()
        if z.empty:
            return PairSignal(
                zscore_latest=float("nan"),
                signal_a="WAIT",
                signal_b="WAIT",
                label="Z-score 계산 불가 (데이터 부족)",
            )

        z_now = float(z.iloc[-1])
        ticker_a = str(spread_result.price_a.name)
        ticker_b = str(spread_result.price_b.name)

        if z_now > self.entry_z:
            # Spread too wide — A overpriced vs B
            return PairSignal(
                zscore_latest=z_now,
                signal_a="SELL",
                signal_b="BUY",
                label=(
                    f"스프레드 과대 (Z={z_now:.2f}) — "
                    f"{ticker_a} SELL / {ticker_b} BUY"
                ),
            )
        if z_now < -self.entry_z:
            # Spread too narrow — A underpriced vs B
            return PairSignal(
                zscore_latest=z_now,
                signal_a="BUY",
                signal_b="SELL",
                label=(
                    f"스프레드 과소 (Z={z_now:.2f}) — "
                    f"{ticker_a} BUY / {ticker_b} SELL"
                ),
            )
        if abs(z_now) < self.exit_z:
            return PairSignal(
                zscore_latest=z_now,
                signal_a="CLOSE",
                signal_b="CLOSE",
                label=f"평균 회귀 완료 (Z={z_now:.2f}) — 포지션 청산",
            )

        return PairSignal(
            zscore_latest=z_now,
            signal_a="WAIT",
            signal_b="WAIT",
            label=f"관망 (Z={z_now:.2f}) — 진입 조건 미충족",
        )

    def run(
        self, ticker_a: str, ticker_b: str
    ) -> tuple[CointegrationResult, SpreadResult, PairSignal]:
        """Convenience method: fetch → test → spread → signal."""
        price_a, price_b = self.fetch_prices(ticker_a, ticker_b)
        coint_result = self.test_cointegration(price_a, price_b)
        spread_result = self.compute_spread(price_a, price_b)
        signal = self.generate_signal(spread_result)
        return coint_result, spread_result, signal
