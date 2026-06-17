"""
Dynamic hedge ratio via Kalman filter.

State-space model (Chan 2013 style):
  Observation:  y_t = H_t @ state_t + ε_t    ε ~ N(0, R)
  Transition:   state_t = state_{t-1} + δ_t   δ ~ N(0, Q)

State vector: [β_t, α_t]  (hedge ratio, intercept — both time-varying)
Observation:  H_t = [price_b_t, 1]

delta controls adaptation speed:
  small delta (1e-5) → slow, stable ratio
  large delta (1e-3) → fast, reactive ratio
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.quant.pairs_trading import PairSignal
from data.collectors.price_collector import PriceCollector


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class KalmanSpreadResult:
    dates: pd.DatetimeIndex
    price_a: pd.Series
    price_b: pd.Series
    hedge_ratio: pd.Series       # time-varying β_t
    intercept: pd.Series         # time-varying α_t
    spread: pd.Series            # price_a − β_t·price_b − α_t
    zscore: pd.Series            # rolling Z-score of spread
    zscore_window: int
    hedge_ratio_stability: float  # 1 − CV(β); higher = more stable


# ── Main class ────────────────────────────────────────────────────────────────

class KalmanHedge:
    """
    Kalman-filter pairs trading.  Drop-in companion to PairsTrading —
    same fetch/generate_signal interface, but hedge_ratio is a pd.Series.
    """

    def __init__(
        self,
        period: str = "1y",
        zscore_window: int = 30,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        delta: float = 1e-4,
    ) -> None:
        self._collector = PriceCollector()
        self.period = period
        self.zscore_window = zscore_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.delta = delta

    # ── Public API ────────────────────────────────────────────────────────

    def fetch_prices(self, ticker_a: str, ticker_b: str) -> tuple[pd.Series, pd.Series]:
        df_a = self._collector.fetch(ticker_a, period=self.period)
        df_b = self._collector.fetch(ticker_b, period=self.period)
        if df_a.empty:
            raise ValueError(f"'{ticker_a}' 가격 데이터를 가져올 수 없습니다.")
        if df_b.empty:
            raise ValueError(f"'{ticker_b}' 가격 데이터를 가져올 수 없습니다.")

        combined = pd.concat(
            [df_a["Close"].rename(ticker_a), df_b["Close"].rename(ticker_b)], axis=1
        ).dropna()
        if len(combined) < 60:
            raise ValueError(
                f"공통 거래일이 {len(combined)}일뿐입니다 — 최소 60일이 필요합니다."
            )
        return combined[ticker_a], combined[ticker_b]

    def compute_spread(
        self, price_a: pd.Series, price_b: pd.Series
    ) -> KalmanSpreadResult:
        betas, alphas = self._run_filter(price_a.values, price_b.values)

        beta_s  = pd.Series(betas,  index=price_a.index, name="hedge_ratio")
        alpha_s = pd.Series(alphas, index=price_a.index, name="intercept")

        spread = price_a - beta_s * price_b - alpha_s

        roll_mean = spread.rolling(self.zscore_window, min_periods=self.zscore_window).mean()
        roll_std  = spread.rolling(self.zscore_window, min_periods=self.zscore_window).std()
        zscore = (spread - roll_mean) / roll_std.replace(0, np.nan)

        # Stability: low coefficient-of-variation → stable hedge ratio
        mean_beta = np.mean(np.abs(betas))
        cv = np.std(betas) / mean_beta if mean_beta > 1e-9 else 1.0
        stability = float(max(0.0, 1.0 - cv))

        return KalmanSpreadResult(
            dates=price_a.index,
            price_a=price_a,
            price_b=price_b,
            hedge_ratio=beta_s,
            intercept=alpha_s,
            spread=spread,
            zscore=zscore,
            zscore_window=self.zscore_window,
            hedge_ratio_stability=stability,
        )

    def generate_signal(self, result: KalmanSpreadResult) -> PairSignal:
        z = result.zscore.dropna()
        if z.empty:
            return PairSignal(
                zscore_latest=float("nan"),
                signal_a="WAIT",
                signal_b="WAIT",
                label="Z-score 계산 불가 (데이터 부족)",
            )

        z_now = float(z.iloc[-1])
        ticker_a = str(result.price_a.name)
        ticker_b = str(result.price_b.name)

        if z_now > self.entry_z:
            return PairSignal(
                zscore_latest=z_now,
                signal_a="SELL",
                signal_b="BUY",
                label=f"스프레드 과대 (Z={z_now:.2f}) — {ticker_a} SELL / {ticker_b} BUY",
            )
        if z_now < -self.entry_z:
            return PairSignal(
                zscore_latest=z_now,
                signal_a="BUY",
                signal_b="SELL",
                label=f"스프레드 과소 (Z={z_now:.2f}) — {ticker_a} BUY / {ticker_b} SELL",
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
    ) -> tuple[KalmanSpreadResult, PairSignal]:
        price_a, price_b = self.fetch_prices(ticker_a, ticker_b)
        result = self.compute_spread(price_a, price_b)
        signal = self.generate_signal(result)
        return result, signal

    # ── Kalman filter core ────────────────────────────────────────────────

    def _run_filter(
        self, price_a: np.ndarray, price_b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Runs the Kalman filter and returns (beta_series, alpha_series).
        Implemented in pure numpy — no pykalman dependency at runtime.
        """
        n = len(price_a)

        # Process noise: Q = delta/(1-delta) * I
        # Small delta → smoother, slower adaptation
        q = self.delta / (1.0 - self.delta)
        Q = np.array([[q, 0.0], [0.0, q]])

        # Measurement noise: variance of price_a scaled to spread scale
        R = float(np.var(price_a))

        # Initialise state with OLS on first 20 bars
        warmup = min(20, n)
        X0 = np.column_stack([price_b[:warmup], np.ones(warmup)])
        try:
            state, *_ = np.linalg.lstsq(X0, price_a[:warmup], rcond=None)
        except np.linalg.LinAlgError:
            state = np.array([1.0, 0.0])

        P = np.eye(2) * 1e4   # large initial uncertainty

        betas  = np.empty(n)
        alphas = np.empty(n)

        for t in range(n):
            h = np.array([price_b[t], 1.0])      # (2,) observation vector

            # Predict
            P_pred = P + Q

            # Update
            innovation = price_a[t] - float(h @ state)
            S = float(h @ P_pred @ h) + R         # scalar
            K = (P_pred @ h) / S                  # (2,) Kalman gain
            state = state + K * innovation
            P = (np.eye(2) - np.outer(K, h)) @ P_pred

            betas[t]  = state[0]
            alphas[t] = state[1]

        return betas, alphas
