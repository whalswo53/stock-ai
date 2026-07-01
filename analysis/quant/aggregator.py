"""
Unified quantitative arbitrage aggregator.

Pair mode:
  Runs OLS pairs trading + Kalman hedge in parallel.
  Weights are derived from objective confidence metrics:
    OLS weight  = R² × max(0, 1 − 2·p_value)
    Kalman weight = hedge_ratio_stability
  Normalized so they sum to 1.  Composite Z = weighted average.

Single mode:
  Delegates directly to MeanReversionAnalyzer (no second model to blend).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from analysis.quant.kalman_hedge import KalmanHedge, KalmanSpreadResult
from analysis.quant.mean_reversion import MeanReversionAnalyzer, MeanReversionResult
from analysis.quant.pairs_trading import (
    CointegrationResult,
    PairSignal,
    PairsTrading,
    SpreadResult,
)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class ModelContribution:
    name: str
    zscore: float
    weight: float              # 0–1, sums to 1 across contributions list
    signal_a: str
    signal_b: str
    confidence_label: str      # human-readable metric that drove the weight
    confidence_value: float    # raw 0–1 value


@dataclass
class AggregatedPairResult:
    ticker_a: str
    ticker_b: str
    composite_zscore: float
    signal_a: str
    signal_b: str
    label: str
    contributions: list[ModelContribution] = field(default_factory=list)
    # Raw results kept for charting
    coint_result: CointegrationResult | None = None
    ols_spread: SpreadResult | None = None
    kalman_spread: KalmanSpreadResult | None = None


# ── Aggregator ────────────────────────────────────────────────────────────────

class QuantAggregator:
    """
    Single entry point for all quant strategies.
    Use run_pair() for two tickers, run_single() for one.
    """

    def __init__(
        self,
        period: str = "1y",
        zscore_window: int = 30,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        kalman_delta: float = 1e-4,
        alpha: float = 0.05,
    ) -> None:
        self.entry_z = entry_z
        self.exit_z  = exit_z

        self._ols = PairsTrading(
            period=period,
            zscore_window=zscore_window,
            entry_z=entry_z,
            exit_z=exit_z,
            alpha=alpha,
        )
        self._kalman = KalmanHedge(
            period=period,
            zscore_window=zscore_window,
            entry_z=entry_z,
            exit_z=exit_z,
            delta=kalman_delta,
        )
        self._mr = MeanReversionAnalyzer(
            period=period,
            zscore_window=zscore_window,
            entry_z=entry_z,
            exit_z=exit_z,
            alpha=alpha,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def run_pair(self, ticker_a: str, ticker_b: str) -> AggregatedPairResult:
        # Fetch prices once, share across models
        price_a, price_b = self._ols.fetch_prices(ticker_a, ticker_b)

        # OLS model
        coint_result = self._ols.test_cointegration(price_a, price_b)
        ols_spread   = self._ols.compute_spread(price_a, price_b)
        ols_signal   = self._ols.generate_signal(ols_spread)

        # Kalman model (prices already aligned — pass directly)
        kalman_spread  = self._kalman.compute_spread(price_a, price_b)
        kalman_signal  = self._kalman.generate_signal(kalman_spread)

        # Weights
        ols_w, kalman_w = self._compute_weights(coint_result, ols_spread, kalman_spread)

        # Composite Z-score
        ols_z    = _safe_z(ols_spread.zscore)
        kalman_z = _safe_z(kalman_spread.zscore)
        composite_z = ols_w * ols_z + kalman_w * kalman_z

        # Final signal from composite
        final_signal = self._classify_pair(composite_z, ticker_a, ticker_b)

        contributions = [
            ModelContribution(
                name="OLS 페어트레이딩",
                zscore=round(ols_z, 4),
                weight=round(ols_w, 4),
                signal_a=ols_signal.signal_a,
                signal_b=ols_signal.signal_b,
                confidence_label=f"R²={ols_spread.rsquared:.3f}  p={coint_result.pvalue:.3f}",
                confidence_value=round(ols_spread.rsquared * max(0.0, 1 - 2 * coint_result.pvalue), 4),
            ),
            ModelContribution(
                name="칼만 필터",
                zscore=round(kalman_z, 4),
                weight=round(kalman_w, 4),
                signal_a=kalman_signal.signal_a,
                signal_b=kalman_signal.signal_b,
                confidence_label=f"헤지비율 안정성={kalman_spread.hedge_ratio_stability:.3f}",
                confidence_value=round(kalman_spread.hedge_ratio_stability, 4),
            ),
        ]

        return AggregatedPairResult(
            ticker_a=ticker_a,
            ticker_b=ticker_b,
            composite_zscore=round(composite_z, 4),
            signal_a=final_signal.signal_a,
            signal_b=final_signal.signal_b,
            label=final_signal.label,
            contributions=contributions,
            coint_result=coint_result,
            ols_spread=ols_spread,
            kalman_spread=kalman_spread,
        )

    def run_single(self, ticker: str) -> MeanReversionResult:
        return self._mr.run(ticker)

    # ── Internals ─────────────────────────────────────────────────────────

    def _compute_weights(
        self,
        coint: CointegrationResult,
        ols: SpreadResult,
        kalman: KalmanSpreadResult,
    ) -> tuple[float, float]:
        """
        OLS confidence  = R² × max(0, 1 − 2·p_value)
          (high when fit is good AND cointegration is strong)
        Kalman confidence = hedge_ratio_stability
          (high when the dynamic ratio is stable and trustworthy)

        Both are clipped to [0.05, 0.95] so neither model is silenced.
        """
        ols_conf    = float(np.clip(ols.rsquared * max(0.0, 1.0 - 2.0 * coint.pvalue), 0.05, 0.95))
        kalman_conf = float(np.clip(kalman.hedge_ratio_stability, 0.05, 0.95))

        total = ols_conf + kalman_conf
        return ols_conf / total, kalman_conf / total

    def _classify_pair(self, z: float, ticker_a: str, ticker_b: str) -> PairSignal:
        if np.isnan(z):
            return PairSignal(float("nan"), "WAIT", "WAIT", "Z-score 계산 불가")
        if z > self.entry_z:
            return PairSignal(z, "SELL", "BUY",
                f"스프레드 과대 (Z={z:.2f}) — {ticker_a} SELL / {ticker_b} BUY")
        if z < -self.entry_z:
            return PairSignal(z, "BUY", "SELL",
                f"스프레드 과소 (Z={z:.2f}) — {ticker_a} BUY / {ticker_b} SELL")
        if abs(z) < self.exit_z:
            return PairSignal(z, "CLOSE", "CLOSE",
                f"평균 회귀 완료 (Z={z:.2f}) — 포지션 청산")
        return PairSignal(z, "WAIT", "WAIT",
            f"관망 (Z={z:.2f}) — 진입 조건 미충족")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_z(zscore_series) -> float:
    """Returns the latest non-NaN Z-score, or 0.0 if none available."""
    valid = zscore_series.dropna()
    return float(valid.iloc[-1]) if not valid.empty else 0.0
