"""
Pure function that converts a single indicator row into a -1.0 ~ +1.0 score.
Positive = bullish (BUY), Negative = bearish (SELL).
"""

import numpy as np
import pandas as pd


def score(row: pd.Series) -> float:
    """
    Weighted composite of RSI, MACD cross, MA trend, and Bollinger Band position.
    Returns 0.0 when insufficient data is available.
    """
    components: list[tuple[float, float]] = []  # (value, weight)

    # ── RSI (weight 0.35) ────────────────────────────────────────────────────
    rsi = row.get("RSI")
    if rsi is not None and not pd.isna(rsi):
        # RSI 50 → 0, RSI 30 → +1, RSI 70 → -1; clipped to ±1
        rsi_score = float(np.clip((50.0 - rsi) / 20.0, -1.0, 1.0))
        components.append((rsi_score, 0.35))

    # ── MACD cross (weight 0.25) ──────────────────────────────────────────────
    macd = row.get("MACD")
    macd_sig = row.get("MACD_Signal")
    if (
        macd is not None and macd_sig is not None
        and not pd.isna(macd) and not pd.isna(macd_sig)
    ):
        macd_score = 1.0 if macd > macd_sig else -1.0
        components.append((macd_score, 0.25))

    # ── MA trend: MA5 vs MA20 (weight 0.20) ──────────────────────────────────
    ma5 = row.get("MA5")
    ma20 = row.get("MA20")
    close = row.get("Close")
    if (
        ma5 is not None and ma20 is not None and close is not None
        and not pd.isna(ma5) and not pd.isna(ma20)
        and close != 0
    ):
        # Percentage deviation between MA5 and MA20; ±5% maps to ±1
        ma_diff = (float(ma5) - float(ma20)) / float(close)
        ma_score = float(np.clip(ma_diff * 20.0, -1.0, 1.0))
        components.append((ma_score, 0.20))

    # ── Bollinger Band position (weight 0.20) ────────────────────────────────
    bb_upper = row.get("BB_Upper")
    bb_lower = row.get("BB_Lower")
    bb_mid = row.get("BB_Mid")
    if (
        bb_upper is not None and bb_lower is not None
        and bb_mid is not None and close is not None
        and not any(pd.isna(v) for v in [bb_upper, bb_lower, bb_mid])
    ):
        band_half = (float(bb_upper) - float(bb_lower)) / 2.0
        if band_half > 0:
            # +1 at bottom of band (cheap), -1 at top of band (expensive)
            bb_pos = (float(close) - float(bb_mid)) / band_half
            bb_score = float(np.clip(-bb_pos, -1.0, 1.0))
            components.append((bb_score, 0.20))

    if not components:
        return 0.0

    total_weight = sum(w for _, w in components)
    weighted_sum = sum(v * w for v, w in components)
    return float(np.clip(weighted_sum / total_weight, -1.0, 1.0))
