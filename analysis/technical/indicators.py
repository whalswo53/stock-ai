import pandas as pd
import numpy as np

from config.settings import (
    MA_WINDOWS, RSI_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD,
)


class TechnicalIndicators:
    """
    Computes technical indicators and appends them as columns to an OHLCV DataFrame.
    All methods are pure — the input DataFrame is never modified in place.
    """

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a new DataFrame with all indicator columns added.
        Input must have columns: Open, High, Low, Close, Volume.
        """
        out = df.copy()
        close = out["Close"]
        volume = out["Volume"]

        for w in MA_WINDOWS:
            out[f"MA{w}"] = self._sma(close, w)

        out["RSI"] = self._rsi(close, RSI_PERIOD)

        out["MACD"], out["MACD_Signal"], out["MACD_Hist"] = self._macd(
            close, MACD_FAST, MACD_SLOW, MACD_SIGNAL
        )

        out["BB_Upper"], out["BB_Mid"], out["BB_Lower"] = self._bollinger(
            close, BB_PERIOD, BB_STD
        )

        out["OBV"] = self._obv(close, volume)

        return out

    # ── Indicator implementations ─────────────────────────────────────────

    @staticmethod
    def _sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=window).mean()

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _macd(
        close: pd.Series, fast: int, slow: int, signal: int
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def _bollinger(
        close: pd.Series, period: int, n_std: float
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        mid = close.rolling(period, min_periods=period).mean()
        std = close.rolling(period, min_periods=period).std()
        return mid + n_std * std, mid, mid - n_std * std

    @staticmethod
    def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        direction = np.sign(close.diff()).fillna(0)
        return (direction * volume).cumsum()
