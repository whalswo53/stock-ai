import pandas as pd
import numpy as np

from config.settings import (
    MA_WINDOWS, RSI_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD,
)

VOL_RATIO_WINDOW = 20   # 거래량 급증 판정용 평균 윈도우
VOL_SPIKE_MULT = 2.0    # 평균 대비 이 배수 이상이면 급증
VWAP_WINDOW = 20        # 롤링 VWAP 윈도우 (일봉이라 장중 VWAP 불가 → 롤링 근사)


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

        out["Vol_Ratio"] = self._vol_ratio(volume, VOL_RATIO_WINDOW)
        out["VWAP"] = self._rolling_vwap(out, VWAP_WINDOW)

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

    @staticmethod
    def _vol_ratio(volume: pd.Series, window: int) -> pd.Series:
        """당일 거래량 ÷ 직전 window일 평균. VOL_SPIKE_MULT 이상 = 급증."""
        avg = volume.shift(1).rolling(window, min_periods=window).mean()
        return volume / avg.replace(0, np.nan)

    @staticmethod
    def _rolling_vwap(df: pd.DataFrame, window: int) -> pd.Series:
        """롤링 VWAP: Σ(대표가격×거래량)/Σ(거래량), 대표가격=(H+L+C)/3."""
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        pv = (tp * df["Volume"]).rolling(window, min_periods=window).sum()
        v = df["Volume"].rolling(window, min_periods=window).sum()
        return pv / v.replace(0, np.nan)
