import yfinance as yf
import pandas as pd


class PriceCollector:
    """Fetches OHLCV price data via yfinance."""

    def fetch(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Returns OHLCV DataFrame indexed by Date.
        Returns empty DataFrame if ticker is invalid or network fails.
        """
        try:
            yf_ticker = yf.Ticker(ticker)
            df = yf_ticker.history(period=period, interval=interval)
        except Exception:
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        # Strip timezone so plotly/pandas comparisons stay simple
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "Date"

        out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        # Drop incomplete intraday rows (e.g. current trading day with no Close yet)
        return out.dropna(subset=["Close"])

    def get_info(self, ticker: str) -> dict:
        """Returns ticker metadata (company name, currency, sector, …)."""
        try:
            return yf.Ticker(ticker).info
        except Exception:
            return {}
