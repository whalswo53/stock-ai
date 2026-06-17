from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
STORAGE_DIR = BASE_DIR / "storage"
CACHE_DIR = STORAGE_DIR / "cache"
PORTFOLIO_DIR = STORAGE_DIR / "portfolio"

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")

# Yahoo Finance suffix by market
MARKET_SUFFIX: dict[str, str] = {
    "KOSPI": ".KS",
    "KOSDAQ": ".KQ",
    "NASDAQ": "",
    "NYSE": "",
}

# Benchmark tickers
BENCHMARK: dict[str, str] = {
    "KOSPI": "^KS11",
    "NASDAQ": "QQQ",
}

# Technical indicator parameters
MA_WINDOWS: list[int] = [5, 20, 60, 120]
RSI_PERIOD: int = 14
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
BB_PERIOD: int = 20
BB_STD: float = 2.0

DEFAULT_PERIOD: str = "1y"
DEFAULT_INTERVAL: str = "1d"
CACHE_TTL_HOURS: int = 1
