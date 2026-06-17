from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class AnalysisResult:
    ticker: str
    signal: str              # "BUY" | "SELL" | "HOLD"
    confidence: float        # 0.0 ~ 1.0
    reasons: list[str] = field(default_factory=list)
    report_md: str = ""
    price_target: Optional[float] = None
    sentiment: Optional[str] = None   # "positive" | "neutral" | "negative"


class AIModel(ABC):
    """Abstract interface — lets ClaudeAnalyst and iTransformer be swapped freely."""

    @abstractmethod
    def analyze(
        self,
        ticker: str,
        df: pd.DataFrame,
        context: dict,
    ) -> AnalysisResult:
        """
        df     — OHLCV + indicator DataFrame (output of TechnicalIndicators.compute)
        context — dict from UserMemory.get_user_context(ticker)
        """
