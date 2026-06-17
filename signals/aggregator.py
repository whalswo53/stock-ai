"""
Combines technical, AI, and sentiment signals into a single composite score.
Weights: technical 0.30 · AI 0.45 · sentiment 0.25
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from analysis.ai.base_model import AnalysisResult
from analysis.sentiment.classifier import SentimentResult


@dataclass
class CompositeSignal:
    ticker: str
    score: float            # -1.0 ~ +1.0
    signal: str             # "BUY" | "SELL" | "HOLD"
    confidence: float       # |score| as a 0~1 certainty proxy
    technical_score: float
    ai_score: float
    sentiment_score: float
    ai_result: Optional[AnalysisResult] = None
    sentiment_results: list[SentimentResult] = field(default_factory=list)


class SignalAggregator:
    WEIGHTS = {"technical": 0.30, "ai": 0.45, "sentiment": 0.25}

    # Threshold: score magnitude at which we issue BUY/SELL vs HOLD
    BUY_THRESHOLD = 0.15
    SELL_THRESHOLD = -0.15

    def aggregate(
        self,
        ticker: str,
        technical_score: float,
        ai_result: Optional[AnalysisResult] = None,
        sentiment_results: Optional[list[SentimentResult]] = None,
    ) -> CompositeSignal:
        s_sent = sentiment_results or []
        ai_s = self._ai_score(ai_result)
        sent_s = self._sentiment_score(s_sent)

        composite = (
            technical_score * self.WEIGHTS["technical"]
            + ai_s * self.WEIGHTS["ai"]
            + sent_s * self.WEIGHTS["sentiment"]
        )
        composite = max(-1.0, min(1.0, composite))

        if composite >= self.BUY_THRESHOLD:
            signal = "BUY"
        elif composite <= self.SELL_THRESHOLD:
            signal = "SELL"
        else:
            signal = "HOLD"

        return CompositeSignal(
            ticker=ticker,
            score=round(composite, 4),
            signal=signal,
            confidence=round(abs(composite), 4),
            technical_score=round(technical_score, 4),
            ai_score=round(ai_s, 4),
            sentiment_score=round(sent_s, 4),
            ai_result=ai_result,
            sentiment_results=s_sent,
        )

    # ── Score mappers ─────────────────────────────────────────────────────────

    @staticmethod
    def _ai_score(result: Optional[AnalysisResult]) -> float:
        if result is None:
            return 0.0
        if result.signal == "BUY":
            return result.confidence
        if result.signal == "SELL":
            return -result.confidence
        return 0.0

    @staticmethod
    def _sentiment_score(results: list[SentimentResult]) -> float:
        if not results:
            return 0.0
        return sum(r.score for r in results) / len(results)

    # ── Human-readable summary ────────────────────────────────────────────────

    def summary_text(self, sig: CompositeSignal) -> str:
        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(sig.signal, "⚪")
        lines = [
            f"{emoji} **{sig.signal}** (종합 점수: {sig.score:+.3f}, 확신도: {sig.confidence:.0%})",
            f"- 기술적: {sig.technical_score:+.3f}  |  AI: {sig.ai_score:+.3f}  |  감성: {sig.sentiment_score:+.3f}",
        ]
        if sig.ai_result and sig.ai_result.reasons:
            lines.append("\n**AI 분석 근거:**")
            for r in sig.ai_result.reasons:
                lines.append(f"  - {r}")
        if sig.ai_result and sig.ai_result.price_target:
            lines.append(f"\n**목표가:** {sig.ai_result.price_target:,.0f}")
        return "\n".join(lines)
