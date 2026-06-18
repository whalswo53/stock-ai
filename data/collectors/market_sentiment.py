"""
VIX 기반 공포·탐욕 지수 (Fear & Greed Index).
CNN 공식 API 차단 시 CBOE VIX로 근사치 산출:
  VIX 낮음(10→) = 탐욕(→90점), VIX 높음(45+) = 공포(→10점 이하)
"""
from __future__ import annotations

from dataclasses import dataclass

import yfinance as yf


@dataclass
class FearGreedResult:
    score: int    # 0–100
    label: str    # 극도의 공포 / 공포 / 중립 / 탐욕 / 극도의 탐욕
    vix: float    # 현재 VIX 값 (데이터 없으면 -1)
    source: str   # 데이터 출처 설명


# ── Score helpers ──────────────────────────────────────────────────────────────

def _vix_to_score(vix: float) -> int:
    """VIX → 0~100 점수. VIX 10 → 90점, VIX 45 → 10점 (선형 변환)."""
    score = 100 - (vix - 10) / 35 * 80
    return max(0, min(100, int(round(score))))


def _score_to_label(score: int) -> str:
    if score <= 20:
        return "극도의 공포"
    if score <= 40:
        return "공포"
    if score <= 60:
        return "중립"
    if score <= 80:
        return "탐욕"
    return "극도의 탐욕"


def score_to_color(score: int) -> str:
    """점수 → 5단계 색상 (hex)."""
    if score <= 20:
        return "#c62828"   # 극도의 공포 — 진한 빨강
    if score <= 40:
        return "#e64a19"   # 공포 — 주황
    if score <= 60:
        return "#f9a825"   # 중립 — 노랑
    if score <= 80:
        return "#558b2f"   # 탐욕 — 초록
    return "#00695c"       # 극도의 탐욕 — 청록


# ── Main collector ─────────────────────────────────────────────────────────────

class MarketSentimentCollector:
    """VIX 지수를 조회해 공포·탐욕 점수를 반환한다."""

    def fetch(self) -> FearGreedResult:
        try:
            hist = yf.Ticker("^VIX").history(period="5d")
            if hist.empty:
                return self._neutral("VIX 데이터 없음")
            vix_val = float(hist["Close"].iloc[-1])
            score = _vix_to_score(vix_val)
            return FearGreedResult(
                score=score,
                label=_score_to_label(score),
                vix=round(vix_val, 2),
                source="CBOE VIX 기반 산출",
            )
        except Exception as exc:
            return self._neutral(f"오류: {exc}")

    @staticmethod
    def _neutral(reason: str) -> FearGreedResult:
        return FearGreedResult(score=50, label="중립", vix=-1.0, source=reason)


def prompt_snippet(result: FearGreedResult) -> str:
    """AI 프롬프트에 삽입할 한 줄 요약을 반환한다."""
    vix_txt = f" (VIX {result.vix:.1f})" if result.vix >= 0 else ""
    return (
        f"현재 시장 전반 분위기: {result.label} "
        f"(공포·탐욕 지수 {result.score}/100{vix_txt}, 출처: {result.source})"
    )
