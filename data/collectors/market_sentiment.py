"""
CNN Fear & Greed Index 수집기.
fear-and-greed 패키지로 실제 CNN 데이터를 가져오고,
VIX는 보조 표시용으로 yfinance에서 별도 조회한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import fear_and_greed
import yfinance as yf


_DESC_TO_KR = {
    "extreme fear":  "극도의 공포",
    "fear":          "공포",
    "neutral":       "중립",
    "greed":         "탐욕",
    "extreme greed": "극도의 탐욕",
}


@dataclass
class FearGreedResult:
    score: int            # 0–100
    label: str            # 극도의 공포 / 공포 / 중립 / 탐욕 / 극도의 탐욕
    vix: float            # 현재 VIX 값 (조회 실패 시 -1)
    last_update: str      # 업데이트 시각 (ISO 문자열)
    source: str           # 데이터 출처


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


def _fetch_vix() -> float:
    """VIX 현재값 조회. 실패 시 -1."""
    try:
        hist = yf.Ticker("^VIX").history(period="5d")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 2)
    except Exception:
        pass
    return -1.0


class MarketSentimentCollector:
    """CNN Fear & Greed Index를 조회한다."""

    def fetch(self) -> FearGreedResult:
        vix = _fetch_vix()
        try:
            r = fear_and_greed.get()
            score = max(0, min(100, int(round(r.value))))
            label = _DESC_TO_KR.get(r.description, _score_to_label(score))
            upd = r.last_update.strftime("%Y-%m-%d %H:%M UTC") if isinstance(r.last_update, datetime) else str(r.last_update)
            return FearGreedResult(
                score=score,
                label=label,
                vix=vix,
                last_update=upd,
                source="CNN Fear & Greed Index",
            )
        except Exception as exc:
            score = 50
            return FearGreedResult(
                score=score,
                label=_score_to_label(score),
                vix=vix,
                last_update="",
                source=f"CNN 조회 실패 (오류: {exc})",
            )

    @staticmethod
    def _neutral(reason: str) -> FearGreedResult:
        return FearGreedResult(score=50, label="중립", vix=-1.0, last_update="", source=reason)


def prompt_snippet(result: FearGreedResult) -> str:
    """AI 프롬프트에 삽입할 한 줄 요약을 반환한다."""
    vix_txt = f", VIX {result.vix:.1f}" if result.vix >= 0 else ""
    upd_txt = f", 기준: {result.last_update}" if result.last_update else ""
    return (
        f"현재 시장 전반 분위기: {result.label} "
        f"(공포·탐욕 지수 {result.score}/100{vix_txt}{upd_txt}, 출처: {result.source})"
    )
