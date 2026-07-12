"""
분석 모듈 레지스트리.

종합분석 페이지는 여기 등록된 모듈을 전부 순회(run_all)하기만 한다 —
개별 모듈을 손으로 골라 배선하지 않으므로, 새 분석에 @register만 붙이면
종합분석에 자동으로 나타난다. 즉 "종합분석 ≡ 등록된 모든 모듈의 합집합"이
구조적으로 보장된다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

ANALYSES: list["AnalysisModule"] = []


@dataclass
class AnalysisResult:
    title: str
    markdown: str                                   # AI 프롬프트/텍스트 표시용 — 항상 채워야 함
    json: dict = field(default_factory=dict)         # 대시보드 JSON 병합용
    render: Callable[[], None] | None = None         # 선택: 차트·게이지 등 리치 UI. 없으면 markdown만 표시
    polarity: str | None = None                      # "bullish" | "neutral" | "bearish" | None(집계 제외)


@dataclass
class AnalysisModule:
    name: str
    fn: Callable[[dict], AnalysisResult]
    order: int = 100      # 종합분석 표시 순서


def register(name: str, order: int = 100):
    def deco(fn):
        ANALYSES.append(AnalysisModule(name=name, fn=fn, order=order))
        return fn
    return deco


def run_all(ctx: dict) -> list[AnalysisResult]:
    out = []
    for m in sorted(ANALYSES, key=lambda x: x.order):
        try:
            out.append(m.fn(ctx))
        except Exception as e:
            out.append(AnalysisResult(title=m.name, markdown=f"⚠️ {m.name} 실패: {e}"))
    return out


def aggregate_verdict(results: list[AnalysisResult]) -> dict:
    """등록 모듈들의 polarity를 다수결로 집계해 종합 판정을 낸다.

    polarity=None인 모듈(뉴스 감성처럼 방향성이 애매한 것)은 자동 제외된다.
    신뢰도는 단순 다수결 비율로 시작한 것 — 추후 모듈 가중치를 반영해
    조정 가능하다.
    """
    polarities = [r.polarity for r in results if r.polarity]
    n_bull = polarities.count("bullish")
    n_bear = polarities.count("bearish")
    n_neu = polarities.count("neutral")
    total = len(polarities)

    if total and n_bull > n_bear and n_bull >= total * 0.5:
        signal, label, polarity = "BUY", "매수 우위", "bullish"
    elif total and n_bear > n_bull and n_bear >= total * 0.5:
        signal, label, polarity = "SELL", "매도 우위", "bearish"
    else:
        signal, label, polarity = "HOLD", "관망", "neutral"

    confidence = max(n_bull, n_bear, n_neu) / total if total else 0.0
    return {
        "signal": signal, "label": label, "polarity": polarity, "confidence": confidence,
        "n_bull": n_bull, "n_neu": n_neu, "n_bear": n_bear, "total": total,
    }
