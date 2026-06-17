"""
Prompt builder + response parser for Claude.ai browser-based workflow.
No API calls — generates a self-contained prompt to paste into claude.ai,
then parses the pasted reply back into an AnalysisResult.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Optional

import pandas as pd

from analysis.ai.base_model import AnalysisResult
from memory.pattern_analyzer import PatternAnalyzer
from memory.user_memory import UserMemory


class ClaudeAnalyst:
    def __init__(self, memory: Optional[UserMemory] = None) -> None:
        self._memory = memory or UserMemory()
        self._analyzer = PatternAnalyzer(self._memory)

    # ── Public: prompt generation ─────────────────────────────────────────────

    def build_prompt(
        self,
        ticker: str,
        df: pd.DataFrame,
        news_articles: Optional[list[dict]] = None,
        market: str = "",
    ) -> str:
        """
        Returns a self-contained analysis prompt ready to paste into claude.ai.
        Includes: current indicators, 5-day price trend, news, personal context.
        """
        if df.empty:
            return f"데이터 없음 — {ticker} 가격 데이터를 불러오지 못했습니다."

        last = df.iloc[-1]
        personal_ctx = self._analyzer.get_summary_text(ticker)

        def _fmt(val: object, fmt: str = ".2f") -> str:
            try:
                return format(float(val), fmt)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return "N/A"

        # Indicator snapshot
        close    = _fmt(last.get("Close"),       ",.0f")
        rsi      = _fmt(last.get("RSI"),          ".1f")
        macd     = _fmt(last.get("MACD"),         ".4f")
        macd_sig = _fmt(last.get("MACD_Signal"),  ".4f")
        macd_h   = _fmt(last.get("MACD_Hist"),    ".4f")
        ma5      = _fmt(last.get("MA5"),          ",.0f")
        ma20     = _fmt(last.get("MA20"),         ",.0f")
        ma60     = _fmt(last.get("MA60"),         ",.0f")
        bb_upper = _fmt(last.get("BB_Upper"),     ",.0f")
        bb_lower = _fmt(last.get("BB_Lower"),     ",.0f")

        # Recent 5-day table
        cols = [c for c in ["Close", "Volume", "RSI", "MACD"] if c in df.columns]
        recent_table = (
            df.tail(5)[cols]
            .to_string(float_format=lambda x: f"{x:,.2f}")
        )

        # MACD cross label
        try:
            cross = "골든크로스 ↑" if float(last.get("MACD", 0)) > float(last.get("MACD_Signal", 0)) else "데드크로스 ↓"
        except (TypeError, ValueError):
            cross = "N/A"

        # RSI zone label
        try:
            rsi_f = float(last.get("RSI", 50))
            rsi_zone = (
                "과매도 구간 (매수 고려)" if rsi_f < 30
                else "과매수 구간 (매도 고려)" if rsi_f > 70
                else "중립 구간"
            )
        except (TypeError, ValueError):
            rsi_zone = "N/A"

        # News section
        if news_articles:
            news_lines = [f"## 최근 뉴스 ({len(news_articles)}건, 최근 48시간)"]
            for i, a in enumerate(news_articles[:10], 1):
                pub = a.get("published_at", "")[:10] if a.get("published_at") else ""
                summary = a.get("summary", "")[:100]
                news_lines.append(
                    f"{i}. [{a.get('source', '')}] {a.get('title', '')} ({pub})"
                    + (f"\n   → {summary}" if summary else "")
                )
            news_section = "\n".join(news_lines)
        else:
            news_section = (
                "## 최근 뉴스\n"
                "관련 뉴스를 찾을 수 없었습니다. "
                "직접 최신 뉴스를 확인하여 분석에 참고해주세요."
            )

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        market_label = f" ({market})" if market else ""

        prompt = f"""# 주식 분석 요청 — {ticker}{market_label}

당신은 전문 주식 투자 분석가입니다. 아래 실시간 데이터를 바탕으로 **{ticker}** 종목을 심층 분석하고, 구체적인 투자 의견을 제시해주세요.

**분석 시각:** {now}

---

## 현재 기술적 지표

| 지표 | 값 | 해석 |
|------|-----|------|
| 현재가 | {close} | — |
| RSI(14) | {rsi} | {rsi_zone} |
| MACD | {macd} / Signal: {macd_sig} / Hist: {macd_h} | {cross} |
| MA5 / MA20 / MA60 | {ma5} / {ma20} / {ma60} | — |
| 볼린저밴드 | 상단: {bb_upper} / 하단: {bb_lower} | — |

## 최근 5거래일 가격 동향

```
{recent_table}
```

{news_section}

## 투자자 개인 컨텍스트 (나의 투자 패턴)

{personal_ctx}

---

## 분석 요청 사항

위 데이터를 종합하여 다음을 포함한 분석을 작성해주세요:

1. **기술적 지표 해석** — RSI·MACD·MA·볼린저밴드가 가리키는 방향
2. **뉴스 감성 분석** — 뉴스가 주가에 미칠 영향
3. **투자 의견** — BUY / SELL / HOLD 중 하나, 명확한 근거와 함께
4. **리스크 요인** — 반드시 확인해야 할 위험 요소
5. **목표가** — 현실적인 수치 (없다면 null)

**⚠️ 응답 마지막에 반드시 아래 JSON 블록을 포함해주세요 (자동 저장에 사용됩니다):**

```json
{{
  "signal": "BUY 또는 SELL 또는 HOLD",
  "confidence": 0.0에서1.0 사이 숫자,
  "price_target": 목표가숫자또는null,
  "reasons": ["핵심 이유 1", "핵심 이유 2", "핵심 이유 3"],
  "report_md": "## 분석 요약\\n마크다운 형식 상세 내용"
}}
```"""

        return prompt

    # ── Public: response parsing ──────────────────────────────────────────────

    def parse_response(self, raw: str, ticker: str) -> AnalysisResult:
        """
        Parses a Claude.ai response (pasted text) into an AnalysisResult.
        Looks for a ```json block first, then falls back to bare {...}.
        """
        json_match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{[^{}]*"signal"[^{}]*\}', raw, re.DOTALL)
            json_str = json_match.group(0) if json_match else ""

        if not json_str:
            # Return a neutral result — user can override signal in the form
            return AnalysisResult(
                ticker=ticker,
                signal="HOLD",
                confidence=0.5,
                reasons=["JSON 블록을 찾지 못했습니다 — 아래에서 직접 시그널을 선택하세요."],
                report_md=raw,
            )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return AnalysisResult(
                ticker=ticker,
                signal="HOLD",
                confidence=0.5,
                reasons=["JSON 파싱 오류 — 아래에서 시그널을 선택하세요."],
                report_md=raw,
            )

        signal = str(data.get("signal", "HOLD")).upper()
        if signal not in {"BUY", "SELL", "HOLD"}:
            signal = "HOLD"

        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        reasons = data.get("reasons", [])
        if not isinstance(reasons, list):
            reasons = [str(reasons)]

        price_target_raw = data.get("price_target")
        try:
            price_target: Optional[float] = float(price_target_raw) if price_target_raw else None
        except (TypeError, ValueError):
            price_target = None

        report_md = str(data.get("report_md", raw))

        return AnalysisResult(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            reasons=reasons,
            report_md=report_md,
            price_target=price_target,
        )
