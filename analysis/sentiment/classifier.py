"""
Sentiment classifier for news articles using Claude.
Batches multiple articles in a single API call for efficiency.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

import anthropic

from config.settings import ANTHROPIC_API_KEY


@dataclass
class SentimentResult:
    title: str
    source: str
    score: float      # -1.0 (매우 부정) ~ +1.0 (매우 긍정)
    label: str        # "positive" | "neutral" | "negative"
    summary: str = ""


class SentimentClassifier:
    MODEL = "claude-opus-4-8"
    MAX_TOKENS = 2048

    def __init__(self, api_key: str = ANTHROPIC_API_KEY) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)

    def classify(self, articles: list[dict]) -> list[SentimentResult]:
        """
        articles: list of {title, source, summary?}
        Returns SentimentResult for each article in same order.
        """
        if not articles:
            return []

        # Batch up to 20 articles per call
        results: list[SentimentResult] = []
        batch_size = 20
        for i in range(0, len(articles), batch_size):
            batch = articles[i : i + batch_size]
            results.extend(self._classify_batch(batch))
        return results

    def aggregate_score(self, results: list[SentimentResult]) -> float:
        """Weighted average; more extreme scores get slightly higher weight."""
        if not results:
            return 0.0
        weights = [abs(r.score) + 0.3 for r in results]
        total_w = sum(weights)
        return sum(r.score * w for r, w in zip(results, weights)) / total_w

    # ── Internal ──────────────────────────────────────────────────────────────

    def _classify_batch(self, articles: list[dict]) -> list[SentimentResult]:
        numbered = "\n".join(
            f"{i+1}. [{a.get('source', '')}] {a.get('title', '')} — {a.get('summary', '')[:120]}"
            for i, a in enumerate(articles)
        )

        prompt = f"""아래 뉴스 기사 목록의 주식 투자 관점 감성을 분석하세요.
각 기사에 대해 -1.0(매우 부정)~+1.0(매우 긍정) 점수와 label(positive/neutral/negative)을 반환하세요.

{numbered}

반드시 아래 형식의 JSON 배열만 반환하세요 (설명 없이):
```json
[
  {{"index": 1, "score": 0.7, "label": "positive", "summary": "한 줄 요약"}},
  ...
]
```"""

        try:
            response = self._client.messages.create(
                model=self.MODEL,
                max_tokens=self.MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )
            return self._parse_batch(articles, raw)
        except Exception:
            return [self._fallback(a) for a in articles]

    def _parse_batch(
        self, articles: list[dict], raw: str
    ) -> list[SentimentResult]:
        json_match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
        json_str = json_match.group(1) if json_match else raw.strip()

        try:
            items = json.loads(json_str)
            if not isinstance(items, list):
                raise ValueError("Expected list")
        except (json.JSONDecodeError, ValueError):
            return [self._fallback(a) for a in articles]

        results: list[SentimentResult] = []
        item_map = {int(item.get("index", 0)): item for item in items}
        for i, article in enumerate(articles):
            item = item_map.get(i + 1)
            if item is None:
                results.append(self._fallback(article))
                continue
            score = max(-1.0, min(1.0, float(item.get("score", 0.0))))
            label = str(item.get("label", "neutral"))
            if label not in {"positive", "neutral", "negative"}:
                label = "neutral"
            results.append(
                SentimentResult(
                    title=article.get("title", ""),
                    source=article.get("source", ""),
                    score=score,
                    label=label,
                    summary=str(item.get("summary", "")),
                )
            )
        return results

    @staticmethod
    def _fallback(article: dict) -> SentimentResult:
        return SentimentResult(
            title=article.get("title", ""),
            source=article.get("source", ""),
            score=0.0,
            label="neutral",
        )
