"""
News collector: Google News RSS (primary per-ticker) + feedparser market RSS (secondary).
Filters by TRUSTED_PUBLISHERS whitelist and recency window.

관련성 점수 (score_relevance):
  1.0  직접 — 종목명/티커가 제목에 등장
  0.8  직접 — 요약(본문 스니펫) 앞 100자에 등장
  0.5  섹터 — TICKER_SECTOR → SECTOR_KEYWORDS 키워드가 제목/요약에 등장
  0.0  일반 — 위 어디에도 해당 없음 (filter_relevant 기본 임계 0.5에서 탈락)
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote_plus

import feedparser

from config.sources import (
    BLOCKED_KEYWORDS, RSS_FEEDS, TRUSTED_PUBLISHERS,
    TICKER_KR_NAME, KOSPI_TICKER_MAP, NASDAQ_TICKER_MAP,
    SECTOR_KEYWORDS, TICKER_SECTOR,
)

# 직접 언급으로 인정할 요약 앞부분 길이
_SUMMARY_HEAD_CHARS = 100
# 제목 유사도가 이 값 이상이면 같은 기사(재탕)로 판정
_DUP_SIMILARITY = 0.75

# 회사명 뒤에 붙는 법인격 접미사 — 검색어/키워드 매칭 노이즈라 제거
_CORP_SUFFIX_RE = re.compile(
    r"[,.]?\s*\b(Inc|Incorporated|Corp|Corporation|Co\.?\s*,?\s*Ltd|Ltd|Holdings?|"
    r"Group|PLC|S\.?A\.?|N\.?V\.?|AG|SE)\b\.?\s*$",
    re.IGNORECASE,
)


def _clean_company_name(name: str) -> str:
    """야후 파이낸스 longName/shortName에서 법인격 접미사를 반복 제거한 검색용 회사명.

    예: "Apple Inc." -> "Apple", "Semiconductor Manufacturing International
    Corporation" -> "Semiconductor Manufacturing International".
    """
    name = (name or "").strip()
    prev = None
    while prev != name:
        prev = name
        name = _CORP_SUFFIX_RE.sub("", name).strip()
    return name


@dataclass
class Article:
    title: str
    source: str
    url: str
    published_at: Optional[datetime]
    summary: str = ""
    ticker: str = ""
    relevance: float = 0.0        # 0.0~1.0 (score_relevance가 채움)
    relevance_tier: str = ""      # "직접" | "섹터" | ""(미채점/일반)


class NewsCollector:
    _GNEWS_KR = "https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"
    _GNEWS_EN = "https://news.google.com/rss/search?q={q}&hl=en&gl=US&ceid=US:en"

    def __init__(self, hours_default: int = 24) -> None:
        self._hours_default = hours_default

    # ── Public interface ──────────────────────────────────────────────────────

    def fetch_by_ticker(
        self,
        ticker: str,
        market: str = "NASDAQ",
        hours: int = 24,
        company_name: str = "",
    ) -> list[Article]:
        """Fetches news via Google News RSS for the given ticker.

        company_name: yfinance .info의 longName/shortName 등 실제 회사명.
        검색어는 항상 회사명 기반이어야 한다 — 티커(특히 0981.HK처럼 숫자
        코드인 경우)로 검색하면 Google News에서 무관한 결과만 걸린다.
        """
        articles = self._from_google_news(ticker, market, hours, company_name)
        return self._deduplicate(articles)

    def fetch_market_news(
        self,
        market: str = "NASDAQ",
        hours: int = 6,
    ) -> list[Article]:
        """Fetches broad market news from configured RSS feeds."""
        return self._from_rss(market, hours)

    # ── Sources ───────────────────────────────────────────────────────────────

    def _from_google_news(
        self,
        ticker: str,
        market: str,
        hours: int,
        company_name: str = "",
    ) -> list[Article]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        results: list[Article] = []

        for feed_url in self._google_news_urls(ticker, market, company_name):
            try:
                parsed = feedparser.parse(feed_url)
            except Exception:
                continue

            for entry in parsed.entries:
                title = entry.get("title") or ""
                if not title or self._has_blocked_keyword(title):
                    continue

                src = entry.get("source")
                source = (src.get("title", "") if isinstance(src, dict) else "") or ""

                url = entry.get("link") or ""
                summary = entry.get("summary") or ""

                pub_dt: Optional[datetime] = None
                published = entry.get("published_parsed")
                if published:
                    try:
                        pub_dt = datetime(*published[:6], tzinfo=timezone.utc)
                        if pub_dt < cutoff:
                            continue
                    except (TypeError, ValueError):
                        pass

                results.append(Article(
                    title=title,
                    source=source,
                    url=url,
                    published_at=pub_dt,
                    summary=summary[:300],
                    ticker=ticker,
                ))

        return results

    def _google_news_urls(self, ticker: str, market: str, company_name: str = "") -> list[str]:
        """Builds Google News RSS URLs for the ticker.

        검색어는 항상 회사명 기반 — 티커/거래소 접미사(0981.HK 등)로 검색하지 않는다.
        숫자 코드(HK/CN 등)를 그대로 쓰면 Google News에서 아무 의미 없는 결과만
        걸리기 때문에, company_name(yfinance longName/shortName)을 최우선으로 쓰고
        정적 맵의 한글/영문명은 보조로만 쓴다.

        KR stocks     → 1 feed: 한국어 로케일, 한글 회사명.
        그 외 전 종목 → 2 feeds: 한국어 로케일(회사명) + 영어 로케일(티커+영문 회사명).
        """
        clean_name = _clean_company_name(company_name)

        if market in ("KOSPI", "KOSDAQ"):
            name = TICKER_KR_NAME.get(ticker) or clean_name or ticker.split(".")[0]
            return [self._GNEWS_KR.format(q=quote_plus(name))]

        base = ticker.split(".")[0]
        kr_query = clean_name or base

        # 정적 맵에 영문 회사명이 있으면 우선 사용, 없으면 info 기반 clean_name으로 보완
        en_name = next(
            (n for n, t in NASDAQ_TICKER_MAP.items()
             if t == base and all(ord(c) < 128 for c in n) and len(n) > len(base)),
            "",
        ) or clean_name

        # base가 순수 숫자 코드(HK/CN 등)면 회사명 없이 붙이지 않는다 — "0981 SMIC" 같은
        # 잡음을 피하고 이름만으로 검색한다.
        if en_name and not base.isdigit():
            en_query = f"{base} {en_name}".strip()
        else:
            en_query = en_name or base

        return [
            self._GNEWS_KR.format(q=quote_plus(kr_query)),
            self._GNEWS_EN.format(q=quote_plus(en_query)),
        ]

    def _from_rss(
        self,
        market: str,
        hours: int,
        ticker_hint: str = "",
    ) -> list[Article]:
        """Fetches from static market RSS feeds (used by fetch_market_news)."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        feeds = RSS_FEEDS.get(market, [])
        trusted = TRUSTED_PUBLISHERS.get(market, [])
        results: list[Article] = []

        for feed_url in feeds:
            try:
                parsed = feedparser.parse(feed_url)
            except Exception:
                continue

            feed_title = parsed.feed.get("title", "")
            for entry in parsed.entries:
                title = entry.get("title") or ""
                if not title or self._has_blocked_keyword(title):
                    continue

                source = entry.get("source", {}).get("title") or feed_title
                if trusted and not any(t in source for t in trusted):
                    source = feed_url.split("/")[2] if "/" in feed_url else source

                url = entry.get("link") or ""
                summary = entry.get("summary") or ""

                pub_dt: Optional[datetime] = None
                published = entry.get("published_parsed")
                if published:
                    try:
                        pub_dt = datetime(*published[:6], tzinfo=timezone.utc)
                        if pub_dt < cutoff:
                            continue
                    except (TypeError, ValueError):
                        pass

                results.append(Article(
                    title=title,
                    source=source,
                    url=url,
                    published_at=pub_dt,
                    summary=summary[:300],
                    ticker=ticker_hint,
                ))

        return results

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _has_blocked_keyword(text: str) -> bool:
        return any(kw in text for kw in BLOCKED_KEYWORDS)

    @staticmethod
    def _deduplicate(articles: list[Article]) -> list[Article]:
        """제목 유사도 기반 중복/재탕 제거.

        완전 일치(소문자 80자)는 물론, 조사·어미·매체명 꼬리표만 다른
        재탕 기사도 SequenceMatcher 유사도 ≥ _DUP_SIMILARITY면 제거한다.
        기사 수가 수십 건 수준이라 O(n²) 비교도 무해하다.
        """
        kept: list[Article] = []
        kept_titles: list[str] = []
        for a in articles:
            title_norm = a.title.strip().lower()
            # 구글 뉴스 제목 꼬리표(" - 매체명") 제거 후 비교
            core = title_norm.rsplit(" - ", 1)[0]
            is_dup = any(
                difflib.SequenceMatcher(None, core, prev).ratio() >= _DUP_SIMILARITY
                for prev in kept_titles
            )
            if not is_dup:
                kept.append(a)
                kept_titles.append(core)
        return kept

    def build_filter_keywords(
        self, ticker: str, market: str, company_name: str = "",
    ) -> set[str]:
        """Returns keywords (company names / ticker symbol) for relevance filtering.

        company_name(정제된 실제 회사명)을 항상 포함한다 — 정적 맵에 없는 종목
        (HK/CN 등)은 이 값이 없으면 "직접 언급" 판정 자체가 불가능해진다.
        base가 순수 숫자 코드(예: "0981")면 키워드로 넣지 않는다 — 기사 본문의
        아무 숫자에나 우연히 매칭되는 잡음이라 정직한 판정에 방해가 된다.
        """
        keywords: set[str] = set()
        if market in ("KOSPI", "KOSDAQ"):
            kr_name = TICKER_KR_NAME.get(ticker, "")
            if kr_name:
                keywords.add(kr_name)
            for name, t in KOSPI_TICKER_MAP.items():
                if t == ticker and len(name) >= 2:
                    keywords.add(name)
        else:
            base = ticker.split(".")[0]
            if len(base) >= 2 and not base.isdigit():
                keywords.add(base)
            for name, t in NASDAQ_TICKER_MAP.items():
                if t == base and len(name) >= 2:
                    keywords.add(name)

        clean_name = _clean_company_name(company_name)
        if len(clean_name) >= 2:
            keywords.add(clean_name)

        return {k for k in keywords if k}

    def score_relevance(
        self, articles: list[Article], ticker: str, market: str, company_name: str = "",
    ) -> list[Article]:
        """각 기사의 relevance/relevance_tier를 채운다 (in-place, 반환은 동일 리스트).

        직접(제목 1.0 / 요약 앞부분 0.8) > 섹터 키워드(0.5) > 일반(0.0).
        """
        keywords = self.build_filter_keywords(ticker, market, company_name)
        is_kr = market in ("KOSPI", "KOSDAQ")
        if not is_kr:
            keywords = {k.lower() for k in keywords}

        sector = TICKER_SECTOR.get(ticker) or TICKER_SECTOR.get(ticker.split(".")[0], "")
        sector_kws = SECTOR_KEYWORDS.get(sector, [])

        def _contains(text: str, kws) -> bool:
            if not is_kr:
                text = text.lower()
            return any(kw in text for kw in kws)

        for a in articles:
            head = a.summary[:_SUMMARY_HEAD_CHARS]
            if keywords and _contains(a.title, keywords):
                a.relevance, a.relevance_tier = 1.0, "직접"
            elif keywords and _contains(head, keywords):
                a.relevance, a.relevance_tier = 0.8, "직접"
            elif sector_kws and _contains(a.title + " " + a.summary, sector_kws):
                a.relevance, a.relevance_tier = 0.5, "섹터"
            else:
                a.relevance, a.relevance_tier = 0.0, ""
        return articles

    def filter_relevant(
        self,
        articles: list[Article],
        ticker: str,
        market: str,
        min_relevance: float = 0.5,
        company_name: str = "",
    ) -> list[Article]:
        """관련도 점수 기반 필터 — 직접 언급 우선, 섹터 키워드는 낮은 점수로 포함.

        점수 내림차순(같으면 최신순)으로 정렬해 반환하므로 직접 언급 기사가
        항상 앞에 온다.
        """
        self.score_relevance(articles, ticker, market, company_name)
        kept = [a for a in articles if a.relevance >= min_relevance]
        kept.sort(
            key=lambda a: (
                a.relevance,
                a.published_at or datetime.min.replace(tzinfo=timezone.utc),
            ),
            reverse=True,
        )
        return kept

    def to_dicts(self, articles: list[Article]) -> list[dict]:
        return [
            {
                "title": a.title,
                "source": a.source,
                "url": a.url,
                "summary": a.summary,
                "published_at": a.published_at.isoformat() if a.published_at else "",
                "ticker": a.ticker,
                "relevance": a.relevance,
                "relevance_tier": a.relevance_tier,
            }
            for a in articles
        ]
