"""
News collector: Google News RSS (primary per-ticker) + feedparser market RSS (secondary).
Filters by TRUSTED_PUBLISHERS whitelist and recency window.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote_plus

import feedparser

from config.sources import (
    BLOCKED_KEYWORDS, RSS_FEEDS, TRUSTED_PUBLISHERS,
    TICKER_KR_NAME, KOSPI_TICKER_MAP, NASDAQ_TICKER_MAP,
)


@dataclass
class Article:
    title: str
    source: str
    url: str
    published_at: Optional[datetime]
    summary: str = ""
    ticker: str = ""


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
    ) -> list[Article]:
        """Fetches news via Google News RSS for the given ticker."""
        articles = self._from_google_news(ticker, market, hours)
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
    ) -> list[Article]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        results: list[Article] = []

        for feed_url in self._google_news_urls(ticker, market):
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

    def _google_news_urls(self, ticker: str, market: str) -> list[str]:
        """Builds Google News RSS URLs for the ticker.

        KR stocks  → 1 feed: Korean locale, Korean company name.
        US stocks  → 2 feeds: Korean locale (ticker only) + English locale (ticker + company name).
        """
        if market in ("KOSPI", "KOSDAQ"):
            name = TICKER_KR_NAME.get(ticker) or ticker.split(".")[0]
            return [self._GNEWS_KR.format(q=quote_plus(name))]

        base = ticker.split(".")[0]
        # Pick the first English company name longer than the ticker symbol itself
        en_name = next(
            (n for n, t in NASDAQ_TICKER_MAP.items()
             if t == base and all(ord(c) < 128 for c in n) and len(n) > len(base)),
            "",
        )
        en_query = f"{base} {en_name}".strip() if en_name else base
        return [
            self._GNEWS_KR.format(q=quote_plus(base)),
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
        seen: set[str] = set()
        out: list[Article] = []
        for a in articles:
            key = a.title.strip().lower()[:80]
            if key not in seen:
                seen.add(key)
                out.append(a)
        return out

    def build_filter_keywords(self, ticker: str, market: str) -> set[str]:
        """Returns keywords (company names / ticker symbol) for relevance filtering."""
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
            if len(base) >= 2:
                keywords.add(base)
            for name, t in NASDAQ_TICKER_MAP.items():
                if t == base and len(name) >= 2:
                    keywords.add(name)
        return {k for k in keywords if k}

    def filter_relevant(self, articles: list[Article], ticker: str, market: str) -> list[Article]:
        """Keeps only articles whose title/summary mentions the company or ticker."""
        keywords = self.build_filter_keywords(ticker, market)
        if not keywords:
            return articles
        is_kr = market in ("KOSPI", "KOSDAQ")
        result = []
        for a in articles:
            text = a.title + " " + a.summary
            if is_kr:
                if any(kw in text for kw in keywords):
                    result.append(a)
            else:
                text_lower = text.lower()
                if any(kw.lower() in text_lower for kw in keywords):
                    result.append(a)
        return result

    def to_dicts(self, articles: list[Article]) -> list[dict]:
        return [
            {
                "title": a.title,
                "source": a.source,
                "url": a.url,
                "summary": a.summary,
                "published_at": a.published_at.isoformat() if a.published_at else "",
                "ticker": a.ticker,
            }
            for a in articles
        ]
