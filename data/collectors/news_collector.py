"""
News collector: yfinance ticker.news (primary) + feedparser RSS (secondary).
Filters by TRUSTED_PUBLISHERS whitelist and recency window.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import feedparser
import yfinance as yf

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
    def __init__(self, hours_default: int = 24) -> None:
        self._hours_default = hours_default

    # ── Public interface ──────────────────────────────────────────────────────

    def fetch_by_ticker(
        self,
        ticker: str,
        market: str = "NASDAQ",
        hours: int = 24,
    ) -> list[Article]:
        """Fetches news for a single ticker. Returns deduplicated list."""
        articles: list[Article] = []
        articles.extend(self._from_yfinance(ticker, market, hours))
        articles.extend(self._from_rss(market, hours, ticker_hint=ticker))
        return self._deduplicate(articles)

    def fetch_market_news(
        self,
        market: str = "NASDAQ",
        hours: int = 6,
    ) -> list[Article]:
        """Fetches broad market news for the given market."""
        return self._from_rss(market, hours)

    # ── Sources ───────────────────────────────────────────────────────────────

    def _from_yfinance(
        self,
        ticker: str,
        market: str,
        hours: int,
    ) -> list[Article]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        results: list[Article] = []
        try:
            raw_news = yf.Ticker(ticker).news or []
        except Exception:
            return []

        trusted = TRUSTED_PUBLISHERS.get(market, [])
        for item in raw_news:
            title = item.get("title") or ""
            source = item.get("publisher") or ""
            url = item.get("link") or item.get("url") or ""
            pub_ts = item.get("providerPublishTime")

            if not title or self._has_blocked_keyword(title):
                continue

            if trusted and not any(t in source for t in trusted):
                continue

            pub_dt: Optional[datetime] = None
            if pub_ts:
                try:
                    pub_dt = datetime.fromtimestamp(int(pub_ts), tz=timezone.utc)
                    if pub_dt < cutoff:
                        continue
                except (ValueError, OSError):
                    pass

            results.append(
                Article(
                    title=title,
                    source=source,
                    url=url,
                    published_at=pub_dt,
                    summary=item.get("summary") or "",
                    ticker=ticker,
                )
            )
        return results

    def _from_rss(
        self,
        market: str,
        hours: int,
        ticker_hint: str = "",
    ) -> list[Article]:
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

                # For ticker-specific queries, filter by company name presence
                if ticker_hint:
                    company_hint = ticker_hint.replace(".KS", "").replace(".KQ", "")
                    if company_hint not in title and company_hint not in (entry.get("summary") or ""):
                        # Don't discard — market RSS may still be relevant; just lower priority
                        pass

                source = entry.get("source", {}).get("title") or feed_title
                if trusted and not any(t in source for t in trusted):
                    # Use feed domain as fallback source label
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

                results.append(
                    Article(
                        title=title,
                        source=source,
                        url=url,
                        published_at=pub_dt,
                        summary=summary[:300],
                        ticker=ticker_hint,
                    )
                )

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
