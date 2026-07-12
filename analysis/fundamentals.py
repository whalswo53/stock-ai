"""
공통 밸류에이션(펀더멘털) 섹션.

4그룹(밸류에이션/수익성/성장성/안정성) 표준 taxonomy로 정규화한다. 소스는
시장별로 층을 쌓는다 — yfinance .info(전 시장 1차) → 부족분만 시장별 보조
소스(한국: 네이버 금융 → pykrx / 미국: Finnhub)로 채운다. 이미 채워진 값은
덮어쓰지 않는다. 전부 실패해도 크래시 없이 "N/A".
"""
from __future__ import annotations

from analysis import fundamentals_sources as src
from analysis.fundamentals_groups import GROUP_FIELDS, FIELD_FMT_KIND
from data.collectors.realtime_quote import is_us_ticker


def _merge_missing(base: dict, patch: dict) -> None:
    """base의 None(미채움) 자리만 patch 값으로 채운다 — 이미 채워진 값은 보존."""
    for group, fields in patch.items():
        for label, val in fields.items():
            if val is None:
                continue
            if base.get(group, {}).get(label) is None:
                base[group][label] = val


def get_fundamentals(ctx: dict) -> dict:
    """반환: {그룹명: {지표명: 값 or 'N/A'}}. ctx = {ticker, info, is_korean, ...}."""
    info = ctx.get("info")
    ticker = ctx.get("ticker", "")
    is_korean = ctx.get("is_korean", False)

    data = src.from_yfinance_info(info)

    if is_korean:
        _merge_missing(data, src.from_naver(ticker))
        _merge_missing(data, src.from_pykrx(ticker))
    elif is_us_ticker(ticker):
        _merge_missing(data, src.from_finnhub(ticker))
    # HK/CN 등 그 외 시장: yfinance .info가 이미 대체로 채워줘서(실측 확인)
    # 별도 보조 소스 없음 — 야후 key-statistics HTML 스크래핑은 접근 자체가
    # 막혀 있어(404) 시도하지 않음.

    return {
        group: {label: (v if v is not None else "N/A") for label, v in fields.items()}
        for group, fields in data.items()
    }


def _group_fully_missing(fields: dict) -> bool:
    return all(v == "N/A" for v in fields.values())


def is_fully_missing(data: dict) -> bool:
    return all(_group_fully_missing(fields) for fields in data.values())


def fmt_value(label: str, v) -> str:
    if v == "N/A":
        return "N/A"
    kind = FIELD_FMT_KIND.get(label, "num")
    try:
        if kind == "pct":
            return f"{float(v) * 100:+.1f}%"
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return str(v)


def to_markdown_table(data: dict) -> str:
    if is_fully_missing(data):
        return "펀더멘털 데이터 미제공"
    parts = []
    for group, fields in data.items():
        parts.append(f"**{group}**")
        if _group_fully_missing(fields):
            parts.append("데이터 미제공")
            continue
        lines = ["| 지표 | 값 |", "|------|-----|"]
        for label, v in fields.items():
            lines.append(f"| {label} | {fmt_value(label, v)} |")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)
