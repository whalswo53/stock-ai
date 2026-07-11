"""
공통 밸류에이션(펀더멘털) 섹션.
소스(yfinance .info / 한국 종목 pykrx 보완)가 달라도 동일한 표준 dict를 반환해,
전 종목에서 같은 형태의 밸류 섹션을 렌더링할 수 있게 한다.
"""
from __future__ import annotations

FIELD_TO_INFO_KEY: dict[str, str] = {
    "시가총액":      "marketCap",
    "P/E(TTM)":      "trailingPE",
    "P/B":           "priceToBook",
    "매출성장(YoY)":  "revenueGrowth",
    "영업이익률":     "operatingMargins",
    "순이익률":       "profitMargins",
}


def _from_info(info: dict | None) -> dict:
    def g(key: str):
        v = (info or {}).get(key)
        return v if v is not None else "N/A"
    return {label: g(key) for label, key in FIELD_TO_INFO_KEY.items()}


def _kr_pykrx_fallback(ticker: str, missing: set[str]) -> dict:
    """yfinance .info가 비어있는 KRX 종목 보완.

    pykrx는 재무제표 API가 없어 시가총액/P·E/P·B만 채울 수 있고, 매출성장·마진은
    이 함수 범위 밖이라 채우지 않는다 (억지로 만들지 않음).
    """
    out: dict = {}
    if not (missing & {"시가총액", "P/E(TTM)", "P/B"}):
        return out
    try:
        from datetime import date, timedelta
        from pykrx import stock as krx

        code = ticker.split(".")[0]
        d = date.today()
        for _ in range(10):  # 최근 영업일 탐색 (주말·휴일 대비)
            ds = d.strftime("%Y%m%d")
            fdf = krx.get_market_fundamental_by_date(ds, ds, code)
            if fdf is not None and not fdf.empty:
                row = fdf.iloc[-1]
                per = float(row.get("PER", 0) or 0)
                pbr = float(row.get("PBR", 0) or 0)
                if "P/E(TTM)" in missing and per > 0:
                    out["P/E(TTM)"] = per
                if "P/B" in missing and pbr > 0:
                    out["P/B"] = pbr
                break
            d -= timedelta(days=1)

        if "시가총액" in missing:
            d2 = date.today()
            for _ in range(10):
                ds2 = d2.strftime("%Y%m%d")
                cdf = krx.get_market_cap_by_date(ds2, ds2, code)
                if cdf is not None and not cdf.empty:
                    cap = int(cdf.iloc[-1].get("시가총액", 0) or 0)
                    if cap > 0:
                        out["시가총액"] = cap
                    break
                d2 -= timedelta(days=1)
    except Exception:
        pass
    return out


def get_fundamentals(info: dict | None, ticker: str = "", is_korean: bool = False) -> dict:
    """표준화된 밸류에이션 dict. 없는 값은 'N/A'."""
    data = _from_info(info)
    if is_korean:
        missing = {k for k, v in data.items() if v == "N/A"}
        if missing:
            data.update(_kr_pykrx_fallback(ticker, missing))
    return data


def is_fully_missing(data: dict) -> bool:
    return all(v == "N/A" for v in data.values())


_FMT: dict[str, str] = {
    "시가총액": "cap", "P/E(TTM)": "num", "P/B": "num",
    "매출성장(YoY)": "pct", "영업이익률": "pct", "순이익률": "pct",
}


def _fmt_value(label: str, v) -> str:
    if v == "N/A":
        return "N/A"
    kind = _FMT.get(label, "num")
    try:
        if kind == "cap":
            v = float(v)
            for unit, div in (("조", 1e12), ("억", 1e8), ("M", 1e6)):
                if v >= div:
                    return f"{v / div:,.1f}{unit}"
            return f"{v:,.0f}"
        if kind == "pct":
            return f"{float(v) * 100:+.1f}%"
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return str(v)


def to_markdown_table(data: dict) -> str:
    if is_fully_missing(data):
        return "펀더멘털 데이터 미제공"
    lines = ["| 지표 | 값 |", "|------|-----|"]
    for label, v in data.items():
        lines.append(f"| {label} | {_fmt_value(label, v)} |")
    return "\n".join(lines)
