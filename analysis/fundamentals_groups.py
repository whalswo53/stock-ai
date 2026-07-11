"""밸류에이션 4그룹 taxonomy — fundamentals.py와 fundamentals_sources.py가 공유."""
from __future__ import annotations

GROUP_FIELDS: dict[str, list[str]] = {
    "밸류에이션": ["PER", "PBR", "PSR", "PCR", "EV/EBITDA", "배당수익률"],
    "수익성": ["ROE", "ROA", "영업이익률", "순이익률", "ROIC"],
    "성장성": ["매출성장(YoY)", "영업이익성장", "EPS성장"],
    "안정성": ["부채비율", "유동비율", "이자보상배율"],
}

# "pct" = 내부에서 분수(0.115=11.5%)로 저장 → ×100 후 % 표기
# "num" = 배수/비율 그대로 소수 2자리 표기
FIELD_FMT_KIND: dict[str, str] = {
    "PER": "num", "PBR": "num", "PSR": "num", "PCR": "num", "EV/EBITDA": "num",
    "배당수익률": "pct",
    "ROE": "pct", "ROA": "pct", "영업이익률": "pct", "순이익률": "pct", "ROIC": "pct",
    "매출성장(YoY)": "pct", "영업이익성장": "pct", "EPS성장": "pct",
    "부채비율": "num", "유동비율": "num", "이자보상배율": "num",
}
