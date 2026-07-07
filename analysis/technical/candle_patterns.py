"""
TA-Lib 기반 캔들 패턴 인식 + 과거 통계 신뢰도.

TA-Lib(C 라이브러리)이 없는 환경에서도 앱이 죽지 않도록,
import 실패 시 is_available() = False를 반환하고 호출부는 섹션을 숨긴다.

신뢰도: 로드된 히스토리에서 같은 패턴(같은 방향)이 발생했던 모든 날의
N일 후 수익률로 승률·평균 수익률·표본 수를 계산한다. 표본이 적으면
통계적 의미가 없으므로 UI는 표본 수를 반드시 함께 표시할 것.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import talib
    _TALIB_OK = True
except ImportError:
    _TALIB_OK = False


def is_available() -> bool:
    return _TALIB_OK


# 패턴 정의: TA-Lib 함수명 → (한글명, 유형)
# CDL 함수는 +100(강세)/-100(약세)/0을 반환한다 (별형은 ±100 외 값도 가능).
PATTERNS: dict[str, tuple[str, str]] = {
    "CDLHAMMER":         ("망치형 (Hammer)",              "반전"),
    "CDLENGULFING":      ("장악형 (Engulfing)",           "반전"),
    "CDLMORNINGSTAR":    ("샛별형 (Morning Star)",        "반전"),
    "CDLEVENINGSTAR":    ("석별형 (Evening Star)",        "반전"),
    "CDLDOJI":           ("도지 (Doji)",                  "반전"),
    "CDL3WHITESOLDIERS": ("적삼병 (Three White Soldiers)", "추세지속"),
    "CDL3BLACKCROWS":    ("흑삼병 (Three Black Crows)",   "추세지속"),
}


# 통계 판정 기준: 이 표본 수 미만이면 "판단 불가"
MIN_SAMPLES = 50
# 거래량 동반 판정: Vol_Ratio(20일 평균 대비)가 이 배수 이상
VOL_ACCOMPANY = 1.5
# 복합 신호용 RSI 기준 (패턴과의 동시 발생이 드물어 30/70보다 완화)
RSI_LOW, RSI_HIGH = 40, 60


@dataclass
class PatternHit:
    date: pd.Timestamp
    func: str            # TA-Lib 함수명
    name: str            # 한글 표기
    kind: str            # 반전 / 추세지속
    direction: str       # 강세 / 약세 / 중립
    sign: int            # +1 강세 / -1 약세 / 0 중립


@dataclass
class PatternStats:
    """한 (패턴, 방향) 조합의 horizon일 후 수익률 통계."""
    n: int
    win_rate: float      # 수익률 > 0 비율 (nan if n=0)
    avg_return: float    # 평균 수익률 % (nan if n=0)
    p_value: float       # 이항검정 (승률이 50%와 유의하게 다른가, 양측)

    def label(self) -> str:
        if self.n == 0:
            return "표본 없음"
        return (
            f"승률 {self.win_rate * 100:.0f}% · 평균 {self.avg_return:+.2f}% "
            f"(n={self.n}, p={self.p_value:.3f})"
        )


def detect(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV DataFrame → 패턴 신호 DataFrame (컬럼=PATTERNS 키, 값=CDL 반환값)."""
    if not _TALIB_OK:
        return pd.DataFrame(index=df.index)
    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)

    out = pd.DataFrame(index=df.index)
    for func in PATTERNS:
        out[func] = getattr(talib, func)(o, h, l, c)
    return out


def _direction(value: float) -> str:
    if value > 0:
        return "강세"
    if value < 0:
        return "약세"
    return "중립"


def recent_hits(
    df: pd.DataFrame,
    signals: pd.DataFrame | None = None,
    lookback_days: int = 5,
) -> list[PatternHit]:
    """최근 lookback_days 거래일 안에 발생한 패턴 목록 (통계는 full_stats로 별도 계산).

    Doji는 거의 매주 발생해 노이즈가 크므로, 다른 패턴과 같은 날 겹치면
    다른 패턴만 남긴다.
    """
    if signals is None:
        signals = detect(df)
    if signals.empty:
        return []

    hits: list[PatternHit] = []
    recent_idx = signals.index[-lookback_days:]
    for func, (name, kind) in PATTERNS.items():
        col = signals[func]
        for dt in recent_idx:
            val = float(col.loc[dt])
            if val == 0:
                continue
            hits.append(PatternHit(
                date=dt, func=func, name=name, kind=kind,
                direction=_direction(val), sign=int(np.sign(val)),
            ))

    # 같은 날 Doji + 다른 패턴 → Doji 제거
    dates_with_other = {h.date for h in hits if h.func != "CDLDOJI"}
    hits = [h for h in hits if not (h.func == "CDLDOJI" and h.date in dates_with_other)]
    hits.sort(key=lambda h: h.date, reverse=True)
    return hits


# ── 통계 (이항검정 + 복합 신호) ───────────────────────────────────────────────

def _make_stats(rets: list[float]) -> PatternStats:
    if not rets:
        return PatternStats(0, float("nan"), float("nan"), float("nan"))
    from scipy.stats import binomtest

    arr = np.array(rets)
    wins = int((arr > 0).sum())
    p = float(binomtest(wins, len(arr), 0.5).pvalue)
    return PatternStats(len(arr), float(wins / len(arr)), float(arr.mean()), p)


def _occurrence_returns(
    df: pd.DataFrame,
    func: str,
    sign: int,
    horizon: int,
    signals: pd.DataFrame | None = None,
    condition=None,
) -> list[float]:
    """같은 (패턴, 방향)의 모든 과거 발생일 horizon일 후 수익률 목록.

    condition: 발생일 행(df.loc[date])을 받아 bool을 반환하는 필터 —
    복합 신호(RSI/MACD/거래량 동반) 계산에 사용. 지표 컬럼이 없거나 NaN이면
    해당 발생일은 조건 불충족으로 처리한다.
    """
    if signals is None:
        signals = detect(df)
    if signals.empty or func not in signals.columns:
        return []

    close = df["Close"]
    col = signals[func]
    hit_dates = col.index[(np.sign(col) == sign) & (col != 0)]

    rets: list[float] = []
    positions = close.index.get_indexer(hit_dates)
    for pos in positions:
        if pos < 0 or pos + horizon >= len(close):
            continue
        if condition is not None:
            try:
                if not bool(condition(df.iloc[pos])):
                    continue
            except Exception:
                continue
        p0, p1 = float(close.iloc[pos]), float(close.iloc[pos + horizon])
        if p0 > 0:
            rets.append((p1 / p0 - 1) * 100)
    return rets


def full_stats(
    dfs: list[pd.DataFrame],
    func: str,
    sign: int,
    horizon: int,
) -> dict[str, PatternStats]:
    """(패턴, 방향)의 기본 + 복합 신호 통계. dfs가 여러 개면 풀 집계.

    dfs의 각 DataFrame은 OHLCV 필수, 복합 신호에는 RSI·MACD_Hist·Vol_Ratio
    컬럼 필요 (TechnicalIndicators.compute 산출물).

    반환 키:
      base    — 전체 발생
      vol_hi  — 거래량 동반 (Vol_Ratio ≥ 1.5)
      vol_lo  — 거래량 미동반
      rsi     — 패턴 방향과 일치하는 RSI 구간 (강세+RSI≤40 / 약세+RSI≥60)
      macd    — MACD 히스토그램 부호가 패턴 방향과 일치
    """
    def _rsi_cond(row) -> bool:
        v = float(row.get("RSI", float("nan")))
        return (v <= RSI_LOW) if sign > 0 else (v >= RSI_HIGH)

    def _macd_cond(row) -> bool:
        v = float(row.get("MACD_Hist", float("nan")))
        return (v > 0) if sign > 0 else (v < 0)

    conditions = {
        "base":   None,
        "vol_hi": lambda row: float(row.get("Vol_Ratio", 0) or 0) >= VOL_ACCOMPANY,
        "vol_lo": lambda row: float(row.get("Vol_Ratio", 0) or 0) < VOL_ACCOMPANY,
        "rsi":    _rsi_cond,
        "macd":   _macd_cond,
    }

    out: dict[str, PatternStats] = {}
    # detect()를 df당 1회만 돌리도록 시그널을 캐시
    sig_cache = [detect(df) for df in dfs]
    for key, cond in conditions.items():
        rets: list[float] = []
        for df, sig in zip(dfs, sig_cache):
            rets.extend(_occurrence_returns(df, func, sign, horizon, sig, cond))
        out[key] = _make_stats(rets)
    return out


def verdict(stats: PatternStats, sign: int) -> str:
    """판정 배지: 표본 부족 → 판단 불가, 그 외 이항검정 유의성 기준.

    강세 패턴은 승률 > 50%, 약세 패턴은 승률 < 50%(= horizon일 후 하락)일 때
    "패턴이 맞은" 것이다.
    """
    if stats.n < MIN_SAMPLES:
        return f"➖ 판단 불가 (표본 {stats.n} < {MIN_SAMPLES})"
    if stats.p_value < 0.05:
        edge_sign = 1 if stats.win_rate > 0.5 else -1
        expected = 1 if sign > 0 else -1
        if edge_sign == expected:
            return f"✅ 통계적으로 유의 (p={stats.p_value:.3f})"
        return f"❌ 역방향 유의 — 패턴 기대와 반대로 작동 (p={stats.p_value:.3f})"
    return f"⚠️ 유의성 없음 (p={stats.p_value:.3f})"
