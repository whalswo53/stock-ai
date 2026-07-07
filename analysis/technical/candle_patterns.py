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


@dataclass
class PatternHit:
    date: pd.Timestamp
    func: str            # TA-Lib 함수명
    name: str            # 한글 표기
    kind: str            # 반전 / 추세지속
    direction: str       # 강세 / 약세 / 중립
    # 과거 동일 패턴(같은 방향) 발생 후 horizon일 수익률 통계
    n_past: int          # 이번 발생 이전의 과거 표본 수
    win_rate: float      # 수익률 > 0 비율 (nan if 표본 없음)
    avg_return: float    # 평균 수익률 % (nan if 표본 없음)


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
    horizon: int = 5,
) -> list[PatternHit]:
    """최근 lookback_days 거래일 안에 발생한 패턴 목록 + 과거 통계.

    Doji는 거의 매주 발생해 노이즈가 크므로, 다른 패턴과 같은 날 겹치면
    다른 패턴만 남긴다.
    """
    if signals is None:
        signals = detect(df)
    if signals.empty:
        return []

    close = df["Close"]
    hits: list[PatternHit] = []

    recent_idx = signals.index[-lookback_days:]
    for func, (name, kind) in PATTERNS.items():
        col = signals[func]
        for dt in recent_idx:
            val = float(col.loc[dt])
            if val == 0:
                continue
            direction = _direction(val)
            n, wr, avg = _past_stats(close, col, dt, val, horizon)
            hits.append(PatternHit(
                date=dt, func=func, name=name, kind=kind,
                direction=direction, n_past=n, win_rate=wr, avg_return=avg,
            ))

    # 같은 날 Doji + 다른 패턴 → Doji 제거
    dates_with_other = {h.date for h in hits if h.func != "CDLDOJI"}
    hits = [h for h in hits if not (h.func == "CDLDOJI" and h.date in dates_with_other)]
    hits.sort(key=lambda h: h.date, reverse=True)
    return hits


def _past_stats(
    close: pd.Series,
    signal_col: pd.Series,
    current_date: pd.Timestamp,
    current_value: float,
    horizon: int,
) -> tuple[int, float, float]:
    """current_date 이전에 같은 방향으로 발생한 패턴들의 horizon일 후 수익률 통계."""
    same_dir = signal_col[
        (signal_col.index < current_date)
        & (np.sign(signal_col) == np.sign(current_value))
        & (signal_col != 0)
    ]
    rets: list[float] = []
    positions = close.index.get_indexer(same_dir.index)
    for pos in positions:
        if pos < 0 or pos + horizon >= len(close):
            continue
        p0, p1 = float(close.iloc[pos]), float(close.iloc[pos + horizon])
        if p0 > 0:
            rets.append((p1 / p0 - 1) * 100)
    if not rets:
        return 0, float("nan"), float("nan")
    arr = np.array(rets)
    return len(arr), float(np.mean(arr > 0)), float(arr.mean())
