"""
Spread diagnostics shared by the pair scanner and the direct-analysis page.

  hurst_exponent : 스프레드의 평균회귀 성향 (H < 0.5 = 평균회귀, 0.5 = 랜덤워크,
                   > 0.5 = 추세). 래그별 차분 표준편차의 로그-로그 기울기로 추정.
  half_life      : OU 반감기 (일). Δs = α + β·s_{t-1} 회귀에서 -ln(2)/β.
                   mean_reversion.py의 단일 종목 반감기와 같은 방식.
  fit_ou         : OU(Ornstein-Uhlenbeck) 프로세스 피팅. AR(1) 이산화로
                   θ(회귀 속도)·μ(장기 평균)·σ_eq(정상분포 표준편차)를 추정하고,
                   σ_eq를 현재 롤링 Z-score 스케일로 환산한 이론 진입 임계값을 제안.
  copula_metrics : 수익률 순위(의사 관측치) 기반 Kendall τ + 꼬리 의존성.
                   Clayton(하방)·Gumbel(상방) θ는 τ 역변환 닫힌형으로 적합 —
                   MLE 없이 값싸게 계산되므로 상위 후보 enrichment에 적합.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ── Hurst exponent ────────────────────────────────────────────────────────────

def hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """Estimates H from the scaling of lagged-difference std (log-log slope).

    Returns NaN if the series is too short or degenerate.
    """
    x = np.asarray(series.dropna().values, dtype=float)
    if len(x) < max_lag * 3:
        return float("nan")

    lags = range(2, max_lag)
    tau = np.array([np.std(x[lag:] - x[:-lag]) for lag in lags])
    if np.any(tau <= 0):
        return float("nan")

    slope = np.polyfit(np.log(np.array(list(lags), dtype=float)), np.log(tau), 1)[0]
    return float(slope)


# ── Half-life ─────────────────────────────────────────────────────────────────

def half_life(series: pd.Series) -> float:
    """OU half-life in trading days; inf when the series does not revert."""
    x = np.asarray(series.dropna().values, dtype=float)
    if len(x) < 30:
        return float("inf")

    delta = np.diff(x)
    lag = x[:-1]
    lag_c = lag - lag.mean()
    denom = float(np.dot(lag_c, lag_c))
    if denom <= 0:
        return float("inf")
    beta = float(np.dot(lag_c, delta - delta.mean()) / denom)
    return float(-np.log(2) / beta) if beta < 0 else float("inf")


# ── OU process fit ────────────────────────────────────────────────────────────

@dataclass
class OUFit:
    theta: float            # 회귀 속도 (1/일)
    mu: float               # 장기 평균 스프레드
    sigma_eq: float         # 정상분포 표준편차 (스프레드 단위)
    half_life_days: float   # ln(2)/θ
    suggested_entry_z: float  # σ_eq 1배 이탈을 현재 롤링 Z 스케일로 환산
    suggested_exit_z: float   # 0.25 σ_eq 환산 (평균 부근 복귀)


def fit_ou(spread: pd.Series, zscore_window: int = 30) -> OUFit | None:
    """Fits dX = θ(μ−X)dt + σdW via the AR(1) discretization
    X_t = c + φ·X_{t−1} + ε  (Δt = 1 trading day).

    suggested_entry_z / exit_z: OU 정상분포 σ_eq 기준 ±1σ_eq / ±0.25σ_eq
    이탈점을, 대시보드가 쓰는 롤링(윈도우 zscore_window) 표준편차 단위로
    환산한 근사 가이드. 롤링 σ가 σ_eq보다 작으면 1보다 큰 Z로 나타난다.
    Returns None when the spread is not mean-reverting (φ ≥ 1 or φ ≤ 0).
    """
    x = np.asarray(spread.dropna().values, dtype=float)
    if len(x) < max(60, zscore_window * 2):
        return None

    y_t, y_lag = x[1:], x[:-1]
    lag_c = y_lag - y_lag.mean()
    denom = float(np.dot(lag_c, lag_c))
    if denom <= 0:
        return None
    phi = float(np.dot(lag_c, y_t - y_t.mean()) / denom)
    c = float(y_t.mean() - phi * y_lag.mean())

    if not (0.0 < phi < 1.0):
        return None  # 랜덤워크(φ≥1) 또는 과잉진동(φ≤0) — OU 부적합

    resid = y_t - (c + phi * y_lag)
    sigma_eps = float(np.std(resid, ddof=1))

    theta = -np.log(phi)                     # Δt=1일
    mu = c / (1.0 - phi)
    sigma_eq = sigma_eps / np.sqrt(1.0 - phi * phi)
    hl = np.log(2) / theta

    roll_std = spread.rolling(zscore_window, min_periods=zscore_window).std()
    latest_roll = float(roll_std.dropna().iloc[-1]) if not roll_std.dropna().empty else float("nan")
    if latest_roll and np.isfinite(latest_roll) and latest_roll > 0:
        entry_z = sigma_eq / latest_roll
        exit_z = 0.25 * sigma_eq / latest_roll
    else:
        entry_z = exit_z = float("nan")

    return OUFit(
        theta=float(theta),
        mu=float(mu),
        sigma_eq=float(sigma_eq),
        half_life_days=float(hl),
        suggested_entry_z=float(entry_z),
        suggested_exit_z=float(exit_z),
    )


# ── Copula-based dependence ───────────────────────────────────────────────────

@dataclass
class CopulaMetrics:
    kendall_tau: float
    tail_dep_lower: float   # 경험적 하방 꼬리 의존 P(V≤q | U≤q), q=0.05
    tail_dep_upper: float   # 경험적 상방 꼬리 의존 P(V>1−q | U>1−q)
    family: str             # "Clayton(하방)" | "Gumbel(상방)" | "독립적"
    family_tail_dep: float  # 선택된 패밀리의 이론 꼬리 의존도 λ


def copula_metrics(
    price_a: pd.Series, price_b: pd.Series, q: float = 0.05,
) -> CopulaMetrics | None:
    """Rank-based (pseudo-observation) dependence metrics on daily returns.

    Clayton/Gumbel θ는 Kendall τ 역변환 닫힌형으로 적합:
      Clayton: θ = 2τ/(1−τ),  λ_L = 2^(−1/θ)
      Gumbel : θ = 1/(1−τ),   λ_U = 2 − 2^(1/θ)
    경험적 꼬리 의존이 더 큰 쪽(하방/상방)의 패밀리를 채택한다.
    """
    from scipy.stats import kendalltau

    combined = pd.concat([price_a, price_b], axis=1).dropna()
    if len(combined) < 60:
        return None
    ra = combined.iloc[:, 0].pct_change().dropna()
    rb = combined.iloc[:, 1].pct_change().dropna()
    n = len(ra)
    if n < 50:
        return None

    tau = float(kendalltau(ra.values, rb.values).statistic)

    # 의사 관측치 (ECDF 순위)
    u = ra.rank(method="average").values / (n + 1)
    v = rb.rank(method="average").values / (n + 1)

    lower = float(np.mean(v[u <= q] <= q)) if np.any(u <= q) else 0.0
    upper = float(np.mean(v[u >= 1 - q] >= 1 - q)) if np.any(u >= 1 - q) else 0.0

    if tau <= 0:
        return CopulaMetrics(tau, lower, upper, "독립적", 0.0)

    theta_c = 2 * tau / (1 - tau)
    theta_g = 1 / (1 - tau)
    lam_clayton = float(2 ** (-1 / theta_c)) if theta_c > 0 else 0.0
    lam_gumbel = float(2 - 2 ** (1 / theta_g)) if theta_g > 1 else 0.0

    if lower >= upper:
        return CopulaMetrics(tau, lower, upper, "Clayton(하방)", lam_clayton)
    return CopulaMetrics(tau, lower, upper, "Gumbel(상방)", lam_gumbel)
