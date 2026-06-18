"""
백테스팅 엔진.
신호 시리즈(1=매수, -1=매도, 0=유지)를 받아 포트폴리오 손익 곡선과
성과 지표를 계산한다. 신호는 당일 종가 기준으로 생성되고,
다음 날 시가에 체결(look-ahead bias 방지).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ── 데이터 구조 ───────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_date:  pd.Timestamp
    entry_price: float
    exit_date:   pd.Timestamp
    exit_price:  float
    return_pct:  float          # 수익률 (%)
    profit:      float          # 손익 (원/달러)
    holding_days: int


@dataclass
class BacktestResult:
    strategy_name:   str
    ticker:          str
    equity:          pd.Series  # 일별 포트폴리오 평가금액
    drawdown:        pd.Series  # 일별 낙폭 (0 ~ -1)
    trades:          list[Trade]
    total_return:    float       # 전체 수익률 (%)
    cagr:            float       # 연평균 수익률 (%)
    mdd:             float       # 최대낙폭 (%, 음수)
    sharpe:          float       # 샤프 비율
    win_rate:        float       # 승률 (0~1)
    n_trades:        int
    initial_capital: float
    commission:      float       # % per trade
    slippage:        float       # % per trade
    alpha:           float = 0.0 # 벤치마크 대비 초과 수익 (%)
    monthly_returns: pd.Series = field(default_factory=pd.Series)


# ── 엔진 ──────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    단순 롱-온리 백테스터.
    포지션 크기: 가용 현금 100% 투입 (분할매수 없음).
    공매도 없음. 소수 주식 허용.
    """

    def __init__(
        self,
        initial_capital: float = 10_000_000,
        commission: float = 0.015,   # % per trade (왕복 아님)
        slippage: float = 0.1,       # % per trade
    ) -> None:
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage   = slippage

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        strategy_name: str = "",
        ticker: str = "",
    ) -> BacktestResult:
        """
        df:      OHLCV DataFrame (Date index)
        signals: 같은 index, 값: 1(매수), -1(매도), 0(유지)
        """
        ic   = self.initial_capital
        comm = self.commission / 100
        slip = self.slippage   / 100

        cash    = float(ic)
        shares  = 0.0
        entry_price: float           = 0.0
        entry_date:  Optional[pd.Timestamp] = None
        trades:      list[Trade]     = []
        equity_vals: list[float]     = []

        n = len(df)

        for i in range(n):
            row   = df.iloc[i]
            close = float(row["Close"])

            # 일별 평가금액 (시가총액 방식)
            equity_vals.append(cash + shares * close)

            # 신호 실행은 다음 날 시가
            if i + 1 >= n:
                continue

            raw = signals.iloc[i]
            sig = 0 if (pd.isna(raw) or raw == 0) else int(raw)

            next_row  = df.iloc[i + 1]
            next_date = df.index[i + 1]
            next_open = float(next_row["Open"])

            if next_open <= 0 or pd.isna(next_open):
                continue

            if sig == 1 and shares == 0:
                buy_px    = next_open * (1 + slip)
                eff_px    = buy_px * (1 + comm)
                shares    = cash / eff_px
                entry_price = eff_px
                entry_date  = next_date
                cash = 0.0

            elif sig == -1 and shares > 0:
                sell_px = next_open * (1 - slip)
                eff_px  = sell_px * (1 - comm)
                proceeds = shares * eff_px
                ret_pct  = (eff_px / entry_price - 1) * 100
                profit   = proceeds - shares * entry_price
                trades.append(Trade(
                    entry_date=entry_date,          # type: ignore[arg-type]
                    entry_price=entry_price,
                    exit_date=next_date,
                    exit_price=eff_px,
                    return_pct=ret_pct,
                    profit=profit,
                    holding_days=(next_date - entry_date).days,  # type: ignore[operator]
                ))
                cash   = proceeds
                shares = 0.0

        # 기간 마지막 날에 열린 포지션 강제 청산
        if shares > 0 and entry_date is not None:
            last_close = float(df.iloc[-1]["Close"])
            sell_px    = last_close * (1 - slip)
            eff_px     = sell_px    * (1 - comm)
            proceeds   = shares * eff_px
            ret_pct    = (eff_px / entry_price - 1) * 100
            profit     = proceeds - shares * entry_price
            trades.append(Trade(
                entry_date=entry_date,
                entry_price=entry_price,
                exit_date=df.index[-1],
                exit_price=eff_px,
                return_pct=ret_pct,
                profit=profit,
                holding_days=(df.index[-1] - entry_date).days,
            ))
            cash = proceeds
            if equity_vals:
                equity_vals[-1] = cash

        equity = pd.Series(equity_vals, index=df.index[:len(equity_vals)], dtype=float)

        return BacktestResult(
            strategy_name=strategy_name,
            ticker=ticker,
            equity=equity,
            drawdown=self._drawdown(equity),
            trades=trades,
            **self._metrics(equity, trades, ic),
            initial_capital=ic,
            commission=self.commission,
            slippage=self.slippage,
            monthly_returns=self._monthly_returns(equity),
        )

    # ── Metric helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _drawdown(equity: pd.Series) -> pd.Series:
        rolling_max = equity.cummax()
        return (equity - rolling_max) / rolling_max.replace(0, np.nan)

    @staticmethod
    def _metrics(equity: pd.Series, trades: list[Trade], ic: float) -> dict:
        final_val   = float(equity.iloc[-1])
        total_ret   = (final_val / ic - 1) * 100

        n_days      = max((equity.index[-1] - equity.index[0]).days, 1)
        n_years     = n_days / 365.25
        cagr        = ((final_val / ic) ** (1 / n_years) - 1) * 100

        rolling_max = equity.cummax()
        dd          = (equity - rolling_max) / rolling_max.replace(0, np.nan)
        mdd         = float(dd.min() * 100)

        daily_rets  = equity.pct_change().dropna()
        std         = float(daily_rets.std())
        sharpe      = float(daily_rets.mean() / std * np.sqrt(252)) if std > 0 else 0.0

        win_rate    = sum(1 for t in trades if t.return_pct > 0) / len(trades) if trades else 0.0

        return dict(
            total_return=total_ret,
            cagr=cagr,
            mdd=mdd,
            sharpe=sharpe,
            win_rate=win_rate,
            n_trades=len(trades),
        )

    @staticmethod
    def _monthly_returns(equity: pd.Series) -> pd.Series:
        monthly = equity.resample("ME").last()
        return monthly.pct_change().dropna() * 100


# ── Benchmark buy-and-hold ────────────────────────────────────────────────────

def benchmark_equity(bm_df: pd.DataFrame, initial_capital: float) -> pd.Series:
    """벤치마크 단순 매수 보유 기준 평가금액 시리즈."""
    if bm_df.empty:
        return pd.Series(dtype=float)
    normalized = bm_df["Close"] / float(bm_df["Close"].iloc[0])
    return (normalized * initial_capital).rename("benchmark")
