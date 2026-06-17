"""
Discovers investment patterns from historical decisions that have recorded outcomes.
Writes back to UserMemory and surfaces rule suggestions.
"""

from __future__ import annotations

from typing import Optional

from memory.user_memory import UserMemory


class PatternAnalyzer:
    def __init__(self, memory: Optional[UserMemory] = None) -> None:
        self.memory = memory or UserMemory()

    # ── Main entry point ──────────────────────────────────────────────────────

    def analyze_patterns(self) -> dict[str, int]:
        """
        Re-runs all pattern discovery from decided trades with outcomes.
        Returns counts of patterns and rules upserted.
        """
        decisions = self.memory.get_decisions_with_outcomes()
        if len(decisions) < 3:
            return {"patterns": 0, "rules": 0}

        n_patterns = 0
        n_patterns += self._analyze_rsi_zone(decisions)
        n_patterns += self._analyze_macd_signal(decisions)
        n_patterns += self._analyze_sector(decisions)
        n_patterns += self._analyze_ai_agreement(decisions)

        n_rules = self._discover_rules()
        return {"patterns": n_patterns, "rules": n_rules}

    # ── Pattern computers ─────────────────────────────────────────────────────

    def _analyze_rsi_zone(self, decisions: list[dict]) -> int:
        buckets: dict[str, list[float]] = {
            "oversold": [],    # RSI < 30
            "low_mid": [],     # 30 ≤ RSI < 50
            "high_mid": [],    # 50 ≤ RSI < 70
            "overbought": [],  # RSI ≥ 70
        }
        for d in decisions:
            rsi = d.get("rsi")
            ret = d.get("outcome_pct")
            if rsi is None or ret is None:
                continue
            if rsi < 30:
                buckets["oversold"].append(ret)
            elif rsi < 50:
                buckets["low_mid"].append(ret)
            elif rsi < 70:
                buckets["high_mid"].append(ret)
            else:
                buckets["overbought"].append(ret)

        labels = {
            "oversold": "RSI < 30 (과매도 구간) 매수",
            "low_mid": "RSI 30~50 구간 매수",
            "high_mid": "RSI 50~70 구간 매수",
            "overbought": "RSI ≥ 70 (과매수 구간) 매수",
        }
        count = 0
        for key, returns in buckets.items():
            if len(returns) < 2:
                continue
            win_rate = sum(1 for r in returns if r > 0) / len(returns)
            avg_ret = sum(returns) / len(returns)
            self.memory.upsert_pattern(
                pattern_type="rsi_zone",
                pattern_key=f"rsi_{key}",
                description=labels[key],
                win_rate=win_rate,
                avg_return=avg_ret,
                sample_count=len(returns),
            )
            count += 1
        return count

    def _analyze_macd_signal(self, decisions: list[dict]) -> int:
        buckets: dict[str, list[float]] = {"golden": [], "dead": []}
        for d in decisions:
            cross = d.get("macd_cross", "")
            ret = d.get("outcome_pct")
            if not cross or ret is None:
                continue
            if "golden" in cross.lower() or "골든" in cross:
                buckets["golden"].append(ret)
            elif "dead" in cross.lower() or "데드" in cross:
                buckets["dead"].append(ret)

        labels = {
            "golden": "MACD 골든크로스 시 매수",
            "dead": "MACD 데드크로스 이후 매수",
        }
        count = 0
        for key, returns in buckets.items():
            if len(returns) < 2:
                continue
            win_rate = sum(1 for r in returns if r > 0) / len(returns)
            avg_ret = sum(returns) / len(returns)
            self.memory.upsert_pattern(
                pattern_type="macd_signal",
                pattern_key=f"macd_{key}",
                description=labels[key],
                win_rate=win_rate,
                avg_return=avg_ret,
                sample_count=len(returns),
            )
            count += 1
        return count

    def _analyze_sector(self, decisions: list[dict]) -> int:
        sector_returns: dict[str, list[float]] = {}
        for d in decisions:
            sector = d.get("sector", "")
            ret = d.get("outcome_pct")
            if not sector or ret is None:
                continue
            sector_returns.setdefault(sector, []).append(ret)

        count = 0
        for sector, returns in sector_returns.items():
            if len(returns) < 2:
                continue
            win_rate = sum(1 for r in returns if r > 0) / len(returns)
            avg_ret = sum(returns) / len(returns)
            self.memory.upsert_pattern(
                pattern_type="sector",
                pattern_key=f"sector_{sector}",
                description=f"{sector} 섹터 투자",
                win_rate=win_rate,
                avg_return=avg_ret,
                sample_count=len(returns),
            )
            count += 1
        return count

    def _analyze_ai_agreement(self, decisions: list[dict]) -> int:
        """Tracks if following AI signal vs ignoring it leads to better outcomes."""
        agreed: list[float] = []
        disagreed: list[float] = []
        for d in decisions:
            signal = d.get("signal", "").upper()
            action = d.get("action", "").upper()
            ret = d.get("outcome_pct")
            if not signal or not action or ret is None:
                continue
            if signal == action:
                agreed.append(ret)
            else:
                disagreed.append(ret)

        count = 0
        for key, returns, desc in [
            ("agreed", agreed, "AI 시그널과 일치하는 결정"),
            ("disagreed", disagreed, "AI 시그널과 반대되는 결정"),
        ]:
            if len(returns) < 2:
                continue
            win_rate = sum(1 for r in returns if r > 0) / len(returns)
            avg_ret = sum(returns) / len(returns)
            self.memory.upsert_pattern(
                pattern_type="ai_agreement",
                pattern_key=f"ai_{key}",
                description=desc,
                win_rate=win_rate,
                avg_return=avg_ret,
                sample_count=len(returns),
            )
            count += 1
        return count

    # ── Rule discovery ────────────────────────────────────────────────────────

    def _discover_rules(self) -> int:
        """
        Converts high-confidence patterns into trading rules.
        win_rate ≥ 0.70 → positive rule; win_rate ≤ 0.35 → warning rule.
        """
        patterns = self.memory.get_patterns(min_samples=3)
        n = 0
        for p in patterns:
            win_rate = p["win_rate"]
            avg_ret = p["avg_return"]
            desc = p["description"]
            if win_rate >= 0.70:
                rule_text = (
                    f"[유리한 패턴] {desc}: "
                    f"승률 {win_rate*100:.0f}%, 평균수익 {avg_ret:+.1f}% "
                    f"(샘플 {p['sample_count']}건)"
                )
                self.memory.add_rule(
                    rule_text=rule_text,
                    category="positive",
                    source="pattern_auto",
                    confidence=win_rate,
                    win_rate=win_rate,
                    sample_count=p["sample_count"],
                )
                n += 1
            elif win_rate <= 0.35:
                rule_text = (
                    f"[주의 패턴] {desc}: "
                    f"승률 {win_rate*100:.0f}%, 평균수익 {avg_ret:+.1f}% "
                    f"(샘플 {p['sample_count']}건) — 신중하게 접근할 것"
                )
                self.memory.add_rule(
                    rule_text=rule_text,
                    category="warning",
                    source="pattern_auto",
                    confidence=1.0 - win_rate,
                    win_rate=win_rate,
                    sample_count=p["sample_count"],
                )
                n += 1
        return n

    # ── Prompt injection text ─────────────────────────────────────────────────

    def get_summary_text(self, ticker: str = "") -> str:
        """
        Returns a Markdown block to inject into Claude analysis prompts.
        Keeps it concise — the most actionable 5 items per section.
        """
        context = self.memory.get_user_context(ticker)
        parts: list[str] = []

        rules = context.get("active_rules", [])[:5]
        if rules:
            parts.append("### 민재의 투자 규칙")
            for r in rules:
                parts.append(f"- {r['rule_text']}")

        patterns = context.get("patterns", [])[:5]
        if patterns:
            parts.append("### 발견된 패턴")
            for p in patterns:
                parts.append(
                    f"- {p['description']}: 승률 {p['win_rate']*100:.0f}%, "
                    f"평균 {p['avg_return']:+.1f}% ({p['sample_count']}건)"
                )

        similar = context.get("similar_decisions", [])
        if similar:
            parts.append(f"### {ticker} 과거 투자 이력")
            for d in similar:
                outcome = (
                    f"{d['outcome_pct']:+.1f}% ({d['outcome_label']})"
                    if d.get("outcome_pct") is not None
                    else "결과 미기록"
                )
                parts.append(
                    f"- {d['created_at'][:10]}: {d['action']} "
                    f"@ ₩{d['price']:,.0f} → {outcome}"
                    if d.get("price") else
                    f"- {d['created_at'][:10]}: {d['action']} → {outcome}"
                )

        recent = context.get("recent_decisions", [])
        if recent:
            parts.append("### 최근 투자 결정 (최신 5건)")
            for d in recent:
                parts.append(
                    f"- {d['created_at'][:10]} {d['ticker']} {d['action']} "
                    f"(신호: {d['signal']})"
                )

        return "\n".join(parts) if parts else "아직 기록된 투자 이력이 없습니다."
