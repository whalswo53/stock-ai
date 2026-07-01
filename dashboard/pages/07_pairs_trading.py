import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import math

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.quant.aggregator import AggregatedPairResult, QuantAggregator
from analysis.quant.mean_reversion import MeanReversionResult
from analysis.quant.pair_scanner import (
    INDUSTRY_GROUPS,
    KMeansPeerDiscovery,
    PairScanner,
    PairScanResult,
    PeerDiscovery,
)
from config.sources import TICKER_KR_NAME, KOSPI_TICKER_MAP, NASDAQ_TICKER_MAP, HK_CN_TICKER_MAP
from data.collectors.price_collector import PriceCollector

# ── Palette ───────────────────────────────────────────────────────────────────
OLS_COLOR    = "#2196F3"
KALMAN_COLOR = "#FF9800"

PRESET_PAIRS = {
    "삼성전자 / SK하이닉스 (반도체)": ("005930.KS", "000660.KS"),
    "AAPL / MSFT (미국 빅테크)":      ("AAPL",       "MSFT"),
    "NVDA / AMD (GPU)":               ("NVDA",       "AMD"),
    "카카오 / 네이버 (플랫폼)":        ("035720.KS",  "035420.KS"),
    "직접 입력":                       None,
}
PRESET_DIRECT     = "직접 입력"
PRESET_DIRECT_IDX = list(PRESET_PAIRS.keys()).index(PRESET_DIRECT)

PERIOD_OPTIONS = {"6개월": "6mo", "1년": "1y", "2년": "2y"}

# ── Ticker display name lookup ─────────────────────────────────────────────────
_NAME_LOOKUP: dict[str, str] = dict(TICKER_KR_NAME)
for _grp in INDUSTRY_GROUPS.values():
    _NAME_LOOKUP.update(_grp["names"])


@st.cache_data(ttl=3600, show_spinner=False)
def _get_name(ticker: str) -> str:
    if ticker in _NAME_LOOKUP:
        return _NAME_LOOKUP[ticker]
    try:
        info = PriceCollector().get_info(ticker)
        return info.get("shortName") or info.get("longName") or ticker
    except Exception:
        return ticker


def _label(ticker: str) -> str:
    name = _get_name(ticker)
    return f"{name} ({ticker})" if name != ticker else ticker


# ── Korean name → ticker resolver ─────────────────────────────────────────────
# Build combined name → ticker map (Korean + English names)
_KR_NAME_TO_TICKER: dict[str, str] = {}
_KR_NAME_TO_TICKER.update(KOSPI_TICKER_MAP)
_KR_NAME_TO_TICKER.update(NASDAQ_TICKER_MAP)
_KR_NAME_TO_TICKER.update(HK_CN_TICKER_MAP)
for _grp in INDUSTRY_GROUPS.values():
    for _t, _n in _grp["names"].items():
        _KR_NAME_TO_TICKER.setdefault(_n, _t)


def _resolve_ticker(raw: str) -> tuple[str, str | None]:
    """
    Converts user input (Korean name or ticker) to a yfinance ticker string.
    Returns (ticker, error_message). On success, error_message is None.
    """
    raw = raw.strip()
    if not raw:
        return "", "종목 코드나 종목명을 입력해주세요."

    has_korean = any("가" <= c <= "힣" for c in raw)

    if not has_korean:
        # Check if it's an English/mixed name in the map (e.g., "NAVER", "AMD", "Apple")
        raw_upper = raw.upper()
        if raw_upper in _KR_NAME_TO_TICKER:
            return _KR_NAME_TO_TICKER[raw_upper], None
        for name, ticker in _KR_NAME_TO_TICKER.items():
            if name.upper() == raw_upper:
                return ticker, None
        # Treat as a raw ticker code
        return raw_upper, None

    # Korean name: exact match first
    if raw in _KR_NAME_TO_TICKER:
        return _KR_NAME_TO_TICKER[raw], None

    # Partial substring match
    matches = {n: t for n, t in _KR_NAME_TO_TICKER.items() if raw in n}
    if len(matches) == 1:
        return list(matches.values())[0], None
    if len(matches) > 1:
        candidates = " · ".join(list(matches.keys())[:4])
        return "", (
            f"'{raw}'와 일치하는 종목이 여러 개입니다: {candidates}. "
            "더 정확한 이름을 입력해주세요."
        )
    return "", (
        f"'{raw}'에 해당하는 종목을 찾을 수 없습니다. "
        "정확한 종목명(예: 삼성전자) 또는 티커(예: 005930.KS)를 입력해주세요."
    )


# ── Scan result table styling ─────────────────────────────────────────────────

def _style_scan(df: pd.DataFrame) -> pd.DataFrame:
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for i, row in df.iterrows():
        pv = row["p-value"]
        if pv < 0.05:
            styles.loc[i, "p-value"] = "color:#26a69a;font-weight:bold"
        elif pv < 0.1:
            styles.loc[i, "p-value"] = "color:#FF9800"
        else:
            styles.loc[i, "p-value"] = "color:#9E9E9E"
        for col in ["A 신호", "B 신호"]:
            sig = row[col]
            if sig == "BUY":
                styles.loc[i, col] = "color:#26a69a;font-weight:bold"
            elif sig == "SELL":
                styles.loc[i, col] = "color:#ef5350;font-weight:bold"
            elif sig == "CLOSE":
                styles.loc[i, col] = "color:#FF9800"
    return styles


def _render_scan_results(
    scan_results: list[PairScanResult],
    source_note: str = "",
    df_key: str = "scan_df",
) -> None:
    scanner = PairScanner()
    df_all  = scanner.to_dataframe(scan_results)
    df_show = df_all.drop(columns=["_ticker_a", "_ticker_b"])
    n_coint = sum(1 for r in scan_results if r.is_cointegrated)

    m1, m2, m3 = st.columns(3)
    m1.metric("검정 쌍 수", len(scan_results))
    m2.metric("공적분 확인",
              f"{n_coint}쌍",
              help="p-value < 0.05 기준으로 장기 동조 관계가 확인된 쌍 수")
    m3.metric("최저 p-value", f"{scan_results[0].pvalue:.4f}" if scan_results else "—",
              help="낮을수록 공적분 관계가 강합니다")

    if source_note:
        st.caption(f"⚠️ {source_note}")

    st.markdown("##### 검정 결과 (p-value 낮은 순 · 상위 5쌍)")
    top5  = df_show.head(5)
    event = st.dataframe(
        top5.style.apply(_style_scan, axis=None),
        width="stretch", hide_index=True,
        selection_mode="single-row", on_select="rerun",
        key=df_key,
    )
    if len(df_show) > 5:
        with st.expander(f"전체 {len(df_show)}쌍 보기"):
            st.dataframe(
                df_show.style.apply(_style_scan, axis=None),
                width="stretch", hide_index=True,
            )

    selected_rows = event.selection.rows if event.selection else []
    if selected_rows:
        picked = scan_results[selected_rows[0]]
        st.session_state["scan_prefill"] = {
            "ticker_a": picked.ticker_a,
            "ticker_b": picked.ticker_b,
            "name_a":   picked.name_a,
            "name_b":   picked.name_b,
        }
        st.success(
            f"**{picked.name_a} ({picked.ticker_a}) / "
            f"{picked.name_b} ({picked.ticker_b})** 선택됨 "
            f"(p={picked.pvalue:.4f}) — "
            "**직접 분석 탭**에서 상세 분석을 확인하세요."
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 퀀트 분석 설정")

    period_label  = st.selectbox("분석 기간", list(PERIOD_OPTIONS.keys()), index=1)
    period        = PERIOD_OPTIONS[period_label]

    st.divider()
    st.subheader("신호 임계값")
    entry_z       = st.slider("진입 Z-score", 1.0, 3.5, 2.0, 0.1,
                               help="이 값을 초과/미달할 때 진입 신호가 발생합니다")
    exit_z        = st.slider("청산 Z-score", 0.1, 1.0, 0.5, 0.1,
                               help="절대값이 이 값 미만이면 청산 신호가 발생합니다")
    zscore_window = st.slider("Z-score 윈도우 (일)", 10, 60, 30, 5,
                               help="Z-score 계산에 사용할 롤링 기간")

    st.divider()
    st.subheader("칼만 필터")
    kalman_delta = st.select_slider(
        "적응 속도 (delta)",
        options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        value=1e-4,
        format_func=lambda v: f"{v:.0e}",
        help="값이 클수록 헤지비율이 빠르게 변합니다. 직접 분석 탭에만 적용됩니다.",
    )


# ── Cached loaders ────────────────────────────────────────────────────────────

def _align_and_serialize(prices: dict[str, pd.Series]) -> dict[str, list]:
    """날짜 교집합으로 정렬 후 직렬화. 종목별 거래일 수 차이로 인한 길이 불일치 방지."""
    if not prices:
        return {"__index__": []}
    common_idx = None
    for series in prices.values():
        common_idx = series.index if common_idx is None else common_idx.intersection(series.index)
    out: dict[str, list] = {t: s.reindex(common_idx).tolist() for t, s in prices.items()}
    out["__index__"] = common_idx.strftime("%Y-%m-%d").tolist()
    return out


def _restore_prices(raw: dict[str, list]) -> dict[str, pd.Series]:
    idx = pd.to_datetime(raw["__index__"])
    return {k: pd.Series(v, index=idx, name=k) for k, v in raw.items() if k != "__index__"}


@st.cache_data(ttl=86400, show_spinner=False)
def _discover_peers(seed_ticker: str, top_n: int, method: str = "sector") -> tuple:
    if method == "kmeans":
        pg = KMeansPeerDiscovery(top_n=top_n).find(seed_ticker)
    else:
        pg = PeerDiscovery(top_n=top_n, scan_depth=80).find(seed_ticker)
    return (pg.tickers, json.dumps(pg.names, ensure_ascii=False), pg.sector, pg.industry, pg.source)


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_peer_prices(tickers_tuple: tuple, period: str) -> dict[str, list]:
    scanner = PairScanner(period=period)
    prices  = scanner.fetch_prices_for(list(tickers_tuple))
    return _align_and_serialize(prices)


@st.cache_data(ttl=1800, show_spinner=False)
def _run_dynamic_scan(
    tickers_tuple: tuple, names_json: str, period: str, window: int, ez: float, xz: float,
    seed_ticker: str = "",
) -> list[PairScanResult]:
    names   = json.loads(names_json)
    raw     = _fetch_peer_prices(tickers_tuple, period)
    prices  = _restore_prices(raw)
    scanner = PairScanner(period=period, zscore_window=window, entry_z=ez, exit_z=xz)
    return scanner.scan_tickers(
        list(tickers_tuple), names, prices,
        seed_ticker=seed_ticker or None,
    )


@st.cache_data(ttl=1800, show_spinner=False)
def _run_pair(ta, tb, period, window, ez, xz, delta) -> AggregatedPairResult:
    return QuantAggregator(
        period=period, zscore_window=window, entry_z=ez, exit_z=xz, kalman_delta=delta,
    ).run_pair(ta, tb)


@st.cache_data(ttl=1800, show_spinner=False)
def _run_single(ticker, period, window, ez, xz) -> MeanReversionResult:
    return QuantAggregator(
        period=period, zscore_window=window, entry_z=ez, exit_z=xz
    ).run_single(ticker)


# ── Page header ───────────────────────────────────────────────────────────────
st.title("📊 통계적 차익거래 분석")
st.caption(
    "**자동 스캔**: 종목 하나 입력 → 동종업종 탐색 → 공적분 쌍 발굴  |  "
    "**직접 분석**: OLS + 칼만 앙상블 / 단일 종목 평균회귀  |  "
    "한글 종목명 입력 지원 (예: 삼성전자, 엔비디아)"
)

tab_scan, tab_direct = st.tabs(["🔍 자동 스캔", "📈 직접 분석"])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1: DYNAMIC PEER SCAN
# ═══════════════════════════════════════════════════════════════════════════════
with tab_scan:
    st.subheader("동종업종 종목 자동 탐색")
    st.caption(
        "종목 코드 또는 **한글 종목명**을 입력하면 동종업종 상위 N개를 자동으로 찾아 "
        "공적분 검정을 실행합니다."
    )

    col_t, col_n, col_btn = st.columns([3, 2, 1])
    with col_t:
        seed_input = st.text_input(
            "기준 종목",
            value="삼성전자",
            placeholder="예: 삼성전자, 005930.KS, 엔비디아, NVDA",
            label_visibility="collapsed",
        ).strip()
    with col_n:
        peer_top_n = st.slider(
            "동종업종 종목 수 N",
            min_value=5, max_value=20, value=10, step=1,
            help="시가총액 상위 N개로 탐색을 제한합니다",
        )
    with col_btn:
        run_discovery = st.button("탐색 시작", type="primary", use_container_width=True)

    disc_method = st.radio(
        "탐색 방식",
        options=["업종 분류 기반", "주가 움직임 기반 (K-means)"],
        index=0,
        horizontal=True,
        help=(
            "**업종 분류 기반**: 네이버 금융 / KRX 업종 코드로 같은 업종 종목 탐색 (빠름)\n\n"
            "**주가 움직임 기반 (K-means)**: 최근 2년 주가 패턴으로 군집화 "
            "— KOSPI/KOSDAQ + S&P500 + 홍콩/중국/대만 종목을 국가 구분 없이 "
            "하나의 글로벌 풀로 합쳐 군집화 (최초 실행 60~90초 소요)"
        ),
    )
    disc_method_key = "kmeans" if "K-means" in disc_method else "sector"

    if disc_method_key == "kmeans":
        st.caption(
            "🌐 **글로벌 통합 풀에서 탐색 중** — KOSPI/KOSDAQ + S&P500 + "
            "INDUSTRY_GROUPS 홍콩/중국/대만 종목을 국가·거래소 구분 없이 "
            "하나의 풀로 합쳐 K-means 클러스터링 (원화 종목은 일별 환율로 USD 환산 후 비교, "
            "최초 실행 후 30분 캐시)"
        )
    else:
        st.caption(
            "🇰🇷 한국: 네이버 금융 업종 분류 + FDR KRX 시가총액 순위  |  "
            "🇺🇸 미국: Yahoo Finance 섹터 분류 + S&P500 구성종목  "
            "(최초 실행 후 30분 캐시)"
        )

    # Resolve Korean name → ticker
    seed_ticker, seed_err = _resolve_ticker(seed_input)

    if seed_err:
        st.error(seed_err)
    else:
        # Show ticker resolution if input was a name
        if seed_ticker != seed_input.strip().upper():
            resolved_name = _get_name(seed_ticker)
            st.caption(f"→ **{resolved_name}** ({seed_ticker}) 로 인식됨")

        disc_key = (seed_ticker, peer_top_n, disc_method_key)
        if run_discovery or st.session_state.get("last_disc_key") == disc_key:
            st.session_state["last_disc_key"] = disc_key

            spinner_msg = (
                f"'{_get_name(seed_ticker)}' 글로벌 통합 풀에서 주가 군집 분석 중… (최초 실행 시 약 60~90초)"
                if disc_method_key == "kmeans"
                else f"'{_get_name(seed_ticker)}' 동종업종 탐색 중… (최초 실행 시 약 10~20초)"
            )
            with st.spinner(spinner_msg):
                try:
                    tickers_list, names_json, sector, industry, source = _discover_peers(
                        seed_ticker, peer_top_n, disc_method_key
                    )
                    disc_error = None
                except (ValueError, RuntimeError) as e:
                    disc_error = str(e)

            if disc_error:
                st.error(disc_error)
            else:
                names        = json.loads(names_json)
                n_discovered = len(tickers_list)
                n_pairs_dyn  = n_discovered - 1  # seed vs 나머지 1:1
                peers_display = "  ·  ".join(f"`{names.get(t, t)}`" for t in tickers_list)

                with st.expander(
                    f"탐색된 종목 {n_discovered}개 ({n_pairs_dyn}쌍 검정 예정)",
                    expanded=True,
                ):
                    c1, c2 = st.columns(2)
                    c1.markdown(f"**업종**: {sector} / {industry}")
                    c2.markdown(f"**출처**: {source}")
                    st.caption(peers_display)

                with st.spinner(f"{n_discovered}개 종목, {n_pairs_dyn}쌍 공적분 검정 중…"):
                    try:
                        scan_results = _run_dynamic_scan(
                            tuple(tickers_list), names_json,
                            period, zscore_window, entry_z, exit_z,
                            seed_ticker,
                        )
                        scan_error = None
                    except Exception as e:
                        scan_error = str(e)

                if scan_error:
                    st.error(f"스캔 오류: {scan_error}")
                elif not scan_results:
                    st.warning("분석 가능한 쌍이 없습니다. 기간을 늘리거나 N을 높여보세요.")
                else:
                    _render_scan_results(
                        scan_results, source_note=source, df_key="scan_df_dynamic"
                    )
        else:
            st.info("기준 종목을 입력하고 **탐색 시작** 버튼을 누르세요.")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2: DIRECT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_direct:

    mode = st.radio("분석 모드", ["페어 분석", "단일 종목 평균회귀"], horizontal=True)

    # ── Ticker inputs ─────────────────────────────────────────────────────────
    if mode == "페어 분석":
        prefill = st.session_state.get("scan_prefill", {})

        if prefill:
            st.info(
                f"자동 스캔에서 선택된 쌍: "
                f"**{prefill['name_a']}** (`{prefill['ticker_a']}`) / "
                f"**{prefill['name_b']}** (`{prefill['ticker_b']}`)"
            )

        prefill_sig    = f"{prefill.get('ticker_a','')}-{prefill.get('ticker_b','')}"
        preset_options = list(PRESET_PAIRS.keys())

        preset_label = st.selectbox(
            "프리셋",
            preset_options,
            index=PRESET_DIRECT_IDX if prefill else 0,
            key=f"preset_sel_{prefill_sig}",
        )
        preset = PRESET_PAIRS[preset_label]

        if preset is None:
            default_a = prefill.get("ticker_a", "AAPL") if prefill else "AAPL"
            default_b = prefill.get("ticker_b", "MSFT") if prefill else "MSFT"
            c1, c2 = st.columns(2)
            raw_a = c1.text_input(
                "종목 A",
                value=default_a,
                key=f"ta_{prefill_sig}",
                help="한글 종목명(예: 삼성전자)이나 티커(예: 005930.KS)를 입력하세요",
            ).strip()
            raw_b = c2.text_input(
                "종목 B",
                value=default_b,
                key=f"tb_{prefill_sig}",
                help="한글 종목명(예: SK하이닉스)이나 티커(예: 000660.KS)를 입력하세요",
            ).strip()
            ticker_a, err_a = _resolve_ticker(raw_a)
            ticker_b, err_b = _resolve_ticker(raw_b)

            # Show resolution feedback
            cap_parts = []
            if ticker_a and ticker_a != raw_a.strip().upper():
                cap_parts.append(f"A → **{ticker_a}** ({_get_name(ticker_a)})")
            if ticker_b and ticker_b != raw_b.strip().upper():
                cap_parts.append(f"B → **{ticker_b}** ({_get_name(ticker_b)})")
            if cap_parts:
                st.caption("   ·   ".join(cap_parts))

            if err_a:
                st.error(f"종목 A: {err_a}")
                st.stop()
            if err_b:
                st.error(f"종목 B: {err_b}")
                st.stop()
        else:
            ticker_a, ticker_b = preset
            st.caption(f"A: {_label(ticker_a)}  ·  B: {_label(ticker_b)}")

        single_ticker = ""

    else:
        raw_single = st.text_input(
            "종목 코드 또는 종목명",
            "AAPL",
            help="한글 종목명(예: 삼성전자)이나 티커(예: 005930.KS, AAPL)를 입력하세요",
        ).strip()
        single_ticker, single_err = _resolve_ticker(raw_single)

        if single_err:
            st.error(single_err)
            st.stop()
        if single_ticker != raw_single.strip().upper():
            st.caption(f"→ **{single_ticker}** ({_get_name(single_ticker)}) 로 인식됨")

        ticker_a = ticker_b = ""

    st.divider()

    # ── Analysis ──────────────────────────────────────────────────────────────
    if mode == "페어 분석":
        label_a = _label(ticker_a)
        label_b = _label(ticker_b)

        with st.spinner(f"'{label_a}' & '{label_b}' 앙상블 분석 중…"):
            try:
                result: AggregatedPairResult = _run_pair(
                    ticker_a, ticker_b, period, zscore_window, entry_z, exit_z, kalman_delta
                )
                error_msg = None
            except ValueError as e:
                error_msg = str(e)

        if error_msg:
            st.error(error_msg)
            st.stop()

        coint  = result.coint_result
        ols    = result.ols_spread
        kalman = result.kalman_spread

        # ── 섹션 1: 공적분 검정 ──────────────────────────────────────────────
        st.subheader("1. 공적분 검정 (Engle-Granger)")
        with st.expander("❓ 공적분이란?", expanded=False):
            st.markdown(
                "두 종목 가격이 **장기적으로 같은 방향으로 움직이는 통계적 관계**를 말합니다.  \n"
                "공적분이 있으면 가격 차이(스프레드)가 일정 범위를 벗어나도 결국 평균으로 "
                "돌아오는 경향이 있어, 이를 이용한 **페어 트레이딩 전략의 기초**가 됩니다.  \n\n"
                "**p-value**: 0.05 미만이면 5% 유의수준에서 공적분을 인정합니다.  \n"
                "**검정 통계량**: 임계값보다 더 작은 음수일수록 공적분 관계가 강합니다."
            )

        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        coint_ok = coint.is_cointegrated
        c1.metric("검정 결과", "공적분 있음 ✅" if coint_ok else "공적분 없음 ❌")
        c2.metric("p-value", f"{coint.pvalue:.4f}",
                  help="낮을수록 공적분 관계가 강합니다. 0.05 미만 = 통계적으로 유의")
        c3.metric("검정 통계량", f"{coint.test_stat:.3f}",
                  help="임계값보다 더 작은 음수일수록 공적분이 강합니다")
        c4.metric("임계값 (5%)", f"{coint.critical_values['5%']:.3f}",
                  help="검정 통계량이 이 값보다 작으면 공적분이 성립합니다")
        st.caption(f"종목쌍: **{label_a}** vs **{label_b}**")

        if coint_ok:
            st.success(
                f"**공적분 관계가 확인되었습니다** (p={coint.pvalue:.4f}).  \n"
                f"두 종목은 장기적으로 같은 방향으로 움직이는 통계적 관계가 있습니다. "
                f"가격 차이가 평균을 벗어나면 다시 좁혀지는 경향이 있어 "
                f"페어 전략이 유효할 가능성이 높습니다."
            )
        else:
            st.warning(
                f"**공적분 관계가 확인되지 않았습니다** (p={coint.pvalue:.4f}).  \n"
                f"두 종목이 장기적으로 같이 움직인다는 통계적 근거가 부족합니다. "
                f"가격 차이가 벌어져도 다시 좁혀진다는 보장이 없어 페어 전략을 신뢰하기 어렵습니다. "
                f"기간을 늘리거나 다른 종목 쌍을 검토해보세요."
            )

        # ── 섹션 2: 모델 기여도 ──────────────────────────────────────────────
        st.subheader("2. 모델별 기여도")
        with st.expander("❓ 두 모델의 차이는?", expanded=False):
            st.markdown(
                "**OLS (고정 헤지비율)**: 분석 기간 전체의 평균 비율로 스프레드를 계산합니다. "
                "관계가 안정적일 때 적합합니다.  \n"
                "**칼만 필터 (동적 헤지비율)**: 시간에 따라 변하는 비율을 실시간 추적합니다. "
                "두 종목의 관계 비율이 변하는 경우에 더 적합합니다.  \n\n"
                "**Z-score**: 현재 스프레드가 역사적 평균에서 몇 표준편차(σ) 떨어져 있는지.  \n"
                "**가중치**: 공적분 강도(OLS)와 헤지비율 안정성(칼만)에 따라 자동 결정됩니다."
            )

        col_sig_a = f"{label_a} 신호"
        col_sig_b = f"{label_b} 신호"

        contrib_rows = []
        for c in result.contributions:
            contrib_rows.append({
                "모델":        c.name,
                "Z-score":     f"{c.zscore:+.3f}",
                "가중치":      f"{c.weight * 100:.1f}%",
                col_sig_a:     c.signal_a,
                col_sig_b:     c.signal_b,
                "신뢰도 지표": c.confidence_label,
            })
        contrib_rows.append({
            "모델":        "**종합 (앙상블)**",
            "Z-score":     f"{result.composite_zscore:+.3f}",
            "가중치":      "100%",
            col_sig_a:     result.signal_a,
            col_sig_b:     result.signal_b,
            "신뢰도 지표": "가중 평균",
        })
        st.dataframe(pd.DataFrame(contrib_rows), width="stretch", hide_index=True)

        w_ols    = result.contributions[0].weight
        w_kalman = result.contributions[1].weight
        wbar_cols = st.columns([w_ols, w_kalman])
        wbar_cols[0].markdown(
            f'<div style="background:{OLS_COLOR};border-radius:4px;padding:4px 8px;'
            f'font-size:12px;text-align:center">OLS {w_ols*100:.0f}%</div>',
            unsafe_allow_html=True,
        )
        wbar_cols[1].markdown(
            f'<div style="background:{KALMAN_COLOR};border-radius:4px;padding:4px 8px;'
            f'font-size:12px;text-align:center;color:#000">Kalman {w_kalman*100:.0f}%</div>',
            unsafe_allow_html=True,
        )

        # Model dominance interpretation
        if w_ols >= w_kalman:
            st.caption(
                f"💡 **OLS가 {w_ols*100:.0f}% 반영됩니다.** "
                "분석 기간 동안 두 종목의 가격 비율이 비교적 안정적으로 유지되었습니다. "
                "고정 헤지비율로 계산한 스프레드를 더 신뢰할 수 있습니다."
            )
        else:
            st.caption(
                f"💡 **칼만 필터가 {w_kalman*100:.0f}% 반영됩니다.** "
                "두 종목의 가격 비율이 시간에 따라 변하고 있어, "
                "이를 동적으로 추적한 칼만 필터 결과가 더 적합합니다."
            )

        # ── 섹션 3: 차트 ─────────────────────────────────────────────────────
        st.subheader("3. 가격·스프레드·Z-score 비교")

        def build_pair_chart(ols_sr, kalman_sr, la, lb, entry_z, exit_z) -> go.Figure:
            dates = ols_sr.dates
            fig = make_subplots(
                rows=4, cols=1, shared_xaxes=True,
                vertical_spacing=0.04, row_heights=[0.30, 0.22, 0.24, 0.24],
                subplot_titles=[
                    f"정규화 가격 ({la} vs {lb})",
                    "칼만 헤지비율 (동적) vs OLS (고정)",
                    "OLS 스프레드 Z-score",
                    "칼만 필터 Z-score",
                ],
            )
            norm_a = ols_sr.price_a / float(ols_sr.price_a.iloc[0]) * 100
            norm_b = ols_sr.price_b / float(ols_sr.price_b.iloc[0]) * 100
            for series, name, color in [(norm_a, la, OLS_COLOR), (norm_b, lb, KALMAN_COLOR)]:
                fig.add_trace(go.Scatter(x=dates, y=series, name=name,
                                         line=dict(color=color, width=1.8)), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=dates, y=kalman_sr.hedge_ratio, name="칼만 β(t)",
                line=dict(color=KALMAN_COLOR, width=1.5),
            ), row=2, col=1)
            fig.add_hline(
                y=ols_sr.hedge_ratio, line_dash="dash", line_color=OLS_COLOR, line_width=1.5,
                annotation_text=f"OLS β={ols_sr.hedge_ratio:.3f}",
                annotation_position="right", row=2, col=1,
            )
            for row_idx, (zseries, name, color) in enumerate([
                (ols_sr.zscore, "OLS Z", OLS_COLOR),
                (kalman_sr.zscore, "Kalman Z", KALMAN_COLOR),
            ], start=3):
                fig.add_trace(go.Scatter(
                    x=dates, y=zseries, name=name,
                    line=dict(color=color, width=1.6), showlegend=False,
                ), row=row_idx, col=1)
                for level, lcolor, dash in [
                    ( entry_z, "rgba(239,83,80,0.55)",  "dash"),
                    (-entry_z, "rgba(38,166,154,0.55)", "dash"),
                    ( exit_z,  "rgba(255,152,0,0.40)",  "dot"),
                    (-exit_z,  "rgba(255,152,0,0.40)",  "dot"),
                    (0,        "rgba(128,128,128,0.25)", "solid"),
                ]:
                    fig.add_hline(y=level, line_dash=dash, line_color=lcolor,
                                  line_width=1, row=row_idx, col=1)
            fig.update_layout(
                height=820, template="plotly_dark",
                margin=dict(l=10, r=10, t=50, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.01,
                            xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
                hovermode="x unified",
            )
            fig.update_yaxes(title_text="정규화 가격", row=1, col=1)
            fig.update_yaxes(title_text="헤지비율 β",  row=2, col=1)
            fig.update_yaxes(title_text="OLS Z",       row=3, col=1)
            fig.update_yaxes(title_text="Kalman Z",    row=4, col=1)
            return fig

        st.plotly_chart(
            build_pair_chart(ols, kalman, label_a, label_b, entry_z, exit_z),
            width="stretch",
        )

        # ── 섹션 4: 종합 신호 ────────────────────────────────────────────────
        st.subheader("4. 종합 신호")

        s1, s2, s3 = st.columns(3)
        s1.metric(f"{label_a} 신호", result.signal_a)
        s2.metric(f"{label_b} 신호", result.signal_b)
        s3.metric(
            "앙상블 Z-score",
            f"{result.composite_zscore:+.3f}",
            help=(
                "현재 스프레드가 역사적 평균에서 몇 표준편차 떨어져 있는지입니다.  "
                f"+{entry_z:.1f} 초과 시 A 고평가(스프레드 과대), "
                f"-{entry_z:.1f} 미만 시 A 저평가(스프레드 과소)."
            ),
        )

        sig = result.signal_a
        z   = result.composite_zscore
        if sig == "WAIT":
            st.info(
                f"**현재 관망** — 두 종목의 가격 차이가 평소 범위(|Z| < {entry_z}σ) 안에 있습니다.  \n"
                "진입 조건이 충족되지 않았습니다. 포지션 없이 기다리세요."
            )
        elif sig == "CLOSE":
            st.success(
                f"**청산 신호** — 스프레드가 정상 범위(|Z| < {exit_z}σ)로 돌아왔습니다.  \n"
                "진입 중인 포지션이 있다면 **지금 청산**하세요. 수익 실현 시점입니다."
            )
        elif sig == "BUY":
            st.success(
                f"**매수/매도 진입 신호** (Z = {z:+.2f}σ)  \n"
                f"**{label_a}** 가격이 상대적으로 너무 낮고, **{label_b}** 가격이 너무 높습니다.  \n"
                f"→ **{label_a} 매수 + {label_b} 매도**를 동시에 진행하세요.  \n"
                f"스프레드가 평균으로 돌아오면 두 포지션 합산 수익이 발생합니다."
            )
        else:
            st.error(
                f"**매도/매수 진입 신호** (Z = {z:+.2f}σ)  \n"
                f"**{label_a}** 가격이 상대적으로 너무 높고, **{label_b}** 가격이 너무 낮습니다.  \n"
                f"→ **{label_a} 매도 + {label_b} 매수**를 동시에 진행하세요.  \n"
                f"스프레드가 평균으로 돌아오면 두 포지션 합산 수익이 발생합니다."
            )

        with st.expander("📋 신호 기준표"):
            st.markdown(
                f"| 앙상블 Z | {label_a} | {label_b} | 의미 |\n"
                f"|---|---|---|---|\n"
                f"| Z > +{entry_z} | SELL (매도) | BUY (매수) | A 고평가 · B 저평가 |\n"
                f"| Z < −{entry_z} | BUY (매수) | SELL (매도) | A 저평가 · B 고평가 |\n"
                f"| \\|Z\\| < {exit_z} | CLOSE (청산) | CLOSE (청산) | 스프레드 정상화 |\n"
                f"| 그 외 | WAIT (관망) | WAIT (관망) | 진입 조건 미충족 |\n\n"
                "**가중치 산출 원리**  \n"
                "- OLS 가중치 = R² × max(0, 1−2·p_value) — 공적분이 강하고 모델 적합도가 높을수록 ↑  \n"
                "- 칼만 가중치 = 헤지비율 안정성 (1 − 변동계수) — 동적 비율이 일관적일수록 ↑  \n"
            )

        with st.expander("최근 20일 Z-score 상세"):
            ols_z  = ols.zscore.dropna().tail(20)
            kal_z  = kalman.zscore.reindex(ols_z.index)
            sp_ols = ols.spread.reindex(ols_z.index)
            tbl = pd.DataFrame({
                "날짜":         ols_z.index.strftime("%Y-%m-%d"),
                "OLS 스프레드": sp_ols.values,
                "OLS Z":        ols_z.values,
                "Kalman Z":     kal_z.values,
            }).reset_index(drop=True)

            def _color_z(val):
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    return ""
                if v > entry_z:     return "color:#ef5350;font-weight:bold"
                if v < -entry_z:    return "color:#26a69a;font-weight:bold"
                if abs(v) < exit_z: return "color:#FF9800"
                return ""

            st.dataframe(
                tbl.style
                .format({"OLS 스프레드": "{:.4f}", "OLS Z": "{:.4f}", "Kalman Z": "{:.4f}"})
                .map(_color_z, subset=["OLS Z", "Kalman Z"]),
                width="stretch",
            )

    # ── SINGLE STOCK MODE ─────────────────────────────────────────────────────
    else:
        single_label = _label(single_ticker)

        with st.spinner(f"'{single_label}' 평균회귀 분석 중…"):
            try:
                mr: MeanReversionResult = _run_single(
                    single_ticker, period, zscore_window, entry_z, exit_z
                )
                error_msg = None
            except ValueError as e:
                error_msg = str(e)

        if error_msg:
            st.error(error_msg)
            st.stop()

        # ── 섹션 1: ADF 단위근 검정 ──────────────────────────────────────────
        st.subheader("1. ADF 단위근 검정 (평균회귀 가능성)")
        with st.expander("❓ 평균회귀란?", expanded=False):
            st.markdown(
                "주가가 장기 평균 수준으로 돌아오려는 성질을 말합니다.  \n"
                "ADF 검정은 가격이 **단위근(랜덤워크)**을 갖는지 검사합니다.  \n"
                "단위근이 없으면 가격이 평균 주변을 맴돌 가능성이 높습니다.  \n\n"
                "**OU 반감기**: 스프레드가 평균에서 절반만큼 되돌아오는 데 걸리는 예상 기간입니다. "
                "짧을수록 빠른 평균회귀 전략에 적합합니다."
            )

        c1, c2, c3 = st.columns([2, 1, 1])
        mr_ok = mr.is_mean_reverting
        c1.metric("검정 결과", "평균회귀 가능 ✅" if mr_ok else "단위근 존재 ❌")
        c2.metric("ADF p-value", f"{mr.adf_pvalue:.4f}",
                  help="낮을수록 평균회귀 성질이 강합니다. 0.05 미만 = 통계적으로 유의")
        hl = mr.half_life_days
        c3.metric(
            "OU 반감기",
            f"{hl:.1f}일" if math.isfinite(hl) else "∞ (비정상)",
            help="스프레드가 평균에서 절반만큼 회귀하는 데 걸리는 예상 일수",
        )
        st.caption(f"종목: **{single_label}**")

        if mr_ok:
            hl_text = f" 반감기 약 **{hl:.0f}일**이므로" if math.isfinite(hl) and hl < 120 else ""
            st.success(
                f"**평균회귀 성질이 확인되었습니다** (p={mr.adf_pvalue:.4f}).  \n"
                f"가격이 장기 평균 주변을 맴도는 경향이 있습니다."
                + (f" {hl_text} 단기 전략에 활용할 수 있습니다." if hl_text else "")
            )
            if math.isfinite(hl) and hl > 60:
                st.info(f"반감기 **{hl:.0f}일** — 회귀 속도가 느려 단기 전략 적용에 주의하세요.")
        else:
            st.warning(
                f"**단위근을 기각하지 못했습니다** (p={mr.adf_pvalue:.4f}).  \n"
                "가격이 추세를 갖고 있어 평균회귀 전략의 신뢰도가 낮습니다. "
                "다른 종목이나 더 긴 기간을 시도해보세요."
            )

        st.subheader("2. 가격 & Z-score")

        def build_single_chart(mr: MeanReversionResult, label: str, entry_z, exit_z) -> go.Figure:
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.06, row_heights=[0.5, 0.5],
                subplot_titles=[f"{label} 가격", f"Z-score (윈도우 {mr.zscore_window}일)"],
            )
            roll_mean = mr.price.rolling(mr.zscore_window).mean()
            fig.add_trace(go.Scatter(x=mr.dates, y=mr.price, name=label,
                                     line=dict(color="#2196F3", width=1.8)), row=1, col=1)
            fig.add_trace(go.Scatter(x=mr.dates, y=roll_mean, name=f"MA{mr.zscore_window}",
                                     line=dict(color="#FF9800", width=1.3, dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(x=mr.dates, y=mr.zscore, name="Z-score",
                                     line=dict(color="#00BCD4", width=1.8),
                                     fill="tozeroy", fillcolor="rgba(0,188,212,0.07)",
                                     showlegend=False), row=2, col=1)
            for level, color, dash in [
                ( entry_z, "rgba(239,83,80,0.55)",  "dash"),
                (-entry_z, "rgba(38,166,154,0.55)", "dash"),
                ( exit_z,  "rgba(255,152,0,0.40)",  "dot"),
                (-exit_z,  "rgba(255,152,0,0.40)",  "dot"),
                (0,        "rgba(128,128,128,0.25)", "solid"),
            ]:
                fig.add_hline(y=level, line_dash=dash, line_color=color, line_width=1, row=2, col=1)
            fig.update_layout(
                height=600, template="plotly_dark",
                margin=dict(l=10, r=10, t=50, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.01,
                            xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
                hovermode="x unified",
            )
            fig.update_yaxes(title_text="가격",    row=1, col=1)
            fig.update_yaxes(title_text="Z-score", row=2, col=1)
            return fig

        st.plotly_chart(build_single_chart(mr, single_label, entry_z, exit_z), width="stretch")

        st.subheader("3. 현재 신호")

        s1, s2 = st.columns(2)
        s1.metric(f"{single_label} 신호", mr.signal)
        z_disp = f"{mr.zscore_latest:+.3f}" if math.isfinite(mr.zscore_latest) else "N/A"
        s2.metric(
            "현재 Z-score", z_disp,
            help=(
                "현재 가격이 이동평균에서 몇 표준편차 떨어져 있는지입니다. "
                f"절대값이 {entry_z}σ를 초과하면 진입 신호가 발생합니다."
            ),
        )

        sig = mr.signal
        zv  = mr.zscore_latest
        hl_note = f" (OU 반감기 약 {hl:.0f}일)" if math.isfinite(hl) else ""

        if sig == "WAIT":
            st.info(
                f"**현재 관망** — {single_label}의 가격이 평소 범위(|Z| < {entry_z}σ) 안에 있습니다.  \n"
                "단기 진입 조건이 충족되지 않았습니다."
            )
        elif sig == "CLOSE":
            st.success(
                f"**청산 신호** — 가격이 평균 부근으로 돌아왔습니다 (|Z| < {exit_z}σ).  \n"
                "매수/매도 포지션이 있다면 **지금 청산**하세요."
            )
        elif sig == "BUY":
            st.success(
                f"**매수 신호** (Z = {zv:+.2f}σ){hl_note}  \n"
                f"{single_label}이 역사적 평균보다 낮은 가격입니다.  \n"
                "평균으로 돌아올 것을 기대하고 **매수**를 고려할 수 있습니다."
            )
        else:
            st.error(
                f"**매도 신호** (Z = {zv:+.2f}σ){hl_note}  \n"
                f"{single_label}이 역사적 평균보다 높은 가격입니다.  \n"
                "평균으로 돌아올 것을 기대하고 **매도(또는 차익 실현)**를 고려할 수 있습니다."
            )

        with st.expander("최근 20일 데이터"):
            z_tail = mr.zscore.dropna().tail(20)
            p_tail = mr.price.reindex(z_tail.index)
            tbl = pd.DataFrame({
                "날짜":    z_tail.index.strftime("%Y-%m-%d"),
                "가격":    p_tail.values,
                "Z-score": z_tail.values,
            }).reset_index(drop=True)

            def _color_z2(val):
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    return ""
                if v > entry_z:     return "color:#ef5350;font-weight:bold"
                if v < -entry_z:    return "color:#26a69a;font-weight:bold"
                if abs(v) < exit_z: return "color:#FF9800"
                return ""

            st.dataframe(
                tbl.style
                .format({"가격": "{:,.2f}", "Z-score": "{:.4f}"})
                .map(_color_z2, subset=["Z-score"]),
                width="stretch",
            )
