-- portfolio/schema.sql
-- Supabase 대시보드 > SQL Editor 에서 실행하세요.
-- (Table Editor 가 아닌 SQL Editor 사용)

-- ── 보유 종목 테이블 ────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS portfolio_holdings (
    id           BIGSERIAL PRIMARY KEY,
    ticker       TEXT        NOT NULL,
    name         TEXT        NOT NULL DEFAULT '',
    quantity     FLOAT8      NOT NULL DEFAULT 0,
    target_qty   FLOAT8,
    avg_cost     FLOAT8      NOT NULL DEFAULT 0,
    group_type   TEXT        NOT NULL DEFAULT 'holding',
    accum_period TEXT        NOT NULL DEFAULT '',
    accum_type   TEXT        NOT NULL DEFAULT '',
    accum_value  FLOAT8      NOT NULL DEFAULT 0,
    sector       TEXT                 DEFAULT '',
    notes        TEXT                 DEFAULT '',
    created_at   TEXT        NOT NULL,
    updated_at   TEXT        NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_holdings_ticker ON portfolio_holdings(ticker);
CREATE INDEX IF NOT EXISTS idx_holdings_group  ON portfolio_holdings(group_type);

-- ── 매수 내역 테이블 ────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS purchase_history (
    id          BIGSERIAL PRIMARY KEY,
    holding_id  BIGINT  NOT NULL REFERENCES portfolio_holdings(id) ON DELETE CASCADE,
    buy_date    TEXT    NOT NULL,
    quantity    FLOAT8  NOT NULL,
    price       FLOAT8  NOT NULL,
    created_at  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_history_holding ON purchase_history(holding_id);

-- ── Row Level Security (RLS) ────────────────────────────────────────────────
-- Service Role Key 는 RLS 를 우회하므로 앱 서버에서만 사용하세요.
-- 공개 접근 차단 (anon key 로는 읽기/쓰기 불가)

ALTER TABLE portfolio_holdings  ENABLE ROW LEVEL SECURITY;
ALTER TABLE purchase_history    ENABLE ROW LEVEL SECURITY;
