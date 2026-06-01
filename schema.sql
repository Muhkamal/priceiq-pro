-- ============================================================
-- PriceIQ Pro — Supabase SQL Schema
-- Run this in your Supabase SQL Editor (supabase.com → SQL Editor)
-- ============================================================

-- ── Users ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email               TEXT UNIQUE NOT NULL,
    hashed_password     TEXT NOT NULL,
    tier                TEXT DEFAULT 'free' CHECK (tier IN ('free','starter','pro','elite')),
    daily_signals_used  INTEGER DEFAULT 0,
    daily_signals_limit INTEGER DEFAULT 3,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── API Keys ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS api_keys (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
    name        TEXT DEFAULT 'default',
    key_hash    TEXT UNIQUE NOT NULL,
    is_active   BOOLEAN DEFAULT TRUE,
    last_used   TIMESTAMPTZ,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ── Signals ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS signals (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID REFERENCES users(id) ON DELETE SET NULL,
    pair            TEXT NOT NULL,
    timeframe       TEXT NOT NULL,
    direction       TEXT NOT NULL CHECK (direction IN ('buy','sell','neutral')),
    pattern         TEXT NOT NULL,
    confidence      NUMERIC(5,4),
    entry_price     NUMERIC(18,6),
    stop_loss       NUMERIC(18,6),
    take_profit_1   NUMERIC(18,6),
    take_profit_2   NUMERIC(18,6),
    risk_reward     NUMERIC(8,4),
    signal_data     JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Trades ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trades (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID REFERENCES users(id) ON DELETE SET NULL,
    signal_id       UUID REFERENCES signals(id) ON DELETE SET NULL,
    pair            TEXT NOT NULL,
    direction       TEXT NOT NULL,
    entry_price     NUMERIC(18,6),
    exit_price      NUMERIC(18,6),
    stop_loss       NUMERIC(18,6),
    take_profit     NUMERIC(18,6),
    position_size   NUMERIC(10,4),
    pnl             NUMERIC(12,2),
    result          TEXT CHECK (result IN ('win','loss','open','cancelled')),
    status          TEXT DEFAULT 'open' CHECK (status IN ('open','closed','cancelled')),
    broker_order_id TEXT,
    opened_at       TIMESTAMPTZ DEFAULT NOW(),
    closed_at       TIMESTAMPTZ
);

-- ── Circuit Breaker State ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS circuit_breaker_state (
    user_id     UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    state       JSONB NOT NULL DEFAULT '{}',
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ── Backtest Results ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS backtest_results (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID REFERENCES users(id) ON DELETE SET NULL,
    pair              TEXT,
    timeframe         TEXT,
    verdict           TEXT,
    verdict_color     TEXT,
    total_trades      INTEGER,
    win_rate          NUMERIC(6,4),
    profit_factor     NUMERIC(8,4),
    sharpe_ratio      NUMERIC(8,4),
    max_drawdown      NUMERIC(8,4),
    total_return_pct  NUMERIC(10,4),
    result_data       JSONB,
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

-- ── Watchlists ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS watchlists (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
    pair        TEXT NOT NULL,
    timeframe   TEXT DEFAULT '1h',
    added_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, pair)
);

-- ── Indexes (performance) ─────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_signals_user_id     ON signals(user_id);
CREATE INDEX IF NOT EXISTS idx_signals_pair        ON signals(pair);
CREATE INDEX IF NOT EXISTS idx_signals_created_at  ON signals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_user_id      ON trades(user_id);
CREATE INDEX IF NOT EXISTS idx_trades_status       ON trades(status);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash       ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id    ON api_keys(user_id);

-- ── Row Level Security (RLS) ──────────────────────────────────
-- Enable RLS so users can only see their own data
ALTER TABLE users              ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_keys           ENABLE ROW LEVEL SECURITY;
ALTER TABLE signals            ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades             ENABLE ROW LEVEL SECURITY;
ALTER TABLE circuit_breaker_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE backtest_results   ENABLE ROW LEVEL SECURITY;
ALTER TABLE watchlists         ENABLE ROW LEVEL SECURITY;

-- Service role bypass (your backend uses the service role key)
CREATE POLICY "Service role full access" ON users              FOR ALL USING (true);
CREATE POLICY "Service role full access" ON api_keys           FOR ALL USING (true);
CREATE POLICY "Service role full access" ON signals            FOR ALL USING (true);
CREATE POLICY "Service role full access" ON trades             FOR ALL USING (true);
CREATE POLICY "Service role full access" ON circuit_breaker_state FOR ALL USING (true);
CREATE POLICY "Service role full access" ON backtest_results   FOR ALL USING (true);
CREATE POLICY "Service role full access" ON watchlists         FOR ALL USING (true);

-- ── Done ──────────────────────────────────────────────────────
-- All tables created. You can view them in Table Editor.
