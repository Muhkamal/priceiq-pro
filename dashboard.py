"""PriceIQ Pro — Dashboard. Run: streamlit run dashboard.py"""
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PriceIQ Pro", layout="wide")
st.title("📈 PriceIQ Pro Dashboard")

def load_json(name, default):
    try:
        with open(name) as f: return json.load(f)
    except Exception: return default

journal = load_json("trade_journal.json", [])
if isinstance(journal, dict): journal = journal.get("entries", [])
mae = load_json("mae_mfe_history.json", [])

df = pd.DataFrame(journal)
if df.empty:
    st.warning("No trades in journal yet."); st.stop()
df["pnl_usd"] = pd.to_numeric(df.get("pnl_usd", 0), errors="coerce").fillna(0)
df["closed_at"] = pd.to_datetime(df.get("closed_at"), errors="coerce")
df = df.sort_values("closed_at")

eq = 100 + df["pnl_usd"].cumsum()
wins, losses = df[df.pnl_usd > 0], df[df.pnl_usd <= 0]
pf = round(wins.pnl_usd.sum()/abs(losses.pnl_usd.sum()), 2) if losses.pnl_usd.sum() else 99
rets = df.pnl_usd/(100 + df.pnl_usd.cumsum().shift(1).fillna(0))
sharpe = round(rets.mean()/rets.std()*(252**0.5), 2) if rets.std() else 0

c = st.columns(6)
c[0].metric("Trades", len(df)); c[1].metric("Win rate", f"{len(wins)/len(df):.0%}")
c[2].metric("Profit factor", pf); c[3].metric("Sharpe", sharpe)
c[4].metric("PnL", f"${df.pnl_usd.sum():+.2f}"); c[5].metric("Equity", f"${eq.iloc[-1]:.2f}")

st.subheader("Equity curve"); st.line_chart(eq.set_axis(df.closed_at))
a, b = st.columns(2)
with a: st.subheader("PnL by agent"); st.bar_chart(df.groupby("agent").pnl_usd.sum())
with b: st.subheader("PnL by pair"); st.bar_chart(df.groupby("pair").pnl_usd.sum())
if mae:
    m = pd.DataFrame(mae)
    st.subheader("MAE vs MFE (R)"); st.scatter_chart(m, x="mae_r", y="mfe_r")
