"""Page OANDA v20 backwards for multi-year EUR_USD M5 -> data/EURUSD_M5_LONG.csv
Requires OANDA_TOKEN (practice token) in env. Never paste the token anywhere.
"""
import os, time, requests
import pandas as pd

TOKEN = os.environ["OANDA_TOKEN"]
ENV = "https://api-fxpractice.oanda.com"   # practice API host
import sys
INST = sys.argv[1] if len(sys.argv) > 1 else "EUR_USD"
OUT_CSV = sys.argv[2] if len(sys.argv) > 2 else "data/EURUSD_M5_LONG.csv"

START = pd.Timestamp(sys.argv[3] if len(sys.argv) > 3 else "2023-01-01", tz="UTC")
cur_to = pd.Timestamp("2026-06-30 23:55", tz="UTC")
frames = []

while True:
    r = requests.get(f"{ENV}/v3/instruments/{INST}/candles",
        params={"granularity": "M5", "count": 5000,
                "to": cur_to.strftime("%Y-%m-%dT%H:%M:%SZ"), "price": "M"},
        headers={"Authorization": f"Bearer {TOKEN}"}, timeout=30)
    if r.status_code != 200:
        print("STOPPED:", r.status_code, r.text[:200]); break
    candles = [c for c in r.json().get("candles", []) if c["complete"]]
    if not candles:
        print("no more data"); break
    df = pd.DataFrame([{
        "dt": c["time"],
        "open": float(c["mid"]["o"]), "high": float(c["mid"]["h"]),
        "low": float(c["mid"]["l"]), "close": float(c["mid"]["c"]),
        "vol": int(c["volume"]),
    } for c in candles])
    df["dt"] = pd.to_datetime(df["dt"])   # RFC3339 with offset -> tz-aware UTC
    df = df.set_index("dt").sort_index()
    frames.append(df)
    oldest = df.index[0]
    print(f"got {len(df)} bars, oldest {oldest}")
    if oldest <= START:
        break
    cur_to = oldest - pd.Timedelta(minutes=5)
    time.sleep(0.2)

m5 = pd.concat(frames).sort_index()
m5 = m5[~m5.index.duplicated(keep="first")]
m5 = m5[m5.index >= START]
m5.index.name = "datetime"
m5.to_csv(OUT_CSV)
print(f"\nWROTE data/EURUSD_M5_LONG.csv: {len(m5)} bars, {m5.index[0]} -> {m5.index[-1]}")
