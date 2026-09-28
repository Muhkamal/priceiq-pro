"""Page TwelveData backwards for multi-year XAUUSD M5 -> data/XAUUSD_M5_LONG.csv
Same source as the existing trusted CSV (timezone consistent by construction).
Requires TWELVEDATA_API_KEY in env (copy from Render dashboard - never paste it here).
"""
import os, time, requests
import pandas as pd

KEY = os.environ["TWELVEDATA_API_KEY"]
URL = "https://api.twelvedata.com/time_series"
START = pd.Timestamp("2023-01-01", tz="UTC")
cur_end = pd.Timestamp("2026-06-30 23:55", tz="UTC")
size = 5000
frames = []

while True:
    params = {"symbol": "XAU/USD", "interval": "5min", "outputsize": size,
              "apikey": KEY, "timezone": "UTC",
              "end_date": cur_end.strftime("%Y-%m-%d %H:%M")}
    r = requests.get(URL, params=params, timeout=30).json()
    if "error" in r:
        if size > 1000:
            size = 1000
            print("downsizing to", size, ":", r["error"][:80]); continue
        print("STOPPED:", r["error"]); break
    vals = r.get("values", [])
    if not vals:
        print("no more data"); break
    df = pd.DataFrame(vals)
    df["dt"] = pd.to_datetime(df["datetime"]).dt.tz_localize("UTC")  # timezone=UTC requested; returned naive
    df = df.set_index("dt")[["open", "high", "low", "close"]].astype(float).sort_index()
    frames.append(df)
    oldest = df.index[0]
    print(f"got {len(df):>5} bars, oldest {oldest}")
    if oldest <= START:
        break
    cur_end = oldest - pd.Timedelta(minutes=5)
    time.sleep(1)

m5 = pd.concat(frames).sort_index()
m5 = m5[~m5.index.duplicated(keep="first")]
m5 = m5[m5.index >= START]
m5.index.name = "datetime"
m5.to_csv("data/XAUUSD_M5_LONG.csv")
print(f"\nWROTE data/XAUUSD_M5_LONG.csv: {len(m5)} bars, {m5.index[0]} -> {m5.index[-1]}")
