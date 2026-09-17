import yfinance as yf
import os
import pandas as pd
import time

os.makedirs("data", exist_ok=True)
ticker = "GC=F"

print(f"Downloading Gold M5 data (last 60 days) using {ticker}...")
df_m5 = yf.download(ticker, period="60d", interval="5m", progress=False, auto_adjust=True)

if df_m5 is None or df_m5.empty:
    print("❌ Failed to download M5 data.")
else:
    if isinstance(df_m5.columns, pd.MultiIndex):
        df_m5.columns = df_m5.columns.get_level_values(0)
    df_m5.columns = [str(c).lower() for c in df_m5.columns]
    df_m5 = df_m5[['open', 'high', 'low', 'close']]
    if hasattr(df_m5.index, 'tz'):
        if df_m5.index.tz is None:
            df_m5.index = df_m5.index.tz_localize("UTC")
        else:
            df_m5.index = df_m5.index.tz_convert("UTC")
    df_m5.to_csv("data/XAUUSD_M5.csv")
    print(f"✅ Saved {len(df_m5)} M5 bars to data/XAUUSD_M5.csv")

print("\nWaiting 3 seconds so Yahoo doesn't block us...")
time.sleep(3)

print(f"Downloading Gold M15 data (last 60 days) using {ticker}...")
df_m15 = yf.download(ticker, period="60d", interval="15m", progress=False, auto_adjust=True)

if df_m15 is None or df_m15.empty:
    print("❌ Failed to download M15 data.")
else:
    if isinstance(df_m15.columns, pd.MultiIndex):
        df_m15.columns = df_m15.columns.get_level_values(0)
    df_m15.columns = [str(c).lower() for c in df_m15.columns]
    df_m15 = df_m15[['open', 'high', 'low', 'close']]
    if hasattr(df_m15.index, 'tz'):
        if df_m15.index.tz is None:
            df_m15.index = df_m15.index.tz_localize("UTC")
        else:
            df_m15.index = df_m15.index.tz_convert("UTC")
    df_m15.to_csv("data/XAUUSD_M15.csv")
    print(f"✅ Saved {len(df_m15)} M15 bars to data/XAUUSD_M15.csv")

print("\n🎉 Data download complete!")
