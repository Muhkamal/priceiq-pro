"""HistData/MT5 M1 CSVs -> data/XAUUSD_M5_LONG.csv (UTC, runner format).
HistData default: US Eastern FIXED (UTC-5, no DST). Verify with the tz-check
step before trusting anything downstream.
Usage: python3 tools/m1_to_m5.py [--src-offset -5] [--pattern 'data/histdata_raw/*.csv']
"""
import argparse, glob
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--src-offset", type=float, default=-5.0,
                help="source timezone as fixed UTC offset hours (HistData=-5)")
ap.add_argument("--pattern", default="data/histdata_raw/DAT_ASCII_XAUUSD_M1_*.csv")
a = ap.parse_args()

frames = []
for f in sorted(glob.glob(a.pattern)):
    df = pd.read_csv(f, sep=";", header=None,
                     names=["pair", "dt", "open", "high", "low", "close", "vol"])
    df["dt"] = pd.to_datetime(df["dt"], format="%Y%m%d %H%M%S")
    df = df.set_index("dt")[["open", "high", "low", "close", "vol"]].astype(float)
    frames.append(df)
    print(f, len(df))

m1 = pd.concat(frames).sort_index()
m1.index = m1.index + pd.Timedelta(hours=-a.src_offset)  # fixed offset -> UTC
m1 = m1[~m1.index.duplicated(keep="last")]

m5 = m1.resample("5min").agg({"open": "first", "high": "max", "low": "min",
                              "close": "last", "vol": "sum"}).dropna()
out = "data/XAUUSD_M5_LONG.csv"
m5.to_csv(out)
print(f"\nWrote {out}: {len(m5)} M5 bars, {m5.index[0]} -> {m5.index[-1]} UTC")
