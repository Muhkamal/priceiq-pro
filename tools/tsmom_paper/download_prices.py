"""
Run this on YOUR OWN computer (not in this sandbox -- it has no live market
data access). It pulls live daily prices from Yahoo Finance via yfinance and
writes them in the exact format monthly_signal.py expects.

Setup (one time):
    pip install yfinance pandas

Usage:
    python3 download_prices.py
    (writes prices.csv in this same folder)

Run this once a month (or whenever you want a fresh signal), then run:
    python3 monthly_signal.py prices.csv
"""

import pandas as pd
import yfinance as yf

# Continuous-futures / FX tickers on Yahoo Finance for the 17-instrument
# paper-trading universe chosen for this system (liquid, easy to track,
# spans equities/bonds/metals/energy/FX/ags -- same spirit as the broader
# 130-market backtest, scaled down to something a person can actually watch).
TICKERS = {
    "SP500": "ES=F",
    "NASDAQ": "NQ=F",
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "CRUDE_W": "CL=F",
    "COPPER": "HG=F",
    "GAS_US": "NG=F",
    "US10": "ZN=F",
    "EUR": "EURUSD=X",
    "GBP": "GBPUSD=X",
    "JPY": "JPY=X",       # this is USDJPY on Yahoo; direction is inverted vs
                           # the other pairs (USD is the base, not the quote)
                           # -- fine for the signal (it just trades the pair),
                           # just be aware of this when you look at the sign.
    "AUD": "AUDUSD=X",
    "CAD": "CAD=X",        # USDCAD
    "CHF": "CHF=X",        # USDCHF
    "CORN": "ZC=F",
    "SOYBEAN": "ZS=F",
    "WHEAT": "ZW=F",
}

LOOKBACK_CALENDAR_DAYS = 500  # need 252+63 trading days (~450-470 calendar days); 500 gives margin


def main():
    frames = {}
    for name, ticker in TICKERS.items():
        print(f"Downloading {name} ({ticker})...")
        data = yf.download(ticker, period=f"{LOOKBACK_CALENDAR_DAYS}d", progress=False)
        if data.empty:
            print(f"  WARNING: no data returned for {ticker}, skipping")
            continue
        close = data["Close"]
        if isinstance(close, pd.DataFrame):   # yfinance >=0.2.3x: MultiIndex cols even for one ticker
            close = close.iloc[:, 0]
        close = close.dropna()
        if close.empty:
            print(f"  WARNING: empty Close series for {ticker}, skipping")
            continue
        frames[name] = close

    if not frames:
        raise SystemExit("ERROR: 0 instruments downloaded (network down?) - prices.csv NOT touched")
    df = pd.DataFrame(frames)
    df.index.name = "date"
    df = df.sort_index()
    df.to_csv("prices.csv")
    print(f"\nWrote prices.csv: {len(df)} rows, {df.index[0].date()} .. {df.index[-1].date()}")
    print("Now run: python3 monthly_signal.py prices.csv")


if __name__ == "__main__":
    main()
