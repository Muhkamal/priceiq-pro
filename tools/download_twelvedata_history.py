"""Download Twelve Data candle history to CSV (paginated backwards).
Usage: python tools/download_twelvedata_history.py XAUUSD 300 65000 data/XAUUSD_M5.csv

Arguments:
  symbol    - trading pair (XAUUSD, EURUSD, etc.)
  interval  - candle interval in seconds (300 = M5, 900 = M15)
  total     - total number of candles to download
  outfile   - output CSV path
"""
import os
import sys
import time
import pandas as pd
import httpx

API_KEY = os.environ.get("TWELVEDATA_API_KEY")
if not API_KEY:
    raise RuntimeError("TWELVEDATA_API_KEY not set")

BASE_URL = "https://api.twelvedata.com/time_series"
TD_SYMBOLS = {"EURUSD": "EUR/USD", "XAUUSD": "XAU/USD", "GBPUSD": "GBP/USD"}

# Map interval seconds to TwelveData format
INTERVAL_MAP = {
    60: "1min",
    300: "5min",
    900: "15min",
    1800: "30min",
    3600: "1h",
    14400: "4h",
    86400: "1day",
}


def fetch_history(symbol: str, interval_sec: int, total: int) -> pd.DataFrame:
    """Download history in paginated chunks."""
    td_symbol = TD_SYMBOLS.get(symbol, symbol)
    interval = INTERVAL_MAP.get(interval_sec)
    if not interval:
        raise ValueError(f"Unsupported interval: {interval_sec}s")
    
    chunk_size = 5000  # TwelveData max per request
    all_data = []
    end_date = None
    
    with httpx.Client(timeout=60) as client:
        while len(all_data) < total:
            params = {
                "symbol": td_symbol,
                "interval": interval,
                "outputsize": min(chunk_size, total - len(all_data)),
                "timezone": "UTC",
                "apikey": API_KEY,
                "format": "JSON",
            }
            if end_date:
                params["end_date"] = end_date
            
            print(f"Fetching chunk {len(all_data)+1}-{len(all_data)+params['outputsize']}...", end=" ")
            r = client.get(BASE_URL, params=params)
            r.raise_for_status()
            data = r.json()
            
            if data.get("status") == "error":
                print(f"\nAPI error: {data.get('message', 'unknown')}")
                break
            
            values = data.get("values")
            if not values:
                print("\nNo more data")
                break
            
            print(f"got {len(values)} bars")
            all_data = values + all_data  # prepend (oldest first)
            
            # Set end_date for next pagination
            oldest = values[-1]["datetime"]
            end_date = oldest
            
            # Rate limit: TwelveData free tier = 8 requests/min
            time.sleep(8)
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data).drop_duplicates("datetime")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    df["volume"] = 1000
    
    # Drop forming candle
    now = pd.Timestamp.now(tz="UTC")
    df = df[df.index + pd.Timedelta(seconds=interval_sec) <= now]
    
    return df[["open", "high", "low", "close", "volume"]]


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    
    symbol = sys.argv[1]
    interval = int(sys.argv[2])
    total = int(sys.argv[3])
    outfile = sys.argv[4]
    
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    
    print(f"Downloading {total} bars of {symbol} @ {interval}s intervals...")
    df = fetch_history(symbol, interval, total)
    
    if df.empty:
        print("FAILED: no candles returned")
        sys.exit(1)
    
    df.to_csv(outfile)
    print(f"\nSaved {len(df)} candles -> {outfile}")
    print(f"  Range: {df.index[0]} to {df.index[-1]}")
    print(f"  Days: {(df.index[-1] - df.index[0]).days}")
