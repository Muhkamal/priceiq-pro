import requests, pandas as pd, os, time

API_KEY = "6d56ac51c23b4cd5b11c64d0ca34d4cf"
os.makedirs("data", exist_ok=True)

def dl(symbol, interval, filename):
    print("Downloading", symbol, interval, "...")
    r = requests.get(
        "https://api.twelvedata.com/time_series",
        params={"symbol": symbol, "interval": interval,
                "outputsize": 5000, "timezone": "UTC", "apikey": API_KEY},
        timeout=30,
    )
    data = r.json()
    if "values" not in data:
        print("API error:", data)
        return
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df[["open", "high", "low", "close"]].to_csv(filename)
    print("Saved", len(df), "bars to", filename)

dl("XAU/USD", "5min", "data/XAUUSD_M5.csv")
time.sleep(3)
dl("XAU/USD", "15min", "data/XAUUSD_M15.csv")
print("DONE")
