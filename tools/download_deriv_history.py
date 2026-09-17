"""Download Deriv synthetic candle history to CSV (paginated backwards).
Usage: python3 tools/download_deriv_history.py R_75 300 15000 data/V75_M5.csv"""
import asyncio, json, os, sys
import pandas as pd
import websockets

APP_ID = os.environ.get("DERIV_APP_ID", "1089")
TOKEN = os.environ.get("DERIV_API_TOKEN", "")
WS = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

async def fetch(symbol, granularity, total):
    out = []
    end = "latest"
    while len(out) < total:
        async with websockets.connect(WS, open_timeout=15, close_timeout=5) as ws:
            if TOKEN:
                await ws.send(json.dumps({"authorize": TOKEN}))
                a = json.loads(await ws.recv())
                if "error" in a:
                    print("auth error:", a["error"].get("message")); break
            await ws.send(json.dumps({
                "ticks_history": symbol, "adjust_start_time": 1,
                "count": min(5000, total - len(out)), "end": end,
                "style": "candles", "granularity": granularity}))
            r = json.loads(await ws.recv())
        c = r.get("candles") or []
        if not c:
            break
        out = c + out
        end = c[0]["epoch"] - 1
        print(f"  fetched {len(out)}/{total}", end="\r")
        await asyncio.sleep(1)
    if not out:
        return None
    df = pd.DataFrame(out).drop_duplicates("epoch").sort_values("epoch")
    df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
    df = df.set_index("datetime")
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    df = df[["open", "high", "low", "close"]]
    now = pd.Timestamp.now(tz="UTC")
    df = df[df.index + pd.Timedelta(seconds=granularity) <= now]
    return df

if __name__ == "__main__":
    symbol, gran, total, outfile = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    print(f"Downloading {symbol} ({gran}s candles, target {total} bars)...")
    df = asyncio.run(fetch(symbol, gran, total))
    if df is None or df.empty:
        print("FAILED: no candles returned")
    else:
        df.to_csv(outfile)
        print(f"\nSaved {len(df)} candles -> {outfile}  ({df.index[0]} .. {df.index[-1]})")
