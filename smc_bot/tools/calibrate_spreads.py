"""Calibrate synthetic spreads by logging live bid/ask for 24 hours."""
import asyncio
import json
import os
import time
import statistics
import websockets
from collections import defaultdict

APP_ID = os.environ.get("DERIV_APP_ID", "1089")
TOKEN = os.environ.get("DERIV_API_TOKEN", "")
WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

async def log_spreads(symbols: list, duration_hours: int = 24):
    """Subscribe to tick streams and log bid/ask spreads."""
    
    spreads = defaultdict(list)
    end_time = time.time() + (duration_hours * 3600)
    
    async with websockets.connect(WS_URL) as ws:
        if TOKEN:
            await ws.send(json.dumps({"authorize": TOKEN}))
            await ws.recv()
        
        # Subscribe to all symbols
        for sym in symbols:
            await ws.send(json.dumps({
                "ticks": sym,
                "subscribe": 1
            }))
        
        while time.time() < end_time:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                
                if "tick" in msg:
                    tick = msg["tick"]
                    symbol = tick["symbol"]
                    bid = tick.get("bid")
                    ask = tick.get("ask")
                    
                    if bid and ask:
                        spread = ask - bid
                        spreads[symbol].append(spread)
                        
                        # Progress update every 100 ticks
                        if len(spreads[symbol]) % 100 == 0:
                            avg = statistics.mean(spreads[symbol][-100:])
                            print(f"{symbol}: {len(spreads[symbol])} samples, recent avg spread: {avg:.4f}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error: {e}")
                break
    
    # Summary
    print("\n" + "="*60)
    print("SPREAD CALIBRATION RESULTS")
    print("="*60)
    
    for symbol, values in spreads.items():
        if not values:
            continue
        
        mean_spread = statistics.mean(values)
        median_spread = statistics.median(values)
        p95 = sorted(values)[int(len(values) * 0.95)]
        
        print(f"\n{symbol}:")
        print(f"  Samples: {len(values)}")
        print(f"  Mean:    {mean_spread:.4f}")
        print(f"  Median:  {median_spread:.4f}")
        print(f"  P95:     {p95:.4f}")
        print(f"  → Use for SPREADS config: {median_spread:.4f}")

if __name__ == "__main__":
    symbols = ["R_75", "STPINDX"]  # Add more as needed
    asyncio.run(log_spreads(symbols, duration_hours=24))
