"""
PriceIQ Pro — 20% Rule Robustness Stress-Tester
Tests if the system's edge survives parameter variations.
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.data_fetcher import DataFetcher
from app.services.agents.trend_agent import TrendAgent
from app.services.agents.mean_reversion_agent import MeanReversionAgent

async def run_stress_test():
    print("🔬 Starting 20% Rule Robustness Stress-Test...")
    fetcher = DataFetcher()
    
    # Fetch 2 years of data for a major pair (e.g., EURUSD)
    print("📡 Fetching 2 years of EURUSD 1D data...")
    candles = await fetcher.get_candles("EURUSD", "1d", limit=500)
    if not candles:
        print("❌ Failed to fetch data.")
        return

    # Define the baseline parameters and the +/- 20% variations
    # Example: Testing Moving Average lengths for Trend/MeanReversion
    baseline_ma = 50
    variations = [
        int(baseline_ma * 0.8),  # -20% (40)
        baseline_ma,             # Baseline (50)
        int(baseline_ma * 1.2)   # +20% (60)
    ]
    
    print(f"\n📊 Testing MA Lengths: {variations}")
    print("-" * 50)
    
    for ma_length in variations:
        # Instantiate agents with the varied parameter
        # (Assuming your agents accept these kwargs, adjust as needed)
        trend_agent = TrendAgent(ma_fast=ma_length, ma_slow=ma_length*2)
        
        # Run a quick backtest simulation
        wins, losses, total_pnl = 0, 0, 0.0
        position = None
        
        for i in range(ma_length*2, len(candles)):
            # Simplified logic: just check if the agent generates a signal
            # In a real test, you'd run the full orchestrator loop
            signal = trend_agent.evaluate(candles[:i+1], "EURUSD", "trending")
            if signal and not position:
                position = {"entry": candles[i].close, "dir": signal.direction}
            elif position:
                # Simple 2% TP / 1% SL for simulation
                price = candles[i].close
                if position["dir"] == "buy":
                    if price >= position["entry"] * 1.02:
                        wins += 1; total_pnl += 2.0; position = None
                    elif price <= position["entry"] * 0.99:
                        losses += 1; total_pnl -= 1.0; position = None
        
        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        status = "✅ ROBUST" if total_pnl > 0 else "❌ BRITTLE"
        
        print(f"MA={ma_length:3d} | Trades: {total_trades:3d} | WinRate: {win_rate:5.1f}% | PnL: {total_pnl:+6.1f}R | {status}")

    print("-" * 50)
    print("💡 If all 3 variations are profitable, your system passes the 20% Rule!")

if __name__ == "__main__":
    asyncio.run(run_stress_test())
