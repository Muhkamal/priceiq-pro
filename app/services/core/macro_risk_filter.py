"""
Macro Risk Filter (The "Go-To-Cash" Switch)
Based on Rayner Teo's Breakout System: Stay in cash if the broad market is bearish.
"""
import httpx
import asyncio
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class MacroRiskFilter:
    def __init__(self):
        # We use SPY (S&P 500 ETF) as the proxy for the broad stock market
        self.symbol = "SPY"
        self.ma_period = 200  # 200-day SMA (approx 40 weeks)
        self._cache = {"date": None, "state": "RISK_ON", "close": 0, "sma": 0}

    async def get_market_state(self) -> str:
        """Returns 'RISK_ON', 'RISK_OFF', or 'NEUTRAL'"""
        today = datetime.now(timezone.utc).date()
        if self._cache["date"] == today:
            return self._cache["state"]

        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{self.symbol}"
            params = {"interval": "1d", "range": "1y", "includePrePost": "false"}
            headers = {"User-Agent": "Mozilla/5.0"}
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
            closes = [c for c in closes if c is not None]

            if len(closes) < self.ma_period:
                return "NEUTRAL"

            current_close = closes[-1]
            sma_200 = sum(closes[-self.ma_period:]) / self.ma_period

            state = "RISK_ON" if current_close > sma_200 else "RISK_OFF"
            
            self._cache = {
                "date": today, 
                "state": state, 
                "close": current_close, 
                "sma": sma_200
            }
            logger.info(f"🌍 MacroFilter: {self.symbol} Close={current_close:.2f} vs 200SMA={sma_200:.2f} -> {state}")
            return state
            
        except Exception as e:
            logger.warning(f"MacroFilter failed to fetch {self.symbol}: {e}")
            return "NEUTRAL"

macro_risk_filter = MacroRiskFilter()
