"""PriceIQ Pro — Fundamentals gate: DXY + US10Y trend (gold), BTC funding (crypto)."""
import asyncio, json, logging, time, urllib.parse, urllib.request
logger = logging.getLogger(__name__)

class FundamentalsGate:
    def __init__(self, refresh_h=6.0):
        self._last, self._refresh_h = 0.0, refresh_h
        self.dxy_chg = self.tnx_chg = self.btc_funding = 0.0

    def _chart_pct(self, symbol):
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}?interval=1d&range=1mo"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r: data = json.load(r)
        closes = [c for c in data["chart"]["result"][0]["indicators"]["quote"][0]["close"] if c]
        if len(closes) < 2: return 0.0
        ref = closes[-6] if len(closes) >= 6 else closes[0]
        return (closes[-1]-ref)/ref*100

    def _funding(self):
        url = "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return float(json.load(r).get("lastFundingRate", 0))*100

    async def maybe_refresh(self):
        if time.time() - self._last < self._refresh_h*3600: return
        self._last = time.time()
        def work():
            try: self.dxy_chg = self._chart_pct("DX-Y.NYB")
            except Exception as e: logger.debug(f"DXY: {e}")
            try: self.tnx_chg = self._chart_pct("^TNX")
            except Exception as e: logger.debug(f"TNX: {e}")
            try: self.btc_funding = self._funding()
            except Exception as e: logger.debug(f"funding: {e}")
        await asyncio.to_thread(work)
        logger.info(f"Fundamentals: {self.status()}")

    def confidence_adjustment(self, pair, direction, cap=0.05) -> float:
        adj = 0.0
        if pair == "XAUUSD":
            if self.dxy_chg > 0.3 and self.tnx_chg > 0.3:   adj = -cap if direction == "buy" else cap
            elif self.dxy_chg < -0.3 and self.tnx_chg < -0.3: adj = cap if direction == "buy" else -cap
        elif pair == "BTCUSD":
            if self.btc_funding > 0.05:   adj = -cap if direction == "buy" else cap*0.5
            elif self.btc_funding < -0.05: adj = -cap if direction == "sell" else cap*0.5
        return adj

    def status(self):
        return f"🏦 Fundamentals: DXY {self.dxy_chg:+.2f}% | US10Y {self.tnx_chg:+.2f}% | BTC funding {self.btc_funding:.3f}%"

fundamentals_gate = FundamentalsGate()
