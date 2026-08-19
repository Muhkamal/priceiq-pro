"""PriceIQ Pro — Statistical Arbitrage (pairs) monitor. Alerts on stretched spreads."""
import logging, math
from datetime import datetime, timezone
from typing import Dict, List, Tuple
logger = logging.getLogger(__name__)

class StatArbMonitor:
    PAIRS = [("EURUSD", "GBPUSD")]   # add more cointegrated pairs here

    def __init__(self, lookback=200, alert_z=2.0, cooldown_h=6.0):
        self._closes: Dict[str, List[float]] = {}
        self._last_alert: Dict[str, float] = {}
        self.lookback, self.alert_z, self.cooldown_h = lookback, alert_z, cooldown_h

    def feed(self, pair: str, candles):
        closes = [c.close for c in candles if getattr(c, "close", None)]
        if closes: self._closes[pair] = closes[-300:]

    def _z(self, a, b):
        ca, cb = self._closes.get(a), self._closes.get(b)
        if not ca or not cb: return None
        n = min(len(ca), len(cb), self.lookback)
        if n < 60: return None
        la = [math.log(x) for x in ca[-n:]]; lb = [math.log(x) for x in cb[-n:]]
        ma, mb = sum(la)/n, sum(lb)/n
        cov = sum((la[i]-ma)*(lb[i]-mb) for i in range(n))
        var = sum((lb[i]-mb)**2 for i in range(n)) or 1e-9
        beta = cov/var
        sp = [la[i]-beta*lb[i] for i in range(n)]
        m = sum(sp)/n; sd = (sum((s-m)**2 for s in sp)/n)**0.5 or 1e-9
        return (sp[-1]-m)/sd, beta

    def scan(self) -> List[Tuple]:
        out = []
        for a, b in self.PAIRS:
            r = self._z(a, b)
            if r: out.append((a, b, r[0], r[1]))
        return out

    def confidence_adjustment(self, pair, direction, cap=0.05) -> float:
        for a, b, z, _ in self.scan():
            if pair not in (a, b) or abs(z) < 1.5: continue
            over = (pair == a and z > 0) or (pair == b and z < 0)
            if over:  return -cap if direction == "buy" else cap*0.5
            else:     return -cap if direction == "sell" else cap*0.5
        return 0.0

    async def maybe_alert(self, telegram):
        if not telegram: return
        now = datetime.now(timezone.utc).timestamp()
        for a, b, z, beta in self.scan():
            key = f"{a}/{b}"
            if abs(z) < self.alert_z: continue
            if now - self._last_alert.get(key, 0) < self.cooldown_h*3600: continue
            self._last_alert[key] = now
            l1 = f"SELL {a}" if z > 0 else f"BUY {a}"
            l2 = f"BUY {b}" if z > 0 else f"SELL {b}"
            try:
                await telegram.send_message(
                    f"📊 <b>STATARB ALERT</b> {key}\nSpread z-score: {z:+.2f} (stretched)\n"
                    f"Idea: {l1} + {l2} (mean reversion, 1-3 days)\nBeta: {beta:.2f}\n"
                    f"⚠️ Manual review required — correlated pair trade.")
            except Exception as e: logger.debug(f"statarb alert: {e}")

stat_arb = StatArbMonitor()
