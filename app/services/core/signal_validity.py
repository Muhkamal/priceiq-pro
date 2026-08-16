"""
Signal Validity — Prevents chasing stale or premature signals.
"""
from datetime import datetime, timezone
from typing import Dict

VALIDITY_CONFIG = {
    "XAUUSD":  {"max_bars": 2, "max_slip_pct": 0.30},
    "EURUSD":  {"max_bars": 3, "max_slip_pct": 0.25},
    "GBPUSD":  {"max_bars": 2, "max_slip_pct": 0.30},
    "USDCHF":  {"max_bars": 3, "max_slip_pct": 0.25},
    "AUDUSD":  {"max_bars": 3, "max_slip_pct": 0.25},
    "BTCUSD":  {"max_bars": 2, "max_slip_pct": 0.50},
}


class SignalValidity:
    def __init__(self):
        self.config = VALIDITY_CONFIG

    def check(self, pair, signal_entry, signal_sl, current_price,
              signal_time, current_time=None, bars_since_signal=0):
        cfg = self.config.get(pair, self.config["EURUSD"])
        sl_distance = abs(signal_entry - signal_sl) or signal_entry * 0.001
        price_slip = abs(current_price - signal_entry)
        slip_pct = price_slip / sl_distance

        if bars_since_signal >= cfg["max_bars"]:
            return {"valid": False, "reason": f"Expired ({bars_since_signal} bars)", "action": "skip"}

        if slip_pct > cfg["max_slip_pct"]:
            return {"valid": False, "reason": f"Price moved {slip_pct:.0%} of SL", "action": "skip"}

        if slip_pct < 0.05:
            return {"valid": True, "reason": f"At entry", "action": "enter_now"}
        if slip_pct < cfg["max_slip_pct"] * 0.7:
            return {"valid": True, "reason": f"Slipped {slip_pct:.0%} — set pending", "action": "set_pending"}

        return {"valid": True, "reason": "Within range", "action": "enter_now"}


validity_checker = SignalValidity()
