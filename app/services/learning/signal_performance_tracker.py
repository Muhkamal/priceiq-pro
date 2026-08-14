"""
Signal Performance Tracker
Records, resolves, and learns from every signal fired by V5.
"""
import json
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

from app.services.ml.signal_outcome_predictor import outcome_predictor

DB_PATH = Path("signal_performance_db.json")
MAX_SIGNALS = 2000
MIN_SAMPLE_SIZE = 5
LOOKBACK_WINDOW = 100


class SignalPerformanceTracker:
    """
    Tracks signal outcomes and auto-adjusts confidence thresholds
    per (pair, agent, regime) based on rolling performance.
    """

    def __init__(self, db_path: str = "signal_performance_db.json"):
        self.db_path = Path(db_path)
        self._lock = asyncio.Lock()
        self._data: Dict[str, Any] = {"signals": [], "thresholds": {}, "version": 1}
        self._load()

    def _load(self):
        if self.db_path.exists():
            try:
                with open(self.db_path, "r") as f:
                    self._data = json.load(f)
                logger.info(f"Tracker loaded {len(self._data.get('signals', []))} signals")
            except Exception as e:
                logger.warning(f"Tracker load failed: {e}")
                self._data = {"signals": [], "thresholds": {}, "version": 1}
        else:
            self._data = {"signals": [], "thresholds": {}, "version": 1}

    async def _save(self):
        async with self._lock:
            try:
                signals = self._data.get("signals", [])
                if len(signals) > MAX_SIGNALS:
                    self._data["signals"] = signals[-MAX_SIGNALS:]
                with open(self.db_path, "w") as f:
                    json.dump(self._data, f, indent=2, default=str)
            except Exception as e:
                logger.warning(f"Tracker save failed: {e}")

    @staticmethod
    def _get_session(dt: datetime) -> str:
        h = dt.hour
        if 7 <= h < 12:
            return "london"
        if 12 <= h < 16:
            return "london_newyork"
        if 16 <= h < 21:
            return "newyork"
        if 21 <= h or h < 2:
            return "newyork_asia"
        if 2 <= h < 7:
            return "tokyo"
        return "other"

    async def record(self, result) -> str:
        """Call when a signal fires."""
        signal_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc)
        entry = float(getattr(result, "fill_price", 0))
        sl = float(getattr(result, "stop_loss", 0))
        tp1 = float(getattr(result, "take_profit_1", 0))
        tp2 = float(getattr(result, "take_profit_2", 0))
        risk = abs(entry - sl) if entry != sl else 1e-9

        # Get AI model prediction + ID
        model_signal_id = None
        try:
            _, _, model_signal_id = outcome_predictor.predict(result, candles)
        except Exception:
            pass

        record = {
            "id": signal_id,
            "model_signal_id": model_signal_id,
            "timestamp": now.isoformat(),
            "pair": getattr(result, "pair", "UNKNOWN"),
            "direction": getattr(result, "direction", ""),
            "agent": getattr(result, "agent_used", ""),
            "regime": getattr(result, "regime", ""),
            "session": self._get_session(now),
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp1_r": round(abs(tp1 - entry) / risk, 2),
            "tp2_r": round(abs(tp2 - entry) / risk, 2),
            "confidence": float(getattr(result, "confidence", 0)),
            "ev": float(getattr(result, "expected_value", 0)),
            "lots": float(getattr(result, "adjusted_lots", 0)),
            "outcome": "open",
            "pnl_r": 0.0,
            "resolved_at": None,
            "bars_held": 0,
        }

        async with self._lock:
            self._data["signals"].append(record)

        await self._save()
        logger.info(f"Signal recorded: {signal_id} {record['pair']} {record['direction']}")
        return signal_id

    async def update_with_candles(self, pair: str, candles: List[Any]) -> List[Dict]:
        """
        Check open signals for this pair against new candles.
        Returns newly resolved signals for Telegram broadcasting.
        """
        if not candles:
            return []

        newly_resolved = []

        async with self._lock:
            open_sigs = [
                s for s in self._data["signals"]
                if s["pair"] == pair and s["outcome"] == "open"
            ]

            for sig in open_sigs:
                sig_ts = datetime.fromisoformat(sig["timestamp"].replace("Z", "+00:00"))
                subsequent = [
                    c for c in candles
                    if hasattr(c, "timestamp") and c.timestamp > sig_ts
                ]
                if not subsequent:
                    continue

                outcome, pnl_r, bars_held = self._resolve_signal(sig, subsequent)
                if outcome != "open":
                    sig["outcome"] = outcome
                    sig["pnl_r"] = pnl_r
                    # Feed AI model
                    try:
                        msid = sig.get("model_signal_id")
                        if msid:
                            await outcome_predictor.on_signal_resolved(msid, outcome, pnl_r)
                    except Exception as e:
                        logger.debug(f"AI feedback failed: {e}")
                    sig["bars_held"] = bars_held
                    sig["resolved_at"] = datetime.now(timezone.utc).isoformat()
                    newly_resolved.append(sig.copy())
                    logger.info(
                        f"Resolved: {sig['id']} {pair} {outcome} {pnl_r:+.2f}R"
                    )

        if newly_resolved:
            await self._save()
            await self._update_thresholds()

        return newly_resolved

    def _resolve_signal(self, sig: Dict, candles: List[Any]) -> tuple:
        direction = sig["direction"]
        sl = sig["sl"]
        tp1 = sig["tp1"]
        tp2 = sig["tp2"]
        bars_held = len(candles)

        # 24-candle timeout (24h on 1H timeframe)
        if bars_held >= 24:
            last_close = float(getattr(candles[-1], "close", sig["entry"]))
            risk = abs(sig["entry"] - sl) if sig["entry"] != sl else 1e-9
            if direction == "buy":
                pnl_r = round((last_close - sig["entry"]) / risk, 2)
            else:
                pnl_r = round((sig["entry"] - last_close) / risk, 2)
            return "timeout", pnl_r, bars_held

        for c in candles:
            high = float(getattr(c, "high", 0))
            low = float(getattr(c, "low", 0))

            if direction == "buy":
                if low <= sl:
                    return "loss", -1.0, bars_held
                if high >= tp2:
                    return "tp2", sig["tp2_r"], bars_held
                if high >= tp1:
                    return "tp1", sig["tp1_r"], bars_held
            else:  # sell
                if high >= sl:
                    return "loss", -1.0, bars_held
                if low <= tp2:
                    return "tp2", sig["tp2_r"], bars_held
                if low <= tp1:
                    return "tp1", sig["tp1_r"], bars_held

        return "open", 0.0, bars_held

    async def _update_thresholds(self):
        """Auto-adjust confidence thresholds from rolling performance."""
        signals = self._data.get("signals", [])
        if not signals:
            return

        groups: Dict[str, List[Dict]] = {}
        for s in signals[-LOOKBACK_WINDOW:]:
            if s["outcome"] == "open":
                continue
            key = f"{s['pair']}::{s['agent']}::{s['regime']}"
            groups.setdefault(key, []).append(s)

        thresholds = self._data.setdefault("thresholds", {})

        for key, group in groups.items():
            if len(group) < MIN_SAMPLE_SIZE:
                continue
            wins = sum(1 for g in group if g["outcome"] in ("tp1", "tp2"))
            wr = wins / len(group)
            avg_pnl = sum(g["pnl_r"] for g in group) / len(group)

            pair, agent, regime = key.split("::")
            current = thresholds.get(pair, {}).get(agent, {}).get(regime, 0.55)

            if wr >= 0.60 and avg_pnl > 0.5:
                new_thresh = max(0.40, current - 0.02)
            elif wr < 0.40 or avg_pnl < -0.2:
                new_thresh = min(0.85, current + 0.03)
            else:
                new_thresh = current

            thresholds.setdefault(pair, {}).setdefault(agent, {})[regime] = round(new_thresh, 2)

        await self._save()

    def get_dynamic_threshold(self, pair: str, agent: str, regime: str) -> float:
        try:
            return self._data["thresholds"][pair][agent][regime]
        except KeyError:
            return 0.55

    def get_recommended_boost(self, pair: str, agent: str, regime: str) -> float:
        """
        Returns -0.05 to +0.05 confidence boost based on recent
        (pair, agent, regime) performance.
        """
        recent = [
            s for s in self._data.get("signals", [])[-LOOKBACK_WINDOW:]
            if s["pair"] == pair and s["agent"] == agent
            and s["regime"] == regime and s["outcome"] != "open"
        ]
        if len(recent) < MIN_SAMPLE_SIZE:
            return 0.0
        wins = sum(1 for r in recent if r["outcome"] in ("tp1", "tp2"))
        wr = wins / len(recent)
        avg_pnl = sum(r["pnl_r"] for r in recent) / len(recent)

        if wr >= 0.65 and avg_pnl > 1.0:
            return 0.05
        elif wr >= 0.55 and avg_pnl > 0.3:
            return 0.02
        elif wr < 0.35:
            return -0.05
        elif avg_pnl < -0.3:
            return -0.03
        return 0.0

    def get_stats(self, pair: Optional[str] = None, agent: Optional[str] = None,
                  regime: Optional[str] = None, lookback: int = LOOKBACK_WINDOW) -> Dict:
        signals = self._data.get("signals", [])
        filtered = [
            s for s in signals[-lookback:]
            if s["outcome"] != "open"
            and (pair is None or s["pair"] == pair)
            and (agent is None or s["agent"] == agent)
            and (regime is None or s["regime"] == regime)
        ]

        if not filtered:
            return {"count": 0, "win_rate": None, "avg_pnl_r": 0, "total_r": 0}

        wins = sum(1 for s in filtered if s["outcome"] in ("tp1", "tp2"))
        total = len(filtered)
        pnl_list = [s["pnl_r"] for s in filtered]
        avg_pnl = sum(pnl_list) / total

        return {
            "count": total,
            "win_rate": round(wins / total, 2),
            "avg_pnl_r": round(avg_pnl, 2),
            "total_r": round(sum(pnl_list), 2),
            "tp1": sum(1 for s in filtered if s["outcome"] == "tp1"),
            "tp2": sum(1 for s in filtered if s["outcome"] == "tp2"),
            "sl": sum(1 for s in filtered if s["outcome"] == "loss"),
            "timeout": sum(1 for s in filtered if s["outcome"] == "timeout"),
        }

    def get_leaderboard(self, top_n: int = 10) -> List[Dict]:
        signals = self._data.get("signals", [])
        combos: Dict[str, List[Dict]] = {}
        for s in signals:
            if s["outcome"] == "open":
                continue
            key = f"{s['agent']} | {s['regime']} | {s['pair']}"
            combos.setdefault(key, []).append(s)

        results = []
        for key, group in combos.items():
            if len(group) < MIN_SAMPLE_SIZE:
                continue
            wins = sum(1 for g in group if g["outcome"] in ("tp1", "tp2"))
            wr = wins / len(group)
            avg_pnl = sum(g["pnl_r"] for g in group) / len(group)
            results.append({
                "combo": key,
                "count": len(group),
                "win_rate": round(wr, 2),
                "avg_pnl_r": round(avg_pnl, 2),
                "total_r": round(sum(g["pnl_r"] for g in group), 2),
            })

        results.sort(key=lambda x: (x["win_rate"], x["avg_pnl_r"]), reverse=True)
        return results[:top_n]

    def get_pair_stats(self) -> Dict[str, Dict]:
        pairs = {s["pair"] for s in self._data.get("signals", []) if s["outcome"] != "open"}
        return {p: self.get_stats(pair=p) for p in sorted(pairs)}

    def export_summary(self) -> Dict:
        return {
            "total_recorded": len(self._data.get("signals", [])),
            "open_signals": sum(1 for s in self._data.get("signals", []) if s["outcome"] == "open"),
            "thresholds": self._data.get("thresholds", {}),
        }


tracker = SignalPerformanceTracker()
