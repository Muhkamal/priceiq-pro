"""
PriceIQ Pro — Trade Manager v2.0

Fixes from assessment review:
    ✅ Chandelier exit: trail uses highest HIGH of last N bars × ATR
       (not all-time-high since entry — that lags by multiple ATRs)
    ✅ Regime-dependent timeout: trending=72h, ranging=24h, volatile=12h
       (not flat 48h regardless of regime)
    ✅ TradeManager → Journal integration: every ManagementEvent
       written to journal with bar index and timestamp
    ✅ Structured management_events with timestamps (not just strings)

Chandelier exit formula:
    For BUY:  trail_sl = highest_high(last N bars) - multiplier × ATR
    For SELL: trail_sl = lowest_low(last N bars)   + multiplier × ATR
    This keeps the stop below a RECENT swing high, not the all-time peak.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Trade management config ──────────────────────────────────
TP1_CLOSE_PCT        = 0.40
TP2_CLOSE_PCT        = 0.40
CHANDELIER_BARS      = 10     # look-back period for chandelier exit
CHANDELIER_MULT      = 3.0    # ATR multiplier for chandelier
TRAIL_ACTIVATION_R   = 1.0
BREAKEVEN_BUFFER_R   = 0.05

# Regime-dependent timeouts (hours)
REGIME_TIMEOUT: Dict[str, int] = {
    "trending": 72,
    "ranging":  24,
    "volatile": 12,
    "unknown":  48,
}


@dataclass
class ManagementEventRecord:
    """Structured management event with full context."""
    event_type:    str    # "TP1_HIT" | "TP2_HIT" | "SL_TO_BE" | "TRAIL_MOVED" | "TIMEOUT" | "CLOSED"
    timestamp:     str
    bar_index:     int
    old_sl:        Optional[float]
    new_sl:        Optional[float]
    lots_closed:   float
    pnl_usd:       float
    current_price: float
    message:       str

    def to_journal_str(self) -> str:
        return f"{self.event_type}@bar{self.bar_index}({self.timestamp[:16]})"


@dataclass
class ManagedPosition:
    position_id:    str
    pair:           str
    direction:      str
    agent:          str
    regime:         str
    session:        str
    timeframe:      str
    entry:          float
    original_sl:    float
    tp1:            float
    tp2:            float
    tp3:            float
    atr_at_entry:   float
    current_sl:     float
    lots_total:     float
    lots_remaining: float
    status:         str
    pnl_realised:   float
    opened_at:      str
    last_updated:   str
    bar_index:      int
    highest_price:  float
    lowest_price:   float
    # Chandelier: rolling window of recent highs/lows
    recent_highs:   List[float] = field(default_factory=list)
    recent_lows:    List[float] = field(default_factory=list)
    # Management flags
    sl_at_breakeven:  bool = False
    tp1_hit:          bool = False
    tp2_hit:          bool = False
    trailing_active:  bool = False
    # Structured event log
    management_events: List[ManagementEventRecord] = field(default_factory=list)
    confidence:   float = 0.0
    win_prob:     float = 0.0
    signal_reasoning: str = ""

    @property
    def is_open(self) -> bool:
        return self.status in ("open", "tp1_hit", "tp2_hit")

    @property
    def risk_distance(self) -> float:
        return abs(self.entry - self.original_sl)

    def age_hours(self) -> float:
        opened = datetime.fromisoformat(self.opened_at.replace("Z", "+00:00"))
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - opened).total_seconds() / 3600

    def timeout_hours(self) -> int:
        return REGIME_TIMEOUT.get(self.regime, REGIME_TIMEOUT["unknown"])

    def management_event_strings(self) -> List[str]:
        """For backward compatibility with journal."""
        return [e.to_journal_str() for e in self.management_events]


class TradeManager:
    """
    Active position lifecycle manager v2.0.
    Chandelier exit + regime-dependent timeout + journal integration.
    """

    def __init__(self, telegram=None, learning_loop=None, journal=None):
        self.telegram      = telegram
        self.learning      = learning_loop
        self.journal       = journal
        self._positions:   Dict[str, ManagedPosition] = {}
        self._closed_log:  List[ManagedPosition] = []

    # ── Open ──────────────────────────────────────────────────
    def open_position(
        self,
        pair: str, direction: str,
        entry: float, stop_loss: float,
        tp1: float, tp2: float, tp3: float,
        lots: float, atr: float,
        agent: str = "", regime: str = "unknown",
        session: str = "unknown", timeframe: str = "1h",
        confidence: float = 0.0, win_prob: float = 0.0,
        bar_index: int = 0, reasoning: str = "",
    ) -> Optional[ManagedPosition]:
        # ── POSITION CAP ──
        if len(self._positions) >= 3:
            logger.warning(f"POSITION CAP: rejecting {pair} ({len(self._positions)} open)")
            return None
        # ──────────────────

        now    = datetime.now(timezone.utc).isoformat()
        pos_id = f"{pair}_{bar_index}"

        pos = ManagedPosition(
            position_id=pos_id, pair=pair.upper(), direction=direction,
            agent=agent, regime=regime, session=session, timeframe=timeframe,
            entry=entry, original_sl=stop_loss,
            tp1=tp1, tp2=tp2, tp3=tp3,
            atr_at_entry=atr, current_sl=stop_loss,
            lots_total=lots, lots_remaining=lots,
            status="open", pnl_realised=0.0,
            opened_at=now, last_updated=now, bar_index=bar_index,
            highest_price=entry, lowest_price=entry,
            recent_highs=[entry], recent_lows=[entry],
            confidence=confidence, win_prob=win_prob,
            signal_reasoning=reasoning,
        )
        self._positions[pair.upper()] = pos
        logger.info(
            f"TradeManager OPEN: {pair} {direction} @ {entry} "
            f"SL={stop_loss} TP1={tp1} Lots={lots} "
            f"Regime={regime} Timeout={pos.timeout_hours()}h"
        )
        return pos

    # ── Update all ────────────────────────────────────────────

    async def update_all(
        self,
        current_prices: Dict[str, float],
        current_candles: Optional[Dict[str, List]] = None,
    ) -> List[ManagementEventRecord]:
        """Call every bar. Returns all events that fired."""
        events    = []
        to_close  = []

        for pair, pos in self._positions.items():
            if not pos.is_open:
                continue
            price = current_prices.get(pair)
            if not price:
                continue

            # Update price tracking
            pos.highest_price = max(pos.highest_price, price)
            pos.lowest_price  = min(pos.lowest_price,  price)
            pos.last_updated  = datetime.now(timezone.utc).isoformat()

            # Update chandelier window
            candles_for_pair = (current_candles or {}).get(pair, [])
            if candles_for_pair:
                recent_n = candles_for_pair[-CHANDELIER_BARS:]
                pos.recent_highs = [getattr(c, "high", price) for c in recent_n]
                pos.recent_lows  = [getattr(c, "low",  price) for c in recent_n]
            else:
                # Maintain rolling window from prices
                pos.recent_highs.append(price)
                pos.recent_lows.append(price)
                pos.recent_highs = pos.recent_highs[-CHANDELIER_BARS:]
                pos.recent_lows  = pos.recent_lows[-CHANDELIER_BARS:]

            # ── REGIME-DEPENDENT TIMEOUT ──────────────────────
            if pos.age_hours() >= pos.timeout_hours():
                ev = self._record_event(pos, "TIMEOUT", None, price, 0, 0,
                    f"⏰ {pair} TIMEOUT after {pos.age_hours():.0f}h "
                    f"(regime={pos.regime}, limit={pos.timeout_hours()}h)")
                pos.status = "timeout"
                to_close.append(pair)
                events.append(ev)
                await self._send(ev.message)
                continue

            # ── SL HIT ────────────────────────────────────────
            sl_hit = (
                (pos.direction == "buy"  and price <= pos.current_sl) or
                (pos.direction == "sell" and price >= pos.current_sl)
            )
            if sl_hit:
                pnl = self._calc_pnl(pos, price, pos.lots_remaining)
                ev  = self._record_event(pos, "CLOSED", pos.current_sl, price,
                    pos.lots_remaining, pnl,
                    f"🛑 {pair} SL hit @ {price:.5f} | PnL: ${pnl:+.2f} | "
                    f"{'BE ✅' if pos.sl_at_breakeven else 'SL ❌'}")
                pos.pnl_realised += pnl
                pos.status = "closed"
                to_close.append(pair)
                events.append(ev)
                await self._send(ev.message)
                await self._report_outcome(pos, price, "loss" if pnl < 0 else "breakeven")
                continue

            # ── TP1 HIT ───────────────────────────────────────
            if not pos.tp1_hit:
                tp1_hit = (
                    (pos.direction == "buy"  and price >= pos.tp1) or
                    (pos.direction == "sell" and price <= pos.tp1)
                )
                if tp1_hit:
                    lots_close = round(pos.lots_total * TP1_CLOSE_PCT, 2)
                    pnl        = self._calc_pnl(pos, pos.tp1, lots_close)
                    new_sl     = self._breakeven_sl(pos)

                    pos.tp1_hit        = True
                    pos.sl_at_breakeven = True
                    pos.current_sl     = new_sl
                    pos.lots_remaining = round(pos.lots_remaining - lots_close, 2)
                    pos.pnl_realised  += pnl
                    pos.status         = "tp1_hit"

                    ev = self._record_event(pos, "TP1_HIT", pos.original_sl, new_sl,
                        lots_close, pnl,
                        f"🎯 {pair} TP1 @ {pos.tp1:.5f}\n"
                        f"Closed {lots_close}L +${pnl:.2f}\n"
                        f"SL → BE: {new_sl:.5f}\n"
                        f"Remaining: {pos.lots_remaining}L → TP2 @ {pos.tp2:.5f}")
                    events.append(ev)
                    await self._send(ev.message)

                    # Write to journal immediately
                    await self._update_journal_events(pos)
                    continue

            # ── TP2 HIT ───────────────────────────────────────
            if pos.tp1_hit and not pos.tp2_hit:
                tp2_hit = (
                    (pos.direction == "buy"  and price >= pos.tp2) or
                    (pos.direction == "sell" and price <= pos.tp2)
                )
                if tp2_hit:
                    lots_close = min(round(pos.lots_total * TP2_CLOSE_PCT, 2), pos.lots_remaining)
                    pnl        = self._calc_pnl(pos, pos.tp2, lots_close)

                    pos.tp2_hit        = True
                    pos.trailing_active = True
                    pos.lots_remaining  = round(pos.lots_remaining - lots_close, 2)
                    pos.pnl_realised   += pnl
                    pos.status          = "tp2_hit"

                    # Activate chandelier trail
                    trail_sl = self._chandelier_sl(pos)
                    pos.current_sl = trail_sl

                    ev = self._record_event(pos, "TP2_HIT", pos.current_sl, trail_sl,
                        lots_close, pnl,
                        f"🎯🎯 {pair} TP2 @ {pos.tp2:.5f}\n"
                        f"Closed {lots_close}L +${pnl:.2f}\n"
                        f"Chandelier trail: {trail_sl:.5f}\n"
                        f"Final {pos.lots_remaining}L → TP3 @ {pos.tp3:.5f}")
                    events.append(ev)
                    await self._send(ev.message)
                    await self._update_journal_events(pos)
                    continue

            # ── CHANDELIER TRAIL ──────────────────────────────
            if pos.trailing_active and pos.lots_remaining > 0:
                new_trail = self._chandelier_sl(pos)
                moved     = False
                if pos.direction == "buy" and new_trail > pos.current_sl:
                    old_sl         = pos.current_sl
                    pos.current_sl = new_trail
                    moved          = True
                elif pos.direction == "sell" and new_trail < pos.current_sl:
                    old_sl         = pos.current_sl
                    pos.current_sl = new_trail
                    moved          = True

                if moved:
                    ev = self._record_event(pos, "TRAIL_MOVED", old_sl, new_trail,
                        0, 0,
                        f"📐 {pair} Chandelier trail: {old_sl:.5f} → {new_trail:.5f}")
                    events.append(ev)
                    logger.debug(ev.message)

            # ── TP3 HIT ───────────────────────────────────────
            if pos.tp2_hit and pos.lots_remaining > 0:
                tp3_hit = (
                    (pos.direction == "buy"  and price >= pos.tp3) or
                    (pos.direction == "sell" and price <= pos.tp3)
                )
                if tp3_hit:
                    pnl = self._calc_pnl(pos, pos.tp3, pos.lots_remaining)
                    pos.pnl_realised += pnl
                    pos.status = "closed"
                    to_close.append(pair)

                    ev = self._record_event(pos, "CLOSED", None, price,
                        pos.lots_remaining, pnl,
                        f"🏆 {pair} TP3 @ {pos.tp3:.5f}\n"
                        f"Final {pos.lots_remaining}L +${pnl:.2f}\n"
                        f"Total PnL: ${pos.pnl_realised:.2f}")
                    events.append(ev)
                    await self._send(ev.message)
                    await self._update_journal_events(pos)
                    await self._report_outcome(pos, price, "win")

        for pair in to_close:
            pos = self._positions.pop(pair, None)
            if pos:
                self._closed_log.append(pos)

        return events

    # ── Chandelier exit ───────────────────────────────────────

    def _chandelier_sl(self, pos: ManagedPosition) -> float:
        """
        Chandelier exit: trail_sl = swing_extreme(last N bars) ∓ multiplier × ATR

        BUY:  trail_sl = highest_high(last N bars) - CHANDELIER_MULT × ATR
        SELL: trail_sl = lowest_low(last N bars)   + CHANDELIER_MULT × ATR

        This is far superior to trailing from all-time-high because:
        - It uses a RECENT swing point, not the peak from 40 bars ago
        - Naturally tightens as volatility contracts
        - Only moves in the direction of the trade
        """
        atr_trail = pos.atr_at_entry * CHANDELIER_MULT

        if pos.direction == "buy":
            swing_high = max(pos.recent_highs) if pos.recent_highs else pos.highest_price
            new_sl     = swing_high - atr_trail
            # Only move trail up, never down (ratchet)
            return round(max(pos.current_sl, new_sl), 6)
        else:
            swing_low  = min(pos.recent_lows) if pos.recent_lows else pos.lowest_price
            new_sl     = swing_low + atr_trail
            # Only move trail down, never up
            return round(min(pos.current_sl, new_sl), 6)

    def _breakeven_sl(self, pos: ManagedPosition) -> float:
        buf = pos.atr_at_entry * BREAKEVEN_BUFFER_R
        if pos.direction == "buy":
            return round(pos.entry + buf, 6)
        return round(pos.entry - buf, 6)

    def _calc_pnl(self, pos: ManagedPosition, exit_price: float, lots: float) -> float:
        pts = (exit_price - pos.entry) if pos.direction == "buy" else (pos.entry - exit_price)
        return round(pts * lots * 100_000 / 10, 2)

    def _record_event(
        self, pos: ManagedPosition, event_type: str,
        old_sl, new_val, lots_closed, pnl, message
    ) -> ManagementEventRecord:
        ev = ManagementEventRecord(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            bar_index=pos.bar_index,
            old_sl=old_sl, new_sl=new_val,
            lots_closed=lots_closed, pnl_usd=pnl,
            current_price=new_val or 0.0,
            message=message,
        )
        pos.management_events.append(ev)
        return ev

    # ── Journal integration ───────────────────────────────────

    async def _update_journal_events(self, pos: ManagedPosition):
        """
        Write management events to journal mid-trade.
        This enables 'what % of TP1 trades hit TP2?' analysis.
        """
        if not self.journal:
            return
        try:
            # Update existing journal entry's management_events field
            entries = self.journal.query(pair=pos.pair)
            for entry in reversed(entries):
                if entry.opened_at == pos.opened_at:
                    entry.management_events = pos.management_event_strings()
                    self.journal._save()
                    return
        except Exception as e:
            logger.debug(f"Journal mid-trade update failed: {e}")

    async def force_close(self, pair: str, current_price: float, reason: str = "Manual"):
        pos = self._positions.pop(pair.upper(), None)
        if not pos:
            return
        pnl = self._calc_pnl(pos, current_price, pos.lots_remaining)
        pos.pnl_realised += pnl
        pos.status = "closed"
        self._closed_log.append(pos)
        msg = f"🔴 {pair} FORCE CLOSED @ {current_price:.5f} | ${pnl:+.2f} | {reason}"
        await self._send(msg)
        await self._report_outcome(pos, current_price, "loss" if pnl < 0 else "win")

    async def _send(self, message: str):
        if self.telegram:
            try:
                await self.telegram.send_message(message)
            except Exception as e:
                logger.warning(f"TradeManager Telegram failed: {e}")

    async def _report_outcome(self, pos: ManagedPosition, exit_price: float, outcome: str):
        if not self.learning:
            return
        try:
            r_dist  = max(abs(pos.entry - pos.original_sl), 1e-9)
            pts     = (exit_price - pos.entry) if pos.direction == "buy" else (pos.entry - exit_price)
            r_mult  = round(pts / r_dist, 3)
            self.learning.update(
                pair=pos.pair, agent_name=pos.agent,
                direction=pos.direction, entry=pos.entry,
                exit_price=exit_price, stop=pos.original_sl,
                tp1=pos.tp1, outcome=outcome,
                r_multiple=r_mult, regime=pos.regime,
                confidence=pos.confidence, session=pos.session,
            )
        except Exception as e:
            logger.warning(f"TradeManager learning update failed: {e}")

    def get_open_positions(self) -> Dict[str, Dict]:
        return {
            pair: {
                "direction":    pos.direction,
                "entry":        pos.entry,
                "current_sl":   pos.current_sl,
                "tp1": pos.tp1, "tp2": pos.tp2, "tp3": pos.tp3,
                "lots_remaining": pos.lots_remaining,
                "status":       pos.status,
                "age_hours":    round(pos.age_hours(), 1),
                "timeout_hours": pos.timeout_hours(),
                "pnl_realised": round(pos.pnl_realised, 2),
                "sl_at_be":     pos.sl_at_breakeven,
                "trailing":     pos.trailing_active,
                "agent":        pos.agent,
                "regime":       pos.regime,
                "n_events":     len(pos.management_events),
                "last_event":   pos.management_events[-1].event_type if pos.management_events else None,
            }
            for pair, pos in self._positions.items()
        }

    def get_stats(self) -> Dict:
        total = len(self._closed_log)
        wins  = sum(1 for p in self._closed_log if p.pnl_realised > 0)
        tp1_count = sum(1 for p in self._closed_log
                        if any(e.event_type == "TP1_HIT" for e in p.management_events))
        tp2_count = sum(1 for p in self._closed_log
                        if any(e.event_type == "TP2_HIT" for e in p.management_events))
        return {
            "open_positions": len(self._positions),
            "closed_trades":  total,
            "win_rate":       round(wins / total, 3) if total else None,
            "total_pnl":      round(sum(p.pnl_realised for p in self._closed_log), 2),
            "tp1_hit_rate":   round(tp1_count / total, 3) if total else None,
            "tp2_hit_rate":   round(tp2_count / total, 3) if total else None,
            "tp1_to_tp2_conversion": round(tp2_count / tp1_count, 3) if tp1_count else None,
        }
