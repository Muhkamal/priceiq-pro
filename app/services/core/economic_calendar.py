"""
PriceIQ Pro — Economic Calendar & News Blackout Filter v1.0

Fetches upcoming high-impact economic events and enforces pre/post-event
signal blackouts. Prevents firing signals into NFP, FOMC, CPI, etc.

Sources (in priority order):
    1. ForexFactory RSS feed (free, no API key)
    2. Hardcoded known recurring events as fallback
    3. Manual override list (from config)

Blackout logic:
    HIGH impact:   block 20 min before → 15 min after
    MEDIUM impact: block 10 min before → 10 min after
    LOW impact:    no blackout (log only)

Currency mapping:
    Event currency → affected pairs
    USD event → blocks EURUSD, GBPUSD, USDJPY, XAUUSD, USDCHF, AUDUSD
    GBP event → blocks GBPUSD, GBPJPY, GBPCAD, EURGBP
    EUR event → blocks EURUSD, EURJPY, EURGBP, EURCAD

Usage:
    calendar = EconomicCalendar()
    await calendar.refresh()   # call once per hour from scheduler

    # Before any signal:
    check = calendar.check_blackout(pair="XAUUSD", dt=datetime.now(utc))
    if check.blocked:
        log(check.reason)
        return None
"""

from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set
from urllib.request import urlopen, Request

logger = logging.getLogger(__name__)

# ── Blackout windows (minutes) ──────────────────────────────
BLACKOUT_PRE  = {"HIGH": 20, "MEDIUM": 10, "LOW": 0}
BLACKOUT_POST = {"HIGH": 15, "MEDIUM": 10, "LOW": 0}

# ── Currency → affected pairs mapping ───────────────────────
CURRENCY_PAIRS: Dict[str, List[str]] = {
    "USD": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD",
            "NZDUSD", "XAUUSD", "XAGUSD"],
    "EUR": ["EURUSD", "EURJPY", "EURGBP", "EURCAD", "EURAUD", "EURCHF"],
    "GBP": ["GBPUSD", "GBPJPY", "GBPCAD", "GBPAUD", "EURGBP", "GBPCHF"],
    "JPY": ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "NZDJPY"],
    "AUD": ["AUDUSD", "AUDCAD", "AUDNZD", "AUDJPY", "AUDCHF"],
    "CAD": ["USDCAD", "GBPCAD", "EURCAD", "CADJPY", "AUDCAD"],
    "CHF": ["USDCHF", "EURCHF", "GBPCHF", "CHFJPY"],
    "NZD": ["NZDUSD", "NZDJPY", "AUDNZD", "NZDCAD"],
    "XAU": ["XAUUSD"],
}

# ── High-impact recurring events (fallback when feed unavailable) ─
KNOWN_HIGH_IMPACT = {
    "NFP":           "USD",   # Non-Farm Payrolls — first Friday of month
    "FOMC":          "USD",   # Federal Reserve rate decision
    "CPI":           "USD",   # Consumer Price Index
    "GDP":           "USD",
    "BOE":           "GBP",   # Bank of England rate decision
    "ECB":           "EUR",   # European Central Bank
    "BOJ":           "JPY",   # Bank of Japan
    "RBA":           "AUD",   # Reserve Bank of Australia
    "RBNZ":          "NZD",
    "SNB":           "CHF",
    "UK_CPI":        "GBP",
    "EUR_CPI":       "EUR",
    "US_RETAIL":     "USD",
    "US_JOBLESS":    "USD",
    "UNEMPLOYMENT":  "USD",
}


@dataclass
class EconomicEvent:
    title:     str
    currency:  str
    impact:    str           # "HIGH" | "MEDIUM" | "LOW"
    dt_utc:    datetime
    actual:    Optional[str] = None
    forecast:  Optional[str] = None


@dataclass
class BlackoutCheck:
    blocked:       bool
    pair:          str
    reason:        str
    event:         Optional[EconomicEvent] = None
    minutes_until: Optional[float] = None   # negative = event already passed
    size_mult:     float = 1.0              # 0.0 blocked, 0.5 reduced


class EconomicCalendar:
    """
    Fetches ForexFactory RSS and maintains upcoming event list.
    Thread-safe for single asyncio event loop.
    """

    FF_RSS_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    REFRESH_INTERVAL_HOURS = 4

    def __init__(self):
        self._events: List[EconomicEvent] = []
        self._last_refresh: Optional[datetime] = None
        self._manual_blackouts: List[Dict] = []   # manually added events

    # ── Public API ────────────────────────────────────────────

    async def refresh(self):
        """Fetch latest events from ForexFactory RSS. Call hourly from scheduler."""
        try:
            events = await asyncio.to_thread(self._fetch_ff_rss)
            if events:
                self._events = events
                self._last_refresh = datetime.now(timezone.utc)
                high_count = sum(1 for e in events if e.impact == "HIGH")
                logger.info(
                    f"Economic calendar refreshed: {len(events)} events "
                    f"({high_count} HIGH impact)"
                )
            else:
                logger.warning("Economic calendar fetch returned no events — using cache")
        except Exception as e:
            logger.warning(f"Economic calendar refresh failed: {e}")

    def check_blackout(
        self,
        pair: str,
        dt: Optional[datetime] = None,
    ) -> BlackoutCheck:
        """
        Check if a signal for `pair` should be blocked right now.
        Returns BlackoutCheck with blocked flag and reason.
        """
        now  = dt or datetime.now(timezone.utc)
        pair = pair.upper()

        for event in self._events:
            if event.impact == "LOW":
                continue

            # Check if this event affects this pair
            affected = CURRENCY_PAIRS.get(event.currency, [])
            if pair not in affected:
                continue

            pre_min  = BLACKOUT_PRE.get(event.impact, 0)
            post_min = BLACKOUT_POST.get(event.impact, 0)

            window_start = event.dt_utc - timedelta(minutes=pre_min)
            window_end   = event.dt_utc + timedelta(minutes=post_min)

            if window_start <= now <= window_end:
                mins_until = (event.dt_utc - now).total_seconds() / 60
                reason = (
                    f"NEWS BLACKOUT: {event.title} ({event.currency} {event.impact}) "
                    f"at {event.dt_utc.strftime('%H:%M UTC')} — "
                    f"{'in {:.0f}min'.format(mins_until) if mins_until > 0 else 'in progress'}"
                )
                return BlackoutCheck(
                    blocked=True, pair=pair, reason=reason,
                    event=event, minutes_until=round(mins_until, 1),
                    size_mult=0.0,
                )

            # Near-event warning (within 2× the blackout window): reduce size
            warn_start = event.dt_utc - timedelta(minutes=pre_min * 2)
            warn_end   = event.dt_utc + timedelta(minutes=post_min * 1.5)
            if warn_start <= now <= window_start or window_end <= now <= warn_end:
                mins_until = (event.dt_utc - now).total_seconds() / 60
                return BlackoutCheck(
                    blocked=False, pair=pair,
                    reason=f"NEWS PROXIMITY: {event.title} soon — size reduced 50%",
                    event=event, minutes_until=round(mins_until, 1),
                    size_mult=0.5,
                )

        return BlackoutCheck(
            blocked=False, pair=pair,
            reason="No news blackout active", size_mult=1.0,
        )

    def upcoming_events(
        self,
        hours_ahead: int = 24,
        impact_filter: Optional[str] = None,
    ) -> List[EconomicEvent]:
        """Return upcoming events in the next N hours."""
        now    = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)
        events = [
            e for e in self._events
            if now <= e.dt_utc <= cutoff
        ]
        if impact_filter:
            events = [e for e in events if e.impact == impact_filter]
        return sorted(events, key=lambda e: e.dt_utc)

    def add_manual_blackout(
        self,
        currency: str,
        dt_utc: datetime,
        title: str = "Manual Blackout",
        impact: str = "HIGH",
    ):
        """Manually add a blackout event (e.g. central bank press conferences)."""
        self._events.append(EconomicEvent(
            title=title, currency=currency.upper(),
            impact=impact, dt_utc=dt_utc,
        ))
        logger.info(f"Manual blackout added: {title} {currency} {dt_utc}")

    def status(self) -> Dict:
        now = datetime.now(timezone.utc)
        return {
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
            "total_events": len(self._events),
            "high_impact":  sum(1 for e in self._events if e.impact == "HIGH"),
            "next_high":    next(
                ({"title": e.title, "currency": e.currency,
                  "dt": e.dt_utc.isoformat(),
                  "in_min": round((e.dt_utc - now).total_seconds() / 60, 1)}
                 for e in sorted(self._events, key=lambda x: x.dt_utc)
                 if e.impact == "HIGH" and e.dt_utc > now),
                None,
            ),
        }

    # ── Internal fetch ────────────────────────────────────────

    def _fetch_ff_rss(self) -> List[EconomicEvent]:
        """Fetch and parse ForexFactory RSS XML calendar."""
        events = []
        try:
            req  = Request(
                self.FF_RSS_URL,
                headers={"User-Agent": "PriceIQ-Pro/5.0 Economic Calendar"},
            )
            with urlopen(req, timeout=10) as resp:
                raw = resp.read()
            root = ET.fromstring(raw)
        except Exception as e:
            logger.warning(f"FF RSS fetch failed: {e}")
            return []

        for item in root.iter("event"):
            try:
                title    = self._text(item, "title",    "Unknown")
                currency = self._text(item, "country",  "").upper()
                impact   = self._text(item, "impact",   "LOW").upper()
                date_str = self._text(item, "date",     "")
                time_str = self._text(item, "time",     "")
                actual   = self._text(item, "actual",   None)
                forecast = self._text(item, "forecast", None)

                if not date_str or currency not in CURRENCY_PAIRS:
                    continue

                dt_utc = self._parse_dt(date_str, time_str)
                if dt_utc is None:
                    continue

                # Normalise impact
                if impact not in ("HIGH", "MEDIUM", "LOW"):
                    if "high" in impact.lower():
                        impact = "HIGH"
                    elif "medium" in impact.lower() or "moderate" in impact.lower():
                        impact = "MEDIUM"
                    else:
                        impact = "LOW"

                events.append(EconomicEvent(
                    title=title, currency=currency, impact=impact,
                    dt_utc=dt_utc, actual=actual, forecast=forecast,
                ))
            except Exception as e:
                logger.debug(f"FF RSS parse error on item: {e}")
                continue

        return events

    def _text(self, element, tag: str, default) -> Optional[str]:
        child = element.find(tag)
        if child is not None and child.text:
            return child.text.strip()
        return default

    def _parse_dt(self, date_str: str, time_str: str) -> Optional[datetime]:
        """Parse ForexFactory date/time strings → UTC datetime."""
        formats = [
            "%m-%d-%Y %I:%M%p",
            "%m-%d-%Y %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
        ]
        combined = f"{date_str} {time_str}".strip()
        for fmt in formats:
            try:
                dt = datetime.strptime(combined, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        # Try date-only
        try:
            dt = datetime.strptime(date_str, "%m-%d-%Y")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
        return None
