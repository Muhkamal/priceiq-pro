"""News blackout gate for forex pairs. Synthetics are unaffected.
Fetches from faireconomy RSS, caches to JSON for 4 hours."""
import json
import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional
from urllib.request import urlopen, Request

logger = logging.getLogger(__name__)

CACHE_FILE = Path(__file__).parent.parent.parent / "data" / "news_cache.json"
CACHE_MAX_AGE_HOURS = 4
RSS_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"

# USD events affect all forex pairs; synthetics (V75, STEP, etc.) are unaffected
USD_PAIRS = {"EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", 
             "AUDUSD", "NZDUSD", "XAUUSD", "BTCUSD", "ETHUSD"}

# Synthetics have no news correlation
SYNTHETICS = {"V75", "V100", "STEP", "BOOM", "CRASH"}

# Blackout windows (minutes)
BLACKOUT_PRE = {"HIGH": 20, "MEDIUM": 10}
BLACKOUT_POST = {"HIGH": 15, "MEDIUM": 10}

@dataclass
class NewsEvent:
    title: str
    currency: str
    impact: str  # "HIGH" | "MEDIUM" | "LOW"
    dt_utc: datetime

class NewsGate:
    """Pre-gate: blocks forex signals during high-impact news."""
    
    def __init__(self):
        self._events: List[NewsEvent] = []
        self._last_refresh: Optional[datetime] = None
        self._load_cache()
    
    def should_refresh(self) -> bool:
        if not self._last_refresh:
            return True
        age = (datetime.now(timezone.utc) - self._last_refresh).total_seconds() / 3600
        return age > CACHE_MAX_AGE_HOURS
    
    def refresh(self):
        """Fetch latest events from RSS. Call once per 4 hours."""
        if not self.should_refresh():
            return
        
        try:
            req = Request(RSS_URL, headers={"User-Agent": "SMC-Bot/1.0"})
            with urlopen(req, timeout=15) as resp:
                raw = resp.read()
            
            root = ET.fromstring(raw)
            events = []
            
            for item in root.iter("event"):
                try:
                    title = self._text(item, "title", "")
                    currency = self._text(item, "country", "").upper()
                    impact = self._text(item, "impact", "LOW").upper()
                    date_str = self._text(item, "date", "")
                    time_str = self._text(item, "time", "")
                    
                    if not date_str or not currency:
                        continue
                    
                    dt_utc = self._parse_dt(date_str, time_str)
                    if dt_utc is None:
                        continue
                    
                    # Normalize impact
                    if impact not in ("HIGH", "MEDIUM", "LOW"):
                        impact_lower = impact.lower()
                        if "high" in impact_lower or "red" in impact_lower:
                            impact = "HIGH"
                        elif "medium" in impact_lower or "orange" in impact_lower:
                            impact = "MEDIUM"
                        else:
                            impact = "LOW"
                    
                    if impact != "LOW":  # Only track HIGH and MEDIUM
                        events.append(NewsEvent(title, currency, impact, dt_utc))
                except Exception as e:
                    logger.debug(f"RSS parse error: {e}")
                    continue
            
            self._events = events
            self._last_refresh = datetime.now(timezone.utc)
            self._save_cache()
            
            high_count = sum(1 for e in events if e.impact == "HIGH")
            logger.info(f"News calendar: {len(events)} events ({high_count} HIGH)")
            
        except Exception as e:
            logger.warning(f"News RSS fetch failed: {e}")
    
    def check(self, pair: str, now: Optional[datetime] = None) -> tuple:
        """Returns (blocked: bool, reason: str)."""
        now = now or datetime.now(timezone.utc)
        
        # Synthetics are unaffected by news
        if pair in SYNTHETICS or pair.startswith(("VOL", "BOOM", "CRASH", "STEP", "JUMP")):
            return False, "Synthetic (no news correlation)"
        
        # Check each event
        for event in self._events:
            # Only USD events affect forex pairs
            if event.currency != "USD":
                continue
            if pair not in USD_PAIRS:
                continue
            
            # Get blackout window
            pre_min = BLACKOUT_PRE.get(event.impact, 0)
            post_min = BLACKOUT_POST.get(event.impact, 0)
            
            window_start = event.dt_utc - timedelta(minutes=pre_min)
            window_end = event.dt_utc + timedelta(minutes=post_min)
            
            if window_start <= now <= window_end:
                return True, f"News: {event.title} ({event.impact}) at {event.dt_utc.strftime('%H:%M UTC')}"
        
        return False, "No news blackout"
    
    def _text(self, element, tag: str, default: str) -> str:
        child = element.find(tag)
        if child is not None and child.text:
            return child.text.strip()
        return default
    
    def _parse_dt(self, date_str: str, time_str: str) -> Optional[datetime]:
        """Parse ForexFactory date/time → UTC datetime."""
        combined = f"{date_str} {time_str}".strip()
        
        formats = [
            "%m-%d-%Y %I:%M%p",
            "%m-%d-%Y %H:%M",
            "%m/%d/%Y %I:%M%p",
            "%m/%d/%Y %H:%M",
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(combined, fmt)
                # ForexFactory uses Eastern Time (EST = UTC-5)
                dt = dt.replace(tzinfo=timezone(timedelta(hours=-5)))
                return dt.astimezone(timezone.utc)
            except ValueError:
                continue
        
        return None
    
    def _save_cache(self):
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
                "events": [
                    {"title": e.title, "currency": e.currency, 
                     "impact": e.impact, "dt_utc": e.dt_utc.isoformat()}
                    for e in self._events
                ]
            }
            with open(CACHE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"Cache save error: {e}")
    
    def _load_cache(self):
        try:
            if not CACHE_FILE.exists():
                return
            with open(CACHE_FILE) as f:
                data = json.load(f)
            
            if data.get("last_refresh"):
                self._last_refresh = datetime.fromisoformat(data["last_refresh"])
                age_h = (datetime.now(timezone.utc) - self._last_refresh).total_seconds() / 3600
                if age_h > CACHE_MAX_AGE_HOURS:
                    return  # Cache too old, will refresh
            
            self._events = [
                NewsEvent(
                    e["title"], e["currency"], e["impact"],
                    datetime.fromisoformat(e["dt_utc"])
                )
                for e in data.get("events", [])
            ]
            logger.info(f"News cache loaded: {len(self._events)} events")
        except Exception as e:
            logger.debug(f"Cache load error: {e}")

# Singleton instance
_news_gate = NewsGate()

def check_news_blackout(pair: str) -> tuple:
    """Convenience function for main.py integration."""
    if _news_gate.should_refresh():
        _news_gate.refresh()
    return _news_gate.check(pair)
