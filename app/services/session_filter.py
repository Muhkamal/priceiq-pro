"""
PriceIQ Pro - Session Filter v3.2
Enhanced with strict mode, overlap detection, and quality modifiers.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from zoneinfo import ZoneInfo
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SessionResult(BaseModel):
    """Result of session filter check."""
    is_allowed: bool
    current_session: str
    session_quality: float
    reason: str
    next_open: Optional[datetime] = None
    is_overlap: bool = False
    active_sessions: List[str] = []

    def dict(self, **kwargs):
        """Convert to dict with ISO format for datetime."""
        data = super().dict(**kwargs)
        if data.get("next_open"):
            data["next_open"] = data["next_open"].isoformat()
        return data


class SessionFilter:
    """
    Filter trades based on market session liquidity with quality scoring.

    Sessions:
        Sydney:   22:00 - 07:00 UTC (Quality: 0.6)
        Tokyo:    00:00 - 09:00 UTC (Quality: 0.7)
        London:   07:00 - 16:00 UTC (Quality: 1.0)
        New York: 12:00 - 21:00 UTC (Quality: 1.0)

    Overlap Boosts:
        London + New York: +0.15 (12:00-16:00 UTC) - Best liquidity
        Tokyo + London:    +0.10 (07:00-09:00 UTC) - Good liquidity
        Sydney + Tokyo:    +0.05 (00:00-07:00 UTC) - Moderate liquidity
    """

    # Session times in UTC
    SESSIONS = {
        "sydney":   {"start": 22, "end": 7, "name": "Sydney", "quality": 0.6},
        "tokyo":    {"start": 0, "end": 9, "name": "Tokyo", "quality": 0.7},
        "london":   {"start": 7, "end": 16, "name": "London", "quality": 1.0},
        "newyork":  {"start": 12, "end": 21, "name": "New York", "quality": 1.0}
    }

    # Best sessions for each pair
    PAIR_SESSIONS = {
        "EURUSD":   ["london", "newyork"],
        "GBPUSD":   ["london", "newyork"],
        "USDJPY":   ["tokyo", "london"],
        "AUDUSD":   ["sydney", "tokyo", "london"],
        "NZDUSD":   ["sydney", "tokyo"],
        "USDCAD":   ["newyork", "london"],
        "USDCHF":   ["london", "newyork"],
        "EURGBP":   ["london"],
        "EURJPY":   ["tokyo", "london"],
        "GBPJPY":   ["tokyo", "london"],
        "EURCHF":   ["london"],
        "AUDJPY":   ["tokyo", "sydney"],
        "CADJPY":   ["tokyo", "newyork"],
        "XAUUSD":   ["london", "newyork"],
        "XAGUSD":   ["london", "newyork"],
    }

    # Quality boost for session overlaps
    OVERLAP_BOOST = {
        ("london", "newyork"): 0.15,  # 12:00-16:00 UTC
        ("tokyo", "london"):   0.10,  # 07:00-09:00 UTC
        ("sydney", "tokyo"):   0.05,  # 00:00-07:00 UTC
    }

    def __init__(self):
        logger.info("SessionFilter v3.2 initialized")

    def check(
        self, 
        pair: str, 
        dt: Optional[datetime] = None, 
        strict: bool = False, 
        **kwargs
    ) -> SessionResult:
        """
        Check if current session is good for trading a pair.

        Args:
            pair: Currency pair (e.g., "EURUSD")
            dt: Datetime to check (default: now UTC)
            strict: If True, only allow highest-quality sessions (quality >= 0.9)
            **kwargs: Ignored extra parameters for compatibility

        Returns:
            SessionResult with is_allowed, quality, reason, etc.
        """
        if dt is None:
            dt = datetime.now(ZoneInfo("UTC"))

        active_sessions = self._get_active_sessions(dt)
        current_session = self._get_primary_session(active_sessions)
        is_overlap = len(active_sessions) > 1

        is_allowed, reason, quality = self._is_pair_allowed(
            pair, current_session, active_sessions, strict
        )
        next_open = self._get_next_session_start(pair, dt)

        return SessionResult(
            is_allowed=is_allowed,
            current_session=current_session,
            session_quality=quality,
            reason=reason,
            next_open=next_open,
            is_overlap=is_overlap,
            active_sessions=active_sessions,
        )

    async def check_session(self, pair: str, dt: Optional[datetime] = None) -> Dict[str, Any]:
        """Async version for backward compatibility."""
        if dt is None:
            dt = datetime.now(ZoneInfo("UTC"))
        result = self.check(pair, dt)
        return {
            "is_allowed": result.is_allowed,
            "current_session": result.current_session,
            "next_open": result.next_open.isoformat() if result.next_open else None,
            "session_quality": result.session_quality,
            "reason": result.reason,
            "is_overlap": result.is_overlap,
            "active_sessions": result.active_sessions,
            "checked_at": datetime.now().isoformat()
        }

    def _get_active_sessions(self, dt: datetime) -> List[str]:
        """Get all currently active sessions."""
        current_hour = dt.hour
        active = []

        if current_hour >= 22 or current_hour < 7:
            active.append("sydney")
        if 0 <= current_hour < 9:
            active.append("tokyo")
        if 7 <= current_hour < 16:
            active.append("london")
        if 12 <= current_hour < 21:
            active.append("newyork")

        return active

    def _get_primary_session(self, active_sessions: List[str]) -> str:
        """Get the primary (highest priority) active session."""
        if not active_sessions:
            return "off_hours"

        priority = {"london": 4, "newyork": 3, "tokyo": 2, "sydney": 1}
        active_sessions.sort(key=lambda x: priority.get(x, 0), reverse=True)
        return active_sessions[0]

    def _is_pair_allowed(
        self, 
        pair: str, 
        primary_session: str, 
        active_sessions: List[str], 
        strict: bool = False
    ) -> tuple:
        """
        Check if pair is allowed to trade in current session.
        
        Returns:
            (is_allowed, reason, quality)
        """
        if primary_session == "off_hours":
            return False, f"{pair} should not be traded during off-hours. Low liquidity.", 0.1

        optimal_sessions = self.PAIR_SESSIONS.get(pair, ["london", "newyork"])

        # Check if any active session is optimal for this pair
        matching_sessions = [s for s in active_sessions if s in optimal_sessions]

        if not matching_sessions:
            quality = 0.3
            session_name = self.SESSIONS.get(primary_session, {}).get("name", primary_session.upper())
            optimal_names = [self.SESSIONS.get(s, {}).get("name", s.upper()) for s in optimal_sessions]
            return False, (
                f"{pair} trades best during {', '.join(optimal_names)} sessions. "
                f"Current {session_name} session has lower liquidity."
            ), quality

        # Calculate quality with overlap boost
        base_quality = max(
            self.SESSIONS.get(s, {}).get("quality", 0.7) 
            for s in matching_sessions
        )

        # Apply overlap boost
        boost = 0.0
        for i, s1 in enumerate(active_sessions):
            for s2 in active_sessions[i+1:]:
                key = tuple(sorted([s1, s2]))
                boost = max(boost, self.OVERLAP_BOOST.get(key, 0.0))

        quality = min(1.0, base_quality + boost)

        # Strict mode: only allow high-quality sessions (overlap periods)
        if strict and quality < 0.9:
            session_names = [self.SESSIONS.get(s, {}).get("name", s.upper()) for s in matching_sessions]
            return False, (
                f"{pair}: Strict mode requires overlap sessions (quality >= 0.9). "
                f"Current {', '.join(session_names)} quality is {quality:.2f}."
            ), quality

        session_names = [self.SESSIONS.get(s, {}).get("name", s.upper()) for s in matching_sessions]
        overlap_text = " (overlap)" if len(active_sessions) > 1 else ""
        return True, (
            f"{pair} trades well during {', '.join(session_names)} session{overlap_text}"
        ), quality

    def _get_next_session_start(self, pair: str, dt: datetime) -> Optional[datetime]:
        """Get the start time of the next optimal session."""
        optimal_sessions = self.PAIR_SESSIONS.get(pair, ["london", "newyork"])
        session_order = ["sydney", "tokyo", "london", "newyork"]

        for session in session_order:
            if session in optimal_sessions:
                session_info = self.SESSIONS.get(session, {})
                session_start_hour = session_info.get("start", 7)

                next_start = dt.replace(hour=session_start_hour, minute=0, second=0, microsecond=0)
                if next_start <= dt:
                    next_start += timedelta(days=1)
                return next_start

        return None

    def get_current_sessions(self, dt: Optional[datetime] = None) -> List[str]:
        """Get all currently active sessions."""
        if dt is None:
            dt = datetime.now(ZoneInfo("UTC"))
        return self._get_active_sessions(dt)

    def get_session_quality_for_pair(self, pair: str, dt: Optional[datetime] = None) -> float:
        """Get the current session quality for a pair (0.0-1.0)."""
        result = self.check(pair, dt)
        return result.session_quality

    def get_session_summary(self, dt: Optional[datetime] = None) -> Dict[str, Any]:
        """Get summary of all active sessions with quality scores."""
        if dt is None:
            dt = datetime.now(ZoneInfo("UTC"))
        
        active = self.get_current_sessions(dt)
        summary = {
            "timestamp": dt.isoformat(),
            "active_sessions": active,
            "is_overlap": len(active) > 1,
            "sessions": {}
        }
        
        for session in active:
            summary["sessions"][session] = {
                "name": self.SESSIONS.get(session, {}).get("name", session.upper()),
                "hours": f"{self.SESSIONS.get(session, {}).get('start', '?')}:00-{self.SESSIONS.get(session, {}).get('end', '?')}:00 UTC",
                "quality": self.SESSIONS.get(session, {}).get("quality", 0.5),
            }
        
        return summary


# Singleton instance
session_filter = SessionFilter()
