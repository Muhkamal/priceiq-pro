"""
PriceIQ Pro — COT Report Agent v1.1 (Production Ready)

Reads CFTC Commitments of Traders (COT) data to detect when
institutional/smart money is at extreme positioning — one of the
most reliable contrarian signals available to retail traders.

Why this matters:
    When non-commercial traders (hedge funds, large speculators) reach
    historically extreme net long/short positions, it signals the trend
    is running out of fuel. The COT report shows WHO is holding WHAT
    before price reverses.

    Key insight: Non-commercials are TREND FOLLOWERS who get crowded.
    When they're maximally long, there's nobody left to buy.
    When they're maximally short, there's nobody left to sell.
    Extreme positioning → mean reversion setup.

    Secondary insight: Commercials (producers/hedgers) are CONTRARIAN.
    When commercials are massively short gold, they're hedging production
    at high prices — a signal gold is near a top.

Data source:
    CFTC Disaggregated COT Report — published every Friday at 3:30 PM ET
    URL: https://www.cftc.gov/dea/newcot/f_year.htm (annual zip files)
    Also via: https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm

    Instruments tracked (CFTC market codes):
        Gold (COMEX):        088691
        Euro FX (CME):       099741
        British Pound (CME): 096742
        Japanese Yen (CME):  097741
        Swiss Franc (CME):   092741

Signal logic:
    Extreme NET LONG non-commercials (>= +1.5σ from 52-week mean):
        → Contrarian SELL signal (crowd is max long, reversal likely)

    Extreme NET SHORT non-commercials (<= -1.5σ from 52-week mean):
        → Contrarian BUY signal (crowd is max short, reversal likely)

    Commercials opposing non-commercials:
        → Boost confidence (smart money disagrees with crowd)

    COT Index (0-100):
        >= 80: Non-commercials near 52-week extreme long → SELL
        <= 20: Non-commercials near 52-week extreme short → BUY
        40-60: No signal

Weekly cadence: refresh every Friday after 3:30 PM ET (20:30 UTC)
Signal validity: 5 trading days (until next report)

Usage:
    agent = COTReportAgent()
    await agent.refresh()    # fetches latest CFTC data
    signal = agent.evaluate(candles, regime="trending", pair="XAUUSD")
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import zipfile
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen, Request

import numpy as np

logger = logging.getLogger(__name__)

# ── CFTC market codes ─────────────────────────────────────────
CFTC_CODES: Dict[str, str] = {
    "XAUUSD": "088691",   # Gold (COMEX)
    "EURUSD": "099741",   # Euro FX (CME)
    "GBPUSD": "096742",   # British Pound (CME)
    "USDJPY": "097741",   # Japanese Yen (CME) — note: JPY futures, inverted
    "USDCHF": "092741",   # Swiss Franc (CME) — inverted
    "AUDUSD": "232741",   # Australian Dollar (CME)
    "USDCAD": "090741",   # Canadian Dollar (CME) — inverted
}
# Pairs where futures are INVERTED vs spot price
# (USDJPY spot up = JPY futures short = invert COT signal)
INVERTED_PAIRS = {"USDJPY", "USDCHF", "USDCAD"}

# CFTC column names in disaggregated report
COT_COLUMNS = {
    "date":              "Report_Date_as_MM_DD_YYYY",
    "market":            "Market_and_Exchange_Names",
    "cftc_code":         "CFTC_Contract_Market_Code",
    "nc_long":           "NonComm_Positions_Long_All",
    "nc_short":          "NonComm_Positions_Short_All",
    "comm_long":         "Comm_Positions_Long_All",
    "comm_short":        "Comm_Positions_Short_All",
    "oi":                "Open_Interest_All",
    "nc_change_long":    "Change_in_NonComm_Long_All",
    "nc_change_short":   "Change_in_NonComm_Short_All"}

# Signal thresholds
COT_INDEX_EXTREME_LONG   = 80    # COT index >= 80 → contrarian SELL
COT_INDEX_EXTREME_SHORT  = 20    # COT index <= 20 → contrarian BUY
ZSCORE_EXTREME           = 1.5   # z-score threshold for extreme positioning


@dataclass
class COTSnapshot:
    """Single week's COT data for one instrument."""
    pair:            str
    report_date:     str
    nc_net:          float    # non-commercial net (long - short)
    nc_long:         float
    nc_short:        float
    comm_net:        float    # commercial net (long - short)
    open_interest:   float
    nc_change:       float    # week-over-week change in nc_net
    cot_index:       float    # 0-100, where nc_net stands vs 52-week range
    nc_zscore:       float    # z-score of nc_net vs 52-week history
    nc_pct_oi:       float    # nc_net as % of open interest


@dataclass
class COTSignal:
    """Output from COTReportAgent."""
    pair:            str
    direction:       Optional[str]    # "buy" | "sell" | None
    signal_type:     str              # "extreme_long" | "extreme_short" | "neutral"
    cot_index:       float
    nc_zscore:       float
    confidence:      float
    reasoning:       str
    report_date:     str
    commercials_confirm: bool         # True if commercials oppose non-commercials


@dataclass
class COTAgentSignal:
    """AgentSignal-compatible output for orchestrator integration."""
    agent_name:      str = "COTReportAgent"
    direction:       Optional[str] = None
    confidence:      float = 0.0
    win_probability: float = 0.0
    expected_value:  float = 0.0
    stop_distance:   float = 0.0
    tp1_distance:    float = 0.0
    tp2_distance:    float = 0.0
    regime_fit:      float = 0.5
    reasoning:       str = ""
    raw_features:    Dict = field(default_factory=dict)
    cot_signal:      Optional[COTSignal] = None

    @property
    def is_valid(self) -> bool:
        return (
            self.direction is not None
            and self.confidence >= 0.40
            and self.stop_distance > 0
        )


class CFTCDataFetcher:
    """
    Fetches COT data from CFTC official sources.

    Primary:   Annual ZIP files from cftc.gov (most complete)
    Fallback:  Quandl/Nasdaq Data Link (requires free API key)
    Fallback2: Manual CSV injection for backtesting
    """

    CFTC_ZIP_URL = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
    CFTC_CURRENT = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_hist_2006_2016.zip"

    def fetch_year(self, year: int = None) -> List[Dict]:
        """Fetch full year of COT data. Returns list of row dicts."""
        year = year or datetime.now().year
        url  = self.CFTC_ZIP_URL.format(year=year)
        try:
            req = Request(url, headers={"User-Agent": "PriceIQ-Pro/5.5"})
            with urlopen(req, timeout=30) as r:
                raw = r.read()
            return self._parse_zip(raw)
        except Exception as e:
            logger.debug(f"CFTC fetch {year}: {e}")
            # Try previous year as fallback
            if year == datetime.now().year:
                return self.fetch_year(year - 1)
            return []

    def fetch_current(self) -> List[Dict]:
        """Fetch most recent available data (current year first, then prior)."""
        data = self.fetch_year(datetime.now().year)
        if not data:
            data = self.fetch_year(datetime.now().year - 1)
        return data

    def _parse_zip(self, raw_bytes: bytes) -> List[Dict]:
        """Parse CFTC annual ZIP file into list of row dicts."""
        rows = []
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                # Find the CSV file inside the ZIP
                csv_files = [f for f in zf.namelist() if f.endswith(".txt") or f.endswith(".csv")]
                if not csv_files:
                    return []
                with zf.open(csv_files[0]) as f:
                    content = f.read().decode("utf-8", errors="ignore")
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                rows.append(dict(row))
        except Exception as e:
            logger.debug(f"COT ZIP parse error: {e}")
        return rows

    def fetch_from_csv(self, csv_content: str) -> List[Dict]:
        """Parse COT data from raw CSV string (for testing/manual import)."""
        reader = csv.DictReader(io.StringIO(csv_content))
        return [dict(row) for row in reader]


class COTHistoryStore:
    """
    Maintains rolling 52-week history of COT snapshots per pair.
    Used to compute COT Index (0-100) and z-scores.
    """

    def __init__(self, path: str = "cot_history.json"):
        self._path    = path
        self._history: Dict[str, List[COTSnapshot]] = {}   # pair → list (newest last)
        self._load()

    def add(self, pair: str, snapshot: COTSnapshot):
        if pair not in self._history:
            self._history[pair] = []
        # Avoid duplicates
        existing_dates = {s.report_date for s in self._history[pair]}
        if snapshot.report_date not in existing_dates:
            self._history[pair].append(snapshot)
            # Keep 52 weeks = 52 snapshots
            self._history[pair] = sorted(
                self._history[pair], key=lambda s: s.report_date
            )[-52:]
            self._save()

    def get_history(self, pair: str) -> List[COTSnapshot]:
        return self._history.get(pair, [])

    def get_latest(self, pair: str) -> Optional[COTSnapshot]:
        history = self._history.get(pair, [])
        return history[-1] if history else None

    def compute_cot_index(self, pair: str, current_nc_net: float) -> float:
        """
        COT Index: where current nc_net stands vs 52-week range.
        0 = lowest nc_net in 52 weeks (extreme short)
        100 = highest nc_net in 52 weeks (extreme long)
        """
        history = self._history.get(pair, [])
        if len(history) < 4:
            return 50.0   # insufficient data → neutral
        nc_nets    = [s.nc_net for s in history]
        min_nc     = min(nc_nets)
        max_nc     = max(nc_nets)
        rang       = max_nc - min_nc
        if rang == 0:
            return 50.0
        return round(((current_nc_net - min_nc) / rang) * 100, 2)

    def compute_zscore(self, pair: str, current_nc_net: float) -> float:
        history = self._history.get(pair, [])
        if len(history) < 4:
            return 0.0
        nc_nets = [s.nc_net for s in history]
        mu  = np.mean(nc_nets)
        std = np.std(nc_nets)
        if std == 0:
            return 0.0
        return round((current_nc_net - mu) / std, 3)

    def _save(self):
        try:
            data = {
                pair: [
                    {k: v for k, v in s.__dict__.items()}
                    for s in snaps
                ]
                for pair, snaps in self._history.items()
            }
            with open(self._path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.debug(f"COT history save: {e}")

    def _load(self):
        try:
            with open(self._path) as f:
                data = json.load(f)
            self._history = {
                pair: [COTSnapshot(**s) for s in snaps]
                for pair, snaps in data.items()
            }
            logger.info(f"COT history loaded: {sum(len(v) for v in self._history.values())} snapshots")
        except Exception:
            pass


class COTReportAgent:
    """
    7th trading agent. Uses CFTC COT data to detect extreme institutional
    positioning — a contrarian signal when crowds are max long/short.

    This agent has a WEEKLY cadence (COT published Fridays) vs the other
    agents' hourly cadence. Signal valid for the entire week until next report.

    Integration:
        Same .evaluate(candles, regime, pair) interface as other agents.
        Lower confidence than intraday agents — COT is a weekly filter,
        not a precise entry trigger. Works best combined with price confirmation.
    """

    NAME = "COTReportAgent"
    SUPPORTED_PAIRS = set(CFTC_CODES.keys())
    REGIME_FIT = {
        "trending": 0.55,    # COT is contrarian — works better at trend extremes
        "ranging":  0.70,    # Better in ranging (mean reversion)
        "volatile": 0.45,    # Volatile = unpredictable, COT less reliable
    }

    def __init__(self, history_path: str = "cot_history.json"):
        self._fetcher = CFTCDataFetcher()
        self._store   = COTHistoryStore(history_path)
        self._latest:  Dict[str, COTSnapshot] = {}
        self._signals: Dict[str, COTSignal]   = {}
        self._last_refresh: Optional[datetime] = None

    # ── Data refresh ─────────────────────────────────────────

    async def refresh(self) -> Dict[str, COTSnapshot]:
        """
        Fetch and process latest CFTC COT data.
        Call every Friday after 20:30 UTC from scheduler.
        Returns dict of pair → COTSnapshot.
        """
        # ═══ THROTTLE: Only fetch once per 24 hours to prevent CFTC IP bans ═══
        if self._last_refresh and (datetime.now(timezone.utc) - self._last_refresh).total_seconds() < 86400:
            logger.debug("COT refresh throttled (already refreshed today)")
            return dict(self._latest)

        try:
            rows = await asyncio.to_thread(self._fetcher.fetch_current)
            if not rows:
                logger.warning("COT: no data fetched from CFTC")
                return {}

            # Process each tracked pair
            for pair, code in CFTC_CODES.items():
                pair_rows = [
                    r for r in rows
                    if r.get("CFTC_Contract_Market_Code", "").strip() == code
                ]
                if not pair_rows:
                    continue

                # Get most recent row
                pair_rows.sort(key=lambda r: r.get("Report_Date_as_MM_DD_YYYY", ""), reverse=True)
                row  = pair_rows[0]
                snap = self._parse_row(row, pair)
                if snap:
                    self._store.add(pair, snap)
                    self._latest[pair] = snap
                    self._signals[pair] = self._build_signal(pair, snap)

            self._last_refresh = datetime.now(timezone.utc)
            logger.info(
                f"COT refreshed: {len(self._latest)} instruments | "
                + ", ".join(
                    f"{p}={s.direction}(idx={self._latest[p].cot_index:.0f})"
                    for p, s in self._signals.items() if s.direction
                )
            )
            return dict(self._latest)

        except Exception as e:
            logger.error(f"COT refresh failed: {e}", exc_info=True)
            return {}

    def load_from_csv(self, csv_content: str) -> int:
        """Load COT history from CSV string. Returns rows processed."""
        rows = self._fetcher.fetch_from_csv(csv_content)
        n = 0
        for pair, code in CFTC_CODES.items():
            pair_rows = [r for r in rows if r.get("CFTC_Contract_Market_Code", "").strip() == code]
            for row in sorted(pair_rows, key=lambda r: r.get("Report_Date_as_MM_DD_YYYY", "")):
                snap = self._parse_row(row, pair)
                if snap:
                    self._store.add(pair, snap)
                    self._latest[pair] = snap
                    n += 1
        # Build signals from latest snapshots
        for pair, snap in self._latest.items():
            self._signals[pair] = self._build_signal(pair, snap)
        logger.info(f"COT loaded from CSV: {n} snapshots across {len(self._latest)} pairs")
        return n

    # ── Signal evaluation ─────────────────────────────────────

    def evaluate(
        self,
        candles: list,
        regime:  str = "trending",
        pair:    str = None,  # ═══ FIX: Changed to None to match BaseAgent signature ═══
    ) -> COTAgentSignal:
        """
        Evaluate COT signal for given pair.
        Same interface as other agents for orchestrator compatibility.
        """
        # ═══ FIX: Safe fallback if orchestrator passes None ═══
        pair_u = (pair or "XAUUSD").upper()
        
        if pair_u not in self.SUPPORTED_PAIRS:
            return self._null_signal(f"{pair} not in COT tracking list")

        if not self._latest:
            return self._null_signal("No COT data loaded — call refresh() first")

        snap = self._latest.get(pair_u)
        if not snap:
            return self._null_signal(f"No COT data for {pair_u}")

        # Check data freshness (COT is weekly, max 10 days old)
        try:
            report_dt = datetime.strptime(snap.report_date, "%m/%d/%Y").replace(tzinfo=timezone.utc)
            age_days  = (datetime.now(timezone.utc) - report_dt).days
            if age_days > 10:
                return self._null_signal(f"COT data stale ({age_days} days old)")
        except ValueError:
            pass

        sig = self._signals.get(pair_u)
        if not sig or not sig.direction:
            return self._null_signal(
                f"COT neutral: index={snap.cot_index:.0f} z={snap.nc_zscore:.2f}"
            )

        # ATR for sizing
        atr       = self._calc_atr(candles)
        stop_dist = atr * 2.0    # COT = weekly signal, wider stops
        tp1_dist  = atr * 3.0
        tp2_dist  = atr * 5.0
        rr        = tp1_dist / max(stop_dist, 1e-9)
        wp        = 0.50 + (sig.confidence - 0.40) * 0.25
        ev        = wp * rr - (1 - wp) * 1.0

        # Regime fit adjustment
        regime_f  = self.REGIME_FIT.get(regime, 0.55)

        return COTAgentSignal(
            direction=sig.direction,
            confidence=round(sig.confidence * regime_f, 3),
            win_probability=round(min(0.60, wp), 3),
            expected_value=round(ev, 4),
            stop_distance=stop_dist,
            tp1_distance=tp1_dist,
            tp2_distance=tp2_dist,
            regime_fit=regime_f,
            reasoning=(
                f"COT {pair_u} {sig.direction.upper()}: "
                f"index={snap.cot_index:.0f}/100 "
                f"z={snap.nc_zscore:+.2f} "
                f"nc_net={snap.nc_net:,.0f} "
                f"{'commercials confirm ✅' if sig.commercials_confirm else ''} "
                f"| {sig.reasoning}"
            ),
            raw_features={
                "cot_index":     snap.cot_index,
                "nc_zscore":     snap.nc_zscore,
                "nc_net":        snap.nc_net,
                "nc_pct_oi":     snap.nc_pct_oi,
                "comm_net":      snap.comm_net,
                "report_date":   snap.report_date,
                "commercials_confirm": sig.commercials_confirm},
            cot_signal=sig,
        )

    # ── Internal signal building ──────────────────────────────

    def _parse_row(self, row: Dict, pair: str) -> Optional[COTSnapshot]:
        """Parse one CFTC CSV row into COTSnapshot."""
        try:
            def safe_float(key: str) -> float:
                val = row.get(key, "0").strip().replace(",", "")
                return float(val) if val else 0.0

            nc_long  = safe_float("NonComm_Positions_Long_All")
            nc_short = safe_float("NonComm_Positions_Short_All")
            nc_net   = nc_long - nc_short

            comm_long  = safe_float("Comm_Positions_Long_All")
            comm_short = safe_float("Comm_Positions_Short_All")
            comm_net   = comm_long - comm_short

            oi           = safe_float("Open_Interest_All") or 1.0
            nc_chg_long  = safe_float("Change_in_NonComm_Long_All")
            nc_chg_short = safe_float("Change_in_NonComm_Short_All")
            nc_change    = nc_chg_long - nc_chg_short

            report_date = row.get("Report_Date_as_MM_DD_YYYY", "").strip()

            # For inverted pairs (USDJPY etc), invert nc_net
            if pair.upper() in INVERTED_PAIRS:
                nc_net   = -nc_net
                comm_net = -comm_net

            cot_index = self._store.compute_cot_index(pair, nc_net)
            nc_zscore = self._store.compute_zscore(pair, nc_net)
            nc_pct_oi = (nc_net / oi) * 100 if oi > 0 else 0.0

            return COTSnapshot(
                pair=pair, report_date=report_date,
                nc_net=nc_net, nc_long=nc_long, nc_short=nc_short,
                comm_net=comm_net, open_interest=oi,
                nc_change=nc_change, cot_index=cot_index,
                nc_zscore=nc_zscore, nc_pct_oi=nc_pct_oi,
            )
        except Exception as e:
            logger.debug(f"COT row parse error ({pair}): {e}")
            return None

    def _build_signal(self, pair: str, snap: COTSnapshot) -> COTSignal:
        """Build trading signal from COT snapshot."""
        # Non-commercials at extreme long → contrarian SELL
        if snap.cot_index >= COT_INDEX_EXTREME_LONG or snap.nc_zscore >= ZSCORE_EXTREME:
            direction    = "sell"
            signal_type  = "extreme_long"
            confidence   = min(0.75, 0.50 + (snap.cot_index - 50) / 100 + abs(snap.nc_zscore) * 0.05)
            reasoning    = (
                f"Non-commercials at extreme LONG ({snap.cot_index:.0f}/100). "
                f"Crowd max long = contrarian SELL."
            )
        # Non-commercials at extreme short → contrarian BUY
        elif snap.cot_index <= COT_INDEX_EXTREME_SHORT or snap.nc_zscore <= -ZSCORE_EXTREME:
            direction    = "buy"
            signal_type  = "extreme_short"
            confidence   = min(0.75, 0.50 + (50 - snap.cot_index) / 100 + abs(snap.nc_zscore) * 0.05)
            reasoning    = (
                f"Non-commercials at extreme SHORT ({snap.cot_index:.0f}/100). "
                f"Crowd max short = contrarian BUY."
            )
        else:
            return COTSignal(
                pair=pair, direction=None, signal_type="neutral",
                cot_index=snap.cot_index, nc_zscore=snap.nc_zscore,
                confidence=0.0, reasoning="COT neutral — no extreme positioning",
                report_date=snap.report_date, commercials_confirm=False,
            )

        # Commercial confirmation: do commercials oppose non-commercials?
        # Commercials are smart money hedgers — they're right at extremes
        comm_confirms = False
        if direction == "sell" and snap.comm_net < 0:
            comm_confirms = True    # commercials also net short = strong signal
            confidence    = min(0.80, confidence + 0.08)
        elif direction == "buy" and snap.comm_net > 0:
            comm_confirms = True    # commercials net long = strong signal
            confidence    = min(0.80, confidence + 0.08)

        return COTSignal(
            pair=pair, direction=direction, signal_type=signal_type,
            cot_index=snap.cot_index, nc_zscore=snap.nc_zscore,
            confidence=round(confidence, 3),
            reasoning=reasoning, report_date=snap.report_date,
            commercials_confirm=comm_confirms,
        )

    def _null_signal(self, reason: str) -> COTAgentSignal:
        return COTAgentSignal(
            direction=None, confidence=0.0, win_probability=0.0,
            expected_value=0.0, stop_distance=0.0, tp1_distance=0.0,
            tp2_distance=0.0, regime_fit=0.0, reasoning=reason,
        )

    def _calc_atr(self, candles: list, period: int = 14) -> float:
        if not candles or len(candles) < period + 1:
            return 0.001
        trs = []
        for i in range(1, min(period + 2, len(candles))):
            c, p = candles[-i], candles[-i - 1]
            try:
                tr = max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close))
                trs.append(tr)
            except AttributeError:
                continue
        return float(np.mean(trs)) if trs else 0.001

    def status_dict(self) -> Dict:
        return {
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
            "pairs_loaded": list(self._latest.keys()),
            "signals": {
                pair: {
                    "direction":   sig.direction,
                    "cot_index":   snap.cot_index if (snap := self._latest.get(pair)) else None,
                    "nc_zscore":   snap.nc_zscore if snap else None,
                    "confidence":  sig.confidence,
                    "report_date": sig.report_date}
                for pair, sig in self._signals.items()
            }}

    def next_refresh_utc(self) -> str:
        """Next expected COT release (Friday 20:30 UTC)."""
        now  = datetime.now(timezone.utc)
        days_until_friday = (4 - now.weekday()) % 7
        next_friday = now + timedelta(days=days_until_friday)
        next_friday = next_friday.replace(hour=20, minute=30, second=0, microsecond=0)
        if next_friday <= now:
            next_friday += timedelta(weeks=1)
        return next_friday.isoformat()
