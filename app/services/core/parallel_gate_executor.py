"""
PriceIQ Pro — Parallel Gate Executor v1.0

Fixes the sequential gate bottleneck in v5_orchestrator_final.py.

Problem:
    Gates 0-3 (Anomaly, Calendar, Drift, DataQuality) are fully independent.
    None depends on the output of another.
    Running them sequentially adds latency with zero benefit.
    At 6 pairs × 2 timeframes = 12 cycles, this compounds significantly.
    At 15M timeframe it becomes a real constraint.

Solution:
    asyncio.gather() runs all independent gates concurrently.
    Total time = slowest gate, not sum of all gates.

    Sequential: 4 gates × ~50ms each = 200ms per pair
    Parallel:   max(50ms, 50ms, 50ms, 50ms) = 50ms per pair
    Saving:     150ms per pair, 1.8 seconds per full 12-cycle scan

Gate dependency graph:
    INDEPENDENT (run in parallel):
        Gate 0: AnomalyDetector.check()
        Gate 1: EconomicCalendar.check_blackout()
        Gate 2: FeatureDriftMonitor.check()
        Gate 3: DataQualityValidator.validate()

    SEQUENTIAL (each depends on previous):
        Gate 4:  RegimeClassifier.predict()   (needs clean data from Gate 3)
        Gate 5:  RegimeTransitionModel.update()
        Gate 6:  AgentOrchestrator.run()      (needs regime from Gate 4)
        Gate 7:  SignalConflictResolver        (needs agent output from Gate 6)
        Gate 8:  WinProbCalibrator             (needs agent + regime)
        Gate 9:  EV check
        Gate 10: Confidence threshold
        Gate 11: RiskGovernor
        Gate 12: DynamicCorrelation
        Gate 13: VaR
        Gate 14: VolSizer
        Gate 15: ExecutionIntelligence

Usage:
    executor = ParallelGateExecutor(v5)
    result   = await executor.run_parallel_gates(candles, pair, now)
    if result.any_hard_block:
        return _no_signal(result.block_reason)
    # proceed with regime + agents using result.drift_features
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ParallelGateResult:
    """Combined output of all four parallel gates."""
    # Gate 0: Anomaly
    anomaly_blocked:    bool
    anomaly_reason:     str
    anomaly_issues:     List[str]
    anomaly_size_mult:  float

    # Gate 1: Calendar
    calendar_blocked:   bool
    calendar_reason:    str
    calendar_size_mult: float

    # Gate 2: Drift
    drift_blocked:      bool
    drift_warned:       bool
    drift_reason:       str
    drift_features:     Optional[Dict]

    # Gate 3: Data Quality
    dq_blocked:         bool
    dq_warned:          bool
    dq_reason:          str

    # Combined
    any_hard_block:     bool
    block_reason:       str
    combined_size_mult: float    # product of all size multipliers
    elapsed_ms:         float    # wall-clock time for all 4 gates

    @property
    def should_proceed(self) -> bool:
        return not self.any_hard_block


class ParallelGateExecutor:
    """
    Runs independent gates 0-3 concurrently using asyncio.gather.
    Integrates with V5OrchestratorFinal.
    """

    def __init__(self, v5):
        """
        Args:
            v5: V5OrchestratorFinal instance (or any object with
                anomaly_det, calendar, drift_monitor, data_validator attrs)
        """
        self.v5 = v5

    async def run_parallel_gates(
        self,
        candles:  List,
        pair:     str,
        now:      datetime,
        timeframe: str = "1h",
    ) -> ParallelGateResult:
        """
        Run all four independent gates in parallel.
        Returns combined result in ~max(gate_times) instead of sum.
        """
        import time
        t0 = time.monotonic()

        # Run all four gates concurrently
        results = await asyncio.gather(
            self._run_anomaly_gate(candles, pair),
            self._run_calendar_gate(pair, now),
            self._run_drift_gate(candles),
            self._run_dq_gate(candles, pair, timeframe),
            return_exceptions=True,   # don't let one failure kill others
        )

        elapsed_ms = (time.monotonic() - t0) * 1000

        # Unpack results (handle any exceptions gracefully)
        anomaly_r  = results[0] if not isinstance(results[0], Exception) else self._anomaly_safe_default()
        calendar_r = results[1] if not isinstance(results[1], Exception) else self._calendar_safe_default()
        drift_r    = results[2] if not isinstance(results[2], Exception) else self._drift_safe_default()
        dq_r       = results[3] if not isinstance(results[3], Exception) else self._dq_safe_default()

        # Log exceptions
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                gate_names = ["anomaly", "calendar", "drift", "data_quality"]
                logger.warning(f"Parallel gate {gate_names[i]} raised: {r}")

        # Combined block check
        any_block    = (anomaly_r["blocked"] or calendar_r["blocked"]
                        or drift_r["blocked"] or dq_r["blocked"])
        block_reason = ""
        if any_block:
            reasons = [
                r["reason"] for r in [anomaly_r, calendar_r, drift_r, dq_r]
                if r["blocked"]
            ]
            block_reason = " | ".join(reasons)

        # Combined size multiplier (product of all reductions)
        combined_mult = (
            anomaly_r["size_mult"]
            * calendar_r["size_mult"]
            * drift_r.get("size_mult", 1.0)
            * dq_r.get("size_mult", 1.0)
        )

        logger.debug(
            f"Parallel gates [{pair}]: {elapsed_ms:.1f}ms | "
            f"anomaly={anomaly_r['blocked']} calendar={calendar_r['blocked']} "
            f"drift={drift_r['blocked']} dq={dq_r['blocked']} "
            f"size_mult={combined_mult:.2f}"
        )

        return ParallelGateResult(
            anomaly_blocked=anomaly_r["blocked"],
            anomaly_reason=anomaly_r["reason"],
            anomaly_issues=anomaly_r.get("issues", []),
            anomaly_size_mult=anomaly_r["size_mult"],
            calendar_blocked=calendar_r["blocked"],
            calendar_reason=calendar_r["reason"],
            calendar_size_mult=calendar_r["size_mult"],
            drift_blocked=drift_r["blocked"],
            drift_warned=drift_r.get("warn", False),
            drift_reason=drift_r["reason"],
            drift_features=drift_r.get("features"),
            dq_blocked=dq_r["blocked"],
            dq_warned=dq_r.get("warn", False),
            dq_reason=dq_r["reason"],
            any_hard_block=any_block,
            block_reason=block_reason,
            combined_size_mult=round(combined_mult, 4),
            elapsed_ms=round(elapsed_ms, 2),
        )

    # ── Individual gate runners ────────────────────────────────

    async def _run_anomaly_gate(self, candles: List, pair: str) -> Dict:
        """Gate 0: Anomaly detection (sync wrapped in thread)."""
        try:
            report = await asyncio.to_thread(
                self.v5.anomaly_det.check, candles, pair
            )
            return {
                "blocked":    report.block_signals,
                "reason":     report.reason,
                "issues":     report.anomalies,
                "size_mult":  report.size_multiplier,
                "warn":       report.warn,
            }
        except Exception as e:
            logger.debug(f"Anomaly gate error: {e}")
            return self._anomaly_safe_default()

    async def _run_calendar_gate(self, pair: str, now: datetime) -> Dict:
        """Gate 1: Economic calendar blackout check (sync)."""
        try:
            check = self.v5.calendar.check_blackout(pair, now)
            return {
                "blocked":   check.blocked,
                "reason":    check.reason,
                "size_mult": check.size_mult,
            }
        except Exception as e:
            logger.debug(f"Calendar gate error: {e}")
            return self._calendar_safe_default()

    async def _run_drift_gate(self, candles: List) -> Dict:
        """Gate 2: Feature drift monitoring (sync wrapped in thread)."""
        try:
            # Feature extraction is CPU-bound → thread pool
            feats = await asyncio.to_thread(
                self.v5.regime_clf.extractor.extract, candles
            )
            if feats:
                self.v5.drift_monitor.update(feats)
                # Check is fast (just comparing cached values)
                report = self.v5.drift_monitor.check()
                if report:
                    if report.severe:
                        return {
                            "blocked":  True,
                            "warn":     True,
                            "reason":   report.recommendation,
                            "features": feats,
                            "size_mult": 0.0,
                        }
                    return {
                        "blocked":  False,
                        "warn":     report.warn,
                        "reason":   "drift OK" if not report.warn else f"drift warn: {report.drifted_features}",
                        "features": feats,
                        "size_mult": 0.75 if report.warn else 1.0,
                    }
            return {
                "blocked": False, "warn": False,
                "reason": "no features extracted", "features": feats,
                "size_mult": 1.0,
            }
        except Exception as e:
            logger.debug(f"Drift gate error: {e}")
            return self._drift_safe_default()

    async def _run_dq_gate(self, candles: List, pair: str, timeframe: str) -> Dict:
        """Gate 3: Data quality validation (sync wrapped in thread)."""
        try:
            report = await asyncio.to_thread(
                self.v5.data_validator.validate, candles, pair, timeframe
            )
            return {
                "blocked":  report.block,
                "warn":     report.warn,
                "reason":   report.summary(),
                "size_mult": 0.5 if (report.warn and not report.block) else 1.0,
            }
        except Exception as e:
            logger.debug(f"DQ gate error: {e}")
            return self._dq_safe_default()

    # ── Safe defaults (on gate error, don't block) ─────────────

    def _anomaly_safe_default(self) -> Dict:
        return {"blocked": False, "reason": "anomaly gate error (bypassed)",
                "issues": [], "size_mult": 1.0, "warn": False}

    def _calendar_safe_default(self) -> Dict:
        return {"blocked": False, "reason": "calendar gate error (bypassed)",
                "size_mult": 1.0}

    def _drift_safe_default(self) -> Dict:
        return {"blocked": False, "warn": False,
                "reason": "drift gate error (bypassed)",
                "features": None, "size_mult": 1.0}

    def _dq_safe_default(self) -> Dict:
        return {"blocked": False, "warn": False,
                "reason": "dq gate error (bypassed)", "size_mult": 1.0}
