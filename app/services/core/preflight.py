"""
PriceIQ Pro — Startup Preflight Validator v1.0

Validates everything required for safe operation BEFORE the system
accepts any signals. Fails loudly with a precise checklist.

Checks:
    ENV       — all required environment variables present
    BROKER    — broker API reachable and authenticated
    TELEGRAM  — bot token valid, chat_id reachable
    SUPABASE  — database connection live
    MODELS    — regime_model.pkl present and loadable
    FILES     — all persistence paths writable
    CALENDAR  — economic calendar feed reachable
    MEMORY    — sufficient RAM for ML operations
    CONFIG    — no contradictory settings (e.g. max_risk > 10%)

Usage in main.py lifespan:
    from app.services.core.preflight import PreflightValidator
    report = await PreflightValidator().run()
    if not report.passed:
        logger.critical(report.failure_summary())
        raise SystemExit(1)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Required environment variables ───────────────────────────
REQUIRED_ENV = {
    "TELEGRAM_BOT_TOKEN":  "Telegram bot token for alerts",
    "TELEGRAM_CHAT_ID":    "Telegram chat ID to send alerts to",
}

OPTIONAL_ENV = {
    "OANDA_API_KEY":       "OANDA broker API key",
    "OANDA_ACCOUNT_ID":    "OANDA account ID",
    "SUPABASE_URL":        "Supabase database URL",
    "SUPABASE_KEY":        "Supabase API key",
    "ALERT_EMAIL":         "Email address for digest alerts",
    "ACCOUNT_BALANCE":     "Starting account balance (default: 10000)",
    "RISK_PERCENT":        "Risk % per trade (default: 2.0)",
    "MAX_DRAWDOWN":        "Max drawdown before halt (default: 0.10)",
}

# ── Config sanity limits ─────────────────────────────────────
CONFIG_LIMITS = {
    "RISK_PERCENT":  (0.1, 10.0),
    "MAX_DRAWDOWN":  (0.01, 0.50),
    "ACCOUNT_BALANCE": (100.0, 10_000_000.0),
}


@dataclass
class CheckResult:
    name:    str
    passed:  bool
    message: str
    critical: bool = True   # if critical and failed → abort


@dataclass
class PreflightReport:
    checks:       List[CheckResult]
    passed:       bool
    timestamp:    str
    python_version: str
    platform:     str

    def failure_summary(self) -> str:
        failed = [c for c in self.checks if not c.passed]
        if not failed:
            return "All preflight checks passed."
        lines = [
            "=" * 60,
            "PREFLIGHT FAILED — System will not start",
            "=" * 60,
        ]
        for c in failed:
            severity = "CRITICAL" if c.critical else "WARNING"
            lines.append(f"  [{severity}] {c.name}: {c.message}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def success_summary(self) -> str:
        passed  = sum(1 for c in self.checks if c.passed)
        warned  = sum(1 for c in self.checks if not c.passed and not c.critical)
        lines   = [
            "=" * 60,
            f"PREFLIGHT PASSED ({passed}/{len(self.checks)} checks)",
        ]
        if warned:
            lines.append(f"  {warned} non-critical warning(s) — see logs")
        for c in self.checks:
            status = "✅" if c.passed else ("⚠️ " if not c.critical else "❌")
            lines.append(f"  {status} {c.name}: {c.message}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "passed":   self.passed,
            "timestamp": self.timestamp,
            "checks":   [
                {"name": c.name, "passed": c.passed,
                 "message": c.message, "critical": c.critical}
                for c in self.checks
            ],
        }


class PreflightValidator:
    """
    Runs all preflight checks and returns a PreflightReport.
    Non-critical failures (OPTIONAL_ENV, non-essential services)
    log warnings but don't abort. Critical failures abort startup.
    """

    def __init__(self, abort_on_failure: bool = True):
        self.abort_on_failure = abort_on_failure
        self._checks: List[CheckResult] = []

    async def run(self) -> PreflightReport:
        """Run all checks. Returns PreflightReport."""
        logger.info("Starting preflight validation...")

        # Run all checks
        self._check_python_version()
        self._check_required_env()
        self._check_optional_env()
        self._check_config_sanity()
        self._check_writable_paths()
        self._check_model_files()
        await self._check_telegram()
        await self._check_broker()
        await self._check_supabase()
        await self._check_calendar_feed()
        self._check_memory()
        self._check_dependencies()

        # Determine overall pass/fail (only critical failures count)
        critical_failures = [c for c in self._checks if not c.passed and c.critical]
        passed = len(critical_failures) == 0

        report = PreflightReport(
            checks=self._checks,
            passed=passed,
            timestamp=datetime.now(timezone.utc).isoformat(),
            python_version=sys.version,
            platform=sys.platform,
        )

        if passed:
            logger.info(f"\n{report.success_summary()}")
        else:
            logger.critical(f"\n{report.failure_summary()}")

        return report

    # ── Individual checks ─────────────────────────────────────

    def _check_python_version(self):
        major, minor = sys.version_info[:2]
        ok  = major == 3 and minor >= 9
        msg = f"Python {major}.{minor} {'(OK)' if ok else '(need ≥ 3.9)'}"
        self._add(CheckResult("Python version", ok, msg, critical=True))

    def _check_required_env(self):
        for var, desc in REQUIRED_ENV.items():
            val = os.environ.get(var, "").strip()
            ok  = bool(val)
            msg = f"{'set' if ok else 'MISSING'} — {desc}"
            self._add(CheckResult(f"ENV:{var}", ok, msg, critical=True))

    def _check_optional_env(self):
        for var, desc in OPTIONAL_ENV.items():
            val = os.environ.get(var, "").strip()
            ok  = bool(val)
            msg = f"{'set' if ok else 'not set (optional)'} — {desc}"
            self._add(CheckResult(f"ENV:{var}", ok, msg, critical=False))

    def _check_config_sanity(self):
        for var, (lo, hi) in CONFIG_LIMITS.items():
            raw = os.environ.get(var, "")
            if not raw:
                continue   # optional, skip if not set
            try:
                val = float(raw)
                ok  = lo <= val <= hi
                msg = f"{val} {'(OK)' if ok else f'(must be {lo}–{hi})'}"
            except ValueError:
                ok  = False
                msg = f"'{raw}' is not a valid number"
            self._add(CheckResult(f"CONFIG:{var}", ok, msg, critical=ok is False))

        # Cross-check: RISK_PERCENT × 5 < MAX_DRAWDOWN (5 consecutive losses shouldn't hit hard stop)
        risk  = float(os.environ.get("RISK_PERCENT",  "2.0"))
        maxdd = float(os.environ.get("MAX_DRAWDOWN",  "0.10"))
        coherent = (risk / 100) * 5 < maxdd
        self._add(CheckResult(
            "CONFIG:risk_coherence",
            coherent,
            f"5× risk ({risk*5:.1f}%) {'<' if coherent else '>='} max_dd ({maxdd*100:.0f}%)",
            critical=False,
        ))

    def _check_writable_paths(self):
        paths = [
            ("regime_model.pkl",          "Regime classifier model"),
            ("learning_state.json",        "Learning loop state"),
            ("regime_weights.json",        "Regime-conditional weights"),
            ("win_prob_calibrator.json",   "Win probability calibrator"),
            ("regime_transitions.json",    "Regime transition matrix"),
            ("trade_journal.json",         "Trade journal"),
        ]
        for path, desc in paths:
            dir_path = os.path.dirname(os.path.abspath(path)) or "."
            writable = os.access(dir_path, os.W_OK)
            self._add(CheckResult(
                f"PATH:{path}",
                writable,
                f"{'writable' if writable else 'NOT WRITABLE'} — {desc}",
                critical=False,
            ))

    def _check_model_files(self):
        model_path = os.environ.get("REGIME_MODEL_PATH", "regime_model.pkl")
        exists     = os.path.exists(model_path)
        if exists:
            try:
                import pickle
                with open(model_path, "rb") as f:
                    data = pickle.load(f)
                loadable = "model" in data and "scaler" in data
                msg = f"found and loadable at {model_path}"
            except Exception as e:
                loadable = False
                msg = f"found but corrupted: {e}"
        else:
            loadable = False
            msg = f"not found at {model_path} — will use heuristic fallback"

        # Not critical — heuristic fallback exists
        self._add(CheckResult("MODEL:regime_classifier", loadable, msg, critical=False))

    async def _check_telegram(self):
        token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            self._add(CheckResult(
                "TELEGRAM:connectivity", False,
                "Token or chat_id not set — skipping connectivity check",
                critical=False,
            ))
            return
        try:
            import asyncio
            from urllib.request import urlopen, Request
            url = f"https://api.telegram.org/bot{token}/getMe"
            req = Request(url, headers={"User-Agent": "PriceIQ-Preflight"})
            resp = await asyncio.to_thread(urlopen, req, 5)
            ok   = resp.status == 200
            self._add(CheckResult(
                "TELEGRAM:connectivity", ok,
                "bot reachable ✅" if ok else "bot unreachable ❌",
                critical=True,
            ))
        except Exception as e:
            self._add(CheckResult(
                "TELEGRAM:connectivity", False,
                f"connection failed: {e}",
                critical=True,
            ))

    async def _check_broker(self):
        api_key    = os.environ.get("OANDA_API_KEY", "")
        account_id = os.environ.get("OANDA_ACCOUNT_ID", "")
        if not api_key or not account_id:
            self._add(CheckResult(
                "BROKER:OANDA", False,
                "API key or account ID not set — broker disabled",
                critical=False,
            ))
            return
        try:
            from urllib.request import urlopen, Request
            url = f"https://api-fxtrade.oanda.com/v3/accounts/{account_id}"
            req = Request(url, headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            })
            resp = await asyncio.to_thread(urlopen, req, 8)
            ok   = resp.status == 200
            self._add(CheckResult(
                "BROKER:OANDA", ok,
                "authenticated and reachable ✅" if ok else "auth failed ❌",
                critical=False,
            ))
        except Exception as e:
            self._add(CheckResult(
                "BROKER:OANDA", False,
                f"connection failed: {e}",
                critical=False,
            ))

    async def _check_supabase(self):
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")
        if not url or not key:
            self._add(CheckResult(
                "DB:Supabase", False,
                "URL or key not set — database disabled",
                critical=False,
            ))
            return
        try:
            from urllib.request import urlopen, Request
            req = Request(
                f"{url}/rest/v1/",
                headers={"apikey": key, "Authorization": f"Bearer {key}"},
            )
            resp = await asyncio.to_thread(urlopen, req, 8)
            ok   = resp.status in (200, 400)   # 400 = no table specified, still reachable
            self._add(CheckResult(
                "DB:Supabase", ok,
                "reachable ✅" if ok else "unreachable ❌",
                critical=False,
            ))
        except Exception as e:
            self._add(CheckResult(
                "DB:Supabase", False,
                f"connection failed: {e}",
                critical=False,
            ))

    async def _check_calendar_feed(self):
        try:
            from urllib.request import urlopen, Request
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
            req = Request(url, headers={"User-Agent": "PriceIQ-Preflight"})
            resp = await asyncio.to_thread(urlopen, req, 8)
            ok   = resp.status == 200
            self._add(CheckResult(
                "CALENDAR:ForexFactory", ok,
                "RSS feed reachable ✅" if ok else "RSS feed unreachable ❌",
                critical=False,
            ))
        except Exception as e:
            self._add(CheckResult(
                "CALENDAR:ForexFactory", False,
                f"unreachable: {e} — will use manual events only",
                critical=False,
            ))

    def _check_memory(self):
        try:
            import psutil
            mem  = psutil.virtual_memory()
            free_mb = mem.available / 1024 / 1024
            ok   = free_mb >= 256
            msg  = f"{free_mb:.0f} MB available {'(OK)' if ok else '(LOW — need ≥ 256 MB)'}"
            self._add(CheckResult("MEMORY:available", ok, msg, critical=False))
        except ImportError:
            self._add(CheckResult(
                "MEMORY:available", True,
                "psutil not installed — skipping memory check",
                critical=False,
            ))

    def _check_dependencies(self):
        deps = {
            "numpy":      "core numerics",
            "sklearn":    "ML utilities",
            "fastapi":    "API layer",
            "apscheduler":"job scheduler",
        }
        optional_deps = {
            "xgboost":    "regime classifier (falls back to GradientBoosting)",
            "psutil":     "memory monitoring",
        }
        for pkg, desc in deps.items():
            try:
                __import__(pkg)
                ok, msg = True, f"installed — {desc}"
            except ImportError:
                ok, msg = False, f"MISSING — {desc}"
            self._add(CheckResult(f"DEP:{pkg}", ok, msg, critical=True))

        for pkg, desc in optional_deps.items():
            try:
                __import__(pkg)
                ok, msg = True, f"installed — {desc}"
            except ImportError:
                ok, msg = False, f"not installed — {desc}"
            self._add(CheckResult(f"DEP:{pkg}", ok, msg, critical=False))

    def _add(self, check: CheckResult):
        self._checks.append(check)
        symbol = "✅" if check.passed else ("⚠️ " if not check.critical else "❌")
        logger.debug(f"Preflight {symbol} {check.name}: {check.message}")
