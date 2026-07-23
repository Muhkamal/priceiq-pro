"""
PriceIQ Pro — AI Explanations v3.2
Updated with connection pooling, retries, caching, and walk-forward integration.
"""

import os
import json
import asyncio  # <-- ADD THIS IMPORT
import logging
from typing import Optional, Dict
from datetime import datetime, timezone, timedelta
from functools import lru_cache
import httpx

from app.models.schemas import TradeSignal, BacktestResult, PatternType
from app.core.config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Confidence tier helper
# ─────────────────────────────────────────────

def _confidence_label(confidence: float) -> str:
    if confidence >= 0.80:
        return "HIGH"
    elif confidence >= 0.55:
        return "MODERATE"
    else:
        return "LOW"

def _confidence_emoji(confidence: float) -> str:
    if confidence >= 0.80:
        return "🟢"
    elif confidence >= 0.55:
        return "🟡"
    else:
        return "🔴"


# ─────────────────────────────────────────────
# Pattern descriptions (all 7 patterns)
# ─────────────────────────────────────────────

PATTERN_DESCRIPTIONS = {
    PatternType.HAMMER: (
        "Hammer",
        "buyers aggressively rejected lower prices — the long lower shadow shows sellers tried to push down but failed completely",
        "buy the dip in an uptrend"
    ),
    PatternType.SHOOTING_STAR: (
        "Shooting Star",
        "sellers rejected higher prices — the long upper shadow shows buyers tried to push up but were overwhelmed",
        "sell the rally in a downtrend"
    ),
    PatternType.BULLISH_ENGULFING: (
        "Bullish Engulfing",
        "buying pressure completely overwhelmed the previous session's sellers — the current candle engulfs the entire prior bearish candle",
        "momentum shift from sellers to buyers at support"
    ),
    PatternType.BEARISH_ENGULFING: (
        "Bearish Engulfing",
        "selling pressure completely overwhelmed the previous session's buyers — the current candle engulfs the entire prior bullish candle",
        "momentum shift from buyers to sellers at resistance"
    ),
    PatternType.DOJI: (
        "Doji",
        "the market is in perfect indecision — open and close are almost identical, showing neither bulls nor bears are in control",
        "a potential turning point at a key level"
    ),
    PatternType.MORNING_STAR: (
        "Morning Star",
        "a 3-candle reversal: strong bearish move, then indecision, then a strong bullish close — a textbook bottom pattern",
        "high-probability reversal at support after a downtrend"
    ),
    PatternType.EVENING_STAR: (
        "Evening Star",
        "a 3-candle reversal: strong bullish move, then indecision, then a strong bearish close — a textbook top pattern",
        "high-probability reversal at resistance after an uptrend"
    ),
}


class AIExplanationService:
    """
    Generates human-readable trade and backtest explanations.
    Uses Claude API when available, falls back to rich local templates.

    Features:
    - Connection pooling for API calls
    - Retry logic for transient failures
    - LRU caching for explanations
    - Walk-forward result integration
    """

    def __init__(self):
        self.api_key = settings.CLAUDE_API_KEY
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.model = getattr(settings, 'CLAUDE_MODEL', "claude-sonnet-4-20250514")
        self.timeout = getattr(settings, 'CLAUDE_TIMEOUT', 15.0)
        self.max_tokens_signal = getattr(settings, 'CLAUDE_MAX_TOKENS_SIGNAL', 600)
        self.max_tokens_backtest = getattr(settings, 'CLAUDE_MAX_TOKENS_BACKTEST', 900)

        # Reusable HTTP client
        self._client: Optional[httpx.AsyncClient] = None

        # Simple explanation cache
        self._explanation_cache: Dict[str, tuple] = {}
        self._cache_ttl = timedelta(minutes=5)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create reusable HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _cache_key(self, signal: TradeSignal) -> str:
        """Generate cache key for signal explanation."""
        return f"{signal.pair}_{signal.pattern.value}_{signal.direction.value}_{_confidence_label(signal.confidence)}"

    def _get_cached(self, key: str) -> Optional[str]:
        """Get cached explanation if not expired."""
        cached = self._explanation_cache.get(key)
        if cached:
            text, timestamp = cached
            if datetime.now(timezone.utc) - timestamp < self._cache_ttl:
                return text
            del self._explanation_cache[key]
        return None

    def _set_cached(self, key: str, text: str):
        """Cache explanation with timestamp."""
        self._explanation_cache[key] = (text, datetime.now(timezone.utc))

    # ──────────────────────────────────────────
    # PUBLIC: Signal explanation
    # ──────────────────────────────────────────

    async def explain_signal(self, signal: TradeSignal, use_cache: bool = True) -> str:
        """Generate signal explanation with caching."""

        # Check cache
        if use_cache:
            cache_key = self._cache_key(signal)
            cached = self._get_cached(cache_key)
            if cached:
                logger.debug(f"AI cache hit for {signal.pair} {signal.pattern.value}")
                return cached

        if not self.api_key:
            explanation = self._generate_local_explanation(signal)
            if use_cache:
                self._set_cached(self._cache_key(signal), explanation)
            return explanation

        prompt = self._build_signal_prompt(signal)
        explanation = await self._call_claude(prompt, self.max_tokens_signal)

        if explanation:
            if use_cache:
                self._set_cached(self._cache_key(signal), explanation)
            return explanation

        # Fallback
        explanation = self._generate_local_explanation(signal)
        if use_cache:
            self._set_cached(self._cache_key(signal), explanation)
        return explanation

    # ──────────────────────────────────────────
    # PUBLIC: Backtest explanation
    # ──────────────────────────────────────────

    async def explain_backtest(
        self, 
        result: BacktestResult,
        walk_forward: Optional[Dict] = None,
    ) -> str:
        """Generate backtest explanation with optional walk-forward results."""

        if not self.api_key:
            return self._generate_local_backtest_summary(result, walk_forward)

        prompt = self._build_backtest_prompt(result, walk_forward)
        explanation = await self._call_claude(prompt, self.max_tokens_backtest)

        if explanation:
            return explanation

        return self._generate_local_backtest_summary(result, walk_forward)

    # ──────────────────────────────────────────
    # INTERNAL: Claude API call with retry
    # ──────────────────────────────────────────

    async def _call_claude(self, prompt: str, max_tokens: int) -> Optional[str]:
        """Call Claude API with retry logic."""
        max_retries = 2

        for attempt in range(max_retries):
            try:
                client = await self._get_client()
                response = await client.post(
                    self.api_url,
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "max_tokens": max_tokens,
                        "system": (
                            "You are a senior price action trader with 15 years of experience. "
                            "You explain trade setups clearly and honestly — you highlight both "
                            "the opportunity AND the risk. You never hype trades. "
                            "Use emojis sparingly. Speak like a mentor, not a salesperson."
                        ),
                        "messages": [{"role": "user", "content": prompt}]
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    return data["content"][0]["text"]
                elif response.status_code == 429:
                    # Rate limit — retry after delay
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt
                        logger.warning(f"Claude rate limit, retrying in {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    else:
                        logger.warning("Claude rate limit exceeded")
                        return None
                else:
                    logger.warning(f"Claude API error {response.status_code}: {response.text[:200]}")
                    return None

            except httpx.TimeoutException:
                if attempt < max_retries - 1:
                    logger.warning(f"Claude timeout, retrying ({attempt + 1}/{max_retries})")
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    logger.warning("Claude timeout after retries")
                    return None
            except Exception as e:
                logger.warning(f"Claude API error: {e}")
                return None

        return None

    # ──────────────────────────────────────────
    # PROMPT BUILDERS
    # ──────────────────────────────────────────

    def _build_signal_prompt(self, signal: TradeSignal) -> str:
        pattern_info = PATTERN_DESCRIPTIONS.get(signal.pattern, ("Unknown Pattern", "", ""))
        pattern_name, pattern_meaning, pattern_context = pattern_info

        conf_label = _confidence_label(signal.confidence)
        conf_emoji = _confidence_emoji(signal.confidence)

        sr_text = "\n".join([
            f"  • {'Support' if sr.was_support else 'Resistance'} at {sr.level:.5f} "
            f"(strength: {sr.strength:.0%}, touches: {sr.touches})"
            for sr in signal.support_resistance[:4]
        ]) or "  • No key levels identified"

        spread_note = (
            f"Note: Entry price ({signal.entry_price:.5f}) already accounts for spread and "
            f"slippage costs — this is the realistic fill price, not the raw close."
        )

        return f"""
Analyze this trade signal and explain it to a retail forex trader:

SIGNAL OVERVIEW
• Pair: {signal.pair} | Timeframe: {signal.timeframe}
• Direction: {signal.direction.value.upper()}
• Pattern: {pattern_name}
• Confidence: {signal.confidence:.0%} ({conf_label}) {conf_emoji}

WHAT THE PATTERN SHOWS
{pattern_meaning}

PRICE LEVELS
• Entry: {signal.entry_price:.5f}
• Stop Loss: {signal.stop_loss:.5f}
• Take Profit 1: {signal.take_profit_1:.5f} (R:R = 1:{signal.risk_reward_1:.1f})
• Take Profit 2: {signal.take_profit_2:.5f if signal.take_profit_2 else 'N/A'} (R:R = 1:{signal.risk_reward_2 or 0:.1f})
• {spread_note}

MARKET CONTEXT
• Trend: {signal.market_structure.trend.value.upper()}
• Market Stage: {signal.market_structure.stage.value}
• Trend Strength: {signal.market_structure.trend_strength:.0%}
• Healthy Trend: {'Yes' if signal.market_structure.is_healthy_trend else 'No — be cautious'}

KEY S/R LEVELS
{sr_text}

Write exactly 2 paragraphs:
1. Why this trade makes sense — market structure + pattern + level confluence
2. Risks and how the setup manages them — be honest, not overly optimistic

Tone: confident but honest. Match the language intensity to the {conf_label} confidence level.
"""

    def _build_backtest_prompt(self, result: BacktestResult, walk_forward: Optional[Dict] = None) -> str:
        m = result.metrics

        pattern_perf = "\n".join([
            f"  • {k.replace('_', ' ').title()}: {v:.1%} win rate"
            for k, v in (m.win_rate_by_pattern or {}).items()
        ]) or "  • No pattern data"

        stage_perf = "\n".join([
            f"  • {k.replace('_', ' ').title()}: {v:.1%} win rate"
            for k, v in (m.win_rate_by_stage or {}).items()
        ]) or "  • No stage data"

        monthly_pnl = "\n".join([
            f"  • {month}: ${pnl:+.2f}"
            for month, pnl in list((m.pnl_by_month or {}).items())[-6:]
        ]) or "  • No monthly data"

        wf_text = ""
        if walk_forward:
            wf_text = f"""
WALK-FORWARD VALIDATION
• Periods Tested: {walk_forward.get('n_periods_run', 'N/A')}
• Robustness Score: {walk_forward.get('robustness_score', 'N/A')}/100
• Efficiency Ratio: {walk_forward.get('efficiency_ratio', 'N/A'):.2f}
• Verdict: {walk_forward.get('verdict', 'N/A')}
"""

        return f"""
Evaluate these backtest results for a price action forex trading system:

VERDICT: {result.verdict} ({result.verdict_color.upper()})

IMPORTANT CONTEXT
- Backtest includes spread and slippage costs per trade (realistic modelling)
- Results use walk-forward validation to reduce curve-fitting bias
{wf_text}

PERFORMANCE METRICS
• Total Trades: {m.total_trades}
• Win Rate: {m.win_rate:.1%}
• Profit Factor: {m.profit_factor:.2f}
• Sharpe Ratio: {m.sharpe_ratio:.2f}
• Sortino Ratio: {m.sortino_ratio:.2f}
• Calmar Ratio: {m.calmar_ratio:.2f}
• Max Drawdown: {m.max_drawdown_percent:.1f}% (${m.max_drawdown_amount:.0f})
• Total Return: {m.total_return_percent:+.1f}%
• Expectancy: ${m.expectancy:.2f} per trade ({m.expectancy_per_r:+.2f}R)
• Avg Win: ${m.avg_win:.2f} | Avg Loss: ${m.avg_loss:.2f}
• Max Consec. Wins: {m.max_consecutive_wins} | Max Consec. Losses: {m.max_consecutive_losses}

PATTERN PERFORMANCE
{pattern_perf}

MARKET STAGE PERFORMANCE
{stage_perf}

RECENT MONTHLY P&L (last 6 months)
{monthly_pnl}

Write exactly 3 paragraphs:
1. Overall verdict — is this strategy viable for live trading? Be direct.
2. Biggest strengths and biggest weaknesses — be specific, not generic.
3. Concrete recommendations before going live — specific steps, not vague advice.

Be honest. If the results are borderline, say so. Traders trust honest analysis.
"""

    # ──────────────────────────────────────────
    # LOCAL FALLBACK — all 7 patterns covered
    # ──────────────────────────────────────────

    def _generate_local_explanation(self, signal: TradeSignal) -> str:
        pattern = signal.pattern
        direction = signal.direction.value.upper()
        trend = signal.market_structure.trend.value.replace("_", " ").title()
        stage = signal.market_structure.stage.value.replace("_", " ").title()
        conf_label = _confidence_label(signal.confidence)
        conf_emoji = _confidence_emoji(signal.confidence)

        pattern_info = PATTERN_DESCRIPTIONS.get(pattern, ("Pattern", "price action shows a setup", "potential trade"))
        pattern_name, pattern_meaning, pattern_context = pattern_info

        nearest_sr = signal.support_resistance[0] if signal.support_resistance else None
        sr_line = (
            f"at a {'support' if nearest_sr.was_support else 'resistance'} level "
            f"({nearest_sr.level:.5f}, strength {nearest_sr.strength:.0%})"
        ) if nearest_sr else "at a key price level"

        risk_pips = abs(signal.entry_price - signal.stop_loss)
        pip_mult = 10000 if "JPY" not in signal.pair else 100
        spread_note = (
            f"The entry price already factors in spread and slippage — "
            f"what you see ({signal.entry_price:.5f}) is your realistic fill, not the raw chart price."
        )

        para1 = (
            f"{conf_emoji} **Why This Trade Makes Sense** ({conf_label} Confidence)\n\n"
            f"The {signal.pair} market is in a **{trend}** during the **{stage}** stage. "
            f"Price pulled back {sr_line}, where a **{pattern_name}** has formed. "
            f"This matters because {pattern_meaning}. "
            f"This is a {pattern_context} — exactly the scenario M.A.E. is designed for. "
            f"Trend strength is {signal.market_structure.trend_strength:.0%}, "
            f"{'which is healthy.' if signal.market_structure.is_healthy_trend else 'though the trend is weakening — be cautious.'}"
        )

        para2 = (
            f"⚠️ **Risk Management**\n\n"
            f"Your stop is {signal.stop_loss:.5f} — {risk_pips * pip_mult:.1f} pips away. "
            f"If the market closes beyond that level, the thesis is invalidated and we exit with "
            f"a controlled loss. TP1 at {signal.take_profit_1:.5f} offers a 1:{signal.risk_reward_1:.1f} R:R, "
            f"and TP2 at {signal.take_profit_2:.5f if signal.take_profit_2 else 'N/A'} offers 1:{signal.risk_reward_2 or 0:.1f}. "
            f"{spread_note} "
            f"{'This is a high-confidence setup — consider full position size.' if conf_label == 'HIGH' else 'Given moderate confidence, consider half position size and wait for TP1 before moving stop to breakeven.' if conf_label == 'MODERATE' else 'Low confidence setup — if you trade it, use minimum position size (0.01 lots) and tight risk control.'}"
        )

        return f"{para1}\n\n{para2}"

    def _generate_local_backtest_summary(
        self, 
        result: BacktestResult,
        walk_forward: Optional[Dict] = None,
    ) -> str:
        m = result.metrics

        # Verdict assessment
        if result.verdict_color == "green":
            assessment = "This strategy demonstrates strong, consistent performance and may be ready for live trading."
        elif result.verdict_color == "yellow":
            assessment = "This strategy is profitable but has areas of concern that need addressing before live capital."
        else:
            assessment = "This strategy is NOT ready for live trading. The numbers do not support risking real money yet."

        # Walk-forward note
        wf_note = ""
        if walk_forward:
            wf_score = walk_forward.get('robustness_score', 'N/A')
            wf_verdict = walk_forward.get('verdict', 'N/A')
            wf_note = f"Walk-forward validation (robustness {wf_score}/100) confirms: {wf_verdict}\n\n"

        # Profit factor rating
        pf_label = "strong" if m.profit_factor > 1.5 else "marginal" if m.profit_factor > 1.0 else "negative"

        # Drawdown rating
        dd_label = "acceptable" if m.max_drawdown_percent < 15 else "concerning" if m.max_drawdown_percent < 25 else "dangerous"

        # Best and worst pattern
        patterns = m.win_rate_by_pattern or {}
        best_pattern = max(patterns, key=patterns.get) if patterns else "N/A"
        worst_pattern = min(patterns, key=patterns.get) if patterns else "N/A"

        # Best stage
        stages = m.win_rate_by_stage or {}
        best_stage = max(stages, key=stages.get) if stages else "N/A"

        return f"""
**Overall Assessment**

{assessment} Over {m.total_trades} trades (with realistic spread and slippage costs included), the strategy achieved a {m.win_rate:.1%} win rate and a {pf_label} profit factor of {m.profit_factor:.2f}. The Sharpe ratio of {m.sharpe_ratio:.2f} suggests {'strong' if m.sharpe_ratio > 1.5 else 'moderate' if m.sharpe_ratio > 0.8 else 'poor'} risk-adjusted returns. Total return over the test period: {m.total_return_percent:+.1f}%.

{wf_note}
**Strengths & Weaknesses**

Strengths: {'Positive expectancy ($' + str(round(m.expectancy, 2)) + ' per trade) means the edge is real.' if m.expectancy > 0 else 'No positive expectancy found.'} The best-performing pattern was **{best_pattern.replace('_', ' ').title()}** — focus your attention there. The strategy performs best in **{best_stage.replace('_', ' ').title()}** market conditions. Weaknesses: The {dd_label} maximum drawdown of {m.max_drawdown_percent:.1f}% (${m.max_drawdown_amount:.0f}) is your biggest psychological challenge — you must be prepared for {m.max_consecutive_losses} consecutive losses. Avoid trading the **{worst_pattern.replace('_', ' ').title()}** pattern until its win rate improves.

**Before Going Live**

1. Paper trade for 30 days minimum — track every signal the system generates, not just the ones you like.
2. Start live with 0.01 micro lots on one pair only — scale up only after 20 profitable live trades.
3. If drawdown exceeds {min(15, m.max_drawdown_percent * 0.6):.0f}% on your live account, stop and review.
4. Focus on {best_stage.replace('_', ' ')} market conditions and the {best_pattern.replace('_', ' ')} pattern — that's where the real edge lives.
5. Keep a trade journal comparing live results to backtest expectations — divergence > 20% means something has changed in the market.
"""


# Singleton instance
ai_service = AIExplanationService()
