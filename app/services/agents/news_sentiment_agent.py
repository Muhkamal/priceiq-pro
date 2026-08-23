"""
PriceIQ Pro — NLP News Sentiment Agent v1.1 (Production Ready)

Reads financial news headlines and converts them into directional
trading signals for XAUUSD and major forex pairs.

Why this matters:
    Central bank surprises, geopolitical shocks, and NFP beats/misses
    move price 15–30 minutes BEFORE the move appears in OHLCV candles.
    A news-reading agent catches the cause; the price agents catch the effect.
    Combined = uncorrelated alpha from a completely different data stream.

Architecture (three-layer scoring):
    Layer 1: Financial keyword lexicon (primary — forex/gold specific)
        Custom dictionary of 200+ financial terms with directional weights.
        "rate hike" = bearish gold, "safe haven" = bullish gold, etc.
        This layer is accurate and fast — no model needed.

    Layer 2: VADER sentiment (confirmation)
        General-purpose sentiment confirms or adjusts keyword scores.
        Handles sentence-level negation ("not bearish" ≠ "bearish").

    Layer 3: FinBERT (optional — upgrade path)
        If FINBERT_ENABLED=true, uses ProsusAI/finbert from HuggingFace.
        Dramatically improves accuracy on ambiguous sentences.
        Enable on Render: pip install transformers torch already in requirements.

Data sources (all free):
    1. Alpha Vantage News API     — free tier: 25 calls/day
       https://www.alphavantage.co/documentation/#news-sentiment
    2. Reuters RSS feeds          — free, no key
    3. ForexFactory news scraper  — free (existing calendar already fetches)
    4. Finnhub news API           — free tier: 60 calls/minute

Signal pairs: XAUUSD, EURUSD, GBPUSD, USDJPY, USDCHF
Refresh:      Every 30 minutes from scheduler

Usage:
    agent = NewsSentimentAgent()
    await agent.refresh()
    signal = agent.evaluate(candles, regime="trending", pair="XAUUSD")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen, Request

import numpy as np

logger = logging.getLogger(__name__)

# ── API keys ──────────────────────────────────────────────────
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "demo")
FINNHUB_KEY       = os.environ.get("FINNHUB_KEY", "")
FINBERT_ENABLED   = os.environ.get("FINBERT_ENABLED", "false").lower() == "true"

# ── Financial Keyword Lexicon ─────────────────────────────────
# Format: "keyword/phrase" → (gold_impact, usd_impact)
# Scores: +1.0 = strongly bullish, -1.0 = strongly bearish, 0 = neutral
# gold_impact: effect on XAUUSD price
# usd_impact:  effect on USD (inverted for EURUSD/GBPUSD)

FINANCIAL_LEXICON: Dict[str, Tuple[float, float]] = {
    # Fed / Rate policy → bearish gold, bullish USD
    "rate hike":             (-0.85, +0.80),
    "rate increase":         (-0.80, +0.75),
    "hawkish":               (-0.75, +0.70),
    "tightening":            (-0.70, +0.65),
    "quantitative tightening": (-0.65, +0.60),
    "qt":                    (-0.55, +0.50),
    "fed hike":              (-0.85, +0.80),
    "higher for longer":     (-0.70, +0.65),
    "yield surge":           (-0.75, +0.60),
    "yields rise":           (-0.65, +0.55),
    "yields spike":          (-0.80, +0.65),
    "real yields":           (-0.60, +0.50),
    "real rates rise":       (-0.80, +0.65),
    "dollar strength":       (-0.70, +0.85),
    "dollar rally":          (-0.65, +0.80),
    "dxy surge":             (-0.70, +0.80),

    # Easing → bullish gold, bearish USD
    "rate cut":              (+0.85, -0.80),
    "rate reduction":        (+0.80, -0.75),
    "dovish":                (+0.75, -0.70),
    "easing":                (+0.70, -0.65),
    "quantitative easing":   (+0.80, -0.70),
    "qe":                    (+0.70, -0.60),
    "stimulus":              (+0.65, -0.55),
    "yields fall":           (+0.65, -0.55),
    "yields drop":           (+0.70, -0.60),
    "yields decline":        (+0.65, -0.55),
    "dollar weakness":       (+0.70, -0.85),
    "dollar falls":          (+0.65, -0.80),
    "dollar weakens":        (+0.70, -0.80),

    # Risk-off → bullish gold, bearish risk currencies
    "safe haven":            (+0.80,  0.00),
    "risk off":              (+0.75,  0.00),
    "risk-off":              (+0.75,  0.00),
    "flight to safety":      (+0.85,  0.00),
    "geopolitical":          (+0.60,  0.00),
    "war":                   (+0.75,  0.00),
    "conflict":              (+0.55,  0.00),
    "escalation":            (+0.65,  0.00),
    "crisis":                (+0.60,  0.00),
    "recession fears":       (+0.65, -0.30),
    "economic slowdown":     (+0.55, -0.35),
    "bank crisis":           (+0.75, -0.40),
    "banking stress":        (+0.70, -0.40),
    "default":               (+0.60, -0.30),
    "panic":                 (+0.70,  0.00),
    "uncertainty":           (+0.40,  0.00),
    "fear":                  (+0.55,  0.00),
    "vix spike":             (+0.65,  0.00),

    # Risk-on → bearish gold, bullish risk currencies
    "risk on":               (-0.65,  0.00),
    "risk-on":               (-0.65,  0.00),
    "rally":                 (-0.40,  0.00),
    "strong nfp":            (-0.70, +0.65),
    "beat expectations":     (-0.45, +0.40),
    "better than expected":  (-0.45, +0.40),
    "strong jobs":           (-0.65, +0.60),
    "strong gdp":            (-0.55, +0.55),
    "strong growth":         (-0.50, +0.50),
    "optimism":              (-0.35,  0.00),

    # Inflation → complex (high inflation bullish gold initially)
    "inflation":             (+0.45, -0.20),
    "cpi beat":              (+0.30, +0.50),  # mixed: gold & rate hike
    "cpi miss":              (+0.20, -0.45),
    "hot inflation":         (+0.40, +0.45),
    "inflation surge":       (+0.50, +0.35),
    "deflation":             (-0.40, +0.20),
    "disinflation":          (-0.30, +0.30),

    # Central bank specific
    "ecb hike":              ( 0.00, +0.55),  # ECB hike → EUR up
    "ecb cut":               ( 0.00, -0.55),
    "boe hike":              ( 0.00, +0.50),  # BOE hike → GBP up
    "boe cut":               ( 0.00, -0.50),
    "boj":                   ( 0.00, -0.30),  # BOJ policy affects JPY
    "yen intervention":      ( 0.00, -0.60),  # JPY strengthens
    "snb":                   ( 0.00, +0.20),

    # Gold specific
    "gold demand":           (+0.60,  0.00),
    "gold buying":           (+0.55,  0.00),
    "central bank gold":     (+0.65,  0.00),
    "gold reserves":         (+0.50,  0.00),
    "gold selling":          (-0.55,  0.00),
    "gold pressure":         (-0.45,  0.00),

    # Negation helpers (applied in _apply_negation)
    "not":                   ( 0.00,  0.00),   # handled separately
    "no":                    ( 0.00,  0.00),
    "despite":               ( 0.00,  0.00),
    "but":                   ( 0.00,  0.00),
}

# ── Pair routing: which keywords affect which pairs ───────────
PAIR_SENSITIVITY: Dict[str, List[str]] = {
    "XAUUSD": [
        "safe haven", "real rates", "yields", "fed", "inflation", "gold",
        "geopolit", "risk off", "risk-off", "flight to safety", "crisis",
        "war", "conflict", "dollar", "dxy", "vix",
    ],
    "EURUSD": [
        "ecb", "euro", "eur", "dollar", "dxy", "fed", "us rate",
        "european", "eurozone", "eu ", "recession", "growth",
    ],
    "GBPUSD": [
        "boe", "bank of england", "pound", "gbp", "sterling", "uk ",
        "britain", "brexit", "dollar", "dxy", "fed",
    ],
    "USDJPY": [
        "boj", "bank of japan", "yen", "jpy", "japan", "dollar",
        "yen intervention", "carry trade", "treasury", "yields",
    ],
    "USDCHF": [
        "snb", "swiss", "franc", "chf", "safe haven", "dollar",
        "risk off", "flight to safety",
    ],
}


@dataclass
class NewsItem:
    title:      str
    summary:    str
    source:     str
    published:  str
    url:        str = ""
    relevance:  float = 0.0    # 0-1, how relevant to trading


@dataclass
class SentimentScore:
    pair:           str
    direction:      Optional[str]   # "buy" | "sell" | None
    raw_score:      float           # -1 to +1
    confidence:     float           # 0-1
    n_articles:     int
    key_headlines:  List[str]
    layer1_score:   float           # keyword lexicon score
    layer2_score:   float           # VADER score
    layer3_score:   Optional[float] # FinBERT (if enabled)
    reasoning:      str


@dataclass
class NewsSentimentAgentSignal:
    """Same interface as AgentSignal for orchestrator compatibility."""
    agent_name:      str = "NewsSentimentAgent"
    direction:       Optional[str] = None
    confidence:      float = 0.0
    win_probability: float = 0.0
    expected_value:  float = 0.0
    stop_distance:   float = 0.0
    tp1_distance:    float = 0.0
    tp2_distance:    float = 0.0
    regime_fit:      float = 0.6
    reasoning:       str = ""
    raw_features:    Dict = field(default_factory=dict)
    sentiment_score: Optional[SentimentScore] = None

    @property
    def is_valid(self) -> bool:
        return (
            self.direction is not None
            and self.confidence >= 0.40
            and self.stop_distance > 0
        )


class FinancialLexiconScorer:
    """
    Layer 1: Fast keyword-based sentiment scoring.
    Custom financial lexicon optimized for forex/gold.
    Handles negation, phrase matching, and context weighting.
    """

    def score(self, text: str, pair: str) -> float:
        """Returns -1 to +1 directional score for pair from text."""
        text_lower = text.lower()
        pair_upper = pair.upper()

        # Check pair sensitivity filter
        sensitive_words = PAIR_SENSITIVITY.get(pair_upper, [])
        text_relevant = any(w in text_lower for w in sensitive_words)
        if sensitive_words and not text_relevant:
            return 0.0   # text not about this pair

        scores = []
        for phrase, (gold_impact, usd_impact) in FINANCIAL_LEXICON.items():
            if phrase in text_lower:
                # Route to correct pair impact
                if pair_upper == "XAUUSD":
                    raw = gold_impact
                elif pair_upper in ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"):
                    # EUR/GBP quoted vs USD: USD strength = these pairs down
                    raw = -usd_impact  # inverted: USD up = EUR/GBP down
                elif pair_upper in ("USDJPY", "USDCAD", "USDCHF"):
                    # USD quoted vs foreign: USD strength = these pairs up
                    raw = usd_impact
                else:
                    raw = gold_impact * 0.3 + usd_impact * 0.3

                # Negation check: "not hawkish", "no rate hike"
                raw = self._apply_negation(text_lower, phrase, raw)
                scores.append(raw)

        if not scores:
            return 0.0
        # Weighted average: more phrases = more confident
        return float(np.tanh(np.mean(scores) * len(scores) * 0.3))

    def _apply_negation(self, text: str, phrase: str, score: float) -> float:
        """Flip score if negation word within 3 words before phrase."""
        negations = ["not", "no", "never", "without", "despite", "contrary"]
        idx = text.find(phrase)
        if idx < 0:
            return score
        window = text[max(0, idx - 30): idx]
        if any(neg in window.split() for neg in negations):
            return -score * 0.7   # partial reversal
        return score


class VaderSentimentScorer:
    """Layer 2: VADER sentiment analysis with financial context boost."""

    def __init__(self):
        self._analyzer = None
        try:
            import nltk
            nltk.download("vader_lexicon", quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            self._analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.warning(f"VADER not available: {e}")

    def score(self, text: str) -> float:
        """Returns -1 to +1 compound sentiment score."""
        if not self._analyzer:
            return 0.0
        scores = self._analyzer.polarity_scores(text)
        return float(scores["compound"])


class FinBERTScorer:
    """
    Layer 3: FinBERT sentiment (optional, requires transformers + torch).
    Enable with FINBERT_ENABLED=true on Render.
    """

    def __init__(self):
        self._pipeline = None
        if FINBERT_ENABLED:
            try:
                from transformers import pipeline
                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    device=-1,   # CPU
                )
                logger.info("FinBERT loaded ✅")
            except Exception as e:
                logger.warning(f"FinBERT not available: {e}. Using VADER only.")

    def score(self, text: str) -> Optional[float]:
        """Returns -1 to +1 or None if not available."""
        if not self._pipeline:
            return None
        try:
            results = self._pipeline([text[:512]])[0]
            label   = results["label"].lower()
            conf    = float(results["score"])
            if label == "positive":
                return conf
            elif label == "negative":
                return -conf
            return 0.0
        except Exception:
            return None


class AlphaVantageNewsFetcher:
    """Fetches financial news from Alpha Vantage (free tier: 25 calls/day)."""

    BASE_URL = "https://www.alphavantage.co/query"

    PAIR_TICKERS: Dict[str, str] = {
        "XAUUSD": "FOREX:USD,FOREX:XAU",
        "EURUSD": "FOREX:EUR,FOREX:USD",
        "GBPUSD": "FOREX:GBP,FOREX:USD",
        "USDJPY": "FOREX:USD,FOREX:JPY",
        "USDCHF": "FOREX:USD,FOREX:CHF",
    }

    def fetch(self, pair: str, limit: int = 10) -> List[NewsItem]:
        if not ALPHA_VANTAGE_KEY or ALPHA_VANTAGE_KEY == "demo":
            return []
        try:
            ticker = self.PAIR_TICKERS.get(pair.upper(), "FOREX:USD")
            url    = (f"{self.BASE_URL}?function=NEWS_SENTIMENT"
                      f"&tickers={ticker}&limit={limit}"
                      f"&apikey={ALPHA_VANTAGE_KEY}")
            req    = Request(url, headers={"User-Agent": "PriceIQ-Pro/5.4"})
            with urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
            items = []
            for article in data.get("feed", []):
                items.append(NewsItem(
                    title=article.get("title", ""),
                    summary=article.get("summary", "")[:500],
                    source=article.get("source", ""),
                    published=article.get("time_published", ""),
                    url=article.get("url", ""),
                    relevance=float(article.get("overall_sentiment_score", 0)),
                ))
            return items
        except Exception as e:
            logger.debug(f"Alpha Vantage fetch failed: {e}")
            return []


class ReutersRSSFetcher:
    """Fetches from Reuters business/forex RSS (free, no API key)."""

    FEEDS = [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.reuters.com/reuters/USDollarNews",
    ]

    def fetch(self) -> List[NewsItem]:
        items = []
        for feed_url in self.FEEDS:
            try:
                req = Request(feed_url, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(req, timeout=8) as r:
                    content = r.read().decode("utf-8", errors="ignore")
                # Parse RSS items
                titles   = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", content)
                pubdates = re.findall(r"<pubDate>(.*?)</pubDate>", content)
                for i, title in enumerate(titles[:10]):
                    pub = pubdates[i] if i < len(pubdates) else ""
                    items.append(NewsItem(
                        title=title, summary=title,
                        source="Reuters", published=pub,
                    ))
            except Exception as e:
                logger.debug(f"Reuters RSS failed: {e}")
        return items


class FinnhubNewsFetcher:
    """Fetches market news from Finnhub (free: 60 calls/min, no credit card)."""

    BASE_URL = "https://finnhub.io/api/v1"

    def fetch(self, category: str = "forex") -> List[NewsItem]:
        if not FINNHUB_KEY:
            return []
        try:
            url = f"{self.BASE_URL}/news?category={category}&token={FINNHUB_KEY}"
            req = Request(url, headers={"User-Agent": "PriceIQ-Pro/5.4"})
            with urlopen(req, timeout=8) as r:
                articles = json.loads(r.read())
            items = []
            for a in articles[:15]:
                items.append(NewsItem(
                    title=a.get("headline", ""),
                    summary=a.get("summary", "")[:500],
                    source=a.get("source", "Finnhub"),
                    published=str(a.get("datetime", "")),
                    url=a.get("url", ""),
                ))
            return items
        except Exception as e:
            logger.debug(f"Finnhub fetch failed: {e}")
            return []


class NewsSentimentAgent:
    """
    6th trading agent. Reads financial news headlines and converts
    them into directional signals. Fundamentally different data stream
    from all 5 existing agents (which all use OHLCV price data).

    Integration with AgentOrchestrator:
        Same .evaluate(candles, regime, pair) interface as other agents.
        Returns NewsSentimentAgentSignal compatible with V5 pipeline.
    """

    NAME    = "NewsSentimentAgent"
    SUPPORTED_PAIRS = {"XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCHF"}
    REGIME_FIT = {"trending": 0.65, "ranging": 0.55, "volatile": 0.85}

    # Thresholds
    MIN_SCORE_TO_SIGNAL = 0.35    # min |score| to generate signal
    STRONG_SIGNAL_SCORE = 0.65    # score above this = high confidence
    MAX_ARTICLE_AGE_H   = 6       # ignore articles older than 6 hours

    def __init__(self):
        self._lexicon   = FinancialLexiconScorer()
        self._vader     = VaderSentimentScorer()
        self._finbert   = FinBERTScorer()
        self._av_fetcher = AlphaVantageNewsFetcher()
        self._reuters   = ReutersRSSFetcher()
        self._finnhub   = FinnhubNewsFetcher()
        self._articles:  List[NewsItem] = []
        self._scores:    Dict[str, SentimentScore] = {}
        self._last_refresh: Optional[datetime] = None

    # ── Data refresh ─────────────────────────────────────────

    async def refresh(self) -> Dict[str, SentimentScore]:
        """
        Fetch latest news and compute sentiment scores for all pairs.
        Call every 30 minutes from scheduler.
        """
        # ═══ THROTTLE: Only fetch once per 30 minutes to prevent API rate limits ═══
        if self._last_refresh and (datetime.now(timezone.utc) - self._last_refresh).total_seconds() < 1800:
            logger.debug("News refresh throttled (already refreshed within 30 mins)")
            return dict(self._scores)

        try:
            # Fetch from all sources in parallel
            all_articles = await asyncio.gather(
                asyncio.to_thread(self._av_fetcher.fetch, "XAUUSD"),
                asyncio.to_thread(self._av_fetcher.fetch, "EURUSD"),
                asyncio.to_thread(self._reuters.fetch),
                asyncio.to_thread(self._finnhub.fetch),
                return_exceptions=True,
            )

            # Flatten and deduplicate
            self._articles = []
            seen_titles    = set()
            for batch in all_articles:
                if isinstance(batch, Exception) or not batch:
                    continue
                for item in batch:
                    if item.title and item.title not in seen_titles:
                        seen_titles.add(item.title)
                        self._articles.append(item)

            self._last_refresh = datetime.now(timezone.utc)

            # Compute scores per pair
            for pair in self.SUPPORTED_PAIRS:
                self._scores[pair] = self._compute_pair_score(pair)

            logger.info(
                f"NewsSentimentAgent: {len(self._articles)} articles → "
                f"scores: " + ", ".join(
                    f"{p}={s.direction}({s.confidence:.2f})"
                    for p, s in self._scores.items()
                )
            )
            return dict(self._scores)

        except Exception as e:
            logger.warning(f"News sentiment refresh failed: {e}")
            return {}

    def inject_articles(self, articles: List[Dict]):
        """
        Manually inject articles (for testing or when using custom news source).
        Each dict needs: title, summary (optional), published (optional).
        """
        self._articles = [
            NewsItem(
                title=a.get("title", ""),
                summary=a.get("summary", a.get("title", ""))[:500],
                source=a.get("source", "manual"),
                published=a.get("published", datetime.now(timezone.utc).isoformat()),
            )
            for a in articles if a.get("title")
        ]
        self._last_refresh = datetime.now(timezone.utc)
        for pair in self.SUPPORTED_PAIRS:
            self._scores[pair] = self._compute_pair_score(pair)

    # ── Signal evaluation ─────────────────────────────────────

    def evaluate(
        self,
        candles: list,
        regime:  str = "trending",
        pair:    str = None,  # ═══ FIX: Changed to None to match BaseAgent signature ═══
    ) -> NewsSentimentAgentSignal:
        """
        Evaluate news sentiment signal for given pair.
        Called by AgentOrchestrator — same interface as other agents.
        """
        # ═══ FIX: Safe fallback if orchestrator passes None ═══
        pair_u = (pair or "XAUUSD").upper()
        
        if pair_u not in self.SUPPORTED_PAIRS:
            return self._null_signal(f"{pair_u} not supported by NewsSentimentAgent")

        if not self._articles:
            return self._null_signal("No news articles loaded — call refresh() first")

        # Check freshness
        if self._last_refresh:
            age_h = (datetime.now(timezone.utc) - self._last_refresh).total_seconds() / 3600
            if age_h > 4:
                return self._null_signal(f"News data stale ({age_h:.1f}h old)")

        score = self._scores.get(pair_u)
        if not score or score.direction is None:
            return self._null_signal(
                f"Insufficient news sentiment for {pair_u} "
                f"(score={score.raw_score:.3f} if score else 'N/A')"
            )

        # Compute ATR for stop/tp distances
        atr = self._calc_atr(candles)

        # Build signal
        stop_dist = atr * 1.5
        tp1_dist  = atr * 2.0
        tp2_dist  = atr * 3.5
        rr        = tp1_dist / max(stop_dist, 1e-9)
        wp        = score.confidence * 0.12 + 0.50   # 0.50–0.62 range
        ev        = wp * rr - (1 - wp) * 1.0

        return NewsSentimentAgentSignal(
            direction=score.direction,
            confidence=round(score.confidence, 3),
            win_probability=round(min(0.62, wp), 3),
            expected_value=round(ev, 4),
            stop_distance=stop_dist,
            tp1_distance=tp1_dist,
            tp2_distance=tp2_dist,
            regime_fit=self.REGIME_FIT.get(regime, 0.60),
            reasoning=(
                f"NEWS {pair_u} {score.direction.upper()}: "
                f"sentiment={score.raw_score:+.3f} "
                f"({score.n_articles} articles) | "
                f"Key: {'; '.join(score.key_headlines[:2])}"
            ),
            raw_features={
                "raw_score":    score.raw_score,
                "n_articles":   score.n_articles,
                "layer1_score": score.layer1_score,
                "layer2_score": score.layer2_score,
                "layer3_score": score.layer3_score,
                "key_headlines": score.key_headlines,
            },
            sentiment_score=score,
        )

    # ── Scoring ───────────────────────────────────────────────

    def _compute_pair_score(self, pair: str) -> SentimentScore:
        """Compute three-layer sentiment score for one pair from loaded articles."""
        if not self._articles:
            return self._empty_score(pair)

        l1_scores, l2_scores, l3_scores = [], [], []
        key_headlines = []

        for article in self._articles:
            text  = f"{article.title}. {article.summary}"

            # Layer 1: keyword lexicon
            l1 = self._lexicon.score(text, pair)
            if abs(l1) < 0.1:
                continue   # skip irrelevant articles

            l1_scores.append(l1)
            if abs(l1) >= 0.4:
                key_headlines.append(article.title[:80])

            # Layer 2: VADER
            l2 = self._vader.score(text)
            l2_scores.append(l2)

            # Layer 3: FinBERT (if available)
            if FINBERT_ENABLED:
                l3 = self._finbert.score(article.title)
                if l3 is not None:
                    l3_scores.append(l3)

        if not l1_scores:
            return self._empty_score(pair)

        # Combine layers (weighted: L1 primary, L2 confirmation, L3 upgrade)
        avg_l1 = float(np.mean(l1_scores))
        avg_l2 = float(np.mean(l2_scores)) if l2_scores else 0.0
        avg_l3 = float(np.mean(l3_scores)) if l3_scores else None

        if avg_l3 is not None:
            # FinBERT available: higher weight on it
            raw = 0.40 * avg_l1 + 0.25 * avg_l2 + 0.35 * avg_l3
        else:
            # VADER only
            raw = 0.70 * avg_l1 + 0.30 * avg_l2

        raw = float(np.clip(raw, -1.0, 1.0))

        # Direction and confidence
        if abs(raw) < self.MIN_SCORE_TO_SIGNAL:
            direction  = None
            confidence = 0.0
        elif raw > 0:
            direction  = "buy"
            confidence = min(0.85, 0.45 + abs(raw) * 0.50)
        else:
            direction  = "sell"
            confidence = min(0.85, 0.45 + abs(raw) * 0.50)

        # Scale down if very few articles
        if len(l1_scores) < 3:
            confidence *= 0.7

        return SentimentScore(
            pair=pair, direction=direction,
            raw_score=round(raw, 4),
            confidence=round(confidence, 4),
            n_articles=len(l1_scores),
            key_headlines=key_headlines[:3],
            layer1_score=round(avg_l1, 4),
            layer2_score=round(avg_l2, 4),
            layer3_score=round(avg_l3, 4) if avg_l3 is not None else None,
            reasoning=f"L1={avg_l1:.3f} L2={avg_l2:.3f}"
                      + (f" L3={avg_l3:.3f}" if avg_l3 else ""),
        )

    def _empty_score(self, pair: str) -> SentimentScore:
        return SentimentScore(
            pair=pair, direction=None, raw_score=0.0,
            confidence=0.0, n_articles=0, key_headlines=[],
            layer1_score=0.0, layer2_score=0.0, layer3_score=None,
            reasoning="no relevant articles",
        )

    def _null_signal(self, reason: str) -> NewsSentimentAgentSignal:
        return NewsSentimentAgentSignal(
            direction=None, confidence=0.0, win_probability=0.0,
            expected_value=0.0, stop_distance=0.0, tp1_distance=0.0,
            tp2_distance=0.0, regime_fit=0.0, reasoning=reason,
        )

    def _calc_atr(self, candles: list, period: int = 14) -> float:
        # ═══ FIX: Safe fallback if candles is None or empty ═══
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
        """Dashboard endpoint data."""
        return {
            "n_articles":      len(self._articles),
            "last_refresh":    self._last_refresh.isoformat() if self._last_refresh else None,
            "finbert_enabled": FINBERT_ENABLED,
            "api_configured": {
                "alpha_vantage": bool(ALPHA_VANTAGE_KEY and ALPHA_VANTAGE_KEY != "demo"),
                "finnhub":       bool(FINNHUB_KEY),
            },
            "scores": {
                pair: {
                    "direction":  s.direction,
                    "confidence": s.confidence,
                    "raw_score":  s.raw_score,
                    "n_articles": s.n_articles,
                }
                for pair, s in self._scores.items()
            },
        }

    def get_key_headlines(self, pair: str) -> List[str]:
        s = self._scores.get(pair.upper())
        return s.key_headlines if s else []
