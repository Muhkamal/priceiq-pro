"""PriceIQ Pro — RAG-lite chatbot. Answers from YOUR bot's live data (Groq optional)."""
import json, logging, os
logger = logging.getLogger(__name__)
_GROQ = None
try:
    from groq import Groq
    if os.environ.get("GROQ_API_KEY"): _GROQ = Groq(api_key=os.environ["GROQ_API_KEY"])
except Exception: pass

def _facts(v5):
    f = []
    try:
        s = v5.get_system_status(); p = s.get("portfolio", {})
        f.append(f"Current balance is ${p.get('current_balance', 0):.2f} with {p.get('open_positions', 0)} open positions.")
        for reg, ag in (s.get("best_per_regime") or {}).items():
            f.append(f"Best agent in {reg} regime is {ag}.")
        f.append(f"Regime model trained: {s.get('regime_model_trained')}.")
    except Exception: pass
    try:
        from app.services.core.performance_analytics import perf_analytics
        m = perf_analytics.metrics()
        f.append(f"Performance: sharpe {m.get('sharpe')}, sortino {m.get('sortino')}, profit factor {m.get('profit_factor')}, win rate {m.get('win_rate')}, max drawdown {m.get('max_dd_pct')}%.")
    except Exception: pass
    try:
        from app.services.core.mae_mfe_analyzer import mae_tracker
        f += mae_tracker.insights()
    except Exception: pass
    try:
        from app.services.core.fundamentals_gate import fundamentals_gate
        f.append(fundamentals_gate.status())
    except Exception: pass
    return f

def _retrieve(q, facts, k=6):
    qw = set(q.lower().split())
    return [f for _, f in sorted(((len(qw & set(f.lower().split())), f) for f in facts), key=lambda x: -x[0])[:k]]

async def ask_ai(question: str, v5) -> str:
    ctx = _retrieve(question, _facts(v5))
    if _GROQ:
        try:
            r = _GROQ.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[{"role": "system", "content": "You are PriceIQ Pro's analyst. Answer ONLY from provided bot data."},
                          {"role": "user", "content": "BOT DATA:\n" + "\n".join(ctx) + f"\n\nQUESTION: {question}"}],
                max_tokens=500)
            return r.choices[0].message.content
        except Exception as e: logger.warning(f"Groq: {e}")
    return "🤖 Relevant bot data:\n• " + "\n• ".join(ctx[:4])
