"""
PriceIQ Pro — Confidence Separation Audit
Answers the key question: do winners carry higher entry confidence than losers?
GET /api/v5/audit
"""
from fastapi import APIRouter
from app.services.v5_orchestrator_final import get_v5

router = APIRouter()

@router.get("/api/v5/audit")
async def confidence_audit():
    v5 = get_v5()
    if not v5:
        return {"error": "orchestrator not initialised"}

    entries = v5.journal.query()
    wins, losses, scratches = [], [], 0
    per_agent = {}

    for e in entries:
        r    = getattr(e, "r_multiple", 0.0) or 0.0
        conf = getattr(e, "confidence", 0.0) or 0.0
        agent = getattr(e, "agent", "?")

        a = per_agent.setdefault(agent, {"n": 0, "w": 0, "r_sum": 0.0, "conf": []})
        a["n"] += 1; a["r_sum"] += r; a["conf"].append(conf)

        if r >= 0.5:                      # decisive win
            a["w"] += 1; wins.append((conf, r))
        elif r <= -0.5:                   # decisive loss
            losses.append((conf, r))
        else:                             # scratch / timeout — not a signal-quality test
            scratches += 1

    n = len(entries)
    if n == 0:
        return {"n": 0, "verdict": "no trades yet"}

    all_r     = [r for _, r in wins] + [r for _, r in losses]
    avg_r     = (sum(all_r) / len(all_r)) if all_r else 0.0
    win_conf  = sum(c for c, _ in wins)   / len(wins)   if wins   else 0.0
    loss_conf = sum(c for c, _ in losses) / len(losses) if losses else 0.0
    separation = win_conf - loss_conf

    decisive = len(wins) + len(losses)
    if decisive < 15:
        verdict = "COLD START — keep running, no recalibration yet"
    elif separation >= 0.08 and win_conf >= 0.65:
        verdict = "YES: M.A.E. IS separating wheat/chaff → run to 100 trades, do NOT touch"
    elif separation <= 0.02:
        verdict = "NO: confidence NOT separating → recalibrate mae_confidence after 50+ trades"
    else:
        verdict = "WEAK separation → collect more data before deciding"

    return {
        "n_trades": n, "decisive": decisive, "scratches": scratches,
        "win_rate": round(len(wins) / decisive, 3) if decisive else 0,
        "avg_R_decisive": round(avg_r, 3),
        "avg_conf_winners": round(win_conf, 3),
        "avg_conf_losers": round(loss_conf, 3),
        "separation": round(separation, 3),
        "verdict": verdict,
        "per_agent": {
            k: {"n": v["n"], "wr": round(v["w"] / v["n"], 2),
                "avg_R": round(v["r_sum"] / v["n"], 3),
                "avg_conf": round(sum(v["conf"]) / len(v["conf"]), 3)}
            for k, v in per_agent.items()
        },
    }
