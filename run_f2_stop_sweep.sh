#!/bin/bash
set -e

PAIR="V75"
DATA_M5="data/V75_M5.csv"
TRAIN_START="2026-03-27"
TRAIN_END="2026-07-09"

echo "=========================================="
echo "Experiment F2: Stop Mode Sweep (Target=dol)"
echo "=========================================="

for MODE in consolidation retest_candle; do
    echo ""
    echo ">>> Testing stop_mode=$MODE..."
    
    if grep -q "stop_mode:" smc_bot/system.yaml; then
        sed -i "s/stop_mode: .*/stop_mode: $MODE/" smc_bot/system.yaml
    else
        sed -i "/^params:/a\  stop_mode: $MODE" smc_bot/system.yaml
    fi
    
    python3 -m smc_bot.backtest.runner "$DATA_M5" \
        --pair "$PAIR" \
        --db "bt_f2_${MODE}.db" \
        --start "$TRAIN_START" \
        --end "$TRAIN_END"
done

echo ""
echo "=========================================="
echo "Stop Mode Results"
echo "=========================================="
python3 << 'PYEOF'
import sqlite3

print(f"{'stop_mode':<18} {'n':<6} {'W/L/T':<14} {'WR':<8} {'avg_win':<10} {'avg_loss':<10} {'total_R':<10}")
print("-" * 80)

for mode in ['consolidation', 'retest_candle']:
    db = f"bt_f2_{mode}.db"
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        
        # Overall stats
        cur.execute("""
            SELECT COUNT(*), 
                   SUM(outcome='WIN'), SUM(outcome='LOSS'), SUM(outcome='TIMEOUT'),
                   COALESCE(SUM(pnl_r),0)
            FROM trades WHERE outcome IS NOT NULL AND outcome != 'SKIPPED'
        """)
        n, w, l, t, total = cur.fetchone()
        
        if n and n > 0:
            wr = (w or 0) * 100.0 / n
            
            # Avg win / avg loss
            cur.execute("SELECT AVG(pnl_r) FROM trades WHERE outcome='WIN'")
            avg_win = cur.fetchone()[0] or 0
            
            cur.execute("SELECT AVG(pnl_r) FROM trades WHERE outcome='LOSS'")
            avg_loss = cur.fetchone()[0] or 0
            
            wlt = f"{w or 0}/{l or 0}/{t or 0}"
            print(f"{mode:<18} {n:<6} {wlt:<14} {wr:>6.1f}% {avg_win:>+9.2f} {avg_loss:>+9.2f} {total:>+9.2f}")
        else:
            print(f"{mode:<18} {'0':<6} {'-':<14} {'-':<8} {'-':<10} {'-':<10} {'-':<10}")
        conn.close()
    except Exception as e:
        print(f"{mode:<18} ERROR: {str(e)[:60]}")
PYEOF
