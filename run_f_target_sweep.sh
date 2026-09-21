#!/bin/bash
set -e

PAIR="V75"
DATA_M5="data/V75_M5.csv"
TRAIN_START="2026-03-27"
TRAIN_END="2026-07-09"

echo "=========================================="
echo "Experiment F: Target Mode Sweep"
echo "=========================================="

for MODE in dol measured_move next_swing; do
    echo ""
    echo ">>> Testing continuation_target=$MODE..."
    
    # Update system.yaml
    if grep -q "continuation_target:" smc_bot/system.yaml; then
        sed -i "s/continuation_target: .*/continuation_target: $MODE/" smc_bot/system.yaml
    else
        # Add to params section
        sed -i "/^params:/a\  continuation_target: $MODE" smc_bot/system.yaml
    fi
    
    python3 -m smc_bot.backtest.runner "$DATA_M5" \
        --pair "$PAIR" \
        --db "bt_f_${MODE}.db" \
        --start "$TRAIN_START" \
        --end "$TRAIN_END"
done

echo ""
echo "=========================================="
echo "Target Mode Results"
echo "=========================================="
python3 << 'PYEOF'
import sqlite3

print(f"{'mode':<15} {'n':<6} {'W/L/T':<12} {'WR':<8} {'exp':<10} {'total_R':<10}")
print("-" * 65)

for mode in ['dol', 'measured_move', 'next_swing']:
    db = f"bt_f_{mode}.db"
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT COUNT(*), 
                   SUM(outcome='WIN'), SUM(outcome='LOSS'), SUM(outcome='TIMEOUT'),
                   COALESCE(SUM(pnl_r),0), AVG(pnl_r)
            FROM trades WHERE outcome IS NOT NULL AND outcome != 'SKIPPED'
        """)
        n, w, l, t, total, avg = cur.fetchone()
        conn.close()
        
        if n and n > 0:
            wr = (w or 0) * 100.0 / n
            wlt = f"{w or 0}/{l or 0}/{t or 0}"
            print(f"{mode:<15} {n:<6} {wlt:<12} {wr:>6.1f}% {avg:>+9.3f} {total:>+9.2f}")
        else:
            print(f"{mode:<15} {'0':<6} {'-':<12} {'-':<8} {'-':<10} {'-':<10}")
    except Exception as e:
        print(f"{mode:<15} ERROR: {str(e)[:50]}")
PYEOF
