#!/bin/bash
set -e

PAIR="V75"
DATA_M5="data/V75_M5.csv"
TRAIN_START="2026-03-27"
TRAIN_END="2026-07-09"
OOS_START="2026-07-10"

echo "=========================================="
echo "PHASE 1: Swing-Length Sweep"
echo "Train Window: $TRAIN_START to $TRAIN_END"
echo "=========================================="

for SWING_LEN in 5 10 15; do
    echo ""
    echo ">>> Testing swing_length=$SWING_LEN..."
    sed -i "s/swing_length: [0-9]*/swing_length: $SWING_LEN/" smc_bot/system.yaml
    
    python3 -m smc_bot.backtest.runner "$DATA_M5" \
        --pair "$PAIR" \
        --db "bt_swing_${SWING_LEN}.db" \
        --start "$TRAIN_START" \
        --end "$TRAIN_END"
done

echo ""
echo "=========================================="
echo "Swing-Length Results (Train Window)"
echo "=========================================="
python3 << 'PYEOF'
import sqlite3
print(f"{'swing_len':<10} {'n':<6} {'total_R':<12} {'expectancy':<12} {'win_rate':<10}")
print("-" * 55)
for sl in [5, 10, 15]:
    conn = sqlite3.connect(f"bt_swing_{sl}.db")
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*), 
               COALESCE(SUM(pnl_r),0), 
               AVG(pnl_r),
               SUM(outcome='WIN') * 100.0 / NULLIF(SUM(outcome IN ('WIN','LOSS','TIMEOUT')),0)
        FROM trades WHERE outcome IS NOT NULL AND outcome != 'SKIPPED'
    """)
    n, total, avg, wr = cur.fetchone()
    conn.close()
    print(f"{sl:<10} {n:<6} {total:+12.2f} {avg:+12.3f} {wr or 0:>9.1f}%")
PYEOF

echo ""
echo "=========================================="
echo ">>> STOP HERE"
echo "=========================================="
echo "Review the swing-length results above."
echo "Update smc_bot/system.yaml with the winning swing_length."
echo "Then run: bash run_v75_experiment.sh phase2"
