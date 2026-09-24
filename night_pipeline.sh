#!/bin/bash
set -e
cd ~/priceiq-pro
LOG=night_log.txt; : > $LOG

echo "=== STEP 1: dedup patch-era corruption ==="
for f in $(git ls-files 'smc_bot/*.py'); do uniq "$f" > /tmp/u && mv /tmp/u "$f"; done
python3 -m compileall -q smc_bot && echo "COMPILE_OK" | tee -a $LOG
git add -A && git commit -m "Remove stacked duplicate lines (patch-era corruption)" | tee -a $LOG

echo "=== STEP 2: regression gate 1 (dedup must not change behavior) ==="
python3 -m smc_bot.backtest.runner data/V75_M5.csv --pair V75 --db /tmp/reg1.db --fresh \
    --start 2026-03-27 --end 2026-07-08 2>&1 | tee -a $LOG | grep -q "n=37 .* exp=+0.29R" \
  && echo "GATE1 PASS: dedup behavior-neutral" | tee -a $LOG \
  || { echo "GATE1 FAIL - STOPPED, see $LOG"; exit 1; }

echo "=== STEP 3: slippage top-up (1.25x spread) + cost echo ==="
python3 << 'PYEOF'
p = 'smc_bot/backtest/runner.py'
src = open(p).read()
old = 'entry = next_open + spread if direction == "BUY" else next_open - spread'
new = 'entry = next_open + 1.25 * spread if direction == "BUY" else next_open - 1.25 * spread'
assert src.count(old) == 1, f"anchor count={src.count(old)} (dedup incomplete?)"
src = src.replace(old, new, 1)
old2 = 'print(f"CONFIG: swing_length={config.params.swing_length} target={config.params.continuation_target}")'
new2 = old2 + '\n        print(f"COST: spread(full,at-entry)=2xhalf + slippage=25% -> model charges 1.25x spread")'
assert src.count(old2) == 1
src = src.replace(old2, new2, 1)
open(p, 'w').write(src)
print("slippage patch applied")
PYEOF
python3 -m compileall -q smc_bot && git add -A && \
git commit -m "Cost model: 1.25x spread at entry (round-trip + 25% slippage buffer)" | tee -a $LOG

echo "=== STEP 4: regression gate 2 (n must stay 37, exp drops by slippage) ==="
python3 -m smc_bot.backtest.runner data/V75_M5.csv --pair V75 --db /tmp/reg2.db --fresh \
    --start 2026-03-27 --end 2026-07-08 2>&1 | tee -a $LOG | grep -q "Double_BOS: n=37" \
  && echo "GATE2 PASS: trade set unchanged, costs now included" | tee -a $LOG \
  || { echo "GATE2 FAIL - STOPPED, see $LOG"; exit 1; }

echo "=== STEP 5: PATH A — XAUUSD train + OOS ==="
python3 -m smc_bot.backtest.runner data/XAUUSD_M5.csv --pair XAUUSD --db bt_a_xau_train.db --fresh \
    --start 2026-03-27 --end 2026-07-08 2>&1 | tee -a $LOG
python3 -m smc_bot.backtest.runner data/XAUUSD_M5.csv --pair XAUUSD --db bt_a_xau_oos.db --fresh \
    --start 2026-07-08 --end 2026-09-17 2>&1 | tee -a $LOG

git push origin main
echo "=== NIGHT COMPLETE — read $LOG in the morning ==="
