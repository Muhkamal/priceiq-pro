#!/bin/bash
# usage: ./run_once.sh <logfile> <db> <start> <end> [extra runner args...]
if pgrep -f "smc_bot.backtest.runner" >/dev/null; then
    echo "REFUSED: backtest already running (PID $(pgrep -f smc_bot.backtest.runner | tr '\n' ' '))"
    exit 1
fi
LOG="$1"; DB="$2"; S="$3"; E="$4"; shift 4
nohup ~/.pyenv/versions/3.11.9/bin/python3 -m smc_bot.backtest.runner data/XAUUSD_M5_LONG.csv \
    --pair XAUUSD --db "$DB" --fresh "$@" --start "$S" --end "$E" > "$LOG" 2>&1 &
echo "launched PID $! -> $LOG"
