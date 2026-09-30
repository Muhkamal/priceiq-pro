#!/bin/bash
# usage: ./run_once.sh <csv> <pair> <logfile> <db> <start> <end> [extra runner args...]
if pgrep -f "smc_bot.backtest.runner" >/dev/null; then
    echo "REFUSED: backtest already running (PID $(pgrep -f smc_bot.backtest.runner | tr '\n' ' '))"
    exit 1
fi
CSV="$1"; PAIR="$2"; LOG="$3"; DB="$4"; S="$5"; E="$6"; shift 6
nohup ~/.pyenv/versions/3.11.9/bin/python3 -m smc_bot.backtest.runner "$CSV" \
    --pair "$PAIR" --db "$DB" --fresh "$@" --start "$S" --end "$E" > "$LOG" 2>&1 &
echo "launched PID $! -> $LOG ($CSV / $PAIR)"
