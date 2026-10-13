"""compare_signal.py - plumbing cross-check for FORWARD-TSMOM-001.

Run ONLY after a fresh download succeeds:
    python3 download_prices.py && python3 compare_signal.py

Layer 1 (the gate): recomputes the signal on data TRUNCATED to the Window-1
date (2026-10-12) and compares to the journaled book (last_signal.csv).
Same inputs -> same outputs, always. Any DIFF = plumbing failure: STOP.
Layer 2 (informational): the live book on the LATEST data. Directions may
legitimately change here (the market moved since Oct-12); weak signals
(small |mom_z|) flip first. This is not an error - it's the book updating.
"""
import sys
import pandas as pd
from monthly_signal import compute_signal   # reuses the frozen logic

W1_DATE = "2026-10-12"

ref = pd.read_csv("last_signal.csv").set_index("instrument")["direction"]
cur = pd.read_csv("prices.csv", parse_dates=["date"]).set_index("date").sort_index()

def book(df):
    return {c: compute_signal(df[c].dropna()).get("direction") for c in df.columns}

# --- Layer 1: determinism / plumbing gate ---
w1_now = book(cur.loc[:W1_DATE])
bad = [c for c in ref.index if (w1_now.get(c) or "None") != (str(ref[c]) if pd.notna(ref[c]) else "None")]
print(f"== PLUMBING: recomputed through {W1_DATE} vs journaled Window-1 book ==")
for c in ref.index:
    j = ref[c] if pd.notna(ref[c]) else "None"
    r = w1_now.get(c) or "None"
    print(f"  {c:10} journal={str(j):6} recomputed={str(r):6} {'MATCH' if c not in bad else 'DIFF <<<'}")
if bad:
    print(f"\nPLUMBING FLAG: {len(bad)} mismatch(es). Do NOT proceed - paste this output.")
    sys.exit(1)
print("\nPLUMBING PASS: identical inputs produce identical directions.")

# --- Layer 2: live book (informational) ---
live = book(cur)
print(f"\n== LIVE BOOK (data through {cur.index[-1].date()}) ==")
for c in cur.columns:
    tag = "" if (live[c] or "None") == (str(ref[c]) if pd.notna(ref[c]) else "None") else "   <- changed since Window 1"
    print(f"  {c:10} {str(live[c] or 'flat'):6}{tag}")
