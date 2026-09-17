"""Split backtest data into in-sample (60%) and out-of-sample (40%) for validation."""
import sqlite3
import pandas as pd
from pathlib import Path

def split_and_validate(db_path: str, split_pct: float = 0.6):
    """Split journal into train/test and compare performance."""
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT * FROM trades 
        WHERE outcome IS NOT NULL AND outcome != 'SKIPPED'
        ORDER BY timestamp
    """, conn)
    conn.close()
    
    if len(df) < 40:
        print(f"❌ Only {len(df)} trades. Need ≥40 for meaningful split.")
        return
    
    split_idx = int(len(df) * split_pct)
    in_sample = df.iloc[:split_idx]
    out_sample = df.iloc[split_idx:]
    
    def calc_stats(subset, label):
        n = len(subset)
        if n == 0:
            return None
        
        total_r = subset['pnl_r'].sum()
        wins = (subset['outcome'] == 'WIN').sum()
        losses = (subset['outcome'] == 'LOSS').sum()
        timeouts = (subset['outcome'] == 'TIMEOUT').sum()
        
        expectancy = total_r / n if n > 0 else 0
        median_r = subset['pnl_r'].median()
        
        # Max drawdown (cumulative R)
        cumulative = subset['pnl_r'].cumsum()
        max_dd = (cumulative.cummax() - cumulative).max()
        
        # Check for outlier dependency (>50% of profit from single trade)
        max_single = subset['pnl_r'].max()
        outlier_dep = (max_single / total_r > 0.5) if total_r > 0 else False
        
        return {
            'label': label,
            'n': n,
            'total_r': total_r,
            'w': wins, 'l': losses, 't': timeouts,
            'expectancy': expectancy,
            'median_r': median_r,
            'max_dd': max_dd,
            'outlier_dep': outlier_dep
        }
    
    in_stats = calc_stats(in_sample, "In-Sample (60%)")
    out_stats = calc_stats(out_sample, "Out-of-Sample (40%)")
    
    print("\n" + "="*70)
    print(f"TIME-SPLIT VALIDATION: {Path(db_path).name}")
    print("="*70)
    
    for stats in [in_stats, out_stats]:
        if not stats:
            continue
        
        print(f"\n{stats['label']}:")
        print(f"  Trades:      {stats['n']}")
        print(f"  W/L/T:       {stats['w']}/{stats['l']}/{stats['t']}")
        print(f"  Total R:     {stats['total_r']:+.2f}R")
        print(f"  Expectancy:  {stats['expectancy']:+.3f}R/trade")
        print(f"  Median R:    {stats['median_r']:+.3f}R")
        print(f"  Max DD:      {stats['max_dd']:.2f}R")
        print(f"  Outlier Dep: {'⚠️  YES' if stats['outlier_dep'] else '✓ No'}")
    
    # Validation rules
    if in_stats and out_stats:
        print("\n" + "-"*70)
        print("VALIDATION CHECKS:")
        print("-"*70)
        
        checks = [
            ("In-sample ≥40 trades", in_stats['n'] >= 40),
            ("Out-of-sample positive R", out_stats['total_r'] > 0),
            ("No outlier dependency", not out_stats['outlier_dep']),
            ("OOS expectancy > 0", out_stats['expectancy'] > 0),
        ]
        
        passed = sum(1 for _, ok in checks if ok)
        
        for desc, ok in checks:
            print(f"  {'✓' if ok else '✗'} {desc}")
        
        print(f"\n  Result: {passed}/{len(checks)} checks passed")
        
        if passed == len(checks):
            print("  🎯 CONFIG VALIDATED")
        else:
            print("  ⚠️  CONFIG NEEDS REVIEW")

if __name__ == "__main__":
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else "bt_v75_ab.db"
    split_and_validate(db_path)
