"""Run from repo root: python smc_bot/tools/eyeball_pass.py data/XAUUSD_M15.csv --pair XAUUSD --windows 50"""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from smartmoneyconcepts import smc

def load_csv(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    return df.sort_index()

def plot_window(df_win, swings, bc, fvg, ob, out_path, title):
    fig, ax = plt.subplots(figsize=(16, 9))
    n = len(df_win)
    for i, (t, row) in enumerate(df_win.iterrows()):
        color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        ax.plot([i, i], [row["low"], row["high"]], color=color, lw=0.8, zorder=2)
        ax.add_patch(Rectangle((i - 0.3, min(row["open"], row["close"])), 0.6,
                               abs(row["close"] - row["open"]) or 1e-5,
                               facecolor=color, edgecolor=color, zorder=3))
    for i in range(len(fvg)):
        r = fvg.iloc[i]
        if np.isnan(r["Top"]):
            continue
        end = r["MitigatedIndex"] if not np.isnan(r["MitigatedIndex"]) else i + 20
        alpha = 0.25 if np.isnan(r["MitigatedIndex"]) else 0.08
        ax.add_patch(Rectangle((i, r["Bottom"]), end - i, r["Top"] - r["Bottom"],
                               facecolor="orange", alpha=alpha, zorder=1))
    for i in range(len(ob)):
        r = ob.iloc[i]
        if np.isnan(r["Top"]):
            continue
        end = r["MitigatedIndex"] if not np.isnan(r["MitigatedIndex"]) else i + 20
        alpha = 0.30 if np.isnan(r["MitigatedIndex"]) else 0.10
        color = "#26a69a" if r["OB"] == 1 else "#ef5350"
        ax.add_patch(Rectangle((i, r["Bottom"]), end - i, r["Top"] - r["Bottom"],
                               facecolor=color, alpha=alpha, zorder=1))
    sh = swings["Highs"].values; sl = swings["Lows"].values; lv = swings["Level"].values
    ax.scatter(np.where(sh == 1)[0], lv[sh == 1], marker="v", s=80, color="red", zorder=4, label="Swing High")
    ax.scatter(np.where(sl == 1)[0], lv[sl == 1], marker="^", s=80, color="blue", zorder=4, label="Swing Low")
    bos = bc["BOS"].values; ch = bc["CHOCH"].values; bl = bc["Level"].values
    for idx in np.where(bos != 0)[0]:
        ax.annotate("BOS", (idx, bl[idx]), textcoords="offset points", xytext=(0, 12),
                    color="green" if bos[idx] == 1 else "magenta",
                    fontsize=8, fontweight="bold", ha="center", zorder=5)
    for idx in np.where(ch != 0)[0]:
        ax.annotate("CHoCH", (idx, bl[idx]), textcoords="offset points", xytext=(0, -16),
                    color="darkorange", fontsize=8, fontweight="bold", ha="center", zorder=5)
    highs = lv[sh == 1]; lows = lv[sl == 1]
    if len(highs) and len(lows):
        leg_hi, leg_lo = highs[-1], lows[-1]
        if leg_hi <= leg_lo:
            leg_hi, leg_lo = leg_lo, leg_hi
        eq = leg_lo + (leg_hi - leg_lo) * 0.5
        ax.axhspan(eq, leg_hi, color="red", alpha=0.05)
        ax.axhspan(leg_lo, eq, color="blue", alpha=0.05)
        ax.axhline(eq, color="gray", ls="--", lw=1)
        ax.text(n - 1, eq, f" EQ {eq:.2f}", va="bottom", fontsize=8, color="gray")
        ax.text(n - 1, leg_hi, f" SH {leg_hi:.2f}", va="bottom", fontsize=8, color="red")
        ax.text(n - 1, leg_lo, f" SL {leg_lo:.2f}", va="top", fontsize=8, color="blue")
    ax.set_title(title)
    ax.set_xlim(-1, n + 8)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--pair", default="PAIR")
    p.add_argument("--windows", type=int, default=50)
    p.add_argument("--bars", type=int, default=96)
    p.add_argument("--step", type=int, default=48)
    p.add_argument("--out", default="eyeball_pass")
    p.add_argument("--swing-length", type=int, default=10)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = load_csv(args.csv)
    log_path = os.path.join(args.out, "review_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("window_id,png,last_bias,swings_agree,bos_agree,zones_agree,notes\n")
    swings = smc.swing_highs_lows(df, swing_length=args.swing_length)
    bc = smc.bos_choch(df, swing_highs_lows=swings, close_break=True)
    fvg = smc.fvg(df)
    ob = smc.ob(df, swing_highs_lows=swings, close_mitigation=False)
    starts = np.arange(0, len(df) - args.bars, args.step)
    if len(starts) > args.windows:
        starts = np.linspace(0, len(df) - args.bars, args.windows).astype(int)
    for w, s in enumerate(starts):
        e = s + args.bars
        win = slice(s, e)
        last_bos = bc["BOS"].iloc[s:e].dropna()
        bias = "BULL" if (not last_bos.empty and last_bos.iloc[-1] == 1) else "BEAR"
        png = os.path.join(args.out, f"win_{w:03d}_{df.index[s]:%Y%m%d_%H%M}.png")
        plot_window(df.iloc[win], swings.iloc[win], bc.iloc[win],
                    fvg.iloc[win], ob.iloc[win], png,
                    f"{args.pair} {df.index[s]:%Y-%m-%d %H:%M}  bias(last BOS)={bias}")
        with open(log_path, "a") as f:
            f.write(f"w{w:03d},{os.path.basename(png)},{bias},,,,\n")
        print(f"[{w+1}/{len(starts)}] {png}  last-BOS-bias={bias}")
    print(f"\nDone. Open {args.out}/ and fill in review_log.csv.")
    print("Pass: >=90% agreement on swings and bias.")

if __name__ == "__main__":
    main()
