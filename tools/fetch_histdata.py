"""Fetch XAUUSD M1 from HistData (free) -> data/histdata_raw/*.csv
Falls back gracefully: any month that fails is logged for manual download.
Usage: python3 tools/fetch_histdata.py 2023-01 2026-08
"""
import sys, io, time, zipfile, pathlib
import requests

PAIR = "XAUUSD"
OUT = pathlib.Path("data/histdata_raw"); OUT.mkdir(parents=True, exist_ok=True)

def months(a, b):
    y, m = map(int, a.split("-")); ey, em = map(int, b.split("-"))
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m == 13: y, m = y + 1, 1

failed = []
for y, m in months(sys.argv[1], sys.argv[2]):
    tag = f"{y}{m:02d}"
    out = OUT / f"DAT_ASCII_{PAIR}_M1_{tag}.csv"
    if out.exists():
        print(f"{tag}: have"); continue
    url = f"https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{PAIR}/{y}/{m}"
    try:
        r = requests.post(url, data={"tk": "1", "date": str(y), "datemonth": tag,
                                     "platform": "ASCII", "timeframe": "M1", "fxpair": PAIR},
                          timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        z = zipfile.ZipFile(io.BytesIO(r.content))
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        out.write_bytes(z.read(name))
        print(f"{tag}: OK ({out.stat().st_size//1024} KB)")
    except Exception as e:
        failed.append(tag); print(f"{tag}: FAILED ({type(e).__name__})")
    time.sleep(2)  # be polite

print("\nFAILED months (download manually from histdata.com if needed):", failed)
