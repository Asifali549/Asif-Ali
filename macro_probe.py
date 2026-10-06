"""MACRO PROBE (2026-10-06) - market chalane wali cheezon ka purana data GitHub se milta hai?"""
import requests

L = []


def emit(x):
    print(x, flush=True)
    L.append(x)


def j(url):
    try:
        r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        return r
    except Exception as e:
        return e


for lab, url in [
    ("DefiLlama stablecoins total", "https://stablecoins.llama.fi/stablecoincharts/all"),
    ("Fear&Greed alternative.me", "https://api.alternative.me/fng/?limit=0"),
    ("FRED DXY (DTWEXBGS)", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS"),
    ("FRED 10y (DGS10)", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"),
    ("FRED Nasdaq (NASDAQCOM)", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=NASDAQCOM"),
    ("FRED Fed balance (WALCL)", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=WALCL"),
    ("FRED RRP (RRPONTSYD)", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=RRPONTSYD"),
]:
    r = j(url)
    if isinstance(r, Exception):
        emit(f"{lab:>30}: FAIL {r}"[:200])
        continue
    t = r.text
    emit(f"{lab:>30}: HTTP {r.status_code} | {len(t)//1024} KB | shuru: {t[:160]!r} | aakhir: {t[-120:]!r}")
with open("macro_probe_RESULTS.txt", "w") as f:
    f.write("\n".join(L))
