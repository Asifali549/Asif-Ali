"""DEPTH PROBE (2026-10-06) - order book (bid/ask) ka purana data kahan milta hai: Binance archive bookDepth / bookTicker + KuCoin live."""
import io
import zipfile

import requests

OUT = "depth_probe_RESULTS.txt"
L = []


def emit(x):
    print(x, flush=True)
    L.append(x)


def z(url):
    try:
        r = requests.get(url, timeout=60)
    except Exception as e:
        return f"FAIL {e}"[:150]
    if r.status_code != 200:
        return f"HTTP {r.status_code}"
    zz = zipfile.ZipFile(io.BytesIO(r.content))
    t = zz.read(zz.namelist()[0]).decode()
    lines = t.split("\n")
    return f"OK {len(r.content)//1024} KB, {len(lines)} lines | " + " || ".join(lines[:4])[:400] + f" || ... || {lines[-2][:120]}"


B = "https://data.binance.vision/data/futures/um/daily"
for lab, url in [
    ("bookDepth BTC 2023-01-05", f"{B}/bookDepth/BTCUSDT/BTCUSDT-bookDepth-2023-01-05.zip"),
    ("bookDepth BTC 2024-06-01", f"{B}/bookDepth/BTCUSDT/BTCUSDT-bookDepth-2024-06-01.zip"),
    ("bookDepth SOL 2026-09-20", f"{B}/bookDepth/SOLUSDT/SOLUSDT-bookDepth-2026-09-20.zip"),
    ("bookTicker BTC 2023-01-05", f"{B}/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2023-01-05.zip"),
    ("bookTicker BTC 2026-09-20", f"{B}/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2026-09-20.zip"),
]:
    emit(f"{lab:>28}: {z(url)}")
try:
    r = requests.get("https://api.kucoin.com/api/v1/market/orderbook/level2_100?symbol=BTC-USDT", timeout=30).json()
    d = r.get("data", {})
    emit(f"{'KuCoin live L2 BTC':>28}: bids {len(d.get('bids', []))} asks {len(d.get('asks', []))} best {d.get('bids', [[0]])[0]} / {d.get('asks', [[0]])[0]}")
except Exception as e:
    emit(f"KuCoin live: FAIL {e}")
with open(OUT, "w") as f:
    f.write("\n".join(L))
