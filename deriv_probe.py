"""DERIV PROBE (2026-10-06) - GitHub runner se funding / OI / volume delta ki purani history kahan se aur kitni milti hai."""
import io
import zipfile

import requests

OUT = "deriv_probe_RESULTS.txt"
L = []


def emit(x):
    print(x, flush=True)
    L.append(x)


def get(url, **kw):
    try:
        r = requests.get(url, timeout=30, **kw)
        return r
    except Exception as e:
        return e


def zip_head(url):
    r = get(url)
    if isinstance(r, Exception):
        return f"FAIL {type(r).__name__}: {r}"[:200]
    if r.status_code != 200:
        return f"HTTP {r.status_code}"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    txt = z.read(z.namelist()[0]).decode()[:300].replace("\n", " || ")
    return f"OK {len(r.content)} bytes | {txt}"


B = "https://data.binance.vision/data/futures/um"
emit("# BINANCE ARCHIVE (data.binance.vision)")
for lab, url in [
    ("funding 2021-01 BTC", f"{B}/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2021-01.zip"),
    ("funding 2026-08 SOL", f"{B}/monthly/fundingRate/SOLUSDT/SOLUSDT-fundingRate-2026-08.zip"),
    ("fut klines 1d 2021-01 BTC", f"{B}/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2021-01.zip"),
    ("fut klines 1d 2026-08 DOGE", f"{B}/monthly/klines/DOGEUSDT/1d/DOGEUSDT-1d-2026-08.zip"),
    ("metrics 2022-01-01 BTC", f"{B}/daily/metrics/BTCUSDT/BTCUSDT-metrics-2022-01-01.zip"),
    ("metrics 2026-09-01 ETH", f"{B}/daily/metrics/ETHUSDT/ETHUSDT-metrics-2026-09-01.zip"),
    ("spot klines 1d 2021-01 ETH", "https://data.binance.vision/data/spot/monthly/klines/ETHUSDT/1d/ETHUSDT-1d-2021-01.zip"),
]:
    emit(f"{lab:>28}: {zip_head(url)}")

emit("\n# OKX (public API)")
for lab, url in [
    ("funding hist BTC", "https://www.okx.com/api/v5/public/funding-rate-history?instId=BTC-USDT-SWAP&limit=100"),
    ("OI hist 1D BTC", "https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-history?instId=BTC-USDT-SWAP&period=1D&limit=100"),
    ("taker vol 1D BTC", "https://www.okx.com/api/v5/rubik/stat/taker-volume-contract?instId=BTC-USDT-SWAP&period=1D&limit=100"),
    ("OI+vol ccy 1D BTC", "https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume?ccy=BTC&period=1D"),
]:
    r = get(url)
    if isinstance(r, Exception):
        emit(f"{lab:>28}: FAIL {r}"[:200])
        continue
    try:
        j = r.json()
        d = j.get("data", [])
        emit(f"{lab:>28}: HTTP {r.status_code} code {j.get('code')} rows {len(d)} | pehla {str(d[-1])[:120] if d else '-'} | aakhri {str(d[0])[:120] if d else '-'}")
    except Exception as e:
        emit(f"{lab:>28}: HTTP {r.status_code} {r.text[:150]}")

with open(OUT, "w") as f:
    f.write("\n".join(L))
