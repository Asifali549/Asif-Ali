"""FUNDING PROBE - GitHub runner se kaun si exchange funding-rate history deti hai aur kitni purani (2026-10-06)."""
import time
import ccxt
import pandas as pd

OUT = "funding_probe_RESULTS.txt"
lines = []


def emit(x):
    print(x, flush=True)
    lines.append(x)


for ex_id in ["binanceusdm", "bybit", "okx", "bitget", "gateio", "kucoinfutures", "mexc", "htx"]:
    try:
        ex = getattr(ccxt, ex_id)({"enableRateLimit": True})
        sym = "BTC/USDT:USDT"
        has = ex.has.get("fetchFundingRateHistory")
        since = int(pd.Timestamp("2020-06-01").timestamp() * 1000)
        rows = ex.fetch_funding_rate_history(sym, since=since, limit=200)
        first = pd.to_datetime(rows[0]["timestamp"], unit="ms") if rows else None
        last = pd.to_datetime(rows[-1]["timestamp"], unit="ms") if rows else None
        # paging test
        n = len(rows)
        for _ in range(3):
            if not rows:
                break
            rows = ex.fetch_funding_rate_history(sym, since=rows[-1]["timestamp"] + 1, limit=200)
            n += len(rows)
            time.sleep(ex.rateLimit / 1000)
        mk = ex.load_markets()
        nswap = sum(1 for m in mk.values() if m.get("swap") and m.get("quote") == "USDT" and m.get("linear"))
        emit(f"{ex_id:>14}: OK has={has} | pehla {first} | 1st page aakhri {last} | 4 pages {n} rows | USDT swaps {nswap}")
    except Exception as e:
        emit(f"{ex_id:>14}: FAIL {type(e).__name__}: {str(e)[:150]}")

with open(OUT, "w") as f:
    f.write("\n".join(lines))
