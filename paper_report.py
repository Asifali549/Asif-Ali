"""
HAFTA-WAR PAPER REPORT (2026-10-06, user ne mana)
=================================================
Har system ki paper (bot) ki band trades ko backtest se milata hai: kitni trades, 10 mein se kitni jeetein (backtest kitni),
PF (har 1 rupay nuqsan par nafa), aur ek binomial jaanch: itni trades mein itni kam jeet sirf badqismati se aa sakti hai ya
koi gadbad (bot ya backtest mein) lagti hai. Backtest number dashboard ke SYSTEMS se (ek hi jagah).
Natija: paper_report_RESULTS.txt + Telegram (Urdu, Urdu hindson ke sath taake mobile par alfaaz aage peeche na hon).
Workflow: paper_report.yml (har peer 03:00 UTC = 8 AM PKT, scheduler bhi chalata hai).
"""
import ast
import json
import math
import os
from datetime import datetime, timezone

import pandas as pd

DASH = "live_colorful_dashboard.py"
OUT = "paper_report_RESULTS.txt"
WIN_MIN = 0.5            # % - backtest jaisa: +0.5% se ziada = jeet
URDU = {"Ichimoku 4H": "پرانا اچیموکو", "Ichimoku TP5": "اچیموکو ٹی پی فائیو", "Donchian Daily": "ڈونچین",
        "Dip Daily": "ڈپ", "Volume Capitulation": "کیپیچولیشن", "Dip+": "ڈپ پلس", "W52": "سال کی چوٹی",
        "W52 Trail": "سال کی چوٹی، چلتا ایس ایل", "Streak": "چار دن گراوٹ", "Flush": "مارکیٹ صفائی"}
DIG = str.maketrans("0123456789.", "۰۱۲۳۴۵۶۷۸۹٫")


def ur(x):
    return str(x).translate(DIG)


def systems():
    tree = ast.parse(open(DASH, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "SYSTEMS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("SYSTEMS nahi mila")


def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def main():
    S = systems()
    rows, tg = [], ["📋 ہفتہ وار پیپر رپورٹ (پیپر بمقابلہ ٹیسٹ)", ""]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows.append(f"PAPER vs BACKTEST - {now}")
    rows.append("=" * 110)
    for name, cfg in S.items():
        bt = cfg.get("backtest", {})
        pw = bt.get("win")
        path = cfg.get("trades")
        n_open = 0
        try:
            st = json.load(open(cfg["state"]))
            n_open = len(st.get("positions", {}))
        except Exception:
            pass
        if not path or not os.path.exists(path):
            r = pd.Series(dtype=float)
        else:
            df = pd.read_csv(path)
            r = pd.to_numeric(df.get("ret_pct", pd.Series(dtype=float)), errors="coerce").dropna()
        n = len(r)
        k = int((r > WIN_MIN).sum())
        g, lo = r[r > 0].sum(), -r[r <= 0].sum()
        pf = g / lo if lo > 0 else (float("inf") if g > 0 else 0.0)
        avg = r.mean() if n else 0.0
        if n < 10:
            verdict, vu = f"kam trades ({n}/10)", f"ابھی کم ٹریڈیں، کم از کم دس چاہئیں"
        elif pw is not None and binom_cdf(k, n, pw / 100) <= 0.05:
            verdict, vu = "GADBAD? jeet backtest se kafi kam (5% se kam imkaan)", "⚠️ جیت ٹیسٹ سے کافی کم، جانچ ضروری"
        else:
            verdict, vu = "theek (backtest ke daire mein)", "✅ ٹیسٹ کے دائرے میں"
        pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
        rows.append(f"{name:<22} band {n:>3} | khuli {n_open:>2} | jeet/10 {k / n * 10 if n else 0:4.1f} (test {pw / 10 if pw else 0:.1f}) | "
                    f"PF {pf_s:>5} (test {bt.get('pf', 0):.2f}) | ausat {avg:+.2f}% | {verdict}")
        win10 = f"{k / n * 10:.1f}" if n else "0"
        pf_u = "بے حد" if pf == float("inf") else ur(f"{pf:.2f}")
        tg.append(f"• {URDU.get(name, name)}: بند ٹریڈیں {ur(n)}، کھلی {ur(n_open)}")
        if n:
            tw = ur(f"{pw / 10:.1f}") if pw else "؟"
            tp = ur(f"{bt.get('pf', 0):.2f}")
            tg.append(f"   دس میں سے جیت {ur(win10)} (ٹیسٹ میں {tw})، پی ایف {pf_u} (ٹیسٹ میں {tp})")
        tg.append(f"   {vu}")
    tg += ["", "پی ایف یعنی ہر ایک روپے نقصان پر کتنا نفع۔ دس سے کم ٹریڈوں پر فیصلہ نہیں ہوتا۔"]
    text = "\n".join(rows)
    print(text)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    if os.environ.get("SEND_TG", "1") == "1":
        try:
            from telegram_alert import send_telegram_alert
            send_telegram_alert("\n".join(tg))
        except Exception as e:
            print("Telegram nakam:", e)
    else:
        print("\n".join(tg))


if __name__ == "__main__":
    main()
