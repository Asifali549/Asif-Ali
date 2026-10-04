"""
SCHEDULER - GitHub ke dheele cron ka mutabadil (2026-10-02)
==========================================================
Masla: 30 Sep se GitHub ke scheduled (cron) runs ghanton der se ya bilkul nahi chal rahe the
(Watchdog din mein 48 ki jagah ~4 baar, Ichimoku 6 mein se ~2).
Hal: ye workflow lagataar ~5 ghante 40 minute chalta hai, har minute ghari dekhta hai aur theek waqt par
bots ko GitHub API (workflow_dispatch, GH_TOKEN) se chalata hai - dispatch foran chalta hai, cron ki tarah
der nahi hoti. Waqt khatam hone se pehle ye KHUD ko dobara chala deta hai (zanjeer). Agar zanjeer kabhi
toote to Watchdog (jo yahi scheduler har 30 min chalata hai, aur apna cron bhi rakhta hai) isay dobara chala deta hai.
Purane cron bhi backup ke taur par maujood hain - bots ek hi candle do baar process nahi karte (last_bar / last_day).
Waqt (UTC): Ichimoku har 4 ghante :10 | Ichimoku TP5 (paper) :12 | Donchian 00:15 | Dip 00:20 | Capitulation 00:25 | Watchdog har :05 aur :35
"""
import os
import time
from datetime import datetime, timedelta, timezone

import requests

REPO = os.environ.get("GITHUB_REPOSITORY", "Asifali549/Asif-Ali")
TOKEN = os.environ.get("GH_TOKEN")
RUN_SECONDS = int(os.environ.get("RUN_SECONDS", 5 * 3600 + 40 * 60))
CATCHUP_MIN = 3


def due(t):
    """Is minute (UTC) mein kaun se workflows chalne chahiye."""
    out = []
    if t.minute == 10 and t.hour % 4 == 0:
        out.append("ichimoku4h_bot.yml")
    if t.minute == 12 and t.hour % 4 == 0:
        out.append("ichi_tp5_bot.yml")
    if t.hour == 0 and t.minute == 15:
        out.append("donchian_daily_bot.yml")
    if t.hour == 0 and t.minute == 20:
        out.append("dip_daily_bot.yml")
    if t.hour == 0 and t.minute == 25:
        out.append("capit_daily_bot.yml")
    if t.minute in (5, 35):
        out.append("watchdog.yml")
    return out


def dispatch(wf):
    if not TOKEN:
        print(f"[NO TOKEN] {wf}")
        return False
    url = f"https://api.github.com/repos/{REPO}/actions/workflows/{wf}/dispatches"
    for k in range(3):
        try:
            r = requests.post(url, json={"ref": "main"}, timeout=20,
                              headers={"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github+json"})
            if r.status_code in (201, 204):
                return True
            print(f"[{wf}] HTTP {r.status_code}: {r.text[:150]}")
        except Exception as e:
            print(f"[{wf}] {e}")
        time.sleep(10 * (k + 1))
    return False


def main():
    t0 = time.time()
    fired = set()
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    # shuru mein: pichle chand minute ke chhoote waqt (zanjeer ke beech ka faasla)
    for m in range(CATCHUP_MIN, 0, -1):
        t = now - timedelta(minutes=m)
        for wf in due(t):
            fired.add((t.isoformat(), wf))
            print(f"{datetime.now(timezone.utc):%H:%M:%S} catch-up {wf} ({t:%H:%M}) -> {dispatch(wf)}")
    while time.time() - t0 < RUN_SECONDS:
        t = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        for wf in due(t):
            key = (t.isoformat(), wf)
            if key not in fired:
                fired.add(key)
                print(f"{datetime.now(timezone.utc):%H:%M:%S} {wf} -> {dispatch(wf)}")
        time.sleep(60 - datetime.now(timezone.utc).second + 1)
    ok = dispatch("scheduler.yml")
    print(f"Waqt poora - agla scheduler chala diya: {ok}")


if __name__ == "__main__":
    main()
