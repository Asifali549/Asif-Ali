"""
Watchdog - GitHub Actions ke zariye har 30 minute khud chalta hai.
Dono live bots (Ichimoku 4H Bot, Donchian Daily Bot) ki taaza-tareen state file
dekhta hai - agar koi loop expected waqt se zyada purana ho gaya ho
(yani beech mein kahin ruk gaya), to us workflow ko khud
"workflow_dispatch" se dobara chalata hai aur Telegram par khabar
deta hai (kamyabi aur nakami, dono soorat mein).

Kisi bhi loop ke andar ke asal scan/trade logic se bilkul chupa hua
hai - sirf "aakhri baar kab update hua" dekhta hai aur zaroorat par
usi workflow ko GitHub API se dobara trigger karta hai (bilkul waisi
hi request jaisi scheduled_dashboard_scan.py khud apne agle scan ke
liye bhejta hai).
"""

import json
import os
from datetime import datetime, timezone

import requests

from telegram_alert import send_telegram_alert

GITHUB_REPO = os.environ.get("GITHUB_REPOSITORY", "Asifali549/Asif-Ali")
GH_TOKEN = os.environ.get("GH_TOKEN")

# (state file, us file ke andar timestamp wali key, jis workflow ko
#  dobara chalana hai, kitne minute ke baad "stale" mana jaye, naam)
#
# Dono bots GitHub ke schedule (cron) par chalte hain:
#   - Ichimoku 4H Bot: har 4 ghante -> 5.5 ghante (330 min) se purana ho to ruka hua
#   - Donchian Daily Bot: rozana 00:15 UTC -> 26 ghante (1560 min) se purana ho to ruka hua
#   - Dip Daily Bot: rozana 00:20 UTC -> 26 ghante (1560 min)
#   - Capitulation Bot: rozana 00:25 UTC -> 26 ghante (1560 min)
LOOPS = [
    ("ichimoku4h_paper_state.json", ["last_updated"], "ichimoku4h_bot.yml", 330, "Ichimoku 4H Bot"),
    ("donchian_paper_state.json", ["last_updated"], "donchian_daily_bot.yml", 1560, "Donchian Daily Bot"),
    ("dip_paper_state.json", ["last_updated"], "dip_daily_bot.yml", 1560, "Dip Daily Bot"),
    ("capit_paper_state.json", ["last_updated"], "capit_daily_bot.yml", 1560, "Capitulation Bot"),
]


def get_timestamp(path, keys):
    """State file ke andar se timestamp nikalta hai. File na ho, ya
    parh na sake, ya key na mile to None (watchdog is loop ko chhor
    deta hai - shayad abhi tak pehli baar chala hi nahi)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        for k in keys:
            data = data.get(k) if isinstance(data, dict) else None
            if data is None:
                return None
        return datetime.fromisoformat(str(data).replace("Z", "+00:00"))
    except Exception:
        return None


def trigger_workflow(workflow_file):
    """GitHub Actions API se us workflow ko workflow_dispatch bhejta
    hai (bilkul scheduled_dashboard_scan.py ke khud-dispatch jaisa)."""
    if not GH_TOKEN:
        return False, "GH_TOKEN nahi mila (workflow permissions check karein)"
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{workflow_file}/dispatches"
    headers = {"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    try:
        resp = requests.post(url, headers=headers, json={"ref": "main"}, timeout=15)
        if resp.status_code in (204, 201):
            return True, ""
        return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return False, str(e)


def main():
    now = datetime.now(timezone.utc)
    any_stale = False

    for state_file, key_path, workflow_file, stale_minutes, label in LOOPS:
        ts = get_timestamp(state_file, key_path)
        if ts is None:
            print(f"[SKIP] {label}: '{state_file}' nahi mili ya parhi nahi ja saki - "
                  f"shayad abhi tak pehli baar chala hi nahi.")
            continue

        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_minutes = (now - ts).total_seconds() / 60

        if age_minutes <= stale_minutes:
            print(f"[OK] {label}: {age_minutes:.0f} min pehle update hua (threshold {stale_minutes} min).")
            continue

        any_stale = True
        print(f"[STALE] {label}: {age_minutes:.0f} min purana (threshold {stale_minutes} min) - "
              f"'{workflow_file}' ko dobara chala rahe hain...")
        ok, msg = trigger_workflow(workflow_file)
        if ok:
            alert = (
                f"🐕 Watchdog: {label} taqreeban {age_minutes/60:.1f} ghante se ruka hua tha - "
                f"maine khud dobara chala diya hai."
            )
            print(alert)
            send_telegram_alert(alert)
        else:
            alert = (
                f"🐕⚠️ Watchdog: {label} taqreeban {age_minutes/60:.1f} ghante se ruka hua hai, "
                f"aur khud dobara chalane ki koshish NAKAM rahi ({msg}). براہِ کرم GitHub → Actions "
                f"mein manually check karein."
            )
            print(alert)
            send_telegram_alert(alert)

    if not any_stale:
        print("Watchdog: sab loops theek chal rahe hain.")


if __name__ == "__main__":
    main()
