import os
import re

import requests

# --- ntfy (mobile push) ---
# Har Telegram alert ntfy app par bhi jata hai. Ye wahi topic hai jo purane screener mein tha
# (ntfy app mein isi naam se subscribe hai). Badalna ho to GitHub secret NTFY_TOPIC daal dein.
NTFY_TOPIC_DEFAULT = "asifali549-strong-signals-9k3m7x"

# --- Bot credentials ---
# Token aur chat ID ab code mein NAHI likhe - GitHub Secrets se aate hain
# (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID). Repo public hai, is liye code mein
# likha token koi bhi dekh kar istemal kar sakta tha.
# Streamlit par chalane ke liye yahi do naam Streamlit Secrets mein daal dein.


def _get_credentials():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        try:
            import streamlit as st
            token = token or st.secrets.get("TELEGRAM_BOT_TOKEN")
            chat_id = chat_id or st.secrets.get("TELEGRAM_CHAT_ID")
        except Exception:
            pass
    return token, chat_id


def send_ntfy(message: str) -> bool:
    """Wahi message ntfy.sh par (HTML tags hata kar). Pehli line title ban jati hai."""
    topic = os.environ.get("NTFY_TOPIC") or NTFY_TOPIC_DEFAULT
    text = re.sub(r"<[^>]+>", "", message).strip()
    first, _, rest = text.partition("\n")
    urgent = any(k in text for k in ("BUY", "BECHEIN", "band", "STOP", "ruk gaya"))
    try:
        # JSON publish - title/message mein emoji aur Urdu bhi theek jate hain
        r = requests.post("https://ntfy.sh/", json={"topic": topic, "title": first, "message": rest.strip() or first,
                                                   "priority": 4 if urgent else 3, "tags": ["chart_with_upwards_trend"]},
                          timeout=10)
        r.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"ntfy alert failed: {e}")
        return False


def send_telegram_alert(message: str) -> bool:
    """
    Telegram bot ke zariye alert message bhejta hai - aur sath hi ntfy app par bhi.
    Returns True agar Telegram par successfully bheja gaya, warna False.
    """
    send_ntfy(message)
    token, chat_id = _get_credentials()
    if not (token and chat_id):
        print("Telegram alert NAHI bheja: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID secrets set nahi hain.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }

    try:
        response = requests.post(url, data=payload, timeout=10)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        # token URL mein hota hai - error message se chhupa dete hain
        print(f"Telegram alert failed: {str(e).replace(token, '***')}")
        return False


if __name__ == "__main__":
    example_message = (
        "🚨 <b>New Signal Detected</b>\n"
        "Symbol: BTC/USDT\n"
        "Strategy: 3-Combo Screener\n"
        "Price: 65,432\n"
        "Time: Just now"
    )
    success = send_telegram_alert(example_message)
    print("Alert sent!" if success else "Alert failed.")
