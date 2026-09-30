import os

import requests

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


def send_telegram_alert(message: str) -> bool:
    """
    Telegram bot ke zariye alert message bhejta hai.
    Returns True agar successfully bheja gaya, warna False.
    """
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
