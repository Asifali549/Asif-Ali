import requests

# --- Bot credentials ---
BOT_TOKEN = "8687734144:AAGEMWRAStFwKQmFv4ghszCxhxL_w3a5si4"
CHAT_ID = "8947945206"


def send_telegram_alert(message: str) -> bool:
    """
    Telegram bot ke zariye alert message bhejta hai.
    Returns True agar successfully bheja gaya, warna False.
    """
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }

    try:
        response = requests.post(url, data=payload, timeout=10)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Telegram alert failed: {e}")
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