"""
DIP DAILY - GEHRI JANCH (Swing Lab mein sirf yehi strategy asli edge dikha rahi thi)
====================================================================================
Swing Lab natija: RSI3<10 PASS (PF 2.07, win 69%, random PF 0.79) lekin 6 mein se sirf 1 variant pass,
aur portfolio return bohat kam (1% risk + 3 ATR stop = chhoti positions).
Yahan:
  1) Grid: RSI3 had 5/8/10/12/15  x  exit SMA 3/5/7  (15 combos) - edge ek "ilaqa" hai ya ek nukta?
  2) RSI3<10 par filter badal kar: BTC filter nahi / stop 2.5 ATR / stop 4 ATR
  3) Sizing: har trade equity ka 10% (stop kam hi lagta hai, is liye 1%-risk sizing bohat chhota tha)
Baqi sab usool Swing Lab wale (lookahead-free, kharcha, random control, folds, bootstrap).
Natija: dip_focus_RESULTS.txt
"""
import swing_lab as L

L.OUT = "dip_focus_RESULTS.txt"
L.FIXED_PCT = 0.10
BASE = {"rsi_lo": 10, "stop_atr": 3.0, "exit_sma": 5, "tp_r": None, "max_hold": 10, "btc": True}
GRID = [(f"RSI3<{r} exit SMA{e}", {"rsi_lo": r, "exit_sma": e}) for r in (5, 8, 10, 12, 15) for e in (3, 5, 7)]
EXTRA = [("RSI3<10 BTC filter nahi", {"btc": False}), ("RSI3<10 stop 2.5 ATR", {"stop_atr": 2.5}),
         ("RSI3<10 stop 4 ATR", {"stop_atr": 4.0})]
L.STRATS = {"DIP_DAILY": (L.dip_daily, "1d", BASE, [v for v in GRID + EXTRA if v[1] != {"rsi_lo": 10, "exit_sma": 5}])}

if __name__ == "__main__":
    L.main()
