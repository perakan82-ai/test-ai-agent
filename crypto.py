"""
Multi-Agent Crypto Scanner Dashboard — v2
=========================================
แผนกต่างๆ (Agents)
  1) Data Agent        : ccxt → Binance (15m / 1h / 4h / 1D, 1000 แท่ง)
                         RSI, EMA20/50/200, MACD, ATR, ADX, Bollinger, Volume, RS เทียบ BTC
  2) Quant Agent       : สร้างแผนเทรด (Entry / SL / TP1 / TP2) ทุกเหรียญทุก TF แบบ rule-based
                         + Backtest ย้อนหลัง → Win Rate จริง และระยะเวลาถือโดยประมาณ
  3) Macro Agent       : เทรนด์ BTC + Fear & Greed Index (alternative.me ฟรี)
  4) News Agent  (AI)  : CryptoPanic (ถ้ามีแพลน) / RSS ฟรี → Gemini สรุป Sentiment รายเหรียญ
  5) Strategist  (AI)  : Gemini รวมทุกแผนก คัดเหรียญดีที่สุด แบ่ง ระยะสั้น / กลาง / ยาว
  6) Risk Manager (AI) : Gemini ตรวจสอบแผน ปรับ Win Rate และแนะนำความเสี่ยงต่อไม้

รัน:  streamlit run app.py
⚠️ ไม่ใช่คำแนะนำการลงทุน  Win Rate คือสถิติย้อนหลัง + การประเมิน ไม่ใช่การรับประกันผล
"""

import concurrent.futures as cf
import re
import time
from datetime import datetime, timedelta, timezone

import ccxt
import feedparser
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

st.set_page_config(page_title="Multi-Agent Crypto Scanner", page_icon="📈", layout="wide")

# =============================================================================
# CONFIG
# =============================================================================
TIMEFRAMES = ["15m", "1h", "4h", "1d"]
TF_LABEL = {"15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D"}
TF_HOURS = {"15m": 0.25, "1h": 1.0, "4h": 4.0, "1d": 24.0}
HIGHER_TF = {"15m": "1h", "1h": "4h", "4h": "1d", "1d": "1d"}
KLINE_LIMIT = 1000

# พารามิเตอร์แผนเทรด / backtest ต่อ TF
SL_ATR = {"15m": 1.5, "1h": 1.5, "4h": 1.5, "1d": 2.0}       # SL = กี่ ATR
RR_TP1 = {"15m": 1.5, "1h": 1.5, "4h": 2.0, "1d": 2.0}       # TP1 = กี่เท่าของความเสี่ยง
RR_TP2 = {"15m": 3.0, "1h": 3.0, "4h": 4.0, "1d": 4.0}       # TP2
MAX_HOLD = {"15m": 32, "1h": 48, "4h": 42, "1d": 60}         # ถือสูงสุดกี่แท่งใน backtest

HORIZONS = {
    "SHORT": {"label": "⚡ ระยะสั้น (Scalping / Day Trade)", "tfs": ["15m", "1h"],
              "hold": "ไม่กี่ชั่วโมง – 2 วัน"},
    "SWING": {"label": "🌊 ระยะกลาง (Swing Trade)", "tfs": ["4h"],
              "hold": "3 – 14 วัน"},
    "LONG": {"label": "🏦 ระยะยาว (Investment / Accumulate)", "tfs": ["1d"],
             "hold": "หลายสัปดาห์ – หลายเดือน"},
}

BINANCE_HOSTS = [
    "https://data-api.binance.vision/api/v3",   # ใช้ได้จาก Streamlit Cloud (US)
    "https://api.binance.com/api/v3",
]

EXCLUDE_BASES = {
    "USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "USDE", "USD1", "EUR", "EURI",
    "AEUR", "PYUSD", "RLUSD", "BFUSD", "XUSD", "USDS", "PAXG", "WBTC", "WBETH", "BNSOL",
}

# 15 เหรียญที่ผู้ใช้สนใจ + 15 เหรียญเสนอเพิ่ม (กระจายหลายกลุ่ม)
USER_COINS = "BTC ETH SOL XRP XLM ADA NEAR ENA HYPE AAVE ZEC BNB PUMP UNI HBAR"
EXTRA_COINS = "DOGE LINK SUI AVAX TAO ONDO TRX LTC BCH WLD JUP PENDLE ARB PEPE RENDER"
DEFAULT_WATCHLIST = USER_COINS + "\n" + EXTRA_COINS
MAX_WATCHLIST = 40

RSS_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
}

COIN_NAMES = {
    "BTC": ["bitcoin"], "ETH": ["ethereum", "ether"], "SOL": ["solana"], "XRP": ["ripple", "xrp"],
    "BNB": ["bnb", "binance coin"], "DOGE": ["dogecoin"], "ADA": ["cardano"], "TRX": ["tron"],
    "AVAX": ["avalanche"], "LINK": ["chainlink"], "DOT": ["polkadot"], "LTC": ["litecoin"],
    "SUI": ["sui"], "TON": ["toncoin"], "SHIB": ["shiba"], "PEPE": ["pepe"], "NEAR": ["near protocol"],
    "APT": ["aptos"], "ARB": ["arbitrum"], "OP": ["optimism"], "UNI": ["uniswap"], "AAVE": ["aave"],
    "ENA": ["ethena"], "WLD": ["worldcoin", "world network"], "FET": ["fetch.ai"], "HBAR": ["hedera"],
    "XLM": ["stellar"], "BCH": ["bitcoin cash"], "ETC": ["ethereum classic"], "FIL": ["filecoin"],
    "TAO": ["bittensor"], "INJ": ["injective"], "SEI": ["sei"], "TIA": ["celestia"], "ONDO": ["ondo"],
    "HYPE": ["hyperliquid"], "ZEC": ["zcash"], "PUMP": ["pump.fun", "pumpfun"], "JUP": ["jupiter"],
    "PENDLE": ["pendle"], "RENDER": ["render network"],
}

GEMINI_MODELS = ["gemini-3.8-flash", "gemini-3.1-flash-lite", "อื่นๆ (พิมพ์เอง)"]

# เทรด Spot อย่างเดียว (ไม่มี Short / ไม่มี Leverage) — เปลี่ยนได้ที่ Sidebar
SPOT_ONLY = True


# =============================================================================
# HELPERS
# =============================================================================
def fmt_price(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "-"
    if np.isnan(x):
        return "-"
    if x >= 1000:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:,.4f}"
    if x >= 0.01:
        return f"{x:.5f}"
    return f"{x:.8f}"


def fmt_hours(h) -> str:
    if h is None or (isinstance(h, float) and np.isnan(h)):
        return "-"
    if h < 1:
        return f"{max(15, round(h * 60 / 15) * 15):.0f} นาที"
    if h < 48:
        return f"{h:.0f} ชม."
    if h < 24 * 60:
        return f"{h / 24:.0f} วัน"
    return f"{h / 24 / 30:.1f} เดือน"


def get_secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, ""))
    except Exception:  # noqa: BLE001
        return ""


def safe_float(v, default=np.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def parse_watchlist(text: str) -> list:
    out = []
    for tok in re.split(r"[\s,;]+", (text or "").upper()):
        tok = re.sub(r"[/\-_]", "", tok)
        if tok.endswith("USDT") and len(tok) > 4:
            tok = tok[:-4]
        if tok and tok not in out:
            out.append(tok)
    return out


# =============================================================================
# 1) DATA AGENT
# =============================================================================
def make_exchange(host: str):
    ex = ccxt.binance({"enableRateLimit": True, "timeout": 20000})
    ex.urls["api"]["public"] = host
    return ex


def connect_binance():
    last_err = None
    for host in BINANCE_HOSTS:
        try:
            ex = make_exchange(host)
            ex.public_get_ping()
            return ex, host
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"เชื่อมต่อ Binance ไม่ได้: {last_err}")


def get_symbols(ex, watchlist, n_top: int, n_discovery: int) -> pd.DataFrame:
    """watchlist=None → Top n_top ตาม volume | มี watchlist → watchlist + Discovery n_discovery ตัว"""
    tickers = ex.public_get_ticker_24hr()
    now_ms = time.time() * 1000
    rows = []
    for t in tickers:
        sym = t.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        base = sym[:-4]
        price = safe_float(t.get("lastPrice"), 0)
        qvol = safe_float(t.get("quoteVolume"), 0)
        close_time = safe_float(t.get("closeTime"), 0)
        if price <= 0 or qvol <= 0 or now_ms - close_time > 86_400_000:
            continue
        rows.append({"symbol": sym, "base": base, "price": price,
                     "change_24h_%": safe_float(t.get("priceChangePercent"), 0),
                     "quote_volume_usdt": qvol})
    if not rows:
        raise RuntimeError("ไม่พบคู่เหรียญ USDT ที่ใช้ได้จาก Binance")
    all_df = pd.DataFrame(rows)
    tradable = all_df[~all_df["base"].isin(EXCLUDE_BASES)].sort_values("quote_volume_usdt", ascending=False)

    if watchlist:
        wl = all_df[all_df["base"].isin(watchlist)].copy()
        missing = [c for c in watchlist if c not in set(wl["base"])]
        if missing:
            st.warning(f"ไม่พบคู่ USDT ที่เทรดอยู่บน Binance: {', '.join(missing)} (ข้ามไป)")
        order = {c: i for i, c in enumerate(watchlist)}
        wl = wl.sort_values("base", key=lambda s: s.map(order))
        wl["source"] = "Watchlist"
        parts = [wl]
        if n_discovery > 0:
            disc = tradable[~tradable["base"].isin(watchlist)].head(n_discovery).copy()
            disc["source"] = "Discovery 🔍"
            parts.append(disc)
        df = pd.concat(parts)
    else:
        df = tradable.head(n_top).copy()
        df["source"] = "Top Volume"

    if "BTC" not in set(df["base"]):  # ต้องมี BTC ไว้วัดภาวะตลาด
        btc = all_df[all_df["base"] == "BTC"].copy()
        btc["source"] = "Benchmark"
        df = pd.concat([df, btc])
    return df.reset_index(drop=True)


def fetch_klines(ex, symbol: str, tf: str, limit: int = KLINE_LIMIT) -> pd.DataFrame:
    raw = ex.public_get_klines({"symbol": symbol, "interval": tf, "limit": limit})
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "qvol", "trades", "tb_base", "tb_quote", "ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(pd.to_numeric(df["open_time"]), unit="ms", utc=True)
    return df[["open_time", "open", "high", "low", "close", "volume"]].dropna().reset_index(drop=True)


# ---------- Indicators ----------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def calc_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(gain.notna())


def calc_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def calc_adx(df: pd.DataFrame, n: int = 14):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = calc_atr(df, n)
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, mdi


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    df["ema20"], df["ema50"], df["ema200"] = ema(c, 20), ema(c, 50), ema(c, 200)
    df["rsi"] = calc_rsi(c)
    macd = ema(c, 12) - ema(c, 26)
    df["macd"] = macd
    df["macd_sig"] = ema(macd, 9)
    df["hist"] = macd - df["macd_sig"]
    df["atr"] = calc_atr(df)
    df["adx"], df["pdi"], df["mdi"] = calc_adx(df)
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    df["bb_up"], df["bb_lo"] = mid + 2 * sd, mid - 2 * sd
    df["bb_width"] = 4 * sd / mid * 100
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    return df


# =============================================================================
# 2) QUANT AGENT — แผนเทรด + Backtest
# =============================================================================
def score_row(r) -> float:
    s = 0.0
    if r["close"] > r["ema50"] > r["ema200"]:
        s += 1.5
    elif r["close"] < r["ema50"] < r["ema200"]:
        s -= 1.5
    s += 0.5 if r["close"] > r["ema50"] else -0.5
    s += 0.5 if r["hist"] > 0 else -0.5
    s += 0.25 if r["close"] > r["ema20"] else -0.25
    if 45 <= r["rsi"] <= 65:
        s += 0.25 * np.sign(s or 1)
    elif r["rsi"] >= 75:
        s -= 0.5
    elif r["rsi"] <= 25:
        s += 0.5
    if r["adx"] >= 25:
        s *= 1.15
    elif r["adx"] < 15:
        s *= 0.8
    return float(np.clip(s, -3, 3))


def backtest(df: pd.DataFrame, tf: str, direction: str) -> dict:
    """จำลองเทรดตาม setup แนวโน้มเดียวกันย้อนหลัง: เข้า close, SL = k·ATR, TP1 = RR·risk
    ชนะ = แตะ TP1 ก่อน SL ภายใน MAX_HOLD แท่ง (ถ้าแตะทั้งคู่ในแท่งเดียว นับว่าแพ้)"""
    c, h, l = df["close"].values, df["high"].values, df["low"].values
    atr = df["atr"].values
    if direction == "LONG":
        sig = ((df["close"] > df["ema50"]) & (df["ema50"] > df["ema200"]) & (df["hist"] > 0)
               & df["rsi"].between(40, 72)).values
    else:
        sig = ((df["close"] < df["ema50"]) & (df["ema50"] < df["ema200"]) & (df["hist"] < 0)
               & df["rsi"].between(28, 60)).values
    k, rr, hold = SL_ATR[tf], RR_TP1[tf], MAX_HOLD[tf]
    n = len(df)
    wins = losses = timeouts = 0
    r_sum, win_bars = 0.0, []
    i = 200
    while i < n - 2:
        if not sig[i] or not np.isfinite(atr[i]) or atr[i] <= 0:
            i += 1
            continue
        e, risk = c[i], k * atr[i]
        sl = e - risk if direction == "LONG" else e + risk
        tp = e + rr * risk if direction == "LONG" else e - rr * risk
        outcome, j = None, i
        for j in range(i + 1, min(n, i + 1 + hold)):
            hit_sl = l[j] <= sl if direction == "LONG" else h[j] >= sl
            hit_tp = h[j] >= tp if direction == "LONG" else l[j] <= tp
            if hit_sl:
                outcome = "loss"
                break
            if hit_tp:
                outcome = "win"
                break
        if outcome is None:
            if j >= n - 1 and j - i < hold:  # เทรดยังไม่จบ (ข้อมูลหมด) → ไม่นับ
                break
            outcome = "timeout"
        if outcome == "win":
            wins += 1
            r_sum += rr
            win_bars.append(j - i)
        elif outcome == "loss":
            losses += 1
            r_sum -= 1
        else:
            timeouts += 1
            move = (c[j] - e) if direction == "LONG" else (e - c[j])
            r_sum += move / risk
        i = j + 1
    total = wins + losses + timeouts
    wr_raw = wins / total * 100 if total else np.nan
    wr_smooth = (wins + 2) / (total + 4) * 100  # Bayesian smoothing กันตัวอย่างน้อย
    hrs = TF_HOURS[tf]
    if win_bars:
        p25, p50, p75 = np.percentile(win_bars, [25, 50, 75])
        hold_txt = f"{fmt_hours(p25 * hrs)} – {fmt_hours(p75 * hrs)}"
        hold_med = p50 * hrs
    else:
        hold_txt, hold_med = f"≤ {fmt_hours(hold * hrs)}", np.nan
    return {"n": total, "wins": wins, "wr_raw": wr_raw, "wr_smooth": wr_smooth,
            "expectancy_r": r_sum / total if total else np.nan,
            "hold_txt": hold_txt, "hold_med_h": hold_med}


def build_plan(last, tf: str, direction: str) -> dict:
    e, a = float(last["close"]), float(last["atr"])
    risk = SL_ATR[tf] * a
    if direction == "LONG":
        sl, tp1, tp2 = e - risk, e + RR_TP1[tf] * risk, e + RR_TP2[tf] * risk
        zone_lo = min(e, max(float(last["ema20"]), e - 0.6 * a))
        zone_hi = e
    else:
        sl, tp1, tp2 = e + risk, e - RR_TP1[tf] * risk, e - RR_TP2[tf] * risk
        zone_lo = e
        zone_hi = max(e, min(float(last["ema20"]), e + 0.6 * a))
    return {"entry": e, "entry_low": zone_lo, "entry_high": zone_hi, "sl": sl, "tp1": tp1, "tp2": tp2}


def analyze_tf(df: pd.DataFrame, tf: str) -> dict:
    if len(df) < 220:
        raise ValueError("ข้อมูลแท่งเทียนน้อยเกินไป")
    df = add_indicators(df)
    last, prev = df.iloc[-1], df.iloc[-2]
    score = score_row(last)
    signal = "LONG" if score >= 0 else "SHORT"           # สัญญาณจริงของตลาด
    direction = "LONG" if SPOT_ONLY else signal          # Spot = วางแผนซื้อเท่านั้น
    if score >= 1:
        spot_status = "🟢 ซื้อได้"
    elif score >= 0:
        spot_status = "🟡 รอจังหวะ"
    else:
        spot_status = "🔴 ยังไม่ควรซื้อ"
    if last["close"] > last["ema50"] > last["ema200"]:
        trend = "UP"
    elif last["close"] < last["ema50"] < last["ema200"]:
        trend = "DOWN"
    else:
        trend = "SIDEWAY"
    recent = np.sign(df["hist"].iloc[-4:].values)
    cross = "bullish" if recent[0] < 0 < recent[-1] else ("bearish" if recent[0] > 0 > recent[-1] else "none")
    bt = backtest(df, tf, direction)
    plan = build_plan(last, tf, direction)
    return {
        "df": df.tail(200).reset_index(drop=True),
        "close": float(last["close"]), "rsi": round(float(last["rsi"]), 1), "adx": round(float(last["adx"]), 1),
        "ema20": float(last["ema20"]), "ema50": float(last["ema50"]), "ema200": float(last["ema200"]),
        "hist": float(last["hist"]), "hist_rising": bool(last["hist"] > prev["hist"]), "macd_cross": cross,
        "atr": float(last["atr"]), "atr_pct": round(float(last["atr"] / last["close"] * 100), 2),
        "bb_width": round(float(last["bb_width"]), 2), "vol_ratio": round(float(last["vol_ratio"]), 2),
        "hi50": float(df["high"].iloc[-50:].max()), "lo50": float(df["low"].iloc[-50:].min()),
        "trend": trend, "score": round(score, 2), "direction": direction, "signal": signal,
        "spot_status": spot_status, "bt": bt, "plan": plan,
    }


def system_win_rate(d: dict, htf: dict | None) -> float:
    """Win Rate ของระบบ = Backtest (smoothed) ปรับด้วย confluence ปัจจุบัน"""
    wr = d["bt"]["wr_smooth"]
    if htf:
        same = (htf["score"] >= 0) == (d["direction"] == "LONG")
        wr += 4 if same and abs(htf["score"]) >= 1 else (-6 if not same else 0)
    if d["adx"] >= 25:
        wr += 2
    elif d["adx"] < 15:
        wr -= 3
    if d["direction"] == "LONG" and d["rsi"] >= 75:
        wr -= 4
    if d["direction"] == "SHORT" and d["rsi"] <= 25:
        wr -= 4
    if d["vol_ratio"] >= 1.5:
        wr += 1
    if d["direction"] != d["signal"]:  # Spot ซื้อทั้งที่สัญญาณยังเป็นขาลง
        wr -= 8
    elif abs(d["score"]) < 1:          # สัญญาณยังไม่ชัด
        wr -= 3
    if d["bt"]["n"] < 8:  # ตัวอย่างน้อย → ดึงเข้าหา 45%
        wr = 0.5 * wr + 0.5 * 45
    return float(np.clip(wr, 20, 85))


def run_data_and_quant(watchlist, n_top, n_discovery, progress):
    ex, host = connect_binance()
    top = get_symbols(ex, watchlist, n_top, n_discovery)
    jobs = [(s, tf) for s in top["symbol"] for tf in TIMEFRAMES]
    results = {s: {} for s in top["symbol"]}
    errors = []

    def work(job):
        sym, tf = job
        return sym, tf, analyze_tf(fetch_klines(ex, sym, tf), tf)

    done = 0
    with cf.ThreadPoolExecutor(max_workers=10) as pool:
        futs = {pool.submit(work, j): j for j in jobs}
        for fut in cf.as_completed(futs):
            done += 1
            try:
                sym, tf, res = fut.result()
                results[sym][tf] = res
            except Exception as e:  # noqa: BLE001
                errors.append(f"{futs[fut][0]} {futs[fut][1]}: {str(e)[:60]}")
            progress.progress(done / len(jobs), text=f"📊 Data + Quant Agent: {done}/{len(jobs)}")
    if not any(results.values()):
        raise RuntimeError("ดึงข้อมูลแท่งเทียนไม่สำเร็จเลย — ลองใหม่อีกครั้ง")
    if errors:
        st.caption(f"ข้ามข้อมูล {len(errors)} ชุด: " + "; ".join(errors[:5]))

    # Win Rate ระบบ (ต้องรู้ TF ใหญ่ก่อน)
    for sym, tfs in results.items():
        for tf, d in tfs.items():
            d["sys_wr"] = round(system_win_rate(d, tfs.get(HIGHER_TF[tf]) if tf != "1d" else None), 1)

    # Relative strength 30 วัน เทียบ BTC
    def ret30(sym):
        d = results.get(sym, {}).get("1d")
        if not d or len(d["df"]) < 31:
            return np.nan
        c = d["df"]["close"]
        return (c.iloc[-1] / c.iloc[-31] - 1) * 100

    btc_ret = ret30("BTCUSDT")
    top["ret30_%"] = top["symbol"].map(ret30)
    top["rs_btc_%"] = top["ret30_%"] - btc_ret
    return top, results, host


# =============================================================================
# 3) MACRO AGENT
# =============================================================================
def fetch_fear_greed() -> dict | None:
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=2", timeout=10)
        data = r.json()["data"]
        return {"value": int(data[0]["value"]), "label": data[0]["value_classification"],
                "prev": int(data[1]["value"]) if len(data) > 1 else None}
    except Exception:  # noqa: BLE001
        return None


def macro_view(results: dict, fng: dict | None) -> dict:
    btc = results.get("BTCUSDT", {})
    view = {"btc_1d": btc.get("1d", {}).get("trend", "-"), "btc_4h": btc.get("4h", {}).get("trend", "-"),
            "btc_score_1d": btc.get("1d", {}).get("score", 0.0), "fng": fng}
    s = view["btc_score_1d"]
    view["regime"] = "Risk-On 🟢" if s >= 1 else ("Risk-Off 🔴" if s <= -1 else "Neutral 🟡")
    return view


# =============================================================================
# 4) NEWS AGENT
# =============================================================================
def fetch_cryptopanic(token: str, plan: str, limit: int = 50) -> list:
    params = {"auth_token": token.strip(), "public": "true", "kind": "news", "regions": "en"}
    plans = [plan] + [p for p in ("growth", "enterprise", "developer") if p != plan]
    r, log = None, []
    for pl in plans:
        r = requests.get(f"https://cryptopanic.com/api/{pl}/v2/posts/", params=params, timeout=15,
                         headers={"User-Agent": "crypto-scanner/2.0"})
        log.append(f"{pl}={r.status_code}")
        if r.status_code != 404:
            break
    if r is None or r.status_code != 200:
        code = r.status_code if r is not None else "-"
        reason = {401: "token ไม่ถูกต้อง", 403: "แพลนไม่มีสิทธิ์ใช้ API",
                  404: "แพลน Developer ฟรีถูกยกเลิกแล้ว", 429: "เรียกถี่เกินโควต้า"}.get(code, f"HTTP {code}")
        raise RuntimeError(f"{reason} [{', '.join(log)}]")
    items = []
    for p in r.json().get("results", [])[:limit]:
        inst = p.get("instruments") or p.get("currencies") or []
        votes = p.get("votes") or {}
        items.append({"time": p.get("published_at", ""), "source": (p.get("source") or {}).get("title", "CryptoPanic"),
                      "title": p.get("title", ""),
                      "coins": [c.get("code", "").upper() for c in inst if isinstance(c, dict)],
                      "url": p.get("original_url") or p.get("url", ""),
                      "votes": f"+{votes.get('positive', 0)}/-{votes.get('negative', 0)}"})
    return items


def tag_coins(text: str, symbols: list) -> list:
    low = text.lower()
    return [s for s in symbols if re.search(rf"\b{re.escape(s)}\b", text)
            or any(re.search(rf"\b{re.escape(n)}\b", low) for n in COIN_NAMES.get(s, []))]


def fetch_rss_news(symbols: list, per_feed: int = 25) -> list:
    items = []
    for src, url in RSS_FEEDS.items():
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            for e in feedparser.parse(resp.content).entries[:per_feed]:
                title = e.get("title", "")
                items.append({"time": e.get("published", ""), "source": src, "title": title,
                              "coins": tag_coins(title, symbols), "url": e.get("link", ""), "votes": ""})
        except Exception:  # noqa: BLE001
            continue
    return items


class CoinSentiment(BaseModel):
    symbol: str = Field(description="base symbol เช่น BTC")
    sentiment: str = Field(description="Bullish, Bearish หรือ Neutral")
    score: int = Field(description="-100 ถึง 100")
    reason: str = Field(description="เหตุผลสั้นๆ ภาษาไทย อ้างอิงข่าว")


class NewsReport(BaseModel):
    overall_sentiment: str = Field(description="Bullish, Bearish หรือ Neutral")
    overall_score: int = Field(description="-100 ถึง 100")
    summary_th: str = Field(description="สรุปภาพรวมข่าวภาษาไทย 3-5 ประโยค")
    key_themes: list[str] = Field(description="ประเด็นสำคัญ 3-6 ข้อ ภาษาไทย")
    coins: list[CoinSentiment]


def run_news_agent(client, model, symbols, cp_token, cp_plan):
    news, src = [], ""
    if cp_token:
        try:
            news, src = fetch_cryptopanic(cp_token, cp_plan), "CryptoPanic"
        except Exception as e:  # noqa: BLE001
            st.info(f"CryptoPanic ใช้ไม่ได้: {str(e).replace(cp_token, '***')} → ใช้ RSS แทน")
    if not news:
        news, src = fetch_rss_news(symbols), "RSS (CoinDesk / Cointelegraph / Decrypt)"
    if not news:
        return None, [], "ไม่มีข่าว"
    headlines = "\n".join(f"- [{n['source']}] {n['title']}" + (f" | coins: {','.join(n['coins'])}" if n["coins"] else "")
                          for n in news[:70])
    prompt = f"""คุณคือ News Sentiment Agent ด้านคริปโต
วิเคราะห์พาดหัวข่าวล่าสุด สรุป Sentiment ตลาดรวม และรายเหรียญ (เฉพาะเหรียญในรายการที่มีข่าวเกี่ยวข้อง
ทั้งทางตรงหรือทางอ้อม เช่น ข่าว ETF / กฎหมาย / ระบบนิเวศ): {', '.join(symbols)}
ข่าวระดับมหภาค (Fed, ดอกเบี้ย, กฎหมาย, ETF flow) ให้สะท้อนใน overall_score
ห้ามแต่งข่าว ใช้เฉพาะข้อมูลที่ให้

ข่าว:
{headlines}
"""
    try:
        report = call_gemini(client, model, prompt, NewsReport, temperature=0.2, max_retries=4)
    except Exception as e:  # noqa: BLE001
        st.info(f"News Agent สรุปไม่สำเร็จ ({str(e)[:100]}) → ส่งพาดหัวข่าวให้ Strategist วิเคราะห์เอง")
        report = None
    return report, news, src


# =============================================================================
# 5) STRATEGIST AGENT
# =============================================================================
class AIPick(BaseModel):
    symbol: str = Field(description="base symbol เช่น SOL")
    horizon: str = Field(description="SHORT, SWING หรือ LONG")
    timeframe: str = Field(description="TF ที่ใช้เข้าเทรด: 15m, 1h, 4h หรือ 1d")
    direction: str = Field(description="LONG หรือ SHORT")
    entry: float = Field(description="ราคาเข้าหลัก")
    entry_low: float = Field(description="ขอบล่างโซนเข้า")
    entry_high: float = Field(description="ขอบบนโซนเข้า")
    stop_loss: float
    tp1: float
    tp2: float
    win_rate: int = Field(description="โอกาสถึง TP1 ก่อน SL (%) เริ่มจาก SysWR แล้วปรับ")
    hold_period: str = Field(description="ระยะเวลาถือที่แนะนำ ภาษาไทย เช่น '4-12 ชั่วโมง', '5-10 วัน', '2-4 เดือน'")
    entry_trigger: str = Field(description="เงื่อนไขก่อนเข้า เช่น 'รอแท่ง 1H ปิดเหนือ EMA20' ภาษาไทย")
    technical_reason: str = Field(description="เหตุผลทางเทคนิค ภาษาไทย 2-3 ประโยค อ้างอิงตัวเลขจริง")
    news_reason: str = Field(description="ผลจากข่าว/Sentiment/Macro ภาษาไทย 1-2 ประโยค")
    invalidation: str = Field(description="สัญญาณที่ทำให้แผนนี้ใช้ไม่ได้ ภาษาไทย")
    confidence: str = Field(description="สูง, กลาง หรือ ต่ำ")


class StrategyReport(BaseModel):
    market_overview: str = Field(description="ภาพรวมตลาด ภาษาไทย 3-5 ประโยค รวมเทคนิค + ข่าว + Fear&Greed")
    best_horizon_now: str = Field(description="ภาวะตลาดตอนนี้เหมาะกับระยะไหนที่สุด พร้อมเหตุผล ภาษาไทย 1-2 ประโยค")
    picks: list[AIPick]


def build_quant_table(top: pd.DataFrame, results: dict, news_map: dict) -> str:
    lines = []
    for _, row in top.iterrows():
        tfs = results.get(row["symbol"], {})
        if not tfs:
            continue
        ns = news_map.get(row["base"])
        head = (f"## {row['base']} [{row['source']}] px={fmt_price(row['price'])} 24h={row['change_24h_%']:+.1f}% "
                f"vol=${row['quote_volume_usdt'] / 1e6:,.0f}M RS30d_vsBTC={row['rs_btc_%']:+.1f}%"
                + (f" news={ns['sentiment']}({ns['score']})" if ns else ""))
        lines.append(head)
        for tf in TIMEFRAMES:
            d = tfs.get(tf)
            if not d:
                continue
            p, bt = d["plan"], d["bt"]
            lines.append(
                f"  {TF_LABEL[tf]}: signal={d['signal']} plan={d['direction']} status={d['spot_status']} "
                f"score={d['score']} trend={d['trend']} RSI={d['rsi']} "
                f"ADX={d['adx']} MACDh={'+' if d['hist'] > 0 else '-'}{'↑' if d['hist_rising'] else '↓'} "
                f"cross={d['macd_cross']} vol×{d['vol_ratio']} ATR%={d['atr_pct']} "
                f"EMA20={fmt_price(d['ema20'])} EMA50={fmt_price(d['ema50'])} EMA200={fmt_price(d['ema200'])} "
                f"Hi50={fmt_price(d['hi50'])} Lo50={fmt_price(d['lo50'])} | PLAN E={fmt_price(p['entry'])} "
                f"zone={fmt_price(p['entry_low'])}-{fmt_price(p['entry_high'])} SL={fmt_price(p['sl'])} "
                f"TP1={fmt_price(p['tp1'])} TP2={fmt_price(p['tp2'])} | BT WR={bt['wr_raw']:.0f}% n={bt['n']} "
                f"expR={bt['expectancy_r']:+.2f} hold={bt['hold_txt']} | SysWR={d['sys_wr']}%"
            )
    return "\n".join(lines)


def run_strategist(client, model, top, results, macro, news_report, news_items, n_picks) -> StrategyReport:
    news_map = {c.symbol.upper(): {"sentiment": c.sentiment, "score": c.score} for c in (news_report.coins if news_report else [])}
    quant = build_quant_table(top, results, news_map)
    fng = macro["fng"]
    macro_txt = (f"BTC 1D trend={macro['btc_1d']} 4H trend={macro['btc_4h']} regime={macro['regime']} | "
                 + (f"Fear&Greed={fng['value']} ({fng['label']}), เมื่อวาน {fng['prev']}" if fng else "Fear&Greed=ไม่มีข้อมูล"))
    if news_report:
        news_txt = (f"Overall {news_report.overall_sentiment} ({news_report.overall_score}): {news_report.summary_th}\n"
                    + "\n".join(f"- {c.symbol}: {c.sentiment} ({c.score}) {c.reason}" for c in news_report.coins))
    else:
        news_txt = "(ไม่มีสรุปจาก News Agent ให้ประเมินเองจากพาดหัว)\n" + "\n".join(
            f"- {n['title']}" for n in news_items[:30])

    prompt = f"""คุณคือ Chief Strategist ของทีมเทรดคริปโต รวมผลจาก Data/Quant/Macro/News Agent
เวลาปัจจุบัน (UTC): {datetime.now(timezone.utc):%Y-%m-%d %H:%M}

=== MACRO ===
{macro_txt}

=== NEWS ===
{news_txt}

=== QUANT (Binance Spot) ===
คำอธิบาย: score -3..+3 | PLAN = แผนที่ระบบคำนวณ (SL = {SL_ATR['1h']}-{SL_ATR['1d']} ATR, TP1/TP2 = R-multiple)
BT WR = อัตราชนะย้อนหลัง (ถึง TP1 ก่อน SL) ของ setup แนวโน้มเดียวกัน, n = จำนวนเทรด, expR = กำไรคาดหวังต่อเทรด (หน่วย R)
hold = ช่วงเวลาที่เทรดชนะใช้จริงในอดีต | SysWR = BT WR ปรับด้วย confluence ปัจจุบัน
{quant}

=== งาน ===
คัดเหรียญที่ดีที่สุดรวม {n_picks} รายการ กระจาย 3 ระยะ (อย่างน้อยระยะละ 1):
- SHORT: Scalping (TF 15m) หรือ Day Trade (TF 1h) — ถือไม่กี่ชั่วโมงถึง 2 วัน
- SWING: TF 4h — ถือ 3-14 วัน
- LONG : TF 1d — ถือหลายสัปดาห์ถึงหลายเดือน (direction = LONG เท่านั้น ถือเป็นการสะสม)

กฎ:
1. ใช้ตัวเลข PLAN ของเหรียญ/TF นั้นเป็นจุดเริ่ม ปรับได้เฉพาะเมื่อมีเหตุผลเชิงโครงสร้าง (EMA, Hi50/Lo50) และห้ามแต่งราคา
2. LONG: SL < entry_low ≤ entry ≤ entry_high < TP1 < TP2 | SHORT: SL > entry_high ≥ entry ≥ entry_low > TP1 > TP2
3. win_rate เริ่มจาก SysWR แล้วปรับได้ไม่เกิน ±10 ตามข่าว/Macro พร้อมบอกเหตุผล ห้ามเกิน 85
4. เลือกเหรียญที่ expR เป็นบวก, TF ใหญ่สอดคล้อง, ADX ไม่ต่ำ และข่าวไม่ขัด
5. ถ้าตลาด Risk-Off ให้ระยะ LONG เน้นโซนสะสมที่ต่ำกว่า (EMA50/EMA200 1D) และลด win_rate
6. hold_period อ้างอิงคอลัมน์ hold ของ TF นั้น
7. ทุกข้อความเป็นภาษาไทย กระชับ อ้างตัวเลขจริง
{SPOT_RULES if SPOT_ONLY else ""}"""
    return call_gemini(client, model, prompt, StrategyReport, temperature=0.3, max_retries=6)


SPOT_RULES = """
=== สำคัญ: ผู้ใช้เทรด SPOT เท่านั้น (ไม่มี Short ไม่มี Leverage) ===
8. direction ต้องเป็น LONG ทุกรายการ ห้ามแนะนำ SHORT เด็ดขาด
9. เลือกเหรียญที่ status = 🟢 ซื้อได้ ก่อน ถ้าเป็น 🟡 รอจังหวะ ให้ระบุ entry_trigger ที่ต้องรอให้ชัด
10. เหรียญ signal=SHORT (🔴 ยังไม่ควรซื้อ) ห้ามเลือกเป็นระยะ SHORT/SWING
    ใช้ได้เฉพาะระยะ LONG แบบตั้ง Limit สะสมที่แนวรับ (Lo50 / EMA200 ของ 1D) ต่ำกว่าราคาปัจจุบัน และต้องลด win_rate
11. ถ้าตลาดขาลงทั้งตลาด ให้คัดน้อยลงได้ และบอกใน best_horizon_now ว่าควรถือเงินสด (USDT) รอ
"""


# =============================================================================
# 6) RISK MANAGER AGENT
# =============================================================================
class RiskReview(BaseModel):
    symbol: str
    horizon: str = Field(description="SHORT, SWING หรือ LONG")
    verdict: str = Field(description="APPROVE, CAUTION หรือ REJECT")
    final_win_rate: int = Field(description="Win Rate สุดท้ายหลังตรวจสอบ (%)")
    risk_per_trade_pct: float = Field(description="แนะนำเสี่ยงกี่ % ของพอร์ตต่อไม้ 0.25-2.0")
    note: str = Field(description="ความเห็นด้านความเสี่ยง ภาษาไทย 1-2 ประโยค")


class RiskReport(BaseModel):
    portfolio_note: str = Field(description="ภาพรวมความเสี่ยงพอร์ต ภาษาไทย 2-3 ประโยค เช่น ความสัมพันธ์ระหว่างเหรียญ")
    max_concurrent_trades: int = Field(description="จำนวนไม้ที่ไม่ควรเปิดพร้อมกันเกิน")
    reviews: list[RiskReview]


def run_risk_manager(client, model, picks: list, macro: dict) -> RiskReport:
    rows = "\n".join(
        f"- {p['symbol']} {p['horizon']} {p['timeframe']} {p['direction']} E={fmt_price(p['entry'])} "
        f"SL={fmt_price(p['stop_loss'])} ({p['sl_pct']:+.2f}%) TP1={fmt_price(p['tp1'])} ({p['tp1_pct']:+.2f}%) "
        f"TP2={fmt_price(p['tp2'])} R:R={p['rr']} WR={p['win_rate']}% SysWR={p['sys_wr']}% "
        f"BTn={p['bt_n']} expR={p['exp_r']:+.2f} confidence={p['confidence']} warnings={p['warnings']}"
        for p in picks)
    fng = macro["fng"]
    spot_txt = ("ผู้ใช้เทรด SPOT เท่านั้น ไม่มี Leverage: ความเสี่ยงหลักคือซื้อเหรียญในช่วงขาลง "
                "ให้ REJECT แผนที่ซื้อสวนเทรนด์ใน TF สั้น/กลาง และระวังการถือ altcoin หลายตัวที่วิ่งตาม BTC พร้อมกัน\n"
                if SPOT_ONLY else "")
    prompt = f"""คุณคือ Risk Manager ตรวจแผนเทรดจาก Strategist อย่างเข้มงวด
{spot_txt}Macro: BTC 1D {macro['btc_1d']}, regime {macro['regime']}, Fear&Greed {fng['value'] if fng else '-'}

แผนที่ต้องตรวจ:
{rows}

ตรวจ: R:R, ระยะ SL (แคบเกินโดนสะบัด / กว้างเกิน), จำนวนตัวอย่าง backtest, ความสัมพันธ์ระหว่างเหรียญ
(เช่น altcoin หลายตัวทิศเดียวกันพร้อมกัน = ความเสี่ยงซ้อน), ภาวะตลาด
- final_win_rate ปรับจาก WR ได้ไม่เกิน ±10
- REJECT เมื่อ expR ติดลบชัดเจน หรือแผนขัดกับภาวะตลาดรุนแรง
- risk_per_trade_pct: setup ดี 1-2%, ปานกลาง 0.5-1%, เสี่ยง 0.25-0.5%
ตอบภาษาไทย
"""
    return call_gemini(client, model, prompt, RiskReport, temperature=0.2, max_retries=4)


# =============================================================================
# VALIDATION / MERGE
# =============================================================================
def normalize_horizon(h: str) -> str:
    h = (h or "").upper()
    if any(k in h for k in ("SHORT", "SCALP", "DAY", "สั้น")):
        return "SHORT"
    if any(k in h for k in ("SWING", "MID", "MEDIUM", "กลาง")):
        return "SWING"
    if any(k in h for k in ("LONG", "INVEST", "ACCUM", "ยาว")):
        return "LONG"
    return "SWING"


def normalize_tf(tf: str, horizon: str) -> str:
    t = (tf or "").lower().replace(" ", "")
    t = {"15": "15m", "15min": "15m", "60m": "1h", "1hr": "1h", "240m": "4h", "4hr": "4h", "1day": "1d", "d": "1d"}.get(t, t)
    if t in TIMEFRAMES and t in HORIZONS[horizon]["tfs"]:
        return t
    return HORIZONS[horizon]["tfs"][-1] if horizon != "SHORT" else "1h"


def valid_levels(direction, sl, lo, e, hi, tp1, tp2) -> bool:
    if direction == "LONG":
        return sl < lo <= e <= hi < tp1 < tp2
    return sl > hi >= e >= lo > tp1 > tp2


def finalize_pick(ai: AIPick, top: pd.DataFrame, results: dict) -> dict | None:
    sym = re.sub(r"[/\-_ ]|USDT$", "", ai.symbol.upper().strip())
    row = top[top["base"] == sym]
    if row.empty:
        return None
    full = row.iloc[0]["symbol"]
    horizon = normalize_horizon(ai.horizon)
    tf = normalize_tf(ai.timeframe, horizon)
    d = results.get(full, {}).get(tf)
    if not d:
        return None
    direction = "SHORT" if "SHORT" in (ai.direction or "").upper() or "SELL" in (ai.direction or "").upper() else "LONG"
    if horizon == "LONG":
        direction = "LONG"  # ระยะยาว = สะสม (ถ้า AI ให้ตัวเลขฝั่ง Short จะถูกแทนด้วยแผนระบบด้านล่าง)
    if SPOT_ONLY and direction == "SHORT":
        return None  # Spot เปิด Short ไม่ได้ → ตัดสัญญาณระยะสั้น/กลางที่เป็น Short ทิ้ง
    warns = []
    lo, hi = min(ai.entry_low, ai.entry_high), max(ai.entry_low, ai.entry_high)
    lv = [ai.stop_loss, lo, ai.entry, hi, ai.tp1, ai.tp2]
    cur = d["close"]
    if not valid_levels(direction, *lv) or abs(ai.entry - cur) / cur > 0.15:
        # AI ให้ตัวเลขไม่สมเหตุสมผล → ใช้แผนของระบบ
        p = d["plan"] if direction == d["direction"] else build_plan(d["df"].iloc[-1], tf, direction)
        lv = [p["sl"], p["entry_low"], p["entry"], p["entry_high"], p["tp1"], p["tp2"]]
        warns.append("ใช้ตัวเลขจากแผนระบบ (ตัวเลขของ AI ไม่ผ่านการตรวจ)")
    sl, lo, e, hi, tp1, tp2 = lv
    risk = abs(e - sl)
    sys_wr = d["sys_wr"]
    wr = int(np.clip(ai.win_rate, max(5, sys_wr - 10), min(85, sys_wr + 10)))
    if direction != d["signal"]:
        warns.append(f"สัญญาณ TF {TF_LABEL[tf]} ยังเป็นขาลง — ซื้อแบบตั้ง Limit ที่โซนล่างเท่านั้น อย่าไล่ราคา"
                     if SPOT_ONLY else f"สวนทางสัญญาณระบบ TF {TF_LABEL[tf]} ({d['signal']})")
    gap = (e - cur) / cur * 100
    if abs(gap) > 5:
        warns.append(f"Entry ห่างราคาปัจจุบัน {gap:+.1f}% (ตั้ง Limit order รอ)")
    pct = lambda x: (x - e) / e * 100  # noqa: E731
    return {
        "symbol": sym, "pair": full, "horizon": horizon, "timeframe": tf, "direction": direction,
        "current": cur, "entry": e, "entry_low": lo, "entry_high": hi, "stop_loss": sl, "tp1": tp1, "tp2": tp2,
        "sl_pct": pct(sl), "tp1_pct": pct(tp1), "tp2_pct": pct(tp2),
        "rr": round(abs(tp1 - e) / risk, 2) if risk else None,
        "rr2": round(abs(tp2 - e) / risk, 2) if risk else None,
        "win_rate": wr, "ai_win_rate": int(ai.win_rate), "sys_wr": sys_wr,
        "bt_wr": d["bt"]["wr_raw"], "bt_n": d["bt"]["n"], "exp_r": d["bt"]["expectancy_r"],
        "bt_hold": d["bt"]["hold_txt"], "hold_period": ai.hold_period, "entry_trigger": ai.entry_trigger,
        "technical_reason": ai.technical_reason, "news_reason": ai.news_reason,
        "invalidation": ai.invalidation, "confidence": ai.confidence,
        "rsi": d["rsi"], "adx": d["adx"], "trend": d["trend"], "spot_status": d["spot_status"],
        "verdict": "-", "risk_pct": None, "risk_note": "", "warnings": warns,
    }


def merge_risk(picks: list, rr: RiskReport | None):
    if not rr:
        return
    rmap = {(re.sub(r"USDT$", "", r.symbol.upper()), normalize_horizon(r.horizon)): r for r in rr.reviews}
    for p in picks:
        r = rmap.get((p["symbol"], p["horizon"]))
        if not r:
            continue
        p["verdict"] = (r.verdict or "").upper()
        p["win_rate"] = int(np.clip(r.final_win_rate, max(5, p["win_rate"] - 10), min(85, p["win_rate"] + 10)))
        p["risk_pct"] = float(np.clip(r.risk_per_trade_pct, 0.1, 3.0))
        p["risk_note"] = r.note


# =============================================================================
# GEMINI WRAPPER
# =============================================================================
def find_replacement_model(client, err_msg: str, tried: set):
    for m in re.findall(r"models/([\w.\-]+)", err_msg):
        if m not in tried:
            return m
    try:
        names = []
        for m in client.models.list():
            name = (m.name or "").replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            if "flash" not in name or (actions and "generateContent" not in actions):
                continue
            if any(k in name for k in ("image", "tts", "live", "audio", "embedding", "thinking")):
                continue
            names.append(name)

        def rank(n):
            ver = [int(x) for x in re.findall(r"\d+", n)[:2]] or [0]
            return ("preview" not in n and "exp" not in n, "lite" not in n, ver)

        for n in sorted(names, key=rank, reverse=True):
            if n not in tried:
                return n
    except Exception:  # noqa: BLE001
        pass
    return None


def call_gemini(client, model: str, prompt: str, schema, temperature: float = 0.3, max_retries: int = 4):
    cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema,
                                      temperature=temperature)
    model = st.session_state.get("model_override", {}).get(model, model)
    requested, tried = model, {model}
    last_err, busy = None, {}
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
            parsed = getattr(resp, "parsed", None)
            if isinstance(parsed, schema):
                return parsed
            if isinstance(parsed, dict):
                return schema.model_validate(parsed)
            text = resp.text or ""
            if not text.strip():
                raise ValueError("Gemini ตอบกลับว่าง")
            return schema.model_validate_json(re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip()))
        except Exception as e:  # noqa: BLE001
            last_err, msg = e, str(e)

            def switch(reason):
                new = find_replacement_model(client, msg if reason == "gone" else "", tried)
                if not new:
                    return None
                st.toast(f"{model} {'ใช้ไม่ได้แล้ว' if reason == 'gone' else 'คนใช้เยอะ/โควต้าเต็ม'} → สลับไปใช้ {new}")
                st.session_state.setdefault("model_override", {})[requested] = new
                st.session_state["active_model"] = new
                tried.add(new)
                return new

            if ("404" in msg or "NOT_FOUND" in msg) and len(tried) < 6:
                new = switch("gone")
                if new:
                    model, attempt = new, attempt - 1
                    continue
            if any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "503", "500", "UNAVAILABLE",
                                       "INTERNAL", "overloaded", "high demand", "DEADLINE")):
                busy[model] = busy.get(model, 0) + 1
                if busy[model] >= 2 and len(tried) < 6:
                    new = switch("busy")
                    if new:
                        model = new
                        continue
                wait = 10 * busy[model]
                st.toast(f"Gemini ไม่ว่างชั่วคราว — รอ {wait} วินาที")
                time.sleep(wait)
                continue
            if isinstance(e, ValueError):
                time.sleep(3)
                continue
            raise
    raise RuntimeError(f"Gemini ไม่ตอบหลังลอง {max_retries} ครั้ง: {last_err}")


# =============================================================================
# CHARTS
# =============================================================================
HZ_COLOR = {"SHORT": "#f5a623", "SWING": "#4a90e2", "LONG": "#7ed321"}


def price_chart(df: pd.DataFrame, p: dict) -> go.Figure:
    d = df.tail(120)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=d["open_time"], open=d["open"], high=d["high"], low=d["low"],
                                 close=d["close"], name="ราคา", increasing_line_color="#26a69a",
                                 decreasing_line_color="#ef5350"))
    for col, color in [("ema20", "#f5c542"), ("ema50", "#42a5f5"), ("ema200", "#ab47bc")]:
        fig.add_trace(go.Scatter(x=d["open_time"], y=d[col], mode="lines", name=col.upper(),
                                 line=dict(width=1.3, color=color)))
    fig.add_hrect(y0=p["entry_low"], y1=p["entry_high"], fillcolor="#2196f3", opacity=0.15, line_width=0)
    for name, y, color in [("Entry", p["entry"], "#2196f3"), ("SL", p["stop_loss"], "#ef5350"),
                           ("TP1", p["tp1"], "#66bb6a"), ("TP2", p["tp2"], "#2e7d32")]:
        fig.add_hline(y=y, line_dash="dash", line_color=color, line_width=1.4,
                      annotation_text=f"{name} {fmt_price(y)}", annotation_position="top left",
                      annotation_font_color=color)
    fig.update_layout(height=430, margin=dict(l=8, r=8, t=40, b=8), xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.1, x=0),
                      title=f"{p['symbol']}/USDT · {TF_LABEL[p['timeframe']]}")
    return fig


def win_rate_chart(picks: list) -> go.Figure:
    df = pd.DataFrame([{"เหรียญ": f"{p['symbol']} ({TF_LABEL[p['timeframe']]})", "Win Rate %": p["win_rate"],
                        "Backtest %": round(p["bt_wr"], 0) if not np.isnan(p["bt_wr"]) else 0,
                        "ระยะ": p["horizon"]} for p in picks]).sort_values("Win Rate %")
    fig = px.bar(df, x="Win Rate %", y="เหรียญ", color="ระยะ", orientation="h", text="Win Rate %",
                 color_discrete_map=HZ_COLOR, range_x=[0, 100])
    fig.add_scatter(x=df["Backtest %"], y=df["เหรียญ"], mode="markers", name="Backtest",
                    marker=dict(symbol="diamond", size=11, color="white", line=dict(color="black", width=1)))
    fig.add_vline(x=50, line_dash="dot", line_color="gray")
    fig.update_layout(height=max(280, 46 * len(df)), margin=dict(l=8, r=8, t=10, b=8),
                      legend=dict(orientation="h", y=-0.15))
    return fig


def heatmap_chart(top: pd.DataFrame, results: dict) -> go.Figure:
    coins, z = [], []
    for _, r in top.iterrows():
        tfs = results.get(r["symbol"], {})
        if tfs:
            coins.append(r["base"])
            z.append([tfs.get(tf, {}).get("score", np.nan) for tf in TIMEFRAMES])
    fig = px.imshow(np.array(z, dtype=float), x=[TF_LABEL[t] for t in TIMEFRAMES], y=coins,
                    color_continuous_scale="RdYlGn", zmin=-3, zmax=3, text_auto=".1f", aspect="auto")
    fig.update_layout(height=max(360, 24 * len(coins)), margin=dict(l=8, r=8, t=10, b=8),
                      coloraxis_colorbar=dict(title="Score"))
    return fig


# =============================================================================
# UI — SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    gemini_key = st.text_input("GEMINI_API_KEY", value=get_secret("GEMINI_API_KEY"), type="password",
                               help="สร้างฟรีที่ aistudio.google.com/apikey")
    model_choice = st.selectbox("Gemini Model", GEMINI_MODELS, index=0)
    if model_choice.startswith("อื่นๆ"):
        model_choice = st.text_input("ชื่อโมเดล", value="gemini-3.8-flash")
    use_risk = st.toggle("ใช้ Risk Manager Agent", value=True, help="+1 การเรียก Gemini ต่อการสแกน")
    SPOT_ONLY = st.toggle("เทรด Spot อย่างเดียว (ไม่มี Short)", value=True,
                          help="ปิดถ้าเทรด Futures และต้องการสัญญาณ Short ด้วย")

    st.divider()
    st.subheader("🪙 เหรียญที่จะสแกน")
    coin_mode = st.radio("เลือกแบบ", ["Watchlist", "Top ตาม Volume อัตโนมัติ"], horizontal=True)
    watchlist, n_top, n_disc = None, 30, 0
    if coin_mode == "Watchlist":
        wl_text = st.text_area("รายชื่อเหรียญ", value=get_secret("WATCHLIST") or DEFAULT_WATCHLIST, height=150,
                               help="คั่นด้วยเว้นวรรค/จุลภาค/ขึ้นบรรทัด เช่น BTC ETH SOL")
        watchlist = parse_watchlist(wl_text)[:MAX_WATCHLIST]
        n_disc = st.slider("🔍 Discovery: เพิ่มเหรียญ Volume สูงสุดที่ไม่อยู่ในลิสต์", 0, 10, 3)
        st.caption(f"Watchlist {len(watchlist)} เหรียญ + Discovery {n_disc}")
    else:
        n_top = st.slider("จำนวนเหรียญ Top Volume", 10, 40, 30)
    n_picks = st.slider("จำนวนเหรียญที่ให้ AI คัด", 5, 10, 8)

    st.divider()
    st.subheader("💰 ขนาดไม้ (Position Size)")
    capital = st.number_input("เงินทุน (USDT)", min_value=10.0, value=1000.0, step=100.0)
    risk_pct_user = st.slider("ยอมเสี่ยงต่อไม้สูงสุด (% ของทุน)", 0.25, 3.0, 1.0, 0.25)

    st.divider()
    with st.expander("📰 CryptoPanic (ไม่บังคับ)"):
        cp_token = st.text_input("auth_token", value=get_secret("CRYPTOPANIC_TOKEN"), type="password",
                                 help="แพลนฟรีถูกยกเลิกแล้ว เว้นว่างเพื่อใช้ RSS ฟรี")
        cp_plan = st.selectbox("plan", ["growth", "enterprise", "developer"], index=0)

    scan = st.button("🚀 Scan Market Now", type="primary", width="stretch")
    st.caption("⚠️ ไม่ใช่คำแนะนำการลงทุน · Win Rate = สถิติย้อนหลัง + การประเมิน ไม่ใช่การรับประกัน")

st.title("📈 Multi-Agent Crypto Scanner v2")
st.caption("Data → Quant (Backtest) → Macro → News (AI) → Strategist (AI) → Risk Manager (AI)")

# =============================================================================
# RUN PIPELINE
# =============================================================================
if scan:
    if not gemini_key:
        st.error("กรุณาใส่ GEMINI_API_KEY ใน Sidebar")
        st.stop()
    if watchlist is not None and len(watchlist) < 3:
        st.error("กรุณาใส่เหรียญใน Watchlist อย่างน้อย 3 เหรียญ")
        st.stop()
    client = genai.Client(api_key=gemini_key)
    st.session_state["active_model"] = st.session_state.get("model_override", {}).get(model_choice, model_choice)
    t0 = time.time()
    with st.status("กำลังสแกนตลาด...", expanded=True) as status:
        try:
            st.write("📊 Data + Quant Agent: ดึงราคา 4 TF คำนวณอินดิเคเตอร์ และ Backtest")
            prog = st.progress(0.0)
            top, results, host = run_data_and_quant(watchlist, n_top, n_disc, prog)
            st.write(f"✅ {len(top)} เหรียญ × {len(TIMEFRAMES)} TF ({KLINE_LIMIT} แท่ง/TF)")

            st.write("🌍 Macro Agent: เทรนด์ BTC + Fear & Greed")
            macro = macro_view(results, fetch_fear_greed())

            st.write("📰 News Agent: ดึงข่าวและวิเคราะห์ Sentiment")
            news_report, news_items, news_src = run_news_agent(client, model_choice, top["base"].tolist(),
                                                               cp_token, cp_plan)
            st.write(f"✅ ข่าว {len(news_items)} รายการ ({news_src})")

            st.write("🧠 Strategist Agent: คัดเหรียญ แบ่งระยะ วางแผนเข้า-ออก")
            strat = run_strategist(client, model_choice, top, results, macro, news_report, news_items, n_picks)
            picks = [p for p in (finalize_pick(a, top, results) for a in strat.picks) if p]
            # กันซ้ำ เหรียญ+ระยะเดียวกัน
            seen, uniq = set(), []
            for p in picks:
                if (p["symbol"], p["horizon"]) not in seen:
                    seen.add((p["symbol"], p["horizon"]))
                    uniq.append(p)
            picks = uniq

            risk_report = None
            if use_risk and picks:
                st.write("🛡️ Risk Manager Agent: ตรวจสอบแผนและความเสี่ยง")
                try:
                    risk_report = run_risk_manager(client, model_choice, picks, macro)
                    merge_risk(picks, risk_report)
                except Exception as e:  # noqa: BLE001
                    st.info(f"Risk Manager ไม่ตอบ ({str(e)[:100]}) → แสดงผลจาก Strategist")

            st.session_state["scan"] = {
                "time": datetime.now(timezone.utc), "top": top, "results": results, "macro": macro,
                "news_report": news_report, "news_items": news_items, "news_src": news_src,
                "strat": strat, "picks": picks, "risk": risk_report,
                "model": st.session_state.get("active_model", model_choice), "secs": time.time() - t0, "spot": SPOT_ONLY,
            }
            status.update(label=f"สแกนเสร็จ ✅ ({time.time() - t0:.0f} วินาที)", state="complete", expanded=False)
        except Exception as e:  # noqa: BLE001
            status.update(label="เกิดข้อผิดพลาด ❌", state="error")
            st.exception(e)
            st.stop()

data = st.session_state.get("scan")
if not data:
    st.info("ใส่ GEMINI_API_KEY แล้วกด **🚀 Scan Market Now** ใน Sidebar")
    with st.expander("📘 ระบบนี้ทำงานอย่างไร", expanded=True):
        st.markdown(
            "- **Data Agent** ดึงแท่งเทียน 1,000 แท่งต่อ TF (15m / 1H / 4H / 1D)\n"
            "- **Quant Agent** วางแผน Entry / SL / TP ทุกเหรียญ และ **Backtest** ว่า setup แบบเดียวกันในอดีต"
            " ไปถึง TP1 ก่อน SL กี่ % และใช้เวลากี่ชั่วโมง/วัน\n"
            "- **Macro Agent** ดูเทรนด์ BTC และ Fear & Greed\n"
            "- **News Agent** สรุป Sentiment ข่าวล่าสุดรายเหรียญ\n"
            "- **Strategist** รวมทุกแผนก คัดเหรียญ แบ่งระยะสั้น/กลาง/ยาว\n"
            "- **Risk Manager** ตรวจแผน ปรับ Win Rate และแนะนำขนาดไม้")
    st.stop()

# =============================================================================
# DASHBOARD
# =============================================================================
top, results, macro, picks = data["top"], data["results"], data["macro"], data["picks"]
nr, strat, rr = data["news_report"], data["strat"], data["risk"]
spot = data.get("spot", True)

# ขนาดไม้ (Spot: ซื้อได้ไม่เกินทุน ไม่มี Leverage)
for p in picks:
    rp = min(risk_pct_user, p["risk_pct"]) if p["risk_pct"] else risk_pct_user
    risk_amt = capital * rp / 100
    per_unit = abs(p["entry"] - p["stop_loss"])
    qty = risk_amt / per_unit if per_unit else 0
    pos = qty * p["entry"]
    p["capped"] = spot and pos > capital
    if p["capped"]:
        pos = capital
        risk_amt = pos / p["entry"] * per_unit
        rp = round(risk_amt / capital * 100, 2)
    p["pos_usdt"], p["risk_amt"], p["risk_used_pct"] = pos, risk_amt, rp
    p["qty"] = pos / p["entry"] if p["entry"] else 0

bkk = data["time"].astimezone(timezone(timedelta(hours=7)))
st.caption(f"สแกนล่าสุด {bkk:%d/%m/%Y %H:%M} (เวลาไทย) · โมเดล {data['model']} · ใช้เวลา {data['secs']:.0f} วินาที")

fng = macro["fng"]
m = st.columns(4)
m[0].metric("ภาวะตลาด (BTC 1D)", macro["regime"], f"4H: {macro['btc_4h']}", delta_color="off")
m[1].metric("Fear & Greed", f"{fng['value']} · {fng['label']}" if fng else "-",
            f"{fng['value'] - fng['prev']:+d} จากเมื่อวาน" if fng and fng["prev"] is not None else None)
m[2].metric("Sentiment ข่าว", nr.overall_sentiment if nr else "-", f"{nr.overall_score:+d}" if nr else None)
m[3].metric("สัญญาณที่คัด", f"{len(picks)} รายการ", f"จาก {len(top)} เหรียญ", delta_color="off")

with st.container(border=True):
    st.markdown(f"**🧠 ภาพรวมตลาด:** {strat.market_overview}")
    st.markdown(f"**⏱️ ระยะที่เหมาะตอนนี้:** {strat.best_horizon_now}")
    if rr:
        st.markdown(f"**🛡️ Risk Manager:** {rr.portfolio_note} · ไม่ควรเปิดพร้อมกันเกิน **{rr.max_concurrent_trades} ไม้**")

tab1, tab2, tab3, tab4 = st.tabs(["🎯 สัญญาณเทรด", "🗺️ ภาพรวมทุกเหรียญ", "📰 ข่าว & Sentiment", "📘 วิธีอ่าน"])

VERDICT_ICON = {"APPROVE": "✅", "CAUTION": "⚠️", "REJECT": "❌", "-": "•"}
HZ_SHORT = {"SHORT": "⚡ สั้น", "SWING": "🌊 กลาง", "LONG": "🏦 ยาว"}

with tab1:
    if not picks:
        st.warning("AI ไม่ได้คัดเหรียญ ลองสแกนใหม่")
    else:
        order = {"SHORT": 0, "SWING": 1, "LONG": 2}
        picks_sorted = sorted(picks, key=lambda p: (order[p["horizon"]], p["verdict"] == "REJECT", -p["win_rate"]))
        st.subheader("📋 ตารางสรุปสัญญาณ")
        table = pd.DataFrame([{
            "สถานะ": VERDICT_ICON.get(p["verdict"], "•"),
            "เหรียญ": p["symbol"], "ระยะ": HZ_SHORT[p["horizon"]], "TF": TF_LABEL[p["timeframe"]],
            "ทิศทาง": p["spot_status"] if spot else ("🟢 LONG" if p["direction"] == "LONG" else "🔴 SHORT"),
            "Win Rate": p["win_rate"],
            "ราคาตอนนี้": fmt_price(p["current"]),
            "โซนเข้า": f"{fmt_price(p['entry_low'])} – {fmt_price(p['entry_high'])}",
            "Entry": fmt_price(p["entry"]),
            "Stop Loss": f"{fmt_price(p['stop_loss'])} ({p['sl_pct']:+.1f}%)",
            "TP1": f"{fmt_price(p['tp1'])} ({p['tp1_pct']:+.1f}%)",
            "TP2": f"{fmt_price(p['tp2'])} ({p['tp2_pct']:+.1f}%)",
            "R:R": f"1:{p['rr']} / 1:{p['rr2']}",
            "ถือประมาณ": p["hold_period"],
            "Backtest": f"{p['bt_wr']:.0f}% (n={p['bt_n']})" if not np.isnan(p["bt_wr"]) else "-",
            "ซื้อ (USDT)": f"{p['pos_usdt']:,.0f}" + (" (เต็มทุน)" if p["capped"] else ""),
            "ความมั่นใจ": p["confidence"],
        } for p in picks_sorted])
        st.dataframe(table, width="stretch", hide_index=True, column_config={
            "Win Rate": st.column_config.ProgressColumn("Win Rate", min_value=0, max_value=100, format="%d%%"),
        })
        st.caption("✅ ผ่าน · ⚠️ ระวัง · ❌ Risk Manager ไม่แนะนำ | ซื้อ (USDT) = จำนวนเงินที่ควรซื้อ เพื่อให้ถ้าโดน SL "
                   f"ขาดทุนไม่เกิน {risk_pct_user}% ของทุน (หรือตาม Risk Manager ถ้าน้อยกว่า)"
                   + (" · โหมด Spot: ไม่มีสัญญาณ Short และไม่ใช้ Leverage" if spot else ""))

        st.subheader("📊 Win Rate เทียบ Backtest")
        st.plotly_chart(win_rate_chart(picks_sorted), width="stretch")

        for hz, cfg in HORIZONS.items():
            group = [p for p in picks_sorted if p["horizon"] == hz]
            st.markdown(f"### {cfg['label']}")
            st.caption(f"ระยะถือโดยทั่วไป: {cfg['hold']}")
            if not group:
                st.caption("ไม่มีสัญญาณในระยะนี้ตอนนี้")
                continue
            for p in group:
                icon = VERDICT_ICON.get(p["verdict"], "•")
                arrow = (f"Spot {p['spot_status']}" if spot
                         else ("🟢 LONG" if p["direction"] == "LONG" else "🔴 SHORT"))
                with st.expander(f"{icon} {p['symbol']}/USDT · {arrow} · TF {TF_LABEL[p['timeframe']]} · "
                                 f"Win {p['win_rate']}% · ถือ {p['hold_period']}", expanded=p is group[0]):
                    c = st.columns(6)
                    c[0].metric("Win Rate", f"{p['win_rate']}%", f"Backtest {p['bt_wr']:.0f}%" if not np.isnan(p['bt_wr']) else None, delta_color="off")
                    c[1].metric("Entry", fmt_price(p["entry"]), f"ตอนนี้ {fmt_price(p['current'])}", delta_color="off")
                    c[2].metric("Stop Loss", fmt_price(p["stop_loss"]), f"{p['sl_pct']:+.2f}%")
                    c[3].metric("TP1", fmt_price(p["tp1"]), f"{p['tp1_pct']:+.2f}%")
                    c[4].metric("TP2", fmt_price(p["tp2"]), f"{p['tp2_pct']:+.2f}%")
                    c[5].metric("R:R", f"1:{p['rr']}", f"TP2 1:{p['rr2']}", delta_color="off")
                    df_chart = results[p["pair"]][p["timeframe"]]["df"]
                    st.plotly_chart(price_chart(df_chart, p), width="stretch",
                                    key=f"chart_{p['symbol']}_{p['horizon']}")
                    left, right = st.columns(2)
                    with left:
                        st.markdown(f"**🎯 เงื่อนไขเข้า:** {p['entry_trigger']}")
                        st.markdown(f"**📈 เหตุผลเทคนิค:** {p['technical_reason']}")
                        st.markdown(f"**📰 ข่าว/Macro:** {p['news_reason']}")
                    with right:
                        st.markdown(f"**⏱️ ระยะถือ:** {p['hold_period']} (สถิติย้อนหลัง {p['bt_hold']})")
                        st.markdown(f"**🚫 แผนเป็นโมฆะเมื่อ:** {p['invalidation']}")
                        if p["risk_note"]:
                            st.markdown(f"**🛡️ Risk Manager ({p['verdict']}):** {p['risk_note']}")
                        st.markdown(f"**💰 ซื้อ:** ~{p['pos_usdt']:,.0f} USDT ≈ {p['qty']:,.4g} {p['symbol']} "
                                    f"(ถ้าโดน SL ขาดทุน {p['risk_amt']:,.2f} USDT = {p['risk_used_pct']}% ของทุน)")
                        st.markdown(f"**🧾 วิธีตั้ง:** Limit Buy ในโซน {fmt_price(p['entry_low'])}–"
                                    f"{fmt_price(p['entry_high'])} → ขายครึ่งที่ TP1 {fmt_price(p['tp1'])} "
                                    f"→ เลื่อน SL มาที่ทุน → ขายที่เหลือที่ TP2 {fmt_price(p['tp2'])} · "
                                    f"ตั้ง Stop-Limit ขายที่ {fmt_price(p['stop_loss'])}")
                        if p["capped"]:
                            st.caption("SL แคบมากจนขนาดไม้เกินทุน → จำกัดไว้ที่ทุนทั้งหมด (Spot ไม่มี Leverage) "
                                       "ควรแบ่งทุนให้หลายเหรียญแทนการลงตัวเดียว")
                    st.caption(f"RSI {p['rsi']} · ADX {p['adx']} · Trend {p['trend']} · Backtest n={p['bt_n']} · "
                               f"Expectancy {p['exp_r']:+.2f}R · ความมั่นใจ {p['confidence']}")
                    for w in p["warnings"]:
                        st.warning(w)

with tab2:
    st.subheader("🌡️ Heatmap คะแนนเทคนิค (เขียว = Bullish, แดง = Bearish)")
    st.plotly_chart(heatmap_chart(top, results), width="stretch")

    rs = top.dropna(subset=["rs_btc_%"]).sort_values("rs_btc_%")
    if not rs.empty:
        st.subheader("💪 ความแข็งแรงเทียบ BTC (30 วัน)")
        fig = px.bar(rs, x="rs_btc_%", y="base", orientation="h", color="rs_btc_%",
                     color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                     labels={"rs_btc_%": "ผลตอบแทนเหนือ BTC (%)", "base": ""})
        fig.update_layout(height=max(320, 22 * len(rs)), margin=dict(l=8, r=8, t=10, b=8))
        st.plotly_chart(fig, width="stretch")

    st.subheader("📑 แผนเทรดของระบบ ทุกเหรียญ")
    hz_sel = st.radio("ระยะ", ["15m (Scalp)", "1H (Day)", "4H (Swing)", "1D (ยาว)"], horizontal=True, index=2)
    tf_sel = {"15m (Scalp)": "15m", "1H (Day)": "1h", "4H (Swing)": "4h", "1D (ยาว)": "1d"}[hz_sel]
    rows = []
    for _, r in top.iterrows():
        d = results.get(r["symbol"], {}).get(tf_sel)
        if not d:
            continue
        p, bt = d["plan"], d["bt"]
        e = p["entry"]
        rows.append({
            "เหรียญ": r["base"], "กลุ่ม": r["source"],
            "สัญญาณ": d["spot_status"] if spot else ("🟢 LONG" if d["direction"] == "LONG" else "🔴 SHORT"),
            "Score": d["score"], "Trend": d["trend"], "RSI": d["rsi"], "ADX": d["adx"],
            "Win Rate ระบบ": d["sys_wr"],
            "Backtest": f"{bt['wr_raw']:.0f}% (n={bt['n']})" if bt["n"] else "-",
            "Exp (R)": round(bt["expectancy_r"], 2) if bt["n"] else None,
            "Entry": fmt_price(e), "SL": f"{fmt_price(p['sl'])} ({(p['sl'] - e) / e * 100:+.1f}%)",
            "TP1": f"{fmt_price(p['tp1'])} ({(p['tp1'] - e) / e * 100:+.1f}%)",
            "TP2": f"{fmt_price(p['tp2'])} ({(p['tp2'] - e) / e * 100:+.1f}%)",
            "ถือ (สถิติ)": bt["hold_txt"],
            "RS vs BTC": round(r["rs_btc_%"], 1) if pd.notna(r["rs_btc_%"]) else None,
        })
    plan_df = pd.DataFrame(rows).sort_values("Win Rate ระบบ", ascending=False)
    st.dataframe(plan_df, width="stretch", hide_index=True, column_config={
        "Score": st.column_config.ProgressColumn("Score", min_value=-3, max_value=3, format="%.2f"),
        "Win Rate ระบบ": st.column_config.ProgressColumn("Win Rate ระบบ", min_value=0, max_value=100, format="%.0f%%"),
        "RS vs BTC": st.column_config.NumberColumn("RS vs BTC", format="%+.1f%%"),
    })
    st.caption("ตารางนี้คำนวณด้วยสูตรล้วน (ไม่ผ่าน AI) ใช้ดูเหรียญที่ AI ไม่ได้คัดได้ด้วย")

with tab3:
    st.caption(f"แหล่งข่าว: {data['news_src']}")
    if nr:
        st.markdown(f"**สรุป:** {nr.summary_th}")
        if nr.key_themes:
            st.markdown("\n".join(f"- {t}" for t in nr.key_themes))
        if nr.coins:
            sdf = pd.DataFrame([c.model_dump() for c in nr.coins]).sort_values("score")
            fig = px.bar(sdf, x="score", y="symbol", orientation="h", color="score", range_x=[-100, 100],
                         color_continuous_scale="RdYlGn", color_continuous_midpoint=0, hover_data=["reason"],
                         labels={"score": "Sentiment", "symbol": ""})
            fig.update_layout(height=max(280, 30 * len(sdf)), margin=dict(l=8, r=8, t=10, b=8))
            st.plotly_chart(fig, width="stretch")
            st.dataframe(sdf.sort_values("score", ascending=False), width="stretch", hide_index=True)
    if data["news_items"]:
        st.markdown("#### พาดหัวข่าวล่าสุด")
        st.dataframe(pd.DataFrame(data["news_items"])[["time", "source", "title", "coins", "url"]],
                     width="stretch", hide_index=True, column_config={"url": st.column_config.LinkColumn("ลิงก์")})

with tab4:
    st.markdown(f"""
**ระยะเวลา**
| ระยะ | TF เข้าเทรด | ถือโดยประมาณ |
|---|---|---|
| ⚡ Scalping | 15m | {fmt_hours(1)} – {fmt_hours(MAX_HOLD['15m'] * 0.25)} |
| ⚡ Day Trade | 1H | หลายชั่วโมง – {fmt_hours(MAX_HOLD['1h'])} |
| 🌊 Swing | 4H | 3 – 14 วัน |
| 🏦 ลงทุน/สะสม | 1D | หลายสัปดาห์ – หลายเดือน |

**คอลัมน์สำคัญ**
- **Win Rate**: โอกาสที่ราคาไปถึง TP1 ก่อนโดน SL เริ่มจาก *Backtest* ของเหรียญนั้นบน TF นั้น ปรับด้วยความสอดคล้องของ TF ใหญ่ ข่าว และ Risk Manager (ปรับได้ไม่เกิน ±10% จากระบบ)
- **Backtest (n)**: ผลจริงย้อนหลัง ~1,000 แท่ง ของ setup แนวโน้มเดียวกัน · n = จำนวนเทรดที่ใช้คำนวณ (n น้อย = เชื่อถือได้น้อย)
- **Expectancy (R)**: กำไรเฉลี่ยต่อเทรดในหน่วยความเสี่ยง เช่น +0.3R = เสี่ยง 100 ได้เฉลี่ย 30 · ติดลบ = setup นี้ขาดทุนในอดีต
- **โซนเข้า**: ทยอยซื้อในช่วงนี้ หรือตั้ง Limit ที่ขอบล่าง (LONG)
- **SL / TP1 / TP2**: SL = {SL_ATR['1h']}–{SL_ATR['1d']} × ATR · แนะนำปิดครึ่งหนึ่งที่ TP1 แล้วเลื่อน SL มาที่ทุน
- **ขนาดไม้**: คำนวณให้ถ้าโดน SL จะขาดทุนไม่เกิน % ที่ตั้งไว้ใน Sidebar
- **Score -3..+3**: คะแนนเทคนิคจาก EMA20/50/200, MACD, RSI, ADX

**โหมด Spot (ค่าเริ่มต้น)**
- 🟢 **ซื้อได้**: แนวโน้มขาขึ้นชัด (Score ≥ 1) เข้าตามโซนได้
- 🟡 **รอจังหวะ**: สัญญาณยังไม่ชัด รอเงื่อนไขในช่อง "เงื่อนไขเข้า" ก่อน
- 🔴 **ยังไม่ควรซื้อ**: ขาลง ถ้าจะสะสมระยะยาว ให้ตั้ง Limit ที่แนวรับด้านล่างเท่านั้น อย่าไล่ราคา
- Spot ไม่มี Stop Loss อัตโนมัติจากตำแหน่ง ต้องตั้ง **Stop-Limit (ขาย)** ไว้ที่ราคา SL เอง
- ช่วงตลาดขาลงทั้งตลาด การถือ USDT รอก็เป็นทางเลือกที่ดี

⚠️ สถิติในอดีตไม่รับประกันอนาคต ข่าวด่วนทำให้ราคาวิ่งผ่าน SL ได้ ควรเช็กกราฟจริงก่อนเข้าเทรดทุกครั้ง
""")
