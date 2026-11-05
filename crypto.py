"""
Multi-Agent Crypto Scanner Dashboard
------------------------------------
Data Agent   : ccxt (Binance public market data) -> RSI, EMA50, EMA200, MACD, ATR บน 15m / 1h / 4h / 1D
News Agent   : CryptoPanic API (ถ้ามี token) + RSS ฟรี (CoinDesk / Cointelegraph / Decrypt) -> Gemini สรุป Sentiment
Master Agent : Gemini รวมข้อมูลเทคนิค + ข่าว -> คัด 5-10 เหรียญ แบ่ง ระยะสั้น / กลาง / ยาว
               พร้อม Win Rate, Entry, SL, TP1, TP2 และเหตุผล

รัน:  streamlit run app.py
⚠️ ไม่ใช่คำแนะนำการลงทุน  Win Rate เป็นค่าประเมินของ AI ไม่ใช่ผล Backtest
"""

import concurrent.futures as cf
import re
import time
from datetime import datetime, timedelta, timezone

import ccxt
import feedparser
import numpy as np
import pandas as pd
import requests
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(page_title="Multi-Agent Crypto Scanner", page_icon="📈", layout="wide")

TIMEFRAMES = ["15m", "1h", "4h", "1d"]
TF_LABEL = {"15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D"}
KLINE_LIMIT = 300  # พอสำหรับ EMA200

# data-api.binance.vision = endpoint ข้อมูลตลาดสาธารณะของ Binance
# ใช้ได้จากเซิร์ฟเวอร์ Streamlit Cloud (US) ที่ api.binance.com มักบล็อก (HTTP 451)
BINANCE_HOSTS = [
    "https://data-api.binance.vision/api/v3",
    "https://api.binance.com/api/v3",
]

EXCLUDE_BASES = {
    # stablecoins / fiat / wrapped / gold
    "USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "USDE", "USD1", "EUR", "EURI",
    "AEUR", "PYUSD", "RLUSD", "BFUSD", "XUSD", "USDS", "PAXG", "WBTC", "WBETH", "BNSOL",
}

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
    "ENA": ["ethena"], "WLD": ["worldcoin"], "FET": ["fetch.ai"], "HBAR": ["hedera"], "XLM": ["stellar"],
    "BCH": ["bitcoin cash"], "ETC": ["ethereum classic"], "FIL": ["filecoin"], "ICP": ["internet computer"],
    "TAO": ["bittensor"], "INJ": ["injective"], "SEI": ["sei"], "TIA": ["celestia"], "ONDO": ["ondo"],
    "HYPE": ["hyperliquid"], "ZEC": ["zcash"], "PUMP": ["pump.fun", "pumpfun"],
}

# Google ปิด gemini-2.5-flash สำหรับผู้ใช้ใหม่แล้ว และแนะนำ gemini-3.8-flash
# ถ้าโมเดลไหนใช้ไม่ได้ แอปจะสลับไปตัวที่ใช้ได้ให้อัตโนมัติ
GEMINI_MODELS = ["gemini-3.8-flash", "gemini-3.1-flash-lite", "อื่นๆ (พิมพ์เอง)"]

DEFAULT_WATCHLIST = (
    "BTC ETH SOL XRP XLM\n"
    "ADA NEAR ENA HYPE AAVE\n"
    "ZEC BNB PUMP UNI HBAR"
)
MAX_WATCHLIST = 40

HORIZONS = {
    "SHORT": "⚡ ระยะสั้น (Scalping / Day Trade)",
    "SWING": "🌊 ระยะกลาง (Swing Trade)",
    "LONG": "🏦 ระยะยาว (Investment / Accumulate)",
}


# =============================================================================
# HELPERS
# =============================================================================
def fmt_price(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    x = float(x)
    if x >= 1000:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:,.4f}"
    if x >= 0.01:
        return f"{x:.5f}"
    return f"{x:.8f}"


def get_secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, ""))
    except Exception:
        return ""


def safe_float(v, default=np.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# =============================================================================
# 1) DATA AGENT
# =============================================================================
def make_exchange(host: str) -> ccxt.binance:
    ex = ccxt.binance({"enableRateLimit": True, "timeout": 20000})
    ex.urls["api"]["public"] = host
    return ex


def connect_binance() -> tuple[ccxt.binance, str]:
    last_err = None
    for host in BINANCE_HOSTS:
        try:
            ex = make_exchange(host)
            ex.public_get_ping()
            return ex, host
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"เชื่อมต่อ Binance ไม่ได้: {last_err}")


def parse_watchlist(text: str) -> list[str]:
    """แปลงข้อความ เช่น 'BTC, ETH/USDT  solusdt' → ['BTC','ETH','SOL'] (ไม่ซ้ำ เรียงตามที่พิมพ์)"""
    out = []
    for tok in re.split(r"[\s,;]+", (text or "").upper()):
        tok = re.sub(r"[/\-_]", "", tok)
        if tok.endswith("USDT") and len(tok) > 4:
            tok = tok[:-4]
        if tok and tok not in out:
            out.append(tok)
    return out


def get_top_symbols(ex: ccxt.binance, n: int = 30, watchlist: list[str] | None = None) -> pd.DataFrame:
    """Top N คู่ USDT ตาม Quote Volume 24h หรือเฉพาะเหรียญใน watchlist (ถ้าระบุ)"""
    tickers = ex.public_get_ticker_24hr()
    now_ms = time.time() * 1000
    wanted = set(watchlist or [])
    rows = []
    for t in tickers:
        sym = t.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        base = sym[:-4]
        if wanted:
            if base not in wanted:
                continue
        elif base in EXCLUDE_BASES:
            continue
        price = safe_float(t.get("lastPrice"), 0)
        qvol = safe_float(t.get("quoteVolume"), 0)
        close_time = safe_float(t.get("closeTime"), 0)
        if price <= 0 or qvol <= 0 or now_ms - close_time > 86_400_000:  # ตัดคู่ที่หยุดเทรด
            continue
        rows.append({
            "symbol": sym,
            "base": base,
            "price": price,
            "change_24h_%": safe_float(t.get("priceChangePercent"), 0),
            "quote_volume_usdt": qvol,
        })
    if wanted:
        missing = [c for c in watchlist if c not in {r["base"] for r in rows}]
        if missing:
            st.warning(f"ไม่พบคู่ USDT ที่เทรดอยู่บน Binance: {', '.join(missing)} (ข้ามไป)")
    if not rows:
        raise RuntimeError("ไม่พบคู่เหรียญ USDT ที่ใช้ได้จาก Binance")
    df = pd.DataFrame(rows)
    if wanted:  # เรียงตามลำดับที่ผู้ใช้พิมพ์
        order = {c: i for i, c in enumerate(watchlist)}
        df = df.sort_values("base", key=lambda s: s.map(order))
    else:
        df = df.sort_values("quote_volume_usdt", ascending=False).head(n)
    return df.reset_index(drop=True)


def fetch_klines(ex: ccxt.binance, symbol: str, tf: str, limit: int = KLINE_LIMIT) -> pd.DataFrame:
    raw = ex.public_get_klines({"symbol": symbol, "interval": tf, "limit": limit})
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "qvol", "trades", "tb_base", "tb_quote", "ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(pd.to_numeric(df["open_time"]), unit="ms", utc=True)
    return df[["open_time", "open", "high", "low", "close", "volume"]]


# ---------- Indicators (คำนวณเองด้วย pandas ไม่ต้องพึ่ง TA-Lib) ----------
def calc_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(100.0).where(avg_gain.notna())


def calc_ema(close: pd.Series, n: int) -> pd.Series:
    return close.ewm(span=n, adjust=False).mean()


def calc_macd(close: pd.Series, fast=12, slow=26, signal=9):
    macd = calc_ema(close, fast) - calc_ema(close, slow)
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig


def calc_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift()
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def analyze_tf(df: pd.DataFrame) -> dict:
    df = df.dropna(subset=["high", "low", "close"]).reset_index(drop=True)
    if len(df) < 35:  # ต้องมีอย่างน้อย ~35 แท่งให้ MACD/RSI มีความหมาย
        raise ValueError("ข้อมูลแท่งเทียนน้อยเกินไป")
    close = df["close"]
    rsi = calc_rsi(close)
    ema50 = calc_ema(close, 50)
    ema200 = calc_ema(close, 200)
    macd, sig, hist = calc_macd(close)
    atr = calc_atr(df)

    last = float(close.iloc[-1])
    has200 = len(df) >= 200
    e50, e200 = float(ema50.iloc[-1]), (float(ema200.iloc[-1]) if has200 else np.nan)
    r = float(rsi.iloc[-1])
    h, h_prev = float(hist.iloc[-1]), float(hist.iloc[-2])

    # MACD cross ภายใน 3 แท่งล่าสุด
    cross = "none"
    recent = np.sign(hist.iloc[-4:].values)
    if len(recent) == 4 and recent[0] < 0 < recent[-1]:
        cross = "bullish"
    elif len(recent) == 4 and recent[0] > 0 > recent[-1]:
        cross = "bearish"

    if has200 and last > e50 > e200:
        trend = "UP"
    elif has200 and last < e50 < e200:
        trend = "DOWN"
    else:
        trend = "MIXED"

    # คะแนนเทคนิคแบบ rule-based (-3 ถึง +3) ใช้เป็นตัวช่วยจัดอันดับให้ AI
    score = 0.0
    score += {"UP": 1.5, "DOWN": -1.5, "MIXED": 0.0}[trend]
    score += 0.5 if last > e50 else -0.5
    score += 0.5 if h > 0 else -0.5
    score += 0.25 if h > h_prev else -0.25
    if 45 <= r <= 65:
        score += 0.25
    elif r >= 75:
        score -= 0.5
    elif r <= 30:
        score += 0.25  # มีโอกาสเด้ง

    hi20 = float(df["high"].iloc[-20:].max())
    lo20 = float(df["low"].iloc[-20:].min())

    return {
        "close": last,
        "rsi": round(r, 1),
        "ema50": e50,
        "ema200": e200,
        "macd": float(macd.iloc[-1]),
        "macd_signal": float(sig.iloc[-1]),
        "macd_hist": h,
        "macd_cross": cross,
        "atr": float(atr.iloc[-1]),
        "atr_pct": round(float(atr.iloc[-1]) / last * 100, 2),
        "trend": trend,
        "swing_high_20": hi20,
        "swing_low_20": lo20,
        "score": round(score, 2),
    }


def run_data_agent(n_coins: int, progress, watchlist: list[str] | None = None) -> tuple[pd.DataFrame, dict, str]:
    ex, host = connect_binance()
    top = get_top_symbols(ex, n_coins, watchlist)
    jobs = [(s, tf) for s in top["symbol"] for tf in TIMEFRAMES]
    results: dict[str, dict[str, dict]] = {s: {} for s in top["symbol"]}
    errors = 0

    def work(sym_tf):
        sym, tf = sym_tf
        return sym, tf, analyze_tf(fetch_klines(ex, sym, tf))

    done = 0
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(work, j) for j in jobs]
        for fut in cf.as_completed(futures):
            done += 1
            try:
                sym, tf, res = fut.result()
                results[sym][tf] = res
            except Exception:  # noqa: BLE001
                errors += 1
            progress.progress(done / len(jobs), text=f"📊 Data Agent: {done}/{len(jobs)} ชุดข้อมูล")

    # คะแนนตามระยะเวลา
    weights = {
        "SHORT": {"15m": 0.5, "1h": 0.5},
        "SWING": {"1h": 0.3, "4h": 0.7},
        "LONG": {"4h": 0.3, "1d": 0.7},
    }
    if not any(results.values()):
        raise RuntimeError("ดึงข้อมูลแท่งเทียนไม่สำเร็จเลย — Binance อาจบล็อกเซิร์ฟเวอร์นี้ ลองใหม่อีกครั้ง")
    for col in ["score_short", "score_swing", "score_long"] + [f"rsi_{tf}" for tf in TIMEFRAMES]:
        top[col] = np.nan
    for tf in TIMEFRAMES:
        top[f"trend_{tf}"] = pd.Series([None] * len(top), dtype="object")
    for i, row in top.iterrows():
        tfs = results.get(row["symbol"], {})
        for hz, w in weights.items():
            if all(tf in tfs for tf in w):
                top.loc[i, f"score_{hz.lower()}"] = round(sum(tfs[tf]["score"] * wt for tf, wt in w.items()), 2)
        for tf in TIMEFRAMES:
            if tf in tfs:
                top.loc[i, f"rsi_{tf}"] = tfs[tf]["rsi"]
                top.loc[i, f"trend_{tf}"] = tfs[tf]["trend"]
    if errors:
        st.warning(f"ดึงข้อมูลไม่สำเร็จ {errors} ชุด (ข้ามไป)")
    return top, results, host


# =============================================================================
# 2) NEWS AGENT
# =============================================================================
def fetch_cryptopanic(token: str, plan: str, limit: int = 50) -> list[dict]:
    url = f"https://cryptopanic.com/api/{plan}/v2/posts/"
    params = {"auth_token": token, "public": "true", "kind": "news", "regions": "en"}
    r = requests.get(url, params=params, timeout=15, headers={"User-Agent": "crypto-scanner/1.0"})
    r.raise_for_status()
    items = []
    for p in r.json().get("results", [])[:limit]:
        instruments = p.get("instruments") or p.get("currencies") or []
        coins = [c.get("code", "").upper() for c in instruments if isinstance(c, dict)]
        votes = p.get("votes") or {}
        items.append({
            "time": p.get("published_at", ""),
            "source": (p.get("source") or {}).get("title", "CryptoPanic"),
            "title": p.get("title", ""),
            "coins": coins,
            "votes_pos": votes.get("positive", 0),
            "votes_neg": votes.get("negative", 0),
            "url": p.get("original_url") or p.get("url", ""),
        })
    return items


def tag_coins(text: str, symbols: list[str]) -> list[str]:
    found = []
    low = text.lower()
    for s in symbols:
        if re.search(rf"\b{re.escape(s)}\b", text) or any(
            re.search(rf"\b{re.escape(n)}\b", low) for n in COIN_NAMES.get(s, [])
        ):
            found.append(s)
    return found


def fetch_rss_news(symbols: list[str], per_feed: int = 20) -> list[dict]:
    items = []
    for src, url in RSS_FEEDS.items():
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            feed = feedparser.parse(resp.content)
            for e in feed.entries[:per_feed]:
                title = e.get("title", "")
                items.append({
                    "time": e.get("published", ""),
                    "source": src,
                    "title": title,
                    "coins": tag_coins(title, symbols),
                    "votes_pos": 0,
                    "votes_neg": 0,
                    "url": e.get("link", ""),
                })
        except Exception:  # noqa: BLE001
            continue
    return items


class CoinSentiment(BaseModel):
    symbol: str = Field(description="ชื่อเหรียญ base เช่น BTC")
    sentiment: str = Field(description="Bullish, Bearish หรือ Neutral")
    score: int = Field(description="-100 (ลบมาก) ถึง 100 (บวกมาก)")
    reason: str = Field(description="เหตุผลสั้นๆ ภาษาไทย")


class NewsReport(BaseModel):
    overall_sentiment: str = Field(description="Bullish, Bearish หรือ Neutral")
    overall_score: int = Field(description="-100 ถึง 100")
    summary_th: str = Field(description="สรุปภาพรวมข่าวภาษาไทย 3-5 ประโยค")
    key_themes: list[str] = Field(description="ประเด็นสำคัญ 3-6 ข้อ ภาษาไทย")
    coins: list[CoinSentiment]


def run_news_agent(client, model, symbols, cp_token, cp_plan) -> tuple[NewsReport | None, list[dict], str]:
    news, source_used = [], ""
    if cp_token:
        try:
            news = fetch_cryptopanic(cp_token, cp_plan)
            source_used = "CryptoPanic"
        except Exception as e:  # noqa: BLE001
            st.warning(f"CryptoPanic ใช้ไม่ได้ ({e}) → สลับไปใช้ RSS")
    if not news:
        news = fetch_rss_news(symbols)
        source_used = "RSS (CoinDesk / Cointelegraph / Decrypt)"
    if not news:
        return None, [], "ไม่มีข่าว"

    headlines = "\n".join(
        f"- [{n['source']}] {n['title']}"
        + (f" | coins: {','.join(n['coins'])}" if n["coins"] else "")
        + (f" | votes +{n['votes_pos']}/-{n['votes_neg']}" if n["votes_pos"] or n["votes_neg"] else "")
        for n in news[:60]
    )
    prompt = f"""คุณคือ News Sentiment Agent ด้านคริปโต
วิเคราะห์พาดหัวข่าวล่าสุดด้านล่าง แล้วสรุป Sentiment ของตลาดโดยรวม และรายเหรียญ
ให้ประเมินเฉพาะเหรียญในรายการนี้ที่มีข่าวเกี่ยวข้อง (ทางตรงหรือทางอ้อม): {', '.join(symbols)}
ถ้าข่าวกระทบทั้งตลาด (เช่น Fed, ETF, กฎหมาย) ให้สะท้อนใน overall_score
ห้ามแต่งข่าวเพิ่มเอง ใช้เฉพาะข้อมูลที่ให้มา

ข่าว:
{headlines}
"""
    report = call_gemini(client, model, prompt, NewsReport, temperature=0.2)
    return report, news, source_used


# =============================================================================
# 3) MASTER AI AGENT
# =============================================================================
class TradePick(BaseModel):
    symbol: str = Field(description="base symbol เช่น SOL")
    horizon: str = Field(description="SHORT, SWING หรือ LONG")
    direction: str = Field(description="LONG หรือ SHORT")
    win_rate: int = Field(description="โอกาสชนะที่ประเมิน 0-100")
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    timeframe_basis: str = Field(description="TF หลักที่ใช้ตัดสินใจ เช่น 1h")
    reasoning: str = Field(description="เหตุผลวิเคราะห์ภาษาไทย 2-4 ประโยค อ้างอิงอินดิเคเตอร์และข่าว")
    risk_note: str = Field(description="ความเสี่ยง/เงื่อนไขที่ทำให้ setup ใช้ไม่ได้ ภาษาไทย")


class MasterReport(BaseModel):
    market_overview: str = Field(description="ภาพรวมตลาดภาษาไทย 3-5 ประโยค")
    picks: list[TradePick]


def build_market_table(top: pd.DataFrame, results: dict) -> str:
    lines = []
    for _, row in top.iterrows():
        tfs = results.get(row["symbol"], {})
        if not tfs:
            continue
        parts = [f"{row['base']} px={fmt_price(row['price'])} 24h={row['change_24h_%']:+.1f}% "
                 f"vol=${row['quote_volume_usdt'] / 1e6:,.0f}M "
                 f"S={row['score_short']} W={row['score_swing']} L={row['score_long']}"]
        for tf in TIMEFRAMES:
            d = tfs.get(tf)
            if not d:
                continue
            parts.append(
                f"[{TF_LABEL[tf]} RSI={d['rsi']} EMA50={fmt_price(d['ema50'])} EMA200={fmt_price(d['ema200'])} "
                f"MACDh={d['macd_hist']:.4g} cross={d['macd_cross']} trend={d['trend']} "
                f"ATR={fmt_price(d['atr'])}({d['atr_pct']}%) H20={fmt_price(d['swing_high_20'])} "
                f"L20={fmt_price(d['swing_low_20'])}]"
            )
        lines.append(" ".join(parts))
    return "\n".join(lines)


def run_master_agent(client, model, top, results, news_report: NewsReport | None, n_picks: int) -> MasterReport:
    market_table = build_market_table(top, results)
    if news_report:
        news_txt = (f"Overall: {news_report.overall_sentiment} ({news_report.overall_score})\n"
                    f"{news_report.summary_th}\n"
                    + "\n".join(f"- {c.symbol}: {c.sentiment} ({c.score}) {c.reason}" for c in news_report.coins))
    else:
        news_txt = "ไม่มีข้อมูลข่าว"

    prompt = f"""คุณคือ Master Trading Analyst ที่รวมผลจาก Data Agent และ News Agent
เวลาปัจจุบัน (UTC): {datetime.now(timezone.utc):%Y-%m-%d %H:%M}

=== ข้อมูลเทคนิค (Binance Spot, {len(top)} เหรียญ) ===
S/W/L = คะแนน rule-based ระยะสั้น/กลาง/ยาว (-3 ถึง +3)
{market_table}

=== Sentiment ข่าว ===
{news_txt}

=== งาน ===
คัดเลือกเหรียญที่น่าสนใจที่สุดรวม {n_picks} รายการ กระจายใน 3 กลุ่ม (แต่ละกลุ่มอย่างน้อย 1 รายการ):
- SHORT = Scalping/Day Trade ใช้ 15m และ 1H เป็นหลัก
- SWING = Swing Trade ใช้ 1H และ 4H เป็นหลัก
- LONG  = Investment/Accumulate ใช้ 4H และ 1D เป็นหลัก

กฎ:
1. ใช้เฉพาะเหรียญและตัวเลขจากตารางด้านบน ห้ามแต่งราคาเอง
2. Entry ต้องอยู่ใกล้ราคาปัจจุบัน (หรือโซนย่อตัวที่สมเหตุสมผล เช่น EMA50 / L20)
3. Stop Loss อิง ATR ของ TF หลัก (ประมาณ 1-2 ATR) หรือใต้/เหนือ swing low/high
4. LONG: SL < Entry < TP1 < TP2  |  SHORT: SL > Entry > TP1 > TP2  และ TP1 ควรให้ R:R อย่างน้อย 1:1.5
5. ระยะ LONG (ลงทุน) ให้ direction = LONG เท่านั้น
6. win_rate ต้องประเมินอย่างระมัดระวังจากความสอดคล้องของ TF, Trend, RSI, MACD และข่าว
   (ปกติอยู่ 45-75%, เกิน 80% เฉพาะ setup ที่สอดคล้องทุกด้านจริงๆ)
7. ถ้าข่าวขัดกับเทคนิค ให้ลด win_rate และระบุใน risk_note
8. reasoning และ risk_note เป็นภาษาไทย กระชับ อ้างอิงตัวเลขจริง
"""
    return call_gemini(client, model, prompt, MasterReport, temperature=0.3)


def normalize_horizon(h: str) -> str:
    h = (h or "").upper()
    if any(k in h for k in ("SHORT", "SCALP", "DAY", "สั้น")):
        return "SHORT"
    if any(k in h for k in ("SWING", "MID", "MEDIUM", "กลาง")):
        return "SWING"
    if any(k in h for k in ("LONG", "INVEST", "ACCUM", "ยาว")):
        return "LONG"
    return "SWING"


def normalize_direction(d: str) -> str:
    d = (d or "").upper()
    return "SHORT" if any(k in d for k in ("SHORT", "SELL", "ขาย")) else "LONG"


def validate_pick(p: TradePick, price_map: dict) -> dict:
    d = p.model_dump()
    d["symbol"] = re.sub(r"[/\-_ ]|USDT$", "", d["symbol"].upper().strip())
    d["horizon"] = normalize_horizon(d["horizon"])
    d["direction"] = normalize_direction(d["direction"])
    if d["horizon"] == "LONG":
        d["direction"] = "LONG"
    d["win_rate"] = int(max(0, min(100, d["win_rate"])))
    d["current"] = price_map.get(d["symbol"])
    e, sl, t1, t2 = d["entry"], d["stop_loss"], d["tp1"], d["tp2"]
    warns = []
    if d["direction"] == "LONG":
        ok = sl < e < t1 <= t2
        risk, reward = e - sl, t1 - e
    else:
        ok = sl > e > t1 >= t2
        risk, reward = sl - e, e - t1
    if not ok:
        warns.append("ลำดับ Entry/SL/TP ไม่สอดคล้องกับทิศทาง")
    d["rr"] = round(reward / risk, 2) if risk > 0 else None
    if d["current"]:
        gap = abs(e - d["current"]) / d["current"] * 100
        d["entry_gap_%"] = round(gap, 2)
        if gap > 10:
            warns.append(f"Entry ห่างราคาปัจจุบัน {gap:.1f}%")
    else:
        warns.append("ไม่พบเหรียญนี้ในข้อมูลที่สแกน")
    d["warnings"] = warns
    return d


# =============================================================================
# GEMINI WRAPPER (retry สำหรับ Free Tier 429)
# =============================================================================
def find_replacement_model(client, err_msg: str, tried: set) -> str | None:
    """หาโมเดลสำรองเมื่อโมเดลที่เลือกถูกยกเลิก (404 NOT_FOUND)"""
    # 1) โมเดลที่ Google แนะนำในข้อความ error เช่น "...use models/gemini-3.8-flash for..."
    for m in re.findall(r"models/([\w.\-]+)", err_msg):
        if m not in tried:
            return m
    # 2) ถามรายชื่อโมเดลที่คีย์นี้ใช้ได้จริง เลือก flash ล่าสุด
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

        def rank(n):  # ไม่ preview ดีกว่า, ไม่ lite ดีกว่า, เวอร์ชันใหม่กว่าดีกว่า
            ver = [int(x) for x in re.findall(r"\d+", n)[:2]] or [0]
            return ("preview" not in n and "exp" not in n, "lite" not in n, ver)

        for n in sorted(names, key=rank, reverse=True):
            if n not in tried:
                return n
    except Exception:  # noqa: BLE001
        pass
    return None


def call_gemini(client, model: str, prompt: str, schema, temperature: float = 0.3, max_retries: int = 4):
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=temperature,
    )
    # ถ้าเคยสลับโมเดลไปแล้วในการสแกนนี้ ใช้ตัวที่สลับไว้เลย
    model = st.session_state.get("model_override", {}).get(model, model)
    requested, tried = model, {model}
    last_err = None
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
                raise ValueError("Gemini ตอบกลับว่าง (อาจถูก safety filter หรือ token ไม่พอ)")
            # กันกรณีมี ```json ... ``` ครอบมา
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
            return schema.model_validate_json(text)
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e)
            # โมเดลถูกยกเลิก / ไม่มีสิทธิ์ใช้ → สลับโมเดลอัตโนมัติ (ไม่นับเป็นการลองซ้ำ)
            if ("404" in msg or "NOT_FOUND" in msg) and len(tried) < 6:
                new_model = find_replacement_model(client, msg, tried)
                if new_model:
                    st.toast(f"โมเดล {model} ใช้ไม่ได้แล้ว → สลับไปใช้ {new_model}")
                    st.session_state.setdefault("model_override", {})[requested] = new_model
                    st.session_state["active_model"] = new_model
                    tried.add(new_model)
                    model = new_model
                    attempt -= 1
                    continue
            if any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "503", "500", "UNAVAILABLE",
                                       "INTERNAL", "overloaded", "DEADLINE")):
                wait = 15 * attempt
                st.toast(f"Gemini ติด rate limit / ไม่ว่าง — รอ {wait} วินาทีแล้วลองใหม่")
                time.sleep(wait)
                continue
            if isinstance(e, ValueError):  # JSON ไม่สมบูรณ์ / ตอบว่าง (ValidationError เป็น ValueError)
                time.sleep(3)
                continue
            raise
    raise RuntimeError(f"Gemini ไม่ตอบหลังลอง {max_retries} ครั้ง: {last_err}")


# =============================================================================
# UI
# =============================================================================
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    gemini_key = st.text_input("GEMINI_API_KEY", value=get_secret("GEMINI_API_KEY"), type="password",
                               help="สร้างฟรีที่ aistudio.google.com/apikey")
    model_choice = st.selectbox("Gemini Model", GEMINI_MODELS, index=0)
    if model_choice.startswith("อื่นๆ"):
        model_choice = st.text_input("ชื่อโมเดล", value="gemini-3.8-flash")

    st.divider()
    st.subheader("📰 News Agent")
    cp_token = st.text_input("CryptoPanic auth_token (ไม่บังคับ)", value=get_secret("CRYPTOPANIC_TOKEN"),
                             type="password", help="ถ้าไม่ใส่หรือใช้ไม่ได้ จะใช้ RSS ฟรีแทน")
    cp_plan = st.selectbox("CryptoPanic plan", ["developer", "growth", "enterprise"], index=0)

    st.divider()
    st.subheader("🪙 เหรียญที่จะสแกน")
    coin_mode = st.radio("เลือกแบบ", ["เลือกเหรียญเอง (Watchlist)", "Top ตาม Volume อัตโนมัติ"], index=0)
    watchlist = None
    n_coins = 30
    if coin_mode.startswith("เลือก"):
        wl_text = st.text_area(
            "รายชื่อเหรียญ (คั่นด้วยเว้นวรรค / จุลภาค / ขึ้นบรรทัดใหม่)",
            value=get_secret("WATCHLIST") or DEFAULT_WATCHLIST, height=140,
            help="พิมพ์แค่ชื่อเหรียญ เช่น BTC ETH SOL หรือ BTCUSDT ก็ได้ ระบบจับคู่กับ USDT ให้เอง",
        )
        watchlist = parse_watchlist(wl_text)
        if len(watchlist) > MAX_WATCHLIST:
            st.warning(f"ใส่ได้สูงสุด {MAX_WATCHLIST} เหรียญ จะใช้ {MAX_WATCHLIST} ตัวแรก")
            watchlist = watchlist[:MAX_WATCHLIST]
        st.caption(f"เลือกไว้ {len(watchlist)} เหรียญ")
    else:
        n_coins = st.slider("จำนวนเหรียญที่สแกน (Top ตาม Volume)", 10, 30, 30)
    n_picks = st.slider("จำนวนเหรียญที่ให้ AI คัด", 5, 10, 7)
    scan = st.button("🚀 Scan Market Now", type="primary", width="stretch")
    st.caption("⚠️ ไม่ใช่คำแนะนำการลงทุน  Win Rate เป็นค่าประเมินของ AI ไม่ใช่ผล Backtest")

st.title("📈 Multi-Agent Crypto Scanner")
st.caption("Data Agent (Binance · ccxt) → News Agent (CryptoPanic / RSS) → Master AI Agent (Gemini)")

if scan:
    if not gemini_key:
        st.error("กรุณาใส่ GEMINI_API_KEY ใน Sidebar")
        st.stop()
    if watchlist is not None and len(watchlist) < 3:
        st.error("กรุณาใส่เหรียญใน Watchlist อย่างน้อย 3 เหรียญ")
        st.stop()
    client = genai.Client(api_key=gemini_key)
    st.session_state["active_model"] = st.session_state.get("model_override", {}).get(model_choice, model_choice)

    with st.status("กำลังสแกนตลาด...", expanded=True) as status:
        try:
            st.write("📊 Data Agent: ดึงราคาและคำนวณอินดิเคเตอร์")
            prog = st.progress(0.0)
            top, results, host = run_data_agent(n_coins, prog, watchlist)
            st.write(f"✅ ได้ข้อมูล {len(top)} เหรียญ × {len(TIMEFRAMES)} TF (จาก {host})")

            st.write("📰 News Agent: ดึงข่าวและวิเคราะห์ Sentiment")
            news_report, news_items, news_src = run_news_agent(
                client, model_choice, top["base"].tolist(), cp_token, cp_plan)
            st.write(f"✅ ข่าว {len(news_items)} รายการ จาก {news_src}")

            st.write("🧠 Master AI Agent: วิเคราะห์รวมและคัดเหรียญ")
            master = run_master_agent(client, model_choice, top, results, news_report, n_picks)
            price_map = dict(zip(top["base"], top["price"]))
            picks = [validate_pick(p, price_map) for p in master.picks]

            st.session_state["scan"] = {
                "time": datetime.now(timezone.utc),
                "top": top, "news_report": news_report, "news_items": news_items,
                "news_src": news_src, "overview": master.market_overview, "picks": picks,
                "model": st.session_state.get("active_model", model_choice),
            }
            status.update(label="สแกนเสร็จแล้ว ✅", state="complete", expanded=False)
        except Exception as e:  # noqa: BLE001
            status.update(label="เกิดข้อผิดพลาด ❌", state="error")
            st.exception(e)
            st.stop()

data = st.session_state.get("scan")
if not data:
    st.info("ใส่ GEMINI_API_KEY แล้วกด **🚀 Scan Market Now** ใน Sidebar เพื่อเริ่มสแกน")
    st.stop()

bkk = data["time"].astimezone(timezone(timedelta(hours=7)))
st.caption(f"สแกนล่าสุด: {bkk:%Y-%m-%d %H:%M} (เวลาไทย) · โมเดล {data['model']}")

nr = data["news_report"]
c1, c2, c3 = st.columns(3)
c1.metric("Market Sentiment (ข่าว)", nr.overall_sentiment if nr else "-",
          f"{nr.overall_score:+d}" if nr else None)
c2.metric("เหรียญที่สแกน", len(data["top"]))
c3.metric("เหรียญที่ AI คัด", len(data["picks"]))

st.subheader("🧠 ภาพรวมตลาดจาก Master Agent")
st.write(data["overview"])

tab_picks, tab_market, tab_news = st.tabs(["🎯 Top Picks", "📊 Market Scan", "📰 News Sentiment"])

with tab_picks:
    for hz, label in HORIZONS.items():
        group = sorted([p for p in data["picks"] if p["horizon"] == hz], key=lambda x: -x["win_rate"])
        st.markdown(f"### {label}")
        if not group:
            st.caption("ไม่มีเหรียญในกลุ่มนี้")
            continue
        for p in group:
            with st.container(border=True):
                arrow = "🟢 LONG" if p["direction"] == "LONG" else "🔴 SHORT"
                st.markdown(f"#### {p['symbol']}/USDT · {arrow} · TF {p['timeframe_basis']}")
                m = st.columns(7)
                m[0].metric("Win Rate", f"{p['win_rate']}%")
                m[1].metric("ราคาปัจจุบัน", fmt_price(p["current"]))
                m[2].metric("Entry", fmt_price(p["entry"]))
                m[3].metric("Stop Loss", fmt_price(p["stop_loss"]))
                m[4].metric("TP1", fmt_price(p["tp1"]))
                m[5].metric("TP2", fmt_price(p["tp2"]))
                m[6].metric("R:R (TP1)", f"1:{p['rr']}" if p["rr"] else "-")
                st.progress(p["win_rate"] / 100)
                st.markdown(f"**เหตุผล:** {p['reasoning']}")
                st.markdown(f"**ความเสี่ยง:** {p['risk_note']}")
                for w in p["warnings"]:
                    st.warning(w)

with tab_market:
    top = data["top"].copy()
    show_cols = ["base", "price", "change_24h_%", "quote_volume_usdt", "score_short", "score_swing",
                 "score_long"] + [c for tf in TIMEFRAMES for c in (f"rsi_{tf}", f"trend_{tf}") if c in top]
    top["quote_volume_usdt"] = (top["quote_volume_usdt"] / 1e6).round(1)
    st.dataframe(
        top[show_cols].rename(columns={"base": "Coin", "quote_volume_usdt": "Vol 24h ($M)"}),
        width="stretch", hide_index=True,
        column_config={
            "score_short": st.column_config.ProgressColumn("Short", min_value=-3, max_value=3, format="%.2f"),
            "score_swing": st.column_config.ProgressColumn("Swing", min_value=-3, max_value=3, format="%.2f"),
            "score_long": st.column_config.ProgressColumn("Long", min_value=-3, max_value=3, format="%.2f"),
            "change_24h_%": st.column_config.NumberColumn("24h %", format="%.2f%%"),
        },
    )
    st.caption("คะแนน -3 ถึง +3 คำนวณจาก Trend (EMA50/200), MACD และ RSI ของแต่ละ TF")

with tab_news:
    st.caption(f"แหล่งข่าว: {data['news_src']}")
    if nr:
        st.write(nr.summary_th)
        if nr.key_themes:
            st.markdown("\n".join(f"- {t}" for t in nr.key_themes))
        if nr.coins:
            st.dataframe(pd.DataFrame([c.model_dump() for c in nr.coins]),
                         width="stretch", hide_index=True)
    if data["news_items"]:
        st.markdown("#### พาดหัวข่าวล่าสุด")
        st.dataframe(
            pd.DataFrame(data["news_items"])[["time", "source", "title", "coins", "url"]],
            width="stretch", hide_index=True,
            column_config={"url": st.column_config.LinkColumn("ลิงก์")},
        )
