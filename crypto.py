import streamlit as st
import ccxt
import pandas as pd
import requests
from google import genai
from google.genai import types

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(page_title="Multi-Agent Crypto Scanner", layout="wide")

st.markdown("""
    <h2 style='text-align: center;'>📈 Multi-Agent Crypto Scanner</h2>
    <p style='text-align: center;'>Data Agent (Binance) ➔ News Agent (CryptoPanic) ➔ Master AI Agent (Gemini 1.5 Flash)</p>
""", unsafe_allow_html=True)

# Sidebar สำหรับตั้งค่า API Key
st.sidebar.markdown("### ⚙️ ตั้งค่า")
api_key = st.sidebar.text_input("GEMINI_API_KEY", type="password")
model_choice = st.sidebar.selectbox("Gemini Model", ["gemini-1.5-flash", "gemini-1.5-pro"])

st.sidebar.markdown("### 📰 News Agent")
cryptopanic_token = st.sidebar.text_input("CryptoPanic auth_token (ไม่บังคับ)", type="password")

st.sidebar.markdown("### 📊 พารามิเตอร์การสแกน")
top_n = st.sidebar.slider("จำนวนเหรียญที่สแกน (Top ตาม Volume)", 10, 50, 30)
selected_limit = st.sidebar.slider("จำนวนเหรียญที่ให้ AI คัด", 5, 15, 8)

scan_button = st.sidebar.button("🚀 Scan Market Now")

# ฟังก์ชันดึงข้อมูลราคาจาก Binance ผ่าน CCXT
def fetch_crypto_data(limit=30):
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        markets = exchange.load_markets()
        
        # ค้นหาคู่เหรียญที่เทรดกับ USDT
        symbols = [s for s in exchange.symbols if '/USDT' in s and not any(x in s for x in ['UP/', 'DOWN/', 'BULL/', 'BEAR/'])]
        
        data_list = []
        tickers = exchange.fetch_tickers(symbols)
        
        # จัดเรียงตาม Volume 24 ชม. สูงสุด
        sorted_tickers = sorted(
            [t for t in tickers.values() if 'quoteVolume' in t and t['quoteVolume'] is not None],
            key=lambda x: x['quoteVolume'],
            reverse=True
        )[:limit]
        
        for ticker in sorted_tickers:
            symbol = ticker['symbol']
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=50)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # คำนวณอินดิเคเตอร์เบื้องต้น (RSI, EMA)
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
                df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
                
                current_price = ticker['last']
                current_rsi = df['rsi'].iloc[-1]
                vol_24h = ticker['quoteVolume']
                
                data_list.append({
                    "symbol": symbol,
                    "price": current_price,
                    "rsi": round(current_rsi, 2) if not pd.isna(current_rsi) else 50.0,
                    "volume_24h": vol_24h
                })
            except Exception:
                continue
                
        return pd.DataFrame(data_list)
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการดึงข้อมูลตลาด: {e}")
        return pd.DataFrame()

# ฟังก์ชันดึงข่าวจาก CryptoPanic
def fetch_crypto_news():
    try:
        url = "https://cryptopanic.com/api/v1/posts/?auth_token=free&public=true"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            results = response.json().get('results', [])
            news_titles = [item['title'] for item in results[:10]]
            return " | ".join(news_titles)
        return "No recent news available."
    except Exception:
        return "Unable to fetch news at the moment."

# เมื่อกดปุ่มสแกน
if scan_button:
    if not api_key:
        st.error("❌ กรุณากรอก GEMINI_API_KEY ใน Sidebar ด่านซ้ายมือก่อนครับ")
    else:
        with st.spinner("🤖 ทีม AI Agents กำลังทำงาน: ดึงราคาตลาด ➔ วิเคราะห์ข่าวสาร ➔ คัดเลือกเหรียญ..."):
            # 1. Data Agent ทำงาน
            df_market = fetch_crypto_data(limit=top_n)
            
            if df_market.empty:
                st.error("❌ ไม่สามารถดึงข้อมูลตลาดจาก Binance ได้ในขณะนี้ ลองใหม่อีกครั้งครับ")
            else:
                # 2. News Agent ทำงาน
                news_summary = fetch_crypto_news()
                
                # แปลงข้อมูลตลาดเป็นข้อความส่งให้ AI วิเคราะห์
                market_summary_str = df_market.to_string(index=False)
                
                # 3. Master AI Agent (Gemini) วิเคราะห์และจัดพอร์ต
                prompt = f"""
                คุณคือ Master Portfolio Manager Agent ผู้เชี่ยวชาญด้านการเทรด Cryptocurrency ระดับมืออาชีพ
                ข้อมูลตลาดปัจจุบันจาก Binance (Top Volume):
                {market_summary_str}
                
                ข่าวสารล่าสุดในตลาดคริปโต:
                {news_summary}
                
                ภารกิจ:
                วิเคราะห์ข้อมูลตลาดและข่าวสารด้านบน คัดเลือกเหรียญที่มีโอกาสชนะสูงที่สุดจำนวน {selected_limit} เหรียญ 
                จัดแบ่งหมวดหมู่อย่างชัดเจนเป็น:
                1. ระยะสั้น (Short-term / Scalping)
                2. ระยะกลาง (Swing Trade)
                3. ระยะยาว (Investment / Accumulate)
                
                สำหรับแต่ละเหรียญที่เลือก ให้ระบุข้อมูลในรูปแบบตารางหรือหัวข้อชัดเจน ดังนี้:
                - ชื่อเหรียญ (Symbol)
                - หมวดหมู่ระยะเวลา
                - คะแนนความน่าจะเป็น / Win Rate (%)
                - จุดเข้าซื้อ (Entry Price)
                - จุดตัดขาดทุน (Stop Loss - SL)
                - เป้าหมายทำกำไร (Take Profit - TP1 / TP2)
                - เหตุผลวิเคราะห์เชิงเทคนิคและปัจจัยพื้นฐานสั้นๆ
                """
                
                try:
                    client = genai.Client(api_key=api_key)
                    response = client.models.generate_content(
                        model=model_choice,
                        contents=prompt,
                    )
                    
                    st.success("✨ วิเคราะห์เสร็จเรียบร้อย!")
                    st.markdown("### 📊 ผลการวิเคราะห์จาก Master AI Agent")
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error(f"❌ เกิดข้อผิดพลาดในการเรียกใช้งาน Gemini API: {e}")
