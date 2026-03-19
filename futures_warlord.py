import os, sys, time, threading, json, requests, warnings, pytz
import pandas as pd
import numpy as np
import xgboost as xgb
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv

# --- 📦 CUSTOM IMPORTS ---
from indicators import engineer_features
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

load_dotenv()
warnings.filterwarnings("ignore")

# --- 🚀 THE FUTURES CONFIG ---
TRADOVATE_URL = "https://demo.tradovateapi.com/v1" 
MNQ_SYMBOL = "MNQM6" # Micro Nasdaq June 2026 Contract
CONTRACT_ID = 0 # We will fetch this dynamically

class TradovateFutures:
    def __init__(self):
        self.user = os.getenv("TRADOVATE_USER").strip()
        self.password = os.getenv("TRADOVATE_PASS").strip()
        self.token = None

    def get_token(self):
        url = f"{TRADOVATE_URL}/auth/accesstokenrequest"
        # Using the standard "SampleApp" credentials from the Tradovate docs
        payload = {
            "name": self.user,
            "password": self.password,
            "appId": "Sample App",
            "appVersion": "1.0",
            "cid": 8, 
            "sec": "f03741b6-f634-48d6-9308-c8fb871150c2" 
        }
        try:
            res = requests.post(url, json=payload)
            data = res.json()
            if "accessToken" in data:
                self.token = data["accessToken"]
                print("✅ Tradovate API Connected via SampleApp.")
                return True
            else:
                # If this fails with "Incorrect username or password," 
                # your trial is officially locked out of the API.
                print(f"❌ API Access Denied: {data.get('errorText')}")
                return False
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            return False

# --- 🧠 MACRO SENTIMENT & AI ---
print("⏳ Loading FinBERT & XGBoost Brains...")
xgb_model = xgb.XGBClassifier()
try: xgb_model.load_model("warlord_friday_v1.json")
except: print("⚠️ XGB Model file missing!")

tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
bert_model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
nlp = pipeline("sentiment-analysis", model=bert_model, tokenizer=tokenizer, device=0)

def get_macro_sentiment():
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search?q=Nasdaq+100+Economy+Inflation+Fed&newsCount=3"
        res = requests.get(url, timeout=5).json()
        headlines = [n['title'] for n in res.get('news', [])]
        if not headlines: return "neutral"
        text = ". ".join(headlines)
        sentiment = nlp(text[:512])[0]
        return sentiment['label']
    except:
        return "neutral"

# --- 🛰️ THE SNIPER ENGINE ---
def futures_execution():
    broker = TradovateFutures()
    if not broker.get_token():
        return
    
    print(f"🎯 FUTURES SNIPER ACTIVATED. TARGET: {MNQ_SYMBOL}")
    print("🛡️ Rules: 20 Tick Stop Loss | 40 Tick Take Profit")
    
    tz = pytz.timezone('US/Eastern')
    
    while True:
        now = datetime.now(tz)
        
        # Futures trading hours check (Closed Friday 5PM to Sunday 6PM EST)
        if now.weekday() == 4 and now.hour >= 17:
            print("🛑 Market Closed for Weekend. Sleeping...")
            time.sleep(3600); continue
            
        try:
            # 1. Get 1-Minute Technicals using continuous NQ futures data from Yahoo
            df = yf.download("NQ=F", period="2d", interval="1m", progress=False).reset_index()
            if df.empty: time.sleep(10); continue
            
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df = df.rename(columns={"Datetime": "timestamp", "Open":"open", "High":"high", "Low":"low", "Close":"close", "Volume":"volume"})
            df['symbol'] = MNQ_SYMBOL
            
            # Run your custom 41 indicators
            latest_df, f_cols = engineer_features(df)
            latest = latest_df.tail(1)
            
            # Predict
            probs = xgb_model.predict_proba(latest[f_cols])
            prob_chop = float(probs[0][0])
            prob_up = float(probs[0][1])
            prob_down = float(probs[0][2])
            adx = float(latest['ADX_14'].iloc[-1])
            
            # 2. Get Macro Sentiment
            sentiment = get_macro_sentiment()
            
            print(f"\n📡 [1-MIN PULSE] {now.strftime('%H:%M:%S')} | {MNQ_SYMBOL} @ {latest['close'].iloc[-1]:.2f}")
            print(f"   👉 AI Call: {prob_up*100:.1f}% | Put: {prob_down*100:.1f}% | Chop: {prob_chop*100:.1f}%")
            print(f"   👉 ADX: {adx:.1f} | Macro Sentiment: {sentiment.upper()}")

            # 3. Paper Trading Simulation Logic (Log the execution before we wire the real POST requests)
            if adx > 20 and prob_chop < 0.40:
                if prob_up > 0.60 and sentiment != "negative":
                    print(f"🚀 [PAPER FIRE] LONG {MNQ_SYMBOL} | Stop: -20 Ticks | Target: +40 Ticks")
                    time.sleep(300) # Cooldown after a trade
                elif prob_down > 0.60 and sentiment != "positive":
                    print(f"🩸 [PAPER FIRE] SHORT {MNQ_SYMBOL} | Stop: -20 Ticks | Target: +40 Ticks")
                    time.sleep(300) # Cooldown after a trade
            
        except Exception as e:
            print(f"⚠️ Loop Error: {e}")
            
        time.sleep(60)

if __name__ == "__main__":
    futures_execution()