import os
import time
import json
import logging
import requests
from datetime import datetime, timezone
from apscheduler.schedulers.blocking import BlockingScheduler
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# --- CONFIGURATION ---
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# FILES AND TIMERS
OUTPUT_FILE = "market_sentiment.json"
TEMP_FILE = "temp_sentiment.json"
SCAN_INTERVAL_MINS = 3

# This will hold our dynamic 100 tickers for the day
CURRENT_WATCHLIST = [] 

# --- LOCAL FINBERT INITIALIZATION ---
logger.info("⏳ Loading local FinBERT model with GPU Acceleration...")
model_name = "ProsusAI/finbert" 
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# device=0 tells the pipeline to use your dedicated GPU for lightning-fast processing
try:
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0)
    logger.info("✅ Local FinBERT Model loaded successfully on GPU!")
except Exception as e:
    logger.warning(f"⚠️ GPU load failed, falling back to CPU: {e}")
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def get_local_sentiment(text):
    return nlp(text)

# --- YAHOO FINANCE ADAPTERS (The Radar) ---
def fetch_premarket_movers():
    """Fetches Top Gainers and Losers from Yahoo Finance's hidden screener."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    dynamic_tickers = []
    logger.info("📡 Fetching Market Movers from Yahoo Screener...")
    
    try:
        # Get Top 50 Gainers
        res_gain = requests.get("https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=false&lang=en-US&region=US&scrIds=day_gainers&count=50", headers=headers, timeout=10)
        if res_gain.status_code == 200:
            quotes = res_gain.json()['finance']['result'][0]['quotes']
            dynamic_tickers.extend([q['symbol'] for q in quotes])
            
        # Get Top 50 Losers
        res_lose = requests.get("https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=false&lang=en-US&region=US&scrIds=day_losers&count=50", headers=headers, timeout=10)
        if res_lose.status_code == 200:
            quotes = res_lose.json()['finance']['result'][0]['quotes']
            dynamic_tickers.extend([q['symbol'] for q in quotes])
            
        # Clean up duplicates and ignore indexes (which start with ^)
        dynamic_tickers = list(set(dynamic_tickers))
        dynamic_tickers = [t for t in dynamic_tickers if "^" not in t and "=" not in t]
        
        logger.info(f"✅ Found {len(dynamic_tickers)} active movers for today's watchlist.")
    except Exception as e:
        logger.error(f"⚠️ Screener Error: {e}")
        
    return dynamic_tickers

def fetch_yahoo_news(ticker):
    """Fetches the latest news headlines for a specific ticker."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}&newsCount=5"
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            news_items = response.json().get('news', [])
            headlines = [item.get('title', '') for item in news_items]
            return ". ".join(headlines)
        return ""
    except:
        return ""

# --- THE JOBS ---
def update_daily_watchlist():
    """Runs every morning to build the hitlist."""
    global CURRENT_WATCHLIST
    logger.info("🌅 MORNING RADAR: Building today's top 100 hitlist...")
    movers = fetch_premarket_movers()
    
    if movers:
        CURRENT_WATCHLIST = movers
    else:
        logger.warning("⚠️ Using fallback watchlist.")
        CURRENT_WATCHLIST = ["NVDA", "TSLA", "MSTR", "COIN", "SPY", "QQQ"]

def scan_and_trade():
    """Runs every 3 minutes to score the watchlist."""
    global CURRENT_WATCHLIST
    
    if not CURRENT_WATCHLIST:
        logger.warning("Watchlist is empty. Updating now...")
        update_daily_watchlist()
        
    logger.info(f"--- Scanning sentiment for {len(CURRENT_WATCHLIST)} dynamic tickers ---")
    
    equity_sentiment = {}

    for ticker in CURRENT_WATCHLIST:
        news_text = fetch_yahoo_news(ticker)
        
        if not news_text:
            equity_sentiment[ticker] = {"sentiment": "neutral", "confidence": 0.0}
            continue
            
        try:
            safe_text = news_text[:1500] 
            result = get_local_sentiment(safe_text)
            
            label = result[0]['label']
            score = result[0]['score']
            
            # Penalize neutral news to prevent spam buying
            if label == "negative":
                final_score = -abs(score)
            elif label == "positive":
                final_score = abs(score)
            else:
                final_score = 0.0
                
            equity_sentiment[ticker] = {
                "sentiment": label,
                "confidence": round(final_score, 4),
                "timestamp": datetime.now(timezone.utc).isoformat() 
            }
            
        except Exception as e:
            equity_sentiment[ticker] = {"sentiment": "neutral", "confidence": 0.0}
            
        time.sleep(0.2)
            
    # --- THE ATOMIC SWAP ---
    logger.info(f"💾 Saving to temporary file: {TEMP_FILE}")
    try:
        with open(TEMP_FILE, 'w') as f:
            json.dump(equity_sentiment, f, indent=4)
            
        # Instantly swap the temp file to the live file
        os.replace(TEMP_FILE, OUTPUT_FILE)
        logger.info(f"🔄 Atomic Swap complete. {OUTPUT_FILE} is live for the Trading Bot.")
    except Exception as e:
        logger.error(f"❌ Failed to save JSON: {e}")

# --- SCHEDULER SETUP ---
if __name__ == "__main__":
    scheduler = BlockingScheduler()
    
    # UPGRADE 1: Refresh the Watchlist every 4 hours, 24/7.
    # This ensures the bot never trades on stale data and catches after-hours movers.
    scheduler.add_job(update_daily_watchlist, 'interval', hours=4)
    
    # UPGRADE 2: Run the FinBERT Scanner every 3 minutes, 24/7/365.
    scheduler.add_job(scan_and_trade, 'interval', minutes=SCAN_INTERVAL_MINS)
    
    logger.info("🐙 24/7 Dynamic Sentiment Engine starting up...")
    
    # Run initialization immediately on startup so you don't have to wait
    update_daily_watchlist()
    scan_and_trade() 
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Engine stopped by user.")