import asyncio
import aiohttp
import requests
import pytz
from datetime import datetime, date

# --- LIQUIDITY GATE ---
# Price floor dropped to $1.00 to catch biotech/small-cap catalysts.
# Volume threshold is the real liquidity filter — price alone means nothing.
MIN_PRICE = 1.00
MIN_VOLUME = 50_000  # minimum shares traded to confirm real activity

# --- STICKINESS ---
# Tickers discovered during premarket (4am-9:30am) are locked onto the
# watchlist for the entire trading day. They don't get evicted by the
# regular-hours scan cycles.
PREMARKET_START_HOUR = 4
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16

# --- EXTENDED HOURS OPTIONS CLOSE TIMES ---
# These tickers trade options past the standard 4:00pm close on Public.com.
# Format: "TICKER": (close_hour, close_minute) in ET.
# Add any new tickers here as Public expands their extended hours offering —
# everything else in the bot will pick them up automatically.
EXTENDED_CLOSE_TIMES = {
    "SPY":  (16, 15),
    "QQQ":  (16, 15),
    "IWM":  (16, 15),
    "DIA":  (16, 15),
    "SPXL": (16, 15),
    "SPXS": (16, 15),
    "TQQQ": (16, 15),
    "SQQQ": (16, 15),
}


async def fetch_news_async(session, ticker):
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}&newsCount=3"
    try:
        async with session.get(url, timeout=3) as response:
            if response.status == 200:
                data = await response.json()
                news_items = data.get("news", [])
                if news_items:
                    return ticker, ". ".join([n["title"] for n in news_items])
    except Exception as e:
        print(f"⚠️ [SCREENER] Yahoo News Error on {ticker}: {e}")
    return ticker, "Neutral"


def _is_premarket(now):
    """Returns True if we are in the premarket power hours (4:00am - 9:30am ET)."""
    if now.hour < PREMARKET_START_HOUR:
        return False
    if now.hour > MARKET_OPEN_HOUR:
        return False
    if now.hour == MARKET_OPEN_HOUR and now.minute >= MARKET_OPEN_MINUTE:
        return False
    return True


def _is_market_hours(now):
    """Returns True if the regular session is open (9:30am - 4:00pm ET)."""
    if now.hour < MARKET_OPEN_HOUR:
        return False
    if now.hour == MARKET_OPEN_HOUR and now.minute < MARKET_OPEN_MINUTE:
        return False
    if now.hour >= MARKET_CLOSE_HOUR:
        return False
    return True


def ticker_is_tradeable(ticker, now):
    """
    Returns True if this specific ticker still has tradeable options right now.
    Extended hours tickers (SPY, QQQ, etc.) stay alive past 4:00pm until
    their actual close time. All other tickers follow the standard 4:00pm cut.
    To add a new extended-hours ticker, just drop it into EXTENDED_CLOSE_TIMES
    at the top of this file — nothing else needs to change.
    """
    # Always require market to be open first (9:30am minimum)
    if now.hour < MARKET_OPEN_HOUR:
        return False
    if now.hour == MARKET_OPEN_HOUR and now.minute < MARKET_OPEN_MINUTE:
        return False

    if ticker in EXTENDED_CLOSE_TIMES:
        close_hour, close_minute = EXTENDED_CLOSE_TIMES[ticker]
        if now.hour > close_hour:
            return False
        if now.hour == close_hour and now.minute >= close_minute:
            return False
        return True

    # Standard ticker — hard stop at 4:00pm
    return now.hour < MARKET_CLOSE_HOUR


def _passes_liquidity(stock_data):
    """
    Smarter liquidity gate. Requires price >= $1.00 AND meaningful volume.
    This catches biotech catalysts, FDA plays, and small-cap runners that
    the old $5.00 floor would have completely missed.
    """
    price = float(stock_data.get("price", 0))
    volume = float(stock_data.get("volume", 0))
    return price >= MIN_PRICE and volume >= MIN_VOLUME


async def _run_scan(ALPACA_KEY, ALPACA_SECRET, nlp, session_label):
    """
    Hits Alpaca for top movers, applies liquidity gate, runs FinBERT,
    and returns a list of approved tickers. Used by both premarket
    and regular-hours scan paths.
    """
    raw_movers = []
    print(f"📡 [SCREENER] {session_label} — pinging Alpaca for top movers...")

    headers = {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
    }

    try:
        res = requests.get(
            "https://data.alpaca.markets/v1beta1/screener/stocks/movers?top=50",
            headers=headers,
            timeout=10,
        )
        if res.status_code != 200:
            print(f"❌ [SCREENER] Alpaca API Error: {res.status_code} - {res.text}")
            return []

        data = res.json()
        gainers = data.get("gainers", [])
        losers = data.get("losers", [])
        print(f"✅ [SCREENER] Alpaca returned {len(gainers)} gainers and {len(losers)} losers.")

        for g in gainers:
            if _passes_liquidity(g):
                raw_movers.append((g["symbol"], "CALL", float(g.get("percent_change", 0))))

        for l in losers:
            if _passes_liquidity(l):
                raw_movers.append((l["symbol"], "PUT", abs(float(l.get("percent_change", 0)))))

    except Exception as e:
        print(f"❌ [SCREENER] Crash during Alpaca fetch: {e}")
        return []

    if not raw_movers:
        print(f"⚠️ [SCREENER] No valid movers found above ${MIN_PRICE} with volume >= {MIN_VOLUME:,}.")
        return []

    # Sort by magnitude of move, take top 40
    raw_movers.sort(key=lambda x: x[2], reverse=True)
    top_40 = raw_movers[:40]
    dynamic_targets = [sym for sym, _, _ in top_40]

    print(f"🧠 [SCREENER] FinBERT scanning news for {len(dynamic_targets)} tickers... (30-60s)")

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_news_async(session, t) for t in dynamic_targets]
        news_results = await asyncio.gather(*tasks)

    approved = []
    for ticker, text in news_results:
        if not text or text == "Neutral":
            approved.append(ticker)
            continue
        sentiment = nlp(text[:512])[0]
        if sentiment["label"] != "negative":
            approved.append(ticker)

    # Strip OTC tickers — no options liquidity
    approved = [t for t in approved if "OTC" not in t]
    return approved


async def dynamic_screener_loop(SYSTEM_STATE, ALPACA_KEY, ALPACA_SECRET, nlp, pos_lock):
    print("🔄 [SCREENER] Booting up... watching for premarket power hours.")

    # Tracks which calendar date the premarket scan has already run on,
    # so we don't re-run it multiple times in the same morning.
    premarket_scan_date = None

    while True:
        tz = pytz.timezone("US/Eastern")
        now = datetime.now(tz)
        today = now.date()

        if not SYSTEM_STATE.get("is_running"):
            await asyncio.sleep(30)
            continue

        if not ALPACA_KEY or not ALPACA_SECRET:
            print("❌ [SCREENER] Missing Alpaca keys in your .env file!")
            await asyncio.sleep(60)
            continue

        # ----------------------------------------------------------------
        # PREMARKET POWER HOUR SCAN (4:00am - 9:30am ET)
        # Runs ONCE per trading day. Finds institutional positioning and
        # high-conviction movers before the bell. These tickers are tagged
        # and LOCKED onto the watchlist for the entire trading day —
        # the regular-hours scan cannot evict them.
        # ----------------------------------------------------------------
        if _is_premarket(now) and premarket_scan_date != today:
            print(f"🌅 [PREMARKET] Power hour scan firing — {now.strftime('%H:%M')} ET")

            approved = await _run_scan(ALPACA_KEY, ALPACA_SECRET, nlp, "PREMARKET")

            if approved:
                with pos_lock:
                    # Store premarket finds separately so they survive the day
                    SYSTEM_STATE["premarket_watchlist"] = approved
                    # Also seed the dynamic watchlist immediately
                    existing = set(SYSTEM_STATE.get("dynamic_watchlist", []))
                    SYSTEM_STATE["dynamic_watchlist"] = list(existing | set(approved))
                    print(
                        f"🏦 [PREMARKET] {len(approved)} institutional-grade tickers locked in "
                        f"for the full trading day: {', '.join(approved[:10])}{'...' if len(approved) > 10 else ''}"
                    )
                premarket_scan_date = today
            else:
                print("⚠️ [PREMARKET] No qualifying premarket movers found. Will retry in 3 minutes.")

        # ----------------------------------------------------------------
        # REGULAR HOURS SCAN (9:30am - 4:00pm ET)
        # Runs every 3 minutes during the session. Refreshes the dynamic
        # watchlist with current movers — but ALWAYS re-merges the
        # premarket sticky tickers so they stay on the list all day.
        # ----------------------------------------------------------------
        elif _is_market_hours(now):
            approved = await _run_scan(ALPACA_KEY, ALPACA_SECRET, nlp, "MARKET HOURS")

            with pos_lock:
                # Pull today's premarket tickers (sticky all day)
                premarket_sticky = SYSTEM_STATE.get("premarket_watchlist", [])

                # Merge: fresh movers + premarket anchors, deduplicated
                combined = list(set(approved) | set(premarket_sticky))
                SYSTEM_STATE["dynamic_watchlist"] = combined

                print(
                    f"🎯 [DYNAMIC UPDATE] Watchlist refreshed — "
                    f"{len(approved)} fresh + {len(premarket_sticky)} premarket sticky = "
                    f"{len(combined)} total tickers active."
                )

        # ----------------------------------------------------------------
        # EXTENDED HOURS WINDOW (4:00pm - 4:15pm ET)
        # Standard market is closed but extended-hours tickers are still
        # live. Keep them on the watchlist and drop everything else so the
        # bot only fires on tickers that actually have open contracts.
        # ----------------------------------------------------------------
        elif now.hour == MARKET_CLOSE_HOUR and now.minute < 15:
            with pos_lock:
                premarket_sticky = SYSTEM_STATE.get("premarket_watchlist", [])
                current = SYSTEM_STATE.get("dynamic_watchlist", [])

                # Retain only tickers whose options window is still open
                extended_only = [
                    t for t in set(current) | set(premarket_sticky)
                    if ticker_is_tradeable(t, now)
                ]
                SYSTEM_STATE["dynamic_watchlist"] = extended_only

                if extended_only:
                    print(
                        f"⏳ [EXTENDED] Post-market window — keeping "
                        f"{len(extended_only)} extended-hours tickers active: "
                        f"{', '.join(extended_only)}"
                    )
                else:
                    print("⏳ [EXTENDED] Post-market window — no extended-hours tickers active.")

        # ----------------------------------------------------------------
        # OVERNIGHT / OFF-HOURS
        # Reset the premarket watchlist at midnight so tomorrow gets a
        # clean slate. Don't burn API calls when nothing is tradeable.
        # ----------------------------------------------------------------
        else:
            if now.hour == 0 and now.minute < 5:
                with pos_lock:
                    if SYSTEM_STATE.get("premarket_watchlist"):
                        print("🌙 [SCREENER] Midnight reset — clearing premarket sticky list for tomorrow.")
                        SYSTEM_STATE["premarket_watchlist"] = []
            await asyncio.sleep(60)
            continue

        # 3-minute cadence during active hours
        await asyncio.sleep(180)