import os
import uuid
import requests
import time
from dotenv import load_dotenv

load_dotenv()

class PublicAPIClient:
    def __init__(self):
        # Your .env holds the SECRET key
        self.secret_key = os.getenv("PUBLIC_API_SECRET_KEY") or os.getenv("PUBLIC_API_KEY")
        self.account_id = os.getenv("PUBLIC_ACCOUNT_ID")
        self.base_url = "https://api.public.com"
        
        # Token Management System
        self._access_token = None
        self._token_expiry = 0

    def _get_active_token(self):
        """Automatically fetches a new token if the current one is expired."""
        if self._access_token and time.time() < self._token_expiry:
            return self._access_token
            
        url = f"{self.base_url}/userapiauthservice/personal/access-tokens"
        payload = {
            "validityInMinutes": 60, 
            "secret": self.secret_key
        }
        
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                self._access_token = response.json().get("accessToken")
                # Set expiry to 55 minutes from now so it refreshes BEFORE dying
                self._token_expiry = time.time() + (55 * 60) 
                return self._access_token
            else:
                print(f"❌ Token Exchange Failed: {response.text}")
                return None
        except Exception as e:
            print(f"⚠️ Token Exchange Error: {e}")
            return None

    def place_order(self, symbol, side, qty, limit_price=None):
        url = f"{self.base_url}/userapigateway/trading/{self.account_id}/order"
        
        # Grab the active session token
        token = self._get_active_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        # Route standard tickers to EQUITY and OSI strings to OPTION
        instrument_type = "OPTION" if len(str(symbol)) > 10 else "EQUITY"

        payload = {
            "orderId": str(uuid.uuid4()), 
            "instrument": {
                "symbol": str(symbol),
                "type": instrument_type
            },
            "orderSide": side.upper(),
            "quantity": str(qty),
            "expiration": {
                "timeInForce": "DAY"
            }
        }

        # THE FINAL FIX: Options require an explicit "OPEN" or "CLOSE" declaration
        if instrument_type == "OPTION":
            # If the bot is Buying, it's opening a position. If Selling, it's closing.
            payload["openCloseIndicator"] = "OPEN" if side.upper() == "BUY" else "CLOSE"

        if limit_price:
            payload["orderType"] = "LIMIT"
            payload["limitPrice"] = str(limit_price)
        else:
            payload["orderType"] = "MARKET"

        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                print(f"✅ SUCCESS: {side} {qty} {symbol} ({instrument_type})")
                return response.json()
            else:
                print(f"❌ API ERROR {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"⚠️ Connection Failed: {e}")
            return None

    def get_account_portfolio_v2(self):
        """Used by the Bot and Liquidator to read your positions."""
        url = f"{self.base_url}/userapigateway/trading/{self.account_id}/portfolio/v2"
        token = self._get_active_token()
        headers = {"Authorization": f"Bearer {token}"}
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
            return {}
        except:
            return {}

    def get_account_info(self):
        # Stub to pass the bot's initial _safe_login check
        return self.get_account_portfolio_v2()

    def authenticate(self):
        """Pings the token server to verify everything works on startup."""
        if self.secret_key and self.account_id and self._get_active_token():
            return True
        print("❌ Auth Failed: Check .env keys.")
        return False