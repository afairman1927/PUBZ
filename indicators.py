import pandas as pd
import numpy as np
import pickle
import os
import logging

logger = logging.getLogger('INDICATORS')

live_buffers = {}
HMM_MODEL = None

if os.path.exists("hydra_hmm_model.pkl"):
    with open("hydra_hmm_model.pkl", "rb") as f:
        HMM_MODEL = pickle.load(f)

def engineer_features(df):
    """Audited 11-feature set. Standardized for training and live."""
    df = df.reset_index(drop=True).copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    grouped = df.groupby('symbol')
    
    # MACD, ROC, BB, EMA, RVOL, ATR (The Audited 11)
    for fast, slow in [(5,35), (12,26)]:
        m_col = f'MACD_{fast}_{slow}'; s_col = f'MACD_Sig_{fast}_{slow}'; h_col = f'MACD_Hist_{fast}_{slow}'
        df[m_col] = grouped['close'].transform(lambda x: x.ewm(span=fast, adjust=False).mean() - x.ewm(span=slow, adjust=False).mean())
        df[s_col] = grouped[m_col].transform(lambda x: x.ewm(span=9, adjust=False).mean())
        df[h_col] = df[m_col] - df[s_col]

    df['ROC_30'] = grouped['close'].transform(lambda x: x.pct_change(30)) * 100

    for period in [10, 20]:
        sma = grouped['close'].transform(lambda x: x.rolling(period).mean())
        std = grouped['close'].transform(lambda x: x.rolling(period).std())
        df[f'BB_Pct_B_{period}'] = (df['close'] - (sma - 2*std)) / (4*std + 1e-8)

    e9 = grouped['close'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    e21 = grouped['close'].transform(lambda x: x.ewm(span=21, adjust=False).mean())
    e50 = grouped['close'].transform(lambda x: x.ewm(span=50, adjust=False).mean())
    df['EMA_Dist_9_21'] = (e9 - e21) / (e21 + 1e-8) * 100
    df['EMA_Dist_21_50'] = (e21 - e50) / (e50 + 1e-8) * 100

    for period in [10, 20, 50]:
        df[f'RVOL_{period}'] = df['volume'] / (grouped['volume'].transform(lambda x: x.rolling(period).mean()) + 1e-8)

    df['HL'] = df['high'] - df['low']
    df['ATR_7'] = grouped['HL'].transform(lambda x: x.rolling(7).mean())

    expected = ['RVOL_10', 'RVOL_20', 'RVOL_50', 'EMA_Dist_9_21', 'EMA_Dist_21_50', 
                'BB_Pct_B_10', 'BB_Pct_B_20', 'MACD_Hist_5_35', 'MACD_Hist_12_26', 'ROC_30', 'ATR_7']
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=expected), expected

def get_live_features(ticker, o, h, l, c, v, t):
    """The Bridge: Ingests a full candle to prevent flatlining the HMM."""
    global live_buffers
    if ticker not in live_buffers: live_buffers[ticker] = []
    
    live_buffers[ticker].append({'ts_event': t, 'symbol': ticker, 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})
    if len(live_buffers[ticker]) > 200: live_buffers[ticker].pop(0)
    if len(live_buffers[ticker]) < 50:
        if len(live_buffers[ticker]) % 10 == 0:  # Log periodically to avoid spam
            logger.debug(f"[{ticker}] Buffering: {len(live_buffers[ticker])}/50 candles")
        return None
        
    df = pd.DataFrame(live_buffers[ticker])
    
    # HMM Gatekeeper check
    if HMM_MODEL is not None:
        ret = df['close'].pct_change().iloc[-1]
        rng = (df['high'].iloc[-1] - df['low'].iloc[-1]) / (df['low'].iloc[-1] + 1e-8)
        vol_chg = df['volume'].pct_change().iloc[-1]
        regime = HMM_MODEL.predict([[np.nan_to_num(ret), np.nan_to_num(rng), np.nan_to_num(vol_chg)]])[0]
        logger.debug(f"[{ticker}] HMM Regime: {regime} (Return: {ret:.4f}, Range: {rng:.4f}, VolChange: {vol_chg:.4f})")
        if regime == 0:
            logger.debug(f"[{ticker}] HMM GATEKEEPER: Choppy market (regime=0), rejecting signal")
            return "CHOP"

    df['ts_event'] = pd.to_datetime(df['ts_event'], unit='s' if isinstance(t, (int, float)) else None)
    df, base_feats = engineer_features(df)
    if df.empty:
        logger.warning(f"[{ticker}] Feature engineering resulted in empty DataFrame")
        return None
    
    df['day_of_week'] = df['ts_event'].dt.dayofweek
    df['hour_of_day'] = df['ts_event'].dt.hour
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    
    final_cols = base_feats + ['day_of_week', 'hour_of_day', 'is_monday', 'is_friday']
    return [df[final_cols].iloc[-1].values]