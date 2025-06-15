import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

def load_csv_sequence_with_sentiment(filepath, window_size=30, indicators=True, sentiment_map=None):
    df = pd.read_csv(filepath)
    df = df.dropna()
    df['date'] = pd.to_datetime(df['date'])

    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

    if indicators:
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['rsi'] = compute_rsi(df['close'], 14)
        df['atr'] = compute_atr(df['high'], df['low'], df['close'], 14)

    df = df.dropna()

    if sentiment_map:
        df['sentiment'] = df['date'].astype(str).map(sentiment_map)
        df['sentiment'] = df['sentiment'].fillna(0.0)
    else:
        df['sentiment'] = 0.0

    features = df.drop(columns=['date']).values
    if len(features) < window_size + 1:
        return torch.empty(0), torch.empty(0), None, None

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i+window_size])
        y.append(features[i+window_size][3])  # index 3 = close price

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)
    last_date = df['date'].iloc[window_size - 1 + len(y)]

    return X, y, scaler, last_date

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period).mean()
    ma_down = down.rolling(window=period).mean()
    rsi = 100 - (100 / (1 + ma_up / ma_down))
    return rsi

def compute_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr
