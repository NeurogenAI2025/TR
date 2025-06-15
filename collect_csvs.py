import requests
import pandas as pd
import os
import time

def get_top_coins(limit=100):
    coins = []
    page = 1
    while len(coins) < limit:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": min(300, limit - len(coins)),
            "page": page
        }
        response = requests.get(url, params=params)
        data = response.json()
        if not data:
            break
        coins.extend([coin['id'] for coin in data])
        page += 1
    return coins[:limit]

def get_historical_data(coin_id, days=300):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "daily"
    }
    response = requests.get(url, params=params)
    if response.status_code == 429:
        print(f"⏳ Rate limit hit for {coin_id}. Retrying in 20 sec...")
        time.sleep(20)
        response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"❌ Failed {coin_id}: HTTP {response.status_code}")
        return None
    data = response.json()
    if not all(k in data for k in ['prices', 'total_volumes']):
        return None

    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
    df['close'] = df['price']
    df['volume'] = [v[1] for v in data['total_volumes']]
    df['high'] = df['close'] * 1.01
    df['low'] = df['close'] * 0.99
    df['open'] = df['close'].shift(1).fillna(df['close'])

    return df[['date', 'open', 'high', 'low', 'close', 'volume']]

def save_all_data(output_dir='crypto_csv_data', top_n=100, days=250):
    os.makedirs(output_dir, exist_ok=True)
    coin_ids = get_top_coins(top_n)
    for coin_id in coin_ids:
        try:
            df = get_historical_data(coin_id, days)
            if df is not None and len(df) >= 220:
                df.to_csv(f"{output_dir}/{coin_id}.csv", index=False)
                print(f"✅ Saved: {coin_id}.csv")
            else:
                print(f"⚠️ Skipped: {coin_id} (not enough data)")
            time.sleep(1.2)  # avoid rate limit
        except Exception as e:
            print(f"❌ Error {coin_id}: {e}")

if __name__ == "__main__":
    save_all_data()



