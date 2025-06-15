import requests

url = "https://api.coingecko.com/api/v3/coins/markets"
params = {
    "vs_currency": "usd",
    "order": "market_cap_desc",
    "per_page": 30,
    "page": 1
}

response = requests.get(url, params=params)
print(f"Status: {response.status_code}")
print("Număr monede returnate:", len(response.json()))
print("Primii 5 ID-uri:", [coin["id"] for coin in response.json()[:5]])
