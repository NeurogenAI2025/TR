import os
import torch
import pandas as pd
from model_transformer import PriceTransformer
from utils import load_csv_sequence_with_sentiment
from torch import nn, optim

CSV_DIR = "crypto_csv_data"
MODEL_DIR = "transformer_models"
SENTIMENT_CSV = "btc_news_sentiment.csv"
SKIPPED_CSV = "skipped_train.csv"

os.makedirs(MODEL_DIR, exist_ok=True)

def load_sentiment():
    if not os.path.exists(SENTIMENT_CSV):
        return {}
    df = pd.read_csv(SENTIMENT_CSV)
    return dict(zip(df["date"].astype(str), df["sentiment_score"]))

def train_symbol(symbol, sentiment_map, skipped):
    try:
        csv_path = os.path.join(CSV_DIR, f"{symbol}.csv")
        X, y, _, _ = load_csv_sequence_with_sentiment(
            csv_path, window_size=150, indicators=True, sentiment_map=sentiment_map
        )

        if X.shape[0] < 10:
            raise ValueError("Insufficient samples after preprocessing.")

        model = PriceTransformer(input_size=X.shape[2], d_model=128, nhead=8, num_layers=3)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(30):
            model.train()
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y.squeeze())
            loss.backward()
            optimizer.step()

        model_path = os.path.join(MODEL_DIR, f"{symbol}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"✅ Trained {symbol} with final loss {loss.item():.4f}")

    except Exception as e:
        print(f"❌ Failed {symbol}: {e}")
        skipped.append({"symbol": symbol, "error": str(e)})

def train_all():
    sentiment_map = load_sentiment()
    skipped = []

    for fname in os.listdir(CSV_DIR):
        if fname.endswith(".csv"):
            symbol = fname.replace(".csv", "")
            train_symbol(symbol, sentiment_map, skipped)

    if skipped:
        pd.DataFrame(skipped).to_csv(SKIPPED_CSV, index=False)
        print(f"🔸 Skipped symbols saved to: {SKIPPED_CSV}")

if __name__ == "__main__":
    train_all()
