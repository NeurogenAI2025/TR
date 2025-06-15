
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from model_transformer import PriceTransformer
from utils import load_csv_sequence
from datetime import datetime

def predict(symbol, folder="crypto_csv_data", model_folder="transformer_models"):
    filepath = os.path.join(folder, f"{symbol}.csv")
    X, _ = load_csv_sequence(filepath)
    model = PriceTransformer()
    model.load_state_dict(torch.load(os.path.join(model_folder, f"{symbol}.pt")))
    model.eval()
    with torch.no_grad():
        prediction = model(X[-1].unsqueeze(0)).item()

    # Salvare CSV
    os.makedirs("predictions", exist_ok=True)
    csv_path = f"predictions/{symbol}_predictions.csv"
    today = datetime.now().strftime("%Y-%m-%d")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=["date", "predicted_close"])
    df = pd.concat([df, pd.DataFrame([{"date": today, "predicted_close": prediction}])], ignore_index=True)
    df.drop_duplicates(subset=["date"], keep="last").to_csv(csv_path, index=False)

    # Grafic
    os.makedirs("charts", exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(df["date"], df["predicted_close"], marker="o", label="Predicted Close")
    plt.title(f"{symbol.upper()} - Predicted Close Price")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"charts/{symbol}.png")
    plt.close()
    print(f"Prediction for {symbol}: {prediction:.2f} saved to CSV and chart.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    args = parser.parse_args()
    predict(args.symbol)
