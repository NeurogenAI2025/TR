import os
import torch
from model_transformer import PriceTransformer
from utils import load_csv_sequence

DATA_DIR = "crypto_csv_data"
MODEL_DIR = "transformer_models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model_for_symbol(symbol):
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    try:
        X, y, _ = load_csv_sequence(file_path, window_size=200, indicators=True)
        model = PriceTransformer(input_size=X.shape[2], d_model=128, nhead=8, num_layers=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.MSELoss()

        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{symbol}.pt"))
        print(f"✅ Trained {symbol} with final loss {loss.item():.4f}")
    except Exception as e:
        print(f"❌ Failed {symbol}: {e}")

def train_all():
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".csv"):
            symbol = filename.replace(".csv", "")
            train_model_for_symbol(symbol)

if __name__ == "__main__":
    train_all()



