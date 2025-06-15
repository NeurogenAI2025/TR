import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from model_transformer import PriceTransformer
from utils import load_csv_sequence_with_sentiment
from sklearn.metrics import mean_absolute_error

CSV_DIR = "crypto_csv_data"
MODEL_DIR = "transformer_models"
PREDICT_DIR = "predictions"
CHART_DIR = "charts"
ALL_PREDICTIONS_PATH = "all_predictions.csv"
EVALUATION_PATH = "evaluation.csv"
SKIPPED_PATH = "skipped_symbols.csv"
SENTIMENT_CSV = "btc_news_sentiment.csv"

os.makedirs(PREDICT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

def load_sentiment():
    if not os.path.exists(SENTIMENT_CSV):
        return {}
    df = pd.read_csv(SENTIMENT_CSV)
    return dict(zip(df["date"].astype(str), df["sentiment_score"]))

def predict_for_symbol(symbol, sentiment_map, skipped):
    try:
        csv_path = os.path.join(CSV_DIR, f"{symbol}.csv")
        model_path = os.path.join(MODEL_DIR, f"{symbol}.pt")

        X, y, scaler, last_date = load_csv_sequence_with_sentiment(
            csv_path, window_size=150, indicators=True, sentiment_map=sentiment_map
        )
        if X.shape[0] == 0:
            raise ValueError("Insufficient samples after preprocessing.")

        X_input = X[-1:].clone().detach()
        y_true = y[-1:].item()

        model = PriceTransformer(input_size=X.shape[2], d_model=128, nhead=8, num_layers=3)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        with torch.no_grad():
            pred_scaled = model(X_input).item()

        last_close = scaler.inverse_transform([[0]*3 + [pred_scaled] + [0]*(X.shape[2]-4)])[0][3]
        true_close = scaler.inverse_transform([[0]*3 + [y_true] + [0]*(X.shape[2]-4)])[0][3]

        predict_date = last_date + pd.Timedelta(days=1)

        result_df = pd.DataFrame([{
            "symbol": symbol,
            "predicted_close": last_close,
            "true_close": true_close,
            "error_abs": abs(last_close - true_close),
            "predicted_date": predict_date.date()
        }])
        result_df.to_csv(
            os.path.join(PREDICT_DIR, f"{symbol}_prediction.csv"), index=False
        )

        if os.path.exists(ALL_PREDICTIONS_PATH):
            all_df = pd.read_csv(ALL_PREDICTIONS_PATH)
            all_df = pd.concat([all_df, result_df], ignore_index=True)
        else:
            all_df = result_df
        all_df.to_csv(ALL_PREDICTIONS_PATH, index=False)

        df = pd.read_csv(csv_path)
        plt.figure(figsize=(10, 5))
        plt.plot(df["close"][-20:], label="Close price")
        plt.axhline(last_close, color="red", linestyle="--", label="Predicted close")
        plt.axhline(true_close, color="green", linestyle=":", label="True last close")
        plt.title(f"{symbol} Prediction — {predict_date.date()}")
        plt.legend()
        plt.savefig(os.path.join(CHART_DIR, f"{symbol}.png"))
        plt.close()

        print(f"✅ Predicted: {symbol} -> {last_close:.2f}, True: {true_close:.2f}, Error: {abs(last_close - true_close):.2f}")

    except Exception as e:
        print(f"❌ Failed for {symbol}: {e}")
        skipped.append({"symbol": symbol, "error": str(e)})

def evaluate_all_predictions():
    if not os.path.exists(ALL_PREDICTIONS_PATH):
        print("No all_predictions.csv found to evaluate.")
        return
    df = pd.read_csv(ALL_PREDICTIONS_PATH)
    if 'true_close' not in df.columns:
        print("Missing true_close column for evaluation.")
        return
    mae = mean_absolute_error(df['true_close'], df['predicted_close'])
    df_sorted = df.sort_values(by="error_abs", ascending=False)
    df_sorted.to_csv(EVALUATION_PATH, index=False)
    print(f"\n📊 Global MAE (mean absolute error): {mae:.2f}")
    print(f"📁 Evaluation saved to: {EVALUATION_PATH}")

def predict_all():
    sentiment_map = load_sentiment()
    skipped = []
    if os.path.exists(ALL_PREDICTIONS_PATH):
        os.remove(ALL_PREDICTIONS_PATH)
    if os.path.exists(EVALUATION_PATH):
        os.remove(EVALUATION_PATH)
    for fname in os.listdir(MODEL_DIR):
        if fname.endswith(".pt"):
            symbol = fname.replace(".pt", "")
            predict_for_symbol(symbol, sentiment_map, skipped)
    evaluate_all_predictions()

    if skipped:
        pd.DataFrame(skipped).to_csv(SKIPPED_PATH, index=False)
        print(f"🔹 Skipped symbols saved to: {SKIPPED_PATH}")

if __name__ == "__main__":
    predict_all()
