
import React, { useState } from "react";
import ReactDOM from "react-dom/client";

function App() {
  const [symbol, setSymbol] = useState("bitcoin");
  const [prediction, setPrediction] = useState(null);

  const handlePredict = async () => {
    const res = await fetch(`http://localhost:8000/predict-transformer?symbol=${symbol}`);
    const data = await res.json();
    setPrediction(data);
  };

  return (
    <div style={{ padding: 20, fontFamily: "sans-serif" }}>
      <h1>Crypto Price Predictor</h1>
      <select value={symbol} onChange={(e) => setSymbol(e.target.value)}>
        {["bitcoin", "ethereum", "solana", "cardano", "ripple"].map((s) => (
          <option key={s} value={s}>{s}</option>
        ))}
      </select>
      <button onClick={handlePredict} style={{ marginLeft: 10 }}>Predict</button>

      {prediction && !prediction.error && (
        <div style={{ marginTop: 20 }}>
          <h2>{prediction.symbol.toUpperCase()} - Predicted Close</h2>
          <img src={`http://localhost:8000${prediction.chart}`} alt="chart" style={{ width: "100%", maxWidth: 600 }} />
        </div>
      )}

      {prediction?.error && <p style={{ color: "red" }}>{prediction.error}</p>}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
