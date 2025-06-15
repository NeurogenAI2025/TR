
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
import os
from predict_transformer import predict

app = FastAPI()

@app.get("/predict-transformer")
def predict_endpoint(symbol: str = Query(...)):
    try:
        predict(symbol)
        chart_path = f"charts/{symbol}.png"
        csv_path = f"predictions/{symbol}_predictions.csv"
        if os.path.exists(chart_path):
            return JSONResponse({
                "symbol": symbol,
                "chart": f"/chart/{symbol}",
                "csv": f"/csv/{symbol}"
            })
        else:
            return JSONResponse({"error": "Chart not found."}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/chart/{symbol}")
def get_chart(symbol: str):
    path = f"charts/{symbol}.png"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    return JSONResponse({"error": "Chart not found."}, status_code=404)

@app.get("/csv/{symbol}")
def get_csv(symbol: str):
    path = f"predictions/{symbol}_predictions.csv"
    if os.path.exists(path):
        return FileResponse(path, media_type="text/csv")
    return JSONResponse({"error": "CSV not found."}, status_code=404)
