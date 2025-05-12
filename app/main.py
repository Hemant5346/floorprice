from fastapi import FastAPI, HTTPException
from app.schema import FloorPriceRequest
from app.model import predict_floor_price
import logging
import time

app = FastAPI(
    title="Floor Price Predictor API",
    version="1.0",
    description="Predicts floor price based on country, domain, browser, and OS"
)

logger = logging.getLogger("uvicorn.error")

@app.post("/predict", summary="Predict Floor Price", tags=["Prediction"])
def predict(request: FloorPriceRequest):
    try:
        start_time = time.time()
        result = predict_floor_price(
            Country=request.Country,
            Domain=request.Domain,
            Browser=request.Browser,
            Os=request.Os
        )
        end_time = time.time()
        prediction_time_ms = round((end_time - start_time) * 1000, 2)
        return {
            "predicted_floor_price": result,
            "prediction_time_ms": prediction_time_ms
        }
    except Exception:
        logger.exception("Prediction endpoint failed.")
        raise HTTPException(status_code=500, detail="Prediction failed.")
