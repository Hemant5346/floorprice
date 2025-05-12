from fastapi import FastAPI, HTTPException
from app.schema import FloorPriceRequest
from app.model import predict_floor_price
import logging

app = FastAPI(
    title="Floor Price Predictor API",
    version="1.0",
    description="Predicts floor price based on country, domain, browser, and OS"
)

logger = logging.getLogger("uvicorn.error")

@app.post("/predict", summary="Predict Floor Price", tags=["Prediction"])
def predict(request: FloorPriceRequest):
    try:
        result = predict_floor_price(
            Country=request.Country,
            Domain=request.Domain,
            Browser=request.Browser,
            Os=request.Os
        )
        return {"predicted_floor_price": result}
    except Exception as e:
        logger.exception("Prediction endpoint failed.")
        raise HTTPException(status_code=500, detail="Prediction failed.")
