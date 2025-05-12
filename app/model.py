import joblib
import pandas as pd
import numpy as np
from typing import Dict
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and metadata
try:
    MODEL_PATH = os.path.join("models", "floor_price_model.pkl")
    METADATA_PATH = os.path.join("models", "floor_price_metadata.pkl")
    model = joblib.load(MODEL_PATH)
    scaler, domain_means, country_freq = joblib.load(METADATA_PATH)
    logger.info("Model and metadata loaded successfully.")
except Exception as e:
    logger.exception("Failed to load model or metadata.")
    raise

def predict_floor_price(Country: str, Domain: str, Browser: str, Os: str) -> float:
    try:
        input_df = pd.DataFrame([{
            'Country': Country,
            'Domain': Domain,
            'Browser': Browser,
            'Os': Os
        }])

        input_df['Domain_te'] = input_df['Domain'].map(domain_means).fillna(np.mean(list(domain_means.values())))
        input_df['Country_freq'] = input_df['Country'].map(country_freq).fillna(0)
        input_df = pd.get_dummies(input_df, columns=['Browser', 'Os'], drop_first=True)

        model_columns = model.get_booster().feature_names
        missing_cols = [col for col in model_columns if col not in input_df.columns]

        if missing_cols:
            padding_df = pd.DataFrame(0, index=input_df.index, columns=missing_cols)
            input_df = pd.concat([input_df, padding_df], axis=1)

        input_df = input_df[model_columns]

        pred = model.predict(input_df)[0]
        return float(round(pred, 4))  # <-- Fix: convert to native float
    except Exception as e:
        logger.exception("Prediction failed.")
        raise ValueError("Error in prediction pipeline.")

