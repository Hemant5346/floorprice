import joblib
import pandas as pd
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Load Model and Metadata at Startup ===
try:
    MODEL_PATH = os.path.join("models", "floor_price_model.pkl")
    METADATA_PATH = os.path.join("models", "floor_price_metadata.pkl")

    model = joblib.load(MODEL_PATH)
    scaler, domain_means, country_freq = joblib.load(METADATA_PATH)

    model_columns = model.get_booster().feature_names
    template_df = pd.DataFrame(columns=model_columns)

    # Precomputed fallbacks
    domain_default = np.mean(list(domain_means.values()))
    country_default = 0

    logger.info("Model and metadata loaded successfully.")
except Exception:
    logger.exception("Failed to load model or metadata.")
    raise

# === Optimized Prediction Function ===
def predict_floor_price(Country: str, Domain: str, Browser: str, Os: str) -> float:
    try:
        row = {
            'Country': Country,
            'Domain': Domain,
            'Browser': Browser,
            'Os': Os
        }
        input_df = pd.DataFrame([row])

        # Fast map + default fallback
        input_df['Domain_te'] = input_df['Domain'].map(domain_means).fillna(domain_default)
        input_df['Country_freq'] = input_df['Country'].map(country_freq).fillna(country_default)

        # One-hot encode
        input_df = pd.get_dummies(input_df, columns=['Browser', 'Os'], drop_first=True)

        # Ensure all required model columns are present
        missing_cols = [col for col in model_columns if col not in input_df.columns]
        if missing_cols:
            padding_df = pd.DataFrame(0, index=input_df.index, columns=missing_cols)
            input_df = pd.concat([input_df, padding_df], axis=1)

        # Reorder columns to match model
        input_df = input_df[model_columns]


        # Predict using NumPy array for speed
        pred = model.predict(input_df.to_numpy())[0]
        return float(round(pred, 4))
    except Exception:
        logger.exception("Prediction failed.")
        raise ValueError("Error in prediction pipeline.")
