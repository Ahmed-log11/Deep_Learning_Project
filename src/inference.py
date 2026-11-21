"""Inference pipeline (same predict_risk() as notebook 04).

predict_risk(input_dict) -> {"claim_probability": float, "prediction": int}
Uses the preprocessor saved in notebook 01 and the best (tuned) model from notebook 03.
Run from the repo root:  python src/inference.py
"""
import json

import joblib
import pandas as pd
from tensorflow import keras

MODEL_PATH = "models/best_model.keras"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
META_PATH = "data/processed/preprocessor_meta.json"
THRESHOLD = 0.5

model = keras.models.load_model(MODEL_PATH, compile=False)
preprocessor = joblib.load(PREPROCESSOR_PATH)
with open(META_PATH) as f:
    META = json.load(f)
INPUT_COLS = META["raw_input_columns"]
ALLOWED = META["allowed_values"]
BINARY_COLS = META["column_groups"]["binary"]


def predict_risk(input_dict: dict, threshold: float = THRESHOLD) -> dict:
    missing = [c for c in INPUT_COLS if c not in input_dict]
    extra = [c for c in input_dict if c not in INPUT_COLS]
    if missing or extra:
        raise ValueError(f"Missing columns: {missing} | Unknown columns: {extra}")

    row = {}
    for c in INPUT_COLS:
        v = input_dict[c]
        if isinstance(v, str):
            v = v.strip().lower()
        if c in ALLOWED and v not in ALLOWED[c]:
            raise ValueError(f"{c}={v!r} is not allowed. Allowed values: {ALLOWED[c]}")
        if c in BINARY_COLS and v not in (0, 1):
            raise ValueError(f"{c} must be 0 or 1, got {v!r}")
        row[c] = v

    X = preprocessor.transform(pd.DataFrame([row], columns=INPUT_COLS)).astype("float32")
    prob = float(model.predict(X, verbose=0)[0, 0])
    return {"claim_probability": round(prob, 4), "prediction": int(prob >= threshold)}


if __name__ == "__main__":
    example = {
        "AGE": "16-25", "GENDER": "female", "RACE": "majority", "DRIVING_EXPERIENCE": "0-9y",
        "EDUCATION": "university", "INCOME": "poverty", "CREDIT_SCORE": 0.30,
        "VEHICLE_OWNERSHIP": 0, "VEHICLE_YEAR": "before 2015", "MARRIED": 0, "CHILDREN": 0,
        "POSTAL_CODE": 10238, "ANNUAL_MILEAGE": 18000, "VEHICLE_TYPE": "sedan",
        "SPEEDING_VIOLATIONS": 0, "DUIS": 0, "PAST_ACCIDENTS": 0,
    }
    print(predict_risk(example))
