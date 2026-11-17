"""Inference pipeline — owned by Member 4.

predict_risk(input_dict) -> {"claim_probability": float, "prediction": int}
Uses the same preprocessor saved in notebook 01 and the best model from notebook 03.
"""

MODEL_PATH = "models/best_model.keras"
PREPROCESSOR_PATH = "models/preprocessor.pkl"


def predict_risk(input_dict: dict) -> dict:
    # TODO (Member 4): load model + preprocessor, transform input, predict
    raise NotImplementedError
