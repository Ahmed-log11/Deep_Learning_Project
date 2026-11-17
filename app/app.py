"""FastAPI app — owned by Member 4.

Run: uvicorn app.app:app --reload
"""
from fastapi import FastAPI

app = FastAPI(title="Insurance Risk Scoring API")


@app.get("/")
def health():
    return {"status": "ok"}


# TODO (Member 4): add POST /predict that calls src.inference.predict_risk
