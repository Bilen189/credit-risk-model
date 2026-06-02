from fastapi import FastAPI
from src.api.pydantic_models import CustomerFeatures

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Credit Risk Model API is running"}


@app.post("/predict")
def predict(data: CustomerFeatures):

    prediction = 1

    return {
        "prediction": prediction,
        "message": "Prediction successful"
    }