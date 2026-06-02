from fastapi import FastAPI
from src.api.pydantic_models import CustomerFeatures
from src.predict import make_prediction

app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting customer credit risk probability",
    version="1.0.0"
)


@app.get("/")
def home():
    return {"message": "Credit Risk Model API is running"}


@app.post("/predict")
def predict(data: CustomerFeatures):
    input_data = data.model_dump()

    result = make_prediction(input_data)

    return {
        "prediction": result["prediction"],
        "risk_probability": result["risk_probability"],
        "message": "Prediction successful"
    }