from pathlib import Path
import joblib
import pandas as pd


MODEL_PATH = Path("models/credit_risk_model.pkl")


def load_model():
    """
    Load the trained model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run train.py first."
        )

    model = joblib.load(MODEL_PATH)

    return model


def make_prediction(data: dict):
    """
    Generate prediction and probability score.
    """

    model = load_model()

    input_df = pd.DataFrame([data])

    # Align API input columns with training columns
    if hasattr(model, "feature_names_in_"):
        input_df = input_df.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "risk_probability": float(probability)
    }