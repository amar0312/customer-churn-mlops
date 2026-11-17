import logging
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using a registered MLflow model",
    version="1.0.0"
)

# Connect to MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Use the registered model name without the trailing period
# Version 2 must exist in your MLflow registry
MODEL_URI = "models:/LogisticRegressionModelRegistered/1"
model = mlflow.sklearn.load_model(MODEL_URI)

# Expected input columns
FEATURE_NAMES = list(model.feature_names_in_)

# Numerical columns from the saved preprocessing pipeline
NUMERICAL_FEATURES = next(
    list(columns)
    for name, transformer, columns
    in model.named_steps["preprocessor"].transformers_
    if name == "num"
)

# Training labels: No = 0, Yes = 1
CHURN_INDEX = list(model.classes_).index(1)


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction FastAPI is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_uri": MODEL_URI
    }


@app.post("/predict")
def predict(customer_data: dict[str, Any]):

    # Validate feature names before attempting prediction
    missing = [
        col for col in FEATURE_NAMES
        if col not in customer_data
    ]

    extra = [
        col for col in customer_data
        if col not in FEATURE_NAMES
    ]

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Invalid input features",
                "missing_features": missing,
                "extra_features": extra
            }
        )

    if any(
        isinstance(value, (dict, list))
        for value in customer_data.values()
    ):
        raise HTTPException(
            status_code=422,
            detail="Each feature must contain a single value"
        )

    try:
        # Convert one customer's JSON into a DataFrame
        input_df = pd.DataFrame(
            [customer_data],
            columns=FEATURE_NAMES
        )

        # Convert numerical inputs; missing values go to the imputer
        for col in NUMERICAL_FEATURES:
            input_df[col] = pd.to_numeric(
                input_df[col],
                errors="raise"
            ).astype("float64")

        # Predict
        prediction = int(model.predict(input_df)[0])

        probabilities = model.predict_proba(input_df)[0]
        churn_prob = float(probabilities[CHURN_INDEX])

        return {
            "prediction": prediction,
            "churn": "Yes" if prediction == 1 else "No",
            "churn_prob": round(churn_prob, 4)
        }

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Invalid feature values",
                "details": str(exc)
            }
        ) from exc

    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        ) from exc