import pandas as pd
import mlflow
import mlflow.sklearn

from flask import Flask, request, jsonify

app = Flask(__name__)

# Connect to the running MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load registered model version 1 (use /1, not /v1)
MODEL_URI = "models:/LogisticRegressionModelRegistered/1"
model = mlflow.sklearn.load_model(MODEL_URI)

# Original input columns expected by the pipeline
FEATURE_NAMES = list(model.feature_names_in_)

# Numerical columns used by your preprocessing pipeline
NUMERICAL_FEATURES = next(
    list(columns)
    for name, transformer, columns
    in model.named_steps["preprocessor"].transformers_
    if name == "num"
)

# Your training code mapped No -> 0 and Yes -> 1
CHURN_INDEX = list(model.classes_).index(1)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Customer Churn Prediction API is running"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_uri": MODEL_URI
    })


@app.route("/predict", methods=["POST"])
def predict():
    # Read and validate the JSON body
    customer_data = request.get_json(silent=True)

    if not isinstance(customer_data, dict):
        return jsonify({
            "error": "Request body must be a valid JSON object"
        }), 400

    # Check for missing and unexpected features
    missing = [
        col for col in FEATURE_NAMES
        if col not in customer_data
    ]

    extra = [
        col for col in customer_data
        if col not in FEATURE_NAMES
    ]

    if missing or extra:
        return jsonify({
            "error": "Invalid input features",
            "missing_features": missing,
            "extra_features": extra
        }), 400

    # Each feature must contain one value
    if any(
        isinstance(value, (dict, list))
        for value in customer_data.values()
    ):
        return jsonify({
            "error": "Each feature must contain a single value"
        }), 400

    try:
        # Create a one-customer DataFrame in the expected order
        input_df = pd.DataFrame(
            [customer_data],
            columns=FEATURE_NAMES
        )

        # Convert numerical inputs; null values go to the imputer
        for col in NUMERICAL_FEATURES:
            input_df[col] = pd.to_numeric(
                input_df[col],
                errors="raise"
            ).astype("float64")

        prediction = int(model.predict(input_df)[0])

        probabilities = model.predict_proba(input_df)[0]
        churn_prob = float(probabilities[CHURN_INDEX])

        return jsonify({
            "prediction": prediction,
            "churn": "Yes" if prediction == 1 else "No",
            "churn_prob": round(churn_prob, 4)
        })

    except (ValueError, TypeError) as exc:
        return jsonify({
            "error": "Invalid feature values",
            "details": str(exc)
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed"
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False
    )
