from pathlib import Path
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request
import numpy as np
import shap

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "crop_recommendation_model.joblib"

app = Flask(__name__)
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]
RANGES = bundle["feature_ranges"]
METRICS = bundle["model_metrics"]
SHAP_GLOBAL = bundle.get("shap_global_importance", [])
# Initialize once at startup to avoid rebuilding the explainer per request.
shap_explainer = shap.TreeExplainer(model)

def render_page(values=None, prediction=None, top_crops=None, shap_explanations=None, error=None):
    return render_template(
        "index.html",
        features=FEATURES,
        ranges=RANGES,
        metrics=METRICS,
        prediction=prediction,
        top_crops=top_crops or [],
        shap_explanations=shap_explanations or [],
        shap_global=SHAP_GLOBAL,
        shap_global_sample_size=bundle.get("shap_global_sample_size", 0),
        error=error,
        values=values or {},
        test_accuracy=bundle["random_forest_holdout_accuracy"],
        macro_auc=bundle["random_forest_macro_roc_auc"],
        training_rows=bundle["training_rows"],
    )

@app.route("/", methods=["GET"])
def home():
    return render_page()

@app.route("/predict", methods=["POST"])
def predict():
    values = request.form.to_dict()
    try:
        row = {}
        for feature in FEATURES:
            raw = values.get(feature, "").strip()
            if not raw:
                raise ValueError(f"Please enter {feature}.")
            try:
                number = float(raw)
            except ValueError:
                raise ValueError(f"{feature} must be a numeric value.")

            limits = RANGES[feature]
            if not (limits["min"] <= number <= limits["max"]):
                raise ValueError(
                    f"{feature} must be between {limits['min']:.3f} and {limits['max']:.3f} "
                    "(the range represented in the training dataset)."
                )
            row[feature] = number

        input_df = pd.DataFrame([row], columns=FEATURES)
        probabilities = model.predict_proba(input_df)[0]
        classes = model.classes_
        ranked = sorted(
            zip(classes, probabilities),
            key=lambda item: item[1],
            reverse=True
        )
        prediction = str(ranked[0][0])
        predicted_class_index = list(classes).index(ranked[0][0])

        # Class-specific SHAP values for the crop predicted for this input.
        explanation = shap_explainer(input_df, check_additivity=False)
        shap_values = np.asarray(explanation.values)
        if shap_values.ndim == 3 and shap_values.shape[1] == len(FEATURES):
            local_values = shap_values[0, :, predicted_class_index]  # samples, features, classes
        elif shap_values.ndim == 3 and shap_values.shape[2] == len(FEATURES):
            local_values = shap_values[0, predicted_class_index, :]  # samples, classes, features
        elif shap_values.ndim == 2:
            local_values = shap_values[0, :]
        else:
            raise ValueError(f"Unexpected local SHAP shape: {shap_values.shape}")
        shap_explanations = [
            {"feature": feature, "value": round(float(row[feature]), 3),
             "shap_value": round(float(value), 6),
             "direction": "supports" if value >= 0 else "opposes",
             "magnitude": abs(float(value))}
            for feature, value in zip(FEATURES, local_values)
        ]
        shap_explanations.sort(key=lambda item: item["magnitude"], reverse=True)

        top_crops = [
            {"crop": str(crop), "probability": round(float(prob) * 100, 2)}
            for crop, prob in ranked[:3]
        ]
        return render_page(values=values, prediction=prediction, top_crops=top_crops, shap_explanations=shap_explanations)
    except ValueError as exc:
        return render_page(values=values, error=str(exc)), 400
    except Exception:
        app.logger.exception("Crop prediction failed")
        return render_page(
            values=values,
            error="Prediction could not be completed. Please verify the inputs and try again."
        ), 500

@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model": "Random Forest Classifier",
        "classes": len(bundle["classes"]),
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
