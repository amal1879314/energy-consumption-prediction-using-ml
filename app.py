# app.py
from flask import Flask, request, render_template, jsonify
import os
import traceback
import logging

import pandas as pd

# import helpers from model_utils
from model_utils import load_model, prepare_input, align_features

# configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder="templates", static_folder="static")
MODEL_PATH = os.path.join(APP_ROOT, "models", "lgb_model_final97.pkl")

# Load model once at startup
try:
    model, meta = load_model(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
except Exception as e:
    logger.exception("Failed to load model at startup: %s", e)
    # Still allow app to start, but requests will fail with a helpful message.
    model, meta = None, {}

# expected features if available
expected_features = meta.get("feature_names")

@app.route("/", methods=["GET"])
def home():
    feature_names = meta.get("feature_names", None)
    return render_template("index.html", feature_names=feature_names)

@app.route("/predict", methods=["POST"])
def predict_form():
    """
    Handles form submission from index.html. Uses prepare_input which returns (df, report).
    """
    if model is None:
        return render_template("index.html", error="Model not loaded. Check server logs.", feature_names=meta.get("feature_names"))

    try:
        # Parse input (either JSON pasted into textarea named 'json_input' or form fields)
        json_input_text = request.form.get("json_input", None)
        if json_input_text:
            # parse the JSON text area value
            try:
                import json
                data = json.loads(json_input_text)
            except Exception:
                # attempt to read raw request json fallback
                data = request.get_json(force=False, silent=True)
                if data is None:
                    raise ValueError("Could not parse JSON input. Make sure the JSON is valid.")
        else:
            # Build dict from form fields
            form_dict = {}
            for key, val in request.form.items():
                if key == "submit":
                    continue
                form_dict[key] = val
            if not form_dict:
                return render_template("index.html", error="No input provided.", feature_names=meta.get("feature_names"))
            data = form_dict

        # Build DataFrame and get conversion report. prepare_input should return (df, report)
        df, prep_report = prepare_input(data, expected_feature_names=expected_features, coerce_numeric=True)
        logger.info("prepare_input report: %s", prep_report)
        # show dtypes for debugging
        logger.info("Prepared DF dtypes:\n%s", df.dtypes)

        # Final safety: ensure columns are ordered as expected (align_features is idempotent if expected_features is None)
        if expected_features:
            df = align_features(df, expected_features, fill_value=float("nan"), verbose=False)

        # Predict
        preds = model.predict(df)
        pred_val = float(preds[0]) if hasattr(preds, "__len__") else float(preds)

        # Render result and include some debug info visible in template
        return render_template(
            "index.html",
            result=pred_val,
            inputs=df.to_dict(orient="records")[0],
            feature_names=meta.get("feature_names"),
            prep_report=prep_report
        )

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Prediction failed: %s", tb)
        # Provide helpful debug info in the UI
        debug = {}
        try:
            # if df exists, include incoming columns
            if 'df' in locals():
                debug["incoming_columns"] = list(df.columns)
                debug["dtypes"] = df.dtypes.apply(lambda x: str(x)).to_dict()
        except Exception:
            pass
        debug["expected_columns"] = expected_features
        return render_template("index.html", error=str(e), traceback=tb, debug=debug, feature_names=meta.get("feature_names"))

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API endpoint. Accepts either {"features": {...}} or {"col1": val1, ...}
    Returns JSON with prediction and a prep_report for debugging.
    """
    if model is None:
        return jsonify({"error": "model_not_loaded", "details": "Model not loaded on server"}), 500

    try:
        payload = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": "invalid_json", "details": str(e)}), 400

    if payload is None:
        return jsonify({"error": "no_json_body"}), 400

    # Accept either {"features": {...}} or the direct feature dict
    if "features" in payload and isinstance(payload["features"], dict):
        features = payload["features"]
    else:
        features = payload

    try:
        df, prep_report = prepare_input(features, expected_feature_names=expected_features, coerce_numeric=True)
        logger.info("API prepare_input report: %s", prep_report)

        if expected_features:
            df = align_features(df, expected_features, fill_value=float("nan"), verbose=False)

        preds = model.predict(df)
        pred0 = preds[0] if hasattr(preds, "__len__") else preds

        return jsonify({
            "prediction": float(pred0),
            "prep_report": prep_report
        })
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("API prediction failed: %s", tb)
        # If sequence/shape issue, include expected/incoming to help debug
        incoming_cols = None
        try:
            if 'df' in locals():
                incoming_cols = list(df.columns)
        except Exception:
            pass
        return jsonify({
            "error": "prediction_failed",
            "details": str(e),
            "traceback": tb,
            "expected_columns": expected_features,
            "incoming_columns": incoming_cols
        }), 500

# Optional debug route (remove or protect in production)
@app.route("/_debug_model", methods=["GET"])
def debug_model():
    return jsonify({
        "model_loaded": model is not None,
        "expected_features": expected_features
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)