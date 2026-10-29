import logging
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g, Response
from prometheus_client import Counter, Histogram, Gauge
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]

# --------------------------------------------------
# Prometheus metrics
# --------------------------------------------------

HTTP_REQUESTS = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests received by the Student Result Prediction API",
    ["method", "path", "status"]
)

HTTP_LATENCY = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"]
)

PREDICTIONS = Counter(
    "student_api_predictions_total",
    "Total Student Result Prediction attempts",
    ["result"]
)

MODEL_LOADED = Gauge(
    "student_api_model_loaded",
    "Whether the Student Result Prediction model is loaded"
)

MODEL_LOADED.set(0)


# --------------------------------------------------
# Request monitoring
# --------------------------------------------------

@app.before_request
def start_request_timer():
    g.start_time = time.perf_counter()


@app.after_request
def record_request_metrics(response):
    path = request.path
    method = request.method

    start_time = getattr(g, "start_time", None)

    if start_time is not None and path != "/metrics":
        duration = time.perf_counter() - start_time

        HTTP_REQUESTS.labels(
            method=method,
            path=path,
            status=str(response.status_code)
        ).inc()

        HTTP_LATENCY.labels(
            method=method,
            path=path
        ).observe(duration)

    return response


# --------------------------------------------------
# Model loading
# --------------------------------------------------

def load_model():
    if not MODEL_PATH.exists():
        MODEL_LOADED.set(0)
        logger.error("Model file not found")

        raise FileNotFoundError(
            "Model not found. Run train_model.py first."
        )

    try:
        logger.info("Loading student result model")

        model = joblib.load(MODEL_PATH)

        MODEL_LOADED.set(1)

        logger.info("event=model_loaded model_file=%s", MODEL_PATH)

        return model

    except Exception:
        MODEL_LOADED.set(0)

        logger.exception(
            "event=model_load_failed model_file=%s",
            MODEL_PATH
        )

        raise


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/")
def health_check():
    logger.info("Health check requested")

    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    })


# --------------------------------------------------
# Prometheus metrics endpoint
# --------------------------------------------------

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST
    )


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict():
    logger.info("Prediction request received")

    data = request.get_json(silent=True)

    if not data:
        logger.warning(
            "Prediction request did not contain JSON data"
        )

        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        logger.warning(
            "Missing required fields: %s",
            missing_fields
        )

        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:
        sample = pd.DataFrame([{
            feature: data[feature]
            for feature in FEATURES
        }])

        model = load_model()

        prediction_code = int(model.predict(sample)[0])

        prediction = (
            "PASS"
            if prediction_code == 1
            else "FAIL"
        )

        PREDICTIONS.labels(
            result="success"
        ).inc()

        logger.info(
            "event=prediction_succeeded prediction=%s",
            prediction
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        })

    except Exception:
        PREDICTIONS.labels(
            result="error"
        ).inc()

        logger.exception(
            "event=prediction_failed"
        )

        return jsonify({
            "error": "Prediction failed"
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
