"""
AWS SageMaker Inference Script — Benefits Eligibility Classifier
================================================================
Predicts employee benefits eligibility based on employee profile data.

Required SageMaker entry-point functions:
  model_fn      → loads model artifacts from model_dir
  input_fn      → parses incoming request body
  predict_fn    → runs inference
  output_fn     → formats prediction response
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# 1. model_fn — Load model + preprocessor from model directory
# ---------------------------------------------------------------------------

def model_fn(model_dir: str) -> dict:
    """
    Load the trained model, preprocessor, and label map from model_dir.
    SageMaker calls this once at container startup.

    Args:
        model_dir (str): Path to the directory containing model artifacts.

    Returns:
        dict: {'model': clf, 'preprocessor': preprocessor, 'label_map': label_map}
    """
    logger.info("Loading model artifacts from: %s", model_dir)

    model_path       = os.path.join(model_dir, "model.pkl")
    preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
    label_map_path   = os.path.join(model_dir, "label_map.json")

    model        = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    with open(label_map_path, "r") as f:
        label_map = json.load(f)          # e.g. {"0": "not_eligible", "1": "eligible"}

    logger.info("Model, preprocessor, and label map loaded successfully.")
    return {"model": model, "preprocessor": preprocessor, "label_map": label_map}


# ---------------------------------------------------------------------------
# 2. input_fn — Parse raw request body into a DataFrame
# ---------------------------------------------------------------------------

EXPECTED_FEATURES = [
    "age",
    "employment_type",       # "full_time" | "part_time" | "contractor"
    "years_of_service",
    "weekly_hours",
    "department",
    "salary_band",           # "band_1" … "band_5"
    "has_dependents",        # 0 or 1
]

def input_fn(request_body: str, content_type: str = "application/json") -> pd.DataFrame:
    """
    Deserialise the incoming request into a pandas DataFrame.

    Accepts:
        - application/json  → single record (dict) or list of records
        - text/csv          → header row + one or more data rows

    Args:
        request_body (str): Raw request payload.
        content_type (str): MIME type of the incoming request.

    Returns:
        pd.DataFrame: One row per inference request.

    Raises:
        ValueError: For unsupported content types or missing fields.
    """
    logger.info("Parsing request with content_type=%s", content_type)

    if content_type == "application/json":
        data = json.loads(request_body)
        if isinstance(data, dict):
            data = [data]
        df = pd.DataFrame(data)

    elif content_type == "text/csv":
        from io import StringIO
        df = pd.read_csv(StringIO(request_body))

    else:
        raise ValueError(
            f"Unsupported content type: '{content_type}'. "
            "Use 'application/json' or 'text/csv'."
        )

    missing = set(EXPECTED_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Request is missing required feature(s): {missing}")

    return df[EXPECTED_FEATURES]


# ---------------------------------------------------------------------------
# 3. predict_fn — Run inference
# ---------------------------------------------------------------------------

def predict_fn(input_data: pd.DataFrame, model_artifacts: dict) -> dict:
    """
    Apply the preprocessor then run the classifier.

    Args:
        input_data (pd.DataFrame): Feature matrix from input_fn.
        model_artifacts (dict):    Output of model_fn.

    Returns:
        dict: {'predictions': [...], 'probabilities': [...]}
    """
    preprocessor = model_artifacts["preprocessor"]
    model        = model_artifacts["model"]
    label_map    = model_artifacts["label_map"]

    logger.info("Running inference on %d record(s).", len(input_data))

    X_transformed = preprocessor.transform(input_data)

    raw_predictions  = model.predict(X_transformed).tolist()
    class_proba      = model.predict_proba(X_transformed).tolist()

    labeled_predictions = [label_map[str(p)] for p in raw_predictions]

    return {
        "predictions":   labeled_predictions,
        "probabilities": class_proba,          # [[p_class0, p_class1], ...]
        "label_order":   [label_map[k] for k in sorted(label_map)],
    }


# ---------------------------------------------------------------------------
# 4. output_fn — Serialise prediction result to HTTP response
# ---------------------------------------------------------------------------

def output_fn(prediction: dict, accept: str = "application/json") -> tuple:
    """
    Convert the prediction dict to the desired response format.

    Args:
        prediction (dict): Output of predict_fn.
        accept (str):      Desired MIME type of the response.

    Returns:
        tuple: (body: str, content_type: str)

    Raises:
        ValueError: For unsupported accept types.
    """
    if accept == "application/json":
        return json.dumps(prediction), "application/json"

    raise ValueError(
        f"Unsupported accept type: '{accept}'. Use 'application/json'."
    )
