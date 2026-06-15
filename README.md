# AWS SAGEMAKER
## Benefits Eligibility Model
### Technical Specification & Model Documentation

**Version 1.0.0**  •  **March 28, 2026**  •  **HR Analytics Team**

---

## 1. Project Overview

This document describes all files required to deploy the Benefits Eligibility Classifier to AWS SageMaker. The model predicts whether an employee is eligible for company benefits based on their employment profile.

| Attribute | Value |
|---|---|
| Use Case | Employee benefits eligibility classification |
| Algorithm | Gradient Boosting Classifier (scikit-learn) |
| Owner | HR Analytics Team |
| Version | 1.0.0 |
| Risk Level | Medium — affects employee compensation |
| Training Date | 2026-03-28 |
| Output Classes | `not_eligible` \| `eligible` |

## 2. Repository & Artifact Structure

All files must be arranged exactly as shown below. The `model/` directory is compressed into `model.tar.gz` before uploading to S3.

```text
benefits-model-sagemaker/
├── model/
│   ├── model.pkl               ← trained GradientBoostingClassifier
│   ├── preprocessor.joblib     ← fitted ColumnTransformer
│   └── label_map.json          ← {"0": "not_eligible", "1": "eligible"}
├── code/
│   ├── inference.py            ← SageMaker entry-point script
│   ├── requirements.txt        ← runtime Python dependencies
│   └── config.yaml             ← (optional) deployment config
├── metadata/
│   └── model_metadata.json     ← governance & audit fields
├── test/
│   ├── sample_request.json     ← validation payload
│   └── expected_response.json  ← expected inference output
└── README.md
```

## 3. Artifact 1 — Model Bundle (MANDATORY)

### 3.1 model.tar.gz contents

The following files must be present inside the tar archive uploaded to S3:

1. `model.pkl` — serialised `GradientBoostingClassifier` (scikit-learn)
2. `preprocessor.joblib` — fitted `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`)
3. `label_map.json` — integer-to-class-name mapping

### 3.2 Feature Schema

The model accepts exactly 7 input features:

| Feature | Type | Description / Allowed Values |
|---|---|---|
| age | int | Employee age in years (22–65) |
| employment_type | string | `full_time` \| `part_time` \| `contractor` |
| years_of_service | int | Completed years at the company (0–40) |
| weekly_hours | int | Contracted weekly hours (10–55) |
| department | string | `engineering` \| `sales` \| `hr` \| `finance` \| `operations` |
| salary_band | string | `band_1` \| `band_2` \| `band_3` \| `band_4` \| `band_5` |
| has_dependents | int | Binary flag — `0 = no dependents`, `1 = has dependents` |

### 3.3 label_map.json

```json
{
  "0": "not_eligible",
  "1": "eligible"
}
```

## 4. Artifact 2 — Inference Script (MANDATORY)

**File:** `code/inference.py` — defines the four SageMaker entry-point functions.

### 4.1 model_fn(model_dir)

Loads `model.pkl`, `preprocessor.joblib`, and `label_map.json` from `model_dir`. Called once at container startup.

```python
def model_fn(model_dir):
    model        = joblib.load(os.path.join(model_dir, 'model.pkl'))
    preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.joblib'))
    with open(os.path.join(model_dir, 'label_map.json')) as f:
        label_map = json.load(f)
    return {'model': model, 'preprocessor': preprocessor, 'label_map': label_map}
```

### 4.2 input_fn(request_body, content_type)

Accepts `application/json` (dict or list of dicts) or `text/csv`. Validates that all 7 expected features are present. Returns a pandas DataFrame.

```python
def input_fn(request_body, content_type='application/json'):
    if content_type == 'application/json':
        data = json.loads(request_body)
        if isinstance(data, dict): data = [data]
        return pd.DataFrame(data)[EXPECTED_FEATURES]
    raise ValueError(f'Unsupported content type: {content_type}')
```

### 4.3 predict_fn(input_data, model_artifacts)

Runs the preprocessor transform then `clf.predict` / `clf.predict_proba`. Returns a dict containing labeled predictions and probability scores.

```python
def predict_fn(input_data, model_artifacts):
    X = model_artifacts['preprocessor'].transform(input_data)
    preds  = model_artifacts['model'].predict(X).tolist()
    probas = model_artifacts['model'].predict_proba(X).tolist()
    labeled = [model_artifacts['label_map'][str(p)] for p in preds]
    return {'predictions': labeled, 'probabilities': probas}
```

### 4.4 output_fn(prediction, accept)

Serialises the prediction dict to JSON. Raises `ValueError` for any accept type other than `application/json`.

```python
def output_fn(prediction, accept='application/json'):
    if accept == 'application/json':
        return json.dumps(prediction), 'application/json'
    raise ValueError(f'Unsupported accept: {accept}')
```

## 5. Artifact 3 — Requirements File (MANDATORY)

**File:** `code/requirements.txt` — installed by SageMaker at container build time.

```text
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
xgboost>=2.0
joblib>=1.3
```

## 6. Artifact 4 — Model Metadata

**File:** `metadata/model_metadata.json` — used for governance, auditing, and model registry.

| Field | Value for This Model |
|---|---|
| model_version | 1.0.0 |
| owner | HR Analytics Team |
| use_case | Employee benefits eligibility classification |
| risk_level | medium |
| protected_attributes | age, has_dependents |
| training_date | 2026-03-28 |
| algorithm | GradientBoostingClassifier |
| training_samples | 1,600 |
| test_samples | 400 |

```json
{
  "model_version": "1.0.0",
  "owner": "HR Analytics Team",
  "use_case": "Employee benefits eligibility classification",
  "risk_level": "medium",
  "protected_attributes": ["age", "has_dependents"],
  "training_date": "2026-03-28",
  "algorithm": "GradientBoostingClassifier",
  "hyperparameters": {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8
  }
}
```

## 7. Artifact 5 — Sample Payloads (Validation)

### 7.1 test/sample_request.json

```json
{
  "instances": [
    {
      "age": 34,
      "employment_type": "full_time",
      "years_of_service": 5,
      "weekly_hours": 40,
      "department": "engineering",
      "salary_band": "band_3",
      "has_dependents": 1
    },
    {
      "age": 27,
      "employment_type": "part_time",
      "years_of_service": 0,
      "weekly_hours": 20,
      "department": "sales",
      "salary_band": "band_1",
      "has_dependents": 0
    }
  ]
}
```

### 7.2 test/expected_response.json

```json
{
  "predictions": ["eligible", "not_eligible"],
  "probabilities": [[0.08, 0.92], [0.87, 0.13]],
  "label_order": ["not_eligible", "eligible"]
}
```

## 8. Deployment Notes

1. Package `model/` as `model.tar.gz` and upload to an S3 bucket accessible by your SageMaker execution role.
2. Point the SageMaker Model object at `inference.py` via `entry_point` and `source_dir`.
3. Use the sklearn container image: `sagemaker-scikit-learn` (`0.23-1` or later).
4. Invoke the endpoint with `Content-Type: application/json` and `Accept: application/json`.
5. Monitor the `protected_attributes` (`age`, `has_dependents`) for bias drift using SageMaker Clarify.

---

End of Document
