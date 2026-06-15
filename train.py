"""
train.py — Benefits Eligibility Classifier
===========================================
Trains a gradient-boosted classifier on synthetic employee data and saves
all model artifacts in the directory structure expected by inference.py
and AWS SageMaker.

Output structure
----------------
benefits-model-sagemaker/
  model/
    model.pkl
    preprocessor.joblib
    label_map.json
  metadata/
    model_metadata.json
  test/
    sample_request.json
    expected_response.json
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline              import Pipeline
from sklearn.compose               import ColumnTransformer
from sklearn.preprocessing         import StandardScaler, OneHotEncoder
from sklearn.ensemble              import GradientBoostingClassifier
from sklearn.model_selection       import train_test_split
from sklearn.metrics               import classification_report

# ---------------------------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# 1. Synthetic dataset
# ---------------------------------------------------------------------------

N = 2_000

employment_types = ["full_time", "part_time", "contractor"]
departments      = ["engineering", "sales", "hr", "finance", "operations"]
salary_bands     = ["band_1", "band_2", "band_3", "band_4", "band_5"]

df = pd.DataFrame({
    "age":              np.random.randint(22, 65, N),
    "employment_type":  np.random.choice(employment_types, N, p=[0.6, 0.25, 0.15]),
    "years_of_service": np.clip(np.random.exponential(5, N), 0, 40).astype(int),
    "weekly_hours":     np.random.randint(10, 55, N),
    "department":       np.random.choice(departments, N),
    "salary_band":      np.random.choice(salary_bands, N, p=[0.15, 0.25, 0.30, 0.20, 0.10]),
    "has_dependents":   np.random.randint(0, 2, N),
})

# Deterministic eligibility rule (mirrors realistic HR logic)
df["eligible"] = (
    (df["employment_type"] == "full_time") &
    (df["weekly_hours"] >= 30) &
    (df["years_of_service"] >= 1)
).astype(int)

# Add a little noise
flip_mask = np.random.rand(N) < 0.05
df.loc[flip_mask, "eligible"] = 1 - df.loc[flip_mask, "eligible"]

X = df.drop(columns=["eligible"])
y = df["eligible"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# 2. Preprocessing pipeline
# ---------------------------------------------------------------------------

numeric_features     = ["age", "years_of_service", "weekly_hours", "has_dependents"]
categorical_features = ["employment_type", "department", "salary_band"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(),                   numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

# ---------------------------------------------------------------------------
# 3. Classifier
# ---------------------------------------------------------------------------

clf = GradientBoostingClassifier(
    n_estimators   = 200,
    learning_rate  = 0.05,
    max_depth      = 4,
    subsample      = 0.8,
    random_state   = RANDOM_STATE,
)

# Fit preprocessor on training data; keep it separate for SageMaker deployment
X_train_t = preprocessor.fit_transform(X_train)
X_test_t  = preprocessor.transform(X_test)

clf.fit(X_train_t, y_train)

y_pred = clf.predict(X_test_t)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["not_eligible", "eligible"]))

# ---------------------------------------------------------------------------
# 4. Label map
# ---------------------------------------------------------------------------

label_map = {"0": "not_eligible", "1": "eligible"}

# ---------------------------------------------------------------------------
# 5. Save artifacts
# ---------------------------------------------------------------------------

BASE  = "benefits-model-sagemaker"
MODEL = os.path.join(BASE, "model")
META  = os.path.join(BASE, "metadata")
TEST  = os.path.join(BASE, "test")

for d in [MODEL, META, TEST]:
    os.makedirs(d, exist_ok=True)

joblib.dump(clf,          os.path.join(MODEL, "model.pkl"))
joblib.dump(preprocessor, os.path.join(MODEL, "preprocessor.joblib"))

with open(os.path.join(MODEL, "label_map.json"), "w") as f:
    json.dump(label_map, f, indent=2)

# ---------------------------------------------------------------------------
# 6. Model metadata
# ---------------------------------------------------------------------------

model_metadata = {
    "model_version":         "1.0.0",
    "owner":                 "HR Analytics Team",
    "use_case":              "Employee benefits eligibility classification",
    "risk_level":            "medium",
    "protected_attributes":  ["age", "has_dependents"],
    "training_date":         "2026-03-28",
    "algorithm":             "GradientBoostingClassifier",
    "framework":             "scikit-learn",
    "input_features":        ["age", "employment_type", "years_of_service",
                              "weekly_hours", "department", "salary_band",
                              "has_dependents"],
    "output_classes":        ["not_eligible", "eligible"],
    "training_samples":      int(len(X_train)),
    "test_samples":          int(len(X_test)),
    "hyperparameters": {
        "n_estimators":  200,
        "learning_rate": 0.05,
        "max_depth":     4,
        "subsample":     0.8,
    },
}

with open(os.path.join(META, "model_metadata.json"), "w") as f:
    json.dump(model_metadata, f, indent=2)

# ---------------------------------------------------------------------------
# 7. Sample payloads for validation
# ---------------------------------------------------------------------------

sample_request = {
    "instances": [
        {
            "age":              34,
            "employment_type":  "full_time",
            "years_of_service": 5,
            "weekly_hours":     40,
            "department":       "engineering",
            "salary_band":      "band_3",
            "has_dependents":   1,
        },
        {
            "age":              27,
            "employment_type":  "part_time",
            "years_of_service": 0,
            "weekly_hours":     20,
            "department":       "sales",
            "salary_band":      "band_1",
            "has_dependents":   0,
        },
    ]
}

expected_response = {
    "predictions":   ["eligible", "not_eligible"],
    "probabilities": [[0.08, 0.92], [0.87, 0.13]],
    "label_order":   ["not_eligible", "eligible"],
}

with open(os.path.join(TEST, "sample_request.json"), "w") as f:
    json.dump(sample_request, f, indent=2)

with open(os.path.join(TEST, "expected_response.json"), "w") as f:
    json.dump(expected_response, f, indent=2)

print("\nAll artifacts saved to:", BASE)
