# AWS SageMaker — Benefits Eligibility Model

## 1. Project Overview

**Project Name:** AWS-Deployment (Benefits Eligibility Model)

### Description

This project packages and deploys a Benefits Eligibility Classifier to AWS SageMaker. The model predicts whether an employee is eligible for company benefits (`eligible` / `not_eligible`) based on 7 employment-profile features, using a scikit-learn `GradientBoostingClassifier`. It includes the model artifact bundle, the SageMaker inference entry-point script, governance metadata, test payloads, and training code.

## 2. Core Features

- Gradient Boosting classifier trained on 7 employment features (age, employment type, years of service, weekly hours, department, salary band, dependents) to predict benefits eligibility.
- SageMaker-compliant `inference.py` implementing the four required entry points: `model_fn`, `input_fn`, `predict_fn`, `output_fn`.
- Accepts `application/json` (single record or batch) and returns labeled predictions with class probabilities.
- Model governance metadata (`model_metadata.json`) tracking owner, risk level, protected attributes (`age`, `has_dependents`), training date, and hyperparameters for bias monitoring via SageMaker Clarify.
- Sample validation payloads (`test/sample_request.json`, `test/expected_response.json`) for endpoint smoke-testing.

## 3. Tech Stack

| Component | Technology |
|---|---|
| Model | scikit-learn `GradientBoostingClassifier`, XGBoost |
| Data handling | pandas, numpy, joblib |
| Platform | AWS SageMaker (sklearn container image, S3 model storage) |

## 4. Project Structure

```
AWS-Deployment/
├── train.py                 # model training script
├── inference.py              # SageMaker entry-point script (model_fn/input_fn/predict_fn/output_fn)
├── requirements.txt
├── test/
│   ├── sample_request.json
│   └── expected_response.json
├── doc/                      # architecture diagram
├── *_guide.md                 # deployment/runbook guides (DL, GB, SDOH variants)
├── README_dependency_environment_guidance.md
└── LICENSE
```

## 5. Prerequisites

- Python 3.x
- An AWS account with a SageMaker execution role and an S3 bucket
- Packages in `requirements.txt`

## 6. How to Run

### Train

```bash
pip install -r requirements.txt
python train.py
```

This produces `model.pkl`, `preprocessor.joblib`, and `label_map.json`.

### Package and deploy

1. Arrange artifacts as shown in Project Structure and compress the `model/` directory into `model.tar.gz`.
2. Upload `model.tar.gz` to an S3 bucket accessible by your SageMaker execution role.
3. Create a SageMaker Model pointing `entry_point` at `inference.py` and `source_dir` at the code directory, using the `sagemaker-scikit-learn` container image (`0.23-1` or later).
4. Deploy to an endpoint and invoke with `Content-Type: application/json`, `Accept: application/json`.
5. Validate using `test/sample_request.json` against `test/expected_response.json`.
6. Monitor `age` and `has_dependents` (protected attributes) for bias drift using SageMaker Clarify.

### Feature schema

| Feature | Type | Allowed values |
| :--- | :--- | :--- |
| age | int | 22-65 |
| employment_type | string | `full_time`, `part_time`, `contractor` |
| years_of_service | int | 0-40 |
| weekly_hours | int | 10-55 |
| department | string | `engineering`, `sales`, `hr`, `finance`, `operations` |
| salary_band | string | `band_1`...`band_5` |
| has_dependents | int | 0 or 1 |

## 7. Testing

`test/sample_request.json` and `test/expected_response.json` provide a smoke-test payload/expected-output pair for validating a deployed endpoint; there is no automated unit test runner in this repository.

## 8. Known Limitations

- No automated unit test suite — validation relies on the sample request/response pair against a live endpoint.
- Bias/drift monitoring on protected attributes (`age`, `has_dependents`) is a manual SageMaker Clarify setup step, not automated in this codebase.

## 9. License

MIT License (see `LICENSE`).
