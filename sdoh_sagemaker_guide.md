# Deploying the SDoH Health Risk Classifier to AWS SageMaker

This guide covers the full path from your local codebase to a live SageMaker
inference endpoint. The model is a PyTorch MLP (`SDoHMLP`) trained on Social
Determinants of Health features. Because it uses PyTorch rather than scikit-learn,
you must use the SageMaker **PyTorch managed container** instead of the sklearn one.

---

## Prerequisites

**AWS account resources**
- An IAM role with the `AmazonSageMakerFullAccess` managed policy. Note the ARN.
- An S3 bucket in the same region as your SageMaker endpoint.
  Example: `s3://my-sdoh-model-bucket`

**Local tooling**

```bash
pip install awscli boto3 sagemaker torch
aws configure          # enter Access Key ID, Secret, region, output format
aws sts get-caller-identity   # confirm identity before proceeding
```

---

## Step 1 — Generate Labeled Training Data (optional)

```bash
python generate_sdoh_examples.py \
    --n 2000 \
    --seed 42 \
    --out-dir ./data
```

Writes `./data/sdoh_examples.csv` and `./data/sdoh_examples.json`.

---

## Step 2 — Train the Model and Build the SageMaker Package Locally

```bash
# With your own labeled data
python train_dl_sdoh.py \
    --platform sagemaker \
    --data-file ./data/sdoh_examples.csv \
    --output-dir ./dist

# With synthetic data (2 000 samples)
python train_dl_sdoh.py \
    --platform sagemaker \
    --samples 2000 \
    --output-dir ./dist
```

After completion, `./dist/sagemaker/` contains:

```
dist/sagemaker/
├── model.tar.gz                  ← model artifacts bundled for SageMaker
├── model/
│   ├── model.pt                  ← PyTorch state dict (weights only)
│   ├── model_config.json         ← architecture spec: input_dim, hidden_dims, dropout
│   ├── preprocessor.joblib       ← sklearn ColumnTransformer
│   └── label_map.json
├── code/
│   ├── inference.py              ← model_fn / input_fn / predict_fn / output_fn
│   └── requirements.txt
├── metadata/
│   └── model_metadata.json
└── test/
    ├── sample_request.json
    └── expected_response.json
```

> **Why model_config.json matters:** PyTorch saves only the weight tensor values
> in `model.pt`, not the class definition. `inference.py` reads `model_config.json`
> first to reconstruct the `SDoHMLP` architecture, then calls `load_state_dict()`
> to populate the weights. Both files must be present in `model_dir`.

---

## Step 3 — Add inference.py Inside model.tar.gz

SageMaker Script Mode requires `inference.py` to be inside `model.tar.gz` under
a `code/` subdirectory. Repack the archive:

```bash
cd dist/sagemaker

mkdir -p repack/code
tar -xzf model.tar.gz -C repack/

cp code/inference.py      repack/code/
cp code/requirements.txt  repack/code/

tar -czf model.tar.gz -C repack .
rm -rf repack
cd -
```

---

## Step 4 — Upload Artifacts to S3

```bash
BUCKET="my-sdoh-model-bucket"
PREFIX="sdoh-classifier/v1"

aws s3 cp dist/sagemaker/model.tar.gz \
    s3://${BUCKET}/${PREFIX}/model.tar.gz

# Verify
aws s3 ls s3://${BUCKET}/${PREFIX}/
```

---

## Step 5 — Register the Model in SageMaker

Use the **PyTorch managed container** — not the sklearn one — because the
inference script imports `torch`.

### Option A — AWS CLI

```bash
ROLE_ARN="arn:aws:iam::123456789012:role/MySageMakerRole"
REGION="us-east-1"
MODEL_URI="s3://${BUCKET}/${PREFIX}/model.tar.gz"

# PyTorch 2.2 CPU container (update region and version tag as needed)
IMAGE="763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-inference:2.2.0-cpu-py310"

aws sagemaker create-model \
    --model-name "sdoh-health-risk-v1" \
    --execution-role-arn "${ROLE_ARN}" \
    --primary-container \
      "Image=${IMAGE},ModelDataUrl=${MODEL_URI},Environment={SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/model/code,SAGEMAKER_PROGRAM=inference.py}" \
    --region "${REGION}"
```

### Option B — Python (SageMaker SDK)

```python
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel

ROLE_ARN  = "arn:aws:iam::123456789012:role/MySageMakerRole"
MODEL_URI = "s3://my-sdoh-model-bucket/sdoh-classifier/v1/model.tar.gz"
REGION    = "us-east-1"

boto3.setup_default_session(region_name=REGION)
sess = sagemaker.Session()

pytorch_model = PyTorchModel(
    model_data        = MODEL_URI,
    role              = ROLE_ARN,
    entry_point       = "inference.py",   # inside model.tar.gz/code/
    framework_version = "2.2",
    py_version        = "py310",
    sagemaker_session = sess,
)
```

---

## Step 6 — Run a SageMaker Training Job (alternative to local training)

If you want SageMaker to manage training on a GPU instance:

```python
from sagemaker.pytorch import PyTorch

ROLE_ARN  = "arn:aws:iam::123456789012:role/MySageMakerRole"
BUCKET    = "my-sdoh-model-bucket"
REGION    = "us-east-1"

boto3.setup_default_session(region_name=REGION)
sess = sagemaker.Session()

# Upload training data
train_s3 = sess.upload_data(
    path       = "./data/sdoh_examples.csv",
    bucket     = BUCKET,
    key_prefix = "sdoh-classifier/data",
)

estimator = PyTorch(
    entry_point       = "train_dl_sdoh.py",
    framework_version = "2.2",
    py_version        = "py310",
    instance_type     = "ml.p3.2xlarge",   # GPU; use ml.m5.xlarge for CPU-only
    instance_count    = 1,
    role              = ROLE_ARN,
    output_path       = f"s3://{BUCKET}/sdoh-classifier/output",
    hyperparameters   = {
        "platform":    "sagemaker",
        "data-file":   "/opt/ml/input/data/train/sdoh_examples.csv",
        "output-dir":  "/opt/ml/model",
    },
    sagemaker_session = sess,
)

estimator.fit({"train": train_s3})
```

> **GPU note:** `train_dl_sdoh.py` calls `torch.cuda.is_available()` and moves the
> model to GPU automatically when one is available. On CPU-only instances the
> training run is slower but produces identical artifacts.

---

## Step 7 — Deploy a Real-Time Inference Endpoint

```python
predictor = pytorch_model.deploy(
    initial_instance_count = 1,
    instance_type          = "ml.m5.large",   # CPU is sufficient for inference
    endpoint_name          = "sdoh-health-risk-endpoint",
    serializer             = sagemaker.serializers.JSONSerializer(),
    deserializer           = sagemaker.deserializers.JSONDeserializer(),
)

print("Endpoint is live:", predictor.endpoint_name)
```

Monitor status:

```bash
aws sagemaker describe-endpoint \
    --endpoint-name sdoh-health-risk-endpoint \
    --query "EndpointStatus"
```

---

## Step 8 — Run Inference Against the Live Endpoint

The `inference.py` entry-point expects `{"instances": [...]}`.

### Python — SageMaker SDK

```python
payload = {
    "instances": [
        {
            "age": 58, "prior_er_visits": 4, "household_size": 5,
            "pct_income_on_housing": 62,
            "income_level": "below_poverty", "insurance_status": "uninsured",
            "food_security": "very_low", "education_level": "less_than_hs",
            "health_literacy": "low", "housing_stability": "unstable",
            "neighborhood_safety": "unsafe", "rural_urban": "rural",
            "social_support": "none", "language_barrier": "yes",
            "has_primary_care": "no"
        },
        {
            "age": 34, "prior_er_visits": 0, "household_size": 3,
            "pct_income_on_housing": 22,
            "income_level": "middle", "insurance_status": "employer",
            "food_security": "secure", "education_level": "bachelors",
            "health_literacy": "high", "housing_stability": "homeowner",
            "neighborhood_safety": "high", "rural_urban": "urban",
            "social_support": "strong", "language_barrier": "no",
            "has_primary_care": "yes"
        }
    ]
}

response = predictor.predict(payload)
print(response)
# Expected:
# {
#   "predictions":   ["high_risk", "low_risk"],
#   "probabilities": [[0.09, 0.91], [0.88, 0.12]],
#   "label_order":   ["low_risk", "high_risk"]
# }
```

### Python — boto3 (no SageMaker SDK)

```python
import boto3, json

runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

payload = json.dumps({
    "instances": [
        {
            "age": 58, "prior_er_visits": 4, "household_size": 5,
            "pct_income_on_housing": 62,
            "income_level": "below_poverty", "insurance_status": "uninsured",
            "food_security": "very_low", "education_level": "less_than_hs",
            "health_literacy": "low", "housing_stability": "unstable",
            "neighborhood_safety": "unsafe", "rural_urban": "rural",
            "social_support": "none", "language_barrier": "yes",
            "has_primary_care": "no"
        }
    ]
})

response = runtime.invoke_endpoint(
    EndpointName = "sdoh-health-risk-endpoint",
    ContentType  = "application/json",
    Accept       = "application/json",
    Body         = payload,
)

result = json.loads(response["Body"].read())
print(result)
```

### AWS CLI

```bash
aws sagemaker-runtime invoke-endpoint \
    --endpoint-name sdoh-health-risk-endpoint \
    --content-type "application/json" \
    --accept "application/json" \
    --body '{"instances":[{"age":58,"prior_er_visits":4,"household_size":5,"pct_income_on_housing":62,"income_level":"below_poverty","insurance_status":"uninsured","food_security":"very_low","education_level":"less_than_hs","health_literacy":"low","housing_stability":"unstable","neighborhood_safety":"unsafe","rural_urban":"rural","social_support":"none","language_barrier":"yes","has_primary_care":"no"}]}' \
    /tmp/sdoh_response.json && cat /tmp/sdoh_response.json
```

---

## Step 9 — (Optional) Batch Inference with SageMaker Batch Transform

```python
from sagemaker.transformer import Transformer

input_s3 = sess.upload_data(
    path       = "./data/sdoh_examples.csv",
    bucket     = BUCKET,
    key_prefix = "sdoh-classifier/batch-input",
)

transformer = pytorch_model.transformer(
    instance_count = 1,
    instance_type  = "ml.m5.xlarge",
    output_path    = f"s3://{BUCKET}/sdoh-classifier/batch-output",
    accept         = "application/json",
    assemble_with  = "Line",
)

transformer.transform(
    data         = input_s3,
    content_type = "text/csv",
    split_type   = "Line",
    wait         = True,
)

print("Batch output at:", transformer.output_path)
```

> **CSV column order for batch transform:** Send only the 15 feature columns in
> this exact order, with no header row and no `high_risk`, `label`, or
> `disadvantage_count` columns:
> `age, prior_er_visits, household_size, pct_income_on_housing, income_level,
> insurance_status, food_security, education_level, health_literacy,
> housing_stability, neighborhood_safety, rural_urban, social_support,
> language_barrier, has_primary_care`

---

## Step 10 — Tear Down the Endpoint

```bash
aws sagemaker delete-endpoint \
    --endpoint-name sdoh-health-risk-endpoint
```

```python
predictor.delete_endpoint()
```

---

## Quick-Reference Checklist

| Step | Action | Done? |
|------|--------|-------|
| 1 | Generate labeled data (`generate_sdoh_examples.py`) | ☐ |
| 2 | Run `train_dl_sdoh.py --platform sagemaker` | ☐ |
| 3 | Repack `model.tar.gz` to include `code/inference.py` inside | ☐ |
| 4 | Upload `model.tar.gz` to S3 | ☐ |
| 5 | Create SageMaker model using PyTorch container | ☐ |
| 6 | Deploy endpoint (`pytorch_model.deploy(...)`) | ☐ |
| 7 | Send a test request and validate output | ☐ |
| 8 | Delete endpoint when finished | ☐ |

---

## Key PyTorch-Specific Notes

| Concern | Detail |
|---------|--------|
| Model artifact | `model.pt` contains **weights only** (state dict), not the class definition |
| Architecture recovery | `inference.py` reads `model_config.json` to rebuild `SDoHMLP` before loading weights |
| Eval mode | `inference.py` calls `model.eval()` after loading; omitting this leaves dropout active and produces non-deterministic predictions |
| No-grad context | `run_inference()` wraps the forward pass in `torch.no_grad()` to reduce memory and disable gradient tracking |
| GPU vs CPU | The endpoint uses CPU (`ml.m5.large`). `map_location="cpu"` in `load_artifacts()` ensures the state dict loads correctly even if the weights were saved from a GPU training run |
| Container image | Must use `pytorch-inference` — not `sagemaker-scikit-learn` — because `inference.py` imports `torch` |
