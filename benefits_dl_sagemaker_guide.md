# Deploying the Benefits Eligibility Classifier (Neural Network) to AWS SageMaker

This guide covers the full path from your local codebase to a live SageMaker
inference endpoint. The model is a PyTorch MLP (`BenefitsMLP`) that predicts
employee benefits eligibility. Because it uses PyTorch, you must use the SageMaker
**PyTorch managed container** rather than the sklearn one.

---

## Prerequisites

**AWS account resources**
- An IAM role with the `AmazonSageMakerFullAccess` managed policy attached. Note
  the role ARN — it is needed in several commands below.
- An S3 bucket in the same region as your SageMaker endpoint.
  Example: `s3://my-benefits-model-bucket`

**Local tooling**

```bash
pip install awscli boto3 sagemaker torch
aws configure          # enter Access Key ID, Secret, region, output format
aws sts get-caller-identity   # confirm identity before proceeding
```

---

## Step 1 — Generate Labeled Training Data (optional)

If you do not already have a labeled CSV, use the bundled generator.

```bash
python generate_benefits_examples.py \
    --n 2000 \
    --seed 42 \
    --out-dir ./data
```

Writes `./data/benefits_examples.csv` and `./data/benefits_examples.json`.

---

## Step 2 — Train the Model and Build the SageMaker Package Locally

```bash
# With your own labeled data
python train_dl_benefits.py \
    --platform sagemaker \
    --data-file ./data/benefits_examples.csv \
    --output-dir ./dist

# With synthetic data (2 000 samples)
python train_dl_benefits.py \
    --platform sagemaker \
    --samples 2000 \
    --output-dir ./dist
```

After completion `./dist/sagemaker/` contains:

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
    ├── sample_request.json       ← {"instances": [...]}
    └── expected_response.json
```

> **Why `model_config.json` is required:** PyTorch saves only the weight tensor
> values in `model.pt`, not the class definition. `inference.py` reads
> `model_config.json` first to reconstruct the `BenefitsMLP` architecture
> (`input_dim`, `hidden_dims`, `dropout`), then calls `load_state_dict()` to
> populate those weights. Both files must be present inside the tar archive.

---

## Step 3 — Add inference.py Inside model.tar.gz

SageMaker Script Mode requires `inference.py` to live **inside** `model.tar.gz`
under a `code/` subdirectory. The training script places it alongside the archive
but not inside it. Repack with:

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

Verify the contents:

```bash
tar -tzf dist/sagemaker/model.tar.gz
# Should include: model.pt, model_config.json, preprocessor.joblib,
#                 label_map.json, code/inference.py, code/requirements.txt
```

---

## Step 4 — Upload Artifacts to S3

```bash
BUCKET="my-benefits-model-bucket"
PREFIX="benefits-classifier-nn/v1"

aws s3 cp dist/sagemaker/model.tar.gz \
    s3://${BUCKET}/${PREFIX}/model.tar.gz

# Verify
aws s3 ls s3://${BUCKET}/${PREFIX}/
```

---

## Step 5 — Register the Model in SageMaker

Use the **PyTorch managed container** — the sklearn container does not include
`torch` and will fail to import `inference.py`.

### Option A — AWS CLI

```bash
ROLE_ARN="arn:aws:iam::123456789012:role/MySageMakerRole"
REGION="us-east-1"
MODEL_URI="s3://${BUCKET}/${PREFIX}/model.tar.gz"

# PyTorch 2.2 CPU inference container
IMAGE="763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-inference:2.2.0-cpu-py310"

aws sagemaker create-model \
    --model-name "benefits-eligibility-nn-v1" \
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
MODEL_URI = "s3://my-benefits-model-bucket/benefits-classifier-nn/v1/model.tar.gz"
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

If you want SageMaker to manage training — useful for GPU instances or large
datasets:

```python
from sagemaker.pytorch import PyTorch

ROLE_ARN = "arn:aws:iam::123456789012:role/MySageMakerRole"
BUCKET   = "my-benefits-model-bucket"
REGION   = "us-east-1"

boto3.setup_default_session(region_name=REGION)
sess = sagemaker.Session()

# Upload training data
train_s3 = sess.upload_data(
    path       = "./data/benefits_examples.csv",
    bucket     = BUCKET,
    key_prefix = "benefits-classifier-nn/data",
)

estimator = PyTorch(
    entry_point       = "train_dl_benefits.py",
    framework_version = "2.2",
    py_version        = "py310",
    instance_type     = "ml.p3.2xlarge",   # GPU; use ml.m5.xlarge for CPU-only
    instance_count    = 1,
    role              = ROLE_ARN,
    output_path       = f"s3://{BUCKET}/benefits-classifier-nn/output",
    hyperparameters   = {
        "platform":   "sagemaker",
        "data-file":  "/opt/ml/input/data/train/benefits_examples.csv",
        "output-dir": "/opt/ml/model",
    },
    sagemaker_session = sess,
)

estimator.fit({"train": train_s3})
```

> **GPU note:** `train_dl_benefits.py` calls `torch.cuda.is_available()` and
> moves the model to GPU automatically. On CPU-only instances the training run is
> slower but produces identical artifacts.

---

## Step 7 — Deploy a Real-Time Inference Endpoint

```python
predictor = pytorch_model.deploy(
    initial_instance_count = 1,
    instance_type          = "ml.m5.large",   # CPU is sufficient for inference
    endpoint_name          = "benefits-eligibility-nn-endpoint",
    serializer             = sagemaker.serializers.JSONSerializer(),
    deserializer           = sagemaker.deserializers.JSONDeserializer(),
)

print("Endpoint is live:", predictor.endpoint_name)
```

Monitor status:

```bash
aws sagemaker describe-endpoint \
    --endpoint-name benefits-eligibility-nn-endpoint \
    --query "EndpointStatus"
```

---

## Step 8 — Run Inference Against the Live Endpoint

`inference.py` expects `{"instances": [...]}`.

### Python — SageMaker SDK

```python
payload = {
    "instances": [
        {
            "age": 34, "employment_type": "full_time", "years_of_service": 5,
            "weekly_hours": 40, "department": "engineering",
            "salary_band": "band_3", "has_dependents": 1
        },
        {
            "age": 27, "employment_type": "part_time", "years_of_service": 0,
            "weekly_hours": 20, "department": "sales",
            "salary_band": "band_1", "has_dependents": 0
        }
    ]
}

response = predictor.predict(payload)
print(response)
# Expected:
# {
#   "predictions":   ["eligible", "not_eligible"],
#   "probabilities": [[0.08, 0.92], [0.87, 0.13]],
#   "label_order":   ["not_eligible", "eligible"]
# }
```

### Python — boto3 (no SageMaker SDK)

```python
import boto3, json

runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

payload = json.dumps({
    "instances": [
        {
            "age": 34, "employment_type": "full_time", "years_of_service": 5,
            "weekly_hours": 40, "department": "engineering",
            "salary_band": "band_3", "has_dependents": 1
        }
    ]
})

response = runtime.invoke_endpoint(
    EndpointName = "benefits-eligibility-nn-endpoint",
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
    --endpoint-name benefits-eligibility-nn-endpoint \
    --content-type "application/json" \
    --accept "application/json" \
    --body '{"instances":[{"age":34,"employment_type":"full_time","years_of_service":5,"weekly_hours":40,"department":"engineering","salary_band":"band_3","has_dependents":1}]}' \
    /tmp/benefits_response.json && cat /tmp/benefits_response.json
```

---

## Step 9 — (Optional) Batch Inference with SageMaker Batch Transform

```python
from sagemaker.transformer import Transformer

input_s3 = sess.upload_data(
    path       = "./data/benefits_examples.csv",
    bucket     = BUCKET,
    key_prefix = "benefits-classifier-nn/batch-input",
)

transformer = pytorch_model.transformer(
    instance_count = 1,
    instance_type  = "ml.m5.xlarge",
    output_path    = f"s3://{BUCKET}/benefits-classifier-nn/batch-output",
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

> **CSV column order for batch transform:** Send only the seven feature columns
> with no header row and no `eligible` or `label` columns:
> `age, employment_type, years_of_service, weekly_hours, department,
> salary_band, has_dependents`

---

## Step 10 — Tear Down the Endpoint

```bash
aws sagemaker delete-endpoint \
    --endpoint-name benefits-eligibility-nn-endpoint
```

```python
predictor.delete_endpoint()
```

---

## Quick-Reference Checklist

| Step | Action | Done? |
|------|--------|-------|
| 1 | Generate labeled data (`generate_benefits_examples.py`) | ☐ |
| 2 | Run `train_dl_benefits.py --platform sagemaker` | ☐ |
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
| Model artifact | `model.pt` is a **state dict only** — it contains weights but not the class definition |
| Architecture recovery | `inference.py` reads `model_config.json` to instantiate `BenefitsMLP`, then calls `load_state_dict()` |
| Both files required | `model.pt` and `model_config.json` must both be inside `model.tar.gz`; missing either causes a startup failure |
| Eval mode | `load_artifacts()` calls `model.eval()` after loading; without this, `BatchNorm1d` and `Dropout` remain in training mode and produce non-deterministic predictions |
| No-grad context | `run_inference()` wraps the forward pass in `torch.no_grad()` to disable gradient tracking during inference |
| CPU loading | `map_location="cpu"` in `load_artifacts()` ensures the state dict loads correctly on CPU endpoints even when trained on GPU |
| Container image | Must use `pytorch-inference:2.2.0-cpu-py310` — not the sklearn image — because `inference.py` imports `torch` |
