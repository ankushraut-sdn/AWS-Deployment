# Porting the Benefits Eligibility Classifier to AWS SageMaker

This guide covers the complete path from your local codebase to a live SageMaker
endpoint — training, packaging, uploading, deploying, and calling the inference API.
Your existing `train_gb_benefits.py` already contains the SageMaker packager, so
much of the heavy lifting is already done.

---

## Prerequisites

Before you begin, make sure you have the following.

**AWS account resources**
- An IAM role with the `AmazonSageMakerFullAccess` managed policy attached (or equivalent custom policy). Note the role ARN — you will use it in several commands below.
- An S3 bucket in the same region where you will run SageMaker. Example: `s3://my-benefits-model-bucket`
- (Optional but recommended) An ECR repository if you plan to use a custom container.

**Local tooling**

```bash
pip install awscli boto3 sagemaker
aws configure          # enter Access Key ID, Secret, region, output format
```

Confirm your identity before proceeding:

```bash
aws sts get-caller-identity
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

This writes `./data/benefits_examples.csv` and `./data/benefits_examples.json`.

---

## Step 2 — Train the Model and Build the SageMaker Package Locally

Run the training script targeting only the SageMaker platform. If you have your
own labeled data pass it with `--data-file`; omit that flag to use the built-in
synthetic data generator.

```bash
# With your own labeled data
python train_gb_benefits.py \
    --platform sagemaker \
    --data-file ./data/benefits_examples.csv \
    --output-dir ./dist

# With synthetic data (2 000 samples)
python train_gb_benefits.py \
    --platform sagemaker \
    --samples 2000 \
    --output-dir ./dist
```

After the script completes the `./dist/sagemaker/` directory will contain:

```
dist/sagemaker/
├── model.tar.gz              ← model artifacts bundled for SageMaker
├── model/
│   ├── model.pkl
│   ├── preprocessor.joblib
│   └── label_map.json
├── code/
│   ├── inference.py          ← model_fn / input_fn / predict_fn / output_fn
│   └── requirements.txt
├── metadata/
│   └── model_metadata.json
└── test/
    ├── sample_request.json
    └── expected_response.json
```

---

## Step 3 — Add inference.py Inside model.tar.gz

SageMaker's Script Mode requires `inference.py` to live **inside** `model.tar.gz`
under a `code/` sub-directory. The training script puts the inference script in
`code/` alongside the archive but does not include it inside the tar. Repack it
with the following commands.

```bash
cd dist/sagemaker

# Extract existing archive
mkdir -p repack/code
tar -xzf model.tar.gz -C repack/

# Copy inference script into the code sub-directory
cp code/inference.py   repack/code/
cp code/requirements.txt repack/code/

# Rebuild the archive
tar -czf model.tar.gz -C repack .

# Clean up
rm -rf repack
cd -
```

---

## Step 4 — Upload Artifacts to S3

```bash
BUCKET="my-benefits-model-bucket"      # ← change to your bucket name
PREFIX="benefits-classifier/v1"

aws s3 cp dist/sagemaker/model.tar.gz \
    s3://${BUCKET}/${PREFIX}/model.tar.gz

# Verify the upload
aws s3 ls s3://${BUCKET}/${PREFIX}/
```

Record the full S3 URI (`s3://my-benefits-model-bucket/benefits-classifier/v1/model.tar.gz`);
you will need it in the next step.

---

## Step 5 — Register the Model in SageMaker

Use the AWS CLI or the Python SDK. Both approaches are shown below.

### Option A — AWS CLI

```bash
ROLE_ARN="arn:aws:iam::123456789012:role/MySageMakerRole"   # ← your IAM role
REGION="us-east-1"                                          # ← your region
MODEL_URI="s3://${BUCKET}/${PREFIX}/model.tar.gz"

# sklearn 1.3 built-in container image (update the tag if your region differs)
IMAGE="683313688378.dkr.ecr.${REGION}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"

aws sagemaker create-model \
    --model-name "benefits-eligibility-v1" \
    --execution-role-arn "${ROLE_ARN}" \
    --primary-container \
      "Image=${IMAGE},ModelDataUrl=${MODEL_URI},Environment={SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/model/code,SAGEMAKER_PROGRAM=inference.py}" \
    --region "${REGION}"
```

### Option B — Python (boto3 / SageMaker SDK)

```python
import boto3
import sagemaker
from sagemaker.sklearn import SKLearnModel

ROLE_ARN   = "arn:aws:iam::123456789012:role/MySageMakerRole"
MODEL_URI  = "s3://my-benefits-model-bucket/benefits-classifier/v1/model.tar.gz"
REGION     = "us-east-1"

boto3.setup_default_session(region_name=REGION)
sess = sagemaker.Session()

sklearn_model = SKLearnModel(
    model_data    = MODEL_URI,
    role          = ROLE_ARN,
    entry_point   = "inference.py",   # located inside model.tar.gz/code/
    framework_version = "1.2-1",
    py_version    = "py3",
    sagemaker_session = sess,
)
```

---

## Step 6 — Run a SageMaker Training Job (alternative to local training)

If you want SageMaker to manage training (useful for large datasets, spot
instances, or experiment tracking), use the Estimator API.

```python
import boto3
import sagemaker
from sagemaker.sklearn import SKLearn

ROLE_ARN  = "arn:aws:iam::123456789012:role/MySageMakerRole"
BUCKET    = "my-benefits-model-bucket"
REGION    = "us-east-1"

boto3.setup_default_session(region_name=REGION)
sess = sagemaker.Session()

# Upload training data
train_s3 = sess.upload_data(
    path   = "./data/benefits_examples.csv",
    bucket = BUCKET,
    key_prefix = "benefits-classifier/data",
)

estimator = SKLearn(
    entry_point       = "train_gb_benefits.py",
    framework_version = "1.2-1",
    py_version        = "py3",
    instance_type     = "ml.m5.large",
    instance_count    = 1,
    role              = ROLE_ARN,
    output_path       = f"s3://{BUCKET}/benefits-classifier/output",
    hyperparameters   = {
        "platform":  "sagemaker",
        "data-file": "/opt/ml/input/data/train/benefits_examples.csv",
    },
    sagemaker_session = sess,
)

estimator.fit({"train": train_s3})

# The estimator produces model.tar.gz in output_path automatically.
```

> **Note:** If you use this path, SageMaker calls `train_gb_benefits.py` as a
> script. Make sure `argparse` reads from `sys.argv` (it already does) and that
> the `--output-dir` is set to `/opt/ml/model` (SageMaker's expected output
> directory). Add `"output-dir": "/opt/ml/model"` to `hyperparameters` above.

---

## Step 7 — Deploy a Real-Time Inference Endpoint

```python
predictor = sklearn_model.deploy(
    initial_instance_count = 1,
    instance_type          = "ml.t3.medium",   # cheapest; upgrade for production
    endpoint_name          = "benefits-eligibility-endpoint",
    serializer             = sagemaker.serializers.JSONSerializer(),
    deserializer           = sagemaker.deserializers.JSONDeserializer(),
)

print("Endpoint is live:", predictor.endpoint_name)
```

Deployment typically takes 3–5 minutes. Monitor status:

```bash
aws sagemaker describe-endpoint \
    --endpoint-name benefits-eligibility-endpoint \
    --query "EndpointStatus"
```

---

## Step 8 — Run Inference Against the Live Endpoint

### Python — SageMaker SDK predictor

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

### Python — boto3 (no SageMaker SDK dependency)

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
    EndpointName = "benefits-eligibility-endpoint",
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
    --endpoint-name benefits-eligibility-endpoint \
    --content-type "application/json" \
    --accept "application/json" \
    --body '{"instances":[{"age":34,"employment_type":"full_time","years_of_service":5,"weekly_hours":40,"department":"engineering","salary_band":"band_3","has_dependents":1}]}' \
    /tmp/response.json && cat /tmp/response.json
```

---

## Step 9 — (Optional) Batch Inference with SageMaker Batch Transform

For scoring large files offline rather than hitting a live endpoint:

```python
from sagemaker.transformer import Transformer

# Upload your scoring data
input_s3 = sess.upload_data(
    path       = "./data/benefits_examples.csv",
    bucket     = BUCKET,
    key_prefix = "benefits-classifier/batch-input",
)

transformer = sklearn_model.transformer(
    instance_count = 1,
    instance_type  = "ml.m5.large",
    output_path    = f"s3://{BUCKET}/benefits-classifier/batch-output",
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

> **Note:** The CSV format for batch transform omits the header row and the
> `eligible` / `label` columns — send only the seven feature columns in the
> order: `age, employment_type, years_of_service, weekly_hours, department,
> salary_band, has_dependents`. You may need a light preprocessing step to strip
> those columns before uploading.

---

## Step 10 — Tear Down the Endpoint

SageMaker endpoints incur charges as long as they are running. Delete when done.

```bash
aws sagemaker delete-endpoint \
    --endpoint-name benefits-eligibility-endpoint
```

Or via Python:

```python
predictor.delete_endpoint()
```

---

## Quick-Reference Checklist

| Step | Command / Action | Done? |
|------|-----------------|-------|
| 1 | Generate labeled data (`generate_benefits_examples.py`) | ☐ |
| 2 | Run `train_gb_benefits.py --platform sagemaker` | ☐ |
| 3 | Repack `model.tar.gz` to include `code/inference.py` inside | ☐ |
| 4 | Upload `model.tar.gz` to S3 | ☐ |
| 5 | Create SageMaker model (`create-model` or `SKLearnModel`) | ☐ |
| 6 | Deploy endpoint (`model.deploy(...)`) | ☐ |
| 7 | Send a test request and validate output | ☐ |
| 8 | Delete endpoint when finished | ☐ |

---

## Key Environment Variables Inside the Container

| Variable | Value set by SageMaker | Purpose |
|----------|------------------------|---------|
| `SM_MODEL_DIR` | `/opt/ml/model` | Where `model_fn` loads artifacts from |
| `SM_CHANNEL_TRAIN` | `/opt/ml/input/data/train` | Training data location (training jobs) |
| `SM_OUTPUT_DATA_DIR` | `/opt/ml/output/data` | Write extra output artifacts here |
| `SAGEMAKER_PROGRAM` | `inference.py` | Entry-point script name |
| `SAGEMAKER_SUBMIT_DIRECTORY` | `/opt/ml/model/code` | Where SageMaker looks for the entry-point |

Your `inference.py` already reads `model_dir` correctly via the `model_fn(model_dir)`
signature — SageMaker passes `SM_MODEL_DIR` to that function automatically.
