# README — AWS CLI, Virtual Environments, and SageMaker Dependency Hygiene

## Why this README exists

During the SageMaker deployment work, several issues became intertwined:

1. **AWS CLI / boto3 / SageMaker SDK installation location**
2. **Model-training virtual environment dependencies**
3. **SageMaker container runtime dependencies**
4. **Cross-cloud packaging requirements**
5. **NumPy / pandas / scikit-learn / XGBoost binary compatibility**

The confusing part is that a package set can work locally, and may even work on Azure or Vertex AI, but still fail in SageMaker because SageMaker runs the model inside a specific prebuilt container with its own Python, NumPy, pandas, scikit-learn, PyTorch, and system libraries.

This document explains what to install where, what to keep out of each environment, and how to avoid breaking otherwise-working local models.

---

# 1. The core rule

Do **not** mix AWS tooling dependencies with model-training dependencies.

Use separate environments:

```text
System / Homebrew / AWS installer
    aws CLI only

ML training venv for GB models
    numpy, pandas, scikit-learn, xgboost, joblib

ML training venv for DL models
    torch, numpy, pandas, scikit-learn, xgboost, joblib

Optional AWS SDK venv
    boto3, sagemaker
```

The AWS CLI does not need to live inside the model-training virtual environment.

---

# 2. What went wrong

## 2.1 Installing AWS tooling inside the ML venv polluted the environment

Installing packages such as:

```bash
pip install awscli boto3 sagemaker
```

inside the model-training venv pulled in a large dependency tree. In the observed case, it upgraded NumPy to a newer incompatible version:

```text
numpy 2.4.4
```

That broke the older pandas / scikit-learn stack that the gradient-boost model needed.

Typical symptom:

```text
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```

or:

```text
ImportError: numpy.core.multiarray failed to import
```

Those errors usually mean binary packages were compiled or installed against incompatible NumPy versions.

---

## 2.2 The wrong `aws` executable caused confusion

There was also a separate issue where the shell found an old Python package named `aws`, not the real AWS CLI.

Symptom:

```text
SyntaxError: Missing parentheses in call to 'print'
```

That happened because the `aws` command being executed came from an outdated Python package, not AWS CLI v2.

Correct AWS CLI should look something like:

```bash
aws --version
```

Expected style:

```text
aws-cli/2.x ...
```

Bad sign:

```text
/Library/Frameworks/Python.framework/Versions/3.11/bin/aws
```

especially if it throws Python syntax errors.

---

# 3. Recommended installation layout

## 3.1 AWS CLI

Install AWS CLI outside model venvs.

Recommended on macOS:

```bash
brew install awscli
```

Then verify:

```bash
which aws
aws --version
```

Expected path is usually one of:

```text
/opt/homebrew/bin/aws
/usr/local/bin/aws
```

Do **not** run this inside the ML venv:

```bash
pip install awscli
```

---

## 3.2 AWS credentials

AWS credentials are configured globally or per user profile, not per model venv.

Run once:

```bash
aws configure
```

Then verify:

```bash
aws sts get-caller-identity
```

You do not need to repeat this for each model.

---

## 3.3 Optional boto3 / SageMaker SDK environment

If you want Python scripts that call AWS APIs, create a separate SDK venv:

```bash
python3.11 -m venv aws-sdk-venv
source aws-sdk-venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install boto3 sagemaker
```

Use this venv for deployment helper scripts, not for model training.

Example:

```python
import boto3

sm = boto3.client("sagemaker", region_name="us-east-2")
```

---

# 4. Model-training environments

## 4.1 Gradient-boost benefits model venv

The gradient-boost benefits model should use a stable, pinned stack that matches the SageMaker scikit-learn container.

Recommended:

```bash
python3.11 -m venv venv_gb
source venv_gb/bin/activate

pip install --upgrade pip setuptools wheel

pip install \
  numpy==1.23.5 \
  pandas==1.5.3 \
  scikit-learn==1.2.1 \
  xgboost==1.7.6 \
  joblib==1.3.2
```

Verify:

```bash
python - <<'PY'
import numpy, pandas, sklearn, xgboost, joblib

print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("sklearn", sklearn.__version__)
print("xgboost", xgboost.__version__)
print("joblib", joblib.__version__)
PY
```

Expected:

```text
numpy 1.23.5
pandas 1.5.3
sklearn 1.2.1
xgboost 1.7.6
joblib 1.3.2
```

This environment should **not** include:

```text
awscli
boto3
sagemaker
torch
```

unless there is a specific reason and compatibility has been tested.

---

## 4.2 Deep-learning model venv

The DL models should use a separate venv.

Example:

```bash
python3.11 -m venv venv_dl
source venv_dl/bin/activate

pip install --upgrade pip setuptools wheel

pip install \
  torch \
  numpy \
  pandas \
  scikit-learn \
  xgboost \
  joblib
```

A stricter pinned version may be preferable once the SageMaker PyTorch image is finalized.

Example:

```bash
pip install \
  torch==2.0.1 \
  numpy==1.24.4 \
  pandas==2.0.3 \
  scikit-learn==1.3.2 \
  xgboost==2.0.3 \
  joblib==1.3.2
```

The exact pins should match the serving container as closely as practical.

This environment should **not** include:

```text
awscli
sagemaker
boto3
```

unless it is being used only as a deployment scripting environment, not a training environment.

---

# 5. requirements.txt is context-specific

This is the biggest source of confusion.

There may be several different `requirements.txt` files, and they do not all serve the same purpose.

## 5.1 Local training requirements

This file is for reproducing local training.

It can include all local dependencies needed to train the model.

Example for a DL model:

```text
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
xgboost>=2.0
joblib>=1.3
torch
```

This is appropriate if local training truly needs those packages.

---

## 5.2 SageMaker inference package requirements

This is the `code/requirements.txt` included inside `model.tar.gz`.

It is used by the SageMaker container at endpoint startup time.

This file should be **minimal**.

It should include only packages that are not already present in the selected SageMaker container, or packages that must be overridden very deliberately.

For a SageMaker scikit-learn container, avoid unnecessarily listing:

```text
numpy
pandas
scikit-learn
```

because the container already has compatible versions. Overriding them can break binary compatibility.

For the GB benefits model, a safer SageMaker inference `code/requirements.txt` might be:

```text
xgboost==1.7.6
joblib==1.3.2
```

or possibly just:

```text
xgboost==1.7.6
```

if joblib is already present and compatible.

---

## 5.3 PyTorch SageMaker inference requirements

For a PyTorch inference container, do not blindly reinstall Torch unless needed.

The PyTorch container already includes PyTorch.

If the selected image is:

```text
pytorch-inference:2.0.1-cpu-py310-ubuntu20.04-sagemaker
```

then `torch` is already in the container.

A DL inference `code/requirements.txt` should include only extra dependencies your `inference.py` needs beyond what the container provides.

Possible minimal DL inference file:

```text
joblib==1.3.2
scikit-learn==1.3.2
xgboost==2.0.3
pandas==2.0.3
numpy==1.24.4
```

But this should be tested against the chosen PyTorch container.

If the model only needs:

```text
torch
joblib
json
os
numpy
```

and the container already provides Torch and NumPy, then the file may be much smaller.

---

# 6. Do not remove packages blindly

Removing package names from SageMaker `code/requirements.txt` is not the same as saying the model does not use those packages.

It means:

> Do not reinstall or override packages already present in the SageMaker base container unless necessary.

For example:

```python
import pandas as pd
```

may still be valid in `inference.py` even if `pandas` is not listed in `code/requirements.txt`, because pandas may already be installed in the base container.

However, if the base container does **not** include pandas, then pandas must be listed.

So the rule is:

```text
Include a package in code/requirements.txt only if:
1. inference.py imports it, and
2. the base container does not already provide a compatible version, or
3. you intentionally need a specific version.
```

Do not use `requirements.txt` as a full copy of the local training environment unless the container is designed for that.

---

# 7. Cross-cloud note: Azure and Vertex may differ

The same local package list may work on:

```text
Azure ML
Vertex AI
Local machine
```

but fail in SageMaker.

Why?

Because each platform uses a different base image and runtime stack.

For multi-cloud packaging, keep separate dependency files when needed:

```text
requirements-local.txt
requirements-sagemaker.txt
requirements-azure.txt
requirements-vertex.txt
```

or generate platform-specific `code/requirements.txt` during packaging.

Recommended structure:

```text
requirements/
├── requirements-local-gb.txt
├── requirements-local-dl.txt
├── requirements-sagemaker-gb.txt
├── requirements-sagemaker-dl.txt
├── requirements-azure-dl.txt
└── requirements-vertex-dl.txt
```

This avoids forcing SageMaker-specific dependency constraints onto other clouds.

---

# 8. Recommended package inclusion/exclusion guide

## 8.1 AWS CLI

Install globally, not in model venv.

Include:

```text
awscli
```

in no model requirements file.

Use:

```bash
brew install awscli
```

Exclude from:

```text
venv_gb
venv_dl
code/requirements.txt
model.tar.gz
```

---

## 8.2 boto3

Use in a separate SDK venv if needed.

Do not include in SageMaker inference `code/requirements.txt` unless `inference.py` explicitly calls AWS services at inference time, which is usually not recommended.

Recommended:

```text
aws-sdk-venv only
```

Exclude from:

```text
training venvs
SageMaker inference package
```

unless specifically required.

---

## 8.3 sagemaker SDK

Use in a separate SDK venv if needed.

Do not include in model-training venv unless using it to launch training jobs from that same environment.

Do not include in inference package.

Recommended:

```text
aws-sdk-venv only
```

Exclude from:

```text
code/requirements.txt
model.tar.gz
```

---

## 8.4 numpy / pandas / scikit-learn

These are binary-sensitive packages.

Include in local training venv with pinned versions.

Be cautious including them in SageMaker inference requirements.

For the GB model served with the scikit-learn 1.2 container:

```text
numpy==1.23.5
pandas==1.5.3
scikit-learn==1.2.1
```

should match training.

But the SageMaker inference `code/requirements.txt` may omit them if the container already includes compatible versions.

For DL PyTorch containers, confirm what versions are present or pin carefully.

---

## 8.5 xgboost

If the model was trained with XGBoost or the inference path imports XGBoost, include it.

Avoid loose version specifiers for SageMaker inference:

Bad:

```text
xgboost>=2.0
```

Better:

```text
xgboost==2.0.3
```

or for the GB model:

```text
xgboost==1.7.6
```

Loose constraints can silently pull newer versions and break compatibility.

---

## 8.6 torch

Include in local DL training venv.

Usually exclude from SageMaker PyTorch inference `code/requirements.txt` because the PyTorch container already provides it.

Include only if using a very custom base container or if the selected container lacks the required version.

---

## 8.7 joblib

Usually safe to include with a pin:

```text
joblib==1.3.2
```

But if a container already has joblib, avoid unnecessary upgrades unless required.

---

# 9. Suggested files

## 9.1 GB local training requirements

```text
# requirements-local-gb.txt
numpy==1.23.5
pandas==1.5.3
scikit-learn==1.2.1
xgboost==1.7.6
joblib==1.3.2
```

## 9.2 GB SageMaker inference requirements

```text
# requirements-sagemaker-gb.txt
xgboost==1.7.6
joblib==1.3.2
```

or, if scikit-learn compatibility must be pinned deliberately:

```text
numpy==1.23.5
pandas==1.5.3
scikit-learn==1.2.1
xgboost==1.7.6
joblib==1.3.2
```

Use the smaller file first.

---

## 9.3 DL local training requirements

```text
# requirements-local-dl.txt
torch
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
xgboost>=2.0
joblib>=1.3
```

This is acceptable for local training, but for reproducibility, pin exact versions after the environment is stable:

```bash
pip freeze > requirements-local-dl.lock.txt
```

---

## 9.4 DL SageMaker inference requirements

Start minimal:

```text
# requirements-sagemaker-dl.txt
joblib==1.3.2
scikit-learn==1.3.2
xgboost==2.0.3
pandas==2.0.3
numpy==1.24.4
```

Do not include `torch` if the PyTorch base image already includes the right version.

If dependency conflicts appear, reduce this file further and rely on the base container where possible.

---

# 10. Recommended deployment practice

Before uploading `model.tar.gz`, inspect it:

```bash
tar -tzf model.tar.gz
```

It should contain:

```text
./model.pt or ./model.pkl
./preprocessor.joblib
./label_map.json
./model_config.json
./code/inference.py
./code/requirements.txt
```

Then inspect the packaged inference requirements:

```bash
tar -xOzf model.tar.gz ./code/requirements.txt
```

Confirm it is the SageMaker-specific requirements file, not the full local training requirements file.

---

# 11. Rule of thumb

Use this decision table.

| Package | Local GB venv | Local DL venv | AWS SDK venv | SageMaker GB inference | SageMaker DL inference |
|---|---:|---:|---:|---:|---:|
| awscli | No | No | Usually no | No | No |
| boto3 | No | No | Yes | Usually no | Usually no |
| sagemaker | No | No | Yes | No | No |
| numpy | Yes, pinned | Yes, pinned/prefer pinned | No | Usually omit or pin carefully | Usually omit or pin carefully |
| pandas | Yes, pinned | Yes, pinned/prefer pinned | No | Usually omit or pin carefully | Usually omit or pin carefully |
| scikit-learn | Yes, pinned | Maybe, pinned | No | Match serving container | Match preprocessor version |
| xgboost | Yes, pinned | Maybe, pinned | No | Include if needed, pinned | Include if needed, pinned |
| torch | No | Yes | No | No | Usually omit if PyTorch image provides it |
| joblib | Yes, pinned | Yes, pinned | No | Include if needed, pinned | Include if needed, pinned |

---

# 12. Bottom line

The problem was not that AWS needs packages removed from requirements in general.

The actual issue was:

> The SageMaker inference requirements file should not blindly duplicate the local training requirements or AWS tooling requirements.

For SageMaker:

- use the right base image
- keep inference requirements minimal
- pin versions when overriding base packages
- never install AWS CLI or SageMaker SDK inside the model runtime package
- keep local training, deployment tooling, and cloud runtime dependencies separate

This prevents the dependency confusion that caused the NumPy, pandas, XGBoost, scikit-learn, and AWS CLI installation issues.
