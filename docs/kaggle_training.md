# Kaggle Model Training

The production model is trained on Kaggle with the CDC Diabetes Health Indicators
dataset, a cleaned BRFSS 2015 dataset with 253,680 survey responses and 21 health
indicator features.

## Why This Dataset

The original repo used a tiny synthetic CSV, which is useful for smoke tests but not
strong enough for a portfolio-grade ML project. The CDC/BRFSS dataset adds:

- enough rows for meaningful train/test evaluation
- lifestyle, access, demographic, and clinical indicator features
- a binary diabetes/prediabetes risk target
- a public Kaggle workflow that produces reproducible artifacts

## Training Outputs

Running `kaggle/kaggle_train.py` creates:

- `artifacts/risk_model.joblib`
- `artifacts/metrics.json`
- `artifacts/model_card.md`

The API loads `artifacts/risk_model.joblib` automatically through `MODEL_ARTIFACT_PATH`.
If the artifact is missing, the app falls back to `data/patients.csv` for local demos
and tests.

## Run On Kaggle

Production training is meant to run on Kaggle, not on your laptop:

```bash
./scripts/kaggle_run.sh
```

That script:

1. syncs `kernel-metadata.json` with your Kaggle username
2. pushes `kaggle/kaggle_train.py`
3. waits for the kernel to finish
4. downloads `risk_model.joblib`, `metrics.json`, and `model_card.md` into `artifacts/`

Manual commands:

```bash
kaggle kernels push -p kaggle
kaggle kernels status aditya2402/healthcare-ai-diabetes-risk-training
kaggle kernels output aditya2402/healthcare-ai-diabetes-risk-training -p artifacts
```

## Local Mirror (optional)

`scripts/train_diabetes_model.py` mirrors the Kaggle script for CI or offline debugging
only. Use Kaggle for the portfolio artifact.

## Model Selection (v4 pipeline)

The training script:

1. Compares logistic regression, histogram gradient boosting, LightGBM, and random forest
2. Ranks models with **5-fold stratified CV** (ROC AUC)
3. Runs **RandomizedSearchCV** on the top 2 models
4. Applies **isotonic calibration** to all candidates
5. Picks the winner on the holdout test set (ROC AUC, then F1)

Metrics include CV scores, Brier score, and tuned classification threshold.
