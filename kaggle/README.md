# Kaggle Training

This folder contains the Kaggle script that trains the production risk artifact from
the CDC Diabetes Health Indicators dataset.

## Dataset

- Kaggle source: `alexteboul/diabetes-health-indicators-dataset`
- File used by the script: `diabetes_binary_health_indicators_BRFSS2015.csv`
- Target: `Diabetes_binary`
- Features: 21 BRFSS health, access, lifestyle, and demographic indicators

## Training pipeline

`training_pipeline.py` performs CV model selection, hyperparameter tuning on the top two
models, isotonic calibration, and writes `artifacts/` outputs. `kaggle_train.py` is the
Kaggle entrypoint.

## Run On Kaggle (recommended)

Training runs entirely on Kaggle GPUs/CPUs. Your laptop only pushes the script and
downloads the trained artifact afterward.

One command from the repo root:

```bash
chmod +x scripts/kaggle_run.sh
./scripts/kaggle_run.sh
```

Or step by step:

```bash
kaggle kernels push -p kaggle
kaggle kernels status aditya2402/healthcare-ai-diabetes-risk-training
kaggle kernels output aditya2402/healthcare-ai-diabetes-risk-training -p artifacts
```

The script auto-fills your Kaggle username from `kaggle config view` if you fork the
repo and change the kernel slug.

The output directory should contain:

- `risk_model.joblib`
- `metrics.json`
- `model_card.md`

After download, the FastAPI app automatically loads `artifacts/risk_model.joblib`.

Live kernel: [aditya2402/healthcare-ai-diabetes-risk-training](https://www.kaggle.com/code/aditya2402/healthcare-ai-diabetes-risk-training)
