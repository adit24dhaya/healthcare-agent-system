# Healthcare Diabetes Risk Model Card

## Intended Use

This model estimates diabetes risk from CDC BRFSS health indicators for an educational
healthcare AI prototype. It is not a diagnostic medical device.

## Dataset

- Source: Kaggle `alexteboul/diabetes-health-indicators-dataset`
- Target: `Diabetes_binary`
- Rows: 253680
- Features: 21

## Training

- Selection: 5-fold_cv_then_holdout_test
- Hyperparameter tuning: hist_gradient_boosting, lightgbm
- Probability calibration: isotonic_cv3
- LightGBM included: True

## Selected Model

- Model: `hist_gradient_boosting`
- CV ROC AUC: 0.83059 (+/- 0.00232)
- Holdout ROC AUC: 0.82698
- Average precision: 0.42299
- Brier score: 0.09739
- Tuned F1: 0.46904
- Tuned threshold: 0.22379
