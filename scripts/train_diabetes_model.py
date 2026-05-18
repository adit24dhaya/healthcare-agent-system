import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = PROJECT_ROOT / "kaggle"
if str(KAGGLE_DIR) not in sys.path:
    sys.path.insert(0, str(KAGGLE_DIR))

from training_pipeline import load_dataset_from_path, train_production_model

DEFAULT_KAGGLE_DATA = Path(
    "/kaggle/input/diabetes-health-indicators-dataset/"
    "diabetes_binary_health_indicators_BRFSS2015.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the production diabetes risk model.")
    parser.add_argument(
        "--data", default=str(DEFAULT_KAGGLE_DATA), help="CSV file or dataset directory."
    )
    parser.add_argument("--output-dir", default="artifacts", help="Directory for model outputs.")
    parser.add_argument("--explanation-sample-size", type=int, default=2500)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_path, X, y = load_dataset_from_path(args.data)
    train_production_model(
        X,
        y,
        data_path=data_path,
        output_dir=args.output_dir,
        explanation_sample_size=args.explanation_sample_size,
    )
