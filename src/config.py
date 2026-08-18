"""Модуль конфигурации проекта."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"

DATA_PATH = os.getenv("DATA_PATH", str(DATASETS_DIR / "Credit Risk Data.csv"))
KAGGLE_DATASET = os.getenv("KAGGLE_DATASET", "alexdister/credit-risk-dataset")
MODEL_DIR = os.getenv("MODEL_DIR", str(MODELS_DIR))
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "lgbm_final_model.pkl")

TARGET_COLUMN = os.getenv("TARGET_COLUMN", "loan_status")
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
CV_FOLDS = int(os.getenv("CV_FOLDS", "5"))
N_TRIALS = int(os.getenv("N_TRIALS", "50"))
INCLUDE_WEAK = os.getenv("INCLUDE_WEAK", "false").lower() == "true"
USE_OPTUNA = os.getenv("USE_OPTUNA", "false").lower() == "true"

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"
API_LOG_LEVEL = os.getenv("API_LOG_LEVEL", "info")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv(
    "LOG_FORMAT",
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOG_DATE_FORMAT = os.getenv("LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S")

MODEL_PATH = Path(MODEL_DIR) / MODEL_FILENAME
