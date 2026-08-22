"""
Тесты для модуля config.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import config


class TestConfig:
    """Тесты для конфигурационных параметров."""

    PROJECT_ROOT_NAME = "credit-risk-classification"
    SRC_DIR_NAME = "src"
    DATASETS_DIR_NAME = "datasets"
    MODELS_DIR_NAME = "models"
    DEFAULT_DATA_PATH_SUBSTRING = "Credit Risk Data.csv"
    DEFAULT_KAGGLE_DATASET = "alexdister/credit-risk-dataset"
    DEFAULT_TARGET_COLUMN = "loan_status"
    DEFAULT_API_HOST = "127.0.0.1"
    DEFAULT_LOG_LEVEL = "INFO"

    def test_project_root(self):
        """Проверка, что PROJECT_ROOT указывает на корень проекта."""
        assert config.PROJECT_ROOT.name == self.PROJECT_ROOT_NAME
        assert (config.PROJECT_ROOT / self.SRC_DIR_NAME).exists()

    def test_datasets_dir(self):
        """Проверка пути к директории datasets."""
        assert config.DATASETS_DIR == config.PROJECT_ROOT / self.DATASETS_DIR_NAME

    def test_models_dir(self):
        """Проверка пути к директории models."""
        assert config.MODELS_DIR == config.PROJECT_ROOT / self.MODELS_DIR_NAME

    def test_model_path_construction(self):
        """Проверка сборки полного пути к модели из MODEL_DIR и MODEL_FILENAME."""
        expected = Path(config.MODEL_DIR) / config.MODEL_FILENAME
        assert config.MODEL_PATH == expected

    def test_data_path_default(self):
        """Проверка значения DATA_PATH по умолчанию."""
        assert self.DEFAULT_DATA_PATH_SUBSTRING in str(config.DATA_PATH)

    def test_kaggle_dataset_default(self):
        """Проверка значения KAGGLE_DATASET по умолчанию."""
        assert config.KAGGLE_DATASET == self.DEFAULT_KAGGLE_DATASET

    def test_target_column_default(self):
        """Проверка значения TARGET_COLUMN по умолчанию."""
        assert config.TARGET_COLUMN == self.DEFAULT_TARGET_COLUMN

    def test_test_size_type(self):
        """Проверка типа TEST_SIZE."""
        assert isinstance(config.TEST_SIZE, float)
        assert 0 < config.TEST_SIZE < 1

    def test_random_state_type(self):
        """Проверка типа RANDOM_STATE."""
        assert isinstance(config.RANDOM_STATE, int)
        assert config.RANDOM_STATE >= 0

    def test_cv_folds_type(self):
        """Проверка типа CV_FOLDS."""
        assert isinstance(config.CV_FOLDS, int)
        assert config.CV_FOLDS >= 2

    def test_n_trials_type(self):
        """Проверка типа N_TRIALS."""
        assert isinstance(config.N_TRIALS, int)
        assert config.N_TRIALS >= 1

    def test_include_weak_type(self):
        """Проверка типа INCLUDE_WEAK."""
        assert isinstance(config.INCLUDE_WEAK, bool)

    def test_use_optuna_type(self):
        """Проверка типа USE_OPTUNA."""
        assert isinstance(config.USE_OPTUNA, bool)

    def test_api_host_default(self):
        """Проверка значения API_HOST по умолчанию."""
        assert config.API_HOST == self.DEFAULT_API_HOST

    def test_api_port_type(self):
        """Проверка типа API_PORT."""
        assert isinstance(config.API_PORT, int)
        assert 1 <= config.API_PORT <= 65535

    def test_api_reload_type(self):
        """Проверка типа API_RELOAD."""
        assert isinstance(config.API_RELOAD, bool)

    def test_log_level_default(self):
        """Проверка значения LOG_LEVEL по умолчанию."""
        assert config.LOG_LEVEL == self.DEFAULT_LOG_LEVEL
