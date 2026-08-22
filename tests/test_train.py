"""
Тесты для модуля train.py
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.train import LightGBMTrainer


@pytest.fixture
def small_train_data():
    """Создаёт небольшой DataFrame для обучения с минимальным набором колонок."""
    n = 50
    np.random.seed(42)
    data = {
        "person_age": np.random.randint(18, 70, n),
        "person_income": np.random.randint(20000, 150000, n),
        "person_emp_length": np.random.randint(0, 30, n),
        "loan_amnt": np.random.randint(1000, 50000, n),
        "loan_int_rate": np.random.uniform(3, 15, n),
        "loan_percent_income": np.random.uniform(0.05, 0.6, n),
        "cb_person_cred_hist_length": np.random.randint(1, 30, n),
        "debt_to_income_ratio": np.random.uniform(0.1, 0.8, n),
        "open_accounts": np.random.randint(1, 10, n),
        "credit_utilization_ratio": np.random.uniform(0, 1, n),
        "loan_term_months": np.random.choice([36, 60], n),
        "past_delinquencies": np.random.randint(0, 5, n),
        "loan_grade": np.random.choice(["A", "B", "C", "D", "E", "F", "G"], n),
        "cb_person_default_on_file": np.random.choice(["Y", "N"], n),
        "gender": np.random.choice(["M", "F"], n),
        "person_home_ownership": np.random.choice(["RENT", "OWN", "MORTGAGE", "OTHER"], n),
        "loan_intent": np.random.choice(["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"], n),
        "marital_status": np.random.choice(["Married", "Single", "Divorced", "Widowed"], n),
        "education_level": np.random.choice(["High School", "Associate", "Bachelor", "Master", "Doctorate"], n),
        "employment_type": np.random.choice(["Employed", "Self-Employed", "Unemployed", "Retired"], n),
        "city": np.random.choice(["NY", "LA", "CHI", "HOU", "PHX"], n),
        "other_debt": np.random.randint(0, 50000, n),
        "loan_status": np.random.choice([0, 1], n, p=[0.8, 0.2]),
    }
    return pd.DataFrame(data)


@pytest.fixture
def trainer_fixed():
    """Создаёт экземпляр LightGBMTrainer с фиксированными параметрами и малым числом деревьев."""
    default_params = {
        "n_estimators": 3,
        "max_depth": 2,
        "learning_rate": 0.1,
        "num_leaves": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 10,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
    }
    return LightGBMTrainer(
        test_size=0.2,
        random_state=42,
        use_optuna=False,
        include_weak=False,
        default_params=default_params,
    )


class TestLightGBMTrainer:
    """Тесты для класса LightGBMTrainer."""

    def test_default_initialization(self):
        """Проверка параметров инициализации по умолчанию."""
        trainer = LightGBMTrainer()
        assert trainer.test_size == 0.2
        assert trainer.random_state == 42
        assert trainer.n_trials == 50
        assert trainer.cv_folds == 5
        assert trainer.include_weak is False
        assert trainer.use_optuna is False
        assert trainer.preprocessor is None
        assert "n_estimators" in trainer.default_params
        assert trainer._pipeline is None

    def test_custom_initialization(self):
        """Проверка пользовательских параметров инициализации."""
        trainer = LightGBMTrainer(
            test_size=0.3,
            random_state=123,
            n_trials=10,
            cv_folds=3,
            include_weak=True,
            use_optuna=True,
        )
        assert trainer.test_size == 0.3
        assert trainer.random_state == 123
        assert trainer.n_trials == 10
        assert trainer.cv_folds == 3
        assert trainer.include_weak is True
        assert trainer.use_optuna is True

    def test_create_preprocessor(self):
        """Проверка создания стандартного ColumnTransformer."""
        trainer = LightGBMTrainer()
        preprocessor = trainer._create_preprocessor()

        assert isinstance(preprocessor, ColumnTransformer)

        expected_transformers = ["num", "grade_ordinal", "onehot_binary", "onehot", "onehot_high"]
        actual_transformers = [name for name, _, _ in preprocessor.transformers]
        assert set(actual_transformers) == set(expected_transformers)

        num_cols = preprocessor.transformers[0][2]
        assert isinstance(num_cols, list)
        assert "person_age" in num_cols
        assert "loan_int_rate" in num_cols

        grade_encoder = preprocessor.transformers[1][1]
        assert hasattr(grade_encoder, "categories")
        assert grade_encoder.categories[0] == ["A", "B", "C", "D", "E", "F", "G"]

    def test_build_pipeline(self, small_train_data, trainer_fixed):
        """Проверка сборки пайплайна с переданными параметрами."""
        preprocessor = trainer_fixed._create_preprocessor()
        trainer_fixed._y_train = small_train_data["loan_status"]
        pipeline = trainer_fixed._build_pipeline(trainer_fixed.default_params, preprocessor)

        assert isinstance(pipeline, Pipeline)
        expected_steps = ["basic_preprocessing", "feature_engineer", "custom_preprocessing", "classifier"]
        assert [step[0] for step in pipeline.steps] == expected_steps

        classifier = pipeline.named_steps["classifier"]
        assert classifier.n_estimators == 3
        assert classifier.random_state == trainer_fixed.random_state
        assert classifier.n_jobs == -1
        assert "scale_pos_weight" in classifier.get_params()

    def test_train_without_optuna(self, small_train_data, trainer_fixed, tmp_path):
        """Проверка полного цикла обучения без Optuna."""
        pipeline, metrics = trainer_fixed.train(
            df=small_train_data,
            target_col="loan_status",
            save_path=tmp_path
        )

        assert isinstance(pipeline, Pipeline)
        assert "roc_auc" in metrics
        assert "recall" in metrics
        assert "precision" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["roc_auc"] <= 1

        assert trainer_fixed._pipeline is pipeline
        assert trainer_fixed._best_params is not None
        assert trainer_fixed._train_metrics is not None
        assert trainer_fixed._test_metrics is not None

        model_file = tmp_path / "lgbm_final_model.pkl"
        info_file = tmp_path / "model_info.json"
        importance_file = tmp_path / "feature_importance.csv"

        assert model_file.exists()
        assert info_file.exists()
        assert importance_file.exists()

        loaded_pipeline = joblib.load(model_file)
        assert isinstance(loaded_pipeline, Pipeline)

        with open(info_file) as f:
            info = json.load(f)
        assert info["model_type"] == "LightGBM"
        assert "best_params" in info
        assert "test_metrics" in info
        assert info["test_metrics"]["roc_auc"] == metrics["roc_auc"]

        importance_df = pd.read_csv(importance_file)
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        assert not importance_df.empty

    def test_train_without_save(self, small_train_data, trainer_fixed):
        """Проверка обучения без сохранения модели."""
        pipeline, metrics = trainer_fixed.train(
            df=small_train_data,
            target_col="loan_status"
        )
        assert isinstance(pipeline, Pipeline)
        assert metrics is not None
        assert trainer_fixed._pipeline is pipeline

    def test_custom_preprocessor(self, small_train_data, tmp_path):
        """Проверка использования кастомного препроцессора."""
        custom_preprocessor = ColumnTransformer([
            ("num", "passthrough", ["person_age", "person_income"])
        ])
        trainer = LightGBMTrainer(
            use_optuna=False,
            default_params={"n_estimators": 3},
            preprocessor=custom_preprocessor
        )
        pipeline, metrics = trainer.train(
            df=small_train_data,
            target_col="loan_status",
            save_path=tmp_path
        )

        assert pipeline.named_steps["custom_preprocessing"] is custom_preprocessor
        assert isinstance(metrics, dict)

    def test_save_metadata_with_numpy_types(self, small_train_data, trainer_fixed, tmp_path):
        """Проверка сохранения метаданных с numpy типами."""
        trainer_fixed.train(
            df=small_train_data,
            target_col="loan_status",
            save_path=tmp_path
        )
        info_file = tmp_path / "model_info.json"
        with open(info_file) as f:
            info = json.load(f)

        assert isinstance(info["cv_best_score"], (float, type(None)))
        assert isinstance(info["train_metrics"]["roc_auc"], float)

    def test_feature_importance_saved(self, small_train_data, trainer_fixed, tmp_path):
        """Проверка сохранения важности признаков."""
        trainer_fixed.train(
            df=small_train_data,
            target_col="loan_status",
            save_path=tmp_path
        )
        importance_file = tmp_path / "feature_importance.csv"
        df_imp = pd.read_csv(importance_file)
        assert len(df_imp) > 0
        assert df_imp["importance"].min() >= 0

    def test_scale_pos_weight_calculation(self, small_train_data, trainer_fixed):
        """Проверка автоматического расчёта scale_pos_weight."""
        trainer_fixed._y_train = small_train_data["loan_status"]
        preprocessor = trainer_fixed._create_preprocessor()
        pipeline = trainer_fixed._build_pipeline(trainer_fixed.default_params, preprocessor)

        scale_pos_weight = pipeline.named_steps["classifier"].scale_pos_weight
        y = small_train_data["loan_status"]
        expected = (y == 0).sum() / (y == 1).sum()
        assert scale_pos_weight == pytest.approx(expected)

    def test_train_with_fixed_params_override(self, small_train_data, tmp_path):
        """Проверка использования переданных фиксированных параметров."""
        fixed_params = {"n_estimators": 5, "max_depth": 3}
        trainer = LightGBMTrainer(use_optuna=False, default_params={})
        pipeline, _ = trainer.train(
            df=small_train_data,
            target_col="loan_status",
            save_path=tmp_path,
            fixed_params=fixed_params
        )
        classifier = pipeline.named_steps["classifier"]
        assert classifier.n_estimators == 5
        assert classifier.max_depth == 3

    def test_train_raises_on_missing_target(self, small_train_data, trainer_fixed):
        """Проверка ошибки при отсутствии целевой колонки."""
        with pytest.raises(KeyError):
            trainer_fixed.train(df=small_train_data, target_col="non_existent")
