"""
Тесты для модуля predict.py
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.predict import LightGBMPredictor
from src.train import LightGBMTrainer


@pytest.fixture
def small_train_data():
    """Создаёт небольшой DataFrame для обучения."""
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
def trained_pipeline(tmp_path, small_train_data):
    """Обучает небольшую модель и возвращает обученный пайплайн."""
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
    trainer = LightGBMTrainer(
        test_size=0.2,
        random_state=42,
        use_optuna=False,
        include_weak=False,
        default_params=default_params,
    )
    pipeline, _ = trainer.train(df=small_train_data, target_col="loan_status")
    return pipeline


@pytest.fixture
def model_path(tmp_path, trained_pipeline):
    """Сохраняет обученный пайплайн в файл и возвращает путь к файлу."""
    model_file = tmp_path / "lgbm_final_model.pkl"
    joblib.dump(trained_pipeline, model_file)
    return model_file


@pytest.fixture
def model_dir(tmp_path, trained_pipeline):
    """Сохраняет пайплайн в директорию и возвращает путь к директории."""
    save_dir = tmp_path / "model_dir"
    save_dir.mkdir()
    model_file = save_dir / "lgbm_final_model.pkl"
    joblib.dump(trained_pipeline, model_file)
    return save_dir


class TestLightGBMPredictor:
    """Тесты для класса LightGBMPredictor."""

    def test_init_with_pipeline(self, trained_pipeline):
        """Проверка инициализации с готовым пайплайном."""
        predictor = LightGBMPredictor(trained_pipeline)
        assert predictor.model is trained_pipeline
        assert predictor.model_path is None
        assert predictor.feature_names is not None

    def test_init_with_model_file(self, model_path):
        """Проверка инициализации с путём к файлу модели."""
        predictor = LightGBMPredictor(model_path)
        assert isinstance(predictor.model, Pipeline)
        assert predictor.model_path == model_path
        assert predictor.feature_names is not None

    def test_init_with_model_directory(self, model_dir):
        """Проверка инициализации с путём к директории, содержащей модель."""
        predictor = LightGBMPredictor(model_dir)
        assert isinstance(predictor.model, Pipeline)
        assert predictor.model_path == model_dir / "lgbm_final_model.pkl"
        assert predictor.feature_names is not None

    def test_init_missing_file(self, tmp_path):
        """Проверка ошибки при отсутствии файла модели."""
        fake_path = tmp_path / "missing.pkl"
        with pytest.raises(FileNotFoundError, match="Модель не найдена по пути"):
            LightGBMPredictor(fake_path)

    def test_init_directory_without_model(self, tmp_path):
        """Проверка ошибки, когда в директории нет файла модели."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="lgbm_final_model.pkl не найден"):
            LightGBMPredictor(empty_dir)

    def test_extract_feature_names(self, trained_pipeline):
        """Проверка извлечения имён признаков из пайплайна."""
        predictor = LightGBMPredictor(trained_pipeline)
        assert predictor.feature_names is not None
        assert "person_age" in predictor.feature_names
        assert "loan_status" not in predictor.feature_names

    def test_validate_input_pass(self, trained_pipeline, small_train_data):
        """Проверка валидации корректного DataFrame."""
        predictor = LightGBMPredictor(trained_pipeline)
        X = small_train_data.drop(columns=["loan_status"])
        predictor._validate_input(X)

    def test_validate_input_missing_columns(self, trained_pipeline, small_train_data):
        """Проверка ошибки при отсутствии обязательных колонок."""
        predictor = LightGBMPredictor(trained_pipeline)
        X = small_train_data.drop(columns=["person_age", "person_income"])
        with pytest.raises(ValueError, match="Отсутствуют обязательные колонки"):
            predictor._validate_input(X)

    def test_predict_proba(self, trained_pipeline, small_train_data):
        """Проверка предсказания вероятностей."""
        predictor = LightGBMPredictor(trained_pipeline)
        X = small_train_data.drop(columns=["loan_status"])
        proba = predictor.predict_proba(X)
        assert isinstance(proba, np.ndarray)
        assert proba.shape == (len(X),)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_predict(self, trained_pipeline, small_train_data):
        """Проверка бинарных предсказаний с порогом по умолчанию."""
        predictor = LightGBMPredictor(trained_pipeline)
        X = small_train_data.drop(columns=["loan_status"])
        preds = predictor.predict(X)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(X),)
        assert set(preds).issubset({0, 1})

    def test_predict_with_custom_threshold(self, trained_pipeline, small_train_data):
        """Проверка бинарных предсказаний с пользовательским порогом."""
        predictor = LightGBMPredictor(trained_pipeline)
        X = small_train_data.drop(columns=["loan_status"])
        preds = predictor.predict(X, threshold=0.7)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(X),)
        assert set(preds).issubset({0, 1})

    def test_predict_single_from_dict(self, trained_pipeline, small_train_data):
        """Проверка предсказания для одного объекта в виде словаря."""
        predictor = LightGBMPredictor(trained_pipeline)
        row = small_train_data.drop(columns=["loan_status"]).iloc[0].to_dict()
        result = predictor.predict_single(row)
        assert "prediction" in result
        assert "probability" in result
        assert result["prediction"] in (0, 1)
        assert 0 <= result["probability"] <= 1

    def test_predict_single_from_series(self, trained_pipeline, small_train_data):
        """Проверка предсказания для одного объекта в виде Series."""
        predictor = LightGBMPredictor(trained_pipeline)
        series = small_train_data.drop(columns=["loan_status"]).iloc[0]
        result = predictor.predict_single(series)
        assert "prediction" in result
        assert "probability" in result

    def test_predict_single_from_dataframe(self, trained_pipeline, small_train_data):
        """Проверка предсказания для одного объекта в виде DataFrame с одной строкой."""
        predictor = LightGBMPredictor(trained_pipeline)
        df = small_train_data.drop(columns=["loan_status"]).iloc[[0]]
        result = predictor.predict_single(df)
        assert "prediction" in result
        assert "probability" in result

    def test_predict_single_dataframe_multiple_rows(self, trained_pipeline, small_train_data):
        """Проверка ошибки при передаче DataFrame с более чем одной строкой."""
        predictor = LightGBMPredictor(trained_pipeline)
        df = small_train_data.drop(columns=["loan_status"]).iloc[:2]
        with pytest.raises(ValueError, match="ровно одну строку"):
            predictor.predict_single(df)

    def test_predict_single_invalid_type(self, trained_pipeline):
        """Проверка ошибки при передаче неподдерживаемого типа."""
        predictor = LightGBMPredictor(trained_pipeline)
        with pytest.raises(TypeError, match="Не поддерживаемый тип данных"):
            predictor.predict_single([1, 2, 3])

    def test_get_feature_names(self, trained_pipeline):
        """Проверка метода get_feature_names."""
        predictor = LightGBMPredictor(trained_pipeline)
        names = predictor.get_feature_names()
        assert names is not None
        assert "person_age" in names

    def test_predict_proba_with_missing_cols(self, trained_pipeline, small_train_data):
        """Проверка ошибки при вызове predict_proba с отсутствующими колонками."""
        predictor = LightGBMPredictor(trained_pipeline)
        X = small_train_data.drop(columns=["person_age"])
        with pytest.raises(ValueError, match="Отсутствуют обязательные колонки"):
            predictor.predict_proba(X)

    def test_validate_input_no_feature_names(self, trained_pipeline, small_train_data):
        """Проверка, что при отсутствии feature_names валидация пропускается."""
        predictor = LightGBMPredictor(trained_pipeline)
        predictor.feature_names = None
        X = small_train_data.drop(columns=["loan_status"])
        predictor._validate_input(X)
