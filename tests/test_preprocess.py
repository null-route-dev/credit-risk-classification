"""
Тесты для модуля preprocess.py
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocess import Preprocessor


@pytest.fixture
def sample_dataframe():
    """Создаёт типовой DataFrame для тестирования."""
    data = {
        "client_ID": [1, 2, 3, 4, 5],
        "loan_to_income_ratio": [0.5, 0.3, 0.8, 0.2, 0.6],
        "city_latitude": [40.7, 34.0, 41.8, 37.7, 33.4],
        "city_longitude": [-74.0, -118.2, -87.6, -122.4, -112.0],
        "state": ["NY", "CA", "IL", "CA", "AZ"],
        "country": ["USA", "USA", "USA", "USA", "USA"],
        "person_age": [25, 120, 45, 30, 150],
        "person_emp_length": [2, 60, 10, np.nan, 3],
        "loan_grade": ["A", "B", "C", "A", "B"],
        "loan_int_rate": [5.0, 6.5, np.nan, 4.8, 7.0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_dataframe_no_grade(sample_dataframe):
    """DataFrame без колонки loan_grade."""
    return sample_dataframe.drop(columns=["loan_grade"])


class TestPreprocessor:
    """Тесты для класса Preprocessor."""

    def test_default_params(self):
        """Проверка параметров инициализации по умолчанию."""
        pp = Preprocessor()
        expected_drop = [
            "client_ID",
            "loan_to_income_ratio",
            "city_latitude",
            "city_longitude",
            "state",
            "country",
        ]
        assert pp.drop_cols == expected_drop
        assert pp.age_threshold == 100
        assert pp.emp_length_threshold == 50
        assert pp.params == {}
        assert pp.feature_names_in_ is None
        assert pp._feature_names_out is None

    def test_custom_params(self):
        """Проверка пользовательских параметров инициализации."""
        custom_drop = ["col1", "col2"]
        pp = Preprocessor(drop_cols=custom_drop, age_threshold=80, emp_length_threshold=40)
        assert pp.drop_cols == custom_drop
        assert pp.age_threshold == 80
        assert pp.emp_length_threshold == 40

    def test_fit_dataframe(self, sample_dataframe):
        """Проверка метода fit с DataFrame."""
        pp = Preprocessor()
        pp.fit(sample_dataframe)

        assert pp.feature_names_in_ == list(sample_dataframe.columns)

        expected = sample_dataframe.groupby("loan_grade")["loan_int_rate"].median().to_dict()
        actual = pp.params["rate_medians"]
        assert set(actual.keys()) == set(expected.keys())
        for key in expected:
            if pd.isna(expected[key]) and pd.isna(actual[key]):
                continue
            assert actual[key] == expected[key]

        assert pp.params["global_rate_median"] == float(sample_dataframe["loan_int_rate"].median())
        assert pp.params["emp_median"] == float(sample_dataframe["person_emp_length"].median())
        assert pp._feature_names_out is not None

    def test_fit_numpy(self, sample_dataframe):
        """Проверка метода fit с numpy-массивом."""
        pp = Preprocessor()
        pp.fit(sample_dataframe)
        X_np = sample_dataframe.values
        pp.fit(X_np)
        assert pp.params["emp_median"] == float(sample_dataframe["person_emp_length"].median())

    def test_fit_numpy_without_feature_names(self, sample_dataframe):
        """Проверка ошибки при вызове fit с numpy без указания имен признаков."""
        pp = Preprocessor()
        X_np = sample_dataframe.values
        with pytest.raises(ValueError, match="Feature names not provided"):
            pp.fit(X_np)

    def test_fit_invalid_type(self):
        """Проверка ошибки при передаче в fit неверного типа данных."""
        pp = Preprocessor()
        with pytest.raises(TypeError, match="Expected pandas DataFrame or numpy array"):
            pp.fit([1, 2, 3])

    def test_fit_missing_column(self, sample_dataframe_no_grade):
        """Проверка ошибки при отсутствии необходимой колонки в данных."""
        pp = Preprocessor()
        with pytest.raises(KeyError):
            pp.fit(sample_dataframe_no_grade)

    def test_transform_not_fitted(self, sample_dataframe):
        """Проверка ошибки при вызове transform до того, как был вызван fit."""
        pp = Preprocessor()
        with pytest.raises(ValueError, match="Transformer not fitted"):
            pp.transform(sample_dataframe)

    def test_transform_dataframe(self, sample_dataframe):
        """Проверка метода transform с DataFrame."""
        pp = Preprocessor()
        pp.fit(sample_dataframe)
        transformed = pp.transform(sample_dataframe)

        for col in pp.drop_cols:
            assert col not in transformed.columns

        assert "emp_length_missing" in transformed.columns

        assert transformed["person_age"].max() <= pp.age_threshold
        assert transformed["person_emp_length"].max() <= pp.emp_length_threshold

        assert not transformed["loan_int_rate"].isna().any()

        median_emp = sample_dataframe["person_emp_length"].median()
        assert transformed.loc[3, "person_emp_length"] == median_emp
        assert transformed.loc[3, "emp_length_missing"] == 1
        assert (transformed["emp_length_missing"] == [0, 0, 0, 1, 0]).all()

        assert list(transformed.columns) == pp.get_feature_names_out()

    def test_transform_numpy(self, sample_dataframe):
        """Проверка метода transform с numpy-массивом."""
        pp = Preprocessor()
        pp.fit(sample_dataframe)
        X_np = sample_dataframe.values
        transformed = pp.transform(X_np)
        assert isinstance(transformed, pd.DataFrame)
        assert list(transformed.columns) == pp.get_feature_names_out()

    def test_transform_numpy_not_fitted(self, sample_dataframe):
        """Проверка ошибки при transform с numpy без предварительного fit."""
        pp = Preprocessor()
        X_np = sample_dataframe.values
        with pytest.raises(ValueError, match="Transformer not fitted"):
            pp.transform(X_np)

    def test_transform_invalid_type(self, sample_dataframe):
        """Проверка ошибки при передаче в transform неверного типа."""
        pp = Preprocessor()
        pp.fit(sample_dataframe)
        with pytest.raises(TypeError, match="Expected pandas DataFrame or numpy array"):
            pp.transform([1, 2, 3])

    def test_fit_transform_dataframe(self, sample_dataframe):
        """Проверка метода fit_transform с DataFrame."""
        pp = Preprocessor()
        transformed = pp.fit_transform(sample_dataframe)
        assert pp.params
        assert transformed["person_age"].max() <= pp.age_threshold
        assert not transformed["loan_int_rate"].isna().any()

    def test_fit_transform_numpy(self, sample_dataframe):
        """Проверка ошибки при вызове fit_transform с numpy без имен признаков."""
        pp = Preprocessor()
        X_np = sample_dataframe.values
        with pytest.raises(ValueError, match="Feature names not provided"):
            pp.fit_transform(X_np)

    def test_get_feature_names_out_not_fitted(self):
        """Проверка ошибки при вызове get_feature_names_out до fit."""
        pp = Preprocessor()
        with pytest.raises(ValueError, match="Transformer not fitted"):
            pp.get_feature_names_out()

    def test_get_feature_names_out_fitted(self, sample_dataframe):
        """Проверка get_feature_names_out после fit."""
        pp = Preprocessor()
        pp.fit(sample_dataframe)
        names = pp.get_feature_names_out()
        all_cols = set(sample_dataframe.columns)
        drop_set = set(pp.drop_cols)
        expected_cols = sorted((all_cols - drop_set) | {"emp_length_missing"})
        assert sorted(names) == expected_cols

    def test_sklearn_is_fitted_not_fitted(self):
        """Проверка __sklearn_is_fitted__ до fit."""
        pp = Preprocessor()
        assert not pp.__sklearn_is_fitted__()

    def test_sklearn_is_fitted_fitted(self, sample_dataframe):
        """Проверка __sklearn_is_fitted__ после fit."""
        pp = Preprocessor()
        pp.fit(sample_dataframe)
        assert pp.__sklearn_is_fitted__()

    def test_drop_columns_ignore_missing(self, sample_dataframe):
        """Проверка игнорирования отсутствующих колонок в списке drop_cols."""
        pp = Preprocessor(drop_cols=["nonexistent_col", "client_ID"])
        pp.fit(sample_dataframe)
        transformed = pp.transform(sample_dataframe)
        assert "client_ID" not in transformed.columns
        assert "nonexistent_col" not in transformed.columns

    def test_capping_with_negative_values(self):
        """Проверка каппинга при наличии отрицательных значений."""
        data = pd.DataFrame({
            "person_age": [-10, 150, 30],
            "person_emp_length": [-5, 60, 10],
            "loan_grade": ["A", "B", "C"],
            "loan_int_rate": [5.0, 6.0, 7.0],
        })
        pp = Preprocessor(age_threshold=100, emp_length_threshold=50)
        pp.fit(data)
        transformed = pp.transform(data)
        assert transformed["person_age"].tolist() == [-10, 100, 30]
        assert transformed["person_emp_length"].tolist() == [-5, 50, 10]

    def test_imputation_with_all_nan_in_group(self):
        """Проверка импутации, когда все значения в группе NaN."""
        data = pd.DataFrame({
            "person_age": [20, 30],
            "person_emp_length": [1, 2],
            "loan_grade": ["A", "A"],
            "loan_int_rate": [np.nan, np.nan],
        })
        pp = Preprocessor()
        pp.fit(data)
        transformed = pp.transform(data)
        assert transformed["loan_int_rate"].isna().all()

    def test_emp_length_missing_for_all_nan(self):
        """Проверка создания колонки emp_length_missing при всех NaN."""
        data = pd.DataFrame({
            "person_age": [20, 30],
            "person_emp_length": [np.nan, np.nan],
            "loan_grade": ["A", "B"],
            "loan_int_rate": [5.0, 6.0],
        })
        pp = Preprocessor()
        pp.fit(data)
        transformed = pp.transform(data)
        assert transformed["person_emp_length"].isna().all()
        assert (transformed["emp_length_missing"] == 1).all()
