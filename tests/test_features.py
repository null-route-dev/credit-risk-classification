"""
Тесты для модуля features.py
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features import FeatureEngineer


@pytest.fixture
def sample_dataframe():
    """Создаёт типовой DataFrame для тестирования FeatureEngineer."""
    data = {
        "debt_to_income_ratio": [0.3, 0.5, 0.7, 0.2, 0.6],
        "loan_percent_income": [0.1, 0.2, 0.3, 0.15, 0.25],
        "person_income": [50000, 60000, 70000, 40000, 80000],
        "other_debt": [10000, 15000, 20000, 5000, 25000],
        "loan_int_rate": [5.0, 6.5, 7.0, 4.8, 8.0],
        "loan_grade": ["A", "B", "C", "A", "B"],
        "city": ["NY", "LA", "CHI", "NY", "LA"],
        "person_home_ownership": ["RENT", "OWN", "RENT", "OWN", "RENT"],
        "cb_person_default_on_file": ["N", "Y", "N", "N", "Y"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_dataframe_missing_cols(sample_dataframe):
    """DataFrame с отсутствующей колонкой loan_grade."""
    return sample_dataframe.drop(columns=["loan_grade"])


class TestFeatureEngineer:
    """Тесты для класса FeatureEngineer."""

    def test_default_params(self):
        """Проверка параметров инициализации по умолчанию."""
        fe = FeatureEngineer()
        assert fe.include_weak is False
        assert fe.params == {}
        assert fe.feature_names_in_ is None
        assert fe._feature_names_out is None

    def test_custom_params(self):
        """Проверка пользовательских параметров инициализации."""
        fe = FeatureEngineer(include_weak=True)
        assert fe.include_weak is True

    def test_fit_dataframe(self, sample_dataframe):
        """Проверка метода fit с DataFrame."""
        fe = FeatureEngineer()
        fe.fit(sample_dataframe)

        assert fe.feature_names_in_ == list(sample_dataframe.columns)
        assert "threshold_debt" in fe.params
        assert "threshold_loan_pct" in fe.params
        assert "threshold_rate" in fe.params
        assert "city_avg_dti" in fe.params
        assert "city_avg_loan_pct" in fe.params
        assert "avg_rate_by_grade" in fe.params
        assert fe._feature_names_out is not None

    def test_fit_numpy(self, sample_dataframe):
        """Проверка метода fit с numpy-массивом."""
        fe = FeatureEngineer()
        fe.fit(sample_dataframe)
        X_np = sample_dataframe.values
        fe.fit(X_np)
        assert "threshold_debt" in fe.params

    def test_fit_numpy_without_feature_names(self, sample_dataframe):
        """Проверка ошибки при вызове fit с numpy без предварительного указания имен признаков."""
        fe = FeatureEngineer()
        X_np = sample_dataframe.values
        with pytest.raises(ValueError, match="Feature names not provided"):
            fe.fit(X_np)

    def test_fit_invalid_type(self):
        """Проверка ошибки при передаче в fit неверного типа данных."""
        fe = FeatureEngineer()
        with pytest.raises(TypeError, match="Expected pandas DataFrame or numpy array"):
            fe.fit([1, 2, 3])

    def test_fit_missing_column(self, sample_dataframe_missing_cols):
        """Проверка ошибки при отсутствии необходимой колонки в данных."""
        fe = FeatureEngineer()
        with pytest.raises(KeyError):
            fe.fit(sample_dataframe_missing_cols)

    def test_transform_not_fitted(self, sample_dataframe):
        """Проверка ошибки при вызове transform до fit."""
        fe = FeatureEngineer()
        with pytest.raises(ValueError, match="Transformer not fitted"):
            fe.transform(sample_dataframe)

    def test_transform_dataframe(self, sample_dataframe):
        """Проверка метода transform с DataFrame (include_weak=False)."""
        fe = FeatureEngineer(include_weak=False)
        fe.fit(sample_dataframe)
        transformed = fe.transform(sample_dataframe)

        expected_cols = [
            "income_debt_balance",
            "is_high_debt",
            "is_high_loan_pct",
            "is_high_rate",
            "grade_ownership",
            "default_grade",
        ]
        for col in expected_cols:
            assert col in transformed.columns

        weak_cols = [
            "dti_loan_pct_ratio",
            "rate_grade_deviation",
            "city_avg_dti",
            "city_avg_loan_pct",
        ]
        for col in weak_cols:
            assert col not in transformed.columns

        for col in sample_dataframe.columns:
            assert col in transformed.columns

        row = transformed.iloc[0]
        assert row["income_debt_balance"] == 50000 - 10000

    def test_transform_dataframe_with_weak(self, sample_dataframe):
        """Проверка метода transform с include_weak=True."""
        fe = FeatureEngineer(include_weak=True)
        fe.fit(sample_dataframe)
        transformed = fe.transform(sample_dataframe)

        weak_cols = [
            "dti_loan_pct_ratio",
            "rate_grade_deviation",
            "city_avg_dti",
            "city_avg_loan_pct",
        ]
        for col in weak_cols:
            assert col in transformed.columns

        row = transformed.iloc[0]
        expected_ratio = row["debt_to_income_ratio"] / (row["loan_percent_income"] + 0.001)
        assert row["dti_loan_pct_ratio"] == pytest.approx(expected_ratio)

    def test_transform_numpy(self, sample_dataframe):
        """Проверка метода transform с numpy-массивом."""
        fe = FeatureEngineer()
        fe.fit(sample_dataframe)
        X_np = sample_dataframe.values
        transformed = fe.transform(X_np)
        assert isinstance(transformed, pd.DataFrame)
        assert "income_debt_balance" in transformed.columns

    def test_transform_invalid_type(self, sample_dataframe):
        """Проверка ошибки при передаче в transform неверного типа."""
        fe = FeatureEngineer()
        fe.fit(sample_dataframe)
        with pytest.raises(TypeError, match="Expected pandas DataFrame or numpy array"):
            fe.transform([1, 2, 3])

    def test_fit_transform_dataframe(self, sample_dataframe):
        """Проверка метода fit_transform с DataFrame."""
        fe = FeatureEngineer(include_weak=False)
        transformed = fe.fit_transform(sample_dataframe)

        assert fe.params
        assert "income_debt_balance" in transformed.columns

    def test_fit_transform_numpy(self, sample_dataframe):
        """Проверка ошибки при вызове fit_transform с numpy без имен признаков."""
        fe = FeatureEngineer()
        X_np = sample_dataframe.values
        with pytest.raises(ValueError, match="Feature names not provided"):
            fe.fit_transform(X_np)

    def test_get_feature_names_out_not_fitted(self):
        """Проверка ошибки при вызове get_feature_names_out до fit."""
        fe = FeatureEngineer()
        with pytest.raises(ValueError, match="Transformer not fitted"):
            fe.get_feature_names_out()

    def test_get_feature_names_out_fitted(self, sample_dataframe):
        """Проверка get_feature_names_out после fit (include_weak=False)."""
        fe = FeatureEngineer(include_weak=False)
        fe.fit(sample_dataframe)
        names = fe.get_feature_names_out()
        expected = set(sample_dataframe.columns).union({
            "income_debt_balance",
            "is_high_debt",
            "is_high_loan_pct",
            "is_high_rate",
            "grade_ownership",
            "default_grade",
        })
        assert set(names) == expected

    def test_get_feature_names_out_with_weak(self, sample_dataframe):
        """Проверка get_feature_names_out с include_weak=True."""
        fe = FeatureEngineer(include_weak=True)
        fe.fit(sample_dataframe)
        names = fe.get_feature_names_out()
        expected = set(sample_dataframe.columns).union({
            "income_debt_balance",
            "is_high_debt",
            "is_high_loan_pct",
            "is_high_rate",
            "grade_ownership",
            "default_grade",
            "dti_loan_pct_ratio",
            "rate_grade_deviation",
            "city_avg_dti",
            "city_avg_loan_pct",
        })
        assert set(names) == expected

    def test_sklearn_is_fitted_not_fitted(self):
        """Проверка __sklearn_is_fitted__ до fit."""
        fe = FeatureEngineer()
        assert not fe.__sklearn_is_fitted__()

    def test_sklearn_is_fitted_fitted(self, sample_dataframe):
        """Проверка __sklearn_is_fitted__ после fit."""
        fe = FeatureEngineer()
        fe.fit(sample_dataframe)
        assert fe.__sklearn_is_fitted__()

    def test_city_avg_features(self, sample_dataframe):
        """Проверка корректности вычисления городских средних."""
        fe = FeatureEngineer(include_weak=True)
        fe.fit(sample_dataframe)
        transformed = fe.transform(sample_dataframe)

        ny_rows = transformed[transformed["city"] == "NY"]
        assert (ny_rows["city_avg_dti"] == 0.25).all()
        assert (ny_rows["city_avg_loan_pct"] == 0.125).all()

    def test_rate_grade_deviation(self, sample_dataframe):
        """Проверка вычисления отклонения ставки от среднего по грейду."""
        fe = FeatureEngineer(include_weak=True)
        fe.fit(sample_dataframe)
        transformed = fe.transform(sample_dataframe)

        row = transformed.iloc[0]
        assert row["rate_grade_deviation"] == pytest.approx(0.1)

    def test_binary_flags(self, sample_dataframe):
        """Проверка бинарных флагов is_high_*."""
        fe = FeatureEngineer()
        fe.fit(sample_dataframe)
        transformed = fe.transform(sample_dataframe)

        debt_threshold = sample_dataframe["debt_to_income_ratio"].quantile(0.75)
        loan_pct_threshold = sample_dataframe["loan_percent_income"].quantile(0.75)
        rate_threshold = sample_dataframe["loan_int_rate"].quantile(0.75)

        for idx, row in transformed.iterrows():
            assert row["is_high_debt"] == int(row["debt_to_income_ratio"] > debt_threshold)
            assert row["is_high_loan_pct"] == int(row["loan_percent_income"] > loan_pct_threshold)
            assert row["is_high_rate"] == int(row["loan_int_rate"] > rate_threshold)

    def test_grade_ownership_combination(self, sample_dataframe):
        """Проверка создания комбинированного признака grade_ownership."""
        fe = FeatureEngineer()
        fe.fit(sample_dataframe)
        transformed = fe.transform(sample_dataframe)

        for idx, row in transformed.iterrows():
            expected = f"{row['loan_grade']}_{row['person_home_ownership']}"
            assert row["grade_ownership"] == expected

    def test_default_grade_combination(self, sample_dataframe):
        """Проверка создания комбинированного признака default_grade."""
        fe = FeatureEngineer()
        fe.fit(sample_dataframe)
        transformed = fe.transform(sample_dataframe)

        for idx, row in transformed.iterrows():
            expected = f"{row['cb_person_default_on_file']}_{row['loan_grade']}"
            assert row["default_grade"] == expected
