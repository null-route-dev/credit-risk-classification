"""
Модуль для создания новых признаков (Feature Engineering).
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Union, List


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Создание новых признаков на основе очищенных данных.

    Параметры:
        include_weak : bool, default=False
            Если False, слабые признаки не создаются.
            Слабые признаки: dti_loan_pct_ratio, rate_grade_deviation,
            city_avg_dti, city_avg_loan_pct.
            Остальные новые признаки создаются всегда.

    Attributes:
        include_weak (bool): Флаг включения слабых признаков.
        params (dict): Параметры, вычисленные в fit.
        feature_names_in_ (List[str]): Имена колонок входных данных.
        _feature_names_out (List[str]): Имена колонок после трансформации.
    """

    def __init__(self, include_weak: bool = False):
        self.include_weak = include_weak
        self.params = {}
        self.feature_names_in_ = None
        self._feature_names_out = None

    def _ensure_dataframe(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Преобразует входные данные в pandas DataFrame, если необходимо."""
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, np.ndarray):
            if self.feature_names_in_ is None:
                raise ValueError(
                    "Transformer not fitted yet. Cannot convert numpy array to DataFrame."
                )
            return pd.DataFrame(X, columns=self.feature_names_in_)
        raise TypeError(f"Expected pandas DataFrame or numpy array, got {type(X)}")

    def _apply_transformations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Применяет все преобразования к DataFrame (без проверок)."""
        data = data.copy()

        data["dti_loan_pct_ratio"] = data["debt_to_income_ratio"] / (data["loan_percent_income"] + 0.001)
        data["income_debt_balance"] = data["person_income"] - data["other_debt"]

        data["rate_grade_deviation"] = data["loan_int_rate"] - data["loan_grade"].map(
            self.params["avg_rate_by_grade"]
        )

        data["is_high_debt"] = (data["debt_to_income_ratio"] > self.params["threshold_debt"]).astype(int)
        data["is_high_loan_pct"] = (data["loan_percent_income"] > self.params["threshold_loan_pct"]).astype(int)
        data["is_high_rate"] = (data["loan_int_rate"] > self.params["threshold_rate"]).astype(int)

        data["city_avg_dti"] = data["city"].map(self.params["city_avg_dti"])
        data["city_avg_loan_pct"] = data["city"].map(self.params["city_avg_loan_pct"])

        data["grade_ownership"] = data["loan_grade"].astype(str) + "_" + data["person_home_ownership"].astype(str)
        data["default_grade"] = data["cb_person_default_on_file"].astype(str) + "_" + data["loan_grade"].astype(str)

        if not self.include_weak:
            weak_cols = [
                "dti_loan_pct_ratio",
                "rate_grade_deviation",
                "city_avg_dti",
                "city_avg_loan_pct",
            ]
            data.drop(columns=weak_cols, inplace=True, errors="ignore")

        return data

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[pd.Series] = None) -> "FeatureEngineer":
        """
        Вычисляет параметры для создания новых признаков и сохраняет имена выходных колонок.

        Args:
            X: pandas DataFrame или numpy array с очищенными данными.
            y: не используется, оставлен для совместимости.

        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        elif isinstance(X, np.ndarray):
            if self.feature_names_in_ is None:
                raise ValueError("Feature names not provided. Please pass a DataFrame for fitting.")
        else:
            raise TypeError(f"Expected pandas DataFrame or numpy array, got {type(X)}")

        data = self._ensure_dataframe(X)

        self.params["threshold_debt"] = data["debt_to_income_ratio"].quantile(0.75)
        self.params["threshold_loan_pct"] = data["loan_percent_income"].quantile(0.75)
        self.params["threshold_rate"] = data["loan_int_rate"].quantile(0.75)

        self.params["city_avg_dti"] = data.groupby("city")["debt_to_income_ratio"].mean().to_dict()
        self.params["city_avg_loan_pct"] = data.groupby("city")["loan_percent_income"].mean().to_dict()
        self.params["avg_rate_by_grade"] = data.groupby("loan_grade")["loan_int_rate"].mean().to_dict()

        dummy = pd.DataFrame(columns=self.feature_names_in_).fillna(0)
        transformed = self._apply_transformations(dummy)
        self._feature_names_out = transformed.columns.tolist()

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Применяет создание новых признаков к данным.

        Args:
            X: pandas DataFrame или numpy array с очищенными данными.

        Returns:
            pandas DataFrame с добавленными новыми признаками.

        Raises:
            ValueError: если не были вызваны fit перед transform.
        """
        if not self.params:
            raise ValueError("Transformer not fitted. Call fit() before transform().")

        data = self._ensure_dataframe(X)
        return self._apply_transformations(data)

    def fit_transform(
        self, X: Union[pd.DataFrame, np.ndarray], y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Сочетание fit и transform.

        Args:
            X: pandas DataFrame или numpy array с очищенными данными.
            y: не используется, оставлен для совместимости.

        Returns:
            pandas DataFrame с добавленными новыми признаками.
        """
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Возвращает имена колонок после трансформации.

        Args:
            input_features: игнорируется, используется для совместимости.

        Returns:
            Список имён колонок выходного DataFrame.

        Raises:
            ValueError: если трансформер не обучен.
        """
        if self._feature_names_out is None:
            raise ValueError("Transformer not fitted. Call fit() first.")
        return self._feature_names_out

    def __sklearn_is_fitted__(self) -> bool:
        """Проверяет, обучен ли трансформер."""
        return hasattr(self, "params") and bool(self.params)
