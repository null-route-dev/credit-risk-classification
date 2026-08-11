"""
Модуль для создания новых признаков (Feature Engineering).
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Union


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Создание новых признаков на основе очищенных данных.

    Параметры:
        include_weak : bool, default=False
            Если False, слабые признаки не создаются.
            Слабые признаки: dti_loan_pct_ratio, rate_grade_deviation,
            city_avg_dti, city_avg_loan_pct.
            Остальные новые признаки (бинарные флаги, комбинации категорий)
            создаются всегда.
    """

    def __init__(self, include_weak: bool = False):
        """
        Инициализирует FeatureEngineer.

        Args:
            include_weak: Флаг включения слабых признаков.
        """
        self.include_weak = include_weak
        self.params = {}

    def _compute_thresholds(self, data: pd.DataFrame) -> None:
        """
        Вычисляет пороговые значения и средние по группам для создания признаков.

        Args:
            data: DataFrame с исходными признаками.
        """
        self.params["threshold_debt"] = data["debt_to_income_ratio"].quantile(0.75)
        self.params["threshold_loan_pct"] = data["loan_percent_income"].quantile(0.75)
        self.params["threshold_rate"] = data["loan_int_rate"].quantile(0.75)

        self.params["city_avg_dti"] = data.groupby("city")["debt_to_income_ratio"].mean().to_dict()
        self.params["city_avg_loan_pct"] = data.groupby("city")["loan_percent_income"].mean().to_dict()
        self.params["avg_rate_by_grade"] = data.groupby("loan_grade")["loan_int_rate"].mean().to_dict()

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        """
        Вычисляет параметры для создания новых признаков.

        Args:
            X: pandas DataFrame с очищенными данными.
            y: не используется, оставлен для совместимости.

        Returns:
            self
        """
        data = X.copy()
        self._compute_thresholds(data)
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Union[pd.DataFrame, tuple]:
        """
        Применяет создание новых признаков к данным.

        Args:
            X: pandas DataFrame с очищенными данными.
            y: опционально, целевая переменная. Если передана, возвращается синхронизированная
                версия y (только для строк, оставшихся после удаления выбросов).

        Returns:
            Если y не передан: pd.DataFrame с добавленными новыми признаками.
            Если y передан: кортеж (pd.DataFrame, pd.Series) — X с новыми признаками и синхронизированный y.

        Raises:
            ValueError: если не были вызваны fit перед transform.
        """
        if not self.params:
            raise ValueError("Модель не обучена. Вызовите fit перед transform.")

        data = X.copy()

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
                "city_avg_loan_pct"
            ]
            data.drop(columns=weak_cols, inplace=True, errors="ignore")

        if y is not None:
            y_sync = y.loc[data.index]
            return data, y_sync
        return data

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Union[pd.DataFrame, tuple]:
        """
        Сочетание fit и transform с синхронизацией y.

        Args:
            X: pandas DataFrame с очищенными данными.
            y: опционально, целевая переменная. Если передана, возвращается синхронизированная
                версия y (только для строк, оставшихся после удаления выбросов).

        Returns:
            Если y не передан: pd.DataFrame с добавленными новыми признаками.
            Если y передан: кортеж (pd.DataFrame, pd.Series) — X с новыми признаками и синхронизированный y.
        """
        self.fit(X, y)
        return self.transform(X, y)
