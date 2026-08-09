"""
Модуль предобработки данных для кредитного скоринга.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional


class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Предобработка данных: удаление признаков, выбросов, импутация пропусков.
    """

    def __init__(self):
        self.params = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Preprocessor":
        """
        Вычисляет параметры импутации на основе данных.

        Args:
            X: pandas DataFrame с исходными данными.
            y: не используется.

        Returns:
            self

        Raises:
            ValueError: если отсутствует одна из обязательных колонок.
        """
        data = X.copy()

        data.drop(columns=["client_ID"], inplace=True)
        data.drop(columns=["loan_to_income_ratio"], inplace=True)
        data.drop(columns=["city_latitude", "city_longitude", "state", "country"], inplace=True)

        data = data[data["person_age"] <= 100].copy()
        data = data[(data["person_emp_length"] <= 50) | data["person_emp_length"].isna()].copy()

        self.params["rate_medians"] = data.groupby("loan_grade")["loan_int_rate"].median().to_dict()
        self.params["emp_median"] = float(data["person_emp_length"].median())

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет предобработку к данным.

        Args:
            X: pandas DataFrame с исходными данными.

        Returns:
            pandas DataFrame с предобработанными данными.

        Raises:
            ValueError: если не были вызваны fit перед transform.
        """
        data = X.copy()

        data.drop(columns=["client_ID"], inplace=True)
        data.drop(columns=["loan_to_income_ratio"], inplace=True)
        data.drop(columns=["city_latitude", "city_longitude", "state", "country"], inplace=True)

        data = data[data["person_age"] <= 100].copy()
        data = data[(data["person_emp_length"] <= 50) | data["person_emp_length"].isna()].copy()

        rate_medians = self.params.get("rate_medians")
        if rate_medians is None:
            raise ValueError("Параметры импутации не найдены. Вызовите fit перед transform.")

        def fill_rate(row: pd.Series) -> float:
            if pd.isna(row["loan_int_rate"]):
                return rate_medians[row["loan_grade"]]
            return row["loan_int_rate"]

        data["loan_int_rate"] = data.apply(fill_rate, axis=1)

        emp_median = self.params.get("emp_median")
        if emp_median is None:
            raise ValueError("Параметр emp_median не найден. Вызовите fit перед transform.")

        data["emp_length_missing"] = data["person_emp_length"].isna().astype(int)
        data["person_emp_length"] = data["person_emp_length"].fillna(emp_median)

        return data

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Сочетание fit и transform.

        Args:
            X: pandas DataFrame с исходными данными.
            y: не используется.

        Returns:
            pandas DataFrame с предобработанными данными.
        """
        return self.fit(X, y).transform(X)
