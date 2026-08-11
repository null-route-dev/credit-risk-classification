"""
Модуль предобработки данных для кредитного скоринга.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Union, List


class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Предобработка данных: удаление признаков, выбросов, импутация пропусков.
    """

    def __init__(
        self,
        drop_cols: Optional[List[str]] = None,
        age_threshold: int = 100,
        emp_length_threshold: int = 50,
    ):
        """
        Инициализирует препроцессор с заданными параметрами.

        Args:
            drop_cols: Список названий колонок для удаления.
                       Если None, используются колонки по умолчанию.
            age_threshold: Максимально допустимый возраст (в годах).
            emp_length_threshold: Максимально допустимый стаж работы (в годах).
        """
        self.drop_cols = drop_cols or [
            "client_ID",
            "loan_to_income_ratio",
            "city_latitude",
            "city_longitude",
            "state",
            "country",
        ]
        self.age_threshold = age_threshold
        self.emp_length_threshold = emp_length_threshold
        self.params = {}

    def _drop_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Удаляет заданные колонки из DataFrame.

        Args:
            X: Исходный DataFrame.

        Returns:
            DataFrame без удалённых колонок.
            Если колонка отсутствует, ошибка игнорируется.
        """
        return X.drop(columns=self.drop_cols, errors="ignore")

    def _filter_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Фильтрует строки по возрасту и стажу, удаляя выбросы.

        Args:
            X: DataFrame с колонками 'person_age' и 'person_emp_length'.

        Returns:
            Отфильтрованный DataFrame (копия).
            Оставляются строки, где возраст <= age_threshold,
            а стаж либо <= emp_length_threshold, либо отсутствует (NaN).
        """
        mask = (X["person_age"] <= self.age_threshold) & (
            (X["person_emp_length"] <= self.emp_length_threshold) | X["person_emp_length"].isna()
        )
        return X.loc[mask].copy()

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Preprocessor":
        """
        Вычисляет параметры импутации на основе обучающих данных.

        Args:
            X: pandas DataFrame с исходными данными.
            y: не используется, оставлен для совместимости.

        Returns:
            self

        Raises:
            ValueError: если отсутствует одна из обязательных колонок.
        """
        data = X.copy()
        data = self._drop_columns(data)
        data = self._filter_outliers(data)

        self.params["rate_medians"] = data.groupby("loan_grade")["loan_int_rate"].median().to_dict()
        self.params["global_rate_median"] = float(data["loan_int_rate"].median())
        self.params["emp_median"] = float(data["person_emp_length"].median())

        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Union[pd.DataFrame, tuple]:
        """
        Применяет предобработку к данным.

        Args:
            X: pandas DataFrame с исходными данными.
            y: опционально, целевая переменная. Если передана, возвращается синхронизированная
                версия y (только для строк, оставшихся после удаления выбросов).

        Returns:
            Если y не передан: pd.DataFrame с предобработанными данными.
            Если y передан: кортеж (pd.DataFrame, pd.Series) — предобработанные X и синхронизированный y.

        Raises:
            ValueError: если не были вызваны fit перед transform.
        """
        if not self.params:
            raise ValueError("Модель не обучена. Вызовите fit перед transform.")

        data = X.copy()
        data = self._drop_columns(data)
        data = self._filter_outliers(data)

        rate_medians = self.params["rate_medians"]
        global_rate_median = self.params["global_rate_median"]
        data["loan_int_rate"] = data["loan_int_rate"].fillna(
            data["loan_grade"].map(rate_medians).fillna(global_rate_median)
        )

        emp_median = self.params["emp_median"]
        data["emp_length_missing"] = data["person_emp_length"].isna().astype(int)
        data["person_emp_length"] = data["person_emp_length"].fillna(emp_median)

        if y is not None:
            y_sync = y.loc[data.index]
            return data, y_sync
        return data

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Union[pd.DataFrame, tuple]:
        """
        Сочетание fit и transform с синхронизацией y.

        Args:
            X: pandas DataFrame с исходными данными.
            y: опционально, целевая переменная. Если передана, возвращается синхронизированная
                версия y (только для строк, оставшихся после удаления выбросов).

        Returns:
            Если y не передан: pd.DataFrame с предобработанными данными.
            Если y передан: кортеж (pd.DataFrame, pd.Series) — предобработанные X и синхронизированный y.
        """
        self.fit(X, y)
        return self.transform(X, y)
