"""
Модуль предобработки данных для кредитного скоринга.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, List, Union


class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Предобработка данных: удаление признаков, выбросов, импутация пропусков.

    Attributes:
        drop_cols (List[str]): Список колонок для удаления.
        age_threshold (int): Максимально допустимый возраст.
        emp_length_threshold (int): Максимально допустимый стаж.
        params (dict): Параметры, вычисленные в fit.
        feature_names_in_ (List[str]): Имена колонок входных данных.
        _feature_names_out (List[str]): Имена колонок после трансформации.
    """

    def __init__(
        self,
        drop_cols: Optional[List[str]] = None,
        age_threshold: int = 100,
        emp_length_threshold: int = 50,
    ):
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

    def _drop_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Удаляет заданные колонки из DataFrame."""
        return X.drop(columns=self.drop_cols, errors="ignore")

    def _cap_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ограничивает значения возраста и стажа заданными порогами (каппинг)."""
        data = X.copy()
        data["person_age"] = data["person_age"].clip(upper=self.age_threshold)
        data["person_emp_length"] = data["person_emp_length"].clip(upper=self.emp_length_threshold)
        return data

    def _apply_transformations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Применяет все преобразования к DataFrame (без проверок)."""
        data = self._drop_columns(data)
        data = self._cap_outliers(data)

        rate_medians = self.params["rate_medians"]
        global_rate_median = self.params["global_rate_median"]
        data["loan_int_rate"] = data["loan_int_rate"].fillna(
            data["loan_grade"].map(rate_medians).fillna(global_rate_median)
        )

        emp_median = self.params["emp_median"]
        data["emp_length_missing"] = data["person_emp_length"].isna().astype(int)
        data["person_emp_length"] = data["person_emp_length"].fillna(emp_median)

        return data

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[pd.Series] = None) -> "Preprocessor":
        """
        Вычисляет параметры импутации и сохраняет имена выходных колонок.

        Args:
            X: pandas DataFrame или numpy array с исходными данными.
            y: не используется, оставлен для совместимости.

        Returns:
            self

        Raises:
            TypeError: Если тип X не поддерживается.
            ValueError: Если отсутствует одна из обязательных колонок.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        elif isinstance(X, np.ndarray):
            if self.feature_names_in_ is None:
                raise ValueError("Feature names not provided. Please pass a DataFrame for fitting.")
        else:
            raise TypeError(f"Expected pandas DataFrame or numpy array, got {type(X)}")

        data = self._ensure_dataframe(X)
        self.params["rate_medians"] = data.groupby("loan_grade")["loan_int_rate"].median().to_dict()
        self.params["global_rate_median"] = float(data["loan_int_rate"].median())
        self.params["emp_median"] = float(data["person_emp_length"].median())

        dummy = pd.DataFrame(columns=self.feature_names_in_).fillna(0)
        transformed = self._apply_transformations(dummy)
        self._feature_names_out = transformed.columns.tolist()

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Применяет предобработку к данным.

        Args:
            X: pandas DataFrame или numpy array с исходными данными.

        Returns:
            pandas DataFrame с предобработанными данными.

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
            X: pandas DataFrame или numpy array с исходными данными.
            y: не используется, оставлен для совместимости.

        Returns:
            pandas DataFrame с предобработанными данными.
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
