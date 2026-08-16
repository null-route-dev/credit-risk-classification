"""
Модуль для получения предсказаний из обученной модели LightGBM.
"""

import logging
from pathlib import Path
from typing import Union, Dict, Any, List, Optional

import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


class LightGBMPredictor:
    """
    Класс для загрузки модели и выполнения предсказаний.

    Attributes:
        model (Optional[Pipeline]): Загруженный или переданный пайплайн.
        model_path (Optional[Path]): Путь к файлу модели (если загружалась из файла).
        feature_names (Optional[List[str]]): Имена признаков, ожидаемых на входе.
    """

    def __init__(self, model: Union[str, Path, Pipeline]):
        """
        Инициализация предиктора.

        Args:
            model: Либо путь к файлу модели (или директории с lgbm_final_model.pkl),
                   либо уже загруженный объект Pipeline.
        """
        self.model: Optional[Pipeline] = None
        self.model_path: Optional[Path] = None
        self.feature_names: Optional[List[str]] = None

        if isinstance(model, Pipeline):
            self.model = model
            self._extract_feature_names()
            logger.info("Предиктор инициализирован с готовым пайплайном")
        else:
            self.model_path = Path(model)
            self._load_model()

    def _extract_feature_names(self) -> None:
        """Извлекает имена признаков из пайплайна."""
        basic_preprocessor = self.model.named_steps.get("basic_preprocessing")
        if basic_preprocessor is not None:
            if hasattr(basic_preprocessor, "feature_names_in_"):
                self.feature_names = basic_preprocessor.feature_names_in_
            else:
                logger.warning("Не удалось получить имена признаков из preprocessor")
        else:
            logger.warning("В пайплайне отсутствует шаг basic_preprocessing")

    def _load_model(self) -> None:
        """Загружает модель из файла и извлекает имена признаков."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Модель не найдена по пути: {self.model_path}")

        if self.model_path.is_dir():
            model_file = self.model_path / "lgbm_final_model.pkl"
            if not model_file.exists():
                raise FileNotFoundError(
                    f"Файл модели lgbm_final_model.pkl не найден в директории {self.model_path}"
                )
            self.model_path = model_file

        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Модель успешно загружена из {self.model_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise

        self._extract_feature_names()

    def _validate_input(self, X: pd.DataFrame) -> None:
        """Проверяет, что входной DataFrame содержит все необходимые колонки."""
        if self.feature_names is None:
            logger.warning("Список ожидаемых признаков не задан, проверка пропущена")
            return
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Возвращает вероятности положительного класса для каждого объекта.

        Args:
            X: pandas DataFrame с данными.

        Returns:
            numpy array вероятностей для класса 1.
        """
        self._validate_input(X)
        proba = self.model.predict_proba(X)
        logger.info(f"Предсказаны вероятности для {len(X)} объектов")
        return proba[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Возвращает бинарные предсказания (0 или 1) на основе порога.

        Args:
            X: pandas DataFrame с данными.
            threshold: Порог отсечения для положительного класса.

        Returns:
            numpy array меток классов (0 или 1).
        """
        proba = self.predict_proba(X)
        preds = (proba >= threshold).astype(int)
        logger.info(f"Предсказаны классы для {len(X)} объектов с порогом {threshold}")
        return preds

    def predict_single(
        self,
        data: Union[Dict[str, Any], pd.Series, pd.DataFrame],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Предсказание для одного объекта (словарь, Series или DataFrame из одной строки).

        Args:
            data: Данные одного объекта.
            threshold: Порог отсечения.

        Returns:
            Словарь с вероятностью положительного класса и предсказанной меткой.
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            df = pd.DataFrame([data.to_dict()])
        elif isinstance(data, pd.DataFrame):
            if len(data) != 1:
                raise ValueError("DataFrame должен содержать ровно одну строку")
            df = data.copy()
        else:
            raise TypeError(f"Не поддерживаемый тип данных: {type(data)}")

        proba = self.predict_proba(df)[0]
        pred = int(proba >= threshold)
        logger.info(f"Предсказание для одного объекта: класс {pred}, вероятность {proba:.4f}")
        return {
            "prediction": pred,
            "probability": float(proba)
        }

    def get_feature_names(self) -> Optional[List[str]]:
        """Возвращает список признаков, ожидаемых моделью."""
        return self.feature_names
