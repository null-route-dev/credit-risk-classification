"""
Модуль для загрузки данных из локального CSV или Kaggle.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union
import logging

from kagglehub import dataset_load, KaggleDatasetAdapter

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Загрузчик данных из CSV или Kaggle.
    """

    @staticmethod
    def load(filepath: Union[str, Path],
             kaggle_dataset: Optional[str] = None,
             force: bool = False) -> pd.DataFrame:
        """
        Загружает данные из локального CSV или с Kaggle.

        Args:
            filepath: Путь к CSV-файлу.
            kaggle_dataset: Имя датасета на Kaggle (owner/dataset).
            force: Если True, игнорирует локальный файл и всегда загружает с Kaggle.

        Returns:
            DataFrame с данными.

        Raises:
            FileNotFoundError: Если файл не найден и kaggle_dataset не задан.
            ImportError: Если kagglehub не установлен.
        """
        path = Path(filepath)

        if not force and path.exists():
            logger.info(f"Загрузка из {path}")
            return pd.read_csv(path)

        if kaggle_dataset is None:
            raise FileNotFoundError(f"Файл {path} не найден, а kaggle_dataset не указан.")

        return DataLoader._load_from_kaggle(kaggle_dataset, path.name)

    @staticmethod
    def _load_from_kaggle(kaggle_dataset: str, file_name: str) -> pd.DataFrame:
        """
        Загружает данные с Kaggle напрямую в DataFrame.

        Args:
            kaggle_dataset: Имя датасета на Kaggle.
            file_name: Имя файла внутри датасета.

        Returns:
            DataFrame.
        """
        logger.info(f"Загрузка {kaggle_dataset}/{file_name} с Kaggle...")
        df = dataset_load(
            KaggleDatasetAdapter.PANDAS,
            kaggle_dataset,
            file_name
        )
        logger.info(f"Загружено {df.shape[0]} строк, {df.shape[1]} столбцов.")
        return df

    @staticmethod
    def save(df: pd.DataFrame, filepath: Union[str, Path]) -> None:
        """
        Сохраняет DataFrame в CSV.

        Args:
            df: Данные для сохранения.
            filepath: Путь для сохранения.
        """
        path = Path(filepath)
        df.to_csv(path)
        logger.info(f"Данные сохранены в {path}")
