"""
Тесты для модуля data_loader.py
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, ANY
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loader import DataLoader


class TestDataLoader:

    def test_load_from_local(self, tmp_path):
        """Проверка загрузки из локального файла."""
        df_expected = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        csv_path = tmp_path / 'data.csv'
        df_expected.to_csv(csv_path, index=False)

        df = DataLoader.load(csv_path)
        pd.testing.assert_frame_equal(df, df_expected)

    def test_load_from_kaggle_when_local_missing(self, tmp_path):
        """Если локальный файл отсутствует и указан kaggle_dataset – загружает с Kaggle."""
        mock_df = pd.DataFrame({'id': [101, 102], 'loan_status': [0, 1]})
        fake_path = tmp_path / 'missing.csv'
        kaggle_ds = 'test/dataset'

        with patch('src.data_loader.dataset_load') as mock_load:
            mock_load.return_value = mock_df
            df = DataLoader.load(fake_path, kaggle_dataset=kaggle_ds)

            mock_load.assert_called_once_with(
                ANY,
                kaggle_ds,
                fake_path.name
            )
            pd.testing.assert_frame_equal(df, mock_df)

    def test_force_download(self, tmp_path):
        """Если force=True, игнорирует локальный файл и загружает с Kaggle."""
        local_df = pd.DataFrame({'col1': [1, 2]})
        csv_path = tmp_path / 'data.csv'
        local_df.to_csv(csv_path, index=False)

        mock_df = pd.DataFrame({'id': [101, 102], 'loan_status': [0, 1]})
        kaggle_ds = 'test/dataset'

        with patch('src.data_loader.dataset_load') as mock_load:
            mock_load.return_value = mock_df
            df = DataLoader.load(csv_path, kaggle_dataset=kaggle_ds, force=True)

            mock_load.assert_called_once_with(
                ANY,
                kaggle_ds,
                csv_path.name
            )
            pd.testing.assert_frame_equal(df, mock_df)

    def test_file_not_found_and_no_kaggle(self):
        """Если файл не найден и kaggle_dataset не указан – выбрасывает FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Файл .* не найден, а kaggle_dataset не указан."):
            DataLoader.load(Path('/nonexistent/file.csv'))

    def test_save_data(self, tmp_path):
        """Проверка сохранения DataFrame."""
        df = pd.DataFrame({'x': [10, 20]})
        file_path = tmp_path / 'out.csv'

        DataLoader.save(df, file_path)
        assert file_path.exists()

        loaded = pd.read_csv(file_path, index_col=0)
        pd.testing.assert_frame_equal(loaded, df)
