"""
Тесты для модуля main.py
"""

import sys
from pathlib import Path
import pytest
import logging
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.api.main import app, lifespan


class TestMain:
    """Тесты для модуля main.py."""

    def test_app_creation(self):
        """Проверка создания приложения FastAPI."""
        assert app.title == "Credit Risk Scoring API"
        assert app.version == "1.0.0"
        assert app.description is not None

    @pytest.mark.asyncio
    async def test_lifespan_success(self):
        """Проверка lifespan при успешной загрузке модели."""
        mock_predictor = MagicMock()
        with patch("src.api.main.LightGBMPredictor", return_value=mock_predictor):
            async with lifespan(app) as context:
                assert app.state.predictor is mock_predictor

    @pytest.mark.asyncio
    async def test_lifespan_failure(self):
        """Проверка lifespan при ошибке загрузки модели."""
        with patch("src.api.main.LightGBMPredictor", side_effect=Exception("Ошибка")):
            with pytest.raises(RuntimeError, match="Не удалось загрузить модель"):
                async with lifespan(app):
                    pass

    @pytest.mark.asyncio
    async def test_lifespan_logger(self, caplog):
        """Проверка логирования в lifespan."""
        caplog.set_level(logging.INFO)
        with patch("src.api.main.LightGBMPredictor", return_value=MagicMock()):
            async with lifespan(app):
                pass
        assert "Загрузка модели..." in caplog.text
        assert "Модель успешно загружена" in caplog.text
        assert "API завершает работу" in caplog.text
