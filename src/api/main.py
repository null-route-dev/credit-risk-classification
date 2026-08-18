"""Главный модуль FastAPI: создание приложения и управление жизненным циклом."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.predict import LightGBMPredictor
from src.config import MODEL_DIR
from .routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка модели при старте и освобождение ресурсов при остановке."""
    logger.info("Загрузка модели...")
    try:
        app.state.predictor = LightGBMPredictor(MODEL_DIR)
        logger.info(f"Модель успешно загружена из {MODEL_DIR}")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        raise RuntimeError("Не удалось загрузить модель")
    yield
    logger.info("API завершает работу")


app = FastAPI(
    title="Credit Risk Scoring API",
    description="API для предсказания кредитного риска на основе модели LightGBM",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
