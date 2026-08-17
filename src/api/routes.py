"""Определение эндпоинтов FastAPI."""

import logging
from typing import List

import pandas as pd
from fastapi import APIRouter, Request, HTTPException, status

from .schemas import (
    ClientData,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Предсказание для одного клиента",
)
async def predict_single(request: Request, client: ClientData):
    """Предсказание для одного клиента."""
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Модель не загружена"
        )
    try:
        data_dict = client.model_dump()
        result = predictor.predict_single(data_dict)
        return PredictionResponse(
            prediction=result["prediction"],
            probability=result["probability"]
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка выполнения: {str(e)}"
        )


@router.post(
    "/predict_batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Пакетное предсказание",
)
async def predict_batch(request: Request, clients: List[ClientData]):
    """Предсказание для нескольких клиентов."""
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Модель не загружена"
        )
    try:
        df = pd.DataFrame([c.model_dump() for c in clients])
        proba = predictor.predict_proba(df)
        preds = (proba >= 0.5).astype(int).tolist()
        return BatchPredictionResponse(
            predictions=preds,
            probabilities=proba.tolist()
        )
    except Exception as e:
        logger.error(f"Ошибка при пакетном предсказании: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка выполнения: {str(e)}"
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Проверка работоспособности",
)
async def health_check(request: Request):
    """Проверяет, загружена ли модель."""
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Модель не загружена"
        )
    return HealthResponse(status="ok", model_loaded=True)


@router.get("/")
async def root():
    """Корневой эндпоинт."""
    return {
        "message": "Credit Risk Scoring API",
        "version": "1.0.0",
        "health": "/health"
    }
