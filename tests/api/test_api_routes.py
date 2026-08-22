"""
Тесты для модуля routes.py
"""

import sys
from pathlib import Path
import pytest
import numpy as np
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.api.main import app


@pytest.fixture
def mock_predictor():
    predictor = MagicMock()

    def predict_single_side_effect(data):
        return {"prediction": 1, "probability": 0.85}

    def predict_proba_side_effect(df):
        return np.array([0.85, 0.3])

    predictor.predict_single.side_effect = predict_single_side_effect
    predictor.predict_proba.side_effect = predict_proba_side_effect
    return predictor


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestRoutes:
    """Тесты для эндпоинтов API."""

    VALID_PAYLOAD = {
        "client_ID": "CUST_12345",
        "person_age": 35,
        "person_income": 75000,
        "person_home_ownership": "OWN",
        "person_emp_length": 8.0,
        "loan_intent": "PERSONAL",
        "loan_grade": "B",
        "loan_amnt": 15000,
        "loan_int_rate": 12.5,
        "loan_percent_income": 0.2,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 15,
        "gender": "Male",
        "marital_status": "Married",
        "education_level": "Bachelor",
        "country": "USA",
        "state": "California",
        "city": "Los Angeles",
        "city_latitude": 34.05,
        "city_longitude": -118.24,
        "employment_type": "Full-time",
        "loan_term_months": 36,
        "loan_to_income_ratio": 0.2,
        "other_debt": 5000.0,
        "debt_to_income_ratio": 0.3,
        "open_accounts": 5,
        "credit_utilization_ratio": 0.4,
        "past_delinquencies": 0
    }

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Credit Risk Scoring API"
        assert data["version"] == "1.0.0"
        assert data["health"] == "/health"

    def test_health_ok(self, client, mock_predictor):
        client.app.state.predictor = mock_predictor
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    def test_health_fail(self, client):
        client.app.state.predictor = None
        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert "Модель не загружена" in data["detail"]

    def test_predict_single_valid(self, client, mock_predictor):
        client.app.state.predictor = mock_predictor
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 1
        assert data["probability"] == 0.85
        mock_predictor.predict_single.assert_called_once()

    def test_predict_single_invalid_age(self, client, mock_predictor):
        client.app.state.predictor = mock_predictor
        payload = self.VALID_PAYLOAD.copy()
        payload["person_age"] = 17
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_single_invalid_client_id(self, client, mock_predictor):
        client.app.state.predictor = mock_predictor
        payload = self.VALID_PAYLOAD.copy()
        payload["client_ID"] = "INVALID"
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_single_model_not_loaded(self, client):
        client.app.state.predictor = None
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        assert response.status_code == 503
        assert "Модель не загружена" in response.text

    def test_predict_single_predictor_raises_value_error(self, client, mock_predictor):
        def predict_single_side_effect(data):
            raise ValueError("Тестовая ошибка")
        mock_predictor.predict_single.side_effect = predict_single_side_effect
        client.app.state.predictor = mock_predictor
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        assert response.status_code == 400
        assert "Тестовая ошибка" in response.text

    def test_predict_single_predictor_raises_generic_error(self, client, mock_predictor):
        def predict_single_side_effect(data):
            raise Exception("Внутренняя ошибка")
        mock_predictor.predict_single.side_effect = predict_single_side_effect
        client.app.state.predictor = mock_predictor
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        assert response.status_code == 500
        assert "Ошибка выполнения" in response.text

    def test_predict_batch_valid(self, client, mock_predictor):
        client.app.state.predictor = mock_predictor
        payload = [self.VALID_PAYLOAD, self.VALID_PAYLOAD.copy()]
        payload[1]["client_ID"] = "CUST_67890"
        response = client.post("/predict_batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["predictions"] == [1, 0]
        assert data["probabilities"] == [0.85, 0.3]
        mock_predictor.predict_proba.assert_called_once()

    def test_predict_batch_model_not_loaded(self, client):
        client.app.state.predictor = None
        payload = [self.VALID_PAYLOAD]
        response = client.post("/predict_batch", json=payload)
        assert response.status_code == 503
        assert "Модель не загружена" in response.text

    def test_predict_batch_predictor_raises_error(self, client, mock_predictor):
        def predict_proba_side_effect(df):
            raise Exception("Ошибка пакетного предсказания")
        mock_predictor.predict_proba.side_effect = predict_proba_side_effect
        client.app.state.predictor = mock_predictor
        payload = [self.VALID_PAYLOAD]
        response = client.post("/predict_batch", json=payload)
        assert response.status_code == 500
        assert "Ошибка выполнения" in response.text
