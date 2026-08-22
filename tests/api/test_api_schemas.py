"""
Тесты для модуля schemas.py
"""

import sys
from pathlib import Path
import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.api.schemas import ClientData, PredictionResponse, BatchPredictionResponse, HealthResponse


class TestClientData:
    """Тесты для Pydantic-схемы ClientData."""

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

    def test_valid_data(self):
        client = ClientData(**self.VALID_PAYLOAD)
        assert client.client_ID == "CUST_12345"
        assert client.person_age == 35

    def test_invalid_age_too_low(self):
        data = self.VALID_PAYLOAD.copy()
        data["person_age"] = 17
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Возраст должен быть от 18 до 100 лет" in str(exc.value)

    def test_invalid_age_too_high(self):
        data = self.VALID_PAYLOAD.copy()
        data["person_age"] = 101
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Возраст должен быть от 18 до 100 лет" in str(exc.value)

    def test_invalid_client_id_not_starting_cust(self):
        data = self.VALID_PAYLOAD.copy()
        data["client_ID"] = "INVALID_123"
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Идентификатор клиента должен начинаться с 'CUST_'" in str(exc.value)

    def test_invalid_client_id_wrong_format(self):
        data = self.VALID_PAYLOAD.copy()
        data["client_ID"] = "CUST_abc"
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Идентификатор клиента должен быть в формате 'CUST_XXXXX'" in str(exc.value)

    def test_invalid_income_too_low(self):
        data = self.VALID_PAYLOAD.copy()
        data["person_income"] = 500
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Доход не может быть меньше 1000" in str(exc.value)

    def test_invalid_income_too_high(self):
        data = self.VALID_PAYLOAD.copy()
        data["person_income"] = 15_000_000
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Доход не может превышать 10,000,000" in str(exc.value)

    def test_invalid_home_ownership(self):
        data = self.VALID_PAYLOAD.copy()
        data["person_home_ownership"] = "INVALID"
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Статус собственности должен быть одним из" in str(exc.value)

    def test_invalid_emp_length_negative(self):
        data = self.VALID_PAYLOAD.copy()
        data["person_emp_length"] = -1.0
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Стаж не может быть отрицательным" in str(exc.value)

    def test_invalid_emp_length_too_high(self):
        data = self.VALID_PAYLOAD.copy()
        data["person_emp_length"] = 51.0
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Стаж не может превышать 50 лет" in str(exc.value)

    def test_invalid_loan_intent(self):
        data = self.VALID_PAYLOAD.copy()
        data["loan_intent"] = "INVALID"
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Цель кредита должна быть одной из" in str(exc.value)

    def test_invalid_loan_grade(self):
        data = self.VALID_PAYLOAD.copy()
        data["loan_grade"] = "Z"
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Грейд кредита должен быть одним из" in str(exc.value)

    def test_invalid_loan_amnt_too_low(self):
        data = self.VALID_PAYLOAD.copy()
        data["loan_amnt"] = 100
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Сумма кредита не может быть меньше 500" in str(exc.value)

    def test_invalid_loan_amnt_too_high(self):
        data = self.VALID_PAYLOAD.copy()
        data["loan_amnt"] = 200_000
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Сумма кредита не может превышать 100,000" in str(exc.value)

    def test_invalid_loan_int_rate_negative(self):
        data = self.VALID_PAYLOAD.copy()
        data["loan_int_rate"] = -5.0
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Процентная ставка не может быть отрицательной" in str(exc.value)

    def test_invalid_loan_int_rate_too_high(self):
        data = self.VALID_PAYLOAD.copy()
        data["loan_int_rate"] = 35.0
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Процентная ставка не может превышать 30%" in str(exc.value)

    def test_invalid_loan_percent_income_negative(self):
        data = self.VALID_PAYLOAD.copy()
        data["loan_percent_income"] = -0.1
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Доля кредита от дохода должна быть от 0 до 1" in str(exc.value)

    def test_invalid_loan_percent_income_too_high(self):
        data = self.VALID_PAYLOAD.copy()
        data["loan_percent_income"] = 1.5
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Доля кредита от дохода должна быть от 0 до 1" in str(exc.value)

    def test_invalid_default_on_file(self):
        data = self.VALID_PAYLOAD.copy()
        data["cb_person_default_on_file"] = "X"
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Значение должно быть 'Y' или 'N'" in str(exc.value)

    def test_invalid_cred_hist_length_negative(self):
        data = self.VALID_PAYLOAD.copy()
        data["cb_person_cred_hist_length"] = -1
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Длина кредитной истории не может быть отрицательной" in str(exc.value)

    def test_invalid_cred_hist_length_too_high(self):
        data = self.VALID_PAYLOAD.copy()
        data["cb_person_cred_hist_length"] = 60
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Длина кредитной истории не может превышать 50 лет" in str(exc.value)

    def test_invalid_gender(self):
        data = self.VALID_PAYLOAD.copy()
        data["gender"] = "Other"
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Пол должен быть 'Male' или 'Female'" in str(exc.value)

    def test_invalid_marital_status(self):
        data = self.VALID_PAYLOAD.copy()
        data["marital_status"] = "Unknown"
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Семейное положение должно быть одним из" in str(exc.value)

    def test_invalid_education_level(self):
        data = self.VALID_PAYLOAD.copy()
        data["education_level"] = "Elementary"
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Уровень образования должен быть одним из" in str(exc.value)

    def test_invalid_country(self):
        data = self.VALID_PAYLOAD.copy()
        data["country"] = "Germany"
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Страна должна быть одной из" in str(exc.value)

    def test_invalid_state(self):
        data = self.VALID_PAYLOAD.copy()
        data["state"] = "Florida"
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Штат/провинция должен быть одним из" in str(exc.value)

    def test_invalid_city(self):
        data = self.VALID_PAYLOAD.copy()
        data["city"] = "Chicago"
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Город должен быть одним из" in str(exc.value)

    def test_invalid_city_latitude(self):
        data = self.VALID_PAYLOAD.copy()
        data["city_latitude"] = 100.0
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Широта должна быть от -90 до 90 градусов" in str(exc.value)

    def test_invalid_city_longitude(self):
        data = self.VALID_PAYLOAD.copy()
        data["city_longitude"] = 200.0
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Долгота должна быть от -180 до 180 градусов" in str(exc.value)

    def test_invalid_employment_type(self):
        data = self.VALID_PAYLOAD.copy()
        data["employment_type"] = "Contractor"
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Тип занятости должен быть одним из" in str(exc.value)

    def test_invalid_loan_term_months(self):
        data = self.VALID_PAYLOAD.copy()
        data["loan_term_months"] = 48
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Срок кредита должен быть одним из [12, 24, 36, 60]" in str(exc.value)

    def test_invalid_loan_to_income_ratio_negative(self):
        data = self.VALID_PAYLOAD.copy()
        data["loan_to_income_ratio"] = -0.1
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Отношение кредита к доходу не может быть отрицательным" in str(exc.value)

    def test_invalid_loan_to_income_ratio_too_high(self):
        data = self.VALID_PAYLOAD.copy()
        data["loan_to_income_ratio"] = 1.5
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Отношение кредита к доходу не может превышать 1" in str(exc.value)

    def test_invalid_other_debt_negative(self):
        data = self.VALID_PAYLOAD.copy()
        data["other_debt"] = -100.0
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Прочие долги не могут быть отрицательными" in str(exc.value)

    def test_invalid_other_debt_too_high(self):
        data = self.VALID_PAYLOAD.copy()
        data["other_debt"] = 3_000_000.0
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Прочие долги не могут превышать 2,000,000" in str(exc.value)

    def test_invalid_debt_to_income_ratio_negative(self):
        data = self.VALID_PAYLOAD.copy()
        data["debt_to_income_ratio"] = -0.1
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Соотношение долга к доходу не может быть отрицательным" in str(exc.value)

    def test_invalid_debt_to_income_ratio_too_high(self):
        data = self.VALID_PAYLOAD.copy()
        data["debt_to_income_ratio"] = 2.5
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Соотношение долга к доходу не может превышать 2" in str(exc.value)

    def test_invalid_open_accounts_negative(self):
        data = self.VALID_PAYLOAD.copy()
        data["open_accounts"] = -1
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Количество открытых счетов не может быть отрицательным" in str(exc.value)

    def test_invalid_open_accounts_too_high(self):
        data = self.VALID_PAYLOAD.copy()
        data["open_accounts"] = 31
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Количество открытых счетов не может превышать 30" in str(exc.value)

    def test_invalid_credit_utilization_ratio_negative(self):
        data = self.VALID_PAYLOAD.copy()
        data["credit_utilization_ratio"] = -0.1
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Коэффициент использования кредита должен быть от 0 до 1" in str(exc.value)

    def test_invalid_credit_utilization_ratio_too_high(self):
        data = self.VALID_PAYLOAD.copy()
        data["credit_utilization_ratio"] = 1.5
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Коэффициент использования кредита должен быть от 0 до 1" in str(exc.value)

    def test_invalid_past_delinquencies_negative(self):
        data = self.VALID_PAYLOAD.copy()
        data["past_delinquencies"] = -1
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Количество просрочек не может быть отрицательным" in str(exc.value)

    def test_invalid_past_delinquencies_too_high(self):
        data = self.VALID_PAYLOAD.copy()
        data["past_delinquencies"] = 11
        with pytest.raises(ValidationError) as exc:
            ClientData(**data)
        assert "Количество просрочек не может превышать 10" in str(exc.value)


class TestResponseSchemas:
    """Тесты для Pydantic-схем ответов."""

    def test_prediction_response_valid(self):
        resp = PredictionResponse(prediction=1, probability=0.85)
        assert resp.prediction == 1
        assert resp.probability == 0.85

    def test_prediction_response_invalid_probability_negative(self):
        with pytest.raises(ValidationError) as exc:
            PredictionResponse(prediction=1, probability=-0.1)
        assert "greater than or equal to 0" in str(exc.value)

    def test_prediction_response_invalid_probability_too_high(self):
        with pytest.raises(ValidationError) as exc:
            PredictionResponse(prediction=1, probability=1.5)
        assert "less than or equal to 1" in str(exc.value)

    def test_batch_prediction_response_valid(self):
        resp = BatchPredictionResponse(predictions=[1, 0], probabilities=[0.85, 0.3])
        assert resp.predictions == [1, 0]
        assert resp.probabilities == [0.85, 0.3]

    def test_health_response_valid(self):
        resp = HealthResponse(status="ok", model_loaded=True)
        assert resp.status == "ok"
        assert resp.model_loaded is True
