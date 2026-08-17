"""Pydantic-схемы для входных и выходных данных API."""

from typing import List
from pydantic import BaseModel, Field, field_validator


class ClientData(BaseModel):
    """Модель входных данных для одного клиента (все признаки)."""

    client_ID: str = Field(..., description="Идентификатор клиента")
    person_age: int = Field(..., description="Возраст клиента")
    person_income: int = Field(..., description="Доход клиента")
    person_home_ownership: str = Field(..., description="Статус собственности")
    person_emp_length: float = Field(..., description="Стаж работы (лет)")
    loan_intent: str = Field(..., description="Цель кредита")
    loan_grade: str = Field(..., description="Грейд кредита (A-G)")
    loan_amnt: int = Field(..., description="Сумма кредита")
    loan_int_rate: float = Field(..., description="Процентная ставка")
    loan_percent_income: float = Field(..., description="Доля кредита от дохода")
    cb_person_default_on_file: str = Field(..., description="Был ли дефолт")
    cb_person_cred_hist_length: int = Field(..., description="Длина кредитной истории")
    gender: str = Field(..., description="Пол")
    marital_status: str = Field(..., description="Семейное положение")
    education_level: str = Field(..., description="Уровень образования")
    country: str = Field(..., description="Страна")
    state: str = Field(..., description="Штат/провинция")
    city: str = Field(..., description="Город")
    city_latitude: float = Field(..., description="Широта города")
    city_longitude: float = Field(..., description="Долгота города")
    employment_type: str = Field(..., description="Тип занятости")
    loan_term_months: int = Field(..., description="Срок кредита в месяцах")
    loan_to_income_ratio: float = Field(..., description="Отношение кредита к доходу")
    other_debt: float = Field(..., description="Прочие долги")
    debt_to_income_ratio: float = Field(..., description="Соотношение долга к доходу")
    open_accounts: int = Field(..., description="Количество открытых счетов")
    credit_utilization_ratio: float = Field(..., description="Коэффициент использования кредита")
    past_delinquencies: int = Field(..., description="Просрочки в прошлом")

    @field_validator("client_ID")
    @classmethod
    def validate_client_id(cls, v: str) -> str:
        if not v.startswith("CUST_"):
            raise ValueError("Идентификатор клиента должен начинаться с 'CUST_'")
        try:
            num_part = int(v.split("_")[1])
            if not (1 <= num_part <= 99999):
                raise ValueError("Номер клиента должен быть от 1 до 99999")
        except (IndexError, ValueError):
            raise ValueError("Идентификатор клиента должен быть в формате 'CUST_XXXXX'")
        return v

    @field_validator("person_age")
    @classmethod
    def validate_person_age(cls, v: int) -> int:
        if not (18 <= v <= 100):
            raise ValueError("Возраст должен быть от 18 до 100 лет")
        return v

    @field_validator("person_income")
    @classmethod
    def validate_person_income(cls, v: int) -> int:
        if v < 1000:
            raise ValueError("Доход не может быть меньше 1000")
        if v > 10_000_000:
            raise ValueError("Доход не может превышать 10,000,000")
        return v

    @field_validator("person_home_ownership")
    @classmethod
    def validate_person_home_ownership(cls, v: str) -> str:
        allowed = {"RENT", "OWN", "MORTGAGE", "OTHER"}
        if v not in allowed:
            raise ValueError(f"Статус собственности должен быть одним из: {', '.join(allowed)}")
        return v

    @field_validator("person_emp_length")
    @classmethod
    def validate_person_emp_length(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Стаж не может быть отрицательным")
        if v > 50:
            raise ValueError("Стаж не может превышать 50 лет")
        return v

    @field_validator("loan_intent")
    @classmethod
    def validate_loan_intent(cls, v: str) -> str:
        allowed = {
            "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
            "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
        }
        if v not in allowed:
            raise ValueError(f"Цель кредита должна быть одной из: {', '.join(allowed)}")
        return v

    @field_validator("loan_grade")
    @classmethod
    def validate_loan_grade(cls, v: str) -> str:
        allowed = {"A", "B", "C", "D", "E", "F", "G"}
        if v not in allowed:
            raise ValueError(f"Грейд кредита должен быть одним из: {', '.join(allowed)}")
        return v

    @field_validator("loan_amnt")
    @classmethod
    def validate_loan_amnt(cls, v: int) -> int:
        if v < 500:
            raise ValueError("Сумма кредита не может быть меньше 500")
        if v > 100_000:
            raise ValueError("Сумма кредита не может превышать 100,000")
        return v

    @field_validator("loan_int_rate")
    @classmethod
    def validate_loan_int_rate(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Процентная ставка не может быть отрицательной")
        if v > 30:
            raise ValueError("Процентная ставка не может превышать 30%")
        return v

    @field_validator("loan_percent_income")
    @classmethod
    def validate_loan_percent_income(cls, v: float) -> float:
        if not (0 <= v <= 1):
            raise ValueError("Доля кредита от дохода должна быть от 0 до 1")
        return v

    @field_validator("cb_person_default_on_file")
    @classmethod
    def validate_cb_person_default_on_file(cls, v: str) -> str:
        if v not in ("Y", "N"):
            raise ValueError("Значение должно быть 'Y' или 'N'")
        return v

    @field_validator("cb_person_cred_hist_length")
    @classmethod
    def validate_cb_person_cred_hist_length(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Длина кредитной истории не может быть отрицательной")
        if v > 50:
            raise ValueError("Длина кредитной истории не может превышать 50 лет")
        return v

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        if v not in ("Male", "Female"):
            raise ValueError("Пол должен быть 'Male' или 'Female'")
        return v

    @field_validator("marital_status")
    @classmethod
    def validate_marital_status(cls, v: str) -> str:
        allowed = {"Married", "Single", "Divorced", "Widowed"}
        if v not in allowed:
            raise ValueError(f"Семейное положение должно быть одним из: {', '.join(allowed)}")
        return v

    @field_validator("education_level")
    @classmethod
    def validate_education_level(cls, v: str) -> str:
        allowed = {"High School", "Bachelor", "Master", "PhD"}
        if v not in allowed:
            raise ValueError(f"Уровень образования должен быть одним из: {', '.join(allowed)}")
        return v

    @field_validator("country")
    @classmethod
    def validate_country(cls, v: str) -> str:
        allowed = {"Canada", "UK", "USA"}
        if v not in allowed:
            raise ValueError(f"Страна должна быть одной из: {', '.join(allowed)}")
        return v

    @field_validator("state")
    @classmethod
    def validate_state(cls, v: str) -> str:
        allowed = {
            "Ontario", "Wales", "BC", "New York", "California",
            "Quebec", "Texas", "Scotland", "England"
        }
        if v not in allowed:
            raise ValueError(f"Штат/провинция должен быть одним из: {', '.join(allowed)}")
        return v

    @field_validator("city")
    @classmethod
    def validate_city(cls, v: str) -> str:
        allowed = {
            "Toronto", "Swansea", "Vancouver", "Buffalo", "San Francisco",
            "Quebec City", "Dallas", "Glasgow", "London", "Montreal",
            "Victoria", "Los Angeles", "New York City", "Ottawa",
            "Edinburgh", "Houston", "Manchester", "Cardiff"
        }
        if v not in allowed:
            raise ValueError(f"Город должен быть одним из: {', '.join(allowed)}")
        return v

    @field_validator("city_latitude")
    @classmethod
    def validate_city_latitude(cls, v: float) -> float:
        if not (-90 <= v <= 90):
            raise ValueError("Широта должна быть от -90 до 90 градусов")
        return v

    @field_validator("city_longitude")
    @classmethod
    def validate_city_longitude(cls, v: float) -> float:
        if not (-180 <= v <= 180):
            raise ValueError("Долгота должна быть от -180 до 180 градусов")
        return v

    @field_validator("employment_type")
    @classmethod
    def validate_employment_type(cls, v: str) -> str:
        allowed = {"Full-time", "Part-time", "Self-employed", "Unemployed"}
        if v not in allowed:
            raise ValueError(f"Тип занятости должен быть одним из: {', '.join(allowed)}")
        return v

    @field_validator("loan_term_months")
    @classmethod
    def validate_loan_term_months(cls, v: int) -> int:
        allowed = [12, 24, 36, 60]
        if v not in allowed:
            raise ValueError(f"Срок кредита должен быть одним из {allowed} (месяцев)")
        return v

    @field_validator("loan_to_income_ratio")
    @classmethod
    def validate_loan_to_income_ratio(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Отношение кредита к доходу не может быть отрицательным")
        if v > 1:
            raise ValueError("Отношение кредита к доходу не может превышать 1")
        return v

    @field_validator("other_debt")
    @classmethod
    def validate_other_debt(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Прочие долги не могут быть отрицательными")
        if v > 2_000_000:
            raise ValueError("Прочие долги не могут превышать 2,000,000")
        return v

    @field_validator("debt_to_income_ratio")
    @classmethod
    def validate_debt_to_income_ratio(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Соотношение долга к доходу не может быть отрицательным")
        if v > 2:
            raise ValueError("Соотношение долга к доходу не может превышать 2")
        return v

    @field_validator("open_accounts")
    @classmethod
    def validate_open_accounts(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Количество открытых счетов не может быть отрицательным")
        if v > 30:
            raise ValueError("Количество открытых счетов не может превышать 30")
        return v

    @field_validator("credit_utilization_ratio")
    @classmethod
    def validate_credit_utilization_ratio(cls, v: float) -> float:
        if not (0 <= v <= 1):
            raise ValueError("Коэффициент использования кредита должен быть от 0 до 1")
        return v

    @field_validator("past_delinquencies")
    @classmethod
    def validate_past_delinquencies(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Количество просрочек не может быть отрицательным")
        if v > 10:
            raise ValueError("Количество просрочек не может превышать 10")
        return v


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Предсказанный класс (0 или 1)")
    probability: float = Field(..., ge=0, le=1, description="Вероятность положительного класса")


class BatchPredictionResponse(BaseModel):
    predictions: List[int] = Field(..., description="Список предсказанных классов")
    probabilities: List[float] = Field(..., description="Список вероятностей положительного класса")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Статус сервиса")
    model_loaded: bool = Field(..., description="Загружена ли модель")
