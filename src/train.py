"""
Модуль для обучения модели LightGBM.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

from .preprocess import Preprocessor
from .features import FeatureEngineer

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


class LightGBMTrainer:
    """
    Класс для обучения модели LightGBM.

    Attributes:
        test_size (float): Доля тестовой выборки.
        random_state (int): Seed для воспроизводимости.
        n_trials (int): Количество итераций Optuna (используется только при use_optuna=True).
        cv_folds (int): Число фолдов для кросс-валидации.
        include_weak (bool): Включать ли слабые признаки в FeatureEngineer.
        use_optuna (bool): Флаг, использовать ли Optuna для подбора гиперпараметров.
        preprocessor (Optional[ColumnTransformer]): Кастомный препроцессор (если None, создаётся стандартный).
        default_params (Dict[str, Any]): Параметры по умолчанию для случая use_optuna=False.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        n_trials: int = 50,
        cv_folds: int = 5,
        include_weak: bool = False,
        use_optuna: bool = False,
        preprocessor: Optional[ColumnTransformer] = None,
        default_params: Optional[Dict[str, Any]] = None,
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.include_weak = include_weak
        self.use_optuna = use_optuna
        self.preprocessor = preprocessor

        self.default_params = default_params or {
            "n_estimators": 1500,
            "max_depth": 4,
            "learning_rate": 0.018,
            "num_leaves": 16,
            "subsample": 0.6,
            "colsample_bytree": 0.6,
            "min_child_samples": 48,
            "reg_alpha": 10,
            "reg_lambda": 1,
        }

        self._pipeline: Optional[Pipeline] = None
        self._best_params: Optional[Dict[str, Any]] = None
        self._train_metrics: Optional[Dict[str, float]] = None
        self._test_metrics: Optional[Dict[str, float]] = None
        self._feature_importance: Optional[pd.DataFrame] = None

    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Создаёт стандартный ColumnTransformer для предобработки признаков для LightGBM.

        Returns:
            ColumnTransformer: Настроенный трансформер для числовых и категориальных признаков.
        """
        num_cols = [
            "person_age", "person_income", "person_emp_length", "loan_amnt",
            "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
            "debt_to_income_ratio", "open_accounts", "credit_utilization_ratio",
            "income_debt_balance", "is_high_debt", "is_high_loan_pct", "is_high_rate",
            "emp_length_missing", "loan_term_months", "past_delinquencies"
        ]

        grade_categories = [["A", "B", "C", "D", "E", "F", "G"]]
        grade_encoder = OrdinalEncoder(
            categories=grade_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

        onehot_binary_cols = ["cb_person_default_on_file", "gender"]
        onehot_cols = [
            "person_home_ownership", "loan_intent", "marital_status",
            "education_level", "employment_type"
        ]
        high_card_cols = ["city", "grade_ownership", "default_grade"]

        return ColumnTransformer([
            ("num", "passthrough", num_cols),
            ("grade_ordinal", grade_encoder, ["loan_grade"]),
            ("onehot_binary", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=True), onehot_binary_cols),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True), onehot_cols),
            ("onehot_high", OneHotEncoder(handle_unknown="ignore", sparse_output=True), high_card_cols)
        ])

    def _build_pipeline(self, params: Dict[str, Any], preprocessor: ColumnTransformer) -> Pipeline:
        """
        Строит полный пайплайн с заданными параметрами модели.

        Args:
            params: Словарь гиперпараметров для LGBMClassifier.
            preprocessor: ColumnTransformer для предобработки признаков.

        Returns:
            Pipeline: Готовый пайплайн.
        """
        params_with_defaults = params.copy()
        params_with_defaults.setdefault("random_state", self.random_state)
        params_with_defaults.setdefault("n_jobs", -1)
        params_with_defaults.setdefault("verbose", -1)

        if "scale_pos_weight" not in params_with_defaults:
            if hasattr(self, "_y_train") and self._y_train is not None:
                params_with_defaults["scale_pos_weight"] = (self._y_train == 0).sum() / (self._y_train == 1).sum()
            else:
                params_with_defaults["scale_pos_weight"] = 3.5

        return Pipeline([
            ("basic_preprocessing", Preprocessor()),
            ("feature_engineer", FeatureEngineer(include_weak=self.include_weak)),
            ("custom_preprocessing", preprocessor),
            ("classifier", LGBMClassifier(**params_with_defaults)),
        ])

    def _objective(self, trial, X_train: pd.DataFrame, y_train: pd.Series,
                   preprocessor: ColumnTransformer, cv: StratifiedKFold) -> float:
        """
        Целевая функция для Optuna.

        Args:
            trial: Объект Optuna Trial.
            X_train: Обучающие признаки.
            y_train: Целевая переменная.
            preprocessor: Препроцессор.
            cv: Объект кросс-валидации.

        Returns:
            float: Средний ROC-AUC по фолдам.
        """
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 1500, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.02, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 24, step=4),
            "subsample": trial.suggest_float("subsample", 0.5, 0.7),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.6),
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 60),
            "reg_alpha": trial.suggest_float("reg_alpha", 5.0, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
            "scale_pos_weight": scale_pos_weight,
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbose": -1,
        }

        pipeline = self._build_pipeline(params, preprocessor)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    def _save_model(self, pipeline: Pipeline, save_dir: Path) -> None:
        """
        Сохраняет обученный пайплайн в файл.

        Args:
            pipeline: Обученный пайплайн.
            save_dir: Путь для сохранения модели.
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / "lgbm_final_model.pkl"
        joblib.dump(pipeline, model_path)
        logger.info(f"Модель сохранена в {model_path}")

    def _save_metadata(self, save_dir: Path) -> None:
        """
        Сохраняет метаданные модели (гиперпараметры, метрики, список признаков) в JSON.

        Args:
            save_dir: Директория для сохранения файла.
        """
        info = {
            "model_type": "LightGBM",
            "best_params": self._best_params,
            "cv_best_score": float(self._cv_best_score) if hasattr(self, "_cv_best_score") else None,
            "train_metrics": self._train_metrics,
            "test_metrics": self._test_metrics,
            "features_used": list(self._X_train.columns) if hasattr(self, "_X_train") else None,
        }
        info_path = save_dir / "model_info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
        logger.info(f"Метаданные сохранены в {info_path}")

    def _save_feature_importance(self, pipeline: Pipeline, save_dir: Path) -> None:
        """
        Сохраняет важность признаков в CSV.

        Args:
            pipeline: Обученный пайплайн.
            save_dir: Директория для сохранения файла.
        """
        feature_names = pipeline[:-1].get_feature_names_out()
        importances = pipeline.named_steps["classifier"].feature_importances_
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        importance_path = save_dir / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"Важность признаков сохранена в {importance_path}")

    def train(
        self,
        df: pd.DataFrame,
        target_col: str,
        save_path: Optional[Union[str, Path]] = None,
        fixed_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Pipeline, Dict[str, float]]:
        """
        Запускает полный цикл обучения модели.

        Args:
            df: pandas DataFrame с данными.
            target_col: Название целевой колонки.
            save_path: Полный путь для сохранения модели.
                       Если указан, все файлы сохраняются в его родительскую директорию.
                       Если не указан, сохранение не выполняется.
            fixed_params: Параметры для модели (используются только если use_optuna=False).

        Returns:
            Кортеж (обученный пайплайн, словарь с метриками на тесте).
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        self._X_train = X_train
        self._y_train = y_train

        logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

        preprocessor = self.preprocessor if self.preprocessor is not None else self._create_preprocessor()

        if self.use_optuna:
            logger.info(f"Запуск Optuna ({self.n_trials} trials)...")
            import optuna
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            study = optuna.create_study(direction="maximize", study_name="lgbm_optimization")
            study.optimize(
                lambda trial: self._objective(trial, X_train, y_train, preprocessor, cv),
                n_trials=self.n_trials,
                show_progress_bar=True,
            )
            self._best_params = study.best_params
            self._cv_best_score = study.best_value
            logger.info(f"Лучший ROC-AUC (CV): {self._cv_best_score:.4f}")
            for k, v in self._best_params.items():
                logger.info(f"  {k}: {v}")
            params_to_use = self._best_params
        else:
            params_to_use = fixed_params if fixed_params is not None else self.default_params
            logger.info("Используются фиксированные гиперпараметры:")
            for k, v in params_to_use.items():
                logger.info(f"  {k}: {v}")
            self._best_params = params_to_use

        final_pipeline = self._build_pipeline(params_to_use, preprocessor)
        logger.info("Финальное обучение на всей обучающей выборке...")
        final_pipeline.fit(X_train, y_train)

        y_train_pred_proba = final_pipeline.predict_proba(X_train)[:, 1]
        y_test_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]
        y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
        y_test_pred = (y_test_pred_proba >= 0.5).astype(int)

        self._train_metrics = {
            "roc_auc": roc_auc_score(y_train, y_train_pred_proba),
            "recall": recall_score(y_train, y_train_pred),
            "precision": precision_score(y_train, y_train_pred),
            "f1": f1_score(y_train, y_train_pred),
        }
        self._test_metrics = {
            "roc_auc": roc_auc_score(y_test, y_test_pred_proba),
            "recall": recall_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred),
            "f1": f1_score(y_test, y_test_pred),
        }

        logger.info("Метрики на обучающей выборке:")
        for k, v in self._train_metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        logger.info("Метрики на тестовой выборке:")
        for k, v in self._test_metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        if save_path is not None:
            save_dir = Path(save_path)
            self._save_model(final_pipeline, save_dir)
            self._save_metadata(save_dir)
            self._save_feature_importance(final_pipeline, save_dir)

        self._pipeline = final_pipeline
        return final_pipeline, self._test_metrics
