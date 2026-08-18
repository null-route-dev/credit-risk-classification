"""
Главный модуль управления проектом: обучение, предсказание и API через CLI.
"""

import argparse
import logging
import sys

import pandas as pd
import uvicorn

from src.config import (
    DATA_PATH, KAGGLE_DATASET, MODEL_DIR, TARGET_COLUMN,
    TEST_SIZE, RANDOM_STATE, CV_FOLDS, N_TRIALS,
    INCLUDE_WEAK, USE_OPTUNA, API_HOST, API_PORT,
    API_RELOAD, API_LOG_LEVEL, LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT
)
from src.data_loader import DataLoader
from src.train import LightGBMTrainer
from src.predict import LightGBMPredictor

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
)


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    """Загружает данные из CSV или Kaggle согласно аргументам."""
    data_path = args.data if hasattr(args, 'data') and args.data else DATA_PATH
    
    if args.kaggle:
        return DataLoader.load(
            filepath=data_path,
            kaggle_dataset=args.kaggle,
            force=args.force_kaggle,
        )
    return DataLoader.load(filepath=data_path)


def train_command(args: argparse.Namespace) -> None:
    """Обучает модель LightGBM."""
    logger.info("Загрузка данных для обучения...")
    df = load_data(args)

    logger.info("Инициализация тренера...")
    trainer = LightGBMTrainer(
        test_size=args.test_size or TEST_SIZE,
        random_state=args.random_state or RANDOM_STATE,
        n_trials=args.n_trials or N_TRIALS,
        cv_folds=args.cv_folds or CV_FOLDS,
        include_weak=args.include_weak if args.include_weak is not None else INCLUDE_WEAK,
        use_optuna=args.use_optuna if args.use_optuna is not None else USE_OPTUNA,
    )

    logger.info("Запуск обучения...")
    pipeline, test_metrics = trainer.train(
        df=df,
        target_col=args.target or TARGET_COLUMN,
        save_path=args.model_dir or MODEL_DIR,
    )

    logger.info("Обучение завершено. Метрики на тесте:")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")


def predict_command(args: argparse.Namespace) -> None:
    """Выполняет предсказание на новых данных."""
    logger.info("Загрузка данных для предсказания...")
    df = load_data(args)

    model_dir = args.model_dir or MODEL_DIR
    logger.info(f"Загрузка модели из {model_dir}...")
    predictor = LightGBMPredictor(model_dir)

    logger.info("Выполнение предсказаний...")
    if args.probabilities:
        results = predictor.predict_proba(df)
        out_df = pd.DataFrame({"probability": results})
    else:
        preds = predictor.predict(df, threshold=args.threshold)
        out_df = pd.DataFrame({"prediction": preds})

    if args.output:
        out_df.to_csv(args.output, index=False)
        logger.info(f"Результаты сохранены в {args.output}")
    else:
        print(out_df.head(10).to_string())


def api_command(args: argparse.Namespace) -> None:
    """Запускает FastAPI сервер."""
    host = args.host or API_HOST
    port = args.port or API_PORT
    reload = args.reload if args.reload is not None else API_RELOAD
    log_level = args.log_level or API_LOG_LEVEL
    
    logger.info(f"Запуск API сервера на {host}:{port}")
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Управление обучением, предсказанием и API кредитного скоринга."
    )
    parser.add_argument("--verbose", action="store_true", help="Подробное логирование")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Команда")

    data_parent = argparse.ArgumentParser(add_help=False)
    data_parent.add_argument(
        "data", type=str, nargs='?', default=DATA_PATH,
        help="Путь к CSV-файлу с данными"
    )
    data_parent.add_argument(
        "--kaggle", type=str, default=KAGGLE_DATASET,
        help="Имя датасета на Kaggle (owner/dataset) для загрузки, если файла нет"
    )
    data_parent.add_argument(
        "--force-kaggle", action="store_true",
        help="Принудительно загружать с Kaggle, игнорируя локальный файл"
    )

    train_parser = subparsers.add_parser(
        "train", parents=[data_parent], help="Обучение модели"
    )
    train_parser.add_argument(
        "--target", type=str, default=TARGET_COLUMN,
        help=f"Имя целевой колонки (по умолчанию: {TARGET_COLUMN})"
    )
    train_parser.add_argument(
        "--model-dir", type=str, default=MODEL_DIR,
        help=f"Директория для сохранения модели (по умолчанию: {MODEL_DIR})"
    )
    train_parser.add_argument(
        "--test-size", type=float, default=TEST_SIZE,
        help=f"Доля тестовой выборки (по умолчанию: {TEST_SIZE})"
    )
    train_parser.add_argument(
        "--random-state", type=int, default=RANDOM_STATE,
        help=f"Seed для воспроизводимости (по умолчанию: {RANDOM_STATE})"
    )
    train_parser.add_argument(
        "--include-weak", action="store_true", default=INCLUDE_WEAK,
        help=f"Включать слабые признаки (по умолчанию: {INCLUDE_WEAK})"
    )
    train_parser.add_argument(
        "--use-optuna", action="store_true", default=USE_OPTUNA,
        help=f"Использовать Optuna для подбора гиперпараметров (по умолчанию: {USE_OPTUNA})"
    )
    train_parser.add_argument(
        "--n-trials", type=int, default=N_TRIALS,
        help=f"Количество итераций Optuna (по умолчанию: {N_TRIALS})"
    )
    train_parser.add_argument(
        "--cv-folds", type=int, default=CV_FOLDS,
        help=f"Число фолдов для кросс-валидации (по умолчанию: {CV_FOLDS})"
    )
    train_parser.set_defaults(func=train_command)

    predict_parser = subparsers.add_parser(
        "predict", parents=[data_parent], help="Предсказание на новых данных"
    )
    predict_parser.add_argument(
        "--model-dir", type=str, default=MODEL_DIR,
        help=f"Директория с сохранённой моделью (по умолчанию: {MODEL_DIR})"
    )
    predict_parser.add_argument(
        "--output", type=str, default=None,
        help="Путь для сохранения результатов (CSV). Если не указан, вывод в консоль."
    )
    predict_parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Порог классификации (по умолчанию: 0.5)"
    )
    predict_parser.add_argument(
        "--probabilities", action="store_true",
        help="Выводить вероятности вместо бинарных классов"
    )
    predict_parser.set_defaults(func=predict_command)

    api_parser = subparsers.add_parser(
        "api",
        help="Запуск FastAPI сервера"
    )
    api_parser.add_argument(
        "--host", type=str, default=API_HOST,
        help=f"Хост для сервера (по умолчанию: {API_HOST})"
    )
    api_parser.add_argument(
        "--port", type=int, default=API_PORT,
        help=f"Порт для сервера (по умолчанию: {API_PORT})"
    )
    api_parser.add_argument(
        "--reload", action="store_true", default=API_RELOAD,
        help=f"Автоматическая перезагрузка при изменениях (по умолчанию: {API_RELOAD})"
    )
    api_parser.add_argument(
        "--log-level", type=str, default=API_LOG_LEVEL,
        choices=["critical", "error", "warning", "info", "debug"],
        help=f"Уровень логирования (по умолчанию: {API_LOG_LEVEL})"
    )
    api_parser.set_defaults(func=api_command)

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Ошибка выполнения: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
