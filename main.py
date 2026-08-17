"""
Главный модуль управления проектом: обучение, предсказание и API через CLI.
"""

import argparse
import logging
import sys

import pandas as pd
import uvicorn

from src.data_loader import DataLoader
from src.train import LightGBMTrainer
from src.predict import LightGBMPredictor

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    """
    Загружает данные из CSV или Kaggle согласно аргументам.
    """
    if args.kaggle:
        return DataLoader.load(
            filepath=args.data,
            kaggle_dataset=args.kaggle,
            force=args.force_kaggle,
        )
    return DataLoader.load(filepath=args.data)


def train_command(args: argparse.Namespace) -> None:
    """Обучает модель LightGBM."""
    logger.info("Загрузка данных для обучения...")
    df = load_data(args)

    logger.info("Инициализация тренера...")
    trainer = LightGBMTrainer(
        test_size=args.test_size,
        random_state=args.random_state,
        n_trials=args.n_trials,
        cv_folds=args.cv_folds,
        include_weak=args.include_weak,
        use_optuna=args.use_optuna,
    )

    logger.info("Запуск обучения...")
    pipeline, test_metrics = trainer.train(
        df=df,
        target_col=args.target,
        save_path=args.model_dir,
    )

    logger.info("Обучение завершено. Метрики на тесте:")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")


def predict_command(args: argparse.Namespace) -> None:
    """Выполняет предсказание на новых данных."""
    logger.info("Загрузка данных для предсказания...")
    df = load_data(args)

    logger.info(f"Загрузка модели из {args.model_dir}...")
    predictor = LightGBMPredictor(args.model_dir)

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
    logger.info(f"Запуск API сервера на {args.host}:{args.port}")
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Управление обучением, предсказанием и API кредитного скоринга."
    )
    parser.add_argument("--verbose", action="store_true", help="Подробное логирование")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Команда")

    data_parent = argparse.ArgumentParser(add_help=False)
    data_parent.add_argument(
        "data", type=str, help="Путь к CSV-файлу с данными"
    )
    data_parent.add_argument(
        "--kaggle", type=str, default=None,
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
        "--target", type=str, default="loan_status",
        help="Имя целевой колонки (по умолчанию: loan_status)"
    )
    train_parser.add_argument(
        "--model-dir", type=str, default="models",
        help="Директория для сохранения модели (по умолчанию: models)"
    )
    train_parser.add_argument(
        "--test-size", type=float, default=0.2, help="Доля тестовой выборки"
    )
    train_parser.add_argument(
        "--random-state", type=int, default=42, help="Seed для воспроизводимости"
    )
    train_parser.add_argument(
        "--include-weak", action="store_true",
        help="Включать слабые признаки (по умолчанию выключено)"
    )
    train_parser.add_argument(
        "--use-optuna", action="store_true",
        help="Использовать Optuna для подбора гиперпараметров"
    )
    train_parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Количество итераций Optuna (по умолчанию: 50)"
    )
    train_parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Число фолдов для кросс-валидации (по умолчанию: 5)"
    )
    train_parser.set_defaults(func=train_command)

    predict_parser = subparsers.add_parser(
        "predict", parents=[data_parent], help="Предсказание на новых данных"
    )
    predict_parser.add_argument(
        "--model-dir", type=str, default="models",
        help="Директория с сохранённой моделью (по умолчанию: models)"
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
        "--host", type=str, default="127.0.0.1",
        help="Хост для сервера (по умолчанию: 127.0.0.1)"
    )
    api_parser.add_argument(
        "--port", type=int, default=8000,
        help="Порт для сервера (по умолчанию: 8000)"
    )
    api_parser.add_argument(
        "--reload", action="store_true",
        help="Автоматическая перезагрузка при изменениях"
    )
    api_parser.add_argument(
        "--log-level", type=str, default="info",
        choices=["critical", "error", "warning", "info", "debug"],
        help="Уровень логирования (по умолчанию: info)"
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
