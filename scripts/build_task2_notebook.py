from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "02_task2_statistical_models.ipynb"


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(source).strip())


def markdown(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def build_notebook() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python (timeseries)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    notebook["cells"] = [
        markdown(
            """
            # Задача №2: статистические модели прогнозирования

            В notebook сравниваются ручные и автоматические статистические
            методы из `statsforecast`. Выбор выполняется по rolling-origin
            backtesting, вероятностным интервалам и анализу ошибок.
            """
        ),
        markdown(
            """
            ## Дизайн эксперимента

            - Горизонт: 6 месяцев.
            - 4 окна backtesting с шагом 6 месяцев.
            - 9 отраслевых временных рядов.
            - Метрики: MAE, RMSE, sMAPE и сезонный MASE.
            - Интервалы: 80% и 95%.
            """
        ),
        code(
            """
            from pathlib import Path
            import sys

            ROOT = Path.cwd().resolve()
            if ROOT.name == "notebooks":
                ROOT = ROOT.parent
            sys.path.insert(0, str(ROOT))

            from IPython.display import Image, display
            import pandas as pd

            REPORTS_DIR = ROOT / "reports" / "task2"
            FIGURES_DIR = ROOT / "figures" / "task2"

            from scripts.task2_statistical_models import (
                build_models,
                run_task2,
            )
            """
        ),
        markdown(
            """
            ## Код эксперимента

            Конфигурация моделей хранится в воспроизводимом Python-модуле.
            Полный расчет можно запустить из notebook, установив флаг ниже.
            По умолчанию используются уже сохраненные результаты, поскольку
            повторный backtesting занимает несколько минут.
            """
        ),
        code(
            """
            build_models()
            """
        ),
        code(
            """
            RUN_FULL_EXPERIMENT = False
            if RUN_FULL_EXPERIMENT:
                run_task2()
            """
        ),
        markdown(
            """
            ## Загрузка результатов

            Полный backtesting выполняется скриптом
            `scripts/task2_statistical_models.py`. Notebook загружает готовые
            артефакты, чтобы не повторять тяжелые расчеты при каждом открытии.
            """
        ),
        code(
            """
            metrics = pd.read_csv(REPORTS_DIR / "model_metrics.csv")
            industry_metrics = pd.read_csv(
                REPORTS_DIR / "model_metrics_by_industry.csv"
            )
            horizon_metrics = pd.read_csv(
                REPORTS_DIR / "model_metrics_by_horizon.csv"
            )
            diagnostics = pd.read_csv(REPORTS_DIR / "residual_diagnostics.csv")
            fitted_diagnostics = pd.read_csv(
                REPORTS_DIR / "fitted_residual_diagnostics.csv"
            )
            tests = pd.read_csv(REPORTS_DIR / "model_comparison_tests.csv")
            final_forecasts = pd.read_csv(REPORTS_DIR / "final_forecasts.csv")
            """
        ),
        markdown("## Сравнение моделей"),
        code(
            """
            metrics.round(3)
            """
        ),
        code(
            """
            display(Image(filename=FIGURES_DIR / "model_metrics.png"))
            """
        ),
        markdown(
            """
            `SeasonalNaive` является основным baseline. MASE меньше 1 означает,
            что модель в среднем точнее сезонного наивного масштаба ошибок.
            """
        ),
        markdown("## Результаты по отраслям"),
        code(
            """
            industry_metrics.pivot(
                index="unique_id",
                columns="model",
                values="mase",
            ).round(3)
            """
        ),
        markdown("## Результаты по горизонту"),
        code(
            """
            horizon_metrics.pivot(
                index="horizon",
                columns="model",
                values="mase",
            ).round(3)
            """
        ),
        markdown("## Последнее окно backtesting"),
        code(
            """
            display(Image(filename=FIGURES_DIR / "best_model_last_window.png"))
            """
        ),
        markdown("## Вероятностные интервалы"),
        code(
            """
            metrics[
                ["model", "coverage_80", "coverage_95"]
            ].round(3)
            """
        ),
        markdown("## Анализ ошибок и остатков"),
        code(
            """
            display(Image(filename=FIGURES_DIR / "forecast_error_boxplots.png"))
            """
        ),
        code(
            """
            diagnostics.groupby("model").agg(
                mean_bias=("mean_error", "mean"),
                white_noise_share=("white_noise_5pct", "mean"),
                median_ljung_box_pvalue=("ljung_box_pvalue_lag6", "median"),
            ).round(3)
            """
        ),
        markdown("### Классические fitted residuals"),
        code(
            """
            fitted_diagnostics.groupby("model").agg(
                mean_residual=("mean_residual", "mean"),
                white_noise_share=("white_noise_5pct", "mean"),
                median_ljung_box_pvalue=("ljung_box_pvalue", "median"),
            ).round(3)
            """
        ),
        markdown("## Статистическое сравнение моделей"),
        code(
            """
            tests
            """
        ),
        markdown("## Финальный шестимесячный прогноз"),
        code(
            """
            display(Image(filename=FIGURES_DIR / "final_forecast.png"))
            """
        ),
        code(
            """
            final_forecasts.head(12)
            """
        ),
        markdown("## Вывод по задаче №2"),
        code(
            """
            print((REPORTS_DIR / "task2_report.md").read_text(encoding="utf-8"))
            """
        ),
    ]
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
