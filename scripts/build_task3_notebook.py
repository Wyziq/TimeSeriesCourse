from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "03_task3_ml_dl_models.ipynb"


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
            # Задача №3: ML и DL для временных рядов

            Сравниваются три глобальные ML-модели из `mlforecast` и три
            глобальные DL-модели из `neuralforecast`.
            """
        ),
        markdown(
            """
            ## Эксперимент

            - горизонт: 6 месяцев;
            - 4 неперекрывающихся backtesting-окна;
            - ML: Ridge, RandomForest, LightGBM;
            - DL: MLP, LSTM, NHITS;
            - основная метрика: сезонный MASE.
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

            from scripts.task3_ml_dl_models import (
                build_dl_models,
                build_mlforecast,
                run_task3,
            )

            REPORTS_DIR = ROOT / "reports" / "task3"
            FIGURES_DIR = ROOT / "figures" / "task3"
            """
        ),
        markdown("## Конфигурация моделей"),
        code(
            """
            build_mlforecast().models
            """
        ),
        code(
            """
            [model.alias for model in build_dl_models(max_steps=150)]
            """
        ),
        markdown(
            """
            ML-признаки включают лаги, rolling statistics, календарные и
            статические отраслевые признаки. MLP получает последние 12
            месяцев, LSTM и NHITS — 24 месяца; модели оптимизируют quantile
            loss.
            """
        ),
        code(
            """
            RUN_FULL_EXPERIMENT = True
            if RUN_FULL_EXPERIMENT:
                run_task3(dl_max_steps=150)
            """
        ),
        markdown("## Загрузка результатов"),
        code(
            """
            metrics = pd.read_csv(REPORTS_DIR / "model_metrics.csv")
            industry_metrics = pd.read_csv(
                REPORTS_DIR / "model_metrics_by_industry.csv"
            )
            horizon_metrics = pd.read_csv(
                REPORTS_DIR / "model_metrics_by_horizon.csv"
            )
            runtime = pd.read_csv(REPORTS_DIR / "runtime.csv")
            tests = pd.read_csv(
                REPORTS_DIR / "model_comparison_tests.csv"
            )
            configuration_study = pd.read_csv(
                REPORTS_DIR / "configuration_study.csv"
            )
            predictions = pd.read_csv(
                REPORTS_DIR / "cross_validation_predictions.csv",
                parse_dates=["ds", "cutoff"],
            )
            """
        ),
        markdown("## Предтестовая настройка"),
        code("configuration_study.round(3)"),
        markdown("## Общее сравнение"),
        code("metrics.round(3)"),
        code(
            """
            display(Image(filename=FIGURES_DIR / "model_comparison.png"))
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
        code(
            """
            display(Image(filename=FIGURES_DIR / "mase_by_horizon.png"))
            """
        ),
        markdown("## Вероятностные оценки DL"),
        code(
            """
            metrics[
                ["model", "coverage_80", "coverage_95"]
            ].dropna().round(3)
            """
        ),
        markdown("## Последнее окно backtesting"),
        code(
            """
            display(
                Image(filename=FIGURES_DIR / "best_model_last_window.png")
            )
            """
        ),
        markdown("## Время расчёта"),
        code("runtime.round(2)"),
        markdown("## Статистическое сравнение"),
        code("tests"),
        markdown("## Вывод"),
        code(
            """
            print(
                (REPORTS_DIR / "task3_report.md").read_text(encoding="utf-8")
            )
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
