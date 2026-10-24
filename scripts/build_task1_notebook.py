from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "01_task1_eda.ipynb"


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
            # Задача №1: подготовка данных и EDA временного ряда

            Цель ноутбука — подготовить набор данных JOLTS/FRED к анализу,
            проверить качество временной панели, сформулировать задачу
            прогнозирования и выполнить базовый EDA.
            """
        ),
        markdown(
            """
            ## Постановка задачи

            Прогнозируется `total_separations` — ежемесячное общее число
            увольнений работников, в тысячах человек. Горизонт прогноза —
            6 месяцев. Единица ряда — отрасль экономики США.

            Данные несезонно скорректированы, поэтому сезонность исследуется
            явно. Признаки `quits` и `layoffs_discharges` являются компонентами
            целевой переменной, поэтому в прогнозных моделях допустимы только
            их лаги или отдельные прогнозы.
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

            from scripts.task1_eda import run_eda, FIGURES_DIR, REPORTS_DIR
            """
        ),
        markdown(
            """
            ## Запуск подготовки EDA

            Функция `run_eda()` читает подготовленную панель, валидирует
            непрерывность месячных рядов, строит таблицы EDA и сохраняет
            графики в `figures/task1`.
            """
        ),
        code(
            """
            results = run_eda()
            data = results["data"]
            summary = results["summary"]
            stationarity = results["stationarity"]
            stl = results["stl"]
            split = results["split"]
            """
        ),
        markdown("## Размерность и качество данных"),
        code(
            """
            print(f"Rows: {len(data):,}")
            print(f"Industries: {data['industry'].nunique()}")
            print(f"Date range: {data['ds'].min().date()} - {data['ds'].max().date()}")
            print("\\nData types:")
            print(data.dtypes.to_string())
            print("\\nMissing values:")
            print(data.isna().sum().to_string())
            print("\\nDuplicates:", data.duplicated(["industry", "ds"]).sum())
            """
        ),
        code(
            """
            data.head()
            """
        ),
        markdown("## Описательные статистики целевого ряда"),
        code(
            """
            summary.round(3)
            """
        ),
        markdown(
            """
            ## Визуальный анализ рядов

            На графиках видны различия масштаба между отраслями, сезонность и
            сильные шоки периода COVID-19, особенно в Leisure and Hospitality.
            """
        ),
        code(
            """
            display(Image(filename=FIGURES_DIR / "total_separations_by_industry.png"))
            """
        ),
        markdown("## Сезонные профили"),
        code(
            """
            display(Image(filename=FIGURES_DIR / "seasonal_profiles.png"))
            """
        ),
        markdown("## STL-декомпозиция на примере Professional and Business Services"),
        code(
            """
            display(Image(filename=FIGURES_DIR / "stl_professional_business_services.png"))
            """
        ),
        code(
            """
            stl.sort_values("seasonal_strength", ascending=False).round(3)
            """
        ),
        markdown("## Проверка стационарности"),
        code(
            """
            stationarity_pivot = stationarity.pivot_table(
                index="industry",
                columns="transformation",
                values=["adf_pvalue", "kpss_pvalue"],
                aggfunc="first",
            )
            stationarity_pivot.round(4)
            """
        ),
        markdown(
            """
            Интерпретация: ADF отвергает гипотезу единичного корня при малом
            p-value, KPSS отвергает стационарность при малом p-value. Поэтому
            для уверенного вывода желательно, чтобы ADF p-value был меньше 0.05,
            а KPSS p-value был не меньше 0.05.
            """
        ),
        markdown("## Описательные корреляции"),
        code(
            """
            display(Image(filename=FIGURES_DIR / "contemporaneous_correlations.png"))
            """
        ),
        markdown(
            """
            Корреляции выше являются описательными. Для прогнозирования нельзя
            использовать одновременные значения `quits` и `layoffs_discharges`,
            потому что они входят в состав `total_separations`.
            """
        ),
        markdown("## Train/test split для следующих задач"),
        code(
            """
            split
            """
        ),
        markdown("## Вывод по задаче №1"),
        code(
            """
            print((REPORTS_DIR / "task1_report.md").read_text(encoding="utf-8"))
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
