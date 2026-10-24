from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "04_task4_pipeline.ipynb"


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
            # Задача №4: итоговый пайплайн

            Пайплайн валидирует панель JOLTS, строит шестимесячный прогноз
            Theta с интервалами и сохраняет SeasonalNaive как baseline.
            """
        ),
        code(
            """
            from pathlib import Path
            import sys
            import json
            import pandas as pd
            from IPython.display import Image, display, Markdown

            ROOT = Path.cwd().resolve()
            if ROOT.name == "notebooks":
                ROOT = ROOT.parent
            sys.path.insert(0, str(ROOT))

            from scripts.forecast_pipeline import load_panel, validate_panel
            from scripts.task4_pipeline_tests import run_task4
            """
        ),
        markdown("## Входные данные и проверки"),
        code(
            """
            data = load_panel(ROOT / "data/processed/total_separations_nf.csv")
            validation = validate_panel(data)
            validation
            """
        ),
        markdown(
            """
            Проверяются обязательные поля, пропуски, дубликаты, типы,
            непрерывность месячной шкалы и достаточная длина истории.
            """
        ),
        markdown("## Полный запуск"),
        code(
            """
            result = run_task4()
            forecast = result["forecast"]
            tests = result["tests"]
            comparison = result["comparison"]
            """
        ),
        markdown("## Сравнение 15 моделей"),
        code(
            """
            comparison[
                ["model", "family", "mase", "smape", "bias"]
            ].round(3)
            """
        ),
        markdown("## Результаты тестирования"),
        code("tests"),
        markdown("## Метаданные запуска"),
        code(
            """
            print(json.dumps(result["metadata"], ensure_ascii=False, indent=2))
            """
        ),
        markdown("## Финальный прогноз"),
        code("forecast.round(2)"),
        code(
            """
            display(
                Image(
                    filename=ROOT
                    / "figures/task4/final_pipeline_forecast.png"
                )
            )
            """
        ),
        markdown("## Итоговый отчет"),
        code(
            """
            display(
                Markdown(
                    (
                        ROOT / "reports/task4/task4_report.md"
                    ).read_text(encoding="utf-8")
                )
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
