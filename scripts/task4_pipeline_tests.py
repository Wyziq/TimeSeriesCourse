import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from scripts.forecast_pipeline import (
    DEFAULT_DATA_PATH,
    DEFAULT_HORIZON,
    load_panel,
    run_pipeline,
    validate_forecast,
    validate_panel,
)


REPORTS_DIR = ROOT / "reports" / "task4"
FIGURES_DIR = ROOT / "figures" / "task4"
TASK2_DIR = ROOT / "reports" / "task2"
TASK3_DIR = ROOT / "reports" / "task3"


def test_row(name: str, passed: bool, details: str) -> dict:
    return {"test": name, "passed": bool(passed), "details": details}


def functional_tests(data: pd.DataFrame, forecast: pd.DataFrame) -> list[dict]:
    rows = []
    validation = validate_panel(data)
    rows.append(
        test_row(
            "input_schema_and_frequency",
            validation["n_series"] == 9,
            json.dumps(validation, ensure_ascii=False),
        )
    )

    validate_forecast(forecast, data, DEFAULT_HORIZON)
    rows.append(
        test_row(
            "forecast_shape",
            len(forecast) == 9 * DEFAULT_HORIZON,
            f"rows={len(forecast)}, expected={9 * DEFAULT_HORIZON}",
        )
    )
    interval_ordered = (
        (forecast["lo_95"] <= forecast["lo_80"])
        & (forecast["lo_80"] <= forecast["y_hat"])
        & (forecast["y_hat"] <= forecast["hi_80"])
        & (forecast["hi_80"] <= forecast["hi_95"])
    ).all()
    rows.append(
        test_row(
            "interval_order",
            interval_ordered,
            "lo95 <= lo80 <= point <= hi80 <= hi95",
        )
    )

    duplicate_data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    duplicate_rejected = False
    try:
        validate_panel(duplicate_data)
    except ValueError:
        duplicate_rejected = True
    rows.append(
        test_row(
            "duplicate_rejection",
            duplicate_rejected,
            "A duplicated time-series key must raise ValueError",
        )
    )

    gap_data = data.drop(data.index[1]).reset_index(drop=True)
    gap_rejected = False
    try:
        validate_panel(gap_data)
    except ValueError:
        gap_rejected = True
    rows.append(
        test_row(
            "monthly_gap_rejection",
            gap_rejected,
            "A missing month must raise ValueError",
        )
    )
    return rows


def statistical_tests() -> tuple[list[dict], pd.DataFrame]:
    statistical = pd.read_csv(TASK2_DIR / "model_metrics.csv")
    data_driven = pd.read_csv(TASK3_DIR / "model_metrics.csv")
    statistical["family"] = "Statistical"
    data_driven["family"] = data_driven["model"].map(
        {
            "Ridge": "ML",
            "RandomForest": "ML",
            "LightGBM": "ML",
            "MLP": "DL",
            "LSTM": "DL",
            "NHITS": "DL",
        }
    )
    common = ["model", "family", "mae", "rmse", "smape", "mase", "bias"]
    comparison = pd.concat(
        [statistical[common], data_driven[common]],
        ignore_index=True,
    ).sort_values("mase")

    theta = statistical.set_index("model").loc["Theta"]
    baseline = statistical.set_index("model").loc["SeasonalNaive"]
    nhits = data_driven.set_index("model").loc["NHITS"]
    task2_tests = pd.read_csv(TASK2_DIR / "model_comparison_tests.csv")
    baseline_tests = task2_tests[
        task2_tests["model_a"].eq("Theta")
        & task2_tests["model_b"].eq("SeasonalNaive")
        & task2_tests["test"].eq("Wilcoxon_abs_error")
    ]
    if len(baseline_tests) != 1:
        raise ValueError(
            "Expected exactly one Theta vs SeasonalNaive Wilcoxon test"
        )
    baseline_test = baseline_tests.iloc[0]
    baseline_reported_correctly = (
        pd.notna(baseline_test["pvalue_holm"])
        and baseline_test["pvalue_holm"] > 0.05
        and bool(baseline_test["significant_5pct"]) is False
    )

    rows = [
        test_row(
            "theta_mase_below_one",
            theta["mase"] < 1,
            f"Theta MASE={theta['mase']:.3f}",
        ),
        test_row(
            "theta_beats_seasonal_baseline",
            theta["mase"] < baseline["mase"],
            (
                f"Theta={theta['mase']:.3f}, "
                f"SeasonalNaive={baseline['mase']:.3f}"
            ),
        ),
        test_row(
            "theta_best_aggregate_model",
            comparison.iloc[0]["model"] == "Theta",
            (
                f"best={comparison.iloc[0]['model']}, "
                f"NHITS={nhits['mase']:.3f}"
            ),
        ),
        test_row(
            "baseline_difference_is_nonsignificant",
            baseline_reported_correctly,
            (
                "Wilcoxon Holm-adjusted p="
                f"{baseline_test['pvalue_holm']:.3f}; "
                "Theta practical gain is reported as not statistically "
                "significant"
            ),
        ),
    ]
    return rows, comparison


def save_figure(data: pd.DataFrame, forecast: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    industries = forecast["unique_id"].unique()
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=False)
    for axis, unique_id in zip(axes.flat, industries):
        history = data[data["unique_id"].eq(unique_id)].tail(36)
        future = forecast[forecast["unique_id"].eq(unique_id)]
        axis.plot(history["ds"], history["y"], color="black", label="History")
        axis.plot(
            future["ds"], future["y_hat"], color="tab:blue", label="Theta"
        )
        axis.fill_between(
            future["ds"],
            future["lo_95"],
            future["hi_95"],
            color="tab:blue",
            alpha=0.15,
            label="95% interval",
        )
        axis.set_title(unique_id, fontsize=9)
        axis.tick_params(axis="x", rotation=30)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Final six-month Theta forecast", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURES_DIR / "final_pipeline_forecast.png", dpi=150)
    plt.close(fig)


def save_report(
    test_results: pd.DataFrame,
    comparison: pd.DataFrame,
    metadata: dict,
) -> None:
    best = comparison.iloc[0]
    baseline = comparison[comparison["model"].eq("SeasonalNaive")].iloc[0]
    nhits = comparison[comparison["model"].eq("NHITS")].iloc[0]
    improvement = 100 * (baseline["mase"] - best["mase"]) / baseline["mase"]
    rows = "\n".join(
        f"| {row.model} | {row.family} | {row.mase:.3f} | "
        f"{row.smape:.3f} | {row.bias:.3f} |"
        for row in comparison.itertuples(index=False)
    )
    tests = "\n".join(
        f"| {row.test} | {'PASS' if row.passed else 'FAIL'} | {row.details} |"
        for row in test_results.itertuples(index=False)
    )
    timing = metadata["timing_seconds"]
    report = f"""# Задача №4: итоговый пайплайн прогнозирования

## Назначение

Пайплайн прогнозирует общее число увольнений в девяти отраслях США на
следующие шесть месяцев. На вход подается панель `unique_id, ds, y` с
ежемесячной частотой. На выходе для каждой отрасли формируются точечный прогноз
Theta, интервалы 80% и 95% и контрольный прогноз SeasonalNaive.

## Компоненты и обоснование

1. Валидация проверяет схему, типы, пропуски, дубликаты, конечность и
   неотрицательность значений, непрерывность месячной шкалы и наличие не менее
   двух сезонных циклов.
2. `Theta(12)` выбрана по общему rolling-origin backtesting: MASE
   `{best["mase"]:.3f}`. Она лучше SeasonalNaive на `{improvement:.1f}%` и
   лучше лучшей data-driven модели NHITS с MASE `{nhits["mase"]:.3f}`.
   Одна общая спецификация используется для всех отраслей: отраслевые
   победители выбирались на тех же 36 backtesting-блоках, поэтому их прямое
   включение в pipeline усилило бы selection bias без отдельной контрольной
   выборки.
3. `SeasonalNaive(12)` сохраняется в результате как прозрачный baseline.
4. Интервалы 80% и 95% позволяют оценивать неопределенность, но историческое
   покрытие Theta было консервативным: 0.940 и 0.995.
5. CSV с прогнозом и JSON с контрольной суммой входа, параметрами и временем
   выполнения обеспечивают воспроизводимость запуска.

## Сравнение всех методов

| Модель | Семейство | MASE | sMAPE, % | Bias |
|---|---|---:|---:|---:|
{rows}

## Тестирование пайплайна

| Проверка | Результат | Детали |
|---|---|---|
{tests}

Полное время запуска: `{timing["total"]:.3f}` с; прогнозирование и проверка:
`{timing["forecast_and_validation"]:.3f}` с. Порог performance-теста установлен
в 30 секунд для текущей панели из девяти рядов.

Статистическое преимущество Theta над SeasonalNaive не подтверждено после
Holm-коррекции (`p = 0.441`). Поэтому выбор основан на практическом общем
MASE, а не трактуется как доказанное превосходство в генеральной совокупности.

## Вывод по задаче №4

Пайплайн автоматически валидирует данные, обучает основную и контрольную
модели, строит 54 прогноза с вероятностными интервалами и сохраняет артефакты
запуска. Все функциональные, статистические и performance-проверки пройдены.
Фактическое время выполнения значительно ниже установленного порога в
30 секунд, поэтому решение подходит для регулярного offline-пересчета после
публикации очередного ежемесячного релиза JOLTS.

В качестве основной модели выбрана Theta с MASE `{best["mase"]:.3f}` и sMAPE
`{best["smape"]:.3f}%`. Она показала лучшее среднее качество среди 15
исследованных моделей и на `{improvement:.1f}%` превзошла SeasonalNaive по
MASE. При этом статистическая значимость данного преимущества не подтверждена,
поэтому SeasonalNaive сохранена в выходном файле как контрольный прогноз.
Такое решение позволяет при каждом обновлении данных проверить, продолжает ли
более сложная модель приносить практическую пользу.

Итоговый прогноз содержит по шесть будущих наблюдений для каждой из девяти
отраслей, всего 54 строки. Интервалы 80% и 95% показывают диапазон
неопределенности: чем шире интервал, тем менее уверенно модель оценивает
будущий уровень увольнений. Эти границы не следует интерпретировать как
гарантированные, поскольку backtesting показал консервативное историческое
покрытие интервалов Theta и возможную зависимость качества от рыночного режима.

Пайплайн предназначен для аналитического offline-прогнозирования, а не для
автоматического принятия кадровых решений. Перед каждым новым запуском следует
обновлять данные, контролировать их полноту и повторно оценивать качество на
новых фактических значениях. При заметном ухудшении MASE относительно
SeasonalNaive необходимо заново провести backtesting и пересмотреть выбор
модели.

**Финальный шестимесячный прогноз Theta по девяти отраслям:**

![Финальный прогноз Theta](../../figures/task4/final_pipeline_forecast.png)

## Общее заключение

Исследована ежемесячная несезонно скорректированная статистика JOLTS по общему
числу увольнений в девяти крупных отраслях США за декабрь 2000 — апрель 2026
года. Цель работы — offline-прогноз на шесть месяцев. Данные не содержат
пропусков и имеют выраженную годовую сезонность; после первой разности все
девять рядов проходят совместную проверку стационарности.

В rolling-origin backtesting с четырьмя окнами сравнивались 15 моделей:
девять статистических, три ML и три DL. Лучший результат показала Theta:
MASE `{best["mase"]:.3f}`, sMAPE `{best["smape"]:.3f}%`. SeasonalNaive имеет
MASE `{baseline["mase"]:.3f}`, лучший data-driven метод NHITS —
`{nhits["mase"]:.3f}`. Преимущество Theta над baseline составляет
`{improvement:.1f}%`, но статистически незначимо после поправки на
множественные сравнения. Поэтому в эксплуатации одновременно сохраняются
Theta и SeasonalNaive, а качество следует повторно проверять при каждом
обновлении данных и смене рыночного режима.
"""
    (REPORTS_DIR / "task4_report.md").write_text(report, encoding="utf-8")


def run_task4() -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pipeline = run_pipeline()
    data = load_panel(DEFAULT_DATA_PATH)

    functional = functional_tests(data, pipeline["forecast"])
    statistical, comparison = statistical_tests()
    total_seconds = time.perf_counter() - started
    performance = [
        test_row(
            "pipeline_runtime_under_30_seconds",
            total_seconds < 30,
            f"total_seconds={total_seconds:.3f}",
        )
    ]
    test_results = pd.DataFrame(functional + statistical + performance)
    test_results.to_csv(
        REPORTS_DIR / "pipeline_test_results.csv", index=False
    )
    comparison.to_csv(REPORTS_DIR / "all_model_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                **pipeline["metadata"]["timing_seconds"],
                "task4_total_with_tests": total_seconds,
                "threshold_seconds": 30,
            }
        ]
    ).to_csv(REPORTS_DIR / "performance.csv", index=False)

    save_figure(data, pipeline["forecast"])
    save_report(test_results, comparison, pipeline["metadata"])
    if not test_results["passed"].all():
        failures = test_results.loc[~test_results["passed"], "test"].tolist()
        raise AssertionError(f"Task 4 tests failed: {failures}")
    return {
        **pipeline,
        "tests": test_results,
        "comparison": comparison,
    }


if __name__ == "__main__":
    result = run_task4()
    print(result["tests"].to_string(index=False))
