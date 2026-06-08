import os
from pathlib import Path
import warnings

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss


DATA_PATH = ROOT / "data" / "processed" / "jolts_industry_panel.csv"
REPORTS_DIR = ROOT / "reports" / "task1"
FIGURES_DIR = ROOT / "figures" / "task1"
TARGET = "total_separations"
HORIZON = 6


def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, parse_dates=["ds"])
    data = data.sort_values(["industry", "ds"]).reset_index(drop=True)
    return data


def validate_data(data: pd.DataFrame) -> None:
    required = {
        "industry",
        "ds",
        "job_openings",
        "hires",
        "total_separations",
        "quits",
        "layoffs_discharges",
    }
    if set(data.columns) != required:
        raise ValueError(f"Unexpected columns: {list(data.columns)}")
    if data.duplicated(["industry", "ds"]).any():
        raise ValueError("Duplicated industry/month rows found")
    if data.isna().any().any():
        raise ValueError("Missing values found")

    expected_dates = pd.date_range(data["ds"].min(), data["ds"].max(), freq="MS")
    for industry, group in data.groupby("industry"):
        dates = pd.DatetimeIndex(group["ds"])
        if not dates.equals(expected_dates):
            raise ValueError(f"{industry} does not have a continuous MS index")


def descriptive_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for industry, group in data.groupby("industry", sort=True):
        y = group[TARGET]
        rows.append(
            {
                "industry": industry,
                "observations": len(group),
                "start": group["ds"].min().date(),
                "end": group["ds"].max().date(),
                "mean": y.mean(),
                "std": y.std(),
                "min": y.min(),
                "max": y.max(),
                "cv": y.std() / y.mean(),
                "acf_lag_1": y.autocorr(1),
                "acf_lag_12": y.autocorr(12),
            }
        )
    return pd.DataFrame(rows)


def stationarity_test(values: pd.Series) -> dict[str, float]:
    values = values.dropna().astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_pvalue = adfuller(values, autolag="AIC")[1]
        kpss_pvalue = kpss(values, regression="c", nlags="auto")[1]
    return {"adf_pvalue": adf_pvalue, "kpss_pvalue": kpss_pvalue}


def stationarity_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    transformations = {
        "level": lambda values: values,
        "diff_1": lambda values: values.diff(),
        "diff_12": lambda values: values.diff(12),
        "diff_1_12": lambda values: values.diff(12).diff(),
    }
    for industry, group in data.groupby("industry", sort=True):
        for transformation, transform in transformations.items():
            result = stationarity_test(transform(group[TARGET]))
            rows.append(
                {
                    "industry": industry,
                    "transformation": transformation,
                    **result,
                    "adf_stationary_5pct": result["adf_pvalue"] < 0.05,
                    "kpss_stationary_5pct": result["kpss_pvalue"] >= 0.05,
                }
            )
    return pd.DataFrame(rows)


def stl_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for industry, group in data.groupby("industry", sort=True):
        y = group.set_index("ds")[TARGET].asfreq("MS")
        result = STL(y, period=12, robust=True).fit()
        residual_variance = np.var(result.resid)
        seasonal_strength = max(
            0.0,
            1.0 - residual_variance / np.var(result.seasonal + result.resid),
        )
        trend_strength = max(
            0.0,
            1.0 - residual_variance / np.var(result.trend + result.resid),
        )
        rows.append(
            {
                "industry": industry,
                "seasonal_strength": seasonal_strength,
                "trend_strength": trend_strength,
                "residual_std": np.std(result.resid),
            }
        )
    return pd.DataFrame(rows)


def seasonal_profile(data: pd.DataFrame) -> pd.DataFrame:
    profile = data.assign(month=data["ds"].dt.month).copy()
    profile["seasonal_index"] = profile[TARGET] / profile.groupby("industry")[
        TARGET
    ].transform("mean")
    return (
        profile.groupby(["industry", "month"], as_index=False)["seasonal_index"]
        .mean()
        .sort_values(["industry", "month"])
    )


def split_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for industry, group in data.groupby("industry", sort=True):
        test = group.tail(HORIZON)
        train = group.iloc[:-HORIZON]
        rows.append(
            {
                "industry": industry,
                "train_observations": len(train),
                "train_end": train["ds"].max().date(),
                "test_observations": len(test),
                "test_start": test["ds"].min().date(),
                "test_end": test["ds"].max().date(),
            }
        )
    return pd.DataFrame(rows)


def save_figures(data: pd.DataFrame, profile: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    industries = sorted(data["industry"].unique())
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
    for industry, axis in zip(industries, axes.flat):
        group = data[data["industry"].eq(industry)]
        axis.plot(group["ds"], group[TARGET], linewidth=1)
        axis.set_title(industry, fontsize=10)
        axis.set_ylabel("Thousand workers")
    fig.suptitle("Monthly total separations by industry", fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "total_separations_by_industry.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
    for industry, axis in zip(industries, axes.flat):
        group = profile[profile["industry"].eq(industry)]
        axis.plot(group["month"], group["seasonal_index"], marker="o")
        axis.axhline(1.0, color="black", linewidth=0.8)
        axis.set_title(industry, fontsize=10)
        axis.set_xticks(range(1, 13))
    fig.suptitle("Average monthly seasonal index", fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "seasonal_profiles.png", dpi=160)
    plt.close(fig)

    representative = "Professional and Business Services"
    y = (
        data[data["industry"].eq(representative)]
        .set_index("ds")[TARGET]
        .asfreq("MS")
    )
    decomposition = STL(y, period=12, robust=True).fit()
    fig = decomposition.plot()
    fig.set_size_inches(14, 10)
    fig.suptitle(f"STL decomposition: {representative}", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "stl_professional_business_services.png", dpi=160)
    plt.close(fig)

    correlations = data[
        [
            "job_openings",
            "hires",
            "total_separations",
            "quits",
            "layoffs_discharges",
        ]
    ].corr()
    fig, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlations, annot=True, fmt=".2f", cmap="coolwarm", ax=axis)
    axis.set_title("Contemporaneous correlations (descriptive only)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "contemporaneous_correlations.png", dpi=160)
    plt.close(fig)


def save_report(
    data: pd.DataFrame,
    summary: pd.DataFrame,
    stationarity: pd.DataFrame,
    stl: pd.DataFrame,
    split: pd.DataFrame,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    level_stationary = stationarity[
        stationarity["transformation"].eq("level")
        & stationarity["adf_stationary_5pct"]
        & stationarity["kpss_stationary_5pct"]
    ]["industry"].nunique()
    transformed_stationary = stationarity[
        stationarity["transformation"].eq("diff_1_12")
        & stationarity["adf_stationary_5pct"]
        & stationarity["kpss_stationary_5pct"]
    ]["industry"].nunique()
    strongest_seasonality = stl.nlargest(3, "seasonal_strength")[
        ["industry", "seasonal_strength"]
    ]
    strongest_trend = stl.nlargest(3, "trend_strength")[
        ["industry", "trend_strength"]
    ]
    dtype_md = "\n".join(
        [
            "| column | dtype |",
            "|---|---|",
            *[f"| {column} | {dtype} |" for column, dtype in data.dtypes.items()],
        ]
    )
    strongest_seasonality_md = "\n".join(
        [
            "| industry | seasonal_strength |",
            "|---|---:|",
            *[
                f"| {row.industry} | {row.seasonal_strength:.3f} |"
                for row in strongest_seasonality.itertuples(index=False)
            ],
        ]
    )
    strongest_trend_md = "\n".join(
        [
            "| industry | trend_strength |",
            "|---|---:|",
            *[
                f"| {row.industry} | {row.trend_strength:.3f} |"
                for row in strongest_trend.itertuples(index=False)
            ],
        ]
    )

    report = f"""# Задача №1: подготовка данных и EDA

## Результаты проверки данных

- Панель содержит {data["industry"].nunique()} отраслей и {len(data):,} строк.
- Каждый ряд содержит {data.groupby("industry").size().iloc[0]} ежемесячных наблюдений.
- Период: {data["ds"].min().date()} — {data["ds"].max().date()}.
- Пропуски и дубликаты отсутствуют.
- Частота всех рядов: начало месяца (`MS`).
- Единица измерения: тысячи работников.

Типы данных после чтения:

{dtype_md}

## Постановка прогноза

- Цель: `total_separations`.
- Горизонт: {HORIZON} месяцев.
- Последняя тестовая выборка: {split["test_start"].min()} — {split["test_end"].max()}.
- Режим: offline batch forecast после выхода очередного месячного релиза.
- Основные метрики следующих этапов: MAE, RMSE и sMAPE; для сравнения отраслей
  разных масштабов дополнительно MASE.
- Локальные модели используются как отраслевые бенчмарки, потому что ряды
  различаются масштабом, сезонностью и реакцией на шоки.
- Глобальные модели обучаются на общей панели, чтобы использовать общие
  закономерности и увеличить число обучающих примеров для ML/DL.

## Основные наблюдения EDA

- Средняя величина увольнений различается между отраслями, поэтому одной
  абсолютной метрики недостаточно.
- Лаговая автокорреляция и сезонная автокорреляция приведены в
  `descriptive_summary.csv`.
- По совместному критерию ADF/KPSS стационарны в уровнях {level_stationary} из
  {data["industry"].nunique()} рядов.
- После комбинированного преобразования `diff_1_12` (сезонная разность с лагом
  12 и затем первая разность) стационарны
  {transformed_stationary} из {data["industry"].nunique()} рядов.
- Наиболее выраженная сезонность по STL:

{strongest_seasonality_md}

- Наиболее выраженный тренд по STL:

{strongest_trend_md}

Графики:

![Целевые ряды по отраслям](../../figures/task1/total_separations_by_industry.png)

![Сезонные профили](../../figures/task1/seasonal_profiles.png)

![STL-декомпозиция Professional and Business Services](../../figures/task1/stl_professional_business_services.png)

![Описательные одновременные корреляции](../../figures/task1/contemporaneous_correlations.png)

## Методологические ограничения

- `quits` и `layoffs_discharges` являются компонентами `total_separations`.
  Их значения за прогнозируемый месяц создают прямую утечку цели.
- В моделях допустимы только прошлые лаги этих показателей либо их отдельные
  прогнозы.
- `job_openings` является stock-показателем на конец месяца, остальные
  показатели — flow за месяц. Для прогноза предпочтительны лаги вакансий.
- Данные FRED содержат пересмотренную историю. Будущий backtesting оценивает
  модели на revised history, а не на исторических vintages.

## Вывод по задаче №1

Данные подготовлены для локальных и глобальных моделей. Девять рядов имеют
одинаковую непрерывную месячную шкалу и достаточную длину для rolling-origin
backtesting. Исходные уровни нельзя считать стационарными без проверки;
для статистических моделей следует сравнить явное дифференцирование с
автоматическим выбором преобразований.
"""
    (REPORTS_DIR / "task1_report.md").write_text(report, encoding="utf-8")

    summary.to_csv(REPORTS_DIR / "descriptive_summary.csv", index=False)
    stationarity.to_csv(REPORTS_DIR / "stationarity_tests.csv", index=False)
    stl.to_csv(REPORTS_DIR / "stl_strength.csv", index=False)
    split.to_csv(REPORTS_DIR / "forecast_split.csv", index=False)


def run_eda() -> dict[str, pd.DataFrame]:
    data = load_data()
    validate_data(data)
    summary = descriptive_summary(data)
    stationarity = stationarity_summary(data)
    stl = stl_summary(data)
    profile = seasonal_profile(data)
    split = split_summary(data)
    save_figures(data, profile)
    save_report(data, summary, stationarity, stl, split)
    return {
        "data": data,
        "summary": summary,
        "stationarity": stationarity,
        "stl": stl,
        "seasonal_profile": profile,
        "split": split,
    }


if __name__ == "__main__":
    results = run_eda()
    print(results["summary"].round(3).to_string(index=False))
