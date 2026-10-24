import logging
import os
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from prophet import Prophet
from scipy.stats import friedmanchisquare, shapiro, wilcoxon
from statsforecast import StatsForecast
from statsforecast.models import (
    ARIMA,
    AutoARIMA,
    AutoETS,
    AutoTheta,
    Naive,
    SeasonalNaive,
    Theta,
)
from statsmodels.stats.diagnostic import acorr_ljungbox

DATA_PATH = ROOT / "data" / "processed" / "total_separations_nf.csv"
REPORTS_DIR = ROOT / "reports" / "task2"
FIGURES_DIR = ROOT / "figures" / "task2"
HORIZON = 6
N_WINDOWS = 4
STEP_SIZE = 6
SEASON_LENGTH = 12
LEVELS = [80, 95]
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").disabled = True

MODEL_NAMES = [
    "SeasonalNaive",
    "Naive",
    "ARIMA_manual",
    "AutoARIMA",
    "ETS_manual",
    "AutoETS",
    "Theta",
    "AutoTheta",
    "Prophet_manual",
]


def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, parse_dates=["ds"])
    return data.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def build_models() -> list:
    return [
        SeasonalNaive(season_length=SEASON_LENGTH),
        Naive(),
        ARIMA(
            order=(1, 1, 1),
            season_length=SEASON_LENGTH,
            seasonal_order=(1, 1, 1),
            alias="ARIMA_manual",
        ),
        AutoARIMA(season_length=SEASON_LENGTH),
        AutoETS(
            season_length=SEASON_LENGTH,
            model="AAA",
            damped=False,
            alias="ETS_manual",
        ),
        AutoETS(season_length=SEASON_LENGTH),
        Theta(season_length=SEASON_LENGTH),
        AutoTheta(season_length=SEASON_LENGTH),
    ]


def run_cross_validation(data: pd.DataFrame) -> pd.DataFrame:
    stats_forecast = StatsForecast(
        models=build_models(),
        freq="MS",
        n_jobs=-1,
    )
    cv = stats_forecast.cross_validation(
        df=data,
        h=HORIZON,
        n_windows=N_WINDOWS,
        step_size=STEP_SIZE,
        level=LEVELS,
        refit=True,
    )
    return add_prophet_cross_validation(cv, data)


def prophet_model(interval_width: float = 0.95) -> Prophet:
    return Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        interval_width=interval_width,
        uncertainty_samples=500,
    )


def prophet_predictions(
    train: pd.DataFrame,
    dates: pd.Series | pd.DatetimeIndex,
) -> pd.DataFrame:
    future = pd.DataFrame({"ds": pd.to_datetime(dates)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = prophet_model().fit(train)
        point = model.predict(future)["yhat"].to_numpy()
        samples = model.predictive_samples(future)["yhat"]
    return pd.DataFrame(
        {
            "ds": future["ds"],
            "Prophet_manual": point,
            "Prophet_manual-lo-80": np.quantile(samples, 0.10, axis=1),
            "Prophet_manual-hi-80": np.quantile(samples, 0.90, axis=1),
            "Prophet_manual-lo-95": np.quantile(samples, 0.025, axis=1),
            "Prophet_manual-hi-95": np.quantile(samples, 0.975, axis=1),
        }
    )


def add_prophet_cross_validation(
    cv: pd.DataFrame,
    data: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for unique_id, cutoff in (
        cv[["unique_id", "cutoff"]].drop_duplicates().itertuples(index=False, name=None)
    ):
        train = data[data["unique_id"].eq(unique_id) & data["ds"].le(cutoff)][
            ["ds", "y"]
        ]
        dates = cv[cv["unique_id"].eq(unique_id) & cv["cutoff"].eq(cutoff)][
            "ds"
        ].sort_values()
        prediction = prophet_predictions(train, dates)
        prediction.insert(0, "cutoff", cutoff)
        prediction.insert(0, "unique_id", unique_id)
        rows.extend(prediction.to_dict("records"))
    prophet_cv = pd.DataFrame(rows)
    return cv.merge(
        prophet_cv,
        on=["unique_id", "cutoff", "ds"],
        how="left",
        validate="one_to_one",
    )


def seasonal_scale(
    data: pd.DataFrame,
    unique_id: str,
    cutoff: pd.Timestamp,
) -> float:
    train = data[data["unique_id"].eq(unique_id) & data["ds"].le(cutoff)]["y"].astype(
        float
    )
    scale = train.diff(SEASON_LENGTH).abs().dropna().mean()
    return float(scale)


def add_scales(cv: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    scales = {
        (unique_id, cutoff): seasonal_scale(data, unique_id, cutoff)
        for unique_id, cutoff in cv[["unique_id", "cutoff"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    result = cv.copy()
    result["mase_scale"] = [
        scales[(unique_id, cutoff)]
        for unique_id, cutoff in result[["unique_id", "cutoff"]].itertuples(
            index=False, name=None
        )
    ]
    return result


def interval_score(
    actual: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    alpha: float,
) -> pd.Series:
    return (
        upper
        - lower
        + (2.0 / alpha) * (lower - actual).clip(lower=0)
        + (2.0 / alpha) * (actual - upper).clip(lower=0)
    )


def metric_row(group: pd.DataFrame, model: str) -> dict:
    errors = group[model] - group["y"]
    absolute_errors = errors.abs()
    denominator = group[model].abs() + group["y"].abs()
    return {
        "mae": absolute_errors.mean(),
        "rmse": np.sqrt(np.mean(errors**2)),
        "smape": np.mean(
            np.where(
                denominator.eq(0),
                0.0,
                200.0 * absolute_errors / denominator,
            )
        ),
        "mase": np.mean(absolute_errors / group["mase_scale"]),
        "bias": errors.mean(),
        "coverage_80": np.mean(
            (group["y"] >= group[f"{model}-lo-80"])
            & (group["y"] <= group[f"{model}-hi-80"])
        ),
        "coverage_95": np.mean(
            (group["y"] >= group[f"{model}-lo-95"])
            & (group["y"] <= group[f"{model}-hi-95"])
        ),
        "mean_width_95": np.mean(group[f"{model}-hi-95"] - group[f"{model}-lo-95"]),
        "interval_score_95": np.mean(
            interval_score(
                group["y"],
                group[f"{model}-lo-95"],
                group[f"{model}-hi-95"],
                alpha=0.05,
            )
        ),
    }


def model_metrics(
    cv: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    industry_rows = []
    horizon_rows = []
    for model in MODEL_NAMES:
        metrics = {"model": model, **metric_row(cv, model)}
        rows.append(metrics)

        for unique_id, group in cv.groupby("unique_id"):
            industry_rows.append(
                {
                    "unique_id": unique_id,
                    "model": model,
                    **metric_row(group, model),
                }
            )
        for horizon, group in cv.groupby("horizon"):
            horizon_rows.append(
                {
                    "horizon": horizon,
                    "model": model,
                    **metric_row(group, model),
                }
            )

    aggregate = pd.DataFrame(rows).sort_values(["mase", "smape"])
    by_industry = pd.DataFrame(industry_rows).sort_values(["unique_id", "mase"])
    by_horizon = pd.DataFrame(horizon_rows).sort_values(["horizon", "mase"])
    return aggregate, by_industry, by_horizon


def residual_diagnostics(cv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_NAMES:
        for unique_id, group in cv.groupby("unique_id"):
            errors = (
                group.sort_values(["cutoff", "ds"])[model]
                - group.sort_values(["cutoff", "ds"])["y"]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ljung_box_pvalue = acorr_ljungbox(
                    errors,
                    lags=[6],
                    return_df=True,
                )["lb_pvalue"].iloc[0]
                shapiro_pvalue = shapiro(errors).pvalue
            rows.append(
                {
                    "model": model,
                    "unique_id": unique_id,
                    "observations": len(errors),
                    "mean_error": errors.mean(),
                    "std_error": errors.std(),
                    "ljung_box_pvalue_lag6": ljung_box_pvalue,
                    "white_noise_5pct": ljung_box_pvalue >= 0.05,
                    "shapiro_pvalue": shapiro_pvalue,
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "unique_id"])


def fitted_residual_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    stats_forecast = StatsForecast(
        models=build_models(),
        freq="MS",
        n_jobs=-1,
    )
    stats_forecast.forecast(df=data, h=1, fitted=True)
    fitted = stats_forecast.forecast_fitted_values()

    prophet_rows = []
    for unique_id, group in data.groupby("unique_id"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = prophet_model().fit(group[["ds", "y"]])
            prediction = model.predict(group[["ds"]])
        prophet_rows.append(
            group[["unique_id", "ds"]].assign(
                Prophet_manual=prediction["yhat"].to_numpy()
            )
        )
    fitted = fitted.merge(
        pd.concat(prophet_rows, ignore_index=True),
        on=["unique_id", "ds"],
        how="left",
        validate="one_to_one",
    )

    rows = []
    for model in MODEL_NAMES:
        for unique_id, group in fitted.groupby("unique_id"):
            residuals = (group["y"] - group[model]).dropna()
            lag = min(12, max(1, len(residuals) // 5))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ljung_box_pvalue = acorr_ljungbox(
                    residuals,
                    lags=[lag],
                    return_df=True,
                )["lb_pvalue"].iloc[0]
            rows.append(
                {
                    "model": model,
                    "unique_id": unique_id,
                    "observations": len(residuals),
                    "mean_residual": residuals.mean(),
                    "std_residual": residuals.std(),
                    "ljung_box_lag": lag,
                    "ljung_box_pvalue": ljung_box_pvalue,
                    "white_noise_5pct": ljung_box_pvalue >= 0.05,
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "unique_id"])


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running_max = 0.0
    count = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * pvalues[index])
        running_max = max(running_max, value)
        adjusted[index] = running_max
    return adjusted.tolist()


def comparison_tests(
    cv: pd.DataFrame,
    aggregate_metrics: pd.DataFrame,
) -> pd.DataFrame:
    blocks = cv[["unique_id", "cutoff"]].copy()
    absolute_errors = {
        model: (
            blocks.assign(error=(cv[model] - cv["y"]).abs())
            .groupby(["unique_id", "cutoff"])["error"]
            .mean()
            .to_numpy()
        )
        for model in MODEL_NAMES
    }
    friedman = friedmanchisquare(*[absolute_errors[model] for model in MODEL_NAMES])
    best_model = aggregate_metrics.iloc[0]["model"]

    rows = [
        {
            "test": "Friedman",
            "model_a": "all",
            "model_b": "all",
            "statistic": friedman.statistic,
            "pvalue": friedman.pvalue,
            "pvalue_holm": friedman.pvalue,
            "significant_5pct": friedman.pvalue < 0.05,
        }
    ]
    raw_pvalues = []
    pair_rows = []
    for model in MODEL_NAMES:
        if model == best_model:
            continue
        result = wilcoxon(
            absolute_errors[best_model],
            absolute_errors[model],
            alternative="less",
            zero_method="zsplit",
        )
        raw_pvalues.append(result.pvalue)
        pair_rows.append(
            {
                "test": "Wilcoxon_abs_error",
                "model_a": best_model,
                "model_b": model,
                "statistic": result.statistic,
                "pvalue": result.pvalue,
            }
        )

    adjusted = holm_adjust(raw_pvalues)
    for row, adjusted_pvalue in zip(pair_rows, adjusted):
        row["pvalue_holm"] = adjusted_pvalue
        row["significant_5pct"] = adjusted_pvalue < 0.05
        rows.append(row)
    return pd.DataFrame(rows)


def final_forecast(data: pd.DataFrame) -> pd.DataFrame:
    stats_forecast = StatsForecast(
        models=build_models(),
        freq="MS",
        n_jobs=-1,
    )
    forecast = stats_forecast.forecast(
        df=data,
        h=HORIZON,
        level=LEVELS,
    )
    rows = []
    for unique_id, group in data.groupby("unique_id"):
        future_dates = pd.date_range(
            group["ds"].max() + pd.offsets.MonthBegin(),
            periods=HORIZON,
            freq="MS",
        )
        prediction = prophet_predictions(group[["ds", "y"]], future_dates)
        prediction.insert(0, "unique_id", unique_id)
        rows.extend(prediction.to_dict("records"))
    return forecast.merge(
        pd.DataFrame(rows),
        on=["unique_id", "ds"],
        how="left",
        validate="one_to_one",
    )


def save_figures(
    data: pd.DataFrame,
    cv: pd.DataFrame,
    aggregate_metrics: pd.DataFrame,
    final_predictions: pd.DataFrame,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    metric_order = aggregate_metrics.sort_values("mase")["model"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(
        data=aggregate_metrics,
        x="mase",
        y="model",
        order=metric_order,
        ax=axes[0],
        color="#4C72B0",
    )
    axes[0].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Rolling-origin MASE")
    sns.barplot(
        data=aggregate_metrics,
        x="smape",
        y="model",
        order=metric_order,
        ax=axes[1],
        color="#55A868",
    )
    axes[1].set_title("Rolling-origin sMAPE, %")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_metrics.png", dpi=160)
    plt.close(fig)

    best_model = aggregate_metrics.iloc[0]["model"]
    last_cutoff = cv["cutoff"].max()
    last_window = cv[cv["cutoff"].eq(last_cutoff)]
    industries = sorted(data["unique_id"].unique())
    fig, axes = plt.subplots(3, 3, figsize=(18, 13), sharex=False)
    for unique_id, axis in zip(industries, axes.flat):
        history = data[data["unique_id"].eq(unique_id)].tail(42)
        forecast = last_window[last_window["unique_id"].eq(unique_id)]
        axis.plot(history["ds"], history["y"], label="Actual", color="black")
        axis.plot(
            forecast["ds"],
            forecast[best_model],
            label=best_model,
            color="#C44E52",
        )
        axis.fill_between(
            forecast["ds"],
            forecast[f"{best_model}-lo-95"],
            forecast[f"{best_model}-hi-95"],
            color="#C44E52",
            alpha=0.15,
            label="95% PI",
        )
        axis.set_title(unique_id, fontsize=10)
        axis.tick_params(axis="x", rotation=30)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle(f"Last backtest window: {best_model}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURES_DIR / "best_model_last_window.png", dpi=160)
    plt.close(fig)

    errors = []
    for model in MODEL_NAMES:
        model_errors = pd.DataFrame(
            {
                "model": model,
                "error": cv[model] - cv["y"],
            }
        )
        errors.append(model_errors)
    error_frame = pd.concat(errors, ignore_index=True)
    fig, axis = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=error_frame, x="model", y="error", ax=axis)
    axis.axhline(0, color="black", linewidth=1)
    axis.tick_params(axis="x", rotation=30)
    axis.set_title("Out-of-sample forecast errors")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "forecast_error_boxplots.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    for unique_id, axis in zip(industries, axes.flat):
        history = data[data["unique_id"].eq(unique_id)].tail(36)
        forecast = final_predictions[final_predictions["unique_id"].eq(unique_id)]
        axis.plot(history["ds"], history["y"], color="black", label="Actual")
        axis.plot(
            forecast["ds"],
            forecast[best_model],
            color="#4C72B0",
            label=best_model,
        )
        axis.fill_between(
            forecast["ds"],
            forecast[f"{best_model}-lo-95"],
            forecast[f"{best_model}-hi-95"],
            color="#4C72B0",
            alpha=0.15,
        )
        axis.set_title(unique_id, fontsize=10)
        axis.tick_params(axis="x", rotation=30)
    fig.suptitle(f"Six-month final forecast: {best_model}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURES_DIR / "final_forecast.png", dpi=160)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = [header, separator]
    for row in selected.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def save_report(
    aggregate_metrics: pd.DataFrame,
    industry_metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    fitted_diagnostics: pd.DataFrame,
    tests: pd.DataFrame,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    best = aggregate_metrics.iloc[0]
    baseline = aggregate_metrics[aggregate_metrics["model"].eq("SeasonalNaive")].iloc[0]
    improvement = 100.0 * (baseline["mase"] - best["mase"]) / baseline["mase"]
    diagnostics_best = diagnostics[diagnostics["model"].eq(best["model"])]
    white_noise_share = diagnostics_best["white_noise_5pct"].mean()
    fitted_best = fitted_diagnostics[fitted_diagnostics["model"].eq(best["model"])]
    fitted_white_noise_share = fitted_best["white_noise_5pct"].mean()
    friedman_pvalue = tests[tests["test"].eq("Friedman")]["pvalue"].iloc[0]
    baseline_test = tests[
        tests["model_b"].eq("SeasonalNaive") & tests["test"].eq("Wilcoxon_abs_error")
    ].iloc[0]
    industry_winners = (
        industry_metrics.sort_values(["unique_id", "mase"])
        .groupby("unique_id", as_index=False)
        .first()[["unique_id", "model"]]
    )
    winner_summary = "; ".join(
        f"{row.unique_id}: {row.model}"
        for row in industry_winners.itertuples(index=False)
    )

    report = f"""# Задача №2: статистические методы прогнозирования

## Дизайн эксперимента

- Фреймворк: `statsforecast`.
- Горизонт одного прогноза: {HORIZON} месяцев.
- Rolling-origin backtesting: {N_WINDOWS} окна с шагом {STEP_SIZE} месяцев.
- Всего оценено {N_WINDOWS * HORIZON * 9} совпадающих прогнозов на модель.
- Метрики: MAE, RMSE, sMAPE и сезонный MASE.
- Вероятностные интервалы: 80% и 95%.

Горизонт совпадает с постановкой задачи №1. Неперекрывающиеся окна покрывают
последние 24 месяца и позволяют проверить модели в нескольких недавних
рыночных режимах без утечки будущих данных. Основная метрика — сезонный MASE,
поскольку масштабы увольнений существенно различаются между отраслями.

## Сравниваемые модели

- `SeasonalNaive(12)` — основной сезонный baseline.
- `Naive` — прогноз последним наблюдением.
- `ARIMA_manual` — ARIMA(1,1,1)(1,1,1)[12].
- `AutoARIMA` — автоматический выбор порядков по AICc.
- `ETS_manual` — фиксированная аддитивная ETS(A,A,A).
- `AutoETS` — автоматический выбор ETS-компонент.
- `Theta` — классическая сезонная Theta.
- `AutoTheta` — автоматический выбор варианта Theta.
- `Prophet_manual` — отдельная модель Prophet с аддитивной годовой
  сезонностью и стандартной регуляризацией тренда. Prophet не является частью
  `statsforecast`, поэтому подключен через официальный пакет `prophet`.

Ручная ARIMA использует первую и сезонную разности, поскольку EDA показал
нестационарность уровней и годовую сезонность. Низкие порядки AR и MA выбраны
как интерпретируемый ограниченный вариант без перебора на тестовых окнах.
Аддитивная ETS соответствует примерно постоянной амплитуде сезонных колебаний
в уровнях. Автоматические варианты служат проверкой того, дает ли подбор
структуры преимущество перед этими фиксированными спецификациями.

## Результаты backtesting

{markdown_table(aggregate_metrics, ["model", "mae", "rmse", "smape", "mase", "bias", "coverage_80", "coverage_95", "interval_score_95"])}

Лучшая модель по MASE: **{best["model"]}** (`{best["mase"]:.3f}`).
Улучшение относительно SeasonalNaive: **{improvement:.1f}%**.

Агрегированный победитель не является лучшим во всех отраслях. Локальные
победители: {winner_summary}.

## Вероятностный прогноз

Для интервалов проверены эмпирическое покрытие, средняя ширина и interval score.
Идеальные значения coverage равны 0.80 и 0.95. Основной рейтинг строится по
MASE; интервальные метрики используются как отдельная проверка надежности, а
не смешиваются с точечной метрикой без заранее заданных весов.

## Анализ остатков

Для каждой модели и отрасли отдельно проверены классические fitted residuals
и последовательности вневыборочных многошаговых ошибок. По Ljung–Box доля
отраслей без обнаруженной автокорреляции fitted residuals лучшей модели равна
**{fitted_white_noise_share:.1%}**. Для out-of-sample ошибок эта доля равна
**{white_noise_share:.1%}**. Последняя диагностика основана только на 24
ошибках на отрасль и интерпретируется как дополнительный сигнал. Низкая доля
для fitted residuals означает, что даже лучшая по прогнозной метрике модель
не описывает всю временную зависимость и не должна считаться окончательно
специфицированной.

## Статистическое сравнение

Friedman-тест применен к MAE, агрегированным по 36 блокам
«отрасль × окно»: p-value = **{friedman_pvalue:.3g}**. Парные сравнения
лучшей модели с остальными выполнены критерием Wilcoxon с Holm-коррекцией.
Поскольку лучшая модель выбрана на тех же данных, p-value являются
исследовательскими, а не подтверждающими.

## Вывод по задаче №2

Выбрана модель **{best["model"]}** как основной статистический кандидат по
сезонному MASE. Интервальные метрики и диагностика вневыборочных ошибок
использованы для проверки надежности вывода. Сезонный baseline сохраняется
как обязательная контрольная модель. Преимущество над SeasonalNaive после
Holm-коррекции статистически не подтверждено
(`p = {baseline_test["pvalue_holm"]:.3f}`), поэтому разницу следует считать
умеренным практическим улучшением, а не окончательной победой модели.

![Сравнение метрик](../../figures/task2/model_metrics.png)

![Последнее окно backtesting](../../figures/task2/best_model_last_window.png)

![Ошибки прогнозов](../../figures/task2/forecast_error_boxplots.png)

![Финальный прогноз](../../figures/task2/final_forecast.png)
"""
    (REPORTS_DIR / "task2_report.md").write_text(report, encoding="utf-8")


def run_task2() -> dict[str, pd.DataFrame]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    cv = run_cross_validation(data)
    cv["horizon"] = (
        (cv["ds"].dt.year - cv["cutoff"].dt.year) * 12
        + cv["ds"].dt.month
        - cv["cutoff"].dt.month
    )
    cv = add_scales(cv, data)
    aggregate_metrics, industry_metrics, horizon_metrics = model_metrics(cv)
    diagnostics = residual_diagnostics(cv)
    fitted_diagnostics = fitted_residual_diagnostics(data)
    tests = comparison_tests(cv, aggregate_metrics)
    final_predictions = final_forecast(data)

    cv.to_csv(REPORTS_DIR / "cross_validation_predictions.csv", index=False)
    aggregate_metrics.to_csv(
        REPORTS_DIR / "model_metrics.csv",
        index=False,
    )
    industry_metrics.to_csv(
        REPORTS_DIR / "model_metrics_by_industry.csv",
        index=False,
    )
    horizon_metrics.to_csv(
        REPORTS_DIR / "model_metrics_by_horizon.csv",
        index=False,
    )
    diagnostics.to_csv(
        REPORTS_DIR / "residual_diagnostics.csv",
        index=False,
    )
    fitted_diagnostics.to_csv(
        REPORTS_DIR / "fitted_residual_diagnostics.csv",
        index=False,
    )
    tests.to_csv(
        REPORTS_DIR / "model_comparison_tests.csv",
        index=False,
    )
    final_predictions.to_csv(
        REPORTS_DIR / "final_forecasts.csv",
        index=False,
    )

    save_figures(data, cv, aggregate_metrics, final_predictions)
    save_report(
        aggregate_metrics,
        industry_metrics,
        diagnostics,
        fitted_diagnostics,
        tests,
    )
    return {
        "data": data,
        "cv": cv,
        "metrics": aggregate_metrics,
        "industry_metrics": industry_metrics,
        "horizon_metrics": horizon_metrics,
        "diagnostics": diagnostics,
        "fitted_diagnostics": fitted_diagnostics,
        "tests": tests,
        "final_forecasts": final_predictions,
    }


if __name__ == "__main__":
    results = run_task2()
    print(results["metrics"].round(3).to_string(index=False))
