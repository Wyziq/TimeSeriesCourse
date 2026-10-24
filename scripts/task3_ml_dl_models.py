import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
from mlforecast.utils import PredictionIntervals
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import LSTM, MLP, NHITS
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = ROOT / "data" / "processed" / "total_separations_nf.csv"
TASK2_CV_PATH = ROOT / "reports" / "task2" / "cross_validation_predictions.csv"
REPORTS_DIR = ROOT / "reports" / "task3"
FIGURES_DIR = ROOT / "figures" / "task3"
HORIZON = 6
N_WINDOWS = 4
STEP_SIZE = 6
SEASON_LENGTH = 12
LEVELS = [80, 95]

ML_MODELS = ["Ridge", "RandomForest", "LightGBM"]
DL_MODELS = ["MLP", "LSTM", "NHITS"]
MODEL_NAMES = ML_MODELS + DL_MODELS


def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, parse_dates=["ds"])
    return data.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def add_static_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    industries = sorted(data["unique_id"].unique())
    result = data.copy()
    static_columns = []
    for index, industry in enumerate(industries):
        column = f"industry_{index}"
        result[column] = result["unique_id"].eq(industry).astype(np.int8)
        static_columns.append(column)
    return result, static_columns


def build_mlforecast() -> MLForecast:
    models = {
        "Ridge": make_pipeline(
            StandardScaler(),
            Ridge(alpha=10.0),
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        ),
    }
    return MLForecast(
        models=models,
        freq="MS",
        lags=[1, 2, 3, 6, 12, 13, 18, 24],
        lag_transforms={
            1: [RollingMean(window_size=3), RollingStd(window_size=6)],
            12: [RollingMean(window_size=3)],
        },
        date_features=["month", "quarter"],
        num_threads=-1,
    )


def build_mlforecast_basic() -> MLForecast:
    return MLForecast(
        models={
            "LightGBM": LGBMRegressor(
                n_estimators=300,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=20,
                random_state=42,
                verbosity=-1,
                n_jobs=-1,
            )
        },
        freq="MS",
        lags=[1, 12],
        date_features=["month"],
        num_threads=-1,
    )


def run_ml_cross_validation(data: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    augmented, static_columns = add_static_features(data)
    forecast = build_mlforecast()
    start = time.perf_counter()
    cv = forecast.cross_validation(
        df=augmented,
        n_windows=N_WINDOWS,
        h=HORIZON,
        step_size=STEP_SIZE,
        static_features=static_columns,
        refit=True,
        prediction_intervals=PredictionIntervals(n_windows=3, h=HORIZON),
        level=LEVELS,
    )
    elapsed = time.perf_counter() - start
    return cv, elapsed


def build_dl_models(max_steps: int = 150) -> list:
    loss = MQLoss(level=LEVELS)
    common = {
        "h": HORIZON,
        "loss": loss,
        "max_steps": max_steps,
        "learning_rate": 1e-3,
        "val_check_steps": 25,
        "early_stop_patience_steps": 3,
        "batch_size": 9,
        "random_seed": 42,
        "enable_progress_bar": False,
        "logger": False,
        "enable_checkpointing": False,
        "enable_model_summary": False,
        "accelerator": "cpu",
    }
    return [
        MLP(
            **common,
            input_size=12,
            num_layers=2,
            hidden_size=128,
            scaler_type="robust",
            alias="MLP",
        ),
        LSTM(
            **common,
            input_size=24,
            encoder_n_layers=1,
            encoder_hidden_size=64,
            decoder_hidden_size=64,
            decoder_layers=1,
            scaler_type="robust",
            alias="LSTM",
        ),
        NHITS(
            **common,
            input_size=24,
            n_blocks=[1, 1, 1],
            mlp_units=[[128, 128], [128, 128], [128, 128]],
            scaler_type="robust",
            alias="NHITS",
        ),
    ]


def run_dl_cross_validation(
    data: pd.DataFrame,
    max_steps: int = 150,
) -> tuple[pd.DataFrame, float]:
    start = time.perf_counter()
    last_cutoff = data["ds"].max() - pd.DateOffset(months=HORIZON)
    cutoffs = [
        last_cutoff - pd.DateOffset(months=STEP_SIZE * offset)
        for offset in reversed(range(N_WINDOWS))
    ]
    windows = []
    for cutoff in cutoffs:
        train = data[data["ds"].le(cutoff)]
        test = data[
            data["ds"].gt(cutoff)
            & data["ds"].le(cutoff + pd.DateOffset(months=HORIZON))
        ][["unique_id", "ds", "y"]]
        neural_forecast = NeuralForecast(
            models=build_dl_models(max_steps=max_steps),
            freq="MS",
        )
        neural_forecast.fit(
            df=train,
            val_size=HORIZON,
            verbose=False,
        )
        prediction = neural_forecast.predict()
        window = test.merge(
            prediction,
            on=["unique_id", "ds"],
            how="left",
            validate="one_to_one",
        )
        window.insert(2, "cutoff", cutoff)
        windows.append(window)
    cv = pd.concat(windows, ignore_index=True)
    elapsed = time.perf_counter() - start
    return cv, elapsed


def normalize_dl_columns(cv: pd.DataFrame) -> pd.DataFrame:
    result = cv.copy()
    for model in DL_MODELS:
        median_candidates = [
            model,
            f"{model}-median",
            f"{model}-ql0.5",
        ]
        point_column = next(
            column for column in median_candidates if column in result.columns
        )
        if point_column != model:
            result[model] = result[point_column]
        for level, quantile in [(80, "0.1"), (95, "0.025")]:
            lower_candidates = [
                f"{model}-lo-{level}",
                f"{model}-ql{quantile}",
            ]
            upper_quantile = "0.9" if level == 80 else "0.975"
            upper_candidates = [
                f"{model}-hi-{level}",
                f"{model}-ql{upper_quantile}",
            ]
            lower = next(
                column for column in lower_candidates if column in result.columns
            )
            upper = next(
                column for column in upper_candidates if column in result.columns
            )
            result[f"{model}-lo-{level}"] = result[lower]
            result[f"{model}-hi-{level}"] = result[upper]
    return result


def seasonal_scales(data: pd.DataFrame, cv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for unique_id, cutoff in (
        cv[["unique_id", "cutoff"]].drop_duplicates().itertuples(index=False, name=None)
    ):
        train = data[data["unique_id"].eq(unique_id) & data["ds"].le(cutoff)]["y"]
        rows.append(
            {
                "unique_id": unique_id,
                "cutoff": cutoff,
                "mase_scale": train.diff(SEASON_LENGTH).abs().dropna().mean(),
            }
        )
    return pd.DataFrame(rows)


def combine_predictions(
    data: pd.DataFrame,
    ml_cv: pd.DataFrame,
    dl_cv: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["unique_id", "ds", "cutoff"]
    ml_columns = keys + ["y"]
    for model in ML_MODELS:
        ml_columns.append(model)
        ml_columns.extend(
            [
                f"{model}-lo-80",
                f"{model}-hi-80",
                f"{model}-lo-95",
                f"{model}-hi-95",
            ]
        )
    dl_columns = keys + DL_MODELS
    for model in DL_MODELS:
        dl_columns.extend(
            [
                f"{model}-lo-80",
                f"{model}-hi-80",
                f"{model}-lo-95",
                f"{model}-hi-95",
            ]
        )
    combined = ml_cv[ml_columns].merge(
        dl_cv[dl_columns],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    combined = combined.merge(
        seasonal_scales(data, combined),
        on=["unique_id", "cutoff"],
        how="left",
        validate="many_to_one",
    )
    combined["horizon"] = (
        (combined["ds"].dt.year - combined["cutoff"].dt.year) * 12
        + combined["ds"].dt.month
        - combined["cutoff"].dt.month
    )
    return combined.sort_values(keys).reset_index(drop=True)


def metric_row(group: pd.DataFrame, model: str) -> dict:
    error = group[model] - group["y"]
    absolute = error.abs()
    denominator = group[model].abs() + group["y"].abs()
    row = {
        "mae": absolute.mean(),
        "rmse": np.sqrt(np.mean(error**2)),
        "smape": np.mean(
            np.where(
                denominator.eq(0),
                0.0,
                200.0 * absolute / denominator,
            )
        ),
        "mase": np.mean(absolute / group["mase_scale"]),
        "bias": error.mean(),
    }
    if f"{model}-lo-80" in group.columns:
        for level in LEVELS:
            lower = group[f"{model}-lo-{level}"]
            upper = group[f"{model}-hi-{level}"]
            row[f"coverage_{level}"] = np.mean(
                (group["y"] >= lower) & (group["y"] <= upper)
            )
            row[f"mean_width_{level}"] = np.mean(upper - lower)
            alpha = 1.0 - level / 100.0
            row[f"interval_score_{level}"] = np.mean(
                upper
                - lower
                + (2.0 / alpha) * (lower - group["y"]).clip(lower=0)
                + (2.0 / alpha) * (group["y"] - upper).clip(lower=0)
            )
        row["quantile_crossing_rate"] = np.mean(
            (group[f"{model}-lo-95"] > group[f"{model}-lo-80"])
            | (group[f"{model}-hi-80"] > group[f"{model}-hi-95"])
        )
    return row


def calculate_metrics(
    cv: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregate = []
    by_industry = []
    by_horizon = []
    for model in MODEL_NAMES:
        aggregate.append({"model": model, **metric_row(cv, model)})
        for unique_id, group in cv.groupby("unique_id"):
            by_industry.append(
                {
                    "unique_id": unique_id,
                    "model": model,
                    **metric_row(group, model),
                }
            )
        for horizon, group in cv.groupby("horizon"):
            by_horizon.append(
                {
                    "horizon": horizon,
                    "model": model,
                    **metric_row(group, model),
                }
            )
    return (
        pd.DataFrame(aggregate).sort_values(["mase", "smape"]),
        pd.DataFrame(by_industry).sort_values(["unique_id", "mase"]),
        pd.DataFrame(by_horizon).sort_values(["horizon", "mase"]),
    )


def validation_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp("2023-10-01")
    valid_start = cutoff + pd.offsets.MonthBegin()
    valid_end = cutoff + pd.DateOffset(months=HORIZON)
    train = data[data["ds"].le(cutoff)].copy()
    valid = data[data["ds"].ge(valid_start) & data["ds"].le(valid_end)].copy()
    return train, valid


def evaluate_validation(
    predictions: pd.DataFrame,
    valid: pd.DataFrame,
    model: str,
) -> float:
    merged = valid.merge(
        predictions[["unique_id", "ds", model]],
        on=["unique_id", "ds"],
        how="left",
        validate="one_to_one",
    )
    rows = []
    for unique_id, group in merged.groupby("unique_id"):
        history = valid[
            valid["unique_id"].eq(unique_id) & valid["ds"].lt(group["ds"].min())
        ]["y"]
        if history.empty:
            history = load_data()[
                load_data()["unique_id"].eq(unique_id)
                & load_data()["ds"].lt(group["ds"].min())
            ]["y"]
        scale = history.diff(SEASON_LENGTH).abs().dropna().mean()
        rows.append((group[model] - group["y"]).abs() / scale)
    return float(pd.concat(rows).mean())


def configuration_study(data: pd.DataFrame) -> pd.DataFrame:
    augmented, static_columns = add_static_features(data)
    train, valid = validation_split(augmented)
    rows = []

    ml_configs = {
        "LightGBM_basic_lags": build_mlforecast_basic(),
        "LightGBM_full_features": build_mlforecast(),
    }
    for name, forecast in ml_configs.items():
        if name == "LightGBM_basic_lags":
            static = []
            train_config = train[["unique_id", "ds", "y"]]
        else:
            static = static_columns
            train_config = train
        forecast.fit(train_config, static_features=static)
        prediction = forecast.predict(h=HORIZON)
        rows.append(
            {
                "family": "ML",
                "configuration": name,
                "validation_mase": evaluate_validation(
                    prediction.rename(columns={"LightGBM": name}),
                    valid,
                    name,
                ),
            }
        )

    train_nf = train[["unique_id", "ds", "y"]]
    valid_nf = valid[["unique_id", "ds", "y"]]
    for input_size in [12, 24]:
        model_name = f"MLP_input_{input_size}"
        model = MLP(
            h=HORIZON,
            input_size=input_size,
            loss=MQLoss(level=LEVELS),
            max_steps=75,
            learning_rate=1e-3,
            val_check_steps=25,
            early_stop_patience_steps=3,
            batch_size=9,
            random_seed=42,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
            accelerator="cpu",
            scaler_type="robust",
            alias=model_name,
        )
        neural_forecast = NeuralForecast(models=[model], freq="MS")
        neural_forecast.fit(train_nf, val_size=HORIZON, verbose=False)
        prediction = neural_forecast.predict()
        point_column = next(
            column
            for column in [
                model_name,
                f"{model_name}-median",
                f"{model_name}-ql0.5",
            ]
            if column in prediction.columns
        )
        prediction[model_name] = prediction[point_column]
        rows.append(
            {
                "family": "DL",
                "configuration": model_name,
                "validation_mase": evaluate_validation(
                    prediction,
                    valid_nf,
                    model_name,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "validation_mase"])


def statistical_benchmarks() -> pd.DataFrame:
    task2 = pd.read_csv(TASK2_CV_PATH, parse_dates=["ds", "cutoff"])
    rows = []
    for model in ["Theta", "SeasonalNaive"]:
        error = task2[model] - task2["y"]
        absolute = error.abs()
        denominator = task2[model].abs() + task2["y"].abs()
        rows.append(
            {
                "model": model,
                "mae": absolute.mean(),
                "rmse": np.sqrt(np.mean(error**2)),
                "smape": np.mean(
                    np.where(
                        denominator.eq(0),
                        0.0,
                        200.0 * absolute / denominator,
                    )
                ),
                "mase": np.mean(absolute / task2["mase_scale"]),
                "bias": error.mean(),
                "family": "Statistical",
            }
        )
    return pd.DataFrame(rows)


def comparison_tests(cv: pd.DataFrame) -> pd.DataFrame:
    task2 = pd.read_csv(TASK2_CV_PATH, parse_dates=["ds", "cutoff"])
    comparison = cv.merge(
        task2[["unique_id", "ds", "cutoff", "Theta"]],
        on=["unique_id", "ds", "cutoff"],
        how="left",
        validate="one_to_one",
    )
    models = ["Theta"] + MODEL_NAMES
    block_errors = {}
    for model in models:
        block_errors[model] = (
            comparison.assign(
                scaled_error=(comparison[model] - comparison["y"]).abs()
                / comparison["mase_scale"]
            )
            .groupby(["unique_id", "cutoff"])["scaled_error"]
            .mean()
            .to_numpy()
        )
    friedman = friedmanchisquare(*[block_errors[model] for model in models])
    best_model = min(
        MODEL_NAMES,
        key=lambda model: np.mean(block_errors[model]),
    )
    pair = wilcoxon(
        block_errors[best_model],
        block_errors["Theta"],
        alternative="two-sided",
        zero_method="zsplit",
    )
    return pd.DataFrame(
        [
            {
                "test": "Friedman_block_MASE",
                "model_a": "all",
                "model_b": "all",
                "statistic": friedman.statistic,
                "pvalue": friedman.pvalue,
            },
            {
                "test": "Wilcoxon_block_MASE",
                "model_a": best_model,
                "model_b": "Theta",
                "statistic": pair.statistic,
                "pvalue": pair.pvalue,
            },
        ]
    )


def save_figures(
    data: pd.DataFrame,
    cv: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    comparison = pd.concat(
        [
            metrics.assign(
                family=np.where(
                    metrics["model"].isin(ML_MODELS),
                    "ML",
                    "DL",
                )
            ),
            statistical_benchmarks(),
        ],
        ignore_index=True,
    )
    fig, axis = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=comparison.sort_values("mase"),
        x="mase",
        y="model",
        hue="family",
        dodge=False,
        ax=axis,
    )
    axis.axvline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_title("ML/DL models and statistical benchmarks")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=160)
    plt.close(fig)

    best_model = metrics.iloc[0]["model"]
    last_cutoff = cv["cutoff"].max()
    last_window = cv[cv["cutoff"].eq(last_cutoff)]
    industries = sorted(data["unique_id"].unique())
    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    for unique_id, axis in zip(industries, axes.flat):
        history = data[data["unique_id"].eq(unique_id)].tail(36)
        forecast = last_window[last_window["unique_id"].eq(unique_id)]
        axis.plot(history["ds"], history["y"], color="black", label="Actual")
        axis.plot(
            forecast["ds"],
            forecast[best_model],
            color="#C44E52",
            label=best_model,
        )
        if best_model in DL_MODELS:
            axis.fill_between(
                forecast["ds"],
                forecast[f"{best_model}-lo-95"],
                forecast[f"{best_model}-hi-95"],
                color="#C44E52",
                alpha=0.15,
            )
        axis.set_title(unique_id, fontsize=10)
        axis.tick_params(axis="x", rotation=30)
    fig.suptitle(f"Last backtest window: {best_model}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURES_DIR / "best_model_last_window.png", dpi=160)
    plt.close(fig)

    horizon = metrics_by_horizon(cv)
    fig, axis = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=horizon,
        x="horizon",
        y="mase",
        hue="model",
        marker="o",
        ax=axis,
    )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_title("MASE by forecast horizon")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mase_by_horizon.png", dpi=160)
    plt.close(fig)


def metrics_by_horizon(cv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_NAMES:
        for horizon, group in cv.groupby("horizon"):
            rows.append(
                {
                    "model": model,
                    "horizon": horizon,
                    "mase": metric_row(group, model)["mase"],
                }
            )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    rows = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        values = [
            f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value)
            for value in row
        ]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def save_report(
    metrics: pd.DataFrame,
    by_industry: pd.DataFrame,
    runtime: pd.DataFrame,
    tests: pd.DataFrame,
    configs: pd.DataFrame,
) -> None:
    best = metrics.iloc[0]
    theta = statistical_benchmarks().query("model == 'Theta'").iloc[0]
    improvement = 100.0 * (theta["mase"] - best["mase"]) / theta["mase"]
    comparison_text = (
        f"лучше Theta на {improvement:.1f}%"
        if improvement >= 0
        else f"хуже Theta на {abs(improvement):.1f}%"
    )
    winners = (
        by_industry.sort_values(["unique_id", "mase"])
        .groupby("unique_id", as_index=False)
        .first()[["unique_id", "model"]]
    )
    winner_summary = "; ".join(
        f"{row.unique_id}: {row.model}" for row in winners.itertuples(index=False)
    )
    friedman_pvalue = tests.loc[tests["test"].eq("Friedman_block_MASE"), "pvalue"].iloc[
        0
    ]
    theta_test = tests.loc[tests["test"].eq("Wilcoxon_block_MASE"), "pvalue"]
    theta_pvalue = theta_test.iloc[0]
    tested_model = tests.loc[tests["test"].eq("Wilcoxon_block_MASE"), "model_a"].iloc[0]
    report = f"""# Задача №3: ML и DL для временных рядов

## Дизайн

- Фреймворки: `mlforecast` и `neuralforecast`.
- Горизонт: {HORIZON} месяцев.
- Backtesting: {N_WINDOWS} неперекрывающихся окна с шагом {STEP_SIZE} месяцев.
- Все модели глобальные: обучаются одновременно на девяти отраслевых рядах.
- Основная метрика: сезонный MASE.

## Feature engineering

ML-модели используют лаги 1, 2, 3, 6, 12, 13, 18 и 24, скользящие средние
и стандартное отклонение, месяц, квартал и one-hot признаки отрасли. Все
лаги рассчитываются внутри каждого ряда. Текущие значения hires, quits и
layoffs не используются, поэтому утечки целевой переменной нет.
Интервалы ML-моделей построены методом conformal prediction внутри
`mlforecast`.

## Предтестовая настройка

Перед основным backtesting выполнена компактная проверка конфигураций на
валидационном окне ноябрь 2023 — апрель 2024, которое предшествует всем
четырём тестовым окнам.

{markdown_table(configs, ["family", "configuration", "validation_mase"])}

Разница LightGBM-конфигураций на одном окне мала. В основном сравнении
сохранен полный единый набор признаков для всех трех ML-моделей: он включает
идентификаторы отраслей и позволяет контролируемо сравнить алгоритмы на одной
матрице признаков. Результат ablation показывает, что дополнительные признаки
сами по себе не гарантируют улучшения.

Это ограниченный tuning study с одним seed и одним временным окном, а не
исчерпывающий поиск гиперпараметров. Остальные размеры моделей выбраны как
небольшие учебные конфигурации с одинаковым бюджетом 150 optimizer steps,
чтобы сравнение было вычислительно воспроизводимым на CPU.

## Модели

- ML: Ridge, RandomForest и LightGBM.
- DL: MLP, LSTM и NHITS.
- MLP использует вход 12 месяцев, выбранный на предтестовом окне; LSTM и
  NHITS используют 24 месяца для более длинного контекста. Все DL-модели
  оптимизируют quantile loss для прогнозов 80% и 95%.

## Результаты

{markdown_table(metrics, ["model", "mae", "rmse", "smape", "mase", "bias"])}

## Вероятностная надежность

{markdown_table(metrics, ["model", "coverage_80", "coverage_95", "mean_width_95", "interval_score_95", "quantile_crossing_rate"])}

Идеальные значения coverage равны 0.80 и 0.95. Интервалы ML построены
conformal-методом, интервалы DL — квантильной функцией потерь. На текущем
backtesting ML-интервалы и интервалы NHITS/MLP недокрывают 95% уровень,
а LSTM слегка перекрывает его. У NHITS также наблюдается небольшая доля
пересечений квантилей.

Лучшая data-driven модель: **{best["model"]}**, MASE = **{best["mase"]:.3f}**.
Она **{comparison_text}** из задачи №2.

Победители по отдельным отраслям: {winner_summary}.

Friedman-тест по MASE в 36 блоках «отрасль × окно» для шести data-driven
моделей и Theta: `p = {friedman_pvalue:.3g}`. Различие {tested_model} и Theta
по двустороннему Wilcoxon-тесту: `p = {theta_pvalue:.3f}`. Лучшая модель
выбрана на тех же данных, поэтому тест является исследовательским и отсутствие
значимости не доказывает эквивалентность моделей.

## Производительность

{markdown_table(runtime, ["family", "seconds"])}

ML и DL переобучаются с нуля в каждом окне, поэтому используют одинаковый
объем доступной на соответствующую дату истории. Время отражает полный
backtesting, но зависит от реализации библиотек и не является чистым
benchmark скорости.

## Вывод

Сравнены три ML- и три DL-метода на едином временном горизонте. Основной выбор
сделан по MASE, дополнительно рассмотрены sMAPE, bias, результаты по отраслям
и горизонту, а для DL — покрытие квантильных интервалов. Data-driven модель
не заменяет автоматически статистический benchmark: итоговый выбор зависит
от фактического улучшения относительно Theta и устойчивости по отраслям.

![Сравнение моделей](../../figures/task3/model_comparison.png)

![Последнее окно](../../figures/task3/best_model_last_window.png)

![MASE по горизонту](../../figures/task3/mase_by_horizon.png)
"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "task3_report.md").write_text(report, encoding="utf-8")


def run_task3(dl_max_steps: int = 150) -> dict[str, pd.DataFrame]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    ml_cv, ml_seconds = run_ml_cross_validation(data)
    dl_cv, dl_seconds = run_dl_cross_validation(data, max_steps=dl_max_steps)
    dl_cv = normalize_dl_columns(dl_cv)
    cv = combine_predictions(data, ml_cv, dl_cv)
    metrics, by_industry, by_horizon = calculate_metrics(cv)
    runtime = pd.DataFrame(
        {
            "family": ["ML", "DL"],
            "seconds": [ml_seconds, dl_seconds],
        }
    )
    tests = comparison_tests(cv)
    configs = configuration_study(data)

    cv.to_csv(REPORTS_DIR / "cross_validation_predictions.csv", index=False)
    metrics.to_csv(REPORTS_DIR / "model_metrics.csv", index=False)
    by_industry.to_csv(
        REPORTS_DIR / "model_metrics_by_industry.csv",
        index=False,
    )
    by_horizon.to_csv(
        REPORTS_DIR / "model_metrics_by_horizon.csv",
        index=False,
    )
    runtime.to_csv(REPORTS_DIR / "runtime.csv", index=False)
    tests.to_csv(REPORTS_DIR / "model_comparison_tests.csv", index=False)
    configs.to_csv(REPORTS_DIR / "configuration_study.csv", index=False)
    save_figures(data, cv, metrics)
    save_report(metrics, by_industry, runtime, tests, configs)
    return {
        "cv": cv,
        "metrics": metrics,
        "industry_metrics": by_industry,
        "horizon_metrics": by_horizon,
        "runtime": runtime,
        "tests": tests,
        "configuration_study": configs,
    }


if __name__ == "__main__":
    results = run_task3()
    print(results["metrics"].round(3).to_string(index=False))
