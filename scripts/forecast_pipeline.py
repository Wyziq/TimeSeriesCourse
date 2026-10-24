import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, Theta

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "processed" / "total_separations_nf.csv"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "task4"
FREQUENCY = "MS"
SEASON_LENGTH = 12
DEFAULT_HORIZON = 6
LEVELS = [80, 95]
REQUIRED_COLUMNS = {"unique_id", "ds", "y"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_panel(path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    data = pd.read_csv(path)
    if "ds" in data.columns:
        data["ds"] = pd.to_datetime(data["ds"], errors="raise")
    return data


def validate_panel(data: pd.DataFrame) -> dict:
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required columns contain missing values")
    if data.duplicated(["unique_id", "ds"]).any():
        raise ValueError("Duplicate (unique_id, ds) observations found")
    if not pd.api.types.is_datetime64_any_dtype(data["ds"]):
        raise ValueError("Column ds must have a datetime dtype")
    if not pd.api.types.is_numeric_dtype(data["y"]):
        raise ValueError("Column y must be numeric")
    if not np.isfinite(data["y"]).all():
        raise ValueError("Column y contains non-finite values")
    if (data["y"] < 0).any():
        raise ValueError("Column y contains negative values")

    lengths = data.groupby("unique_id").size()
    if (lengths < 2 * SEASON_LENGTH).any():
        short = lengths[lengths < 2 * SEASON_LENGTH].index.tolist()
        raise ValueError(f"Series shorter than 24 months: {short}")

    for unique_id, group in data.groupby("unique_id"):
        actual = pd.DatetimeIndex(group["ds"].sort_values())
        expected = pd.date_range(actual.min(), actual.max(), freq=FREQUENCY)
        if not actual.equals(expected):
            raise ValueError(f"Series {unique_id!r} is not a continuous monthly series")

    return {
        "n_series": int(data["unique_id"].nunique()),
        "n_rows": len(data),
        "min_date": data["ds"].min().date().isoformat(),
        "max_date": data["ds"].max().date().isoformat(),
        "min_series_length": int(lengths.min()),
        "max_series_length": int(lengths.max()),
    }


def forecast_panel(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if horizon <= 0:
        raise ValueError("Horizon must be positive")

    models = [
        Theta(season_length=SEASON_LENGTH),
        SeasonalNaive(season_length=SEASON_LENGTH),
    ]
    engine = StatsForecast(models=models, freq=FREQUENCY, n_jobs=1)
    raw = engine.forecast(
        df=data.sort_values(["unique_id", "ds"]),
        h=horizon,
        level=LEVELS,
    )
    result = raw.rename(
        columns={
            "Theta": "y_hat",
            "Theta-lo-80": "lo_80",
            "Theta-hi-80": "hi_80",
            "Theta-lo-95": "lo_95",
            "Theta-hi-95": "hi_95",
            "SeasonalNaive": "baseline_y_hat",
        }
    )
    selected = [
        "unique_id",
        "ds",
        "y_hat",
        "lo_80",
        "hi_80",
        "lo_95",
        "hi_95",
        "baseline_y_hat",
    ]
    result = result[selected].copy()
    result["model"] = "Theta"
    result["baseline_model"] = "SeasonalNaive"
    return result


def validate_forecast(
    forecast: pd.DataFrame,
    history: pd.DataFrame,
    horizon: int,
) -> None:
    expected_rows = history["unique_id"].nunique() * horizon
    if len(forecast) != expected_rows:
        raise ValueError(f"Expected {expected_rows} forecasts, got {len(forecast)}")
    if forecast.duplicated(["unique_id", "ds"]).any():
        raise ValueError("Forecast contains duplicate keys")

    numeric = [
        "y_hat",
        "lo_80",
        "hi_80",
        "lo_95",
        "hi_95",
        "baseline_y_hat",
    ]
    if not np.isfinite(forecast[numeric].to_numpy()).all():
        raise ValueError("Forecast contains non-finite values")
    if (forecast[["y_hat", "baseline_y_hat"]] < 0).any().any():
        raise ValueError("Point forecast contains negative values")
    if not (
        (forecast["lo_95"] <= forecast["lo_80"])
        & (forecast["lo_80"] <= forecast["y_hat"])
        & (forecast["y_hat"] <= forecast["hi_80"])
        & (forecast["hi_80"] <= forecast["hi_95"])
    ).all():
        raise ValueError("Prediction intervals are not correctly ordered")

    last_dates = history.groupby("unique_id")["ds"].max()
    for unique_id, group in forecast.groupby("unique_id"):
        expected_dates = pd.date_range(
            last_dates.loc[unique_id] + pd.offsets.MonthBegin(),
            periods=horizon,
            freq=FREQUENCY,
        )
        if not pd.DatetimeIndex(group["ds"].sort_values()).equals(expected_dates):
            raise ValueError(f"Invalid forecast dates for {unique_id!r}")


def run_pipeline(
    data_path: Path | str = DEFAULT_DATA_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    horizon: int = DEFAULT_HORIZON,
) -> dict:
    started = time.perf_counter()
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_started = time.perf_counter()
    data = load_panel(data_path)
    validation = validate_panel(data)
    load_validation_seconds = time.perf_counter() - load_started

    forecast_started = time.perf_counter()
    forecast = forecast_panel(data, horizon)
    validate_forecast(forecast, data, horizon)
    forecast_seconds = time.perf_counter() - forecast_started

    forecast_path = output_dir / "final_pipeline_forecast.csv"
    metadata_path = output_dir / "pipeline_metadata.json"
    forecast.to_csv(forecast_path, index=False)

    metadata = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "data_path": str(data_path),
        "data_sha256": file_sha256(data_path),
        "frequency": FREQUENCY,
        "horizon": horizon,
        "selected_model": "Theta",
        "baseline_model": "SeasonalNaive",
        "prediction_intervals": LEVELS,
        "validation": validation,
        "timing_seconds": {
            "load_and_validation": load_validation_seconds,
            "forecast_and_validation": forecast_seconds,
            "total": time.perf_counter() - started,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "forecast": forecast,
        "metadata": metadata,
        "forecast_path": forecast_path,
        "metadata_path": metadata_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the final JOLTS forecasting pipeline."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_pipeline(args.data, args.output_dir, args.horizon)
    print(result["forecast"].round(2).to_string(index=False))
    print(json.dumps(result["metadata"]["timing_seconds"], indent=2))
