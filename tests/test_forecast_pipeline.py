import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.forecast_pipeline import (
    DEFAULT_DATA_PATH,
    forecast_panel,
    load_panel,
    run_pipeline,
    validate_forecast,
    validate_panel,
)


class ForecastPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_panel(DEFAULT_DATA_PATH)

    def test_valid_panel(self) -> None:
        result = validate_panel(self.data)
        self.assertEqual(result["n_series"], 9)
        self.assertEqual(result["min_series_length"], 305)

    def test_duplicate_is_rejected(self) -> None:
        invalid = pd.concat(
            [self.data, self.data.iloc[[0]]], ignore_index=True
        )
        with self.assertRaises(ValueError):
            validate_panel(invalid)

    def test_missing_required_column_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_panel(self.data.drop(columns=["y"]))

    def test_missing_value_is_rejected(self) -> None:
        invalid = self.data.copy()
        invalid.loc[0, "y"] = np.nan
        with self.assertRaises(ValueError):
            validate_panel(invalid)

    def test_negative_value_is_rejected(self) -> None:
        invalid = self.data.copy()
        invalid.loc[0, "y"] = -1
        with self.assertRaises(ValueError):
            validate_panel(invalid)

    def test_nonfinite_value_is_rejected(self) -> None:
        invalid = self.data.copy()
        invalid["y"] = invalid["y"].astype(float)
        invalid.loc[0, "y"] = np.inf
        with self.assertRaises(ValueError):
            validate_panel(invalid)

    def test_non_datetime_ds_is_rejected(self) -> None:
        invalid = self.data.copy()
        invalid["ds"] = invalid["ds"].dt.strftime("%Y-%m-%d")
        with self.assertRaises(ValueError):
            validate_panel(invalid)

    def test_short_series_is_rejected(self) -> None:
        invalid = self.data.groupby("unique_id").head(23).copy()
        with self.assertRaises(ValueError):
            validate_panel(invalid)

    def test_monthly_gap_is_rejected(self) -> None:
        invalid = self.data.drop(self.data.index[1]).reset_index(drop=True)
        with self.assertRaises(ValueError):
            validate_panel(invalid)

    def test_nonpositive_horizon_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            forecast_panel(self.data, 0)

    def test_forecast_contract(self) -> None:
        horizon = 2
        forecast = forecast_panel(self.data, horizon)
        validate_forecast(forecast, self.data, horizon)
        self.assertEqual(len(forecast), 18)

    def test_bad_forecast_intervals_are_rejected(self) -> None:
        forecast = forecast_panel(self.data, 1)
        forecast.loc[0, "lo_80"] = forecast.loc[0, "hi_80"] + 1
        with self.assertRaises(ValueError):
            validate_forecast(forecast, self.data, 1)

    def test_pipeline_writes_forecast_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_pipeline(DEFAULT_DATA_PATH, Path(tmp), horizon=1)
            self.assertTrue(result["forecast_path"].exists())
            self.assertTrue(result["metadata_path"].exists())
            self.assertEqual(len(result["metadata"]["data_sha256"]), 64)
            self.assertEqual(len(result["forecast"]), 9)


if __name__ == "__main__":
    unittest.main()
