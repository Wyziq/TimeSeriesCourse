import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from functools import reduce
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MANIFEST_PATH = PROCESSED_DIR / "jolts_manifest.json"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

INDUSTRIES = {
    "Construction": "2300",
    "Manufacturing": "3000",
    "Trade, Transportation, and Utilities": "4000",
    "Information": "5100",
    "Financial Activities": "510099",
    "Professional and Business Services": "540099",
    "Education and Health Services": "6000",
    "Leisure and Hospitality": "7000",
    "Government": "9000",
}

MEASURES = {
    "job_openings": "JOL",
    "hires": "HIL",
    "total_separations": "TSL",
    "quits": "QUL",
    "layoffs_discharges": "LDL",
}


def series_id(industry_code: str, measure: str) -> str:
    if industry_code == "540099" and measure == "quits":
        # FRED uses this historical identifier for Professional and Business
        # Services quits; the other industries follow the QUL suffix pattern.
        return "JTU540099UL"
    return f"JTU{industry_code}{MEASURES[measure]}"


def download_file(fred_id: str, destination: Path, refresh: bool) -> None:
    if destination.exists() and not refresh:
        return

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(
        FRED_URL.format(series_id=fred_id),
        headers={"User-Agent": "Mozilla/5.0 (JOLTS course project)"},
    )

    last_error = None
    for attempt in range(1, 4):
        try:
            print(f"Downloading {fred_id} (attempt {attempt})...")
            with urlopen(request, timeout=60) as response:
                content = response.read()
            if not content.startswith(b"observation_date,"):
                raise ValueError(f"Unexpected CSV header for {fred_id}")
            temporary.write_bytes(content)
            temporary.replace(destination)
            return
        except (OSError, URLError, ValueError) as error:
            last_error = error
            if temporary.exists():
                temporary.unlink()
            time.sleep(2**attempt)

    raise RuntimeError(f"Failed to download {fred_id}") from last_error


def download_series(refresh: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for industry_code in INDUSTRIES.values():
        for measure in MEASURES:
            fred_id = series_id(industry_code, measure)
            download_file(fred_id, RAW_DIR / f"{fred_id}.csv", refresh)


def read_series(fred_id: str, column_name: str) -> pd.DataFrame:
    path = RAW_DIR / f"{fred_id}.csv"
    data = pd.read_csv(path).rename(
        columns={"observation_date": "ds", fred_id: column_name}
    )
    expected_columns = {"ds", column_name}
    if set(data.columns) != expected_columns:
        raise ValueError(f"{path} has unexpected columns: {list(data.columns)}")
    if data["ds"].duplicated().any():
        raise ValueError(f"{path} contains duplicated dates")

    data["ds"] = pd.to_datetime(data["ds"])
    data[column_name] = pd.to_numeric(data[column_name], errors="raise")
    if (data[column_name] < 0).any():
        raise ValueError(f"{path} contains negative values")
    return data


def build_industry_dataset(industry: str, industry_code: str) -> pd.DataFrame:
    frames = [
        read_series(series_id(industry_code, measure), measure)
        for measure in MEASURES
    ]
    data = reduce(
        lambda left, right: left.merge(right, on="ds", how="outer"),
        frames,
    )
    data = data.sort_values("ds").reset_index(drop=True)
    data.insert(0, "industry", industry)
    return data


def build_dataset() -> pd.DataFrame:
    data = pd.concat(
        [
            build_industry_dataset(industry, industry_code)
            for industry, industry_code in INDUSTRIES.items()
        ],
        ignore_index=True,
    )
    return data.sort_values(["industry", "ds"]).reset_index(drop=True)


def validate_panel(data: pd.DataFrame) -> None:
    expected_columns = {"industry", "ds", *MEASURES.keys()}
    if set(data.columns) != expected_columns:
        raise ValueError(f"Unexpected panel columns: {list(data.columns)}")
    if data.duplicated(["industry", "ds"]).any():
        raise ValueError("Panel contains duplicated industry/month rows")
    if data[list(MEASURES)].isna().any().any():
        raise ValueError("Panel contains missing values")

    expected_dates = pd.date_range(data["ds"].min(), data["ds"].max(), freq="MS")
    for industry, group in data.groupby("industry"):
        dates = pd.DatetimeIndex(group["ds"].sort_values())
        if not dates.equals(expected_dates):
            raise ValueError(f"{industry} has non-continuous monthly dates")


def save_outputs(data: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    data = data.copy()
    data["ds"] = pd.to_datetime(data["ds"]).dt.strftime("%Y-%m-%d")

    data.to_csv(
        PROCESSED_DIR / "jolts_industry_panel.csv",
        index=False,
    )

    target = data[["industry", "ds", "total_separations"]].rename(
        columns={"industry": "unique_id", "total_separations": "y"}
    )
    target.to_csv(PROCESSED_DIR / "total_separations_nf.csv", index=False)

    quality = (
        data.groupby("industry", as_index=False)
        .agg(
            observations=("ds", "size"),
            start=("ds", "min"),
            end=("ds", "max"),
            **{
                f"missing_{measure}": (
                    measure,
                    lambda values: int(values.isna().sum()),
                )
                for measure in MEASURES
            },
        )
        .sort_values("industry")
    )
    quality.to_csv(PROCESSED_DIR / "data_quality.csv", index=False)

    print(f"Rows: {len(data):,}")
    print(f"Industries: {data['industry'].nunique()}")
    print(f"Date range: {data['ds'].min()} - {data['ds'].max()}")
    print("Missing values:")
    print(data.isna().sum().to_string())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_manifest() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    series = []
    for industry, industry_code in INDUSTRIES.items():
        for measure in MEASURES:
            fred_id = series_id(industry_code, measure)
            path = RAW_DIR / f"{fred_id}.csv"
            series.append(
                {
                    "industry": industry,
                    "measure": measure,
                    "fred_id": fred_id,
                    "file": str(path.relative_to(ROOT)),
                    "sha256": file_sha256(path),
                }
            )

    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "source": "https://fred.stlouisfed.org/",
                "series": series,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare JOLTS/FRED panel data.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download raw FRED CSV files again before rebuilding processed data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_series(refresh=args.refresh)
    data = build_dataset()
    validate_panel(data)
    save_outputs(data)
    save_manifest()


if __name__ == "__main__":
    main()
