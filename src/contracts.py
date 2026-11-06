"""Shared monthly data contracts. Missing market prices are never imputed."""
from __future__ import annotations

import argparse
import re
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
START_DATE = date(2015, 1, 1)
RIYADH = timezone(timedelta(hours=3))
# Broad sanity limits in the stated units, not price forecasts.
MATERIALS = {
    "brent_oil": dict(material_id=1, material_name="Brent Crude Oil", category="Global Benchmark",
                      unit="barrel", currency="USD", source="Yahoo Finance / BZ=F",
                      ticker="BZ=F", is_simulated=False, price_basis="unadjusted monthly close",
                      minimum=1.0, maximum=500.0),
    "steel_index": dict(material_id=2, material_name="Global Steel Index", category="Global Benchmark",
                        unit="ETF share", currency="USD", source="Yahoo Finance / SLX (ETF proxy)",
                        ticker="SLX", is_simulated=False, price_basis="unadjusted monthly close",
                        minimum=1.0, maximum=1000.0),
    "cement": dict(material_id=3, material_name="Portland Cement", category="Local GASTAT",
                   unit="50 kg bag", currency="SAR", source="GASTAT_SIMULATED",
                   ticker="not_applicable", is_simulated=True, price_basis="simulated monthly price",
                   minimum=1.0, maximum=200.0),
    "rebar": dict(material_id=4, material_name="Reinforcing Rebar", category="Local GASTAT",
                  unit="metric tonne", currency="SAR", source="GASTAT_SIMULATED",
                  ticker="not_applicable", is_simulated=True, price_basis="simulated monthly price",
                  minimum=100.0, maximum=20000.0),
    "concrete": dict(material_id=5, material_name="Ready-Mix Concrete", category="Local GASTAT",
                     unit="m³", currency="SAR", source="GASTAT_SIMULATED",
                     ticker="not_applicable", is_simulated=True, price_basis="simulated monthly price",
                     minimum=20.0, maximum=3000.0),
}
DATASETS = {"brent_oil": ("brent_oil",), "steel_index": ("steel_index",),
            "gastat": ("cement", "rebar", "concrete")}
METADATA_COLUMNS = ("material_name", "category", "unit", "currency", "source",
                    "ticker", "is_simulated", "price_basis")
SILVER_COLUMNS = ("date", "material_code", *METADATA_COLUMNS, "price",
                  "is_partial_month", "ingestion_timestamp")


def resolve_as_of(value: str | date | None = None) -> date:
    today = datetime.now(RIYADH).date()
    result = date.fromisoformat(value) if isinstance(value, str) else value or today
    if not START_DATE <= result <= today:
        raise ValueError(f"as-of must be between {START_DATE} and {today}.")
    return result


def parse_as_of() -> date:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", help="Inclusive cutoff YYYY-MM-DD; Riyadh today by default")
    return resolve_as_of(parser.parse_args().as_of)


def expected_months(as_of: date) -> pd.DatetimeIndex:
    return pd.date_range(START_DATE, as_of.replace(day=1), freq="MS").as_unit("ns")


def standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")
                      for c in result.columns]
    if result.columns.duplicated().any() or any(not re.fullmatch(r"[a-z][a-z0-9_]*", c) for c in result.columns):
        raise ValueError("Column normalization produced ambiguous or invalid names.")
    return result


def assert_complete_values(frame: pd.DataFrame) -> None:
    if frame.empty or frame.isna().any().any():
        raise ValueError("Dataset is empty or contains NULL/NaN/NaT values.")
    numbers = frame.select_dtypes(include="number")
    if not np.isfinite(numbers.to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains infinite numeric values.")
    for column in frame.select_dtypes(include=["object", "string"]):
        if frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Empty strings in {column}.")


def attach_metadata(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    result = frame.copy()
    result["material_code"] = code
    for column in METADATA_COLUMNS:
        result[column] = MATERIALS[code][column]
    return result


def validate_monthly(frame: pd.DataFrame, dataset: str, as_of: date, *, silver: bool = True) -> None:
    """Reject unknown metadata, duplicates, missing months and invalid prices."""
    assert_complete_values(frame)
    if silver and tuple(frame.columns) != SILVER_COLUMNS:
        raise ValueError(f"{dataset}: Silver columns differ from the common contract.")
    if any(not re.fullmatch(r"[a-z][a-z0-9_]*", c) for c in frame.columns):
        raise ValueError(f"{dataset}: column names must be lowercase snake_case.")
    codes = DATASETS[dataset]
    if set(frame["material_code"]) != set(codes):
        raise ValueError(f"{dataset}: unexpected material set.")
    dates = pd.to_datetime(frame["date"], errors="raise")
    if dates.dt.tz is not None or not dates.dt.is_month_start.all() or not dates.eq(dates.dt.normalize()).all():
        raise ValueError(f"{dataset}: dates must be timezone-naive month starts at midnight.")
    if frame.duplicated(["date", "material_code"]).any():
        raise ValueError(f"{dataset}: duplicate material/month keys.")
    months = expected_months(as_of)
    current_is_partial = as_of < (pd.Timestamp(as_of) + pd.offsets.MonthEnd(0)).date()
    for code in codes:
        group = frame.loc[frame["material_code"].eq(code)].sort_values("date")
        if not pd.DatetimeIndex(group["date"]).as_unit("ns").equals(months.as_unit("ns")):
            actual = pd.DatetimeIndex(group["date"])
            missing = months.difference(actual).strftime("%Y-%m-%d").tolist()
            extra = actual.difference(months).strftime("%Y-%m-%d").tolist()
            raise ValueError(f"{dataset}/{code}: incomplete monthly coverage; missing={missing}, extra={extra}.")
        metadata = MATERIALS[code]
        for column in METADATA_COLUMNS:
            if not group[column].eq(metadata[column]).all():
                raise ValueError(f"{code}: inconsistent {column}.")
        prices = pd.to_numeric(group["price"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(prices).all() or not ((prices >= metadata["minimum"]) & (prices <= metadata["maximum"])).all():
            raise ValueError(f"{code}: prices outside sanity bounds [{metadata['minimum']}, {metadata['maximum']}].")
        if silver and not np.allclose(prices * 100, np.round(prices * 100), rtol=0, atol=1e-8):
            raise ValueError(f"{code}: price precision must be two decimal places.")
        expected_partial = group["date"].eq(months[-1]) & current_is_partial
        if not group["is_partial_month"].eq(expected_partial).all():
            raise ValueError(f"{code}: incorrect partial-month flags.")
    timestamps = pd.to_datetime(frame["ingestion_timestamp"], utc=True, errors="raise")
    if timestamps.isna().any() or (timestamps > pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=5)).any():
        raise ValueError("Invalid ingestion timestamp.")


def save_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Stage, reread, and atomically replace one Parquet file."""
    frame = frame.copy()
    for column in frame.select_dtypes(include=["datetime", "datetimetz"]):
        frame[column] = frame[column].dt.as_unit("ns")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression="snappy", index=False)
        pd.testing.assert_frame_equal(frame.reset_index(drop=True), pd.read_parquet(temporary))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
