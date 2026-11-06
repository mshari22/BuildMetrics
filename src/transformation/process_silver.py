"""Publish complete monthly Silver series; reject bad data rather than invent prices."""
from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.contracts import (DATASETS, MATERIALS, METADATA_COLUMNS, PROJECT_ROOT, SILVER_COLUMNS,
                           assert_complete_values, expected_months, parse_as_of, save_parquet,
                           standardize_columns, validate_monthly)

LOGGER = logging.getLogger(__name__)
BRONZE_DIR = PROJECT_ROOT / "data/bronze"
SILVER_DIR = PROJECT_ROOT / "data/silver"


def transform_frame(frame: pd.DataFrame, name: str, as_of: date) -> pd.DataFrame:
    result = standardize_columns(frame)
    assert_complete_values(result)
    required = {"date", "material_code", *METADATA_COLUMNS, "ingestion_timestamp"}
    price_column = "price" if name == "gastat" else "close"
    if missing := (required | {price_column}) - set(result.columns):
        raise ValueError(f"{name}: missing columns {sorted(missing)}.")
    # Preserve exchange calendar dates; UTC conversion can change the calendar month.
    parsed = pd.to_datetime(result["date"], errors="raise")
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)
    if ((parsed < pd.Timestamp("2015-01-01")) | (parsed > pd.Timestamp(as_of))).any():
        raise ValueError(f"{name}: observations outside the requested history.")
    result["source_date"] = parsed
    result["date"] = parsed.dt.to_period("M").dt.to_timestamp()
    result["price"] = pd.to_numeric(result[price_column], errors="raise").astype(float)
    for code, group in result.groupby("material_code"):
        if code not in DATASETS[name]:
            raise ValueError(f"{name}: unknown material {code}.")
        metadata = MATERIALS[code]
        price_columns = [c for c in ("price", "open", "high", "low", "close") if c in group]
        for column in price_columns:
            values = pd.to_numeric(group[column], errors="raise")
            if not values.between(metadata["minimum"], metadata["maximum"]).all():
                raise ValueError(f"{name}/{code}: {column} outside sanity bounds.")
        if any(group[c].nunique(dropna=False) != 1 for c in METADATA_COLUMNS):
            raise ValueError(f"{name}: conflicting material metadata.")
    if result.duplicated(["source_date", "material_code"]).any():
        raise ValueError(f"{name}: duplicate source observation keys.")
    result = (result.sort_values("source_date").groupby(["date", "material_code"], as_index=False)
              .last().sort_values(["date", "material_code"]).reset_index(drop=True))
    result["price"] = result["price"].round(2)
    result["date"] = result["date"].dt.as_unit("ns")
    result["ingestion_timestamp"] = pd.to_datetime(result["ingestion_timestamp"], utc=True, errors="raise").dt.as_unit("ns")
    result["is_partial_month"] = result["date"].eq(expected_months(as_of)[-1]) & (
        as_of < (pd.Timestamp(as_of) + pd.offsets.MonthEnd(0)).date())
    result = result[list(SILVER_COLUMNS)]
    validate_monthly(result, name, as_of)
    return result


def process_dataset(name: str, as_of: date) -> pd.DataFrame:
    path = BRONZE_DIR / f"{name}_raw.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return transform_frame(pd.read_parquet(path), name, as_of)


def main(as_of: date) -> None:
    frames = {name: process_dataset(name, as_of) for name in DATASETS}
    for name, frame in frames.items():
        path = SILVER_DIR / f"{name}_silver.parquet"
        save_parquet(frame, path)
        LOGGER.info("Verified %s: %d rows, %s through %s", name, len(frame),
                    frame["date"].min().date(), frame["date"].max().date())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(parse_as_of())
