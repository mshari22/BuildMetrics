"""Download daily Yahoo OHLCV; Silver derives month-end observed closes."""
from __future__ import annotations

import logging
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.contracts import (MATERIALS, PROJECT_ROOT, START_DATE, assert_complete_values,
                           attach_metadata, expected_months, parse_as_of, save_parquet,
                           standardize_columns, validate_monthly)

LOGGER = logging.getLogger(__name__)
BRONZE_DIR = PROJECT_ROOT / "data/bronze"


def download_commodity(code: str, as_of: date) -> pd.DataFrame:
    ticker = MATERIALS[code]["ticker"]
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            raw = yf.download(ticker, start=START_DATE.isoformat(),
                              end=(as_of + timedelta(days=1)).isoformat(), interval="1d",
                              auto_adjust=False, actions=False, progress=False, threads=False,
                              ignore_tz=True, keepna=True, multi_level_index=False, timeout=30)
            if raw is None or raw.empty:
                raise ValueError(f"{ticker}: provider returned no observations.")
            frame = standardize_columns(raw.reset_index())
            required = ["date", "open", "high", "low", "close", "volume"]
            if not set(required).issubset(frame):
                raise ValueError(f"{ticker}: missing OHLCV columns.")
            # Adj Close is provider-derived and is not used by this warehouse.
            frame = frame[required].copy()
            frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
            frame = frame.loc[frame["date"].between(pd.Timestamp(START_DATE), pd.Timestamp(as_of))].copy()
            # Yahoo includes occasional holiday placeholders: no prices and zero volume.
            # They are not observations. Partially missing traded records still fail below.
            no_trade = frame[["open", "high", "low", "close"]].isna().all(axis=1) & frame["volume"].eq(0)
            if no_trade.any():
                LOGGER.info("%s: excluded %d non-trading placeholders", ticker, int(no_trade.sum()))
                frame = frame.loc[~no_trade].copy()
            assert_complete_values(frame)
            for column in ("open", "high", "low", "close"):
                frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
                if not frame[column].between(MATERIALS[code]["minimum"], MATERIALS[code]["maximum"]).all():
                    raise ValueError(f"{ticker}: invalid {column} prices.")
            if (frame["volume"] < 0).any() or not np.equal(frame["volume"], np.floor(frame["volume"])).all():
                raise ValueError(f"{ticker}: invalid traded volume.")
            # A futures settlement close can fall outside the reported intraday range.
            if (frame["high"] < frame["low"]).any():
                raise ValueError(f"{ticker}: high is below low.")
            frame = attach_metadata(frame, code)
            frame["ingestion_timestamp"] = pd.Timestamp(datetime.now(timezone.utc))
            projection = frame.sort_values("date").copy()
            projection["date"] = projection["date"].dt.to_period("M").dt.to_timestamp()
            projection = projection.groupby("date", sort=True, as_index=False).last()
            projection["price"] = projection["close"]
            projection["is_partial_month"] = projection["date"].eq(expected_months(as_of)[-1]) & (
                as_of < (pd.Timestamp(as_of) + pd.offsets.MonthEnd(0)).date())
            validate_monthly(projection, code, as_of, silver=False)
            return frame.sort_values("date").reset_index(drop=True)
        except Exception as error:
            last_error = error
            LOGGER.warning("%s attempt %d/3 failed: %s", ticker, attempt + 1, error)
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"{ticker}: fresh ingestion failed; existing Bronze remains unchanged.") from last_error


def ingest_commodities(as_of: date) -> None:
    frames = {code: download_commodity(code, as_of) for code in ("brent_oil", "steel_index")}
    for code, frame in frames.items():
        save_parquet(frame, BRONZE_DIR / f"{code}_raw.parquet")
        LOGGER.info("Saved %s: %d provider observations", code, len(frame))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ingest_commodities(parse_as_of())
