"""Generate explicitly simulated Saudi monthly prices, never official GASTAT data."""
from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.contracts import (PROJECT_ROOT, SILVER_COLUMNS, attach_metadata, expected_months,
                           parse_as_of, resolve_as_of, save_parquet, validate_monthly)

LOGGER = logging.getLogger(__name__)
OUTPUT_PATH = PROJECT_ROOT / "data/bronze/gastat_raw.parquet"
SIMULATION = (("cement", 14.0, 0.020, 0.010), ("rebar", 2300.0, 0.030, 0.040),
              ("concrete", 180.0, 0.025, 0.015))


def generate_gastat_data(seed: int = 42, as_of: date | None = None) -> pd.DataFrame:
    cutoff = resolve_as_of(as_of)
    dates = expected_months(cutoff)
    timestamp = pd.Timestamp(datetime.now(timezone.utc))
    periods = np.arange(len(dates))
    frames = []
    for stream, (code, base_price, growth, volatility) in enumerate(SIMULATION):
        rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
        shocks = rng.normal(0.0, volatility, len(dates))
        shocks[0] = 0.0
        prices = base_price * np.exp(np.log1p(growth) * periods / 12
                                    + 0.015 * np.sin(2 * np.pi * periods / 12)
                                    + np.cumsum(shocks))
        frame = attach_metadata(pd.DataFrame({"date": dates, "price": prices.round(2)}), code)
        frame["is_partial_month"] = frame["date"].eq(dates[-1]) & (
            cutoff < (pd.Timestamp(cutoff) + pd.offsets.MonthEnd(0)).date())
        frame["ingestion_timestamp"] = timestamp
        frames.append(frame[list(SILVER_COLUMNS)])
    result = pd.concat(frames, ignore_index=True).sort_values(["date", "material_code"]).reset_index(drop=True)
    validate_monthly(result, "gastat", cutoff)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data = generate_gastat_data(as_of=parse_as_of())
    save_parquet(data, OUTPUT_PATH)
    LOGGER.info("Saved %d explicitly simulated records to %s", len(data), OUTPUT_PATH)