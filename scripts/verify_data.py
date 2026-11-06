"""Verify persisted Bronze/Silver contracts without changing any data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.contracts import (DATASETS, PROJECT_ROOT, assert_complete_values,
                           resolve_as_of, validate_monthly)
from src.transformation.process_silver import transform_frame


def verify_data(as_of, layers=("bronze", "silver")) -> list[dict]:
    results = []
    for layer in layers:
        for name in DATASETS:
            suffix = "raw" if layer == "bronze" else "silver"
            path = PROJECT_ROOT / "data" / layer / f"{name}_{suffix}.parquet"
            if not path.is_file():
                raise AssertionError(f"Missing {path}")
            frame = pd.read_parquet(path)
            if len(frame) <= 100:
                raise AssertionError(f"{path}: expected more than 100 rows")
            assert_complete_values(frame)
            if layer == "silver":
                validate_monthly(frame, name, as_of)
                bronze = pd.read_parquet(PROJECT_ROOT / "data/bronze" / f"{name}_raw.parquet")
                pd.testing.assert_frame_equal(frame, transform_frame(bronze, name, as_of))
            else:
                transform_frame(frame, name, as_of)
            results.append(dict(file=str(path.relative_to(PROJECT_ROOT)), rows=len(frame),
                                nulls=int(frame.isna().sum().sum()),
                                start=str(pd.to_datetime(frame["date"]).min().date()),
                                end=str(pd.to_datetime(frame["date"]).max().date())))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of")
    parser.add_argument("--layer", choices=["bronze", "silver", "all"], default="all")
    args = parser.parse_args()
    layers = ("bronze", "silver") if args.layer == "all" else (args.layer,)
    print(json.dumps(verify_data(resolve_as_of(args.as_of), layers), indent=2))
