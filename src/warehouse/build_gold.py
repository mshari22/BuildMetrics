"""Build a fresh constrained DuckDB star schema and publish it atomically."""
from __future__ import annotations

import hashlib
import logging
import sys
import tempfile
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.contracts import DATASETS, MATERIALS, PROJECT_ROOT, expected_months, parse_as_of, validate_monthly

LOGGER = logging.getLogger(__name__)
SILVER_DIR = PROJECT_ROOT / "data/silver"
DATABASE_PATH = PROJECT_ROOT / "database/buildmetrics.duckdb"
CORRELATION_COLUMNS = ("brent_oil_price", "steel_index_price", "portland_cement_price",
                       "reinforcing_rebar_price", "ready_mix_concrete_price")


def load_silver(connection: duckdb.DuckDBPyConnection, as_of: date) -> list[dict]:
    audit = []
    paths = []
    for name in DATASETS:
        path = SILVER_DIR / f"{name}_silver.parquet"
        frame = pd.read_parquet(path)
        validate_monthly(frame, name, as_of)
        paths.append(str(path))
        audit.append(dict(dataset=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          row_count=len(frame)))
    connection.execute("CREATE TEMP TABLE silver_prices AS SELECT * FROM read_parquet(?)", [paths])
    catalog = pd.DataFrame([dict(material_code=code, **values) for code, values in MATERIALS.items()])
    connection.register("material_catalog", catalog)
    connection.register("input_audit", pd.DataFrame(audit))
    return audit


def create_star_schema(connection: duckdb.DuckDBPyConnection, as_of: date) -> None:
    connection.execute("""
        CREATE TABLE dim_date (
            date_key INTEGER PRIMARY KEY,
            date DATE NOT NULL UNIQUE CHECK (day(date) = 1),
            year SMALLINT NOT NULL,
            quarter TINYINT NOT NULL CHECK (quarter BETWEEN 1 AND 4),
            month TINYINT NOT NULL CHECK (month BETWEEN 1 AND 12),
            month_name VARCHAR NOT NULL,
            CHECK (date_key = CAST(year AS INTEGER) * 10000 + CAST(month AS INTEGER) * 100 + 1),
            CHECK (year = year(date) AND month = month(date) AND quarter = quarter(date))
        );
        INSERT INTO dim_date
        SELECT CAST(strftime(CAST(date AS DATE), '%Y%m%d') AS INTEGER), CAST(date AS DATE),
               year(date), quarter(date), month(date), monthname(date)
        FROM (SELECT DISTINCT date FROM silver_prices) ORDER BY date;

        CREATE TABLE dim_material (
            material_id INTEGER PRIMARY KEY,
            material_name VARCHAR NOT NULL UNIQUE,
            category VARCHAR NOT NULL CHECK (category IN ('Global Benchmark', 'Local GASTAT')),
            unit VARCHAR NOT NULL,
            currency VARCHAR NOT NULL CHECK (currency IN ('USD', 'SAR')),
            source VARCHAR NOT NULL
        );
        INSERT INTO dim_material
        SELECT material_id, material_name, category, unit, currency, source
        FROM material_catalog ORDER BY material_id;

        CREATE TABLE fact_prices (
            date DATE NOT NULL REFERENCES dim_date(date),
            material_id INTEGER NOT NULL REFERENCES dim_material(material_id),
            price DOUBLE NOT NULL CHECK (isfinite(price) AND price > 0),
            mom_change_pct DOUBLE CHECK (mom_change_pct IS NULL OR (isfinite(mom_change_pct) AND mom_change_pct > -100)),
            yoy_change_pct DOUBLE CHECK (yoy_change_pct IS NULL OR (isfinite(yoy_change_pct) AND yoy_change_pct > -100)),
            PRIMARY KEY (date, material_id)
        );
        INSERT INTO fact_prices
        WITH aligned AS (
            SELECT CAST(s.date AS DATE) AS date, m.material_id, CAST(s.price AS DOUBLE) AS price,
                   lag(s.price, 1) OVER w AS previous_month,
                   lag(s.price, 12) OVER w AS previous_year
            FROM silver_prices s JOIN material_catalog m USING (material_code)
            WINDOW w AS (PARTITION BY material_id ORDER BY s.date)
        )
        SELECT date, material_id, price,
               round(100.0 * (price / previous_month - 1), 6),
               round(100.0 * (price / previous_year - 1), 6)
        FROM aligned ORDER BY date, material_id;

        CREATE VIEW vw_global_local_correlation AS
        SELECT b.date, b.price AS brent_oil_price, s.price AS steel_index_price,
               c.price AS portland_cement_price, r.price AS reinforcing_rebar_price,
               x.price AS ready_mix_concrete_price
        FROM fact_prices b
        JOIN fact_prices s ON s.date = b.date AND s.material_id = 2
        JOIN fact_prices c ON c.date = b.date AND c.material_id = 3
        JOIN fact_prices r ON r.date = b.date AND r.material_id = 4
        JOIN fact_prices x ON x.date = b.date AND x.material_id = 5
        WHERE b.material_id = 1;

        CREATE TABLE warehouse_metadata (
            schema_version INTEGER PRIMARY KEY,
            as_of DATE NOT NULL,
            built_at TIMESTAMPTZ NOT NULL,
            latest_ingestion TIMESTAMPTZ NOT NULL,
            current_month_is_partial BOOLEAN NOT NULL
        );
        CREATE TABLE warehouse_inputs (
            dataset VARCHAR PRIMARY KEY,
            sha256 VARCHAR NOT NULL,
            row_count INTEGER NOT NULL CHECK (row_count > 0)
        );
        INSERT INTO warehouse_inputs SELECT * FROM input_audit;
    """)
    connection.execute("""
        INSERT INTO warehouse_metadata
        SELECT 2, CAST(? AS DATE), current_timestamp, max(ingestion_timestamp), bool_or(is_partial_month)
        FROM silver_prices
    """, [as_of])


def verify_gold(connection: duckdb.DuckDBPyConnection, as_of: date) -> dict[str, int]:
    months = len(expected_months(as_of))
    counts = {}
    for table, expected in (("dim_date", months), ("dim_material", 5),
                            ("fact_prices", months * 5), ("vw_global_local_correlation", months)):
        frame = connection.execute(f"SELECT * FROM {table}").df()
        if len(frame) != expected:
            raise AssertionError(f"{table}: expected {expected}, got {len(frame)}.")
        required = frame.drop(columns=["mom_change_pct", "yoy_change_pct"], errors="ignore")
        if required.isna().any().any():
            raise AssertionError(f"{table}: unexpected NULL.")
        counts[table] = len(frame)
    changes = connection.execute("""
        SELECT f.date, f.material_id, f.mom_change_pct, f.yoy_change_pct,
               100 * (f.price / p.price - 1) AS expected_mom,
               100 * (f.price / y.price - 1) AS expected_yoy
        FROM fact_prices f
        LEFT JOIN fact_prices p ON p.material_id = f.material_id AND p.date = f.date - INTERVAL '1 month'
        LEFT JOIN fact_prices y ON y.material_id = f.material_id AND y.date = f.date - INTERVAL '1 year'
    """).df()
    for actual, expected, nulls in (("mom_change_pct", "expected_mom", 5),
                                     ("yoy_change_pct", "expected_yoy", min(12, months) * 5)):
        if int(changes[actual].isna().sum()) != nulls:
            raise AssertionError(f"Unexpected {actual} NULL count.")
        if not np.array_equal(changes[actual].isna(), changes[expected].isna()):
            raise AssertionError(f"{actual}: wrong missing-baseline behavior.")
        valid = changes[expected].notna()
        np.testing.assert_allclose(changes.loc[valid, actual], changes.loc[valid, expected], atol=0.00000051, rtol=0)
    return counts


def build_gold(as_of: date, database_path: Path = DATABASE_PATH) -> dict[str, int]:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gold_build_", dir=database_path.parent) as staging:
        staged = Path(staging) / "buildmetrics.duckdb"
        with duckdb.connect(str(staged)) as connection:
            connection.execute("SET TimeZone = 'UTC'")
            connection.execute("BEGIN TRANSACTION")
            try:
                load_silver(connection, as_of)
                create_star_schema(connection, as_of)
                counts = verify_gold(connection, as_of)
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
            connection.execute("CHECKPOINT")
        with duckdb.connect(str(staged), read_only=True) as connection:
            verify_gold(connection, as_of)
        # Failure to replace (e.g. an external writer) leaves the previous warehouse intact.
        staged.replace(database_path)
    LOGGER.info("Published verified Gold warehouse: %s", counts)
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_gold(parse_as_of())
