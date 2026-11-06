"""Execute SQL statements against the Gold warehouse through a read-only connection."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_file(path: Path, database: Path) -> int:
    sql = path.read_text(encoding="utf-8-sig")
    with duckdb.connect(str(database), read_only=True) as connection:
        statements = connection.extract_statements(sql)
        if not statements:
            raise ValueError("SQL file contains no statements.")
        for number, statement in enumerate(statements, 1):
            frame = connection.execute(statement).df()
            print(f"Statement {number}: {len(frame)} rows")
            print(frame.to_string(index=False, max_rows=25))
    return len(statements)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--database", type=Path, default=PROJECT_ROOT / "database/buildmetrics.duckdb")
    args = parser.parse_args()
    sql_file = args.file if args.file.is_absolute() else PROJECT_ROOT / args.file
    database = args.database if args.database.is_absolute() else PROJECT_ROOT / args.database
    try:
        run_file(sql_file, database)
    except (OSError, ValueError, duckdb.Error) as error:
        print(f"Query failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
