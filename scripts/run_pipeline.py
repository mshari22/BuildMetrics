"""Run the Medallion stages in order and stop immediately if any stage fails."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.contracts import PROJECT_ROOT, parse_as_of


if __name__ == "__main__":
    cutoff = parse_as_of().isoformat()
    for script in ("src/ingestion/ingest_commodities.py", "src/ingestion/ingest_gastat.py",
                   "src/transformation/process_silver.py", "scripts/verify_data.py",
                   "src/warehouse/build_gold.py"):
        subprocess.run([sys.executable, str(PROJECT_ROOT / script), "--as-of", cutoff],
                       cwd=PROJECT_ROOT, check=True)
    subprocess.run([sys.executable, str(PROJECT_ROOT / "src/warehouse/query_runner.py"),
                    "--file", "queries/analytics_demo.sql"], cwd=PROJECT_ROOT, check=True)
