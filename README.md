# BuildMetrics

An Arabic–English price observatory for Saudi construction materials and global commodity benchmarks, backed by a validated monthly analytics pipeline.

**العربية:** مرصد أسعار مواد البناء والسلع الأساسية  
**English:** Construction Materials & Commodities Price Observatory

The portal defaults to Arabic and uses a traditional white, light-gray, and institutional-green design. Its presentation uses economic terminology; implementation details remain in the documentation and source code.

## Verified scope

The persisted warehouse is verified through **2026-11-06**. All five material series cover **143 monthly periods, 2015-01-01 through 2026-11-01**. The current month is provisional.

The latest recorded verification passed **23 of 23 tests**, with **zero failures, errors, or skips**. See [the verification report](reports/verification.json) for the recorded environment and evidence.

| Dataset / object | Rows |
| --- | ---: |
| Bronze Brent observations | 2,981 |
| Bronze steel observations | 2,980 |
| Bronze simulated local prices | 429 |
| Silver Brent prices | 143 |
| Silver steel prices | 143 |
| Silver local prices | 429 |
| Gold monthly dates | 143 |
| Gold materials | 5 |
| Gold price observations | 715 |
| Complete global/local monthly comparisons | 143 |

Local prices are reproducible simulations labelled `GASTAT_SIMULATED`, not official GASTAT observations. The “Global Steel Index” material uses the **SLX ETF proxy in USD per ETF share**, not a physical steel price. Real local data must replace the simulation before using the product for real procurement or economic decisions.

## Quick start

Use Python 3.12. The checked environment versions are pinned in `requirements.txt`; the exact installed dependency set is captured in `requirements.lock.txt`.

Run the following commands from the project root. To open the portal with the existing data:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\venv\Scripts\python.exe -m streamlit run app/main.py
```

Open the local URL printed by Streamlit, normally [http://localhost:8501](http://localhost:8501).

To refresh the data and verify the complete project:

```powershell
.\venv\Scripts\python.exe scripts/run_pipeline.py --as-of 2026-11-06
.\venv\Scripts\python.exe scripts/verify_project.py --as-of 2026-11-06
```

The example cutoff matches the supplied data. For a new run, omit `--as-of` from both commands to use today's date in Riyadh, or supply the same cutoff to both. Fresh market ingestion requires internet access. The pipeline stops at the first failed stage, and the verification command exits nonzero for any failed check.

## Using the portal

Choose **العربية** or **English** at the top of the sidebar. Arabic uses right-to-left text, controls, cards, and tables; English uses left-to-right layout. Changing language preserves the selected period and materials.

The sidebar provides start/end year and month controls, category and material selections, and **تحديث العرض / Refresh View**. Refresh View reloads the available published data; run the pipeline command above to retrieve new market observations. All categories and materials are selected initially. Empty selections and reversed dates produce clear messages.

Four bordered KPI cards show the latest selected Brent, steel, rebar, and cement prices with their monthly changes.

| Arabic tab | English tab | Features |
| --- | --- | --- |
| اتجاهات الأسعار | Price Trends | Interactive price charts and a Base 100 comparison toggle |
| مقارنة السوق المحلي والعالمي | Local vs. Global Market | Correlation table, monthly-change or price-level comparison, and scatter trend |
| البيانات التفصيلية | Detailed Data | Business column labels, newest/oldest ordering, pagination, and CSV export |

Detailed tables offer 25, 50, or 100 rows per page. **تصدير البيانات (CSV) / Export CSV** exports all filtered rows with localized headers and values, regardless of the current page. UTF-8 with a byte-order mark supports Arabic text in Excel.

Charts use Plotly's `plotly_white` template with green, charcoal, muted blue, and gray. The theme is configured in `.streamlit/config.toml`; typography, directional layout, KPI borders, and semantic HTML tables are defined in `app/main.py`. Arabic requests Tajawal/Cairo and English requests Inter, with Segoe UI and system sans-serif fallbacks; fonts are not downloaded by the app.

A professional footnote identifies local prices as simulated. The Arabic steel display uses the requested “دولار / نقطة” label; its underlying values remain SLX share prices, with “USD / Share” in English and an explanatory KPI help text.

## Project structure

```text
app/main.py                         Bilingual Streamlit presentation
.streamlit/config.toml              Light theme settings
src/contracts.py                   Shared material metadata and validation rules
src/ingestion/ingest_commodities.py  Daily global market ingestion
src/ingestion/ingest_gastat.py       Reproducible local-price simulation
src/transformation/process_silver.py Monthly normalization and validation
src/warehouse/build_gold.py         Constrained DuckDB schema and atomic publication
src/warehouse/query_runner.py       Read-only SQL command-line runner
scripts/run_pipeline.py             Ordered pipeline execution
scripts/verify_data.py              Bronze/Silver file verification
scripts/verify_project.py           End-to-end data and presentation QA
queries/analytics_demo.sql          Analytical query examples
data/bronze/                       Validated source observations
data/silver/                       Aligned monthly prices
database/buildmetrics.duckdb        Published analytics database
reports/                           Verification evidence and logs
```

## Data contracts

- Bronze preserves observed daily unadjusted Yahoo OHLCV for `BZ=F` and `SLX`, together with canonical metadata and UTC ingestion timestamps. It excludes empty, zero-volume holiday placeholders and rejects partially missing or invalid traded records. January 1, 2015 was not a trading day, so the first market observation is January 2.
- Silver uses the **last observed daily close** within each calendar month, rounded to two decimal places, labelled with the first day of that month. It uses a common column contract across all three files and preserves provenance and the provisional-month flag.
- All expected months and material combinations must exist. Missing prices are never interpolated, forward-filled, or fabricated. A provider outage or a new month without an observed trading price fails validation and preserves the previous published warehouse.
- Prices must be finite, strictly positive, and within the material-specific sanity limits in `src/contracts.py`. These broad limits detect corrupt data; they are not economic forecasts. Unexpected spellings, duplicate keys, metadata changes, nulls, and infinite values fail validation.
- Parquet files are staged and reread before individual atomic replacement. All datasets are validated before publishing a layer. Run one pipeline writer at a time; publication of multiple Parquet files is not a single filesystem transaction.

## Gold schema and mathematics

| Object | Grain / columns |
| --- | --- |
| `dim_date` | One row per month; `date_key, date, year, quarter, month, month_name` |
| `dim_material` | Five fixed material IDs; `material_id, material_name, category, unit, currency, source` |
| `fact_prices` | One row per date/material; `date, material_id, price, mom_change_pct, yoy_change_pct` |
| `vw_global_local_correlation` | One complete monthly panel with the five material prices |
| `warehouse_metadata` | Data cutoff, build and ingestion timestamps, schema version, provisional status |
| `warehouse_inputs` | Silver file SHA-256 hashes and row counts |

Primary keys, foreign keys, NOT NULL constraints, and finite positive-price checks are enforced in DuckDB. Gold is built and verified in a new database, closed, and atomically substituted for the previous file. A failed replacement leaves the previous database intact.

`MoM = 100 × (price / price one calendar month earlier − 1)`

`YoY = 100 × (price / price twelve calendar months earlier − 1)`

Percent changes are stored to six decimal places. The five initial MoM values and sixty initial YoY values are intentionally NULL because their baselines do not exist. These are the only expected missing values in the fact table; replacing them with zero would be mathematically misleading.

Base 100 divides every material price by its first selected monthly price and multiplies by 100. It does not convert currencies. Raw charts retain their original units.

Correlation uses exact shared monthly dates. Monthly percentage changes are the dashboard default, and the provisional month is excluded by default. Users may select levels explicitly. Fewer than three matched months or constant series produce informative messages. Pearson correlation and the scatter's OLS line are descriptive, not evidence of causality.

## Application data access and SQL

The app uses short-lived `read_only=True` connections inside a cached snapshot helper. Connections close before data is returned. Read-only mode prevents writes; closing connections releases their read locks. Database modification time and file size invalidate the cache after rebuilds.

All application SQL lives in `APP_QUERIES` and is executed by QA. The query registry still retrieves internal row counts for verification, but database names and counts are not displayed in the portal. Translations are applied to presentation copies; the stored material names, prices, data contracts, and mathematical calculations retain their canonical values.

```powershell
.\venv\Scripts\python.exe src/warehouse/query_runner.py --file queries/analytics_demo.sql
```

The SQL runner uses a read-only connection and exits nonzero on SQL or file errors.

## Verification evidence

`scripts/verify_project.py` checks all six Parquet files, complete calendars, exact schema types, key/price constraints, Silver-to-Gold price reconciliation, independently recalculated MoM/YoY values, dimension dates, correlation alignment, source hashes, adversarial inputs, deterministic simulation history, every app query, Python compilation, SQL CLI execution, dashboard interactions, and the HTTP health endpoint.

Presentation checks cover Arabic defaults, both languages, filter persistence, translated month options, RTL/LTR table markup, light-theme settings, Plotly styling, KPI delta signs, localized export labels, pagination and sorting, and absence of engineering terms in the displayed business content.

Results are written to [`reports/verification.json`](reports/verification.json), with SQL CLI output and server logs in `reports/`. The most recent captured terminal run is [`reports/verification_run.log`](reports/verification_run.log). A fully passing run establishes that these checks passed for the stated cutoff and environment; it does not prove the absence of every possible defect or make simulated data suitable for real production analysis.
