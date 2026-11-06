"""Independent end-to-end QA. Writes an evidence report and exits nonzero on any failure."""
from __future__ import annotations

import argparse
import ast
import hashlib
import http.client
import importlib.metadata
import importlib.util
import json
import logging
import os
import py_compile
import subprocess
import sys
import tempfile
import time
import tomllib
import unittest
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.contracts import DATASETS, MATERIALS, PROJECT_ROOT, expected_months, resolve_as_of, standardize_columns
from src.ingestion.ingest_gastat import generate_gastat_data
from src.transformation.process_silver import transform_frame
from src.warehouse.build_gold import DATABASE_PATH, create_star_schema, load_silver, verify_gold
from scripts.verify_data import verify_data

APP_PATH = PROJECT_ROOT / "app/main.py"
spec = importlib.util.spec_from_file_location("buildmetrics_dashboard", APP_PATH)
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)
EVIDENCE: dict = {}
AS_OF = None


def connect():
    return duckdb.connect(str(DATABASE_PATH), read_only=True)


class DataAndWarehouseQA(unittest.TestCase):
    def test_01_all_bronze_and_silver_parquets(self):
        EVIDENCE["parquet"] = verify_data(AS_OF)
        for layer in ("bronze", "silver"):
            expected = {f"{name}_{'raw' if layer == 'bronze' else 'silver'}.parquet" for name in DATASETS}
            self.assertEqual({p.name for p in (PROJECT_ROOT / "data" / layer).glob("*.parquet")}, expected)

    def test_02_exact_gold_schema_types(self):
        expected = {
            "dim_date": [("date_key", "INTEGER"), ("date", "DATE"), ("year", "SMALLINT"),
                         ("quarter", "TINYINT"), ("month", "TINYINT"), ("month_name", "VARCHAR")],
            "dim_material": [("material_id", "INTEGER"), ("material_name", "VARCHAR"), ("category", "VARCHAR"),
                             ("unit", "VARCHAR"), ("currency", "VARCHAR"), ("source", "VARCHAR")],
            "fact_prices": [("date", "DATE"), ("material_id", "INTEGER"), ("price", "DOUBLE"),
                            ("mom_change_pct", "DOUBLE"), ("yoy_change_pct", "DOUBLE")],
        }
        with connect() as connection:
            EVIDENCE["gold_counts"] = verify_gold(connection, AS_OF)
            for table, columns in expected.items():
                with self.subTest(table=table):
                    actual = connection.execute(f"DESCRIBE {table}").fetchall()
                    self.assertEqual([(row[0], row[1]) for row in actual], columns)
            for table in ("warehouse_inputs", "warehouse_metadata"):
                frame = connection.execute(f"SELECT * FROM {table}").df()
                self.assertFalse(frame.empty)
                self.assertFalse(frame.isna().any().any())

    def test_03_fact_prices_reconcile_to_silver(self):
        frames = []
        for name in DATASETS:
            frame = pd.read_parquet(PROJECT_ROOT / "data/silver" / f"{name}_silver.parquet")
            frame["material_id"] = frame["material_code"].map({k: v["material_id"] for k, v in MATERIALS.items()})
            frames.append(frame[["date", "material_id", "price"]])
        expected = pd.concat(frames).sort_values(["material_id", "date"]).reset_index(drop=True)
        with connect() as connection:
            actual = connection.execute("SELECT * FROM fact_prices ORDER BY material_id, date").df()
        np.testing.assert_array_equal(actual["date"].to_numpy(dtype="datetime64[D]"),
                                      expected["date"].to_numpy(dtype="datetime64[D]"))
        np.testing.assert_array_equal(actual["material_id"], expected["material_id"])
        np.testing.assert_array_equal(actual["price"], expected["price"])
        for lag, column in ((1, "mom_change_pct"), (12, "yoy_change_pct")):
            prior = expected.groupby("material_id")["price"].shift(lag)
            changes = 100.0 * (expected["price"] / prior - 1)
            np.testing.assert_allclose(actual[column], changes, atol=0.00000051, rtol=0, equal_nan=True)
        EVIDENCE["expected_null_growth_rates"] = {
            "mom_change_pct": int(actual["mom_change_pct"].isna().sum()),
            "yoy_change_pct": int(actual["yoy_change_pct"].isna().sum()),
        }

    def test_04_date_dimension_and_correlation_view(self):
        with connect() as connection:
            dates = connection.execute("SELECT * FROM dim_date ORDER BY date").df()
            facts = connection.execute("SELECT date, material_id, price FROM fact_prices").df()
            view = connection.execute("SELECT * FROM vw_global_local_correlation ORDER BY date").df()
        np.testing.assert_array_equal(dates["date"].to_numpy(dtype="datetime64[D]"),
                                      expected_months(AS_OF).to_numpy(dtype="datetime64[D]"))
        np.testing.assert_array_equal(dates["date_key"], dates["date"].dt.strftime("%Y%m%d").astype(int))
        np.testing.assert_array_equal(dates["year"], dates["date"].dt.year)
        np.testing.assert_array_equal(dates["quarter"], dates["date"].dt.quarter)
        np.testing.assert_array_equal(dates["month"], dates["date"].dt.month)
        np.testing.assert_array_equal(dates["month_name"], dates["date"].dt.month_name())
        self.assertFalse(view.isna().any().any())
        self.assertEqual(len(view), len(dates))
        pivot = facts.pivot(index="date", columns="material_id", values="price").sort_index()
        for material_id, column in enumerate(dashboard.CORRELATION_MATERIALS, 1):
            np.testing.assert_array_equal(view[column], pivot[material_id])

    def test_05_primary_foreign_and_check_constraints(self):
        # A separate in-memory build tests actual constraint behavior without modifying Gold.
        with duckdb.connect(":memory:") as connection:
            load_silver(connection, AS_OF)
            create_star_schema(connection, AS_OF)
            invalid = [
                "INSERT INTO fact_prices VALUES ('2015-01-01',1,10,NULL,NULL)",
                "INSERT INTO fact_prices VALUES ('2015-01-02',1,10,NULL,NULL)",
                "INSERT INTO fact_prices VALUES ('2015-01-01',999,10,NULL,NULL)",
                "UPDATE fact_prices SET price = 0 WHERE material_id = 1",
                "UPDATE fact_prices SET price = -1 WHERE material_id = 1",
                "UPDATE fact_prices SET price = 'NaN'::DOUBLE WHERE material_id = 1",
                "UPDATE fact_prices SET price = 'Infinity'::DOUBLE WHERE material_id = 1",
                "UPDATE fact_prices SET price = NULL WHERE material_id = 1",
                "UPDATE fact_prices SET mom_change_pct = 'Infinity'::DOUBLE WHERE material_id = 1",
                "INSERT INTO dim_date VALUES (20990102,'2099-01-02',2099,1,1,'January')",
                "INSERT INTO dim_date VALUES (20990102,'2099-01-01',2099,1,1,'January')",
                "INSERT INTO dim_material VALUES (999,'Invalid','unknown','bag','SAR','invalid')",
            ]
            for sql in invalid:
                with self.subTest(sql=sql), self.assertRaises(duckdb.ConstraintException):
                    connection.execute(sql)
            constraints = connection.execute("""
                SELECT table_name, constraint_type FROM duckdb_constraints()
                WHERE table_name IN ('dim_date','dim_material','fact_prices')
            """).fetchall()
            self.assertEqual(sum(kind == "PRIMARY KEY" for _, kind in constraints), 3)
            self.assertEqual(sum(kind == "FOREIGN KEY" for _, kind in constraints), 2)
            verify_gold(connection, AS_OF)
        EVIDENCE["invalid_writes_rejected"] = len(invalid)

    def test_06_lineage_and_cutoff(self):
        with connect() as connection:
            audit = connection.execute("SELECT * FROM warehouse_inputs").df()
            metadata = connection.execute("SELECT * FROM warehouse_metadata").df().iloc[0]
            materials = connection.execute("SELECT * FROM dim_material ORDER BY material_id").df()
        self.assertEqual(pd.Timestamp(metadata["as_of"]).date(), AS_OF)
        for row in audit.itertuples(index=False):
            path = PROJECT_ROOT / "data/silver" / f"{row.dataset}_silver.parquet"
            self.assertEqual(row.sha256, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(row.row_count, len(pd.read_parquet(path)))
        for expected, actual in zip(MATERIALS.values(), materials.to_dict("records")):
            for column in actual:
                self.assertEqual(actual[column], expected[column])
        EVIDENCE["lineage_sha256_verified"] = len(audit)

    def test_07_silver_rejects_invalid_data(self):
        source = pd.read_parquet(PROJECT_ROOT / "data/bronze/brent_oil_raw.parquet")
        bad_values = (None, np.nan, np.inf, -np.inf, 0, -10, 1000000)
        for value in bad_values:
            with self.subTest(price=value):
                bad = source.copy()
                # An early daily record must be rejected even if it is not the month's last.
                bad.loc[0, "close"] = value
                with self.assertRaises(ValueError):
                    transform_frame(bad, "brent_oil", AS_OF)
        for column, value in (("unit", "wrong"), ("material_name", "Brent Crude Oill"),
                              ("currency", "usd"), ("source", None)):
            with self.subTest(column=column):
                bad = source.copy()
                bad.loc[0, column] = value
                with self.assertRaises(ValueError):
                    transform_frame(bad, "brent_oil", AS_OF)
        bad = source.loc[source["date"].dt.to_period("M") != pd.Period("2020-01")].copy()
        with self.assertRaises(ValueError):
            transform_frame(bad, "brent_oil", AS_OF)
        with self.assertRaises(ValueError):
            transform_frame(pd.concat([source, source.iloc[[0]]]), "brent_oil", AS_OF)
        with self.assertRaises(ValueError):
            standardize_columns(pd.DataFrame(columns=["Adj Close", "adj_close"]))

    def test_08_simulated_history_is_reproducible(self):
        short = generate_gastat_data(as_of=resolve_as_of("2020-12-31")).drop(columns="ingestion_timestamp")
        long = generate_gastat_data(as_of=AS_OF).drop(columns="ingestion_timestamp")
        historical = long.loc[long["date"] <= short["date"].max()].reset_index(drop=True)
        pd.testing.assert_frame_equal(short, historical)

    def test_09_query_runner_cli(self):
        run = subprocess.run([sys.executable, str(PROJECT_ROOT / "src/warehouse/query_runner.py"),
                              "--file", "queries/analytics_demo.sql"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        (PROJECT_ROOT / "reports/query_runner_output.txt").write_text(run.stdout + run.stderr, encoding="utf-8")
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(run.stdout.count("Statement "), 5)
        EVIDENCE["query_runner_exit_code"] = run.returncode
        EVIDENCE["analytics_demo_statements"] = 5

    def test_10_compile_all_sources_and_app_sql(self):
        paths = [*PROJECT_ROOT.glob("src/**/*.py"), *PROJECT_ROOT.glob("app/*.py"),
                 *PROJECT_ROOT.glob("scripts/*.py")]
        for path in paths:
            py_compile.compile(str(path), doraise=True)
        with connect() as connection:
            rows = {name: len(connection.execute(sql).df()) for name, sql in dashboard.APP_QUERIES.items()}
            for count in rows.values():
                self.assertGreater(count, 0)
        tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "execute"]
        self.assertEqual(len(calls), 1, "Keep all dashboard SQL in APP_QUERIES for complete pretesting")
        self.assertIsInstance(calls[0].args[0], ast.Name)
        self.assertEqual(calls[0].args[0].id, "sql")
        EVIDENCE["app_query_row_counts"] = rows
        EVIDENCE["compiled_python_files"] = len(paths)

    def test_11_read_only_snapshot_releases_connection(self):
        stat = DATABASE_PATH.stat()
        frames = dashboard.load_warehouse(str(DATABASE_PATH), (stat.st_mtime_ns, stat.st_size))
        self.assertEqual(len(frames["prices"]), len(expected_months(AS_OF)) * 5)
        with connect() as connection, self.assertRaises(duckdb.Error):
            connection.execute("CREATE TABLE qa_must_not_write (id INTEGER)")
        # A read-write connection can open because no cached read-only handle remains.
        with duckdb.connect(str(DATABASE_PATH)) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM fact_prices").fetchone()[0],
                             len(frames["prices"]))

    def test_12_normalization_and_correlation_math(self):
        with connect() as connection:
            prices = connection.execute(dashboard.APP_QUERIES["prices"]).df()
            panel = connection.execute(dashboard.APP_QUERIES["correlation"]).df()
            meta = connection.execute(dashboard.APP_QUERIES["metadata"]).df().iloc[0]
        prices["date"] = pd.to_datetime(prices["date"])
        panel["date"] = pd.to_datetime(panel["date"])
        index = dashboard.normalized_index(prices)
        np.testing.assert_allclose(index.iloc[0], 100)
        for name in index:
            rows = prices.loc[prices["material_name"].eq(name)].sort_values("date")
            self.assertAlmostEqual(index[name].iloc[-1], rows["price"].iloc[-1] / rows["price"].iloc[0] * 100)
        start, end = pd.Timestamp("2020-01-01"), prices["date"].max()
        observed = dashboard.correlation_observations(panel, list(index), start, end, monthly_changes=True,
                    exclude_partial=True, as_of=meta["as_of"], is_partial=bool(meta["current_month_is_partial"]))
        expected = panel.set_index("date").rename(columns=dashboard.CORRELATION_MATERIALS).pct_change(fill_method=None) * 100
        expected = expected.loc[expected.index >= start]
        if meta["current_month_is_partial"]:
            expected = expected.loc[expected.index < pd.Timestamp(meta["as_of"]).replace(day=1)]
        pd.testing.assert_frame_equal(observed, expected)
        self.assertTrue(np.isfinite(observed.corr()).all().all())
        self.assertTrue((observed.corr().abs() <= 1.000000001).all().all())


class DashboardQA(unittest.TestCase):
    def app(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=45)
        self.assertEqual(len(app.exception), 0, str(app.exception))
        return app

    def clean(self, app):
        self.assertEqual(len(app.exception), 0, str(app.exception))

    def tables(self, app):
        return [item.proto.body for item in app.get("html") if '<table class="price-table"' in item.proto.body]

    def test_13_default_executive_dashboard(self):
        app = self.app()
        self.assertEqual(len(app.tabs), 3)
        self.assertEqual(len(app.metric), 4)
        self.assertEqual(len(self.tables(app)), 2)
        self.assertEqual(app.selectbox(key="language").value, "ar")
        self.assertEqual(app.title[0].value, dashboard.TEXT["title"][0])
        self.assertEqual([tab.label for tab in app.tabs], [dashboard.TEXT[key][0] for key in
                         ("tab_trends", "tab_comparison", "tab_details")])
        self.assertTrue(all('dir="rtl"' in table for table in self.tables(app)))
        # The sign must lead the delta so Streamlit chooses the correct direction arrow.
        with connect() as connection:
            latest = connection.execute(dashboard.APP_QUERIES["prices"]).df()
        for metric, name in zip(app.metric, dashboard.KPI_NAMES):
            change = latest.loc[latest["material_name"].eq(name)].iloc[-1]["mom_change_pct"]
            self.assertTrue(metric.delta.startswith(f"{change:+.2f}%"))
        self.assertEqual(len(app.multiselect(key="materials").value), 5)
        self.assertEqual(app.selectbox(key="start_year").value, 2015)
        self.assertEqual(app.selectbox(key="end_year").value, AS_OF.year)
        self.assertEqual(app.selectbox(key="end_month_ar").value, AS_OF.month)

    def test_14_empty_filters_show_warning(self):
        for key in ("materials", "categories"):
            with self.subTest(filter=key):
                app = self.app()
                app.multiselect(key=key).set_value([]).run(timeout=45)
                self.clean(app)
                self.assertGreater(len(app.warning), 0)

    def test_15_reversed_date_range(self):
        app = self.app()
        app.selectbox(key="start_year").set_value(AS_OF.year)
        app.selectbox(key="end_year").set_value(2015).run(timeout=45)
        self.clean(app)
        self.assertEqual(app.warning[0].value, dashboard.TEXT["reversed"][0])

    def test_16_single_month_and_no_growth_baseline(self):
        app = self.app()
        app.selectbox(key="end_year").set_value(2015)
        app.selectbox(key="end_month_ar").set_value(1).run(timeout=45)
        self.clean(app)
        self.assertEqual(len(app.metric), 4)
        self.assertTrue(any(dashboard.TEXT["few_months"][0] == element.value for element in app.info))
        for metric in app.metric[:4]:
            self.assertEqual(metric.delta, "")

    def test_17_normalized_index_and_price_level_correlation(self):
        app = self.app()
        app.toggle(key="normalized").set_value(True).run(timeout=45)
        self.clean(app)
        app.radio(key="comparison_basis").set_value("levels").run(timeout=45)
        self.clean(app)
        self.assertEqual(len(self.tables(app)), 2)

    def test_18_only_global_or_only_local_materials(self):
        for names in (["Brent Crude Oil"], ["Portland Cement", "Reinforcing Rebar"]):
            with self.subTest(materials=names):
                app = self.app()
                app.multiselect(key="materials").set_value(names).run(timeout=45)
                self.clean(app)
                self.assertEqual(len(app.metric), 4)
                expected = [dashboard.TEXT[key][0] for key in ("few_materials", "scatter_needs_pair")]
                self.assertTrue(any(item.value in expected for item in app.info))

    def test_19_missing_database_has_graceful_error(self):
        # Exercise main's actual error path without changing the project database.
        script = ("import importlib.util\nfrom pathlib import Path\n"
                  f"s=importlib.util.spec_from_file_location('missing_db_app',{str(APP_PATH)!r})\n"
                  "m=importlib.util.module_from_spec(s)\ns.loader.exec_module(m)\n"
                  f"m.DATABASE_PATH=Path({str(PROJECT_ROOT / 'reports/nonexistent.duckdb')!r})\n"
                  "m.main()\n")
        app = AppTest.from_string(script).run(timeout=45)
        self.clean(app)
        self.assertEqual(len(app.error), 1)

    def test_20_streamlit_server_health(self):
        import socket
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        command = [sys.executable, "-m", "streamlit", "run", str(APP_PATH), "--server.headless=true",
                   "--server.address=127.0.0.1", f"--server.port={port}", "--browser.gatherUsageStats=false"]
        log_path = PROJECT_ROOT / "reports/streamlit_server.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            try:
                deadline = time.monotonic() + 30
                healthy = False
                while time.monotonic() < deadline and process.poll() is None:
                    try:
                        client = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
                        client.request("GET", "/_stcore/health")
                        response = client.getresponse()
                        healthy = response.status == 200 and response.read() == b"ok"
                        client.close()
                        if healthy:
                            break
                    except OSError:
                        pass
                    time.sleep(0.2)
                self.assertTrue(healthy, "Streamlit health endpoint did not return 200/ok")
                EVIDENCE["streamlit_health"] = "HTTP 200 / ok"
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

    def test_21_bilingual_labels_and_filter_persistence(self):
        app = self.app()
        selected = ["Brent Crude Oil", "Reinforcing Rebar"]
        app.multiselect(key="materials").set_value(selected).run(timeout=45)
        prices = [metric.value for metric in app.metric]
        app.selectbox(key="language").set_value("en").run(timeout=45)
        self.clean(app)
        self.assertEqual(app.title[0].value, dashboard.TEXT["title"][1])
        self.assertEqual([tab.label for tab in app.tabs], [dashboard.TEXT[key][1] for key in
                         ("tab_trends", "tab_comparison", "tab_details")])
        self.assertEqual(app.multiselect(key="materials").value, selected)
        self.assertEqual(app.selectbox(key="start_month_en").options[0], "January")
        self.assertEqual(app.selectbox(key="end_month_en").options[-1], dashboard.ENGLISH_MONTHS[AS_OF.month - 1])
        self.assertEqual([metric.value for metric in app.metric], prices)
        self.assertTrue(all('dir="ltr"' in table for table in self.tables(app)))
        detail = self.tables(app)[-1]
        for header in ("Date", "Material", "Category", "Price", "Unit", "Monthly Change %", "Annual Change %"):
            self.assertIn(f">{header}</th>", detail)
        self.assertEqual(app.get("download_button")[0].proto.label, "Export CSV")
        app.selectbox(key="language").set_value("ar").run(timeout=45)
        self.clean(app)
        self.assertEqual(app.multiselect(key="materials").value, selected)
        for header in ("التاريخ", "المادة", "التصنيف", "السعر", "الوحدة", "التغير الشهري %", "التغير السنوي %"):
            self.assertIn(f">{header}</th>", self.tables(app)[-1])
        self.assertEqual(app.get("download_button")[0].proto.label, "تصدير البيانات (CSV)")

    def test_22_light_theme_plotly_and_no_visible_engineering_terms(self):
        theme = tomllib.loads((PROJECT_ROOT / ".streamlit/config.toml").read_text(encoding="utf-8"))["theme"]
        self.assertEqual({key: theme[key] for key in ("base", "primaryColor", "backgroundColor",
                         "secondaryBackgroundColor", "textColor")},
                         dict(base="light", primaryColor="#166534", backgroundColor="#FFFFFF",
                              secondaryBackgroundColor="#F3F4F6", textColor="#1F2937"))
        app = self.app()
        forbidden = ("bronze", "silver", "gold", "duckdb", "star schema", "olap", "warehouse",
                     "parquet", "data cutoff", "slx", "etf proxy", "ols", "fact_prices",
                     "dim_material", "dim_date", "مستودع", "برونز", "طبقة الذهب")
        for language in ("ar", "en"):
            app.selectbox(key="language").set_value(language).run(timeout=45)
            self.clean(app)
            strings = []
            for kind in ("title", "subheader", "caption", "markdown", "info", "warning", "error"):
                strings.extend(str(item.value) for item in app.get(kind))
            strings.extend(self.tables(app))
            strings.extend(metric.label for metric in app.metric)
            for widget in list(app.selectbox) + list(app.multiselect) + list(app.radio):
                strings.append(widget.label)
                strings.extend(widget.options)
            strings.extend(tab.label for tab in app.tabs)
            strings.extend(item.proto.label for item in app.get("download_button"))
            for chart in app.get("plotly_chart"):
                figure = json.loads(chart.proto.spec)
                self.assertEqual(figure["layout"]["template"]["layout"]["paper_bgcolor"], "white")
                self.assertEqual(figure["layout"]["paper_bgcolor"], "#FFFFFF")
                self.assertEqual(figure["layout"]["colorway"], dashboard.PALETTE)
                self.assertEqual(figure["layout"]["yaxis"]["side"], "right" if language == "ar" else "left")
                strings.extend(trace.get("name", "") for trace in figure["data"])
                for axis in ("xaxis", "yaxis"):
                    strings.append(figure["layout"][axis].get("title", {}).get("text", ""))
            combined = " ".join(strings).lower()
            for term in forbidden:
                self.assertNotIn(term, combined, f"Visible engineering term in {language}: {term}")
            footer = [item.proto.body for item in app.get("html") if "<footer" in item.proto.body]
            self.assertEqual(len(footer), 1)
            self.assertIn(dashboard.TEXT["footnote"][0 if language == "ar" else 1], footer[0])

    def test_23_detail_pagination_sorting_and_filter_changes(self):
        app = self.app()
        app.selectbox(key="detail_page").set_value(2).run(timeout=45)
        self.clean(app)
        self.assertTrue(any("عرض 26–50" in caption.value for caption in app.caption))
        app.selectbox(key="detail_order").set_value("oldest").run(timeout=45)
        self.clean(app)
        app.selectbox(key="detail_page").set_value(1).run(timeout=45)
        self.assertIn("2015-01", self.tables(app)[-1])
        app.selectbox(key="end_year").set_value(2015)
        app.selectbox(key="end_month_ar").set_value(1).run(timeout=45)
        self.clean(app)
        self.assertEqual(app.selectbox(key="detail_page").value, 1)
        self.assertEqual(self.tables(app)[-1].count("<tbody><tr>"), 1)
        self.assertEqual(self.tables(app)[-1].count("</tr>"), 6)


def main() -> int:
    global AS_OF
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of")
    args = parser.parse_args()
    AS_OF = resolve_as_of(args.as_of)
    (PROJECT_ROOT / "reports").mkdir(exist_ok=True)
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(DataAndWarehouseQA),
                                unittest.defaultTestLoader.loadTestsFromTestCase(DashboardQA)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    versions = {name: importlib.metadata.version(name) for name in
                ("pandas", "numpy", "pyarrow", "duckdb", "yfinance", "streamlit", "altair", "plotly")}
    report = dict(verified_at=datetime.now(timezone.utc).isoformat(), as_of=AS_OF.isoformat(),
                  passed=result.wasSuccessful(), tests_run=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                  dependencies=versions, evidence=EVIDENCE,
                  limitations=["Local GASTAT prices are simulated, not official statistics.",
                               "Global Steel Index uses the SLX ETF proxy in USD per share.",
                               "The current month is provisional.",
                               "Passing QA is not proof of the absence of every possible defect."])
    (PROJECT_ROOT / "reports/verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
