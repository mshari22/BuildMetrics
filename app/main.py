"""Traditional executive BI dashboard. Run: python -m streamlit run app/main.py"""
from __future__ import annotations

import logging
from pathlib import Path

from html import escape

import plotly.graph_objects as go
import duckdb
import numpy as np
import pandas as pd
import streamlit as st

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "database/buildmetrics.duckdb"
# Every statement executed by the dashboard is registered here and exercised by QA.
APP_QUERIES = {
    "prices": """
        SELECT f.date, f.material_id, m.material_name, m.category, m.unit, m.currency,
               m.source, f.price, f.mom_change_pct, f.yoy_change_pct,
               d.date_key, d.year, d.quarter, d.month, d.month_name
        FROM fact_prices f
        JOIN dim_material m USING (material_id)
        JOIN dim_date d ON d.date = f.date
        ORDER BY f.date, f.material_id
    """,
    "correlation": "SELECT * FROM vw_global_local_correlation ORDER BY date",
    "counts": """
        SELECT 'fact_prices' AS table_name, count(*) AS row_count FROM fact_prices
        UNION ALL SELECT 'dim_material', count(*) FROM dim_material
        UNION ALL SELECT 'dim_date', count(*) FROM dim_date
    """,
    "metadata": "SELECT * FROM warehouse_metadata",
}
CORRELATION_MATERIALS = {
    "brent_oil_price": "Brent Crude Oil",
    "steel_index_price": "Global Steel Index",
    "portland_cement_price": "Portland Cement",
    "reinforcing_rebar_price": "Reinforcing Rebar",
    "ready_mix_concrete_price": "Ready-Mix Concrete",
}
KPI_NAMES = ("Brent Crude Oil", "Global Steel Index", "Reinforcing Rebar", "Portland Cement")


@st.cache_data(ttl=60, max_entries=4, show_spinner=False)
def load_warehouse(database_path: str, warehouse_version: tuple[int, int]) -> dict[str, pd.DataFrame]:
    """Cache data snapshots; close each read-only connection before returning.

    Read-only mode prevents writes, and promptly closing the connection releases
    its read lock. No connection object is cached or shared between sessions.
    """
    with duckdb.connect(database_path, read_only=True) as connection:
        frames = {name: connection.execute(sql).df() for name, sql in APP_QUERIES.items()}
    for name in ("prices", "correlation"):
        frames[name]["date"] = pd.to_datetime(frames[name]["date"])
    if frames["prices"].empty or frames["metadata"].shape[0] != 1:
        raise ValueError("The Gold warehouse is empty or its build metadata is invalid.")
    if frames["prices"].drop(columns=["mom_change_pct", "yoy_change_pct"]).isna().any().any():
        raise ValueError("The Gold warehouse contains unexpected missing values.")
    return frames


def filter_prices(frame: pd.DataFrame, materials: list[str], start, end) -> pd.DataFrame:
    return frame.loc[frame["material_name"].isin(materials)
                     & frame["date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()


def normalized_index(frame: pd.DataFrame) -> pd.DataFrame:
    wide = frame.pivot(index="date", columns="material_name", values="price").sort_index()
    if wide.empty or wide.isna().any().any() or (wide <= 0).any().any():
        raise ValueError("A normalized index requires aligned, positive price observations.")
    return wide.div(wide.iloc[0]).mul(100.0)


def correlation_observations(panel: pd.DataFrame, materials: list[str], start, end,
                             *, monthly_changes: bool, exclude_partial: bool,
                             as_of, is_partial: bool) -> pd.DataFrame:
    """Compute returns before filtering so the first selected month has a valid prior month."""
    wide = panel.set_index("date").rename(columns=CORRELATION_MATERIALS).sort_index()
    if exclude_partial and is_partial:
        wide = wide.loc[wide.index < pd.Timestamp(as_of).replace(day=1)]
    if monthly_changes:
        wide = wide.pct_change(fill_method=None).mul(100.0)
    columns = [name for name in CORRELATION_MATERIALS.values() if name in materials]
    return wide.loc[wide.index.to_series().between(pd.Timestamp(start), pd.Timestamp(end)), columns].dropna()


LANGUAGES = {"ar": "العربية", "en": "English"}
PALETTE = ["#166534", "#374151", "#0284C7", "#9CA3AF"]
TEXT = {
    "title": ("مرصد أسعار مواد البناء والسلع الأساسية", "Construction Materials & Commodities Price Observatory"),
    "subtitle": ("متابعة دورية لمؤشرات أسعار مواد البناء في السوق المحلي وأسعار السلع العالمية",
                 "Periodic tracking of local construction material prices and global commodity benchmarks"),
    "time_period": ("الفترة الزمنية (من - إلى)", "Time Period (From - To)"),
    "start_year": ("من سنة", "From Year"), "start_month": ("من شهر", "From Month"),
    "end_year": ("إلى سنة", "To Year"), "end_month": ("إلى شهر", "To Month"),
    "materials": ("المواد", "Materials"), "material": ("المادة", "Material"),
    "category": ("التصنيف", "Category"), "date": ("التاريخ", "Date"),
    "price": ("السعر", "Price"), "unit": ("الوحدة", "Unit"),
    "monthly": ("التغير الشهري", "Monthly Change"), "annual": ("التغير السنوي", "Annual Change"),
    "monthly_pct": ("التغير الشهري %", "Monthly Change %"),
    "annual_pct": ("التغير السنوي %", "Annual Change %"),
    "normalized": ("توحيد مقياس المقارنة (سنة الأساس = 100)", "Normalized Comparison Scale (Base = 100)"),
    "tab_trends": ("اتجاهات الأسعار", "Price Trends"),
    "tab_comparison": ("مقارنة السوق المحلي والعالمي", "Local vs. Global Market"),
    "tab_details": ("البيانات التفصيلية", "Detailed Data"),
    "refresh": ("تحديث العرض", "Refresh View"),
    "coverage": ("البيانات المتاحة: {start} إلى {end}", "Available period: {start} to {end}"),
    "selection": ("الفترة: {start} إلى {end} | عدد المواد: {count}", "Period: {start} to {end} | Materials: {count}"),
    "updated": ("آخر تحديث: {date}", "Last updated: {date}"),
    "provisional": ("أسعار الشهر الجاري أولية وقابلة للتحديث.", "Current-month prices are provisional and subject to revision."),
    "excluded": ("غير مشمول في الاختيار الحالي", "Outside the current selection"),
    "kpi_note": ("الأسعار لأحدث شهر ضمن الفترة المختارة، ويُحسب التغير الشهري مقارنة بالشهر السابق.",
                 "Prices reflect the latest selected month; monthly change compares with the preceding month."),
    "steel_help": ("قيمة استرشادية لحركة شركات الصلب العالمية؛ يعتمد المرجع على سعر السهم.",
                   "An indicative measure of global steel companies, based on share prices."),
    "index_note": ("تبدأ جميع المواد من 100 في {date}. وتشير القيمة 110 إلى ارتفاع بنسبة 10% عن فترة الأساس.",
                   "All materials start at 100 in {date}. A value of 110 represents a 10% increase from the base period."),
    "index_axis": ("مؤشر المقارنة (الأساس = 100)", "Comparison Index (Base = 100)"),
    "raw_note": ("تُعرض أسعار كل مادة بوحدتها الأصلية. فعّل توحيد مقياس المقارنة لمقارنة التغيرات بين المواد.",
                 "Each material retains its original unit. Enable the normalized scale to compare changes across materials."),
    "correlation_note": ("توضح المقارنة مدى تزامن تحركات الأسعار خلال الأشهر المشتركة. يُفضّل استخدام التغير الشهري عند مقارنة اتجاهات الأسواق. الارتباط لا يعني وجود علاقة سببية.",
                         "Compare price movements over matching months. Monthly changes provide a useful basis for comparing markets. Correlation does not establish causality."),
    "comparison_basis": ("أساس المقارنة", "Comparison Basis"),
    "levels": ("مستويات الأسعار", "Price Levels"),
    "exclude_partial": ("استبعاد الشهر الجاري غير المكتمل", "Exclude the Incomplete Current Month"),
    "few_months": ("اختر ثلاثة أشهر مشتركة على الأقل لعرض المقارنة.", "Select at least three matching months to compare movements."),
    "few_materials": ("اختر مادتين على الأقل تتغير أسعارهما لعرض المقارنة.",
                      "Select at least two materials with changing prices to compare movements."),
    "constant": ("استُبعدت المواد التي لم تتغير أسعارها خلال الفترة المختارة.", "Materials with unchanged prices over the selected period are excluded."),
    "correlation_title": ("ترابط تحركات الأسعار", "Price Movement Correlation"),
    "correlation_scale": ("تشمل المقارنة {count} شهرًا. تتراوح قيمة الارتباط بين −1 و+1.",
                          "The comparison covers {count} months. Correlation ranges from −1 to +1."),
    "scatter_needs_pair": ("اختر مؤشرًا عالميًا ومادة من السوق المحلي لعرض العلاقة بينهما.",
                           "Select a global benchmark and a local material to compare their movements."),
    "global": ("المؤشر العالمي", "Global Benchmark"), "local": ("المادة المحلية", "Local Material"),
    "observations": ("الأسعار الشهرية", "Monthly Prices"), "trend": ("الاتجاه العام", "Overall Trend"),
    "scatter_note": ("قيمة الارتباط: {value}. يوضح الخط الاتجاه العام خلال الفترة المختارة.",
                     "Correlation: {value}. The line shows the overall relationship over the selected period."),
    "export": ("تصدير البيانات (CSV)", "Export CSV"),
    "detail_note": ("تشمل البيانات المواد والفترة المحددة في خيارات البحث.", "Prices follow the selected materials and time period."),
    "missing_change": ("تشير الشرطة إلى عدم توفر سعر فترة المقارنة السابقة.", "A dash indicates that the earlier comparison period is unavailable."),
    "page": ("الصفحة", "Page"), "page_size": ("عدد الصفوف في الصفحة", "Rows per Page"),
    "sort": ("ترتيب البيانات", "Sort By"), "newest": ("الأحدث أولًا", "Newest First"),
    "oldest": ("الأقدم أولًا", "Oldest First"),
    "page_note": ("عرض {first}–{last} من {total} | صفحة {page} من {pages}",
                  "Showing {first}–{last} of {total} | Page {page} of {pages}"),
    "empty": ("اختر تصنيفًا واحدًا ومادة واحدة على الأقل.", "Select at least one category and one material."),
    "reversed": ("يجب أن تسبق بداية الفترة نهايتها أو تتوافق معها.", "The start month must be on or before the end month."),
    "no_results": ("لا توجد أسعار تطابق الخيارات المحددة.", "No prices match the selected filters."),
    "unavailable": ("يتعذر عرض الأسعار حاليًا. يُرجى المحاولة مجددًا بعد قليل.", "Prices are temporarily unavailable. Please try again shortly."),
    "footnote": ("ملاحظة: بيانات الأسعار المحلية المعروضة هي بيانات محاكاة لأغراض العرض التحليلي فقط ولا تمثل نشرات رسمية.",
                 "Note: Local market prices shown are simulated for analytical demonstration purposes and do not represent official releases."),
}
MATERIAL_LABELS = {
    "Brent Crude Oil": ("نفط خام برنت", "Brent Crude Oil"),
    "Global Steel Index": ("مؤشر الصلب العالمي", "Global Steel Index"),
    "Reinforcing Rebar": ("حديد التسليح", "Reinforcing Rebar"),
    "Portland Cement": ("الأسمنت البورتلاندي", "Portland Cement"),
    "Ready-Mix Concrete": ("الخرسانة الجاهزة", "Ready-Mix Concrete"),
}
CATEGORY_LABELS = {"Global Benchmark": ("المؤشرات العالمية", "Global Benchmarks"),
                   "Local GASTAT": ("السوق المحلي", "Local Market")}
UNIT_LABELS = {
    "Brent Crude Oil": ("دولار / برميل", "USD / Barrel"),
    "Global Steel Index": ("دولار / نقطة", "USD / Share"),
    "Reinforcing Rebar": ("ريال / طن", "SAR / Tonne"),
    "Portland Cement": ("ريال / كيس (50 كجم)", "SAR / 50 kg Bag"),
    "Ready-Mix Concrete": ("ريال / م³", "SAR / m³"),
}
MONTH_LABELS = ("يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
                "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر")
ENGLISH_MONTHS = ("January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December")


def is_arabic() -> bool:
    return st.session_state.get("language", "ar") == "ar"


def tr(key: str, **values) -> str:
    return TEXT[key][0 if is_arabic() else 1].format(**values)


def material_label(name: str) -> str:
    return MATERIAL_LABELS[name][0 if is_arabic() else 1]


def category_label(name: str) -> str:
    return CATEGORY_LABELS[name][0 if is_arabic() else 1]


def unit_label(name: str) -> str:
    return UNIT_LABELS[name][0 if is_arabic() else 1]


def font_family() -> str:
    return "'Tajawal', 'Cairo', 'Segoe UI', sans-serif" if is_arabic() else "'Inter', 'Segoe UI', sans-serif"


def apply_presentation_style() -> None:
    direction, alignment = ("rtl", "right") if is_arabic() else ("ltr", "left")
    st.html(f"""
    <style>
    [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stMainBlockContainer"], [data-testid="stSidebar"],
    [data-testid="stSidebarContent"] {{
        direction: {direction}; text-align: {alignment};
        font-family: {font_family()};
    }}
    [data-testid="stAppViewContainer"] {{ background: #FFFFFF; color: #1F2937; }}
    [data-testid="stSidebar"] {{ background: #F3F4F6; overflow: hidden; }}
    [data-testid="stToolbarActions"], [data-testid="stAppDeployButton"],
    [data-testid="stMainMenu"] {{ display: none; }}
    [data-testid="stMainBlockContainer"] {{ max-width: 1440px; padding-top: 2rem; padding-bottom: 2rem; }}
    [data-testid="stAppViewContainer"] h1, [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3, [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] label, [data-testid="stAppViewContainer"] button,
    [data-testid="stAppViewContainer"] input, [data-testid="stAppViewContainer"] select {{
        font-family: {font_family()}; text-align: {alignment};
    }}
    [data-testid="stAppViewContainer"] h1 {{ font-size: 1.85rem; font-weight: 600; line-height: 1.5; color: #166534; }}
    [data-testid="stAppViewContainer"] h2 {{ font-size: 1.15rem; font-weight: 600; }}
    [data-testid="stAppViewContainer"] h3 {{ font-size: 1rem; font-weight: 600; }}
    [data-testid="stHorizontalBlock"], [data-testid="stMetric"],
    [data-testid="stTabs"] [role="tablist"],
    [data-testid="stSidebar"] [data-baseweb="select"] {{
        direction: {direction}; text-align: {alignment};
    }}
    .st-key-kpi_0, .st-key-kpi_1, .st-key-kpi_2, .st-key-kpi_3 {{
        background: #FFFFFF; border: 1px solid #E5E7EB; border-top: 3px solid #166534;
        border-radius: 6px; padding: 16px; box-shadow: none;
        direction: {direction}; text-align: {alignment}; height: 100%;
    }}
    [data-testid="stMetricValue"] {{ font-size: 1.7rem; }}
    [data-testid="stMetricValue"] > div {{
        direction: ltr; unicode-bidi: isolate; text-align: {alignment};
    }}
    [data-testid="stMetricDelta"] > div {{
        direction: ltr; unicode-bidi: isolate; text-align: {alignment};
    }}
    [data-testid="stTabs"] [role="tablist"] {{ border-bottom: 1px solid #E5E7EB; }}
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{ color: #166534; font-weight: 600; }}
    .price-table-wrap {{
        direction: {direction}; overflow-x: auto; border: 1px solid #E5E7EB;
        border-radius: 6px; background: #FFFFFF; width: 100%;
    }}
    .price-table {{
        direction: {direction}; text-align: {alignment}; width: 100%;
        border-collapse: collapse; font-family: {font_family()}; font-size: .9rem;
    }}
    .price-table th {{
        text-align: {alignment}; background: #F3F4F6; color: #1F2937;
        font-weight: 600; border-bottom: 2px solid #166534; padding: 12px 14px;
    }}
    .price-table td {{ text-align: {alignment}; border-bottom: 1px solid #E5E7EB; padding: 10px 14px; }}
    .price-table tbody tr:last-child td {{ border-bottom: none; }}
    .price-table tbody tr:hover {{ background: #F3F4F6; }}
    .price-table bdi {{ unicode-bidi: isolate; white-space: nowrap; }}
    .portal-footnote {{
        direction: {direction}; text-align: {alignment}; font-family: {font_family()};
        margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #E5E7EB;
        color: #6B7280; font-size: .8rem; line-height: 1.8;
    }}
    </style>
    """)


def render_footnote() -> None:
    direction = "rtl" if is_arabic() else "ltr"
    st.html(f'<footer class="portal-footnote" lang="{st.session_state.get("language", "ar")}" '
            f'dir="{direction}">{escape(tr("footnote"))}</footer>')

def display_number(value: str) -> str:
    """Isolate numbers and calendar labels from surrounding Arabic punctuation."""
    return f"\u2066{value}\u2069" if is_arabic() else value


def _select(label: str, options: list, key: str, default, format_func=str):
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = default
    labels = {value: format_func(value) for value in options}
    # Recreate translated select controls on language changes. Some clients retain
    # the previous selected label when only format_func changes on an existing key.
    widget_key = f"{key}_{st.session_state.get('language', 'ar')}" if format_func is not str else key
    if widget_key not in st.session_state or st.session_state[widget_key] not in options:
        st.session_state[widget_key] = st.session_state[key]
    selected = st.selectbox(label, options, key=widget_key, format_func=labels.get)
    if widget_key != key:
        st.session_state[key] = selected
    return selected


def render_sidebar(frame: pd.DataFrame):
    months = sorted(frame["date"].drop_duplicates().tolist())
    years = sorted({month.year for month in months})
    categories = sorted(frame["category"].unique().tolist())
    month_names = MONTH_LABELS if is_arabic() else ENGLISH_MONTHS
    with st.sidebar:
        st.subheader(tr("time_period"))
        start_year = _select(tr("start_year"), years, "start_year", years[0])
        start_options = [month.month for month in months if month.year == start_year]
        start_month = _select(tr("start_month"), start_options, "start_month", start_options[0],
                              lambda value: month_names[value - 1])
        end_year = _select(tr("end_year"), years, "end_year", years[-1])
        end_options = [month.month for month in months if month.year == end_year]
        end_month = _select(tr("end_month"), end_options, "end_month", end_options[-1],
                            lambda value: month_names[value - 1])
        st.divider()
        st.subheader(tr("materials"))
        if "categories" not in st.session_state:
            st.session_state["categories"] = categories
        st.session_state["categories"] = [c for c in st.session_state["categories"] if c in categories]
        category_names = {value: category_label(value) for value in categories}
        selected_categories = st.multiselect(tr("category"), categories, key="categories",
                                             format_func=category_names.get)
        available = (frame.loc[frame["category"].isin(selected_categories), "material_name"]
                     .drop_duplicates().sort_values().tolist())
        if "materials" not in st.session_state:
            st.session_state["materials"] = available
        st.session_state["materials"] = [m for m in st.session_state["materials"] if m in available]
        material_names = {value: material_label(value) for value in available}
        materials = st.multiselect(tr("material"), available, key="materials", format_func=material_names.get)
        st.caption(tr("coverage", start=display_number(f"{months[0]:%Y-%m}"),
                      end=display_number(f"{months[-1]:%Y-%m}")))
        if st.button(tr("refresh")):
            load_warehouse.clear()
            st.rerun()
    return (materials, selected_categories,
            pd.Timestamp(year=start_year, month=start_month, day=1),
            pd.Timestamp(year=end_year, month=end_month, day=1))


def render_kpis(frame: pd.DataFrame) -> None:
    for index, (column, name) in enumerate(zip(st.columns(4), KPI_NAMES)):
        with column, st.container(key=f"kpi_{index}"):
            rows = frame.loc[frame["material_name"].eq(name)].sort_values("date")
            help_text = tr("steel_help") if name == "Global Steel Index" else None
            if rows.empty:
                st.metric(material_label(name), "—", help=help_text)
                st.caption(tr("excluded"))
                continue
            latest = rows.iloc[-1]
            delta = None if pd.isna(latest["mom_change_pct"]) else (
                f"{latest['mom_change_pct']:+.2f}% {tr('monthly')}")
            st.metric(material_label(name), f"{latest['price']:,.2f}", delta=delta,
                      delta_color="off", help=help_text)
            st.caption(unit_label(name))
            st.caption(display_number(f"{latest['date']:%Y-%m}"))
    st.caption(tr("kpi_note"))


def style_chart(figure: go.Figure, *, y_title: str, height: int = 320) -> go.Figure:
    figure.update_layout(
        template="plotly_white", colorway=PALETTE, height=height,
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family=font_family(), color="#1F2937", size=13),
        margin=dict(l=24, r=24, t=20, b=40),
        legend=dict(title_text="", orientation="h", y=-0.2,
                    x=1 if is_arabic() else 0, xanchor="right" if is_arabic() else "left"),
        hoverlabel=dict(bgcolor="#FFFFFF", font_family=font_family(), align="right" if is_arabic() else "left"),
        xaxis=dict(title=tr("date"), gridcolor="#E5E7EB", zeroline=False),
        yaxis=dict(title=y_title, gridcolor="#E5E7EB", zeroline=False,
                   side="right" if is_arabic() else "left"),
    )
    return figure


def show_chart(figure: go.Figure, key: str) -> None:
    st.plotly_chart(figure, width="stretch", theme=None, key=key,
                    config={"displayModeBar": False, "scrollZoom": False})


def render_trends(frame: pd.DataFrame) -> None:
    indexed = st.toggle(tr("normalized"), key="normalized")
    if indexed:
        values = normalized_index(frame)
        figure = go.Figure()
        for index, name in enumerate(values):
            figure.add_trace(go.Scatter(x=values.index, y=values[name], name=material_label(name), mode="lines",
                                         line=dict(color=PALETTE[index % len(PALETTE)], width=2,
                                                   dash="dash" if index >= len(PALETTE) else "solid"),
                                         hovertemplate="%{x|%Y-%m}<br>%{y:,.2f}<extra>%{fullData.name}</extra>"))
        figure.update_xaxes(tickformat="%Y-%m")
        show_chart(style_chart(figure, y_title=tr("index_axis"), height=440), "normalized_chart")
        st.caption(tr("index_note", date=display_number(f"{values.index[0]:%Y-%m}")))
    else:
        st.caption(tr("raw_note"))
        columns = st.columns(2)
        for index, name in enumerate(MATERIAL_LABELS):
            group = frame.loc[frame["material_name"].eq(name)].sort_values("date")
            if group.empty:
                continue
            with columns[index % 2]:
                st.subheader(material_label(name))
                figure = go.Figure(go.Scatter(
                    x=group["date"], y=group["price"], mode="lines", name=material_label(name),
                    line=dict(color=PALETTE[index % len(PALETTE)], width=2),
                    hovertemplate="%{x|%Y-%m}<br>%{y:,.2f}<extra>%{fullData.name}</extra>"))
                figure.update_xaxes(tickformat="%Y-%m")
                figure.update_layout(showlegend=False)
                show_chart(style_chart(figure, y_title=unit_label(name)), f"price_chart_{index}")


def table_html(frame: pd.DataFrame, *, caption: str, numeric_columns=(), percent_columns=(),
               decimal_places: int = 2) -> str:
    """Semantic HTML supports real RTL headers/cells, including keyboard horizontal scrolling."""
    direction = "rtl" if is_arabic() else "ltr"
    lang = "ar" if is_arabic() else "en"
    header = "".join(f'<th scope="col">{escape(str(column))}</th>' for column in frame.columns)
    rows = []
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for column, value in zip(frame.columns, row):
            number = column in numeric_columns or column in percent_columns
            if pd.isna(value):
                text = "—"
            elif column in percent_columns:
                text = f"{value:+,.2f}%"
            elif number:
                text = f"{value:,.{decimal_places}f}"
            else:
                text = str(value)
            cell_direction = "ltr" if number or column == tr("date") else "auto"
            cells.append(f'<td><bdi dir="{cell_direction}">{escape(text)}</bdi></td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<div class="price-table-wrap" dir="{direction}" lang="{lang}" role="region" tabindex="0" '
            f'aria-label="{escape(caption, quote=True)}"><table class="price-table" dir="{direction}" '
            f'aria-label="{escape(caption, quote=True)}"><thead><tr>{header}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def render_correlation(panel: pd.DataFrame, frame: pd.DataFrame, start, end, metadata: pd.Series) -> None:
    st.caption(tr("correlation_note"))
    basis_labels = {"monthly": tr("monthly_pct"), "levels": tr("levels")}
    basis = st.radio(tr("comparison_basis"), ["monthly", "levels"], horizontal=True,
                     format_func=basis_labels.get,
                     key="comparison_basis")
    exclude_partial = st.checkbox(tr("exclude_partial"), value=True, key="exclude_partial")
    materials = frame["material_name"].drop_duplicates().tolist()
    observations = correlation_observations(
        panel, materials, start, end, monthly_changes=basis == "monthly",
        exclude_partial=exclude_partial, as_of=metadata["as_of"],
        is_partial=bool(metadata["current_month_is_partial"]))
    if len(observations) < 3:
        st.info(tr("few_months"))
        return
    variable = [name for name in observations if observations[name].nunique() > 1]
    if len(variable) < 2:
        st.info(tr("few_materials"))
        return
    if len(variable) < len(observations.columns):
        st.info(tr("constant"))
    observations = observations[variable]
    matrix = observations.corr(method="pearson", min_periods=3)
    st.subheader(tr("correlation_title"))
    display = matrix.rename(index=material_label, columns=material_label)
    numeric_columns = tuple(display.columns)
    display.insert(0, tr("material"), display.index)
    st.html(table_html(display.reset_index(drop=True), caption=tr("correlation_title"),
                       numeric_columns=numeric_columns, decimal_places=3))
    st.caption(tr("correlation_scale", count=f"{len(observations):,}"))
    globals_ = [name for name in variable if name in ("Brent Crude Oil", "Global Steel Index")]
    locals_ = [name for name in variable if name not in ("Brent Crude Oil", "Global Steel Index")]
    if not globals_ or not locals_:
        st.info(tr("scatter_needs_pair"))
        return
    global_name = _select(tr("global"), globals_, "global_material", globals_[0], material_label)
    local_name = _select(tr("local"), locals_, "local_material", locals_[0], material_label)
    x, y = observations[global_name].to_numpy(), observations[local_name].to_numpy()
    figure = go.Figure(go.Scatter(
        x=x, y=y, mode="markers", name=tr("observations"),
        customdata=observations.index.strftime("%Y-%m"),
        marker=dict(color=PALETTE[0], size=7, opacity=0.7),
        hovertemplate="%{customdata}<br>%{x:,.2f} · %{y:,.2f}<extra></extra>"))
    slope = np.mean((x - x.mean()) * (y - y.mean())) / np.var(x)
    trend_x = np.array([x.min(), x.max()])
    figure.add_trace(go.Scatter(x=trend_x, y=y.mean() + slope * (trend_x - x.mean()),
                                mode="lines", name=tr("trend"), line=dict(color=PALETTE[1], width=2)))
    x_title = f"{material_label(global_name)} ({tr('monthly_pct') if basis == 'monthly' else unit_label(global_name)})"
    y_title = f"{material_label(local_name)} ({tr('monthly_pct') if basis == 'monthly' else unit_label(local_name)})"
    style_chart(figure, y_title=y_title, height=400)
    figure.update_xaxes(title=x_title, tickformat=".2f")
    figure.update_yaxes(tickformat=".2f")
    show_chart(figure, "market_scatter")
    st.caption(tr("scatter_note", value=display_number(f"{matrix.loc[global_name, local_name]:+.3f}")))


def business_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Translate a presentation copy. Calculations and source data remain untouched."""
    result = frame[["date", "material_name", "category", "price", "unit", "mom_change_pct", "yoy_change_pct"]].copy()
    result["unit"] = result["material_name"].map(unit_label)
    result["material_name"] = result["material_name"].map(material_label)
    result["category"] = result["category"].map(category_label)
    result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m")
    result = result.rename(columns={
        "date": tr("date"), "material_name": tr("material"), "category": tr("category"),
        "price": tr("price"), "unit": tr("unit"), "mom_change_pct": tr("monthly_pct"),
        "yoy_change_pct": tr("annual_pct"),
    })
    return result


def render_explorer(frame: pd.DataFrame) -> None:
    st.caption(tr("detail_note"))
    controls = st.columns(3)
    with controls[0]:
        order_labels = {value: tr(value) for value in ("newest", "oldest")}
        order = st.selectbox(tr("sort"), ["newest", "oldest"], key="detail_order", format_func=order_labels.get)
    with controls[1]:
        page_size = st.selectbox(tr("page_size"), [25, 50, 100], key="detail_page_size")
    ordered = frame.sort_values(["date", "material_name"], ascending=[order == "oldest", True])
    preview = business_data(ordered)
    pages = max(1, (len(preview) + page_size - 1) // page_size)
    with controls[2]:
        page = _select(tr("page"), list(range(1, pages + 1)), "detail_page", 1)
    offset = (page - 1) * page_size
    st.html(table_html(preview.iloc[offset:offset + page_size], caption=tr("tab_details"),
                       numeric_columns=(tr("price"),),
                       percent_columns=(tr("monthly_pct"), tr("annual_pct"))))
    st.caption(tr("page_note", first=f"{offset + 1:,}", last=f"{min(offset + page_size, len(preview)):,}",
                  total=f"{len(preview):,}", page=page, pages=pages))
    st.download_button(tr("export"), data=preview.to_csv(index=False, float_format="%.2f").encode("utf-8-sig"),
                       file_name="construction_material_prices.csv", mime="text/csv", key="export_csv")
    st.caption(tr("missing_change"))


def main() -> None:
    st.set_page_config(page_title=tr("title"), layout="wide")
    with st.sidebar:
        st.selectbox("اللغة / Language", ["ar", "en"], key="language", format_func=LANGUAGES.get)
    apply_presentation_style()
    st.title(tr("title"))
    st.write(tr("subtitle"))
    try:
        version = DATABASE_PATH.stat()
        data = load_warehouse(str(DATABASE_PATH), (version.st_mtime_ns, version.st_size))
    except (OSError, ValueError, KeyError, duckdb.Error):
        LOGGER.exception("Cannot read the Gold warehouse")
        st.error(tr("unavailable"))
        render_footnote()
        st.stop()
    frame, metadata = data["prices"], data["metadata"].iloc[0]
    materials, categories, start, end = render_sidebar(frame)
    if not categories or not materials:
        st.warning(tr("empty"))
        render_footnote()
        st.stop()
    if start > end:
        st.warning(tr("reversed"))
        render_footnote()
        st.stop()
    filtered = filter_prices(frame, materials, start, end)
    if filtered.empty:
        st.warning(tr("no_results"))
        render_footnote()
        st.stop()
    st.caption(tr("selection", start=display_number(f"{start:%Y-%m}"),
                  end=display_number(f"{end:%Y-%m}"), count=len(materials)))
    st.caption(tr("updated", date=display_number(f"{pd.Timestamp(metadata['latest_ingestion']):%Y-%m-%d}")))
    if metadata["current_month_is_partial"]:
        st.caption(tr("provisional"))
    render_kpis(filtered)
    st.divider()
    history, correlation, explorer = st.tabs([tr("tab_trends"), tr("tab_comparison"), tr("tab_details")])
    with history:
        render_trends(filtered)
    with correlation:
        render_correlation(data["correlation"], filtered, start, end, metadata)
    with explorer:
        render_explorer(filtered)
    render_footnote()


if __name__ == "__main__":
    main()
