-- Gold star schema counts.
SELECT 'fact_prices' AS table_name, count(*) AS row_count FROM fact_prices
UNION ALL SELECT 'dim_material', count(*) FROM dim_material
UNION ALL SELECT 'dim_date', count(*) FROM dim_date;

-- Latest available price and calendar-based changes for each material.
SELECT f.date, m.material_name, m.category, m.unit, m.currency, m.source,
       f.price, f.mom_change_pct, f.yoy_change_pct
FROM fact_prices f JOIN dim_material m USING (material_id)
QUALIFY row_number() OVER (PARTITION BY f.material_id ORDER BY f.date DESC) = 1
ORDER BY f.material_id;

-- Full monthly calendar and provenance remain available through dimensions.
SELECT d.year, d.month, d.month_name, m.material_name, f.price
FROM fact_prices f
JOIN dim_date d ON d.date = f.date
JOIN dim_material m USING (material_id)
ORDER BY f.date DESC, f.material_id LIMIT 15;

-- Monthly percentage changes avoid comparing unrelated price scales.
WITH changes AS (
    SELECT date,
           100.0 * (brent_oil_price / lag(brent_oil_price) OVER w - 1) AS brent_mom,
           100.0 * (steel_index_price / lag(steel_index_price) OVER w - 1) AS steel_mom,
           100.0 * (reinforcing_rebar_price / lag(reinforcing_rebar_price) OVER w - 1) AS rebar_mom
    FROM vw_global_local_correlation WINDOW w AS (ORDER BY date)
)
SELECT count(*) FILTER (WHERE brent_mom IS NOT NULL AND rebar_mom IS NOT NULL) AS paired_months,
       corr(brent_mom, rebar_mom) AS brent_rebar_mom_pearson_r,
       corr(steel_mom, rebar_mom) AS steel_rebar_mom_pearson_r
FROM changes;

SELECT * FROM vw_global_local_correlation ORDER BY date DESC LIMIT 5;
