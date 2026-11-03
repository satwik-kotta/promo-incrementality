-- Product x week panel for the products this study covers.
-- Units, gross revenue (before loyalty discount) and loyalty discount dollars come from
-- panel household transactions. Promotion flags come from the weekly mailer and in store
-- display records. Weeks with no sales are kept as zero rows so the baseline is not biased upward.

CREATE OR REPLACE TABLE products_in_scope AS
WITH volume AS (
    SELECT t.product_id,
           SUM(t.quantity)                AS units,
           COUNT(DISTINCT t.week)         AS weeks_sold,
           COUNT(DISTINCT t.household_id) AS households
    FROM raw_transactions t
    JOIN raw_products p USING (product_id)
    WHERE t.week BETWEEN 2 AND 52
      AND p.department NOT IN ('FUEL', 'MISCELLANEOUS', 'KIOSK-GAS', 'MISC SALES TRAN')
      AND COALESCE(p.product_category, '') NOT IN ('COUPON/MISC ITEMS', 'FUEL')
      AND t.quantity BETWEEN 1 AND 50
    GROUP BY 1
)
SELECT v.*, p.department, p.product_category, p.product_type, p.brand
FROM volume v
JOIN raw_products p USING (product_id)
WHERE v.units >= 200 AND v.weeks_sold >= 45 AND v.households >= 30;

CREATE OR REPLACE TABLE weekly_sales AS
SELECT t.product_id, t.week,
       SUM(t.quantity)                       AS units,
       SUM(t.sales_value + t.retail_disc)    AS gross_revenue,
       SUM(t.retail_disc)                    AS loyalty_discount,
       COUNT(DISTINCT t.household_id)        AS buyers
FROM raw_transactions t
WHERE t.product_id IN (SELECT product_id FROM products_in_scope)
  AND t.week BETWEEN 2 AND 52
  AND t.quantity BETWEEN 1 AND 50
GROUP BY 1, 2;

CREATE OR REPLACE TABLE weekly_promo AS
SELECT product_id, week,
       MAX(CASE WHEN mailer_location  <> '0' THEN 1 ELSE 0 END) AS mailer,
       AVG(CASE WHEN display_location <> '0' THEN 1 ELSE 0 END) AS display_share
FROM raw_promotions
WHERE product_id IN (SELECT product_id FROM products_in_scope)
  AND week BETWEEN 2 AND 52
GROUP BY 1, 2;

CREATE OR REPLACE TABLE panel AS
WITH grid AS (
    SELECT p.product_id, w.week
    FROM products_in_scope p
    CROSS JOIN (SELECT UNNEST(RANGE(2, 53)) AS week) w
)
SELECT g.product_id, g.week,
       p.department, p.product_category,
       COALESCE(s.units, 0)            AS units,
       COALESCE(s.gross_revenue, 0)    AS gross_revenue,
       COALESCE(s.loyalty_discount, 0) AS loyalty_discount,
       COALESCE(s.buyers, 0)           AS buyers,
       COALESCE(m.mailer, 0)           AS mailer,
       COALESCE(m.display_share, 0)    AS display_share
FROM grid g
JOIN products_in_scope p USING (product_id)
LEFT JOIN weekly_sales s USING (product_id, week)
LEFT JOIN weekly_promo m USING (product_id, week);
