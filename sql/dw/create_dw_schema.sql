-- Olist ETL project: star-schema Data Warehouse.
-- Dimensions use surrogate keys (IDENTITY) with the Olist natural key kept
-- as an attribute for lookups. Facts are split by grain:
--   fact_order_items : one row per order item   (transaction grain)
--   fact_orders      : one row per order          (order grain)
--   fact_payments    : one row per payment record
--   fact_reviews     : one row per review (a review_id can be attached to several orders)
-- Table bodies are (re)created here; src/etl/dw_builder.py populates them
-- from the cleaned staging.* tables.
--
-- Besides the business columns, tables carry data-quality flags (is_*, has_*,
-- *_imputed, zip_*): the cleaning layer flags suspicious rows instead of dropping
-- them, so analyses can include or exclude them explicitly.

CREATE SCHEMA IF NOT EXISTS dw;

-- ---------------------------------------------------------------------
-- Dimensions
-- ---------------------------------------------------------------------

DROP TABLE IF EXISTS dw.dim_customers CASCADE;
CREATE TABLE dw.dim_customers (
    customer_key INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id VARCHAR(64) UNIQUE NOT NULL,
    customer_unique_id VARCHAR(64),
    customer_zip_code_prefix VARCHAR(5),
    customer_city VARCHAR(128),
    customer_state VARCHAR(2),
    customer_region VARCHAR(16),
    zip_in_geolocation BOOLEAN,        -- CEP found in dim_geolocation (can be plotted)
    zip_state_mismatch BOOLEAN         -- UF disagrees with the CEP range (Correios)
);

DROP TABLE IF EXISTS dw.dim_sellers CASCADE;
CREATE TABLE dw.dim_sellers (
    seller_key INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    seller_id VARCHAR(64) UNIQUE NOT NULL,
    seller_zip_code_prefix VARCHAR(5),
    seller_city VARCHAR(128),
    seller_state VARCHAR(2),
    seller_region VARCHAR(16),
    zip_in_geolocation BOOLEAN,
    zip_state_mismatch BOOLEAN
);

DROP TABLE IF EXISTS dw.dim_products CASCADE;
CREATE TABLE dw.dim_products (
    product_key INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    product_id VARCHAR(64) UNIQUE NOT NULL,
    product_category_name VARCHAR(128),
    product_category_name_english VARCHAR(128),
    category_group VARCHAR(64),        -- ~14 business macro-groups over the 73 categories
    product_weight_g NUMERIC,
    product_length_cm NUMERIC,
    product_height_cm NUMERIC,
    product_width_cm NUMERIC,
    product_volume_cm3 NUMERIC,
    product_photos_qty NUMERIC,
    product_name_length NUMERIC,
    product_description_length NUMERIC,
    has_listing_metadata BOOLEAN,      -- FALSE when photos/title/description info is missing
    dimensions_imputed BOOLEAN         -- weight/size filled with the category median
);

DROP TABLE IF EXISTS dw.dim_geolocation CASCADE;
CREATE TABLE dw.dim_geolocation (
    geo_key INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    zip_code_prefix VARCHAR(5) UNIQUE NOT NULL,
    city VARCHAR(128),
    state VARCHAR(2),
    region VARCHAR(16),
    avg_lat NUMERIC,
    avg_lng NUMERIC,
    n_points INT,                      -- raw GPS points kept in the average
    n_points_discarded INT             -- raw GPS points discarded as outliers
);

DROP TABLE IF EXISTS dw.dim_date CASCADE;
CREATE TABLE dw.dim_date (
    date_key INT PRIMARY KEY,          -- YYYYMMDD
    full_date DATE UNIQUE NOT NULL,
    year INT NOT NULL,
    quarter INT NOT NULL,
    month INT NOT NULL,
    day INT NOT NULL,
    day_of_week INT NOT NULL,          -- 0=Monday .. 6=Sunday
    is_weekend BOOLEAN NOT NULL,
    year_month VARCHAR(7) NOT NULL,    -- 'YYYY-MM', sortable label
    month_name VARCHAR(12) NOT NULL,   -- pt-BR
    day_name VARCHAR(12) NOT NULL,     -- pt-BR
    week_of_year INT NOT NULL
);

-- ---------------------------------------------------------------------
-- Facts
-- ---------------------------------------------------------------------

DROP TABLE IF EXISTS dw.fact_order_items CASCADE;
CREATE TABLE dw.fact_order_items (
    order_item_key INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    order_id VARCHAR(64) NOT NULL,
    order_item_id INT NOT NULL,
    customer_key INT REFERENCES dw.dim_customers (customer_key),
    seller_key INT REFERENCES dw.dim_sellers (seller_key),
    product_key INT REFERENCES dw.dim_products (product_key),
    order_purchase_date_key INT REFERENCES dw.dim_date (date_key),
    price NUMERIC,
    freight_value NUMERIC,
    is_price_outlier BOOLEAN,          -- price above Q3 + 3*IQR
    is_freight_above_price BOOLEAN,
    UNIQUE (order_id, order_item_id)
);

DROP TABLE IF EXISTS dw.fact_orders CASCADE;
CREATE TABLE dw.fact_orders (
    order_key INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    order_id VARCHAR(64) UNIQUE NOT NULL,
    customer_key INT REFERENCES dw.dim_customers (customer_key),
    order_status VARCHAR(32),
    order_purchase_date_key INT REFERENCES dw.dim_date (date_key),
    order_delivered_date_key INT REFERENCES dw.dim_date (date_key),
    order_estimated_delivery_date_key INT REFERENCES dw.dim_date (date_key),
    purchase_hour SMALLINT,            -- 0-23
    approval_time_hours NUMERIC,       -- purchase -> payment approval
    carrier_handoff_days NUMERIC,      -- purchase -> handed to carrier
    delivery_time_days NUMERIC,        -- purchase -> delivered (delivered orders only)
    delivery_delay_days NUMERIC,       -- delivered date - estimated date, in days (>0 = late)
    is_late BOOLEAN,                   -- delivered after the estimated date
    has_date_anomaly BOOLEAN,          -- impossible/out-of-order timestamps (see staging.dq_issues)
    items_count INT,
    items_value NUMERIC,
    freight_value NUMERIC
);

DROP TABLE IF EXISTS dw.fact_payments CASCADE;
CREATE TABLE dw.fact_payments (
    payment_key INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    order_id VARCHAR(64) NOT NULL,
    payment_sequential INT,
    payment_type VARCHAR(32),
    payment_installments INT,
    payment_value NUMERIC,
    is_valid_payment BOOLEAN           -- FALSE for 'not_defined' type / missing value
);

DROP TABLE IF EXISTS dw.fact_reviews CASCADE;
CREATE TABLE dw.fact_reviews (
    review_key INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    review_id VARCHAR(64) NOT NULL,
    order_id VARCHAR(64) NOT NULL,
    review_score INT,
    review_creation_date_key INT REFERENCES dw.dim_date (date_key),
    review_answer_date_key INT REFERENCES dw.dim_date (date_key),
    has_comment_title BOOLEAN,
    has_comment_message BOOLEAN,
    review_comment_title TEXT,         -- cleaned free text (case/accents preserved)
    review_comment_message TEXT,
    comment_length INT,
    is_low_information BOOLEAN,        -- 'ok', '.', '10'...
    is_shouting BOOLEAN,               -- all caps
    is_latest_for_order BOOLEAN,       -- most recent review of the order
    response_time_hours NUMERIC,       -- review creation -> answer
    UNIQUE (review_id, order_id)       -- review_id alone is NOT unique in Olist
);
