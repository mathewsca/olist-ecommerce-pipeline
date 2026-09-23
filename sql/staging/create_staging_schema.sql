-- Olist ETL teaching project: staging layer.
-- Typed, deduplicated tables built from raw.* by src/etl/staging_transform.py
-- via pandas.to_sql (if_exists="replace"). Only the schema is created here;
-- table shapes are defined by the DataFrames produced in StagingTransformer.
CREATE SCHEMA IF NOT EXISTS staging;
