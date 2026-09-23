-- Olist ETL teaching project: raw layer.
-- Tables here are near-verbatim copies of the Kaggle CSVs, created by
-- src/etl/raw_loader.py via pandas.to_sql (if_exists="replace"), so no
-- explicit CREATE TABLE is needed here beyond the schema itself.
CREATE SCHEMA IF NOT EXISTS raw;
