#!/usr/bin/env bash
# Creates the olist_dw database (separate from the "airflow" metadata database)
# the first time the postgres container initializes its data volume.
# Runs automatically via /docker-entrypoint-initdb.d/ — see docker-compose.yml.
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    SELECT 'CREATE DATABASE olist_dw OWNER $POSTGRES_USER'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'olist_dw')\gexec
EOSQL

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname olist_dw <<-EOSQL
    CREATE SCHEMA IF NOT EXISTS raw;
    CREATE SCHEMA IF NOT EXISTS staging;
    CREATE SCHEMA IF NOT EXISTS dw;
EOSQL
