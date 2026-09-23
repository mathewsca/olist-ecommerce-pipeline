"""
FabricaIA ETL - Tests for src/etl/db.py configuration resolution.
"""

from src.etl.db import get_engine, load_database_config


class TestLoadDatabaseConfig:
    def test_falls_back_to_yaml_when_no_env_vars(self, monkeypatch):
        for var in ("DW_HOST", "DW_PORT", "DW_NAME", "DW_USER", "DW_PASSWORD"):
            monkeypatch.delenv(var, raising=False)

        config = load_database_config(config_path="config/config.yaml")

        assert config["name"] == "olist_dw"
        assert config["schema_raw"] == "raw"
        assert config["schema_staging"] == "staging"
        assert config["schema_dw"] == "dw"

    def test_env_vars_override_yaml(self, monkeypatch):
        monkeypatch.setenv("DW_HOST", "custom-host")
        monkeypatch.setenv("DW_PORT", "6543")
        monkeypatch.setenv("DW_NAME", "custom_db")
        monkeypatch.setenv("DW_USER", "custom_user")
        monkeypatch.setenv("DW_PASSWORD", "custom_pass")

        config = load_database_config(config_path="config/config.yaml")

        assert config["host"] == "custom-host"
        assert config["port"] == 6543
        assert config["name"] == "custom_db"
        assert config["user"] == "custom_user"
        assert config["password"] == "custom_pass"

    def test_uses_defaults_when_config_file_missing(self, monkeypatch):
        for var in ("DW_HOST", "DW_PORT", "DW_NAME", "DW_USER", "DW_PASSWORD"):
            monkeypatch.delenv(var, raising=False)

        config = load_database_config(config_path="does/not/exist.yaml")

        assert config["host"] == "localhost"
        assert config["port"] == 5432
        assert config["name"] == "olist_dw"


class TestGetEngine:
    def test_builds_postgres_engine_without_connecting(self, monkeypatch):
        monkeypatch.setenv("DW_HOST", "somehost")
        monkeypatch.setenv("DW_PORT", "5432")
        monkeypatch.setenv("DW_NAME", "olist_dw")
        monkeypatch.setenv("DW_USER", "airflow")
        monkeypatch.setenv("DW_PASSWORD", "airflow")

        engine = get_engine(config_path="config/config.yaml")

        assert engine.url.host == "somehost"
        assert engine.url.database == "olist_dw"
        assert engine.url.drivername == "postgresql+psycopg2"
