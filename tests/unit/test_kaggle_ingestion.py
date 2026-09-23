"""
FabricaIA ETL - Tests for KaggleIngestion (mocks the kaggle package; never
hits the network).
"""

import sys
from unittest.mock import MagicMock

import pytest

from src.etl.kaggle_ingestion import EXPECTED_CSV_FILES, MIN_ROWS, KaggleIngestion


class TestKaggleIngestion:
    def setup_method(self):
        self.ingestion = KaggleIngestion()

    def _touch_all_expected_files(self, dest_dir):
        for name in EXPECTED_CSV_FILES:
            (dest_dir / name).write_text("col1,col2\n" + "1,2\n" * MIN_ROWS[name])

    def _write_sample_files(self, dest_dir):
        for name in EXPECTED_CSV_FILES:
            (dest_dir / name).write_text("col1,col2\n1,2\n")

    def test_skips_download_when_all_files_present(self, tmp_path, monkeypatch):
        self._touch_all_expected_files(tmp_path)

        # If the kaggle package were imported/used, this would fail loudly.
        monkeypatch.setitem(sys.modules, "kaggle", None)

        files = self.ingestion.download_dataset(dest_dir=str(tmp_path))

        assert files == EXPECTED_CSV_FILES

    def test_raises_when_credentials_missing_and_files_absent(self, tmp_path, monkeypatch):
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)
        monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)

        with pytest.raises(RuntimeError, match="KAGGLE_USERNAME"):
            self.ingestion.download_dataset(dest_dir=str(tmp_path))

    def test_downloads_via_kaggle_api_when_credentials_present(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "test_user")
        monkeypatch.setenv("KAGGLE_KEY", "test_key")

        mock_api = MagicMock()

        def fake_download(dataset_slug, path, unzip):
            self._touch_all_expected_files(tmp_path)

        mock_api.dataset_download_files.side_effect = fake_download

        mock_kaggle_module = MagicMock()
        mock_kaggle_module.api.kaggle_api_extended.KaggleApi.return_value = mock_api
        monkeypatch.setitem(sys.modules, "kaggle", MagicMock())
        monkeypatch.setitem(sys.modules, "kaggle.api", MagicMock())
        monkeypatch.setitem(
            sys.modules,
            "kaggle.api.kaggle_api_extended",
            MagicMock(KaggleApi=lambda: mock_api),
        )

        files = self.ingestion.download_dataset(dest_dir=str(tmp_path))

        assert files == EXPECTED_CSV_FILES
        mock_api.authenticate.assert_called_once()
        mock_api.dataset_download_files.assert_called_once()

    def test_redownloads_when_existing_files_are_truncated_samples(self, tmp_path, monkeypatch):
        self._write_sample_files(tmp_path)
        monkeypatch.setenv("KAGGLE_USERNAME", "test_user")
        monkeypatch.setenv("KAGGLE_KEY", "test_key")

        mock_api = MagicMock()
        mock_api.dataset_download_files.side_effect = lambda *a, **k: self._touch_all_expected_files(tmp_path)

        monkeypatch.setitem(sys.modules, "kaggle", MagicMock())
        monkeypatch.setitem(sys.modules, "kaggle.api", MagicMock())
        monkeypatch.setitem(
            sys.modules,
            "kaggle.api.kaggle_api_extended",
            MagicMock(KaggleApi=lambda: mock_api),
        )

        self.ingestion.download_dataset(dest_dir=str(tmp_path))

        mock_api.dataset_download_files.assert_called_once()
