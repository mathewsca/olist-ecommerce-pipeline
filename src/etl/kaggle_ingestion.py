"""
FabricaIA - Kaggle Ingestion Module

Downloads the Brazilian E-Commerce Public Dataset by Olist from Kaggle into
data/raw/. Idempotent: if the expected CSVs are already present and look
complete (e.g. a student placed them there manually), the download is skipped
— this is what lets the same data/raw/ folder serve both the automated
pipeline and manual student inspection. Truncated/sample CSVs (far fewer rows
than the real dataset) are treated as incomplete and trigger a full download.
"""

import logging
import os
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

KAGGLE_DATASET_SLUG = "olistbr/brazilian-ecommerce"

EXPECTED_CSV_FILES: List[str] = [
    "olist_customers_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "product_category_name_translation.csv",
]

# Conservative lower bounds on line counts (header included) for the full
# dataset (real sizes: ~99k orders/customers, ~1M geolocation, 71 categories...).
# A CSV below its threshold is a sample/stub, not the real data.
MIN_ROWS = {
    "olist_customers_dataset.csv": 90_000,
    "olist_orders_dataset.csv": 90_000,
    "olist_order_items_dataset.csv": 100_000,
    "olist_order_payments_dataset.csv": 100_000,
    "olist_order_reviews_dataset.csv": 90_000,
    "olist_products_dataset.csv": 30_000,
    "olist_sellers_dataset.csv": 3_000,
    "olist_geolocation_dataset.csv": 900_000,
    "product_category_name_translation.csv": 70,
}


def _count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


class KaggleIngestion:
    """Downloads the Olist dataset from Kaggle into a local raw data directory."""

    def __init__(self, dataset_slug: str = KAGGLE_DATASET_SLUG):
        self.dataset_slug = dataset_slug

    def _missing_files(self, dest_dir: Path) -> List[str]:
        return [name for name in EXPECTED_CSV_FILES if not (dest_dir / name).exists()]

    def _incomplete_files(self, dest_dir: Path) -> List[str]:
        """Files that exist but have suspiciously few rows (sample/stub data)."""
        return [
            name
            for name in EXPECTED_CSV_FILES
            if (dest_dir / name).exists() and _count_lines(dest_dir / name) < MIN_ROWS[name]
        ]

    def download_dataset(self, dest_dir: str = "data/raw", force: bool = False) -> List[str]:
        """
        Download and unzip the Olist dataset from Kaggle into dest_dir.

        Skips the download when all expected CSVs are already present and
        complete (not truncated samples), unless force=True. Requires KAGGLE_API_TOKEN or KAGGLE_USERNAME/KAGGLE_KEY
        environment variables (or ~/.kaggle/kaggle.json) to be configured.

        Args:
            dest_dir: Directory the CSVs should end up in (created if missing).
            force: Re-download even if the CSVs already exist.

        Returns:
            The list of CSV filenames now present in dest_dir.
        """
        dest_path = Path(dest_dir)
        dest_path.mkdir(parents=True, exist_ok=True)

        missing = self._missing_files(dest_path)
        incomplete = self._incomplete_files(dest_path)
        if not force and not missing and not incomplete:
            logger.info("All Olist CSVs already present in %s, skipping Kaggle download.", dest_dir)
            return EXPECTED_CSV_FILES
        if incomplete:
            logger.warning(
                "CSVs in %s look truncated (sample data), re-downloading full dataset: %s",
                dest_dir,
                incomplete,
            )

        has_legacy_creds = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
        if not has_legacy_creds and not os.environ.get("KAGGLE_API_TOKEN"):
            raise RuntimeError(
                "KAGGLE_USERNAME/KAGGLE_KEY (or KAGGLE_API_TOKEN) are not set and the following CSVs are "
                f"missing or incomplete in {dest_dir}: {missing + incomplete}. Either configure Kaggle API "
                "credentials (see .env.example) or place the CSVs manually."
            )

        # Imported lazily so importing this module never requires the kaggle
        # package (and its credential check at import time) unless a download
        # is actually needed.
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        logger.info("Downloading dataset %s into %s ...", self.dataset_slug, dest_dir)
        api.dataset_download_files(self.dataset_slug, path=str(dest_path), unzip=True)

        still_missing = self._missing_files(dest_path)
        if still_missing:
            raise RuntimeError(f"Download completed but CSVs are still missing: {still_missing}")

        logger.info("Olist dataset ready in %s.", dest_dir)
        return EXPECTED_CSV_FILES
