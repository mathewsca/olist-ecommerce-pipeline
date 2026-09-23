#!/usr/bin/env python3
"""
Baixa o Brazilian E-Commerce Public Dataset by Olist (Kaggle) para data/raw/.
Usado pela DAG do Airflow e disponível também via `make ingest` para uso manual.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.etl.kaggle_ingestion import KaggleIngestion  # noqa: E402

logging.basicConfig(level=logging.INFO)


def main():
    ingestion = KaggleIngestion()
    files = ingestion.download_dataset(dest_dir="data/raw")
    print(f"CSVs disponiveis em data/raw/: {len(files)}")
    for name in files:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
