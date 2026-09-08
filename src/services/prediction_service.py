"""
FabricaIA - Prediction Service

Encapsulates instance lookup, data treatment (train-serve consistency),
and model inference to avoid training-serving skew.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from src.data.processor import DataProcessor
from src.features.engineering import FeatureEngineer
from src.models.trainer import ModelTrainer

logger = logging.getLogger(__name__)


class PredictionService:
    """Service to handle lookup, consistent feature preprocessing, and model inference."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.data_processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer(config_path=config_path)
        self._raw_cache: Optional[pd.DataFrame] = None

    def _load_config(self, config_path: str) -> dict:
        if Path(config_path).exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        return {}

    def get_raw_dataset(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """Load and cache reference dataset for lookup operations."""
        if self._raw_cache is not None and data_path is None:
            return self._raw_cache

        target_path = data_path or "data/raw/obras_publicas.csv"
        if Path(target_path).exists():
            self._raw_cache = pd.read_csv(target_path)
            return self._raw_cache

        # Fallback to searching data/raw/
        raw_dir = Path("data/raw")
        if raw_dir.exists():
            for f in raw_dir.glob("*.csv"):
                self._raw_cache = pd.read_csv(f)
                return self._raw_cache

        raise FileNotFoundError(f"No reference dataset found for lookup at {target_path}")

    def lookup_instance(
        self, lookup_id: str, id_column: str = "id_obra", data_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Lookup a data instance by its unique identifier.

        Args:
            lookup_id: Unique identifier value (e.g. 'OBR-2024-0001')
            id_column: Name of the identifier column
            data_path: Optional path to specific dataset

        Returns:
            Dictionary containing raw feature attributes of the instance
        """
        df = self.get_raw_dataset(data_path)

        if id_column not in df.columns:
            # Fallback: check index if id_column is not explicitly present
            if lookup_id.isdigit() and int(lookup_id) < len(df):
                row = df.iloc[int(lookup_id)].to_dict()
                row.pop("atraso_risco", None)
                return row
            raise KeyError(f"Identifier column '{id_column}' not found in reference data.")

        matched = df[df[id_column].astype(str) == str(lookup_id)]
        if matched.empty:
            raise ValueError(f"Instance with {id_column}='{lookup_id}' not found.")

        row = matched.iloc[0].to_dict()
        # Remove target if present to prevent leakage during inference
        row.pop("atraso_risco", None)
        row.pop("target", None)
        return row

    def preprocess_instance(
        self, raw_features: Dict[str, Any], model: Any
    ) -> pd.DataFrame:
        """
        Transform raw input features using identical treatment as training pipeline
        to guarantee train-serve consistency (preventing training-serving skew).

        Args:
            raw_features: Dictionary of raw feature attributes
            model: Trained scikit-learn model with expected feature_names_in_

        Returns:
            Processed single-row DataFrame ready for model inference
        """
        # Convert dictionary to single-row DataFrame
        df = pd.DataFrame([raw_features])

        # Remove identifier columns if present
        for id_col in ["id_obra", "id", "contract_id"]:
            if id_col in df.columns:
                df = df.drop(columns=[id_col])

        # 1. Clean data using DataProcessor
        df_clean = self.data_processor.clean_data(df, drop_duplicates=False, handle_missing="fill")

        # 2. Categorical encoding
        cat_cols = df_clean.select_dtypes(include=["object", "category"]).columns.tolist()
        df_encoded = self.data_processor.encode_categorical(df_clean, columns=cat_cols if cat_cols else None, method="label")

        # 3. Scale numeric features
        num_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
        df_scaled = self.data_processor.scale_features(df_encoded, columns=num_cols if num_cols else None)

        # 4. Feature engineering (polynomial and interaction terms)
        numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        if len(numerical_cols) >= 2:
            poly_deg = self.config.get("FEATURES", {}).get("polynomial_degree", 2)
            df_poly = self.feature_engineer.create_polynomial_features(
                df_scaled, numerical_cols[:2], degree=poly_deg
            )
            df_features = self.feature_engineer.create_interaction_features(
                df_poly, [(numerical_cols[0], numerical_cols[1])]
            )
        else:
            df_features = df_scaled

        # 5. Align columns strictly with what the trained model expects
        expected_features = getattr(model, "feature_names_in_", None)
        if expected_features is not None:
            aligned_df = pd.DataFrame(index=df_features.index)
            for feat in expected_features:
                if feat in df_features.columns:
                    aligned_df[feat] = df_features[feat]
                else:
                    # Impute missing engineered feature with 0.0
                    aligned_df[feat] = 0.0
            return aligned_df

        return df_features

    def predict_instance(
        self,
        model: Any,
        model_name: str,
        raw_features: Optional[Dict[str, Any]] = None,
        lookup_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute end-to-end lookup, transformation, and prediction.

        Args:
            model: Trained model instance
            model_name: Name of the model
            raw_features: Optional dictionary of raw attributes
            lookup_id: Optional identifier for lookup

        Returns:
            Dictionary with prediction, probabilities, confidence, and metadata
        """
        source_id = None
        if lookup_id:
            raw_data = self.lookup_instance(lookup_id)
            source_id = lookup_id
        elif raw_features:
            raw_data = raw_features.copy()
            source_id = raw_data.get("id_obra")
        else:
            raise ValueError("Either 'features' or 'lookup_id' must be provided.")

        # Transform features through consistent service pipeline
        processed_df = self.preprocess_instance(raw_data, model)

        # Predict
        prediction = self.model_trainer.predict(model, processed_df)

        # Probabilities
        probability = None
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = self.model_trainer.predict_proba(model, processed_df)
            classes = getattr(model, "classes_", None)
            if classes is not None:
                probability = {
                    str(classes[i]): float(proba[0][i]) for i in range(len(classes))
                }
                confidence = max(probability.values())

        pred_val = prediction[0].item() if len(prediction) == 1 else prediction.tolist()

        return {
            "prediction": pred_val,
            "probability": probability,
            "confidence": confidence,
            "model_name": model_name,
            "lookup_id": source_id,
            "raw_input_summary": {k: v for k, v in raw_data.items() if k not in ["features"]},
        }
