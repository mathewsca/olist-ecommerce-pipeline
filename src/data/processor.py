"""
FabricaIA - Data Processing Module

This module contains reusable functions for data preprocessing,
feature engineering, and data validation.
"""

import logging
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class DataProcessor:
    """Main class for data processing operations."""

    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = None
        self.target_column = None

    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from various file formats.

        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments for pandas read functions

        Returns:
            Loaded DataFrame
        """
        try:
            if file_path.endswith(".csv"):
                return pd.read_csv(file_path, **kwargs)
            elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
                return pd.read_excel(file_path, **kwargs)
            elif file_path.endswith(".json"):
                return pd.read_json(file_path, **kwargs)
            elif file_path.endswith(".parquet"):
                return pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise

    def clean_data(
        self,
        df: pd.DataFrame,
        drop_duplicates: bool = True,
        handle_missing: str = "drop",
    ) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and duplicates.

        Args:
            df: Input DataFrame
            drop_duplicates: Whether to drop duplicate rows
            handle_missing: Strategy for handling missing values
                ('drop', 'fill', 'interpolate')

        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()

        if drop_duplicates:
            df_clean = df_clean.drop_duplicates()

        if handle_missing == "drop":
            df_clean = df_clean.dropna()
        elif handle_missing == "fill":
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            categorical_cols = df_clean.select_dtypes(include=["object"]).columns

            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(
                df_clean[numeric_cols].mean()
            )
            df_clean[categorical_cols] = df_clean[categorical_cols].fillna(
                df_clean[categorical_cols].mode().iloc[0]
            )
        elif handle_missing == "interpolate":
            df_clean = df_clean.interpolate()

        logger.info(f"Data cleaned. Shape: {df_clean.shape}")
        return df_clean

    def encode_categorical(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "label",
    ) -> pd.DataFrame:
        """
        Encode categorical variables.

        Args:
            df: Input DataFrame
            columns: List of columns to encode (if None, auto-detect)
            method: Encoding method ('label', 'onehot')

        Returns:
            DataFrame with encoded categorical variables
        """
        df_encoded = df.copy()

        if columns is None:
            columns = df_encoded.select_dtypes(include=["object"]).columns.tolist()

        for col in columns:
            if method == "label":
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = encoder
            elif method == "onehot":
                dummies = pd.get_dummies(df_encoded[col], prefix=col)
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)

        logger.info(f"Categorical encoding completed for columns: {columns}")
        return df_encoded

    def scale_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        scaler_type: str = "standard",
        exclude_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Scale numerical features.

        Args:
            df: Input DataFrame
            columns: List of columns to scale (if None, auto-detect numeric)
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')

        Returns:
            DataFrame with scaled features
        """
        df_scaled = df.copy()

        if columns is None:
            columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude target-like columns from scaling by default
        default_excludes = {self.target_column} if self.target_column else set()
        default_excludes.update({"target", "label", "y", "class"})
        exclude_set = set(exclude_columns or []) | {
            c for c in default_excludes if c in df_scaled.columns
        }
        columns = [c for c in columns if c not in exclude_set]

        if scaler_type == "standard":
            scaler = StandardScaler()
        else:
            scaler = StandardScaler()  # Default to StandardScaler

        if columns:
            df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
        self.scalers["features"] = scaler

        logger.info(f"Feature scaling completed for columns: {columns}")
        return df_scaled

    def split_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple:
        """
        Split data into train and test sets.

        Args:
            df: Input DataFrame
            target_column: Name of the target column
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.target_column = target_column
        feature_columns = [col for col in df.columns if col != target_column]
        self.feature_columns = feature_columns

        X = df[feature_columns]
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info(
            f"Data split completed. Train: {X_train.shape}, Test: {X_test.shape}"
        )
        return X_train, X_test, y_train, y_test

    def get_feature_importance(
        self, model: Any, feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract feature importance from trained model.

        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names

        Returns:
            DataFrame with feature importance
        """
        if feature_names is None:
            feature_names = self.feature_columns

        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        return importance_df
