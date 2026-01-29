"""
FabricaIA - Feature Engineering Module

This module contains reusable functions for feature engineering,
feature selection, and feature transformation.
"""

import logging
from typing import List, Tuple

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, f_classif, f_regression
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Main class for feature engineering operations."""

    def __init__(self):
        self.feature_selectors = {}
        self.transformers = {}
        self.feature_names = None

    def create_polynomial_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        degree: int = 2,
        include_bias: bool = False,
    ) -> pd.DataFrame:
        """
        Create polynomial features for specified columns.

        Args:
            df: Input DataFrame
            columns: Columns to create polynomial features for
            degree: Degree of polynomial features
            include_bias: Whether to include bias term

        Returns:
            DataFrame with polynomial features
        """
        df_poly = df.copy()

        for col in columns:
            poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
            poly_features = poly.fit_transform(df[[col]])

            # Create column names for polynomial features
            feature_names = [f"{col}_poly_{i}" for i in range(poly_features.shape[1])]

            # Add polynomial features to DataFrame
            poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
            df_poly = pd.concat([df_poly, poly_df], axis=1)

        logger.info(f"Polynomial features created for columns: {columns}")
        return df_poly

    def create_interaction_features(
        self, df: pd.DataFrame, feature_pairs: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """
        Create interaction features between specified feature pairs.

        Args:
            df: Input DataFrame
            feature_pairs: List of tuples with feature pairs to interact

        Returns:
            DataFrame with interaction features
        """
        df_interact = df.copy()

        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                interaction_name = f"{feat1}_x_{feat2}"
                df_interact[interaction_name] = df[feat1] * df[feat2]

        logger.info(f"Interaction features created for {len(feature_pairs)} pairs")
        return df_interact

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        window_sizes: List[int],
        operations: List[str] = ["mean", "std"],
    ) -> pd.DataFrame:
        """
        Create rolling window features.

        Args:
            df: Input DataFrame
            columns: Columns to create rolling features for
            window_sizes: List of window sizes
            operations: List of operations to apply

        Returns:
            DataFrame with rolling features
        """
        df_rolling = df.copy()

        for col in columns:
            if col in df.columns:
                for window in window_sizes:
                    for op in operations:
                        feature_name = f"{col}_rolling_{window}_{op}"
                        if op == "mean":
                            df_rolling[feature_name] = (
                                df[col].rolling(window=window).mean()
                            )
                        elif op == "std":
                            df_rolling[feature_name] = (
                                df[col].rolling(window=window).std()
                            )
                        elif op == "min":
                            df_rolling[feature_name] = (
                                df[col].rolling(window=window).min()
                            )
                        elif op == "max":
                            df_rolling[feature_name] = (
                                df[col].rolling(window=window).max()
                            )

        logger.info(f"Rolling features created for columns: {columns}")
        return df_rolling

    def create_lag_features(
        self, df: pd.DataFrame, columns: List[str], lag_periods: List[int]
    ) -> pd.DataFrame:
        """
        Create lag features for time series data.

        Args:
            df: Input DataFrame
            columns: Columns to create lag features for
            lag_periods: List of lag periods

        Returns:
            DataFrame with lag features
        """
        df_lag = df.copy()

        for col in columns:
            if col in df.columns:
                for lag in lag_periods:
                    feature_name = f"{col}_lag_{lag}"
                    df_lag[feature_name] = df[col].shift(lag)

        logger.info(f"Lag features created for columns: {columns}")
        return df_lag

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "univariate",
        k: int = 10,
        model=None,
    ) -> pd.DataFrame:
        """
        Select best features using various methods.

        Args:
            X: Feature DataFrame
            y: Target Series
            method: Selection method ('univariate', 'rfe', 'pca')
            k: Number of features to select
            model: Model for RFE (required if method='rfe')

        Returns:
            DataFrame with selected features
        """
        if method == "univariate":
            # Use F-test for feature selection
            if len(y.unique()) > 2:  # Classification
                selector = SelectKBest(score_func=f_classif, k=k)
            else:  # Regression
                selector = SelectKBest(score_func=f_regression, k=k)

            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            self.feature_selectors["univariate"] = selector

        elif method == "rfe":
            if model is None:
                raise ValueError("Model required for RFE method")

            selector = RFE(estimator=model, n_features_to_select=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            self.feature_selectors["rfe"] = selector

        elif method == "pca":
            selector = PCA(n_components=k)
            X_selected = selector.fit_transform(X)
            selected_features = [f"PC_{i+1}" for i in range(k)]
            self.feature_selectors["pca"] = selector

        else:
            raise ValueError(f"Unsupported selection method: {method}")

        # Create DataFrame with selected features
        X_selected_df = pd.DataFrame(
            X_selected, columns=selected_features, index=X.index
        )

        logger.info(
            f"Feature selection completed using {method}. "
            f"Selected {len(selected_features)} features"
        )
        return X_selected_df

    def create_binning_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        bins: int = 5,
        method: str = "quantile",
    ) -> pd.DataFrame:
        """
        Create binning features for continuous variables.

        Args:
            df: Input DataFrame
            columns: Columns to bin
            bins: Number of bins
            method: Binning method ('quantile', 'uniform')

        Returns:
            DataFrame with binning features
        """
        df_binned = df.copy()

        for col in columns:
            if col in df.columns and df[col].dtype in ["int64", "float64"]:
                bin_col_name = f"{col}_bin"

                if method == "quantile":
                    df_binned[bin_col_name] = pd.qcut(
                        df[col], q=bins, labels=False, duplicates="drop"
                    )
                elif method == "uniform":
                    df_binned[bin_col_name] = pd.cut(df[col], bins=bins, labels=False)

                # Create dummy variables for bins
                dummies = pd.get_dummies(df_binned[bin_col_name], prefix=f"{col}_bin")
                df_binned = pd.concat([df_binned, dummies], axis=1)

        logger.info(f"Binning features created for columns: {columns}")
        return df_binned

    def create_statistical_features(
        self,
        df: pd.DataFrame,
        groupby_columns: List[str],
        agg_columns: List[str],
        operations: List[str] = ["mean", "std", "min", "max"],
    ) -> pd.DataFrame:
        """
        Create statistical features by grouping.

        Args:
            df: Input DataFrame
            groupby_columns: Columns to group by
            agg_columns: Columns to aggregate
            operations: Statistical operations to apply

        Returns:
            DataFrame with statistical features
        """
        df_stats = df.copy()

        for col in agg_columns:
            if col in df.columns:
                for group_col in groupby_columns:
                    if group_col in df.columns:
                        for op in operations:
                            feature_name = f"{col}_{op}_by_{group_col}"

                            if op == "mean":
                                df_stats[feature_name] = df.groupby(group_col)[
                                    col
                                ].transform("mean")
                            elif op == "std":
                                df_stats[feature_name] = df.groupby(group_col)[
                                    col
                                ].transform("std")
                            elif op == "min":
                                df_stats[feature_name] = df.groupby(group_col)[
                                    col
                                ].transform("min")
                            elif op == "max":
                                df_stats[feature_name] = df.groupby(group_col)[
                                    col
                                ].transform("max")
                            elif op == "count":
                                df_stats[feature_name] = df.groupby(group_col)[
                                    col
                                ].transform("count")

        logger.info(f"Statistical features created for {len(agg_columns)} columns")
        return df_stats
