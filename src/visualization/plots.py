"""
FabricaIA - Visualization Module

This module contains reusable functions for data visualization
and model performance visualization.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class DataVisualizer:
    """Main class for data visualization operations."""

    def __init__(self, figsize: Tuple[int, int] = (10, 6), dpi: int = 100):
        """
        Initialize the visualizer.

        Args:
            figsize: Default figure size
            dpi: Default DPI for plots
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colors = sns.color_palette("husl", 10)

    def plot_distribution(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        plot_type: str = "histogram",
        bins: int = 30,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot distribution of numerical columns.

        Args:
            df: Input DataFrame
            columns: Columns to plot (if None, plot all numerical)
            plot_type: Type of plot ('histogram', 'kde', 'box')
            bins: Number of bins for histogram
            save_path: Path to save the plot
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(self.figsize[0] * n_cols, self.figsize[1] * n_rows)
        )
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        for i, col in enumerate(columns):
            if plot_type == "histogram":
                axes[i].hist(
                    df[col].dropna(),
                    bins=bins,
                    alpha=0.7,
                    color=self.colors[i % len(self.colors)],
                )
            elif plot_type == "kde":
                df[col].dropna().plot(
                    kind="kde", ax=axes[i], color=self.colors[i % len(self.colors)]
                )
            elif plot_type == "box":
                axes[i].boxplot(df[col].dropna())

            axes[i].set_title(f"Distribution of {col}")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frequency")

        # Hide empty subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Distribution plot saved to {save_path}")

        plt.show()

    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        method: str = "pearson",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot correlation matrix heatmap.

        Args:
            df: Input DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')
            save_path: Path to save the plot
        """
        # Calculate correlation matrix
        corr_matrix = df.select_dtypes(include=[np.number]).corr(method=method)

        # Create heatmap
        plt.figure(
            figsize=(
                max(8, len(corr_matrix.columns) * 0.8),
                max(6, len(corr_matrix.columns) * 0.6),
            )
        )

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )

        plt.title(f"Correlation Matrix ({method.title()})")
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Correlation matrix saved to {save_path}")

        plt.show()

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot feature importance.

        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to show
            save_path: Path to save the plot
        """
        # Get top N features
        top_features = importance_df.head(top_n)

        plt.figure(figsize=self.figsize)
        # Add hue and disable legend to comply with seaborn >=0.14 behavior
        sns.barplot(
            data=top_features,
            x="importance",
            y="feature",
            hue="feature",
            dodge=False,
            legend=False,
            palette="viridis",
        )
        plt.title(f"Top {top_n} Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Feature importance plot saved to {save_path}")

        plt.show()

    def plot_model_performance(
        self,
        metrics: Dict[str, Any],
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot model performance metrics.

        Args:
            metrics: Dictionary with model metrics
            model_name: Name of the model
            save_path: Path to save the plot
        """
        # Determine if classification or regression
        if "accuracy" in metrics or "classification_report" in metrics:
            # Classification metrics
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # Accuracy
            if "accuracy" in metrics:
                axes[0].bar(["Accuracy"], [metrics["accuracy"]], color=self.colors[0])
                axes[0].set_title(f"{model_name} - Accuracy")
                axes[0].set_ylabel("Score")
                axes[0].set_ylim(0, 1)

            # Confusion Matrix
            if "confusion_matrix" in metrics:
                cm = np.array(metrics["confusion_matrix"])
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1])
                axes[1].set_title(f"{model_name} - Confusion Matrix")
                axes[1].set_xlabel("Predicted")
                axes[1].set_ylabel("Actual")

        else:
            # Regression metrics
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            metrics_to_plot = ["r2_score", "mse", "rmse"]
            metric_names = ["R² Score", "MSE", "RMSE"]

            for i, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
                if metric in metrics:
                    axes[i].bar([name], [metrics[metric]], color=self.colors[i])
                    axes[i].set_title(f"{model_name} - {name}")
                    axes[i].set_ylabel("Score")

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Model performance plot saved to {save_path}")

        plt.show()

    def plot_prediction_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot predicted vs actual values.

        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Path to save the plot
        """
        plt.figure(figsize=self.figsize)

        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.6, color=self.colors[0])

        # Perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            lw=2,
            label="Perfect Prediction",
        )

        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"{model_name} - Predicted vs Actual")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Prediction plot saved to {save_path}")

        plt.show()

    def plot_time_series(
        self,
        df: pd.DataFrame,
        time_column: str,
        value_columns: List[str],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot time series data.

        Args:
            df: Input DataFrame
            time_column: Name of the time column
            value_columns: Columns to plot
            save_path: Path to save the plot
        """
        plt.figure(figsize=(self.figsize[0] * 1.5, self.figsize[1]))

        for i, col in enumerate(value_columns):
            if col in df.columns:
                plt.plot(
                    df[time_column],
                    df[col],
                    label=col,
                    color=self.colors[i % len(self.colors)],
                )

        plt.xlabel(time_column)
        plt.ylabel("Value")
        plt.title("Time Series Plot")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Time series plot saved to {save_path}")

        plt.show()

    def plot_missing_data(
        self, df: pd.DataFrame, save_path: Optional[str] = None
    ) -> None:
        """
        Plot missing data patterns.

        Args:
            df: Input DataFrame
            save_path: Path to save the plot
        """
        # Calculate missing data
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100

        # Create DataFrame for plotting
        missing_df = pd.DataFrame(
            {
                "Column": missing_data.index,
                "Missing Count": missing_data.values,
                "Missing Percent": missing_percent.values,
            }
        ).sort_values("Missing Percent", ascending=False)

        # Filter columns with missing data
        missing_df = missing_df[missing_df["Missing Count"] > 0]

        if len(missing_df) == 0:
            print("No missing data found!")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Missing count plot
        sns.barplot(
            data=missing_df,
            x="Missing Count",
            y="Column",
            ax=axes[0],
            palette="viridis",
        )
        axes[0].set_title("Missing Data Count by Column")

        # Missing percentage plot
        sns.barplot(
            data=missing_df,
            x="Missing Percent",
            y="Column",
            ax=axes[1],
            palette="viridis",
        )
        axes[1].set_title("Missing Data Percentage by Column")
        axes[1].set_xlabel("Missing Percentage (%)")

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Missing data plot saved to {save_path}")

        plt.show()
