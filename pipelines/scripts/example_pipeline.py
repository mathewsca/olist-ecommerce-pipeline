"""
FabricaIA - Example Pipeline

This is an example pipeline showing how to use the FabricaIA modules
for a complete machine learning workflow.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Import FabricaIA modules
from src.data.processor import DataProcessor
from src.features.engineering import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.visualization.plots import DataVisualizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FabricaIAPipeline:
    """Main pipeline class for FabricaIA workflow."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the pipeline.

        Args:
            config_path: Path to configuration file
        """
        self.config = self.load_config(config_path)
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer()
        self.feature_engineer = FeatureEngineer()
        self.visualizer = DataVisualizer()

        # Create necessary directories
        self.create_directories()

    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config

    def create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.config["DATA_PATHS"]["raw"],
            self.config["DATA_PATHS"]["processed"],
            self.config["DATA_PATHS"]["external"],
            self.config["MODEL_PATHS"]["trained"],
            self.config["MODEL_PATHS"]["artifacts"],
            "logs",
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def load_and_explore_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and perform initial data exploration.

        Args:
            data_path: Path to the data file

        Returns:
            Loaded DataFrame
        """
        logger.info("Loading data...")
        df = self.data_processor.load_data(data_path)

        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Data types:\n{df.dtypes}")
        logger.info(f"Missing values:\n{df.isnull().sum()}")

        # Visualize missing data
        self.visualizer.plot_missing_data(df, "logs/missing_data.png")

        # Plot distributions for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            self.visualizer.plot_distribution(
                df, numerical_cols[:6], save_path="logs/distributions.png"
            )

        # Plot correlation matrix
        if len(numerical_cols) > 1:
            self.visualizer.plot_correlation_matrix(
                df, save_path="logs/correlation_matrix.png"
            )

        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data.

        Args:
            df: Input DataFrame

        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing data...")

        # Clean data
        df_clean = self.data_processor.clean_data(
            df, drop_duplicates=True, handle_missing="fill"
        )

        # Encode categorical variables
        df_encoded = self.data_processor.encode_categorical(df_clean, method="label")

        # Scale features
        df_scaled = self.data_processor.scale_features(df_encoded)

        logger.info(f"Preprocessed data shape: {df_scaled.shape}")
        return df_scaled

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer new features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features...")

        # Get numerical columns for feature engineering
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Create polynomial features
        if len(numerical_cols) >= 2:
            df_poly = self.feature_engineer.create_polynomial_features(
                df,
                numerical_cols[:2],
                degree=self.config["FEATURES"]["polynomial_degree"],
            )
        else:
            df_poly = df

        # Create interaction features
        if len(numerical_cols) >= 2:
            interaction_pairs = [(numerical_cols[0], numerical_cols[1])]
            df_interact = self.feature_engineer.create_interaction_features(
                df_poly, interaction_pairs
            )
        else:
            df_interact = df_poly

        logger.info(f"Feature engineered data shape: {df_interact.shape}")
        return df_interact

    def train_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict:
        """
        Train multiple models and evaluate them.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary with model results
        """
        logger.info("Training models...")

        results = {}
        algorithms = self.config["MODELS"]["algorithms"]

        for algorithm in algorithms:
            logger.info(f"Training {algorithm}...")

            # Train model
            model = self.model_trainer.train_model(X_train, y_train, algorithm)

            # Evaluate model
            metrics = self.model_trainer.evaluate_model(model, X_test, y_test)

            # Cross-validation
            cv_results = self.model_trainer.cross_validate(
                model, pd.concat([X_train, X_test]), pd.concat([y_train, y_test])
            )

            # Get feature importance (if available)
            if hasattr(model, "feature_importances_"):
                importance_df = self.data_processor.get_feature_importance(model)
                self.visualizer.plot_feature_importance(
                    importance_df, save_path=f"logs/{algorithm}_feature_importance.png"
                )

            # Plot predictions vs actual
            y_pred = self.model_trainer.predict(model, X_test)
            self.visualizer.plot_prediction_vs_actual(
                y_test, y_pred, algorithm, f"logs/{algorithm}_predictions.png"
            )

            # Save model
            model_path = (
                f"{self.config['MODEL_PATHS']['trained']}/{algorithm}_model.pkl"
            )
            self.model_trainer.save_model(model, model_path)

            results[algorithm] = {
                "model": model,
                "metrics": metrics,
                "cv_results": cv_results,
                "model_path": model_path,
            }

        return results

    def run_pipeline(self, data_path: str, target_column: str):
        """
        Run the complete pipeline.

        Args:
            data_path: Path to the data file
            target_column: Name of the target column
        """
        logger.info("Starting FabricaIA Pipeline...")

        # 1. Load and explore data
        df = self.load_and_explore_data(data_path)

        # 2. Preprocess data
        df_processed = self.preprocess_data(df)

        # 3. Engineer features
        df_features = self.engineer_features(df_processed)

        # 4. Split data
        X_train, X_test, y_train, y_test = self.data_processor.split_data(
            df_features,
            target_column,
            test_size=self.config["PIPELINE"]["test_size"],
            random_state=self.config["PIPELINE"]["random_state"],
        )

        # 5. Train models
        results = self.train_models(X_train, y_train, X_test, y_test)

        # 6. Summary
        logger.info("Pipeline completed successfully!")
        logger.info("Model Results Summary:")
        for algorithm, result in results.items():
            logger.info(f"{algorithm}: {result['metrics']}")

        return results


def main():
    """Main function to run the pipeline."""
    # Example usage
    # pipeline = FabricaIAPipeline()

    # You would replace this with your actual data path
    # data_path = "data/raw/your_dataset.csv"
    # target_column = "your_target_column"
    # results = pipeline.run_pipeline(data_path, target_column)

    print("FabricaIA Pipeline initialized successfully!")
    print("To run the pipeline, provide your data path and target column.")


if __name__ == "__main__":
    main()
