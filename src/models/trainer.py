"""
FabricaIA - Model Training Module

This module contains reusable classes and functions for model training,
evaluation, and management.
"""

import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC, SVR

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Main class for model training and evaluation."""

    def __init__(
        self,
        model_type: str = "classification",
        use_mlflow: bool = True,
        config_path: str = "config/config.yaml",
    ):
        """
        Initialize the model trainer.

        Args:
            model_type: Type of problem ('classification' or 'regression')
            use_mlflow: Whether to use MLflow for tracking
            config_path: Path to configuration file
        """
        self.model_type = model_type
        self.models: Dict[str, Any] = {}
        self.best_model = None
        self.best_params = None
        self.training_history: Dict[str, Any] = {}
        self.use_mlflow = use_mlflow
        self.mlflow_tracker = None

        # Load MLflow configuration
        # (lazy import to avoid hanging during pytest collection)
        if self.use_mlflow:
            # Check if MLflow is disabled via environment variable (e.g., during tests)
            import os

            if os.environ.get("MLFLOW_DISABLE_TRACKING", "").lower() == "true":
                logger.info(
                    "MLflow disabled via MLFLOW_DISABLE_TRACKING environment variable"
                )
                self.use_mlflow = False
                return

            try:
                # Lazy import MLflowTracker to avoid import-time initialization
                from .mlflow_tracker import MLflowTracker

                # Check if config file exists before trying to read it
                config_file = Path(config_path)
                if config_file.exists():
                    with open(config_path, "r") as f:
                        config = yaml.safe_load(f)
                        mlflow_config = config.get("MLFLOW", {})
                        if mlflow_config.get("enable_tracking", True):
                            self.mlflow_tracker = MLflowTracker(
                                tracking_uri=mlflow_config.get(
                                    "tracking_uri", "sqlite:///mlflow.db"
                                ),
                                experiment_name=mlflow_config.get(
                                    "experiment_name", "fabricaia_experiments"
                                ),
                            )
                            logger.info("MLflow tracking enabled")
                else:
                    logger.warning(
                        f"Config file not found: {config_path}, MLflow disabled"
                    )
                    self.use_mlflow = False
            except Exception as e:
                logger.warning(f"Could not initialize MLflow: {e}")
                self.use_mlflow = False

    def _filter_params(
        self, model_class: Type[Any], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Filter parameters to only include those valid for the given model class.

        Args:
            model_class: The model class to check parameters for
            params: Dictionary of parameters to filter

        Returns:
            Filtered dictionary with only valid parameters
        """
        # Get valid parameters for the model class
        # Access __init__ safely to satisfy mypy type checking
        init_method = getattr(model_class, "__init__", None)
        if init_method is None:
            return {}
        sig = inspect.signature(init_method)
        valid_params = set(sig.parameters.keys()) - {"self"}

        # Filter params to only include valid ones
        filtered_params = {k: v for k, v in params.items() if k in valid_params}

        # Log if any parameters were filtered out
        removed_params = set(params.keys()) - set(filtered_params.keys())
        if removed_params:
            logger.warning(
                f"Removed invalid parameters for {model_class.__name__}: "
                f"{removed_params}"
            )

        return filtered_params

    def get_model(self, algorithm: str, **params):
        """
        Get a model instance based on algorithm name.

        Args:
            algorithm: Name of the algorithm
            **params: Model parameters
                (only valid parameters for the algorithm will be used)

        Returns:
            Model instance
        """
        if self.model_type == "classification":
            model_classes = {
                "random_forest": RandomForestClassifier,
                "logistic_regression": LogisticRegression,
                "svm": SVC,
            }
        else:  # regression
            model_classes = {
                "random_forest": RandomForestRegressor,
                "linear_regression": LinearRegression,
                "svm": SVR,
            }

        if algorithm not in model_classes:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Filter parameters to only include valid ones for this algorithm
        model_class = model_classes[algorithm]
        filtered_params = self._filter_params(model_class, params)

        return model_class(**filtered_params)

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        algorithm: str = "random_forest",
        run_name: Optional[str] = None,
        **params,
    ) -> Any:
        """
        Train a model with given data.

        Args:
            X_train: Training features
            y_train: Training target
            algorithm: Algorithm to use
            run_name: Name for MLflow run
            **params: Model parameters

        Returns:
            Trained model
        """
        # Start MLflow run if enabled
        if self.mlflow_tracker and run_name:
            self.mlflow_tracker.start_run(
                run_name=run_name, tags={"model_type": self.model_type}
            )

        model = self.get_model(algorithm, **params)

        # Log parameters to MLflow
        if self.mlflow_tracker:
            mlflow_params = {"algorithm": algorithm, **params}
            self.mlflow_tracker.log_params(mlflow_params)
            self.mlflow_tracker.log_params(
                {"n_samples": len(X_train), "n_features": X_train.shape[1]}
            )

        logger.info(f"Training {algorithm} model...")
        model.fit(X_train, y_train)

        self.models[algorithm] = model
        logger.info(f"Model {algorithm} trained successfully")

        return model

    def evaluate_model(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Evaluate model performance.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = model.predict(X_test)

        if self.model_type == "classification":
            metrics = {
                "accuracy": model.score(X_test, y_test),
                "classification_report": classification_report(
                    y_test, y_pred, output_dict=True
                ),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            }

            # Log metrics to MLflow
            if self.mlflow_tracker:
                mlflow_metrics = {"test_accuracy": metrics["accuracy"]}
                # Extract additional metrics from classification report
                if "1" in metrics["classification_report"]:
                    report = metrics["classification_report"]["1"]
                    mlflow_metrics.update(
                        {
                            "precision": report.get("precision", 0),
                            "recall": report.get("recall", 0),
                            "f1_score": report.get("f1-score", 0),
                        }
                    )
                self.mlflow_tracker.log_metrics(mlflow_metrics)
        else:  # regression
            metrics = {
                "r2_score": r2_score(y_test, y_pred),
                "mse": mean_squared_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            }

            # Log metrics to MLflow
            if self.mlflow_tracker:
                self.mlflow_tracker.log_metrics(
                    {
                        "test_r2_score": metrics["r2_score"],
                        "test_mse": metrics["mse"],
                        "test_rmse": metrics["rmse"],
                    }
                )

        logger.info(f"Model evaluation completed. Metrics: {metrics}")
        return metrics

    def cross_validate(
        self, model: Any, X: pd.DataFrame, y: pd.Series, cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation.

        Args:
            model: Model to validate
            X: Features
            y: Target
            cv: Number of folds

        Returns:
            Cross-validation scores
        """
        scores = cross_val_score(model, X, y, cv=cv)

        cv_results = {
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "scores": scores.tolist(),
        }

        logger.info(
            f"Cross-validation completed. Mean score: {cv_results['mean_score']:.4f}"
        )
        return cv_results

    def hyperparameter_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        algorithm: str,
        param_grid: Dict[str, List],
        cv: int = 5,
    ) -> Any:
        """
        Perform hyperparameter tuning using GridSearchCV.

        Args:
            X_train: Training features
            y_train: Training target
            algorithm: Algorithm to tune
            param_grid: Parameter grid for tuning
            cv: Number of folds for cross-validation

        Returns:
            Best model with tuned parameters
        """
        base_model = self.get_model(algorithm)

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring="accuracy" if self.model_type == "classification" else "r2",
        )

        logger.info(f"Starting hyperparameter tuning for {algorithm}...")
        grid_search.fit(X_train, y_train)

        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        # Log hyperparameter tuning results to MLflow
        if self.mlflow_tracker:
            self.mlflow_tracker.log_params({"best_params": str(self.best_params)})
            self.mlflow_tracker.log_metrics({"best_cv_score": grid_search.best_score_})
            # Log relevant metrics from cv_results_
            cv_metrics = {}
            for metric, value in grid_search.cv_results_.items():
                if metric.startswith("mean_test") or metric.startswith("std_test"):
                    if isinstance(value, (int, float)):
                        cv_metrics[metric] = value
                    elif isinstance(value, np.ndarray):
                        cv_metrics[metric] = value[0] if len(value) > 0 else 0
            if cv_metrics:
                self.mlflow_tracker.log_metrics(cv_metrics)

        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")

        return self.best_model

    def save_model(
        self, model: Any, file_path: str, artifact_path: str = "model"
    ) -> None:
        """
        Save trained model to disk and MLflow.

        Args:
            model: Trained model
            file_path: Path to save the model
            artifact_path: Path within MLflow run where the model should be logged
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, file_path)
        logger.info(f"Model saved to {file_path}")

        # Log model to MLflow
        if self.mlflow_tracker:
            self.mlflow_tracker.log_model(model, artifact_path=artifact_path)

    def load_model(self, file_path: str) -> Any:
        """
        Load model from disk.

        Args:
            file_path: Path to the model file

        Returns:
            Loaded model
        """
        model = joblib.load(file_path)
        logger.info(f"Model loaded from {file_path}")
        return model

    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using trained model.

        Args:
            model: Trained model
            X: Features for prediction

        Returns:
            Predictions
        """
        predictions = model.predict(X)
        logger.info(f"Predictions generated for {len(X)} samples")
        return predictions

    def predict_proba(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities (for classification models).

        Args:
            model: Trained classification model
            X: Features for prediction

        Returns:
            Prediction probabilities
        """
        if self.model_type != "classification":
            raise ValueError(
                "Probability prediction only available for classification models"
            )

        probabilities = model.predict_proba(X)
        logger.info(f"Probabilities generated for {len(X)} samples")
        return probabilities

    def end_run(self):
        """
        End the current MLflow run.
        """
        if self.mlflow_tracker:
            self.mlflow_tracker.end_run()
            logger.info("Ended MLflow run")
