"""
FabricaIA - Model Training Module

This module contains reusable classes and functions for model training,
evaluation, and management.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
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

    def __init__(self, model_type: str = "classification"):
        """
        Initialize the model trainer.

        Args:
            model_type: Type of problem ('classification' or 'regression')
        """
        self.model_type = model_type
        self.models = {}
        self.best_model = None
        self.best_params = None
        self.training_history = {}

    def get_model(self, algorithm: str, **params):
        """
        Get a model instance based on algorithm name.

        Args:
            algorithm: Name of the algorithm
            **params: Model parameters

        Returns:
            Model instance
        """
        if self.model_type == "classification":
            models = {
                "random_forest": RandomForestClassifier(**params),
                "logistic_regression": LogisticRegression(**params),
                "svm": SVC(**params),
            }
        else:  # regression
            models = {
                "random_forest": RandomForestRegressor(**params),
                "linear_regression": LinearRegression(**params),
                "svm": SVR(**params),
            }

        if algorithm not in models:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        return models[algorithm]

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        algorithm: str = "random_forest",
        **params,
    ) -> Any:
        """
        Train a model with given data.

        Args:
            X_train: Training features
            y_train: Training target
            algorithm: Algorithm to use
            **params: Model parameters

        Returns:
            Trained model
        """
        model = self.get_model(algorithm, **params)

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
        else:  # regression
            metrics = {
                "r2_score": r2_score(y_test, y_pred),
                "mse": mean_squared_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            }

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

        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")

        return self.best_model

    def save_model(self, model: Any, file_path: str) -> None:
        """
        Save trained model to disk.

        Args:
            model: Trained model
            file_path: Path to save the model
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, file_path)
        logger.info(f"Model saved to {file_path}")

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
