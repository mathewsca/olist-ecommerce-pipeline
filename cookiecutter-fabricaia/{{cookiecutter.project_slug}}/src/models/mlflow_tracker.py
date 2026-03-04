"""
FabricaIA - MLflow Tracking Module

This module provides integration with MLflow for experiment tracking,
model versioning, and deployment management.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Lazy imports to avoid hanging during pytest collection
# MLflow imports are done inside methods when needed


class MLflowTracker:
    """Class for tracking ML experiments with MLflow."""

    def _get_mlflow(self):
        """Lazy import mlflow to avoid hanging during pytest collection."""
        # Check if MLflow is disabled via environment variable
        if os.environ.get("MLFLOW_DISABLE_TRACKING", "").lower() == "true":
            # Return a mock if disabled
            from unittest.mock import MagicMock

            mock = MagicMock()
            mock.set_tracking_uri = lambda x: None
            mock.get_experiment_by_name = lambda x: None
            mock.create_experiment = lambda x: None
            mock.set_experiment = lambda x: None
            mock.start_run = lambda **kwargs: MagicMock()
            mock.end_run = lambda: None
            mock.log_params = lambda x: None
            mock.log_metrics = lambda x, step=None: None
            mock.log_artifacts = lambda x, y=None: None
            mock.log_artifact = lambda x: None
            mock.register_model = lambda x, y: None
            mock.active_run = lambda: MagicMock(info=MagicMock(run_id="test"))
            return mock

        import mlflow

        return mlflow

    def __init__(
        self,
        tracking_uri: str = "sqlite:///mlflow.db",
        experiment_name: str = "fabricaia_experiments",
    ):
        """
        Initialize MLflow tracker.

        Args:
            tracking_uri: URI of the MLflow tracking server
            experiment_name: Name of the MLflow experiment
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        # Lazy import MLflow to avoid hanging during import
        try:
            mlflow_module = self._get_mlflow()
            from mlflow.tracking import MlflowClient  # type: ignore

            mlflow_module.set_tracking_uri(tracking_uri)

            # Get or create experiment
            try:
                experiment = mlflow_module.get_experiment_by_name(experiment_name)
                if experiment is None:
                    mlflow_module.create_experiment(experiment_name)
                    logger.info(f"Created MLflow experiment: {experiment_name}")
                else:
                    logger.info(f"Using existing MLflow experiment: {experiment_name}")
            except Exception as e:
                logger.warning(f"Could not create MLflow experiment: {e}")

            mlflow_module.set_experiment(experiment_name)
            self.client = MlflowClient(tracking_uri=tracking_uri)
        except Exception as e:
            logger.warning(f"Could not initialize MLflow client: {e}")
            self.client = None

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Start a new MLflow run.

        Args:
            run_name: Name for this run
            tags: Dictionary of tags for this run
        """
        if self.client is None:
            raise RuntimeError("MLflow client not initialized")
        mlflow_module = self._get_mlflow()
        run = mlflow_module.start_run(run_name=run_name, tags=tags)
        logger.info(f"Started MLflow run: {run_name}")
        return run

    def end_run(self):
        """End the current MLflow run."""
        mlflow_module = self._get_mlflow()
        mlflow_module.end_run()
        logger.info("Ended MLflow run")

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow.

        Args:
            params: Dictionary of parameters to log
        """
        mlflow_module = self._get_mlflow()
        mlflow_module.log_params(params)
        logger.info(f"Logged parameters: {params}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step/epoch number
        """
        mlflow_module = self._get_mlflow()
        mlflow_module.log_metrics(metrics, step=step)
        logger.info(f"Logged metrics: {metrics}")

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Log model to MLflow.

        Args:
            model: Trained model to log
            artifact_path: Path within the run where the model should be logged
            registered_model_name: Name under which to register the model
            **kwargs: Additional arguments for model logging
        """
        mlflow_module = self._get_mlflow()
        # Auto-detect model type
        model_type = type(model).__module__

        if "sklearn" in model_type or "scikit_learn" in model_type:
            import mlflow.sklearn  # type: ignore

            mlflow_module.sklearn.log_model(model, artifact_path, **kwargs)
        elif "torch" in model_type or "pytorch" in model_type:
            import mlflow.pytorch  # type: ignore

            mlflow_module.pytorch.log_model(model, artifact_path, **kwargs)
        elif "tensorflow" in model_type or "keras" in model_type:
            import mlflow.tensorflow  # type: ignore  # noqa: F401

            mlflow_module.tensorflow.log_model(model, artifact_path, **kwargs)
        else:
            # Fallback to generic model logging
            import pickle

            model_path = f"models/{artifact_path}"
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)

            with open(f"{model_path}.pkl", "wb") as f:
                pickle.dump(model, f)

            mlflow_module.log_artifact(f"{model_path}.pkl")

        logger.info(f"Logged model to {artifact_path}")

        # Register model if name provided
        if registered_model_name:
            try:
                mlflow_module.register_model(
                    f"runs:/{mlflow_module.active_run().info.run_id}/{artifact_path}",
                    registered_model_name,
                )
                logger.info(f"Registered model: {registered_model_name}")
            except Exception as e:
                logger.warning(f"Could not register model: {e}")

    def log_artifacts(self, artifact_path: str, artifact_dir: Optional[str] = None):
        """
        Log artifacts to MLflow.

        Args:
            artifact_path: Path to artifact file or directory
            artifact_dir: Directory to store artifacts in MLflow
        """
        mlflow_module = self._get_mlflow()
        mlflow_module.log_artifacts(artifact_path, artifact_dir)
        logger.info(f"Logged artifacts from {artifact_path}")

    def load_model(self, model_uri: str, flavor: Optional[str] = None):
        """
        Load a model from MLflow.

        Args:
            model_uri: URI of the model to load
            flavor: Flavor of the model (sklearn, pytorch, tensorflow)

        Returns:
            Loaded model
        """
        mlflow_module = self._get_mlflow()
        if flavor == "sklearn" or flavor is None:
            import mlflow.sklearn  # type: ignore

            return mlflow_module.sklearn.load_model(model_uri)
        elif flavor == "pytorch" or flavor == "torch":
            import mlflow.pytorch  # type: ignore

            return mlflow_module.pytorch.load_model(model_uri)
        elif flavor == "tensorflow" or flavor == "tf":
            import mlflow.tensorflow  # type: ignore  # noqa: F401

            return mlflow_module.tensorflow.load_model(model_uri)
        else:
            raise ValueError(f"Unsupported model flavor: {flavor}")

    def search_runs(self, filter_string: str = "", max_results: int = 100):
        """
        Search runs in the current experiment.

        Args:
            filter_string: Filter expression
            max_results: Maximum number of results to return

        Returns:
            List of runs matching the filter
        """
        mlflow_module = self._get_mlflow()
        experiment = mlflow_module.get_experiment_by_name(self.experiment_name)
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            max_results=max_results,
        )
        return runs

    def get_best_run(self, metric_name: str, ascending: bool = True):
        """
        Get the best run based on a metric.

        Args:
            metric_name: Name of the metric to optimize
            ascending: If True, lower is better; if False, higher is better

        Returns:
            Best run based on the specified metric
        """
        runs = self.search_runs()
        if not runs:
            raise ValueError("No runs found")

        # Sort runs by the specified metric
        sorted_runs = sorted(
            runs,
            key=lambda r: r.data.metrics.get(metric_name, float("inf"))
            if ascending
            else -r.data.metrics.get(metric_name, -float("inf")),
        )

        return sorted_runs[0]

    def compare_runs(self, run_ids: list) -> Dict[str, Any]:
        """
        Compare multiple runs.

        Args:
            run_ids: List of run IDs to compare

        Returns:
            Dictionary with comparison results
        """
        comparison = {}
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            comparison[run_id] = {
                "params": run.data.params,
                "metrics": run.data.metrics,
                "tags": run.data.tags,
            }
        return comparison
