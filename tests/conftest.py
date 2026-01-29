"""
Pytest configuration and fixtures for FabricaIA tests.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest


def pytest_configure(config):
    """
    Hook that runs before test collection.
    This completely mocks MLflow BEFORE any imports happen to prevent hanging.
    """
    # Set environment variable to disable MLflow completely
    os.environ["MLFLOW_DISABLE_TRACKING"] = "true"
    os.environ["MLFLOW_TRACKING_URI"] = ""

    # Create a complete mock for MLflow module
    class MockMLflow:
        """Complete mock of MLflow module."""

        def __init__(self):
            # Mock all functions as no-ops
            self.set_tracking_uri = lambda *args, **kwargs: None
            self.get_experiment_by_name = lambda *args, **kwargs: None
            self.create_experiment = lambda *args, **kwargs: None
            self.set_experiment = lambda *args, **kwargs: None
            self.start_run = lambda *args, **kwargs: MagicMock()
            self.end_run = lambda *args, **kwargs: None
            self.log_params = lambda *args, **kwargs: None
            self.log_metrics = lambda *args, **kwargs: None
            self.log_artifacts = lambda *args, **kwargs: None
            self.log_artifact = lambda *args, **kwargs: None
            self.register_model = lambda *args, **kwargs: None
            self.active_run = lambda: MagicMock(info=MagicMock(run_id="test-run-id"))

            # Mock submodules
            self.sklearn = MagicMock()
            self.sklearn.log_model = lambda *args, **kwargs: None
            self.sklearn.load_model = lambda *args, **kwargs: MagicMock()

            self.pytorch = MagicMock()
            self.pytorch.log_model = lambda *args, **kwargs: None
            self.pytorch.load_model = lambda *args, **kwargs: MagicMock()

            self.tensorflow = MagicMock()
            self.tensorflow.log_model = lambda *args, **kwargs: None
            self.tensorflow.load_model = lambda *args, **kwargs: MagicMock()

            # Mock tracking module with MlflowClient
            mock_client = MagicMock()
            mock_client.search_runs = lambda *args, **kwargs: []
            mock_client.get_run = lambda *args, **kwargs: MagicMock(
                data=MagicMock(params={}, metrics={}, tags={})
            )

            self.tracking = MagicMock()
            self.tracking.MlflowClient = lambda *args, **kwargs: mock_client

    # Inject mock into sys.modules BEFORE any real import
    mock_mlflow = MockMLflow()
    sys.modules["mlflow"] = mock_mlflow
    sys.modules["mlflow.sklearn"] = mock_mlflow.sklearn
    sys.modules["mlflow.pytorch"] = mock_mlflow.pytorch
    sys.modules["mlflow.tensorflow"] = mock_mlflow.tensorflow
    sys.modules["mlflow.tracking"] = mock_mlflow.tracking


@pytest.fixture(autouse=True)
def disable_mlflow_in_tests(monkeypatch):
    """
    Automatically disable MLflow in tests to prevent hanging.
    This fixture runs before every test.

    The MLflow initialization can hang if it tries to connect to a server
    or create database connections during test collection. This fixture
    ensures MLflow is disabled by default in tests.
    """
    # Set environment variable to disable MLflow tracking
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
    monkeypatch.setenv("MLFLOW_DISABLE_TRACKING", "true")

    # Mock MLflow functions to prevent any actual initialization
    try:
        from unittest.mock import MagicMock

        import mlflow

        # Create a mock that does nothing
        def noop(*args, **kwargs):
            return None

        # Mock only attributes that exist in mlflow module
        # Use setattr with create=True to avoid AttributeError
        attrs_to_mock = [
            "set_tracking_uri",
            "get_experiment_by_name",
            "create_experiment",
            "set_experiment",
            "start_run",
            "end_run",
            "log_params",
            "log_metrics",
            "log_artifacts",
            "register_model",
            "active_run",
        ]

        for attr in attrs_to_mock:
            if hasattr(mlflow, attr):
                if attr == "start_run":
                    monkeypatch.setattr(mlflow, attr, lambda **kwargs: MagicMock())
                elif attr == "get_experiment_by_name":
                    monkeypatch.setattr(mlflow, attr, lambda x: None)
                elif attr == "active_run":
                    monkeypatch.setattr(
                        mlflow,
                        attr,
                        lambda: MagicMock(info=MagicMock(run_id="test-run-id")),
                    )
                else:
                    monkeypatch.setattr(mlflow, attr, noop)
            else:
                # Create attribute if it doesn't exist (for safety)
                monkeypatch.setattr(mlflow, attr, noop, raising=False)

        # Mock mlflow submodules
        try:
            import mlflow.sklearn

            monkeypatch.setattr(mlflow.sklearn, "log_model", noop)
            monkeypatch.setattr(mlflow.sklearn, "load_model", lambda x: MagicMock())
        except (ImportError, AttributeError):
            pass

        try:
            import mlflow.pytorch

            monkeypatch.setattr(mlflow.pytorch, "log_model", noop)
            monkeypatch.setattr(mlflow.pytorch, "load_model", lambda x: MagicMock())
        except (ImportError, AttributeError):
            pass

        try:
            import mlflow.tensorflow

            monkeypatch.setattr(mlflow.tensorflow, "log_model", noop)
            monkeypatch.setattr(mlflow.tensorflow, "load_model", lambda x: MagicMock())
        except (ImportError, AttributeError):
            pass

        # Mock MlflowClient
        try:
            mock_client = MagicMock()
            mock_client.search_runs = lambda *args, **kwargs: []
            mock_client.get_run = lambda x: MagicMock(
                data=MagicMock(params={}, metrics={}, tags={})
            )
            # Mock the class directly without importing
            monkeypatch.setattr(
                "mlflow.tracking.MlflowClient", lambda *args, **kwargs: mock_client
            )
        except (ImportError, AttributeError):
            pass

    except ImportError:
        pass  # MLflow not installed, that's fine
