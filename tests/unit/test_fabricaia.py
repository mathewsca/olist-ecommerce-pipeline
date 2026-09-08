"""
FabricaIA - Test Suite

This module contains unit tests for FabricaIA modules.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

# Import FabricaIA modules
from src.data.processor import DataProcessor
from src.features.engineering import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.visualization.plots import DataVisualizer


class TestDataProcessor:
    """Test cases for DataProcessor class."""

    def setup_method(self):
        """Setup test data."""
        self.processor = DataProcessor()

        # Create sample data
        self.sample_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
                "category": ["A", "B", "A", "C", "B"],
                "target": [0, 1, 0, 1, 0],
            }
        )

    def test_clean_data_drop_duplicates(self):
        """Test data cleaning with duplicate removal."""
        # Add duplicates
        data_with_duplicates = pd.concat([self.sample_data, self.sample_data.iloc[:2]])

        cleaned_data = self.processor.clean_data(
            data_with_duplicates, drop_duplicates=True
        )

        assert len(cleaned_data) == len(self.sample_data)
        assert cleaned_data.equals(self.sample_data)

    def test_clean_data_handle_missing(self):
        """Test data cleaning with missing value handling."""
        # Add missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0, "feature1"] = np.nan
        data_with_missing.loc[1, "category"] = np.nan

        # Test drop strategy
        cleaned_drop = self.processor.clean_data(
            data_with_missing, handle_missing="drop"
        )
        assert not cleaned_drop.isnull().any().any()

        # Test fill strategy
        cleaned_fill = self.processor.clean_data(
            data_with_missing, handle_missing="fill"
        )
        assert not cleaned_fill.isnull().any().any()

    def test_encode_categorical(self):
        """Test categorical encoding."""
        encoded_data = self.processor.encode_categorical(
            self.sample_data, method="label"
        )

        # Check that categorical column is encoded
        assert encoded_data["category"].dtype in ["int64", "int32"]
        assert len(encoded_data["category"].unique()) == len(
            self.sample_data["category"].unique()
        )

    def test_scale_features(self):
        """Test feature scaling."""
        scaled_data = self.processor.scale_features(self.sample_data)

        # Check that numerical features are scaled
        assert scaled_data["feature1"].std() < 2  # Should be close to 1 after scaling
        assert scaled_data["feature2"].std() < 2

    def test_split_data(self):
        """Test data splitting."""
        X_train, X_test, y_train, y_test = self.processor.split_data(
            self.sample_data, "target", test_size=0.4, random_state=42
        )

        # Check shapes
        assert len(X_train) + len(X_test) == len(self.sample_data)
        assert len(y_train) + len(y_test) == len(self.sample_data)

        # Check that target column is not in features
        assert "target" not in X_train.columns
        assert "target" not in X_test.columns


class TestModelTrainer:
    """Test cases for ModelTrainer class."""

    def setup_method(self):
        """Setup test data."""
        self.trainer = ModelTrainer(model_type="classification", use_mlflow=False)

        # Create sample data
        np.random.seed(42)
        self.X_train = pd.DataFrame(
            {"feature1": np.random.randn(100), "feature2": np.random.randn(100)}
        )
        self.y_train = pd.Series(np.random.randint(0, 2, 100))

        self.X_test = pd.DataFrame(
            {"feature1": np.random.randn(20), "feature2": np.random.randn(20)}
        )
        self.y_test = pd.Series(np.random.randint(0, 2, 20))

    def test_get_model_classification(self):
        """Test getting classification models."""
        rf_model = self.trainer.get_model("random_forest")
        lr_model = self.trainer.get_model("logistic_regression")
        svm_model = self.trainer.get_model("svm")

        assert rf_model is not None
        assert lr_model is not None
        assert svm_model is not None

    def test_train_model(self):
        """Test model training."""
        model = self.trainer.train_model(self.X_train, self.y_train, "random_forest")

        assert model is not None
        assert "random_forest" in self.trainer.models

    def test_evaluate_model(self):
        """Test model evaluation."""
        model = self.trainer.train_model(self.X_train, self.y_train, "random_forest")
        metrics = self.trainer.evaluate_model(model, self.X_test, self.y_test)

        assert "accuracy" in metrics
        assert isinstance(metrics["accuracy"], float)

    def test_cross_validate(self):
        """Test cross-validation."""
        model = self.trainer.train_model(self.X_train, self.y_train, "random_forest")
        cv_results = self.trainer.cross_validate(model, self.X_train, self.y_train)

        assert "mean_score" in cv_results
        assert "std_score" in cv_results
        assert isinstance(cv_results["mean_score"], float)

    def test_save_load_model(self):
        """Test model saving and loading."""
        model = self.trainer.train_model(self.X_train, self.y_train, "random_forest")

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Save model
            self.trainer.save_model(model, tmp_path)
            assert os.path.exists(tmp_path)

            # Load model
            loaded_model = self.trainer.load_model(tmp_path)
            assert loaded_model is not None

            # Test predictions are the same
            pred1 = model.predict(self.X_test)
            pred2 = loaded_model.predict(self.X_test)
            np.testing.assert_array_equal(pred1, pred2)

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class."""

    def setup_method(self):
        """Setup test data."""
        self.engineer = FeatureEngineer()

        # Create sample data
        self.sample_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
                "category": ["A", "B", "A", "C", "B"],
            }
        )

    def test_create_polynomial_features(self):
        """Test polynomial feature creation."""
        poly_data = self.engineer.create_polynomial_features(
            self.sample_data, ["feature1"], degree=2
        )

        # Check that polynomial features are created
        poly_cols = [col for col in poly_data.columns if "poly" in col]
        assert len(poly_cols) > 0

    def test_create_interaction_features(self):
        """Test interaction feature creation."""
        interact_data = self.engineer.create_interaction_features(
            self.sample_data, [("feature1", "feature2")]
        )

        # Check that interaction feature is created
        assert "feature1_x_feature2" in interact_data.columns

    def test_create_binning_features(self):
        """Test binning feature creation."""
        binned_data = self.engineer.create_binning_features(
            self.sample_data, ["feature1"], bins=3
        )

        # Check that binning features are created
        bin_cols = [col for col in binned_data.columns if "bin" in col]
        assert len(bin_cols) > 0

    def test_select_features(self):
        """Test feature selection."""
        # Create target for feature selection
        y = pd.Series([0, 1, 0, 1, 0])

        selected_data = self.engineer.select_features(
            self.sample_data[["feature1", "feature2"]], y, method="univariate", k=1
        )

        # Check that only 1 feature is selected
        assert selected_data.shape[1] == 1


class TestDataVisualizer:
    """Test cases for DataVisualizer class."""

    def setup_method(self):
        """Setup test data."""
        self.visualizer = DataVisualizer()

        # Create sample data
        np.random.seed(42)
        self.sample_data = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "target": np.random.randint(0, 2, 100),
            }
        )

    def test_plot_distribution(self):
        """Test distribution plotting."""
        # This test mainly checks that the function runs without error
        # In a real test environment, you might want to check the plot output
        try:
            self.visualizer.plot_distribution(self.sample_data, ["feature1"])
            assert True
        except Exception as e:
            pytest.fail(f"plot_distribution failed: {str(e)}")

    def test_plot_correlation_matrix(self):
        """Test correlation matrix plotting."""
        try:
            self.visualizer.plot_correlation_matrix(self.sample_data)
            assert True
        except Exception as e:
            pytest.fail(f"plot_correlation_matrix failed: {str(e)}")

    def test_plot_feature_importance(self):
        """Test feature importance plotting."""
        importance_df = pd.DataFrame(
            {"feature": ["feature1", "feature2"], "importance": [0.6, 0.4]}
        )

        try:
            self.visualizer.plot_feature_importance(importance_df)
            assert True
        except Exception as e:
            pytest.fail(f"plot_feature_importance failed: {str(e)}")


# Integration tests
class TestIntegration:
    """Integration tests for FabricaIA pipeline."""

    def test_full_pipeline(self):
        """Test a complete pipeline workflow."""
        # Create sample data
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "category": np.random.choice(["A", "B", "C"], 100),
                "target": np.random.randint(0, 2, 100),
            }
        )

        # Initialize components
        processor = DataProcessor()
        trainer = ModelTrainer(model_type="classification", use_mlflow=False)

        # Process data
        cleaned_data = processor.clean_data(data)
        encoded_data = processor.encode_categorical(cleaned_data)
        scaled_data = processor.scale_features(encoded_data)

        # Split data
        X_train, X_test, y_train, y_test = processor.split_data(scaled_data, "target")

        # Train model
        model = trainer.train_model(X_train, y_train, "random_forest")

        # Evaluate model
        metrics = trainer.evaluate_model(model, X_test, y_test)

        # Check that everything worked
        assert "accuracy" in metrics
        assert metrics["accuracy"] >= 0  # Should be a valid accuracy score


class TestPredictionService:
    """Test cases for PredictionService class."""

    def test_prediction_service_raw_inference(self):
        from src.services.prediction_service import PredictionService
        from sklearn.ensemble import RandomForestClassifier

        service = PredictionService()
        X = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [10.0, 20.0, 30.0, 40.0],
        })
        y = np.array([0, 1, 0, 1])
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        res = service.predict_instance(
            model=model,
            model_name="test_rf",
            raw_features={"f1": 2.5, "f2": 25.0},
        )

        assert "prediction" in res
        assert "probability" in res
        assert "confidence" in res
        assert res["prediction"] in [0, 1]


if __name__ == "__main__":
    pytest.main([__file__])
